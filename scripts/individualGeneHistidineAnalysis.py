#!/usr/bin/env python3
"""
Individual Gene Histidine TadA Editing Analysis
================================================
For each gene with sufficient coverage in both BAMs, builds a per-position
A→G editing fraction matrix across the CDS, computes log2FC between the two
libraries, and highlights histidine codons. One PDF figure per gene.

Usage:
    python3 individualGeneHistidineAnalysis.py \
        --bam1 condition1.bam --label1 "WT" \
        --bam2 condition2.bam --label2 "3-AT" \
        --ref reference.fa \
        --gtf annotation.gtf \
        --output output_prefix \
        [--min_coverage 50] \
        [--min_mapq 20] \
        [--min_baseq 10] \
        [--pseudo 1e-3] \
        [--gene_list genes.txt]

Requirements:
    pip install pysam pandas numpy matplotlib
"""

import argparse
import sys
import re
import collections
from pathlib import Path

import pysam
import numpy as np
import pandas as pd


# ── Histidine codons (DNA, transcript coordinates) ──────────────────────────
HIS_CODONS = {"CAT", "CAC"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. GTF parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_gtf(gtf_path: str) -> dict:
    """
    Parse GTF CDS features. Returns:
        gene_name → {
            "chrom":      str,
            "strand":     "+" | "-",
            "transcript": str,
            "gene_name":  str,
            "cds":        sorted list of (start0, end0)
        }
    One entry per gene_name, using the first transcript encountered.
    """
    genes = {}
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "CDS":
                continue
            chrom  = fields[0]
            start  = int(fields[3]) - 1   # GTF 1-based → 0-based
            end    = int(fields[4])
            strand = fields[6]
            m_tid  = re.search(r'transcript_id "([^"]+)"', fields[8])
            m_gn   = re.search(r'gene_name "([^"]+)"', fields[8])
            tid    = m_tid.group(1) if m_tid else "."
            gname  = m_gn.group(1)  if m_gn  else tid

            if gname not in genes:
                genes[gname] = {
                    "chrom":      chrom,
                    "strand":     strand,
                    "transcript": tid,
                    "gene_name":  gname,
                    "cds":        [],
                }
            genes[gname]["cds"].append((start, end))

    # Sort CDS intervals; for minus-strand genes sort descending so we can
    # walk them in transcript order later
    for g in genes.values():
        g["cds"].sort(key=lambda x: x[0],
                      reverse=(g["strand"] == "-"))
    return genes


# ─────────────────────────────────────────────────────────────────────────────
# 2. Coverage filtering
# ─────────────────────────────────────────────────────────────────────────────

def mean_cds_coverage(bam: pysam.AlignmentFile,
                      gene: dict,
                      min_mapq: int) -> float:
    """Mean per-position depth across all CDS intervals of a gene."""
    total_depth = 0
    total_bases = 0
    for (start, end) in gene["cds"]:
        for col in bam.pileup(
            gene["chrom"], start, end,
            truncate=True,
            min_mapping_quality=min_mapq,
            stepper="samtools",
        ):
            total_depth += col.nsegments
            total_bases += 1
    return total_depth / total_bases if total_bases > 0 else 0.0


def filter_high_coverage_genes(genes: dict,
                                bam1_path: str,
                                bam2_path: str,
                                min_coverage: float,
                                min_mapq: int) -> list:
    """
    Return list of gene_names where BOTH BAMs have mean CDS coverage
    >= min_coverage.
    """
    passing = []
    bam1 = pysam.AlignmentFile(bam1_path, "rb")
    bam2 = pysam.AlignmentFile(bam2_path, "rb")

    for i, (gname, gene) in enumerate(genes.items()):
        if (i + 1) % 100 == 0:
            print(f"  Checking coverage {i+1}/{len(genes)}…", file=sys.stderr)
        cov1 = mean_cds_coverage(bam1, gene, min_mapq)
        cov2 = mean_cds_coverage(bam2, gene, min_mapq)
        if cov1 >= min_coverage and cov2 >= min_coverage:
            passing.append(gname)

    bam1.close()
    bam2.close()
    print(f"  {len(passing):,} / {len(genes):,} genes pass coverage filter.",
          file=sys.stderr)
    return passing


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build per-position editing matrix for a gene
# ─────────────────────────────────────────────────────────────────────────────

def complement_base(b: str) -> str:
    return b.translate(str.maketrans("ACGTacgt", "TGCAtgca"))


def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]


def build_editing_matrix(bam: pysam.AlignmentFile,
                          ref_fasta: pysam.FastaFile,
                          gene: dict,
                          min_mapq: int,
                          min_baseq: int) -> pd.DataFrame:
    """
    For every CDS position in transcript order, compute:
        - ref_base (transcript coordinates)
        - A, G, C, T counts (transcript coordinates)
        - coverage
        - ag_edit_frac = G/(A+G) where ref_base == A, else NaN

    Returns a DataFrame indexed by transcript position (0-based).
    Strand handling: uses read.is_reverse XOR (strand=="-") to convert
    each read to transcript coordinates before counting.
    """
    chrom  = gene["chrom"]
    strand = gene["strand"]
    records = []
    tx_pos  = 0   # transcript coordinate counter

    for (cds_start, cds_end) in gene["cds"]:
        # Genomic positions in transcript order
        if strand == "+":
            gpos_range = range(cds_start, cds_end)
        else:
            gpos_range = range(cds_end - 1, cds_start - 1, -1)

        for gpos in gpos_range:
            ref_base_genomic = ref_fasta.fetch(chrom, gpos, gpos + 1).upper()
            ref_base_tx = complement_base(ref_base_genomic) \
                          if strand == "-" else ref_base_genomic

            counts = collections.Counter()
            for col in bam.pileup(
                chrom, gpos, gpos + 1,
                truncate=True,
                min_mapping_quality=min_mapq,
                min_base_quality=min_baseq,
                stepper="samtools",
            ):
                if col.reference_pos != gpos:
                    continue
                for pread in col.pileups:
                    if pread.is_del or pread.is_refskip:
                        continue
                    qbase_raw = pread.alignment.query_sequence[
                        pread.query_position
                    ].upper()
                    needs_complement = pread.alignment.is_reverse != (strand == "-")
                    qbase = complement_base(qbase_raw) if needs_complement \
                            else qbase_raw
                    counts[qbase] += 1

            cov      = sum(counts.values())
            ag_denom = counts["A"] + counts["G"]
            ag_frac  = counts["G"] / ag_denom \
                       if ref_base_tx == "A" and ag_denom > 0 else np.nan

            records.append({
                "tx_pos":       tx_pos,
                "gpos":         gpos,
                "ref_base":     ref_base_tx,
                "A":            counts["A"],
                "G":            counts["G"],
                "C":            counts["C"],
                "T":            counts["T"],
                "coverage":     cov,
                "ag_edit_frac": ag_frac,
            })
            tx_pos += 1

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Find histidine codon positions in transcript coordinates
# ─────────────────────────────────────────────────────────────────────────────

def find_his_codon_tx_positions(ref_fasta: pysam.FastaFile,
                                 gene: dict) -> list:
    """
    Returns list of transcript positions (0-based) that are the middle base
    (A) of each His codon in the CDS, in transcript order.
    """
    his_positions = []
    tx_pos = 0
    chrom  = gene["chrom"]
    strand = gene["strand"]

    # Concatenate CDS sequence in transcript order
    tx_seq = ""
    for (cds_start, cds_end) in gene["cds"]:
        seg = ref_fasta.fetch(chrom, cds_start, cds_end).upper()
        if strand == "-":
            seg = reverse_complement(seg)
        tx_seq += seg

    # Walk codons
    for i in range(0, len(tx_seq) - 2, 3):
        codon = tx_seq[i:i+3]
        if codon in HIS_CODONS:
            his_positions.append(i + 1)   # middle base of codon

    return his_positions


# ─────────────────────────────────────────────────────────────────────────────
# 5. Plot one gene
# ─────────────────────────────────────────────────────────────────────────────

def plot_gene(gene_name: str,
              df1: pd.DataFrame,
              df2: pd.DataFrame,
              label1: str,
              label2: str,
              his_positions: list,
              pseudo: float,
              pdf_path: str,
              rolling_window: int = 10):
    """
    For one gene, produce a PDF with three stacked panels sharing an x-axis:
      Panel 0: rolling mean edit frequency along CDS, BAM1
      Panel 1: rolling mean edit frequency along CDS, BAM2
      Panel 2: log2FC (BAM2 / BAM1) at each ref=A position

    His codon A positions are marked with vertical lines.
    Uses the pyx package.
    """
    from pyx import canvas, graph, color, style, deco, text as pyx_text

    # Merge on tx_pos, restrict to ref=A
    merged = df1[["tx_pos", "ref_base", "ag_edit_frac"]].merge(
        df2[["tx_pos", "ag_edit_frac"]],
        on="tx_pos", suffixes=(f"_{label1}", f"_{label2}")
    )
    ref_a  = merged[merged["ref_base"] == "A"].copy()
    pos    = ref_a["tx_pos"].values
    frac1  = ref_a[f"ag_edit_frac_{label1}"].fillna(0).values
    frac2  = ref_a[f"ag_edit_frac_{label2}"].fillna(0).values
    log2fc = np.log2((frac2 + pseudo) / (frac1 + pseudo))

    # Rolling mean — use pandas with min_periods=1 so edges aren't dropped
    def rolling_mean(arr):
        return pd.Series(arr).rolling(
            rolling_window, center=True, min_periods=1
        ).mean().values

    frac1_smooth  = rolling_mean(frac1)
    frac2_smooth  = rolling_mean(frac2)
    log2fc_smooth = rolling_mean(log2fc)

    cds_len = len(merged)
    x_min, x_max = 0, max(cds_len, 1)

    col1    = color.cmyk(0, 0, 0, 1)        # black  — BAM1
    col2    = color.cmyk(1, 0.5, 0, 0)      # blue   — BAM2
    col_fc  = color.cmyk(0.5, 0.5, 0, 0.3) # slate  — log2FC line
    col_his = color.cmyk(0, 1, 1, 0)        # red    — His lines
    col_zero = color.cmyk(0, 0, 0, 1)       # black  — zero line

    panel_width  = 12
    panel_height = 3
    panel_gap    = 1.5

    c = canvas.canvas()

    def make_panel(ypos, frac_data, col, bam_label, share_xaxis=None):
        """Build one editing fraction panel with rolling mean line."""
        y_title = f"{bam_label} Edit Freq"
        if share_xaxis is None:
            g = graph.graphxy(
                width=panel_width,
                height=panel_height,
                ypos=ypos,
                x=graph.axis.linear(min=x_min, max=x_max, title=""),
                y=graph.axis.linear(min=0, max=1, title=y_title),
            )
        else:
            g = graph.graphxy(
                width=panel_width,
                height=panel_height,
                ypos=ypos,
                x=graph.axis.linkedaxis(share_xaxis.axes["x"]),
                y=graph.axis.linear(min=0, max=1, title=y_title),
            )

        # His codon vertical lines
        for hp in his_positions:
            g.plot(
                graph.data.function(f"x(y)={hp}", min=0, max=1),
                [graph.style.line([col_his, style.linewidth.thin,
                                   style.linestyle.solid])]
            )

        # Rolling mean edit frequency line
        points = list(zip(pos.tolist(), frac_data.tolist()))
        if points:
            g.plot(
                graph.data.points(points, x=1, y=2),
                [graph.style.line([col, style.linewidth.thin,
                                   style.linestyle.solid])]
            )

        return g

    def make_log2fc_panel(ypos, share_xaxis):
        """Build the log2FC panel with black dashed zero line."""
        y_min = float(np.nanmin(log2fc_smooth)) if len(log2fc_smooth) > 0 else -2
        y_max = float(np.nanmax(log2fc_smooth)) if len(log2fc_smooth) > 0 else  2
        y_abs = max(abs(y_min), abs(y_max), 0.5)
        y_min, y_max = -y_abs * 1.1, y_abs * 1.1

        g = graph.graphxy(
            width=panel_width,
            height=panel_height,
            ypos=ypos,
            x=graph.axis.linkedaxis(share_xaxis.axes["x"]),
            y=graph.axis.linear(min=y_min, max=y_max,
                                title=f"log2FC ({label2}/{label1})"),
        )

        # Black dashed zero line
        g.plot(
            graph.data.function("x(y)=0", min=y_min, max=y_max),
            [graph.style.line([col_zero, style.linewidth.thin,
                               style.linestyle.dashed])]
        )

        # His codon vertical lines
        for hp in his_positions:
            g.plot(
                graph.data.function(f"x(y)={hp}", min=y_min, max=y_max),
                [graph.style.line([col_his, style.linewidth.thin,
                                   style.linestyle.solid])]
            )

        # Rolling mean log2FC line
        points = list(zip(pos.tolist(), log2fc_smooth.tolist()))
        if points:
            g.plot(
                graph.data.points(points, x=1, y=2),
                [graph.style.line([col_fc, style.linewidth.thin,
                                   style.linestyle.solid])]
            )

        return g

    # Build panels — BAM1 first to establish reference x-axis
    y0 = 0
    y1 = panel_height + panel_gap
    y2 = 2 * (panel_height + panel_gap)

    g_bam1   = make_panel(ypos=y2, frac_data=frac1_smooth, col=col1,
                           bam_label=label1, share_xaxis=None)
    g_bam2   = make_panel(ypos=y1, frac_data=frac2_smooth, col=col2,
                           bam_label=label2, share_xaxis=g_bam1)
    g_log2fc = make_log2fc_panel(ypos=y0, share_xaxis=g_bam1)

    c.insert(g_log2fc)
    c.insert(g_bam2)
    c.insert(g_bam1)

    # Title
    pyx_text.set(pyx_text.LatexRunner)
    c.text(panel_width / 2, y2 + panel_height + 0.3,
           gene_name.replace("_", r"\_"),
           [pyx_text.halign.center, pyx_text.size.normalsize])

    c.writePDFfile(pdf_path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Per-gene TadA editing analysis with His codon highlighting."
    )
    p.add_argument("--bam1",   required=True)
    p.add_argument("--bam2",   required=True)
    p.add_argument("--label1", default="BAM1")
    p.add_argument("--label2", default="BAM2")
    p.add_argument("--ref",    required=True)
    p.add_argument("--gtf",    required=True)
    p.add_argument("--output", default="gene_his_analysis")
    p.add_argument("--min_coverage", type=float, default=50.0,
                   help="Min mean CDS coverage in both BAMs (default: 50)")
    p.add_argument("--min_mapq",  type=int, default=20)
    p.add_argument("--min_baseq", type=int, default=10)
    p.add_argument("--pseudo", type=float, default=1e-3,
                   help="Pseudocount for log2FC (default: 1e-3)")
    p.add_argument("--rolling_window", type=int, default=10,
                   help="Rolling mean window size for smoothing (default: 10)")
    p.add_argument("--gene_list", default=None,
                   help="Optional text file of gene names to restrict analysis")
    return p.parse_args()


def main():
    args = parse_args()
    out  = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print("=== Individual Gene Histidine TadA Editing Analysis ===",
          file=sys.stderr)

    # ── Parse GTF ─────────────────────────────────────────────────────────────
    print("\nParsing GTF…", file=sys.stderr)
    genes = parse_gtf(args.gtf)
    print(f"  {len(genes):,} genes parsed.", file=sys.stderr)

    # ── Optional gene list filter ─────────────────────────────────────────────
    if args.gene_list:
        with open(args.gene_list) as fh:
            allowed = {l.strip() for l in fh if l.strip()}
        genes = {k: v for k, v in genes.items() if k in allowed}
        print(f"  {len(genes):,} genes after gene list filter.", file=sys.stderr)

    # ── Coverage filter ───────────────────────────────────────────────────────
    print(f"\nFiltering genes with mean CDS coverage ≥ {args.min_coverage}x "
          f"in both BAMs…", file=sys.stderr)
    passing_genes = filter_high_coverage_genes(
        genes, args.bam1, args.bam2,
        min_coverage=args.min_coverage,
        min_mapq=args.min_mapq,
    )

    if not passing_genes:
        print("ERROR: No genes passed coverage filter.", file=sys.stderr)
        sys.exit(1)

    # ── Open shared resources ─────────────────────────────────────────────────
    ref_fasta = pysam.FastaFile(args.ref)
    bam1      = pysam.AlignmentFile(args.bam1, "rb")
    bam2      = pysam.AlignmentFile(args.bam2, "rb")

    pdf_dir = Path(f"{out}_gene_pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    print(f"\nBuilding editing matrices and plotting {len(passing_genes):,} "
          f"genes → {pdf_dir}/", file=sys.stderr)

    for i, gname in enumerate(passing_genes):
        if (i + 1) % 10 == 0:
            print(f"  Gene {i+1}/{len(passing_genes)}: {gname}…",
                  file=sys.stderr)

        gene = genes[gname]

        # Build editing matrices
        df1 = build_editing_matrix(
            bam1, ref_fasta, gene, args.min_mapq, args.min_baseq
        )
        df2 = build_editing_matrix(
            bam2, ref_fasta, gene, args.min_mapq, args.min_baseq
        )

        if df1.empty or df2.empty:
            continue

        # Find His codon positions in transcript coordinates
        his_positions = find_his_codon_tx_positions(ref_fasta, gene)

        # Summary stats
        ref_a1 = df1[df1["ref_base"] == "A"]["ag_edit_frac"].dropna()
        ref_a2 = df2[df2["ref_base"] == "A"]["ag_edit_frac"].dropna()
        his_a1 = df1[df1["tx_pos"].isin(his_positions)]["ag_edit_frac"].dropna()
        his_a2 = df2[df2["tx_pos"].isin(his_positions)]["ag_edit_frac"].dropna()

        summary_rows.append({
            "gene":                          gname,
            "chrom":                         gene["chrom"],
            "strand":                        gene["strand"],
            "n_his_codons":                  len(his_positions),
            "cds_length":                    len(df1),
            f"mean_edit_{args.label1}":      ref_a1.mean() if len(ref_a1) else np.nan,
            f"mean_edit_{args.label2}":      ref_a2.mean() if len(ref_a2) else np.nan,
            f"mean_his_edit_{args.label1}":  his_a1.mean() if len(his_a1) else np.nan,
            f"mean_his_edit_{args.label2}":  his_a2.mean() if len(his_a2) else np.nan,
        })

        # One PDF per gene
        safe_name = re.sub(r"[^\w\-]", "_", gname)
        gene_pdf  = str(pdf_dir / safe_name)
        plot_gene(
            gname, df1, df2,
            args.label1, args.label2,
            his_positions,
            pseudo=args.pseudo,
            pdf_path=gene_pdf,
            rolling_window=args.rolling_window,
        )

    ref_fasta.close()
    bam1.close()
    bam2.close()

    # ── Save summary table ────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    pseudo = args.pseudo
    summary_df["his_vs_background_log2fc"] = np.log2(
        (summary_df[f"mean_his_edit_{args.label2}"] + pseudo) /
        (summary_df[f"mean_his_edit_{args.label1}"] + pseudo)
    ) - np.log2(
        (summary_df[f"mean_edit_{args.label2}"] + pseudo) /
        (summary_df[f"mean_edit_{args.label1}"] + pseudo)
    )
    summary_df = summary_df.sort_values(
        "his_vs_background_log2fc", ascending=True
    )
    summary_csv = f"{out}_gene_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n  Saved summary → {summary_csv}", file=sys.stderr)
    print(f"  Gene PDFs → {pdf_dir}/", file=sys.stderr)
    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()