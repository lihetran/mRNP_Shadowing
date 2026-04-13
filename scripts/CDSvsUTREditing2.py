#!/usr/bin/env python3
"""
CDS vs UTR Editing CDF
=======================
For each high-coverage gene, computes the A->G editing fraction at every
ref=A position across the full gene body, then separates positions into
CDS and UTR. Plots CDFs of per-position editing fraction for CDS vs UTR
for each BAM, overlaid on the same axes.

Usage:
    python3 cdsVsUtrEditingCDF.py \
        --bam1 condition1.bam --label1 "WT" \
        --bam2 condition2.bam --label2 "3-AT" \
        --ref reference.fa \
        --gtf annotation.gtf \
        --output output_prefix \
        [--min_coverage 50] \
        [--min_mapq 20] \
        [--min_baseq 10] \
        [--gene_list genes.txt]

Requirements:
    pip install pysam pandas numpy
    pyx (for plotting)
"""

import argparse
import sys
import re
import collections
from pathlib import Path

import pysam
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. GTF parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_gtf(gtf_path: str) -> dict:
    """
    Parse GTF for gene body extent, CDS intervals, and UTR intervals.
    Returns:
        gene_name -> {
            chrom, strand, gene_start, gene_end,
            cds:  [(start0, end0), ...],
            utrs: [(start0, end0), ...],   # 5' and 3' UTR combined
        }
    """
    genes        = {}
    gene_extents = {}
    cds_by_gene  = collections.defaultdict(list)
    utr_by_gene  = collections.defaultdict(list)

    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            feature = fields[2]
            chrom   = fields[0]
            start   = int(fields[3]) - 1
            end     = int(fields[4])
            strand  = fields[6]
            m_gn    = re.search(r'gene_name "([^"]+)"', fields[8])
            m_tid   = re.search(r'transcript_id "([^"]+)"', fields[8])
            gname   = m_gn.group(1)  if m_gn  else None
            tid     = m_tid.group(1) if m_tid else "."
            if gname is None:
                continue

            if feature == "gene":
                gene_extents[gname] = (chrom, strand, start, end, tid)
            elif feature == "CDS":
                cds_by_gene[gname].append((start, end))
                if gname not in genes:
                    genes[gname] = {
                        "chrom": chrom, "strand": strand,
                        "gene_name": gname, "transcript": tid,
                    }
            elif feature in ("UTR", "five_prime_utr", "three_prime_utr"):
                utr_by_gene[gname].append((start, end))

    # Attach extents and intervals
    for gname, g in genes.items():
        if gname in gene_extents:
            _, _, gs, ge, _ = gene_extents[gname]
            g["gene_start"] = gs
            g["gene_end"]   = ge
        else:
            g["gene_start"] = min(s for s, e in cds_by_gene[gname])
            g["gene_end"]   = max(e for s, e in cds_by_gene[gname])
        g["cds"]  = sorted(cds_by_gene[gname])
        g["utrs"] = sorted(utr_by_gene.get(gname, []))

    # Only keep genes that have both CDS and UTR annotations
    genes = {k: v for k, v in genes.items()
             if v["cds"] and v["utrs"]}
    print(f"  {len(genes):,} genes with both CDS and UTR annotations.",
          file=sys.stderr)
    return genes


def complement_base(b: str) -> str:
    return b.translate(str.maketrans("ACGTacgt", "TGCAtgca"))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Coverage filter (CDS only, consistent with other scripts)
# ─────────────────────────────────────────────────────────────────────────────

def mean_cds_coverage(bam: pysam.AlignmentFile, gene: dict,
                       min_mapq: int) -> float:
    total, bases = 0, 0
    for (s, e) in gene["cds"]:
        for col in bam.pileup(gene["chrom"], s, e, truncate=True,
                              min_mapping_quality=min_mapq,
                              stepper="samtools"):
            total += col.nsegments
            bases += 1
    return total / bases if bases > 0 else 0.0


def filter_genes(genes: dict, bam1_path: str, bam2_path: str,
                  min_coverage: float, min_mapq: int) -> list:
    passing = []
    bam1 = pysam.AlignmentFile(bam1_path, "rb")
    bam2 = pysam.AlignmentFile(bam2_path, "rb")
    for i, (gname, gene) in enumerate(genes.items()):
        if (i + 1) % 200 == 0:
            print(f"  Coverage check {i+1}/{len(genes)}…", file=sys.stderr)
        if (mean_cds_coverage(bam1, gene, min_mapq) >= min_coverage and
                mean_cds_coverage(bam2, gene, min_mapq) >= min_coverage):
            passing.append(gname)
    bam1.close()
    bam2.close()
    print(f"  {len(passing):,}/{len(genes):,} genes pass coverage filter.",
          file=sys.stderr)
    return passing


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build per-position editing fractions across the gene body
# ─────────────────────────────────────────────────────────────────────────────

def build_position_edit_fracs(bam: pysam.AlignmentFile,
                               ref_fasta: pysam.FastaFile,
                               gene: dict,
                               min_mapq: int,
                               min_baseq: int) -> pd.DataFrame:
    """
    For every ref=A position in the full gene body, compute ag_edit_frac
    and label it as CDS or UTR.

    Returns DataFrame with columns:
        gpos, region (CDS|UTR), ag_edit_frac
    Only positions with at least one A or G read are included.
    """
    chrom      = gene["chrom"]
    strand     = gene["strand"]
    gene_start = gene["gene_start"]
    gene_end   = gene["gene_end"]

    # Build genomic position sets for fast lookup
    cds_positions = set()
    for (s, e) in gene["cds"]:
        cds_positions.update(range(s, e))

    utr_positions = set()
    for (s, e) in gene["utrs"]:
        utr_positions.update(range(s, e))

    chrom_seq = ref_fasta.fetch(chrom).upper()
    records   = []

    for gpos in range(gene_start, gene_end):
        ref_base_genomic = chrom_seq[gpos]
        ref_base_tx      = complement_base(ref_base_genomic) \
                           if strand == "-" else ref_base_genomic

        if ref_base_tx != "A":
            continue

        # Determine region
        if gpos in cds_positions:
            region = "CDS"
        elif gpos in utr_positions:
            region = "UTR"
        else:
            continue   # intronic or unannotated — skip

        counts = collections.Counter()
        for col in bam.pileup(chrom, gpos, gpos + 1, truncate=True,
                              min_mapping_quality=min_mapq,
                              min_base_quality=min_baseq,
                              stepper="samtools"):
            if col.reference_pos != gpos:
                continue
            for pread in col.pileups:
                if pread.is_del or pread.is_refskip:
                    continue
                qbase_raw = pread.alignment.query_sequence[
                    pread.query_position].upper()
                needs_complement = (pread.alignment.is_reverse !=
                                    (strand == "-"))
                qbase = complement_base(qbase_raw) \
                        if needs_complement else qbase_raw
                if qbase in ("A", "G"):
                    counts[qbase] += 1

        ag_total = counts["A"] + counts["G"]
        if ag_total == 0:
            continue

        records.append({
            "gpos":         gpos,
            "region":       region,
            "ag_edit_frac": counts["G"] / ag_total,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Plot CDFs using pyx
# ─────────────────────────────────────────────────────────────────────────────

def plot_cdfs_pyx(fracs: dict, label1: str, label2: str,
                   output_prefix: str):
    """
    fracs = {
        label1: {"CDS": array, "UTR": array},
        label2: {"CDS": array, "UTR": array},
    }

    Layout: 2 panels side by side — one per BAM.
    Each panel shows two CDF lines: CDS (solid) and UTR (dashed).
    """
    from pyx import canvas, graph, color, style, path, text as pyx_text

    col1    = color.cmyk(0, 0, 0, 1)        # black — BAM1
    col2    = color.cmyk(1, 0.5, 0, 0)      # blue  — BAM2
    ls_cds  = style.linestyle.solid
    ls_utr  = style.linestyle.dashed

    panel_w = 6
    panel_h = 5
    gap     = 2.0

    c = canvas.canvas()

    def _cdf_graph(xpos, bam_label, bam_col, frac_dict):
        g = graph.graphxy(
            width=panel_w, height=panel_h,
            xpos=xpos, ypos=0,
            x=graph.axis.linear(min=0, max=1,
                                title="A->G edit frac"),
            y=graph.axis.linear(min=0, max=1,
                                title="Cumulative fraction"),
        )

        for region, ls in [("CDS", ls_cds), ("UTR", ls_utr)]:
            vals = frac_dict.get(region, np.array([]))
            if len(vals) == 0:
                continue
            sf  = np.sort(vals)
            cdf = np.arange(1, len(sf) + 1) / len(sf)
            pts = list(zip(sf.tolist(), cdf.tolist()))
            g.plot(graph.data.points(pts, x=1, y=2),
                   [graph.style.line([bam_col, style.linewidth.normal, ls])])

        c.insert(g)

        # Title
        c.text(g.xpos + g.width / 2., g.ypos + g.height + 0.4,
               bam_label,
               [pyx_text.halign.center, pyx_text.size.normalsize])

        return g

    g1 = _cdf_graph(xpos=0,            bam_label=label1,
                    bam_col=col1, frac_dict=fracs[label1])
    g2 = _cdf_graph(xpos=panel_w + gap, bam_label=label2,
                    bam_col=col2, frac_dict=fracs[label2])

    # Legend on the right of the second panel
    leg_x   = g2.xpos + g2.width + 0.4
    leg_lw  = 0.8
    leg_dy  = 0.55

    # Use black for legend lines since styles distinguish CDS vs UTR
    for j, (region, ls) in enumerate([("CDS", ls_cds), ("UTR", ls_utr)]):
        ly = g2.ypos + g2.height - 0.3 - j * leg_dy
        c.stroke(path.line(leg_x, ly, leg_x + leg_lw, ly),
                 [color.cmyk(0, 0, 0, 1), style.linewidth.normal, ls])
        c.text(leg_x + leg_lw + 0.15, ly, region,
               [pyx_text.valign.middle, pyx_text.size.small])

    # Summary stats as text below each panel
    for g, bam_label in [(g1, label1), (g2, label2)]:
        fd = fracs[bam_label]
        lines = []
        for region in ["CDS", "UTR"]:
            vals = fd.get(region, np.array([]))
            if len(vals) > 0:
                lines.append(f"{region}: n={len(vals):,}, "
                             f"med={np.median(vals):.4f}")
        for k, line in enumerate(lines):
            c.text(g.xpos, g.ypos - 0.4 - k * 0.45, line,
                   [pyx_text.size.tiny])

    plot_path = f"{output_prefix}_cds_vs_utr_cdf_pyx"
    c.writePDFfile(plot_path)
    print(f"  Saved CDS vs UTR CDF -> {plot_path}.pdf", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CDF of A->G editing fraction: CDS vs UTR positions."
    )
    p.add_argument("--bam1",   required=True)
    p.add_argument("--bam2",   required=True)
    p.add_argument("--label1", default="BAM1")
    p.add_argument("--label2", default="BAM2")
    p.add_argument("--ref",    required=True)
    p.add_argument("--gtf",    required=True)
    p.add_argument("--output", default="cds_vs_utr")
    p.add_argument("--min_coverage", type=float, default=50.0,
                   help="Min mean CDS coverage in both BAMs (default: 50)")
    p.add_argument("--min_mapq",  type=int, default=20)
    p.add_argument("--min_baseq", type=int, default=10)
    p.add_argument("--gene_list", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out  = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print("=== CDS vs UTR Editing CDF ===", file=sys.stderr)

    # ── Parse GTF ─────────────────────────────────────────────────────────────
    print("\nParsing GTF...", file=sys.stderr)
    genes = parse_gtf(args.gtf)

    if args.gene_list:
        with open(args.gene_list) as fh:
            allowed = {l.strip() for l in fh if l.strip()}
        genes = {k: v for k, v in genes.items() if k in allowed}
        print(f"  {len(genes):,} after gene list filter.", file=sys.stderr)

    if not genes:
        print("ERROR: No genes with CDS and UTR annotations.", file=sys.stderr)
        sys.exit(1)

    # ── Coverage filter ───────────────────────────────────────────────────────
    print(f"\nFiltering genes (mean CDS coverage >= {args.min_coverage}x)...",
          file=sys.stderr)
    passing = filter_genes(
        genes, args.bam1, args.bam2,
        min_coverage=args.min_coverage, min_mapq=args.min_mapq,
    )
    if not passing:
        print("ERROR: No genes passed coverage filter.", file=sys.stderr)
        sys.exit(1)

    # ── Collect per-position edit fracs ───────────────────────────────────────
    ref_fasta = pysam.FastaFile(args.ref)

    # fracs[label][region] = list of ag_edit_frac values
    fracs = {
        args.label1: {"CDS": [], "UTR": []},
        args.label2: {"CDS": [], "UTR": []},
    }

    for key, bam_path, label in [
        ("bam1", args.bam1, args.label1),
        ("bam2", args.bam2, args.label2),
    ]:
        print(f"\nPiling up {label}...", file=sys.stderr)
        bam = pysam.AlignmentFile(bam_path, "rb")

        for i, gname in enumerate(passing):
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(passing)}: {gname}...",
                      file=sys.stderr)
            gene = genes[gname]
            df   = build_position_edit_fracs(
                bam, ref_fasta, gene, args.min_mapq, args.min_baseq
            )
            if df.empty:
                continue
            for region in ["CDS", "UTR"]:
                vals = df[df["region"] == region]["ag_edit_frac"].values
                fracs[label][region].extend(vals.tolist())

        bam.close()
        for region in ["CDS", "UTR"]:
            n = len(fracs[label][region])
            med = np.median(fracs[label][region]) if n > 0 else float("nan")
            print(f"  [{label}] {region}: {n:,} positions, "
                  f"median edit frac = {med:.4f}", file=sys.stderr)

    ref_fasta.close()

    # Convert to arrays
    for label in [args.label1, args.label2]:
        for region in ["CDS", "UTR"]:
            fracs[label][region] = np.array(fracs[label][region])

    # ── Save summary CSV ───────────────────────────────────────────────────────
    rows = []
    for label in [args.label1, args.label2]:
        for region in ["CDS", "UTR"]:
            vals = fracs[label][region]
            if len(vals) == 0:
                continue
            rows.append({
                "label":         label,
                "region":        region,
                "n_positions":   len(vals),
                "mean_edit":     float(np.mean(vals)),
                "median_edit":   float(np.median(vals)),
                "pct25_edit":    float(np.percentile(vals, 25)),
                "pct75_edit":    float(np.percentile(vals, 75)),
            })
    pd.DataFrame(rows).to_csv(f"{out}_summary.csv", index=False)
    print(f"\n  Saved summary -> {out}_summary.csv", file=sys.stderr)

    # ── Plot ───────────────────────────────────────────────────────────────────
    print("\nPlotting CDFs (pyx)...", file=sys.stderr)
    try:
        plot_cdfs_pyx(fracs, args.label1, args.label2, out)
    except Exception as e:
        print(f"  WARNING: pyx plotting failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()