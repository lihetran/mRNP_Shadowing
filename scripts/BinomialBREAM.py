#!/usr/bin/env python3
"""
Binomial Shadowing Analysis (parquet input)
===========================================
For each high-coverage gene, slides a window across the transcript and
tests whether each read shows fewer A->G edits than expected under a
binomial null model derived from the reference/WT library (parquet1).

At each window position for each read:
  - n = number of ref=A sites covered by the read in the window
  - k = number of those sites showing a G (i.e. edited)
  - p = mean background edit probability across those sites (from parquet1)
  - p_value = P(X <= k | n, p)  [lower-tail binomial test for protection]

A low p-value means the read has fewer edits than expected — consistent
with protection (shadowing) in that window.

Uses absolute_indices + edit_string (sense-oriented by the parquet generator)
rather than aligned_pairs, matching the approach in histidineMetaFromParquet.py.
Loads all parquet chunks once upfront rather than per gene.

Usage:
    python3 binomialShadowing.py \
        --parquet1 reference_chunks/ --label1 "WT" \
        --parquet2 query_chunks/     --label2 "3-AT" \
        --ref reference.fa \
        --gtf annotation.gtf \
        --output output_prefix \
        [--window 30] \
        [--min_coverage 50] \
        [--min_sites 5] \
        [--num_reads 10] \
        [--gene_list genes.txt]
"""

import argparse
import sys
import re
import math
import collections
from pathlib import Path

import pysam
import numpy as np
import pandas as pd
import scipy.stats
from logJosh import Tee


HIS_CODONS = {"CAT", "CAC"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Helpers
# ─────────────────────────────────────────────────────────────────────────────

def complement_base(b: str) -> str:
    return b.translate(str.maketrans("ACGTacgt", "TGCAtgca"))

def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]


# ─────────────────────────────────────────────────────────────────────────────
# 2. GTF parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_gtf(gtf_path: str) -> dict:
    genes = {}
    gene_extents = {}
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
                gene_extents[gname] = (start, end)
            if feature == "CDS":
                if gname not in genes:
                    genes[gname] = {
                        "chrom": chrom, "strand": strand,
                        "transcript": tid, "gene_name": gname, "cds": [],
                    }
                genes[gname]["cds"].append((start, end))
    for gname, g in genes.items():
        if gname in gene_extents:
            g["gene_start"], g["gene_end"] = gene_extents[gname]
        else:
            g["gene_start"] = min(s for s, e in g["cds"])
            g["gene_end"]   = max(e for s, e in g["cds"])
        g["cds"].sort(key=lambda x: x[0], reverse=(g["strand"] == "-"))
        g["cds_genomic_start"] = min(s for s, e in g["cds"])
        g["cds_genomic_end"]   = max(e for s, e in g["cds"])
    return genes


def cds_length(gene: dict) -> int:
    return sum(ce - cs for cs, ce in gene["cds"])


def find_his_codon_tx_positions(ref_fasta, gene):
    chrom, strand = gene["chrom"], gene["strand"]
    tx_seq = ""
    for (cs, ce) in gene["cds"]:
        seg = ref_fasta.fetch(chrom, cs, ce).upper()
        if strand == "-":
            seg = reverse_complement(seg)
        tx_seq += seg
    his_pos, seen = [], set()
    for i in range(0, len(tx_seq) - 2, 3):
        if tx_seq[i:i+3] in HIS_CODONS:
            p = i + 1
            if p not in seen:
                seen.add(p)
                his_pos.append(p)
    return his_pos


# ─────────────────────────────────────────────────────────────────────────────
# 3. Parquet loading — load once upfront
# ─────────────────────────────────────────────────────────────────────────────

def load_all_parquet_chunks(parquet_dir: str) -> pd.DataFrame:
    parquet_dir = Path(parquet_dir)
    chunks = sorted(parquet_dir.glob("*.parquet"))
    if not chunks:
        return pd.DataFrame()
    dfs = [pd.read_parquet(c) for c in chunks]
    df  = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df):,} reads from {len(chunks)} chunk(s).",
          file=sys.stderr)
    return df


def get_gene_df(df_all: pd.DataFrame, gene: dict,
                cds_spanning: bool = False) -> pd.DataFrame:
    """
    Fast vectorised pre-filter to reads overlapping this gene.

    If cds_spanning is True, only keep reads whose alignment spans the full
    CDS (read_start <= cds_genomic_start and read_end >= cds_genomic_end),
    so every read had the opportunity to be edited at every position.
    """
    mask = ((df_all["chrom"]       == gene["chrom"]) &
            (df_all["gene_strand"] == gene["strand"]))
    if "read_start" in df_all.columns and "read_end" in df_all.columns:
        if cds_spanning:
            cds_start = gene.get("cds_genomic_start", gene["gene_start"])
            cds_end   = gene.get("cds_genomic_end",   gene["gene_end"])
            mask &= ((df_all["read_start"] <= cds_start) &
                     (df_all["read_end"]   >= cds_end))
        else:
            mask &= ((df_all["read_start"] < gene["gene_end"]) &
                     (df_all["read_end"]   > gene["gene_start"]))
    return df_all[mask]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Build gpos -> tx_pos map and ref frequencies
#    Uses absolute_indices + edit_string (sense-oriented, no aligned_pairs)
# ─────────────────────────────────────────────────────────────────────────────

def _gpos_to_tx_map(gene: dict, ref_fasta: pysam.FastaFile) -> dict:
    """
    Map genomic position -> spliced CDS tx_pos for all ref=A positions.
    tx_pos is contiguous over CDS segments in transcript order.
    Only ref=A (transcript-sense) positions are included.
    """
    chrom     = gene["chrom"]
    strand    = gene["strand"]
    chrom_seq = ref_fasta.fetch(chrom).upper()

    gpos_to_tx = {}
    tx_pos = 0
    for (cs, ce) in gene["cds"]:
        if strand == "+":
            for gpos in range(cs, ce):
                if chrom_seq[gpos] == "A":
                    gpos_to_tx[gpos] = tx_pos
                tx_pos += 1
        else:
            for gpos in range(ce - 1, cs - 1, -1):
                if chrom_seq[gpos] == "T":
                    gpos_to_tx[gpos] = tx_pos
                tx_pos += 1
    return gpos_to_tx


def build_reference_freq(df: pd.DataFrame, gpos_to_tx: dict,
                          gene: dict) -> dict:
    """
    {tx_pos: p_edit} from parquet1 using absolute_indices + edit_string.

    absolute_indices and edit_string are both sense-oriented by the parquet
    generator, so index i in edit_string matches index i in absolute_indices.
    edit_string '1' = A->G edit, '0' = no edit at ref=A, '2' = indel/skip.
    Ref=A verification is implicit: gpos_to_tx only contains ref=A positions.
    """
    if df.empty:
        return {}

    gene_start = gene["gene_start"]
    gene_end   = gene["gene_end"]
    min_gp     = min(gpos_to_tx.keys(), default=None)
    max_gp     = max(gpos_to_tx.keys(), default=None)
    if min_gp is None:
        return {}

    # Pre-filter to reads spanning ref=A positions
    if "read_start" in df.columns and "read_end" in df.columns:
        sub = df[(df["read_start"] <= max_gp) & (df["read_end"] >= min_gp)]
    else:
        sub = df

    edit_counts = collections.defaultdict(lambda: [0, 0])

    for read in sub.itertuples():
        edit_str    = read.edit_string
        abs_indices = read.absolute_indices
        n_edit      = len(edit_str)

        for i, ref_pos in enumerate(abs_indices):
            if ref_pos is None:
                continue
            if isinstance(ref_pos, float) and ref_pos != ref_pos:
                continue
            ref_pos = int(ref_pos)
            if ref_pos < min_gp or ref_pos > max_gp:
                continue
            if ref_pos not in gpos_to_tx:
                continue
            if i >= n_edit:
                continue
            ev = edit_str[i]
            if ev == "2":
                continue
            edit_counts[gpos_to_tx[ref_pos]][int(ev)] += 1

    ref_freq = {}
    for tx, (n0, n1) in edit_counts.items():
        total = n0 + n1
        if total > 0:
            ref_freq[tx] = max(1e-6, min(1 - 1e-6, n1 / total))
    return ref_freq


def collect_read_edits(df: pd.DataFrame, gpos_to_tx: dict,
                        gene: dict) -> dict:
    """
    {read_id: {tx_pos: 0_or_1}} from parquet using absolute_indices +
    edit_string. Restricts to ref=A positions via gpos_to_tx.
    """
    if df.empty:
        return {}

    min_gp = min(gpos_to_tx.keys(), default=None)
    max_gp = max(gpos_to_tx.keys(), default=None)
    if min_gp is None:
        return {}

    if "read_start" in df.columns and "read_end" in df.columns:
        sub = df[(df["read_start"] <= max_gp) & (df["read_end"] >= min_gp)]
    else:
        sub = df

    read_edits = collections.defaultdict(dict)

    for read in sub.itertuples():
        edit_str    = read.edit_string
        abs_indices = read.absolute_indices
        n_edit      = len(edit_str)

        for i, ref_pos in enumerate(abs_indices):
            if ref_pos is None:
                continue
            if isinstance(ref_pos, float) and ref_pos != ref_pos:
                continue
            ref_pos = int(ref_pos)
            if ref_pos < min_gp or ref_pos > max_gp:
                continue
            if ref_pos not in gpos_to_tx:
                continue
            if i >= n_edit:
                continue
            ev = edit_str[i]
            if ev == "2":
                continue
            read_edits[read.read_id][gpos_to_tx[ref_pos]] = int(ev)

    return dict(read_edits)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Coverage check — fast read count rather than pileup depth
# ─────────────────────────────────────────────────────────────────────────────

def passes_coverage(df_ref: pd.DataFrame, df_qry: pd.DataFrame,
                    min_coverage: float) -> bool:
    """
    Use read count as a fast coverage proxy. Both libraries must have at
    least min_coverage reads overlapping the gene.
    """
    return len(df_ref) >= min_coverage and len(df_qry) >= min_coverage


# ─────────────────────────────────────────────────────────────────────────────
# 6. Binomial p-values per read per window
# ─────────────────────────────────────────────────────────────────────────────

def compute_binomial_pvals_per_read(read_edits: dict, ref_freq: dict,
                                     nt_window: int, min_sites: int,
                                     gene_len: int) -> dict:
    """
    Returns {read_id: [(window_centre, -log10(p_value)), ...]}
    Lower-tail binomial: small p = fewer edits than expected = protection.
    """
    results = {}

    for read_id, pos_dict in read_edits.items():
        trace = []
        for start in range(0, gene_len):
            ks, ps = [], []
            for tx in range(start, start + nt_window):
                if tx in pos_dict and tx in ref_freq:
                    ks.append(pos_dict[tx])
                    ps.append(ref_freq[tx])

            n = len(ks)
            if n < min_sites:
                continue

            k      = sum(ks)
            p_mean = float(np.mean(ps))
            p_val  = scipy.stats.binom.cdf(k, n, p_mean)
            p_val  = max(p_val, 1e-300)

            trace.append((start + nt_window / 2., -math.log10(p_val)))

        if trace:
            results[read_id] = trace

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7. Meta aggregation
# ─────────────────────────────────────────────────────────────────────────────

def get_meta_read(binomial_scores: dict) -> list:
    by_pos = collections.defaultdict(list)
    for trace in binomial_scores.values():
        for centre, v in trace:
            by_pos[centre].append(v)

    return [
        (centre,
         float(np.quantile(vals, 0.05)),
         float(np.quantile(vals, 0.50)),
         float(np.quantile(vals, 0.95)))
        for centre, vals in sorted(by_pos.items())
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 8. Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_gene_pyx(gene_name, meta, binomial_scores, his_positions,
                   read_edits, label1, label2, gene_len, num_reads, pdf_path):
    from pyx import canvas, graph, color, style, text as pyx_text

    col_qry  = color.cmyk(1, 0.5, 0, 0)
    col_his  = color.cmyk(0, 1, 1, 0)
    col_sig  = color.cmyk(0, 0, 0, 0.4)
    col_edit = color.cmyk(0, 0, 0, 1)
    col_no   = color.cmyk(0, 0.8, 1, 0)

    x_min, x_max = 0, gene_len
    panel_w  = 12
    tick_h   = 0.25
    read_h   = 1.2
    gap      = 0.6
    sig_line = -math.log10(0.05)

    all_y      = [e[3] for e in meta]
    y_max_meta = max(max(all_y) * 1.1, 2.0) if all_y else 5.0

    c         = canvas.canvas()
    meta_ypos = num_reads * (read_h + tick_h + gap) + gap * 2

    g_meta = graph.graphxy(
        width=panel_w, height=3, xpos=0, ypos=meta_ypos,
        x=graph.axis.linear(min=x_min, max=x_max,
                            title="Position Along Transcript (nt)"),
        y=graph.axis.linear(min=0, max=y_max_meta,
                            title="-log10(p)"),
    )
    for hp in his_positions:
        g_meta.plot(
            graph.data.function(f"x(y)={hp}", min=0, max=y_max_meta),
            [graph.style.line([col_his, style.linewidth.thin,
                               style.linestyle.solid])])
    g_meta.plot(
        graph.data.function(f"y(x)={sig_line:.4f}", min=x_min, max=x_max),
        [graph.style.line([col_sig, style.linewidth.thin,
                           style.linestyle.dashed])])
    for col_idx in [1, 3]:
        pts = [(e[0], e[col_idx]) for e in meta]
        if pts:
            g_meta.plot(graph.data.points(pts, x=1, y=2),
                        [graph.style.line([col_qry, style.linewidth.thin,
                                           style.linestyle.dotted])])
    med_pts = [(e[0], e[2]) for e in meta]
    if med_pts:
        g_meta.plot(graph.data.points(med_pts, x=1, y=2),
                    [graph.style.line([col_qry, style.linewidth.normal,
                                       style.linestyle.solid])])
    c.insert(g_meta)
    c.text(g_meta.xpos + g_meta.width / 2.,
           g_meta.ypos + g_meta.height + 0.4,
           f"{gene_name} - {label2} binomial protection (ref: {label1})",
           [pyx_text.halign.center, pyx_text.size.normalsize])

    read_ids = list(binomial_scores.keys())[:num_reads]
    for jj, read_id in enumerate(read_ids):
        trace = binomial_scores[read_id]
        ypos  = (num_reads - 1 - jj) * (read_h + tick_h + gap)
        y_max_read = max(max(v for _, v in trace) * 1.1, 2.0)

        g_read = graph.graphxy(
            width=panel_w, height=read_h, xpos=0, ypos=ypos + tick_h,
            x=graph.axis.linkedaxis(g_meta.axes["x"]),
            y=graph.axis.linear(min=0, max=y_max_read, title=""),
        )
        for hp in his_positions:
            g_read.plot(
                graph.data.function(f"x(y)={hp}", min=0, max=y_max_read),
                [graph.style.line([col_his, style.linewidth.thin,
                                   style.linestyle.solid])])
        if sig_line <= y_max_read:
            g_read.plot(
                graph.data.function(f"y(x)={sig_line:.4f}",
                                    min=x_min, max=x_max),
                [graph.style.line([col_sig, style.linewidth.thin,
                                   style.linestyle.dashed])])
        g_read.plot(
            graph.data.points(list(trace), x=1, y=2),
            [graph.style.line([col_qry, style.linewidth.thin,
                               style.linestyle.solid])])
        c.insert(g_read)

        g_ticks = graph.graphxy(
            width=panel_w, height=tick_h, xpos=0, ypos=ypos,
            x=graph.axis.linkedaxis(g_meta.axes["x"]),
            y=graph.axis.linear(min=0, max=1),
        )
        for tx, v in read_edits.get(read_id, {}).items():
            col = col_edit if v == 1 else col_no
            g_ticks.plot(
                graph.data.function(f"x(y)={tx}", min=0, max=1),
                [graph.style.line([col, style.linewidth.thin,
                                   style.linestyle.solid])])
        c.insert(g_ticks)
        c.text(panel_w + 0.15, ypos + tick_h + read_h / 2.,
               read_id[:20], [pyx_text.valign.middle, pyx_text.size.tiny])

    c.writePDFfile(pdf_path)


# ─────────────────────────────────────────────────────────────────────────────
# 9. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Binomial test for editing protection in sliding windows."
    )
    p.add_argument("--parquet1",     required=True)
    p.add_argument("--parquet2",     required=True)
    p.add_argument("--label1",       default="BAM1")
    p.add_argument("--label2",       default="BAM2")
    p.add_argument("--ref",          required=True)
    p.add_argument("--gtf",          required=True)
    p.add_argument("--output",       default="binomialShadow")
    p.add_argument("--window",       type=int,   default=30)
    p.add_argument("--min_coverage", type=float, default=50.0)
    p.add_argument("--min_sites",    type=int,   default=5)
    p.add_argument("--num_reads",    type=int,   default=10)
    p.add_argument("--top_n_plots",  type=int,   default=10,
                   help="Number of genes to plot, ranked by number of "
                        "CDS-spanning query reads (default: 10)")
    p.add_argument("--gene_list",    default=None)
    p.add_argument("--cds_spanning", action="store_true",
                   help="Only include reads whose alignment spans the full "
                        "CDS of the gene.")
    return p.parse_args()


def main():
    args = parse_args()
    out  = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print("=== Binomial Shadowing Analysis (parquet) ===", file=sys.stderr)
    if args.cds_spanning:
        print("  CDS spanning filter: ON", file=sys.stderr)

    print("\nParsing GTF...", file=sys.stderr)
    genes = parse_gtf(args.gtf)
    print(f"  {len(genes):,} genes.", file=sys.stderr)

    if args.gene_list:
        with open(args.gene_list) as fh:
            allowed = {l.strip() for l in fh if l.strip()}
        genes = {k: v for k, v in genes.items() if k in allowed}
        print(f"  {len(genes):,} after gene list filter.", file=sys.stderr)

    # Load both parquets once upfront
    print(f"\nLoading {args.label1} parquet...", file=sys.stderr)
    df_all_ref = load_all_parquet_chunks(args.parquet1)
    print(f"\nLoading {args.label2} parquet...", file=sys.stderr)
    df_all_qry = load_all_parquet_chunks(args.parquet2)

    if df_all_ref.empty or df_all_qry.empty:
        print("ERROR: One or both parquet directories are empty.",
              file=sys.stderr)
        sys.exit(1)

    ref_fasta = pysam.FastaFile(args.ref)

    pdf_dir = Path(f"{out}_gene_pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    n_pass = 0

    gene_names = list(genes.keys())
    print(f"\nProcessing {len(gene_names):,} genes...", file=sys.stderr)

    # Diagnostics for spanning filter
    max_spanning_ref = 0
    max_spanning_qry = 0
    genes_with_any_spanning = 0

    # Stash plot data per gene; plot only the top-N after the loop
    plot_data = {}

    for i, gname in enumerate(gene_names):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(gene_names)} scanned, "
                  f"{n_pass} passing...", file=sys.stderr)

        gene     = genes[gname]
        gene_len = cds_length(gene)

        # Fast vectorised pre-filter from the already-loaded DataFrame
        df_ref = get_gene_df(df_all_ref, gene, cds_spanning=args.cds_spanning)
        df_qry = get_gene_df(df_all_qry, gene, cds_spanning=args.cds_spanning)

        # Track best-case spanning counts for diagnostics
        if len(df_ref) > 0 and len(df_qry) > 0:
            genes_with_any_spanning += 1
        max_spanning_ref = max(max_spanning_ref, len(df_ref))
        max_spanning_qry = max(max_spanning_qry, len(df_qry))

        if not passes_coverage(df_ref, df_qry, args.min_coverage):
            continue
        n_pass += 1

        gpos_to_tx = _gpos_to_tx_map(gene, ref_fasta)
        if not gpos_to_tx:
            continue

        ref_freq = build_reference_freq(df_ref, gpos_to_tx, gene)
        if not ref_freq:
            continue

        read_edits = collect_read_edits(df_qry, gpos_to_tx, gene)
        if not read_edits:
            continue

        binomial_scores = compute_binomial_pvals_per_read(
            read_edits, ref_freq,
            nt_window=args.window,
            min_sites=args.min_sites,
            gene_len=gene_len,
        )
        if not binomial_scores:
            continue

        meta          = get_meta_read(binomial_scores)
        his_positions = find_his_codon_tx_positions(ref_fasta, gene)

        all_scores = [v for trace in binomial_scores.values()
                      for _, v in trace]
        sig_line   = -math.log10(0.05)
        frac_sig   = (sum(1 for v in all_scores if v >= sig_line)
                      / len(all_scores)) if all_scores else 0.0

        # Number of query reads spanning the CDS — the ranking metric
        n_spanning_qry = len(df_qry)

        summary_rows.append({
            "gene":              gname,
            "n_reads":           len(binomial_scores),
            "n_his_codons":      len(his_positions),
            "gene_len":          gene_len,
            "n_ref_a_sites":     len(ref_freq),
            "n_reads_ref":       len(df_ref),
            "n_reads_qry":       n_spanning_qry,
            "median_neg_log10p": float(np.median(all_scores)),
            "frac_sig_windows":  frac_sig,
        })

        # Stash everything needed to draw this gene's plot later
        plot_data[gname] = {
            "n_spanning":       n_spanning_qry,
            "meta":             meta,
            "binomial_scores":  binomial_scores,
            "his_positions":    his_positions,
            "read_edits":       read_edits,
            "gene_len":         gene_len,
        }

    ref_fasta.close()

    # ── Plot only the top-N genes by number of CDS-spanning query reads ───────
    if plot_data:
        top_genes = sorted(plot_data.keys(),
                           key=lambda g: plot_data[g]["n_spanning"],
                           reverse=True)[:args.top_n_plots]
        print(f"\n  Plotting top {len(top_genes)} genes by spanning reads:",
              file=sys.stderr)
        for gname in top_genes:
            d = plot_data[gname]
            print(f"    {gname}: {d['n_spanning']:,} spanning reads",
                  file=sys.stderr)
            safe_name = re.sub(r"[^\w\-]", "_", gname)
            try:
                plot_gene_pyx(
                    gene_name=gname,
                    meta=d["meta"],
                    binomial_scores=d["binomial_scores"],
                    his_positions=d["his_positions"],
                    read_edits=d["read_edits"],
                    label1=args.label1,
                    label2=args.label2,
                    gene_len=d["gene_len"],
                    num_reads=args.num_reads,
                    pdf_path=str(pdf_dir / safe_name),
                )
            except Exception as e:
                print(f"  WARNING: pyx plot failed for {gname}: {e}",
                      file=sys.stderr)

    print(f"\n  {n_pass:,}/{len(gene_names):,} genes passed coverage filter.",
          file=sys.stderr)
    if args.cds_spanning:
        print(f"  [spanning diagnostics] "
              f"{genes_with_any_spanning:,} genes had >=1 spanning read in "
              f"both libraries.", file=sys.stderr)
        print(f"  [spanning diagnostics] max spanning reads in any gene: "
              f"ref={max_spanning_ref:,}  qry={max_spanning_qry:,}",
              file=sys.stderr)
        if n_pass == 0:
            print(f"  HINT: no gene reached --min_coverage="
                  f"{int(args.min_coverage)} spanning reads. "
                  f"Lower --min_coverage (try a value <= {max(1, min(max_spanning_ref, max_spanning_qry))}).",
                  file=sys.stderr)

    if not summary_rows:
        print("ERROR: No genes produced output.", file=sys.stderr)
        sys.exit(1)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "median_neg_log10p", ascending=False)
    summary_csv = f"{out}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n  Summary -> {summary_csv}", file=sys.stderr)
    print(f"  Gene PDFs -> {pdf_dir}/", file=sys.stderr)
    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    Tee()
    main()