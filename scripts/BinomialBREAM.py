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

Outputs per gene:
  - Meta plot: -log10(p-value) median +/- deciles across all reads at each
    window centre, with His codon positions marked.
  - Individual read plots: per-read -log10(p-value) traces with edit/no-edit
    tick marks below each trace.
  - Summary CSV with median -log10(p) per gene.

Input format
------------
Both libraries are read from directories of parquet chunks. Each chunk is a
DataFrame with (at least) these columns:
  - chrom        : reference contig name
  - gene_strand  : "+" or "-"
  - read_id      : unique read identifier
  - aligned_pairs: list of (read_pos, ref_pos) tuples (None for indels)
  - edit_string  : per-aligned-pair code, "0"=ref(A), "1"=edit(G), "2"=skip

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

Requirements:
    pip install pysam pandas numpy scipy pyarrow
    pyx (for plotting)
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


# ── Histidine codons ──────────────────────────────────────────────────────────
HIS_CODONS = {"CAT", "CAC"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Parquet input helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_parquet_chunks(parquet_dir: str, gene: dict) -> pd.DataFrame:
    """
    Read all parquet chunks in a directory and return the rows belonging to
    this gene's contig and strand. Streams one chunk at a time to keep peak
    memory bounded by a single chunk plus the matching rows.
    """
    parquet_dir = Path(parquet_dir)
    chunks = sorted(parquet_dir.glob("*.parquet"))
    if not chunks:
        return pd.DataFrame()
    chrom  = gene["chrom"]
    strand = gene["strand"]
    dfs = []
    for chunk_path in chunks:
        df = pd.read_parquet(chunk_path)
        mask = (df["chrom"] == chrom) & (df["gene_strand"] == strand)
        df = df[mask]
        if not df.empty:
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def cds_length(gene: dict) -> int:
    """Total length of the spliced coding sequence (sum of CDS segments)."""
    return sum(ce - cs for (cs, ce) in gene["cds"])


def _gpos_to_tx_map(gene: dict, ref_fasta: pysam.FastaFile) -> dict:
    """
    Map genomic positions to spliced-CDS (transcript) positions for ref=A sites.

    tx_pos increments contiguously over the concatenated CDS segments in
    transcript order (5'->3'); introns and UTRs are skipped entirely, so
    positions in this map line up exactly with the His-codon coordinates
    produced by find_his_codon_tx_positions (which concatenates the same CDS
    segments). Only ref=A (transcript-strand A) positions are recorded.

    Note: gene["cds"] is already sorted in transcript order by parse_gtf
    (ascending for "+" strand, descending for "-"), so iterating the list
    in order walks the CDS 5'->3'.
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
                if chrom_seq[gpos] == "T":  # ref=T on minus strand = A in transcript
                    gpos_to_tx[gpos] = tx_pos
                tx_pos += 1
    return gpos_to_tx


def mean_cds_coverage_parquet(df: pd.DataFrame, gene: dict) -> float:
    """
    Mean read depth across CDS positions, computed from aligned_pairs.
    Depth is averaged over CDS positions that have at least one aligned read
    (mirrors the pileup-based behaviour of the previous BAM implementation).
    """
    if df.empty:
        return 0.0
    cds_positions = set()
    for (s, e) in gene["cds"]:
        cds_positions.update(range(s, e))
    if not cds_positions:
        return 0.0

    coverage = collections.Counter()
    for _, row in df.iterrows():
        for read_pos, ref_pos in row["aligned_pairs"]:
            if ref_pos is None or read_pos is None:
                continue
            if ref_pos in cds_positions:
                coverage[ref_pos] += 1

    if not coverage:
        return 0.0
    return sum(coverage.values()) / len(coverage)


def build_reference_freq_parquet(df: pd.DataFrame, gpos_to_tx: dict) -> dict:
    """
    Returns {tx_pos: p_edit} where p_edit = G/(A+G) at each ref=A position
    across the gene body, from the reference library.
    Clamped to [1e-6, 1-1e-6] to avoid degenerate binomial probabilities.
    """
    if df.empty:
        return {}
    edit_counts = collections.defaultdict(lambda: [0, 0])
    for _, row in df.iterrows():
        es = row["edit_string"]
        for i, (read_pos, ref_pos) in enumerate(row["aligned_pairs"]):
            if ref_pos is None or read_pos is None:
                continue
            if i >= len(es):
                continue
            ev = es[i]
            if ev == "2" or ref_pos not in gpos_to_tx:
                continue
            edit_counts[gpos_to_tx[ref_pos]][int(ev)] += 1

    ref_freq = {}
    for tx, (n0, n1) in edit_counts.items():
        total = n0 + n1
        if total > 0:
            p = n1 / total
            ref_freq[tx] = max(1e-6, min(1 - 1e-6, p))
    return ref_freq


def collect_read_edits_parquet(df: pd.DataFrame, gpos_to_tx: dict) -> dict:
    """
    Returns {read_id: {tx_pos: 0_or_1}} for all reads covering the gene,
    from the query library.
    """
    if df.empty:
        return {}
    read_edits = collections.defaultdict(dict)
    for _, row in df.iterrows():
        es = row["edit_string"]
        for i, (read_pos, ref_pos) in enumerate(row["aligned_pairs"]):
            if ref_pos is None or read_pos is None:
                continue
            if i >= len(es):
                continue
            ev = es[i]
            if ev == "2" or ref_pos not in gpos_to_tx:
                continue
            read_edits[row["read_id"]][gpos_to_tx[ref_pos]] = int(ev)
    return dict(read_edits)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def complement_base(b: str) -> str:
    return b.translate(str.maketrans("ACGTacgt", "TGCAtgca"))


def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]


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
    return genes


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
# 3. Compute binomial p-values per read per window position
#
#    At each window centre:
#      n = number of ref=A sites in window covered by this read
#      k = number of those sites showing an edit
#      p = mean background edit probability across those sites (from ref_freq)
#      p_value = scipy.stats.binom.cdf(k, n, p)  [lower tail: P(X <= k)]
#
#    A small p_value means fewer edits than expected — protection/shadowing.
#    Store as -log10(p_value) directly to avoid underflow.
# ─────────────────────────────────────────────────────────────────────────────

def compute_binomial_pvals_per_read(read_edits, ref_freq, nt_window,
                                     min_sites, gene_len):
    """
    Returns {read_name: [(window_centre, -log10(p_value)), ...]}
    """
    results = {}

    for read_name, pos_dict in read_edits.items():
        trace = []
        for start in range(0, gene_len):
            end = start + nt_window

            # Collect sites covered by this read in this window
            ks, ps = [], []
            for tx in range(start, end):
                if tx in pos_dict and tx in ref_freq:
                    ks.append(pos_dict[tx])
                    ps.append(ref_freq[tx])

            n = len(ks)
            if n < min_sites:
                continue

            k = sum(ks)
            p_mean = float(np.mean(ps))

            # Lower-tail binomial: P(X <= k | n, p_mean)
            # Protection shows as low p (fewer edits than expected)
            p_val = scipy.stats.binom.cdf(k, n, p_mean)

            # Clamp before log to avoid -inf
            p_val = max(p_val, 1e-300)
            neg_log_p = -math.log10(p_val)

            centre = start + nt_window / 2.
            trace.append((centre, neg_log_p))

        if trace:
            results[read_name] = trace

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Meta read aggregation
# ─────────────────────────────────────────────────────────────────────────────

def get_meta_read(binomial_scores):
    """
    Aggregate -log10(p) scores across all reads at each window centre.
    Returns [(centre, 5th pctile, median, 95th pctile), ...].
    Higher score = more protected (fewer edits than expected).
    """
    by_pos = collections.defaultdict(list)
    for read_name, trace in binomial_scores.items():
        for centre, neg_log_p in trace:
            by_pos[centre].append(neg_log_p)

    meta = []
    for centre in sorted(by_pos.keys()):
        vals = by_pos[centre]
        meta.append((
            centre,
            float(np.quantile(vals, 0.05)),
            float(np.quantile(vals, 0.50)),
            float(np.quantile(vals, 0.95)),
        ))
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# 5. Plotting (pyx)
# ─────────────────────────────────────────────────────────────────────────────

def plot_gene_pyx(gene_name, meta, binomial_scores, his_positions,
                   read_edits, label1, label2, gene_len, num_reads, pdf_path):
    """
    Meta plot + individual read traces with edit/no-edit tick marks.
    Y-axis is -log10(binomial p-value). Higher = more protected.
    Horizontal dashed line at -log10(0.05) = 1.30 marks nominal significance.
    """
    from pyx import canvas, graph, color, style, path, text as pyx_text

    col_qry  = color.cmyk(1, 0.5, 0, 0)     # blue  — query traces
    col_his  = color.cmyk(0, 1, 1, 0)       # red   — His codon lines
    col_sig  = color.cmyk(0, 0, 0, 0.4)     # grey  — significance threshold
    col_edit = color.cmyk(0, 0, 0, 1)       # black — edit ticks
    col_no   = color.cmyk(0, 0.8, 1, 0)     # orange — no-edit ticks

    x_min, x_max = 0, gene_len
    panel_w  = 12
    tick_h   = 0.25
    read_h   = 1.2
    gap      = 0.6
    sig_line = -math.log10(0.05)   # 1.301

    all_y = [entry[3] for entry in meta]
    y_max_meta = max(all_y) * 1.1 if all_y else 5.0
    y_max_meta = max(y_max_meta, 2.0)

    c = canvas.canvas()

    # ── Meta plot ─────────────────────────────────────────────────────────────
    meta_ypos = num_reads * (read_h + tick_h + gap) + gap * 2

    g_meta = graph.graphxy(
        width=panel_w, height=3,
        xpos=0, ypos=meta_ypos,
        x=graph.axis.linear(min=x_min, max=x_max,
                            title="Position Along Transcript (nt)"),
        y=graph.axis.linear(min=0, max=y_max_meta,
                            title="-log10(p)"),
    )

    # His codon lines
    for hp in his_positions:
        g_meta.plot(
            graph.data.function(f"x(y)={hp}", min=0, max=y_max_meta),
            [graph.style.line([col_his, style.linewidth.thin,
                               style.linestyle.solid])]
        )

    # Nominal significance threshold at p=0.05
    g_meta.plot(
        graph.data.function(f"y(x)={sig_line:.4f}",
                            min=x_min, max=x_max),
        [graph.style.line([col_sig, style.linewidth.thin,
                           style.linestyle.dashed])]
    )

    # 5th/95th percentile dotted bounds
    for col_idx in [1, 3]:
        pts = [(entry[0], entry[col_idx]) for entry in meta]
        if pts:
            g_meta.plot(graph.data.points(pts, x=1, y=2),
                        [graph.style.line([col_qry, style.linewidth.thin,
                                           style.linestyle.dotted])])

    # Median solid line
    med_pts = [(entry[0], entry[2]) for entry in meta]
    if med_pts:
        g_meta.plot(graph.data.points(med_pts, x=1, y=2),
                    [graph.style.line([col_qry, style.linewidth.normal,
                                       style.linestyle.solid])])

    c.insert(g_meta)
    c.text(g_meta.xpos + g_meta.width / 2.,
           g_meta.ypos + g_meta.height + 0.4,
           f"{gene_name} - {label2} binomial protection (ref: {label1})",
           [pyx_text.halign.center, pyx_text.size.normalsize])

    # ── Individual read plots ─────────────────────────────────────────────────
    read_names = list(binomial_scores.keys())[:num_reads]

    for jj, read_name in enumerate(read_names):
        trace = binomial_scores[read_name]
        ypos  = (num_reads - 1 - jj) * (read_h + tick_h + gap)

        read_y_vals = [v for _, v in trace]
        y_max_read  = max(read_y_vals) * 1.1 if read_y_vals else 5.0
        y_max_read  = max(y_max_read, 2.0)
        y_min_read  = 0.0

        g_read = graph.graphxy(
            width=panel_w, height=read_h,
            xpos=0, ypos=ypos + tick_h,
            x=graph.axis.linkedaxis(g_meta.axes["x"]),
            y=graph.axis.linear(min=y_min_read, max=y_max_read, title=""),
        )

        # His codon lines
        for hp in his_positions:
            g_read.plot(
                graph.data.function(f"x(y)={hp}", min=y_min_read,
                                    max=y_max_read),
                [graph.style.line([col_his, style.linewidth.thin,
                                   style.linestyle.solid])]
            )

        # Significance line
        if sig_line <= y_max_read:
            g_read.plot(
                graph.data.function(f"y(x)={sig_line:.4f}",
                                    min=x_min, max=x_max),
                [graph.style.line([col_sig, style.linewidth.thin,
                                   style.linestyle.dashed])]
            )

        # -log10(p) trace
        if trace:
            g_read.plot(
                graph.data.points(list(trace), x=1, y=2),
                [graph.style.line([col_qry, style.linewidth.thin,
                                   style.linestyle.solid])]
            )

        c.insert(g_read)

        # Edit/no-edit tick marks
        g_ticks = graph.graphxy(
            width=panel_w, height=tick_h,
            xpos=0, ypos=ypos,
            x=graph.axis.linkedaxis(g_meta.axes["x"]),
            y=graph.axis.linear(min=0, max=1),
        )

        pos_dict = read_edits.get(read_name, {})
        for tx, v in pos_dict.items():
            col = col_edit if v == 1 else col_no
            g_ticks.plot(
                graph.data.function(f"x(y)={tx}", min=0, max=1),
                [graph.style.line([col, style.linewidth.thin,
                                   style.linestyle.solid])]
            )

        c.insert(g_ticks)

        c.text(panel_w + 0.15, ypos + tick_h + read_h / 2.,
               read_name[:20],
               [pyx_text.valign.middle, pyx_text.size.tiny])

    c.writePDFfile(pdf_path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Binomial test for editing protection in sliding windows "
                    "(parquet input)."
    )
    p.add_argument("--parquet1", required=True,
                   help="Parquet chunk directory for the reference/WT library.")
    p.add_argument("--parquet2", required=True,
                   help="Parquet chunk directory for the query library.")
    p.add_argument("--label1", default="BAM1")
    p.add_argument("--label2", default="BAM2")
    p.add_argument("--ref",    required=True)
    p.add_argument("--gtf",    required=True)
    p.add_argument("--output", default="binomialShadow")
    p.add_argument("--window", type=int, default=30,
                   help="Sliding window size in nt (default: 30)")
    p.add_argument("--min_coverage", type=float, default=50.0)
    p.add_argument("--min_sites", type=int, default=5,
                   help="Min ref=A sites per window to run the test (default: 5)")
    p.add_argument("--num_reads", type=int, default=10,
                   help="Number of individual reads to plot per gene (default: 10)")
    p.add_argument("--gene_list", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out  = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print("=== Binomial Shadowing Analysis (parquet) ===", file=sys.stderr)

    print("\nParsing GTF…", file=sys.stderr)
    genes = parse_gtf(args.gtf)
    print(f"  {len(genes):,} genes.", file=sys.stderr)

    if args.gene_list:
        with open(args.gene_list) as fh:
            allowed = {l.strip() for l in fh if l.strip()}
        genes = {k: v for k, v in genes.items() if k in allowed}
        print(f"  {len(genes):,} after gene list filter.", file=sys.stderr)

    ref_fasta = pysam.FastaFile(args.ref)

    pdf_dir = Path(f"{out}_gene_pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    gene_names = list(genes.keys())
    n_pass = 0

    print(f"\nProcessing {len(gene_names):,} genes "
          f"(mean CDS coverage >= {args.min_coverage}x)…", file=sys.stderr)

    for i, gname in enumerate(gene_names):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(gene_names)} scanned, "
                  f"{n_pass} passing…", file=sys.stderr)

        gene     = genes[gname]
        # Window sliding and His-codon marks live in spliced-CDS coordinates,
        # so the transcript extent is the summed CDS length (not the genomic
        # span, which would include introns/UTRs).
        gene_len = cds_length(gene)

        # Load this gene's reads once per library.
        df_ref = load_parquet_chunks(args.parquet1, gene)
        df_qry = load_parquet_chunks(args.parquet2, gene)
        if df_ref.empty or df_qry.empty:
            continue

        # Coverage filter (both libraries must pass).
        cov_ref = mean_cds_coverage_parquet(df_ref, gene)
        cov_qry = mean_cds_coverage_parquet(df_qry, gene)
        if cov_ref < args.min_coverage or cov_qry < args.min_coverage:
            continue
        n_pass += 1

        gpos_to_tx = _gpos_to_tx_map(gene, ref_fasta)

        ref_freq = build_reference_freq_parquet(df_ref, gpos_to_tx)
        if not ref_freq:
            continue

        read_edits = collect_read_edits_parquet(df_qry, gpos_to_tx)
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
        # Fraction of window-positions with nominal significance (p < 0.05)
        sig_line   = -math.log10(0.05)
        frac_sig   = sum(1 for v in all_scores if v >= sig_line) / len(all_scores) \
                     if all_scores else 0.0

        summary_rows.append({
            "gene":              gname,
            "n_reads":           len(binomial_scores),
            "n_his_codons":      len(his_positions),
            "gene_len":          gene_len,
            "n_ref_a_sites":     len(ref_freq),
            "mean_cov_ref":      cov_ref,
            "mean_cov_query":    cov_qry,
            "median_neg_log10p": float(np.median(all_scores)),
            "frac_sig_windows":  frac_sig,
        })

        safe_name = re.sub(r"[^\w\-]", "_", gname)
        try:
            plot_gene_pyx(
                gene_name=gname,
                meta=meta,
                binomial_scores=binomial_scores,
                his_positions=his_positions,
                read_edits=read_edits,
                label1=args.label1,
                label2=args.label2,
                gene_len=gene_len,
                num_reads=args.num_reads,
                pdf_path=str(pdf_dir / safe_name),
            )
        except Exception as e:
            print(f"  WARNING: pyx plot failed for {gname}: {e}",
                  file=sys.stderr)

    ref_fasta.close()

    print(f"\n  {n_pass:,}/{len(gene_names):,} genes passed coverage filter.",
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
    main()