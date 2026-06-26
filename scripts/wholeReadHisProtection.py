#!/usr/bin/env python3
"""
Window-based His codon protection analysis
============================================
For each read covering at least one His codon, tests whether any window
around a His codon shows global suppression of A->G editing, consistent
with ribosome occupancy.

For each His codon window on each read:
  - Collect all ref=A positions within +-window nt of the His A
  - n = number of those positions covered by this read
  - k_unedited = number showing A (not edited)
  - p_i = per-position background P(unedited) from parquet1
  - p_mean = mean of p_i across covered positions
  - p_value = P(X >= k_unedited | n, p_mean)  [upper-tail binomial]

Per-read summary:
  - min_p: most significant window (most protected)
  - combined_p: Fisher's combined p-value across all His windows
    (tests whether the read shows global His codon suppression)

Outputs:
  - Per-read CSV with window-level and combined statistics
  - CDF of -log10(min_p) per read comparing the two libraries
  - Scatter of observed vs expected unedited sites in the best window
  - Gene-level summary

Usage:
    python3 windowHisProtection.py \
        --parquet1 wt_parquet/   --label1 "-3AT" \
        --parquet2 3at_parquet/  --label2 "+3AT" \
        --ref  reference.fa \
        --gtf  annotation.gtf \
        --output output_prefix \
        [--window 50] \
        [--min_sites 5] \
        [--min_coverage 50] \
        [--cds_spanning]

Requirements:
    pip install pysam pandas numpy scipy
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


# ─────────────────────────────────────────────────────────────────────────────
# 3. Find His codon genomic positions
# ─────────────────────────────────────────────────────────────────────────────

def find_his_gpos(ref_fasta: pysam.FastaFile, gene: dict) -> list:
    """
    Returns list of {rank, gpos, tx_pos, codon} for every His codon A,
    in transcript order.
    """
    chrom  = gene["chrom"]
    strand = gene["strand"]

    tx_seq     = ""
    tx_to_gpos = []

    for (cs, ce) in gene["cds"]:
        seg = ref_fasta.fetch(chrom, cs, ce).upper()
        if strand == "+":
            for gp in range(cs, ce):
                tx_to_gpos.append(gp)
            tx_seq += seg
        else:
            for gp in range(ce - 1, cs - 1, -1):
                tx_to_gpos.append(gp)
            tx_seq += reverse_complement(seg)

    his_sites = []
    seen      = set()
    for i in range(0, len(tx_seq) - 2, 3):
        if tx_seq[i:i+3] in HIS_CODONS:
            tx_a = i + 1
            if tx_a >= len(tx_to_gpos):
                continue
            gpos = tx_to_gpos[tx_a]
            if gpos not in seen:
                seen.add(gpos)
                his_sites.append({
                    "rank":   len(his_sites) + 1,
                    "gpos":   gpos,
                    "tx_pos": tx_a,
                    "codon":  tx_seq[i:i+3],
                })

    return his_sites


# ─────────────────────────────────────────────────────────────────────────────
# 4. Parquet loading
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


def filter_genes_by_parquet_count(genes: dict,
                                   df1: pd.DataFrame,
                                   df2: pd.DataFrame,
                                   min_reads: int = 50) -> list:
    passing    = []
    use_coords = ("read_start" in df1.columns and "read_end" in df1.columns)

    for gname, gene in genes.items():
        chrom      = gene["chrom"]
        strand     = gene["strand"]
        gene_start = gene["gene_start"]
        gene_end   = gene["gene_end"]

        for df in (df1, df2):
            mask = (df["chrom"] == chrom) & (df["gene_strand"] == strand)
            if use_coords:
                mask &= (df["read_start"] < gene_end) & \
                        (df["read_end"]   > gene_start)
            if mask.sum() < min_reads:
                break
        else:
            passing.append(gname)

    print(f"  {len(passing):,}/{len(genes):,} genes pass coverage filter.",
          file=sys.stderr)
    return passing


# ─────────────────────────────────────────────────────────────────────────────
# 5. Build per-position background edit probability from parquet1
#    across all ref=A positions in the windows around each His codon
# ─────────────────────────────────────────────────────────────────────────────

def build_window_background_freq(df: pd.DataFrame,
                                  ref_fasta: pysam.FastaFile,
                                  gene: dict,
                                  his_sites: list,
                                  window: int) -> dict:
    """
    For every ref=A position within +-window nt (genomic) of any His codon A,
    compute the background editing probability P(edit) from parquet1.

    Returns {gpos: p_edit} clamped to [1e-6, 1-1e-6].
    P(unedited) = 1 - p_edit.

    Uses absolute_indices + edit_string (sense-oriented).
    ref=A positions are identified from the reference FASTA.
    """
    if df.empty or not his_sites:
        return {}

    chrom      = gene["chrom"]
    strand     = gene["strand"]
    gene_minus = (strand == "-")
    chrom_len  = ref_fasta.get_reference_length(chrom)
    chrom_seq  = ref_fasta.fetch(chrom).upper()

    # Collect all genomic positions that are ref=A (in transcript coords)
    # within any His codon window
    window_gpos = set()
    for s in his_sites:
        gpos = s["gpos"]
        lo   = max(0, gpos - window)
        hi   = min(chrom_len - 1, gpos + window)
        for gp in range(lo, hi + 1):
            ref_g  = chrom_seq[gp]
            ref_tx = complement_base(ref_g) if gene_minus else ref_g
            if ref_tx == "A":
                window_gpos.add(gp)

    if not window_gpos:
        return {}

    min_gp = min(window_gpos)
    max_gp = max(window_gpos)

    # Pre-filter reads to those spanning the window region
    if "read_start" in df.columns and "read_end" in df.columns:
        sub = df[(df["read_start"] <= max_gp) & (df["read_end"] >= min_gp)]
    else:
        sub = df

    edit_counts = collections.defaultdict(lambda: [0, 0])  # [n_A, n_G]

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
            if ref_pos not in window_gpos:
                continue
            if i >= n_edit:
                continue
            ev = edit_str[i]
            if ev == "2":
                continue
            edit_counts[ref_pos][int(ev)] += 1

    freq = {}
    for gpos, (n_a, n_g) in edit_counts.items():
        total = n_a + n_g
        if total > 0:
            p = max(1e-6, min(1 - 1e-6, n_g / total))
            freq[gpos] = p

    return freq


# ─────────────────────────────────────────────────────────────────────────────
# 6. Per-read window-based binomial test
# ─────────────────────────────────────────────────────────────────────────────

def test_read_window_protection(df: pd.DataFrame,
                                 ref_fasta: pysam.FastaFile,
                                 gene: dict,
                                 his_sites: list,
                                 window_freq: dict,
                                 window: int,
                                 min_sites: int = 5,
                                 cds_spanning: bool = False) -> pd.DataFrame:
    """
    For each read covering at least one His codon window with >= min_sites
    ref=A positions:

      Per window:
        - n      = ref=A positions covered by this read in the window
        - k      = number of those positions showing A (unedited)
        - p_mean = mean background P(unedited) across those positions
        - p_val  = P(X >= k | n, p_mean)  upper-tail binomial

      Per read:
        - best_p       = minimum p-value across all His windows (most protected)
        - combined_p   = Fisher's method combining all window p-values
        - n_windows    = number of windows tested on this read

    Returns one row per read.
    """
    if df.empty or not his_sites or not window_freq:
        return pd.DataFrame()

    chrom      = gene["chrom"]
    strand     = gene["strand"]
    gene_minus = (strand == "-")
    chrom_len  = ref_fasta.get_reference_length(chrom)
    cds_start  = gene.get("cds_genomic_start", gene["gene_start"])
    cds_end    = gene.get("cds_genomic_end",   gene["gene_end"])

    # For each His site, pre-compute the set of window gpos with known freq
    his_window_gpos = []
    for s in his_sites:
        gpos = s["gpos"]
        lo   = max(0, gpos - window)
        hi   = min(chrom_len - 1, gpos + window)
        wgpos = {gp: window_freq[gp]
                 for gp in range(lo, hi + 1)
                 if gp in window_freq}
        if len(wgpos) >= min_sites:
            his_window_gpos.append((s, wgpos))

    if not his_window_gpos:
        return pd.DataFrame()

    # Global span for pre-filtering reads
    all_gpos = [gp for _, wg in his_window_gpos for gp in wg]
    min_gp   = min(all_gpos)
    max_gp   = max(all_gpos)

    if "read_start" in df.columns and "read_end" in df.columns:
        mask = (df["read_start"] <= max_gp) & (df["read_end"] >= min_gp)
        if cds_spanning:
            mask &= (df["read_start"] <= cds_start) & \
                    (df["read_end"]   >= cds_end)
        sub = df[mask]
    else:
        sub = df

    if sub.empty:
        return pd.DataFrame()

    # Collect per-position edit status per read across all window positions
    # read_pos_status: {read_id: {gpos: 0_or_1}}
    read_pos_status = collections.defaultdict(dict)

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
            if ref_pos not in window_freq:
                continue
            if i >= n_edit:
                continue
            ev = edit_str[i]
            if ev == "2":
                continue
            # 0 = edited (G), 1 = unedited (A) for our protection test
            read_pos_status[read.read_id][ref_pos] = 1 - int(ev)

    rows = []
    for read_id, pos_status in read_pos_status.items():
        window_results = []

        for site, wgpos in his_window_gpos:
            # Find positions in this window that the read covered
            covered = {gp: pos_status[gp]
                       for gp in wgpos if gp in pos_status}
            n = len(covered)
            if n < min_sites:
                continue

            k_unedited     = sum(covered.values())
            p_vals_unedited = [1 - wgpos[gp] for gp in covered]
            p_mean         = float(np.mean(p_vals_unedited))
            expected       = sum(p_vals_unedited)

            # Upper-tail binomial P(X >= k | n, p_mean)
            if k_unedited == 0:
                p_val = 1.0
            else:
                p_val = 1.0 - scipy.stats.binom.cdf(
                    k_unedited - 1, n, p_mean
                )
            p_val = max(p_val, 1e-300)

            window_results.append({
                "his_rank":    site["rank"],
                "his_gpos":   site["gpos"],
                "n_sites":    n,
                "k_unedited": k_unedited,
                "expected":   expected,
                "p_mean":     p_mean,
                "p_val":      p_val,
            })

        if not window_results:
            continue

        p_vals = [w["p_val"] for w in window_results]

        # Most protected window
        best_idx  = int(np.argmin(p_vals))
        best      = window_results[best_idx]
        best_p    = best["p_val"]

        # Fisher's combined p-value across all windows
        if len(p_vals) == 1:
            combined_p = best_p
        else:
            chi2_stat  = -2 * sum(math.log(p) for p in p_vals)
            combined_p = scipy.stats.chi2.sf(chi2_stat, df=2 * len(p_vals))
        combined_p = max(combined_p, 1e-300)

        rows.append({
            "read_id":             read_id,
            "gene":                gene["gene_name"],
            "n_windows_tested":    len(window_results),
            "best_his_rank":       best["his_rank"],
            "best_n_sites":        best["n_sites"],
            "best_k_unedited":     best["k_unedited"],
            "best_expected":       best["expected"],
            "best_p_mean":         best["p_mean"],
            "best_p_val":          best_p,
            "best_neg_log10_p":    -math.log10(best_p),
            "combined_p_val":      combined_p,
            "combined_neg_log10_p": -math.log10(combined_p),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results_pyx(df1: pd.DataFrame, df2: pd.DataFrame,
                      label1: str, label2: str,
                      output_prefix: str):
    """
    Figure 1: CDF of best_neg_log10_p per read (most protected window).
    Figure 2: CDF of combined_neg_log10_p per read (all windows jointly).
    Figure 3: Scatter of best_k_unedited vs best_expected per read.
    """
    from pyx import canvas, graph, color, style, path, deco, text as pyx_text

    col1 = color.cmyk(0, 0, 0, 1)
    col2 = color.cmyk(1, 0.5, 0, 0)

    def _cdf_figure(col_key, title, filename):
        c       = canvas.canvas()
        panel_w = 8
        panel_h = 6
        sig     = -math.log10(0.05)

        all_vals = []
        for df in [df1, df2]:
            if not df.empty and col_key in df.columns:
                all_vals.extend(df[col_key].dropna().tolist())
        x_max = max(all_vals) * 1.1 if all_vals else 5.0
        x_max = max(x_max, 3.0)

        g = graph.graphxy(
            width=panel_w, height=panel_h,
            xpos=0, ypos=0,
            x=graph.axis.linear(min=0, max=x_max,
                                title=r"$-\log_{10}$(p)"),
            y=graph.axis.linear(min=0, max=1,
                                title="Cumulative fraction of reads"),
        )

        g.plot(
            graph.data.function(f"x(y)={sig:.4f}", min=0, max=1),
            [graph.style.line([color.gray(0.5), style.linewidth.thin,
                               style.linestyle.dashed])]
        )

        leg_x  = panel_w + 0.5
        leg_lw = 0.8
        leg_dy = 0.6
        leg_y0 = panel_h - 0.3

        for j, (df, col, label) in enumerate([
            (df1, col1, label1), (df2, col2, label2)
        ]):
            if df.empty or col_key not in df.columns:
                continue
            vals     = np.sort(df[col_key].dropna().values)
            cdf      = np.arange(1, len(vals) + 1) / len(vals)
            frac_sig = float((df[col_key] >= sig).mean())
            g.plot(
                graph.data.points(list(zip(vals.tolist(), cdf.tolist())),
                                  x=1, y=2),
                [graph.style.line([col, style.linewidth.normal,
                                   style.linestyle.solid])]
            )
            ly = leg_y0 - j * leg_dy
            c.stroke(path.line(leg_x, ly, leg_x + leg_lw, ly),
                     [col, style.linewidth.normal, style.linestyle.solid])
            c.text(leg_x + leg_lw + 0.15, ly,
                   f"{label} ({frac_sig*100:.1f}\\% sig.)",
                   [pyx_text.valign.middle, pyx_text.size.small])

        c.insert(g)
        c.text(panel_w / 2., panel_h + 0.5, title,
               [pyx_text.halign.center, pyx_text.size.normalsize])

        c.writePDFfile(filename)
        print(f"  Saved -> {filename}.pdf", file=sys.stderr)

    # Figure 1: best window CDF
    _cdf_figure(
        "best_neg_log10_p",
        "Most protected His window per read",
        f"{output_prefix}_best_window_cdf_pyx",
    )

    # Figure 2: combined CDF
    _cdf_figure(
        "combined_neg_log10_p",
        "Fisher combined p across His windows per read",
        f"{output_prefix}_combined_cdf_pyx",
    )

    # Figure 3: scatter observed vs expected in best window
    c3      = canvas.canvas()
    panel_w = 6
    panel_h = 6
    gap     = 2.0

    for col_idx, (df, col, label) in enumerate([
        (df1, col1, label1), (df2, col2, label2)
    ]):
        if df.empty:
            continue
        xpos = col_idx * (panel_w + gap)
        obs  = df["best_k_unedited"].tolist()
        exp  = df["best_expected"].tolist()
        ax_max = max(max(obs), max(exp)) * 1.1 if obs else 5.0
        ax_max = max(ax_max, 2.0)

        g_sc = graph.graphxy(
            width=panel_w, height=panel_h,
            xpos=xpos, ypos=0,
            x=graph.axis.linear(min=0, max=ax_max,
                                title="Expected unedited sites in window"),
            y=graph.axis.linear(min=0, max=ax_max,
                                title="Observed unedited sites in window"),
        )
        g_sc.plot(
            graph.data.function("y(x)=x", min=0, max=ax_max),
            [graph.style.line([color.gray(0.5), style.linewidth.thin,
                               style.linestyle.dashed])]
        )

        sig     = 0.05
        p_vals  = df["best_p_val"].tolist()
        pts_sig = [(e, o) for e, o, p in zip(exp, obs, p_vals) if p < sig]
        pts_ns  = [(e, o) for e, o, p in zip(exp, obs, p_vals) if p >= sig]

        if pts_ns:
            g_sc.plot(
                graph.data.points(pts_ns, x=1, y=2),
                [graph.style.symbol(graph.style.symbol.circle,
                                    symbolattrs=[color.gray(0.6), deco.filled],
                                    size=0.05)]
            )
        if pts_sig:
            g_sc.plot(
                graph.data.points(pts_sig, x=1, y=2),
                [graph.style.symbol(graph.style.symbol.circle,
                                    symbolattrs=[col, deco.filled],
                                    size=0.07)]
            )

        c3.insert(g_sc)
        c3.text(g_sc.xpos + g_sc.width / 2.,
                g_sc.ypos + g_sc.height + 0.4, label,
                [pyx_text.halign.center, pyx_text.size.normalsize])

    c3.text((2 * panel_w + gap) / 2., -0.8,
            "Best window: observed vs expected unedited sites (coloured = p$<$0.05)",
            [pyx_text.halign.center, pyx_text.size.small])

    scatter_path = f"{output_prefix}_scatter_pyx"
    c3.writePDFfile(scatter_path)
    print(f"  Saved -> {scatter_path}.pdf", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Window-based His codon protection binomial test."
    )
    p.add_argument("--parquet1",     required=True)
    p.add_argument("--parquet2",     required=True)
    p.add_argument("--label1",       default="-3AT")
    p.add_argument("--label2",       default="+3AT")
    p.add_argument("--ref",          required=True)
    p.add_argument("--gtf",          required=True)
    p.add_argument("--output",       default="window_his_protection")
    p.add_argument("--window",       type=int, default=50,
                   help="nt each side of His A to include in window (default: 50)")
    p.add_argument("--min_sites",    type=int, default=5,
                   help="Min ref=A sites a read must cover in a window "
                        "to test that window (default: 5)")
    p.add_argument("--min_coverage", type=float, default=50.0)
    p.add_argument("--gene_list",    default=None)
    p.add_argument("--cds_spanning", action="store_true",
                   help="Only include reads spanning the full CDS")
    return p.parse_args()


def main():
    args = parse_args()
    out  = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print("=== Window-based His Codon Protection Analysis ===", file=sys.stderr)
    print(f"  Window: +/-{args.window} nt around each His A", file=sys.stderr)
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

    print(f"\nLoading {args.label1} parquet...", file=sys.stderr)
    df_ref = load_all_parquet_chunks(args.parquet1)
    print(f"\nLoading {args.label2} parquet...", file=sys.stderr)
    df_qry = load_all_parquet_chunks(args.parquet2)

    print(f"\nFiltering genes (min reads >= {int(args.min_coverage)})...",
          file=sys.stderr)
    passing = filter_genes_by_parquet_count(
        genes, df_ref, df_qry, min_reads=int(args.min_coverage)
    )
    if not passing:
        print("ERROR: No genes passed coverage filter.", file=sys.stderr)
        sys.exit(1)

    ref_fasta    = pysam.FastaFile(args.ref)
    all_ref_rows = []
    all_qry_rows = []
    gene_summary = []

    print(f"\nProcessing {len(passing):,} genes...", file=sys.stderr)

    for i, gname in enumerate(passing):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(passing)}: {gname}...", file=sys.stderr)

        gene      = genes[gname]
        his_sites = find_his_gpos(ref_fasta, gene)
        if not his_sites:
            continue

        chrom  = gene["chrom"]
        strand = gene["strand"]

        def _gene_df(df):
            mask = (df["chrom"] == chrom) & (df["gene_strand"] == strand)
            if "read_start" in df.columns and "read_end" in df.columns:
                mask &= (df["read_start"] < gene["gene_end"]) & \
                        (df["read_end"]   > gene["gene_start"])
            return df[mask]

        gene_ref = _gene_df(df_ref)
        gene_qry = _gene_df(df_qry)

        if gene_ref.empty or gene_qry.empty:
            continue

        # Build per-position background frequencies from parquet1
        window_freq = build_window_background_freq(
            gene_ref, ref_fasta, gene, his_sites, args.window
        )
        if not window_freq:
            continue

        # Run per-read window tests on both libraries
        ref_df = test_read_window_protection(
            gene_ref, ref_fasta, gene, his_sites, window_freq,
            window=args.window, min_sites=args.min_sites,
            cds_spanning=args.cds_spanning,
        )
        qry_df = test_read_window_protection(
            gene_qry, ref_fasta, gene, his_sites, window_freq,
            window=args.window, min_sites=args.min_sites,
            cds_spanning=args.cds_spanning,
        )

        if not ref_df.empty:
            all_ref_rows.append(ref_df)
        if not qry_df.empty:
            all_qry_rows.append(qry_df)

        sig = 0.05
        for df, label in [(ref_df, args.label1), (qry_df, args.label2)]:
            if df.empty:
                continue
            gene_summary.append({
                "gene":                gname,
                "label":               label,
                "n_reads":             len(df),
                "n_his_codons":        len(his_sites),
                "mean_best_neg_log10p": float(df["best_neg_log10_p"].mean()),
                "frac_sig_best":       float((df["best_p_val"] < sig).mean()),
                "mean_combined_neg_log10p": float(
                    df["combined_neg_log10_p"].mean()),
                "frac_sig_combined":   float(
                    (df["combined_p_val"] < sig).mean()),
            })

    ref_fasta.close()

    full_ref = pd.concat(all_ref_rows, ignore_index=True) \
               if all_ref_rows else pd.DataFrame()
    full_qry = pd.concat(all_qry_rows, ignore_index=True) \
               if all_qry_rows else pd.DataFrame()

    if full_ref.empty and full_qry.empty:
        print("No results — exiting.", file=sys.stderr)
        sys.exit(0)

    # Save per-read results
    for df, label in [(full_ref, args.label1), (full_qry, args.label2)]:
        if df.empty:
            continue
        safe     = label.replace(" ", "_").replace("/", "_")
        csv_path = f"{out}_{safe}_per_read.csv.gz"
        df.to_csv(csv_path, index=False, compression="gzip")
        frac_sig = float((df["best_p_val"] < 0.05).mean())
        print(f"\n  [{label}] {len(df):,} reads  |  "
              f"{frac_sig*100:.1f}% significant best window  |  "
              f"saved -> {csv_path}", file=sys.stderr)

    # Save gene summary sorted by mean protection signal
    if gene_summary:
        gs_df   = pd.DataFrame(gene_summary)
        gs_df   = gs_df.sort_values("frac_sig_combined", ascending=False)
        gs_path = f"{out}_gene_summary.csv"
        gs_df.to_csv(gs_path, index=False)
        print(f"\n  Saved gene summary -> {gs_path}", file=sys.stderr)

    # Plot
    print("\nGenerating plots...", file=sys.stderr)
    try:
        plot_results_pyx(full_ref, full_qry,
                         args.label1, args.label2, out)
    except Exception as e:
        print(f"  WARNING: pyx plotting failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    Tee()
    main()