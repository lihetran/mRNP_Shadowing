#!/usr/bin/env python3
"""
His Codon Footprint Facilitation Analysis
==========================================
Clean single-molecule question:

  On reads where His codon i shows a protected footprint (editing suppressed
  in a +-window nt region, z-score < threshold), is the editing rate in the
  footprint around downstream His codon j higher, lower, or the same as on
  reads where His codon i is unprotected?

For each read covering two His codons i and j (j downstream of i):
  1. Compute z-score for window i:
       z_i = (observed_edit_frac_i - expected_edit_frac_i) / std_i
     where expected and std come from the per-position background frequencies
     estimated from parquet1 (reference library).
     z_i << 0 means suppressed editing = protected footprint.

  2. Classify the read as "i protected" (z_i < z_threshold) or not.

  3. Record the edit fraction in window j on the same read.

  4. Compare mean edit_frac_j between protected vs unprotected groups
     across all reads, per ordered pair (i, j), aggregated by codon distance.

Usage:
    python3 hisFootprintFacilitation.py \
        --parquet1 wt_parquet/   --label1 "-3AT" \
        --parquet2 3at_parquet/  --label2 "+3AT" \
        --ref  reference.fa \
        --gtf  annotation.gtf \
        --output output_prefix \
        [--window 25] \
        [--min_sites 5] \
        [--z_threshold -1.0] \
        [--min_reads_per_group 10] \
        [--min_coverage 50] \
        [--gene_list genes.txt] \
        [--cds_spanning]
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
# 4. Parquet loading and gene filtering
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


def filter_genes(genes: dict, df1: pd.DataFrame, df2: pd.DataFrame,
                  min_reads: int) -> list:
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
# 5. Build per-position background edit frequency from parquet1
# ─────────────────────────────────────────────────────────────────────────────

def build_window_background_freq(df: pd.DataFrame,
                                  ref_fasta: pysam.FastaFile,
                                  gene: dict,
                                  his_sites: list,
                                  window: int) -> dict:
    """
    For every ref=A position within +-window nt of any His codon A,
    compute background P(edit) from parquet1.
    Returns {gpos: p_edit}.
    """
    if df.empty or not his_sites:
        return {}

    chrom      = gene["chrom"]
    strand     = gene["strand"]
    gene_minus = (strand == "-")
    chrom_len  = ref_fasta.get_reference_length(chrom)
    chrom_seq  = ref_fasta.fetch(chrom).upper()

    window_gpos = set()
    for s in his_sites:
        lo = max(0, s["gpos"] - window)
        hi = min(chrom_len - 1, s["gpos"] + window)
        for gp in range(lo, hi + 1):
            ref_tx = complement_base(chrom_seq[gp]) if gene_minus \
                     else chrom_seq[gp]
            if ref_tx == "A":
                window_gpos.add(gp)

    if not window_gpos:
        return {}

    min_gp = min(window_gpos)
    max_gp = max(window_gpos)

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
            freq[gpos] = max(1e-6, min(1 - 1e-6, n_g / total))
    return freq


# ─────────────────────────────────────────────────────────────────────────────
# 6. Per-read window z-score and footprint facilitation
# ─────────────────────────────────────────────────────────────────────────────

def compute_read_window_zscore(pos_status: dict,
                                wgpos: dict) -> tuple:
    """
    Given edit status at positions in a window and their background
    edit probabilities, compute:
      - observed edit fraction
      - expected edit fraction (mean of background p_edit)
      - z-score = (observed - expected) / std
        where std = sqrt(mean_p * (1 - mean_p) / n)

    pos_status: {gpos: 0_or_1} where 1 = edited, 0 = unedited
    wgpos:      {gpos: p_edit}  background edit probability

    Returns (edit_frac, expected_frac, z_score, n_sites) or
            (nan, nan, nan, 0) if insufficient coverage.
    """
    covered = {gp: pos_status[gp] for gp in wgpos if gp in pos_status}
    n = len(covered)
    if n == 0:
        return np.nan, np.nan, np.nan, 0

    obs_frac  = sum(covered.values()) / n
    exp_frac  = float(np.mean([wgpos[gp] for gp in covered]))
    variance  = exp_frac * (1 - exp_frac) / n
    std       = math.sqrt(variance) if variance > 0 else 1e-9

    z = (obs_frac - exp_frac) / std
    return obs_frac, exp_frac, z, n


def collect_footprint_facilitation(df: pd.DataFrame,
                                    ref_fasta: pysam.FastaFile,
                                    gene: dict,
                                    his_sites: list,
                                    window_freq: dict,
                                    window: int,
                                    min_sites: int,
                                    z_threshold: float,
                                    cds_spanning: bool) -> pd.DataFrame:
    """
    For each read covering at least two His codon windows:
      1. Compute z-score for each window (how suppressed is editing?)
      2. For each ordered pair (i upstream, j downstream):
         - Record z_i, edit_frac_j, and whether i is protected (z_i < threshold)

    Returns one row per (read, ordered His pair).
    """
    if df.empty or not his_sites or not window_freq:
        return pd.DataFrame()

    chrom     = gene["chrom"]
    strand    = gene["strand"]
    chrom_len = ref_fasta.get_reference_length(chrom)
    cds_start = gene.get("cds_genomic_start", gene["gene_start"])
    cds_end   = gene.get("cds_genomic_end",   gene["gene_end"])

    # Pre-compute valid window positions per His site
    his_window_gpos = []
    for s in his_sites:
        lo    = max(0, s["gpos"] - window)
        hi    = min(chrom_len - 1, s["gpos"] + window)
        wgpos = {gp: window_freq[gp]
                 for gp in range(lo, hi + 1) if gp in window_freq}
        if len(wgpos) >= min_sites:
            his_window_gpos.append((s, wgpos))

    if len(his_window_gpos) < 2:
        return pd.DataFrame()

    all_gpos = [gp for _, wg in his_window_gpos for gp in wg]
    min_gp   = min(all_gpos)
    max_gp   = max(all_gpos)

    # Pre-filter reads
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

    # Collect per-position edit status per read
    # 1 = edited (G), 0 = unedited (A)
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
            read_pos_status[read.read_id][ref_pos] = int(ev)

    rows = []
    for read_id, pos_status in read_pos_status.items():

        # Compute z-score and edit fraction for each His window on this read
        window_stats = []
        for site, wgpos in his_window_gpos:
            obs_frac, exp_frac, z, n = compute_read_window_zscore(
                pos_status, wgpos)
            if n < min_sites:
                continue
            window_stats.append({
                "rank":      site["rank"],
                "obs_frac":  obs_frac,
                "exp_frac":  exp_frac,
                "z_score":   z,
                "n_sites":   n,
                "protected": z < z_threshold,
            })

        # Need at least 2 windows with sufficient coverage
        if len(window_stats) < 2:
            continue

        # Emit one row per ordered pair (i upstream, j downstream)
        for idx_i in range(len(window_stats)):
            for idx_j in range(idx_i + 1, len(window_stats)):
                wi = window_stats[idx_i]
                wj = window_stats[idx_j]

                rows.append({
                    "read_id":        read_id,
                    "gene":           gene["gene_name"],
                    "rank_i":         wi["rank"],
                    "rank_j":         wj["rank"],
                    "codon_distance": wj["rank"] - wi["rank"],
                    "z_i":            wi["z_score"],
                    "i_protected":    wi["protected"],
                    "obs_frac_i":     wi["obs_frac"],
                    "exp_frac_i":     wi["exp_frac"],
                    "n_sites_i":      wi["n_sites"],
                    "obs_frac_j":     wj["obs_frac"],
                    "exp_frac_j":     wj["exp_frac"],
                    "n_sites_j":      wj["n_sites"],
                    "z_j":            wj["z_score"],
                })

    return pd.DataFrame(rows)


def aggregate_facilitation(pair_df: pd.DataFrame,
                            min_reads_per_group: int = 10) -> pd.DataFrame:
    """
    For each ordered His pair (rank_i, rank_j), compare edit_frac_j
    between reads classified as i_protected vs i_unprotected.

    Statistics:
      - mean_obs_frac_j in each group
      - facilitation_ratio = mean_j_protected / mean_j_unprotected
        > 1: upstream protection associated with more editing downstream
        < 1: upstream protection associated with less editing downstream (co-protection)
      - Mann-Whitney U test (one-tailed, alternative="greater")
        tests whether obs_frac_j is higher when i is protected

    Returns one row per (codon_distance) aggregated across all gene pairs.
    """
    if pair_df.empty:
        return pd.DataFrame()

    rows = []
    for dist, grp in pair_df.groupby("codon_distance"):
        prot   = grp[grp["i_protected"]]
        unprot = grp[~grp["i_protected"]]

        n_prot   = len(prot)
        n_unprot = len(unprot)

        if n_prot < min_reads_per_group or n_unprot < min_reads_per_group:
            continue

        mean_j_prot   = float(prot["obs_frac_j"].mean())
        mean_j_unprot = float(unprot["obs_frac_j"].mean())

        # Also record mean z_j in each group to see if downstream protection
        # changes too
        mean_z_j_prot   = float(prot["z_j"].mean())
        mean_z_j_unprot = float(unprot["z_j"].mean())

        facilitation_ratio = mean_j_prot / mean_j_unprot \
                             if mean_j_unprot > 0 else np.nan

        _, mw_p = scipy.stats.mannwhitneyu(
            prot["obs_frac_j"].values,
            unprot["obs_frac_j"].values,
            alternative="greater",
        )
        mw_p = max(mw_p, 1e-300)

        # Also test whether z_j differs (two-tailed — could go either way)
        _, mw_p_z = scipy.stats.mannwhitneyu(
            prot["z_j"].values,
            unprot["z_j"].values,
            alternative="two-sided",
        )

        # Binomial test: across all protected-i reads, is the number of
        # sites edited in window j greater than expected under the background
        # rate?  k = total edited sites in j across all protected reads,
        # n = total sites covered in j across all protected reads,
        # p_mean = mean background edit rate across those positions.
        # Upper-tail: P(X >= k | n, p_mean)
        all_j_obs   = prot["obs_frac_j"].values
        all_j_exp   = prot["exp_frac_j"].values
        # Approximate n per read from min_sites lower bound;
        # weight by n_sites if available, else assume equal weight
        if "n_sites_j" in prot.columns:
            n_total = int(prot["n_sites_j"].sum())
            k_total = int((prot["obs_frac_j"] * prot["n_sites_j"]).sum())
            p_bg    = float((prot["exp_frac_j"] * prot["n_sites_j"]).sum()
                            / n_total) if n_total > 0 else float(np.mean(all_j_exp))
        else:
            # Fallback: treat each read as contributing equally
            n_total = len(prot)
            k_total = int(prot["obs_frac_j"].sum() * n_total
                          / n_total * n_total)  # approximate
            k_total = int(round(float(prot["obs_frac_j"].mean()) * n_total))
            p_bg    = float(prot["exp_frac_j"].mean())

        if k_total == 0 or n_total == 0:
            binom_p_upper = 1.0
            binom_p_lower = 1.0
        else:
            # Upper tail: is j MORE edited than expected? (facilitation)
            binom_p_upper = 1.0 - scipy.stats.binom.cdf(
                k_total - 1, n_total, p_bg)
            # Lower tail: is j LESS edited than expected? (co-protection)
            binom_p_lower = scipy.stats.binom.cdf(
                k_total, n_total, p_bg)
        binom_p_upper = max(binom_p_upper, 1e-300)
        binom_p_lower = max(binom_p_lower, 1e-300)

        rows.append({
            "codon_distance":              int(dist),
            "n_i_protected":               n_prot,
            "n_i_unprotected":             n_unprot,
            "mean_obs_frac_j_protected":   mean_j_prot,
            "mean_obs_frac_j_unprotected": mean_j_unprot,
            "facilitation_ratio":          facilitation_ratio,
            "mean_z_j_protected":          mean_z_j_prot,
            "mean_z_j_unprotected":        mean_z_j_unprot,
            "delta_z_j":                   mean_z_j_prot - mean_z_j_unprot,
            "mannwhitney_p_frac":          mw_p,
            "neg_log10_p_mw":              -math.log10(mw_p),
            "mannwhitney_p_z":             mw_p_z,
            "neg_log10_p_z":               -math.log10(max(mw_p_z, 1e-300)),
            "binomial_p_upper":            binom_p_upper,
            "neg_log10_p_binom_upper":      -math.log10(binom_p_upper),
            "binomial_p_lower":            binom_p_lower,
            "neg_log10_p_binom_lower":      -math.log10(binom_p_lower),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_pyx(summary1: pd.DataFrame, summary2: pd.DataFrame,
              label1: str, label2: str, output_prefix: str):
    """
    Figure 1: facilitation_ratio and delta_z_j vs codon distance.
    Figure 2: -log10(binomial p) and -log10(Mann-Whitney p) vs codon distance.
    """
    from pyx import canvas, graph, color, style, deco, path, text as pyx_text

    col1    = color.cmyk(0, 0, 0, 1)
    col2    = color.cmyk(1, 0.5, 0, 0)
    panel_w = 7
    panel_h = 5
    gap     = 2.0

    def _make_canvas(summary1, summary2, left_col, right_col,
                     left_title, right_title,
                     left_hline, right_hline, filename):
        c = canvas.canvas()

        def _panel(xpos, ypos, x_vals, y_vals, col, x_title, y_title,
                   hline_y, title):
            if not x_vals:
                return
            finite = [v for v in y_vals if np.isfinite(v)]
            if not finite:
                return
            pad   = max(abs(max(finite) - hline_y),
                        abs(min(finite) - hline_y), 0.01)
            y_min = hline_y - pad * 1.3
            y_max = hline_y + pad * 1.3
            x_max = max(x_vals) + 1

            g = graph.graphxy(
                width=panel_w, height=panel_h, xpos=xpos, ypos=ypos,
                x=graph.axis.linear(min=0, max=x_max, title=x_title),
                y=graph.axis.linear(min=y_min, max=y_max, title=y_title),
            )
            g.plot(
                graph.data.function(f"y(x)={hline_y:.4f}", min=0, max=x_max),
                [graph.style.line([color.gray(0.5), style.linewidth.thin,
                                   style.linestyle.dashed])])
            pts = [(x, y) for x, y in zip(x_vals, y_vals) if np.isfinite(y)]
            if pts:
                g.plot(graph.data.points(pts, x=1, y=2),
                       [graph.style.symbol(graph.style.symbol.circle,
                                           symbolattrs=[col, deco.filled],
                                           size=0.12)])
                g.plot(graph.data.points(sorted(pts), x=1, y=2),
                       [graph.style.line([col, style.linewidth.thin,
                                          style.linestyle.solid])])
            c.insert(g)
            c.text(g.xpos + g.width / 2., g.ypos + g.height + 0.35,
                   title, [pyx_text.halign.center, pyx_text.size.small])

        for row_idx, (label, col, summary) in enumerate([
            (label1, col1, summary1), (label2, col2, summary2)
        ]):
            if summary.empty:
                continue
            ybase = (1 - row_idx) * (panel_h + gap)
            dists = summary["codon_distance"].tolist()

            _panel(0, ybase, dists, summary[left_col].tolist(), col,
                   "His codon distance (codons apart)", left_title,
                   left_hline, f"{label} {left_title}")

            _panel(panel_w + gap, ybase, dists, summary[right_col].tolist(),
                   col, "His codon distance (codons apart)", right_title,
                   right_hline, f"{label} {right_title}")

        c.writePDFfile(filename)
        print(f"  Saved -> {filename}.pdf", file=sys.stderr)

    # Figure 1: effect size panels
    _make_canvas(
        summary1, summary2,
        left_col="facilitation_ratio",
        right_col="delta_z_j",
        left_title="Facilitation ratio",
        right_title=r"$\Delta z_j$ (protected $-$ free)",
        left_hline=1.0,
        right_hline=0.0,
        filename=f"{output_prefix}_footprint_facilitation_pyx",
    )

    # Figure 2: significance panels — three panels side by side
    from pyx import canvas, graph, color, style, deco, text as pyx_text
    sig_line = -math.log10(0.05)
    c2       = canvas.canvas()
    col1     = color.cmyk(0, 0, 0, 1)
    col2     = color.cmyk(1, 0.5, 0, 0)

    def _sig_panel(xpos, ypos, col, summary, p_col, title):
        if summary.empty or p_col not in summary.columns:
            return
        vals  = summary[p_col].tolist()
        dists = summary["codon_distance"].tolist()
        finite = [v for v in vals if np.isfinite(v)]
        if not finite:
            return
        y_max = max(max(finite) * 1.15, sig_line * 1.5)
        x_max = max(dists) + 1

        g = graph.graphxy(
            width=panel_w, height=panel_h, xpos=xpos, ypos=ypos,
            x=graph.axis.linear(min=0, max=x_max,
                                title="His codon distance (codons apart)"),
            y=graph.axis.linear(min=0, max=y_max,
                                title=r"$-\log_{10}$(p)"),
        )
        g.plot(
            graph.data.function(f"y(x)={sig_line:.4f}", min=0, max=x_max),
            [graph.style.line([color.gray(0.5), style.linewidth.thin,
                               style.linestyle.dashed])])
        pts = [(x, y) for x, y in zip(dists, vals) if np.isfinite(y)]
        if pts:
            g.plot(graph.data.points(pts, x=1, y=2),
                   [graph.style.symbol(graph.style.symbol.circle,
                                       symbolattrs=[col, deco.filled],
                                       size=0.12)])
            g.plot(graph.data.points(sorted(pts), x=1, y=2),
                   [graph.style.line([col, style.linewidth.thin,
                                      style.linestyle.solid])])
        c2.insert(g)
        c2.text(g.xpos + g.width / 2., g.ypos + g.height + 0.35,
                title, [pyx_text.halign.center, pyx_text.size.small])

    for row_idx, (label, col, summary) in enumerate([
        (label1, col1, summary1), (label2, col2, summary2)
    ]):
        if summary.empty:
            continue
        ybase = (1 - row_idx) * (panel_h + gap)
        _sig_panel(0,                  ybase, col, summary,
                   "neg_log10_p_binom_lower",
                   f"{label} binomial lower (co-protection)")
        _sig_panel(panel_w + gap,      ybase, col, summary,
                   "neg_log10_p_binom_upper",
                   f"{label} binomial upper (facilitation)")
        _sig_panel(2*(panel_w + gap),  ybase, col, summary,
                   "neg_log10_p_mw",
                   f"{label} Mann-Whitney")

    sig_path = f"{output_prefix}_footprint_significance_pyx"
    c2.writePDFfile(sig_path)
    print(f"  Saved -> {sig_path}.pdf", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="His codon footprint facilitation analysis."
    )
    p.add_argument("--parquet1",     required=True,
                   help="Reference parquet directory (background frequencies)")
    p.add_argument("--parquet2",     required=True,
                   help="Query parquet directory")
    p.add_argument("--label1",       default="-3AT")
    p.add_argument("--label2",       default="+3AT")
    p.add_argument("--ref",          required=True)
    p.add_argument("--gtf",          required=True)
    p.add_argument("--output",       default="footprint_facilitation")
    p.add_argument("--window",       type=int,   default=25,
                   help="nt each side of His A defining the footprint "
                        "(default: 25)")
    p.add_argument("--min_sites",    type=int,   default=5,
                   help="Min ref=A sites a read must cover in a window "
                        "(default: 5)")
    p.add_argument("--z_threshold",  type=float, default=-1.0,
                   help="Z-score threshold to classify a window as protected. "
                        "More negative = stricter. Default: -1.0 "
                        "(editing ~1 SD below expected)")
    p.add_argument("--min_reads_per_group", type=int, default=10,
                   help="Min reads in each group per codon distance "
                        "(default: 10)")
    p.add_argument("--min_coverage", type=float, default=50.0)
    p.add_argument("--gene_list",    default=None)
    p.add_argument("--cds_spanning", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out  = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print("=== His Codon Footprint Facilitation Analysis ===", file=sys.stderr)
    print(f"  Footprint window:  +/-{args.window} nt", file=sys.stderr)
    print(f"  Z-score threshold: {args.z_threshold} "
          f"(protected = z < {args.z_threshold})", file=sys.stderr)
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
    passing = filter_genes(genes, df_ref, df_qry,
                            min_reads=int(args.min_coverage))
    if not passing:
        print("ERROR: No genes passed coverage filter.", file=sys.stderr)
        sys.exit(1)

    ref_fasta     = pysam.FastaFile(args.ref)
    all_pairs_ref = []
    all_pairs_qry = []

    print(f"\nProcessing {len(passing):,} genes...", file=sys.stderr)

    for i, gname in enumerate(passing):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(passing)}: {gname}...", file=sys.stderr)

        gene      = genes[gname]
        his_sites = find_his_gpos(ref_fasta, gene)

        # Need at least 2 His codons
        if len(his_sites) < 2:
            continue

        chrom  = gene["chrom"]
        strand = gene["strand"]

        def _gdf(df):
            mask = (df["chrom"] == chrom) & (df["gene_strand"] == strand)
            if "read_start" in df.columns and "read_end" in df.columns:
                mask &= (df["read_start"] < gene["gene_end"]) & \
                        (df["read_end"]   > gene["gene_start"])
            return df[mask]

        gene_ref = _gdf(df_ref)
        gene_qry = _gdf(df_qry)

        if gene_ref.empty or gene_qry.empty:
            continue

        # Background frequencies always from reference library
        window_freq = build_window_background_freq(
            gene_ref, ref_fasta, gene, his_sites, args.window)
        if not window_freq:
            continue

        for df_lib, store in [(gene_ref, all_pairs_ref),
                               (gene_qry, all_pairs_qry)]:
            pair_df = collect_footprint_facilitation(
                df_lib, ref_fasta, gene, his_sites, window_freq,
                window=args.window,
                min_sites=args.min_sites,
                z_threshold=args.z_threshold,
                cds_spanning=args.cds_spanning,
            )
            if not pair_df.empty:
                store.append(pair_df)

    ref_fasta.close()

    if not all_pairs_ref and not all_pairs_qry:
        print("No results — exiting.", file=sys.stderr)
        sys.exit(0)

    full_ref = pd.concat(all_pairs_ref, ignore_index=True) \
               if all_pairs_ref else pd.DataFrame()
    full_qry = pd.concat(all_pairs_qry, ignore_index=True) \
               if all_pairs_qry else pd.DataFrame()

    # Save per-read-pair data
    for df, label in [(full_ref, args.label1), (full_qry, args.label2)]:
        if df.empty:
            continue
        safe = label.replace(" ", "_").replace("/", "_")
        path = f"{out}_{safe}_per_read_pairs.csv.gz"
        df.to_csv(path, index=False, compression="gzip")
        n_prot = int(df.drop_duplicates(["read_id", "rank_i"])
                       ["i_protected"].sum())
        n_total = int(df.drop_duplicates(["read_id", "rank_i"])
                        ["i_protected"].count())
        print(f"\n  [{label}] {len(df):,} read-pair observations  |  "
              f"{n_prot:,}/{n_total:,} ({100*n_prot/n_total:.1f}%) "
              f"windows classified as protected  |  -> {path}",
              file=sys.stderr)

    # Aggregate by codon distance
    summary_ref = aggregate_facilitation(full_ref, args.min_reads_per_group)
    summary_qry = aggregate_facilitation(full_qry, args.min_reads_per_group)

    for df, label in [(summary_ref, args.label1), (summary_qry, args.label2)]:
        if df.empty:
            continue
        safe = label.replace(" ", "_").replace("/", "_")
        path = f"{out}_{safe}_summary.csv"
        df.to_csv(path, index=False)
        print(f"  Saved summary -> {path}", file=sys.stderr)
        print(f"  [{label}] Distances with data: "
              f"{sorted(df['codon_distance'].tolist())}",
              file=sys.stderr)

    # Plot
    print("\nGenerating plots...", file=sys.stderr)
    try:
        plot_pyx(summary_ref, summary_qry, args.label1, args.label2, out)
    except Exception as e:
        print(f"  WARNING: pyx plot failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    Tee()
    main()