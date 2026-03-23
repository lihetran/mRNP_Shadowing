#!/usr/bin/env python3
"""
Histidine Codon A→G Editing Comparison Between Two Nanopore BAM Files
======================================================================
Runs the same meta-analysis as histidine_edit_analysis.py on two BAM files,
then computes log2 fold-change in G mismatch rate between them and overlays
CDFs of His A editing.

Usage:
    python histidine_edit_compare.py \
        --bam1 condition1.bam --label1 "WT" \
        --bam2 condition2.bam --label2 "KO" \
        --ref reference.fa \
        --gtf annotation.gtf \
        [--window 50] \
        [--min_coverage 10] \
        [--min_edit_fraction 0.01] \
        [--output comparison]

Requirements:
    pip install pysam pandas numpy matplotlib seaborn
"""

import argparse
import sys
import re
import collections
from pathlib import Path

import pysam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Histidine codon sets (DNA, forward strand) ──────────────────────────────
HIS_CODONS = {"CAT", "CAC"}


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (identical to histidine_edit_analysis.py)
# ─────────────────────────────────────────────────────────────────────────────

def parse_gtf_cds(gtf_path: str) -> dict:
    cds_by_chrom = collections.defaultdict(list)
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "CDS":
                continue
            chrom  = fields[0]
            start  = int(fields[3]) - 1
            end    = int(fields[4])
            strand = fields[6]
            m = re.search(r'transcript_id "([^"]+)"', fields[8])
            tid = m.group(1) if m else "."
            cds_by_chrom[chrom].append((start, end, strand, tid))
    for chrom in cds_by_chrom:
        cds_by_chrom[chrom].sort()
    return dict(cds_by_chrom)


def find_his_positions(ref_fasta: pysam.FastaFile,
                       cds_by_chrom: dict,
                       window: int) -> list:
    # Collect all sites first, then assign per-transcript rank
    sites = []
    for chrom, intervals in cds_by_chrom.items():
        try:
            chrom_len = ref_fasta.get_reference_length(chrom)
        except KeyError:
            continue
        for (cds_start, cds_end, strand, tid) in intervals:
            cds_seq = ref_fasta.fetch(chrom, cds_start, cds_end).upper()
            cds_len = cds_end - cds_start
            for i in range(0, cds_len - 2, 3):
                codon = cds_seq[i:i+3]
                if strand == "-":
                    codon = reverse_complement(codon)
                if codon in HIS_CODONS:
                    if strand == "+":
                        codon_ref_start = cds_start + i
                    else:
                        codon_ref_start = cds_end - i - 3
                    edit_pos  = codon_ref_start + 1
                    win_start = max(0, edit_pos - window)
                    win_end   = min(chrom_len, edit_pos + window + 1)
                    sites.append({
                        "chrom":       chrom,
                        "edit_pos":    edit_pos,
                        "codon_start": codon_ref_start,
                        "strand":      strand,
                        "codon":       codon,
                        "transcript":  tid,
                        "win_start":   win_start,
                        "win_end":     win_end,
                    })

    # Assign 1-based rank of each His codon within its transcript,
    # ordered by position in the CDS walk (transcript order).
    tid_counter: dict = collections.defaultdict(int)
    for site in sites:
        tid_counter[site["transcript"]] += 1
        site["his_rank"] = tid_counter[site["transcript"]]

    return sites


def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]


def complement_base(b: str) -> str:
    return b.translate(str.maketrans("ACGTacgt", "TGCAtgca"))


def count_mismatches_at_site(bam: pysam.AlignmentFile,
                             ref_fasta: pysam.FastaFile,
                             site: dict,
                             min_mapq: int = 20,
                             min_baseq: int = 10) -> dict:
    chrom     = site["chrom"]
    edit_pos  = site["edit_pos"]
    win_start = site["win_start"]
    win_end   = site["win_end"]
    strand    = site["strand"]
    pos_data  = {}

    for pcolumn in bam.pileup(
        chrom, win_start, win_end,
        truncate=True,
        min_mapping_quality=min_mapq,
        min_base_quality=min_baseq,
        stepper="samtools",
        ignore_overlaps=False,
    ):
        ref_pos = pcolumn.reference_pos
        if ref_pos < win_start or ref_pos >= win_end:
            continue
        ref_base = ref_fasta.fetch(chrom, ref_pos, ref_pos + 1).upper()
        rel_pos  = ref_pos - edit_pos

        counts = collections.Counter()
        for pread in pcolumn.pileups:
            if pread.is_del or pread.is_refskip:
                continue
            qbase = pread.alignment.query_sequence[pread.query_position].upper()
            if strand == "-":
                qbase = complement_base(qbase)
            counts[qbase] += 1

        total = sum(counts.values())
        pos_data[rel_pos] = {
            "ref_pos":  ref_pos,
            "ref_base": ref_base if strand == "+" else complement_base(ref_base),
            "A": counts.get("A", 0),
            "G": counts.get("G", 0),
            "C": counts.get("C", 0),
            "T": counts.get("T", 0),
            "cov": total,
        }
    return pos_data


def aggregate_sites(sites: list,
                    bam: pysam.AlignmentFile,
                    ref_fasta: pysam.FastaFile,
                    min_coverage: int,
                    min_mapq: int,
                    min_baseq: int,
                    label: str) -> pd.DataFrame:
    records = []
    for i, site in enumerate(sites):
        if (i + 1) % 100 == 0:
            print(f"  [{label}] Processing site {i+1}/{len(sites)}…", file=sys.stderr)

        pos_data = count_mismatches_at_site(
            bam, ref_fasta, site,
            min_mapq=min_mapq, min_baseq=min_baseq
        )
        for rel_pos, counts in pos_data.items():
            if counts["cov"] < min_coverage:
                continue
            ag_denom = counts["A"] + counts["G"]
            ag_edit  = counts["G"] / ag_denom \
                       if counts["ref_base"] == "A" and ag_denom > 0 else np.nan
            records.append({
                "site_id":      f"{site['chrom']}:{site['edit_pos']}",
                "transcript":   site["transcript"],
                "his_rank":     site["his_rank"],
                "chrom":        site["chrom"],
                "edit_pos":     site["edit_pos"],
                "strand":       site["strand"],
                "codon":        site["codon"],
                "rel_pos":      rel_pos,
                "ref_base":     counts["ref_base"],
                "in_his_codon": rel_pos in (-1, 0, 1),
                "A":            counts["A"],
                "G":            counts["G"],
                "C":            counts["C"],
                "T":            counts["T"],
                "coverage":     counts["cov"],
                "ag_edit_frac": ag_edit,
                "is_his_A":     rel_pos == 0 and counts["ref_base"] == "A",
            })
    return pd.DataFrame(records)


def transcript_normalised_agg(df: pd.DataFrame,
                              group_cols: list = None) -> pd.DataFrame:
    """
    Two-stage aggregation restricted to ref=A positions only, using
    ag_edit_frac = G/(A+G) as the editing metric:
      1. Average ag_edit_frac across all His sites within a transcript at each rel_pos
      2. Average those transcript means across transcripts at each rel_pos

    Non-A ref positions are excluded entirely — they are not informative
    for A→G editing and would mix in sequencing error rates.

    group_cols: additional columns to group by before rel_pos (e.g. ["his_rank"])
    Returns DataFrame with rel_pos (+ group_cols), mean_edit_frac, sem_edit_frac, n_transcripts.
    """
    if group_cols is None:
        group_cols = []

    # Keep ref=A positions that are either:
    #   - the His A itself (rel_pos=0, in_his_codon=True), OR
    #   - outside the His codon entirely (in_his_codon=False)
    # This excludes rel_pos=-1 (C) and +1 (T/C) of THIS site's codon,
    # but also prevents incidental ref=A hits at those positions from
    # OTHER sites' windows contaminating the codon-flanking positions.
    ref_a = df[
        df["ag_edit_frac"].notna() &
        (df["ref_base"] == "A") &
        (~df["in_his_codon"] | df["is_his_A"])
    ].copy()

    # Stage 1: per-transcript mean ag_edit_frac at each rel_pos
    tx_mean = (
        ref_a.groupby(group_cols + ["transcript", "rel_pos"])["ag_edit_frac"]
             .mean()
             .reset_index()
             .rename(columns={"ag_edit_frac": "tx_mean_edit_frac"})
    )

    # Stage 2: grand mean across transcripts
    agg = (
        tx_mean.groupby(group_cols + ["rel_pos"])
               .agg(
                   mean_edit_frac=("tx_mean_edit_frac", "mean"),
                   sem_edit_frac=("tx_mean_edit_frac", lambda x: x.sem()),
                   n_transcripts=("transcript", "nunique"),
               )
               .reset_index()
    )
    return agg


def compute_summaries(df: pd.DataFrame, min_edit_frac: float) -> dict:
    his_a_df = df[df["is_his_A"]].copy()

    # Overall meta-aggregation: ref=A only, ag_edit_frac, transcript-normalised
    rel_agg = transcript_normalised_agg(df)

    # Per-rank meta-aggregation
    rank_agg = {}
    for rank in [1, 2, 3]:
        sub = df[df["his_rank"] == rank]
        if sub.empty:
            rank_agg[rank] = pd.DataFrame()
            continue
        rank_agg[rank] = transcript_normalised_agg(sub)

    edited = his_a_df[his_a_df["ag_edit_frac"] >= min_edit_frac].copy()
    edited = edited.sort_values("ag_edit_frac", ascending=False)

    return {
        "his_a_sites":       his_a_df,
        "rel_position_agg":  rel_agg,
        "rank_agg":          rank_agg,
        "edit_frac_dist":    his_a_df["ag_edit_frac"].dropna(),
        "edited_sites":      edited,
    }


def compute_log2fc_agg(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple:
    """
    Transcript-normalised log2FC of A→G editing fraction at ref=A positions only.

    Steps:
      1. Restrict each BAM's data to ref=A positions with valid ag_edit_frac
      2. Per-transcript mean ag_edit_frac at each rel_pos
      3. Merge on (transcript, his_rank, rel_pos)
      4. Compute log2FC where both conditions have ag_edit_frac > 0
      5. Average log2FC across transcripts at each rel_pos

    Returns:
        log2fc_agg      — DataFrame(rel_pos, mean_log2fc, sem_log2fc, n_transcripts)
        rank_log2fc_agg — dict: rank → same DataFrame restricted to that rank
    """
    def _tx_mean(df):
        ref_a = df[
            df["ag_edit_frac"].notna() &
            (df["ref_base"] == "A") &
            (~df["in_his_codon"] | df["is_his_A"])
        ].copy()
        return (
            ref_a.groupby(["transcript", "his_rank", "rel_pos"])["ag_edit_frac"]
                 .mean()
                 .reset_index()
        )

    tm1 = _tx_mean(df1)
    tm2 = _tx_mean(df2)

    merged = tm1.merge(tm2,
                       on=["transcript", "his_rank", "rel_pos"],
                       suffixes=("_1", "_2"))

    # Only compute log2FC where both conditions have observed editing —
    # no pseudocount, so transcripts with ag_edit_frac == 0 in either BAM are excluded
    merged = merged[
        (merged["ag_edit_frac_1"] > 0) & (merged["ag_edit_frac_2"] > 0)
    ].copy()
    merged["log2fc"] = np.log2(merged["ag_edit_frac_2"] / merged["ag_edit_frac_1"])

    def _agg(sub):
        if sub.empty:
            return pd.DataFrame()
        return (
            sub.groupby("rel_pos")
               .agg(
                   mean_log2fc=("log2fc", "mean"),
                   sem_log2fc=("log2fc", lambda x: x.sem()),
                   n_transcripts=("transcript", "nunique"),
               )
               .reset_index()
        )

    log2fc_agg = _agg(merged)

    rank_log2fc_agg = {}
    for rank in [1, 2, 3]:
        sub = merged[merged["his_rank"] == rank]
        rank_log2fc_agg[rank] = _agg(sub)

    return log2fc_agg, rank_log2fc_agg




COLORS = {"bam1": "steelblue", "bam2": "coral"}


def plot_comparison(s1: dict, s2: dict,
                    label1: str, label2: str,
                    output_prefix: str,
                    window: int,
                    min_edit_frac: float,
                    strat_raw_dfs: dict = None):

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Histidine A→G Editing Comparison: {label1} vs {label2}",
        fontsize=14, fontweight="bold"
    )

    c1, c2 = COLORS["bam1"], COLORS["bam2"]

    # ── Panel 1: Meta-plot BAM1 ───────────────────────────────────────────────
    _meta_panel(axes[0, 0], s1["rel_position_agg"], label1, c1, window)

    # ── Panel 2: Meta-plot BAM2 ───────────────────────────────────────────────
    _meta_panel(axes[0, 1], s2["rel_position_agg"], label2, c2, window)

    # ── Panel 3: log2FC meta-plot ─────────────────────────────────────────────
    ax = axes[0, 2]
    log2fc_agg = s1["log2fc_agg"].set_index("rel_pos")
    log2fc  = log2fc_agg["mean_log2fc"]
    sem_fc  = log2fc_agg["sem_log2fc"]
    common  = log2fc.index

    ax.axhline(0, color="grey", lw=1, ls="--")
    ax.axvline(0, color="crimson", lw=1.5, ls="--", label="His A")
    ax.axvspan(-1, 1, alpha=0.08, color="grey", label="His codon")
    ax.fill_between(common, log2fc, 0,
                    where=(log2fc >= 0), interpolate=True,
                    color=c2, alpha=0.6, label=f"Higher in {label2}")
    ax.fill_between(common, log2fc, 0,
                    where=(log2fc < 0), interpolate=True,
                    color=c1, alpha=0.6, label=f"Higher in {label1}")
    ax.plot(common, log2fc, color="black", lw=1.2)
    ax.fill_between(common, log2fc - sem_fc, log2fc + sem_fc,
                    alpha=0.15, color="black")
    ax.set_xlim(-window, window)
    ax.set_xlabel("Position relative to His codon A")
    ax.set_ylabel(f"log2FC ({label2} / {label1})")
    ax.set_title("log2 Fold-Change in G mismatch rate\n(mean ± SEM across sites)")
    ax.legend(fontsize=8)

    # ── Panel 4: Overlaid CDFs ────────────────────────────────────────────────
    ax = axes[1, 0]
    for fracs, label, color in [
        (s1["edit_frac_dist"], label1, c1),
        (s2["edit_frac_dist"], label2, c2),
    ]:
        if len(fracs) > 0:
            sf = np.sort(fracs)
            cdf = np.arange(1, len(sf) + 1) / len(sf)
            median = float(np.median(sf))
            ax.plot(sf, cdf, color=color, lw=2,
                    label=f"{label} (n={len(sf):,}, med={median:.3f})")
            ax.axvline(median, color=color, lw=1, ls="--", alpha=0.7)
    ax.axhline(0.5, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("A→G edit fraction at His A")
    ax.set_ylabel("Cumulative fraction of sites")
    ax.set_title("CDF of editing at His A")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9)

    # ── Panel 5: Overall per-read editing efficiency CDF ─────────────────────
    ax = axes[1, 1]
    if strat_raw_dfs is not None:
        for key, label, color in [("bam1", label1, c1), ("bam2", label2, c2)]:
            raw = strat_raw_dfs[key]
            read_rows = (
                raw[raw["rel_pos"].isna()]
                   .drop_duplicates(subset=["read_name"])
                   [["read_name", "read_edit_eff"]]
                   .dropna(subset=["read_edit_eff"])
            )
            if len(read_rows) == 0:
                continue
            eff    = np.sort(read_rows["read_edit_eff"].values)
            cdf    = np.arange(1, len(eff) + 1) / len(eff)
            median = float(np.median(eff))
            ax.plot(eff, cdf, color=color, lw=2,
                    label=f"{label} (n={len(eff):,}, med={median:.3f})")
            ax.axvline(median, color=color, lw=1, ls="--", alpha=0.7)
        ax.axhline(0.5, color="grey", lw=0.8, ls=":", alpha=0.6)
        ax.set_xlabel("Per-read A→G editing efficiency")
        ax.set_ylabel("Cumulative fraction of reads")
        ax.set_title("Overall editing efficiency CDF\n(QC: should overlap if libraries matched)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No read-level data\n(run with strat_raw_dfs)",
                ha="center", va="center", transform=ax.transAxes)

    # ── Panel 6: log2FC distribution at His A sites only ─────────────────────
    ax = axes[1, 2]
    ha1 = s1["his_a_sites"][["site_id", "ag_edit_frac"]].dropna()
    ha2 = s2["his_a_sites"][["site_id", "ag_edit_frac"]].dropna()
    merged = ha1.merge(ha2, on="site_id", suffixes=("_1", "_2"))
    valid  = merged[
        (merged["ag_edit_frac_1"] > 0) & (merged["ag_edit_frac_2"] > 0)
    ]
    if len(valid) > 0:
        fc = np.log2(valid["ag_edit_frac_2"] / valid["ag_edit_frac_1"])
        ax.hist(fc, bins=60, color="mediumpurple", edgecolor="white")
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.axvline(fc.median(), color="crimson", lw=1.5, ls="--",
                   label=f"Median log2FC = {fc.median():.2f}")
        ax.set_xlabel(f"log2FC ({label2} / {label1}) at His A")
        ax.set_ylabel("Number of sites")
        ax.set_title(f"Per-site log2FC at His A\n(n={len(valid):,} sites with G>0 in both)")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No sites with G>0 in both conditions",
                ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plot_path = f"{output_prefix}_comparison_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved comparison plots → {plot_path}", file=sys.stderr)

    # ── Second figure: per-rank meta and log2FC ───────────────────────────────
    plot_rank_comparison(s1, s2, label1, label2, output_prefix, window)


def _meta_panel(ax, rel_agg: pd.DataFrame, label: str, color: str, window: int):
    ax.axvline(0, color="crimson", lw=1.5, ls="--", label="His A")
    ax.axvspan(-1, 1, alpha=0.08, color=color, label="His codon")
    ax.plot(rel_agg["rel_pos"], rel_agg["mean_edit_frac"], color=color, lw=2)
    ax.fill_between(
        rel_agg["rel_pos"],
        rel_agg["mean_edit_frac"] - rel_agg["sem_edit_frac"],
        rel_agg["mean_edit_frac"] + rel_agg["sem_edit_frac"],
        alpha=0.25, color=color,
    )
    ax.set_xlim(-window, window)
    ax.set_xlabel("Position relative to His codon A")
    ax.set_ylabel("Mean A→G edit fraction [G/(A+G)]")
    ax.set_title(f"Meta-analysis: {label}\n(ref=A positions only)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.legend(fontsize=9)


def plot_rank_comparison(s1: dict, s2: dict,
                         label1: str, label2: str,
                         output_prefix: str,
                         window: int):
    """
    3-column × 2-row figure.
    Columns = His codon rank 1, 2, 3.
    Row 0   = overlaid meta-plots (BAM1 + BAM2 G/cov) for that rank.
    Row 1   = log2FC meta-plot for that rank.
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    c1, c2 = COLORS["bam1"], COLORS["bam2"]

    rank_labels = {1: "1st His codon", 2: "2nd His codon", 3: "3rd His codon"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(
        f"His A→G Editing by Codon Rank: {label1} vs {label2}",
        fontsize=14, fontweight="bold"
    )

    for col, rank in enumerate([1, 2, 3]):
        r1 = s1["rank_agg"].get(rank, pd.DataFrame())
        r2 = s2["rank_agg"].get(rank, pd.DataFrame())
        rank_title = rank_labels[rank]

        # ── Row 0: overlaid meta-plots ────────────────────────────────────────
        ax = axes[0, col]
        ax.axvline(0, color="crimson", lw=1.5, ls="--", label="His A")
        ax.axvspan(-1, 1, alpha=0.08, color="grey")

        for df_r, label, color in [(r1, label1, c1), (r2, label2, c2)]:
            if df_r.empty:
                continue
            n = int(df_r["n_transcripts"].max()) if "n_transcripts" in df_r.columns else 0
            ax.plot(df_r["rel_pos"], df_r["mean_edit_frac"],
                    color=color, lw=2, label=f"{label} (n={n:,} tx)")
            ax.fill_between(
                df_r["rel_pos"],
                df_r["mean_edit_frac"] - df_r["sem_edit_frac"],
                df_r["mean_edit_frac"] + df_r["sem_edit_frac"],
                alpha=0.2, color=color,
            )

        if r1.empty and r2.empty:
            ax.text(0.5, 0.5, "No sites at this rank",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(-window, window)
        ax.set_xlabel("Position relative to His A")
        ax.set_ylabel("Mean A→G edit fraction [G/(A+G)]")
        ax.set_title(f"{rank_title} — meta-analysis\n(ref=A positions only)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax.legend(fontsize=8)

        # ── Row 1: log2FC ─────────────────────────────────────────────────────
        ax = axes[1, col]
        rank_log2fc = s1["rank_log2fc_agg"].get(rank, pd.DataFrame())
        if not rank_log2fc.empty:
            rfc = rank_log2fc.set_index("rel_pos")
            log2fc = rfc["mean_log2fc"]
            sem_fc = rfc["sem_log2fc"]
            common = log2fc.index

            ax.axhline(0, color="grey", lw=1, ls="--")
            ax.axvline(0, color="crimson", lw=1.5, ls="--", label="His A")
            ax.axvspan(-1, 1, alpha=0.08, color="grey", label="His codon")
            ax.fill_between(common, log2fc, 0,
                            where=(log2fc >= 0), interpolate=True,
                            color=c2, alpha=0.6, label=f"Higher in {label2}")
            ax.fill_between(common, log2fc, 0,
                            where=(log2fc < 0), interpolate=True,
                            color=c1, alpha=0.6, label=f"Higher in {label1}")
            ax.plot(common, log2fc, color="black", lw=1.2)
            ax.fill_between(common, log2fc - sem_fc, log2fc + sem_fc,
                            alpha=0.15, color="black")
        else:
            ax.text(0.5, 0.5, "Insufficient data for one or both conditions",
                    ha="center", va="center", transform=ax.transAxes)

        ax.set_xlim(-window, window)
        ax.set_xlabel("Position relative to His A")
        ax.set_ylabel(f"log2FC ({label2} / {label1})")
        ax.set_title(f"{rank_title} — log2FC\n(mean ± SEM across sites)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = f"{output_prefix}_rank_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved rank comparison plots → {plot_path}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Periodicity analyses: autocorrelation + co-occurrence
# ─────────────────────────────────────────────────────────────────────────────

def compute_autocorrelation(rel_agg: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute the normalised autocorrelation of the mean A→G edit fraction
    vector across the window (ref=A positions only). Positions not present
    in rel_agg (non-A ref bases) are filled with the mean before correlating.
    A peak at lag 21 would indicate ~21 nt periodicity in editing.

    Returns DataFrame(lag, autocorr) for lags 0..window.
    """
    all_pos = np.arange(-window, window + 1)
    idx = rel_agg.set_index("rel_pos")["mean_edit_frac"]
    vec = np.array([idx.get(p, np.nan) for p in all_pos])

    # Fill missing positions (non-A ref) with the mean so they don't
    # artificially drive the autocorrelation
    vec[np.isnan(vec)] = np.nanmean(vec)

    # Mean-centre and autocorrelate
    vec     = vec - vec.mean()
    full_ac = np.correlate(vec, vec, mode="full")
    mid     = len(full_ac) // 2
    ac      = full_ac[mid:]
    ac      = ac / ac[0]   # normalise so lag-0 = 1

    lags = np.arange(len(ac))
    return pd.DataFrame({"lag": lags[:window + 1], "autocorr": ac[:window + 1]})


def build_co_occurrence_matrix(sites: list,
                                bam: pysam.AlignmentFile,
                                ref_fasta: pysam.FastaFile,
                                window: int,
                                min_mapq: int = 20,
                                min_baseq: int = 10) -> np.ndarray:
    """
    For every read that spans a His site window, record which relative
    positions show an A→G mismatch (ref=A, read=G). Accumulate a
    (2*window+1) × (2*window+1) co-occurrence count matrix where entry
    [i, j] = number of (site, read) observations where both rel_pos i
    and rel_pos j showed A→G mismatches simultaneously.

    Diagonal entry [i, i] = number of reads showing a mismatch at rel_pos i
    (i.e. the marginal count), useful for normalisation.

    Returns the raw co-occurrence count matrix (float64).
    """
    n     = 2 * window + 1
    mat   = np.zeros((n, n), dtype=np.float64)
    # rel_pos → matrix index: rel_pos 0 maps to index `window`
    offset = window

    for site in sites:
        chrom    = site["chrom"]
        edit_pos = site["edit_pos"]
        win_start = site["win_start"]
        win_end   = site["win_end"]

        # Fetch reference sequence for the window once
        ref_seq = ref_fasta.fetch(chrom, win_start, win_end).upper()

        for read in bam.fetch(chrom, win_start, win_end):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality < min_mapq:
                continue
            if read.query_sequence is None:
                continue

            # Build ref_pos → query_base map using the alignment pairs,
            # restricted to the window and to positions where ref=A
            ag_rel_positions = []
            for qpos, rpos in read.get_aligned_pairs(matches_only=True):
                if rpos < win_start or rpos >= win_end:
                    continue
                ref_idx  = rpos - win_start
                ref_base = ref_seq[ref_idx] if ref_idx < len(ref_seq) else "N"
                if ref_base != "A":
                    continue
                # Check base quality
                if read.query_qualities is not None:
                    if read.query_qualities[qpos] < min_baseq:
                        continue
                qbase = read.query_sequence[qpos].upper()
                if qbase == "G":
                    rel = rpos - edit_pos
                    if -window <= rel <= window:
                        ag_rel_positions.append(rel)

            # Accumulate co-occurrences for all pairs on this read
            if len(ag_rel_positions) == 0:
                continue
            idxs = [r + offset for r in ag_rel_positions]
            for ii in idxs:
                for jj in idxs:
                    mat[ii, jj] += 1

    return mat


def normalise_co_occurrence(mat: np.ndarray) -> np.ndarray:
    """
    Normalise co-occurrence matrix so entry [i,j] = P(edit at j | edit at i).
    Divide each row by its diagonal (marginal count). Positions with zero
    marginal are set to NaN.
    """
    diag   = np.diag(mat).copy()
    normed = mat.astype(np.float64).copy()
    for i in range(len(diag)):
        if diag[i] > 0:
            normed[i, :] = normed[i, :] / diag[i]
        else:
            normed[i, :] = np.nan
    return normed


def plot_periodicity(sites: list,
                     bam1: pysam.AlignmentFile,
                     bam2: pysam.AlignmentFile,
                     ref_fasta: pysam.FastaFile,
                     s1: dict, s2: dict,
                     label1: str, label2: str,
                     output_prefix: str,
                     window: int,
                     min_mapq: int,
                     min_baseq: int):
    """
    Two-row figure:
      Row 0: autocorrelation of G/cov rate for BAM1 and BAM2 (overlaid)
      Row 1: normalised co-occurrence heatmaps for BAM1 and BAM2
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    c1, c2 = COLORS["bam1"], COLORS["bam2"]

    print("  Computing autocorrelations…", file=sys.stderr)
    ac1 = compute_autocorrelation(s1["rel_position_agg"], window)
    ac2 = compute_autocorrelation(s2["rel_position_agg"], window)

    print("  Building co-occurrence matrices (read-level, this may take a while)…",
          file=sys.stderr)
    mat1 = build_co_occurrence_matrix(
        sites, bam1, ref_fasta, window, min_mapq=min_mapq, min_baseq=min_baseq)
    mat2 = build_co_occurrence_matrix(
        sites, bam2, ref_fasta, window, min_mapq=min_mapq, min_baseq=min_baseq)

    norm1 = normalise_co_occurrence(mat1)
    norm2 = normalise_co_occurrence(mat2)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Editing Periodicity: {label1} vs {label2}",
        fontsize=14, fontweight="bold"
    )
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    tick_step  = max(1, window // 5)
    tick_pos   = np.arange(0, 2 * window + 1, tick_step)
    tick_label = [str(t - window) for t in tick_pos]

    # ── Panel [0, 0:2]: overlaid autocorrelation ──────────────────────────────
    ax_ac = fig.add_subplot(gs[0, :2])
    ax_ac.axhline(0, color="grey", lw=0.8, ls="--")
    ax_ac.axvline(21, color="black", lw=1, ls=":", alpha=0.6, label="lag = 21 nt")
    for ac, label, color in [(ac1, label1, c1), (ac2, label2, c2)]:
        ax_ac.plot(ac["lag"], ac["autocorr"], color=color, lw=2, label=label)
    ax_ac.set_xlabel("Lag (nt)")
    ax_ac.set_ylabel("Normalised autocorrelation")
    ax_ac.set_title("Autocorrelation of G/cov rate across window")
    ax_ac.set_xlim(0, window)
    ax_ac.legend(fontsize=9)

    # ── Panel [0, 2]: autocorrelation difference ──────────────────────────────
    ax_diff = fig.add_subplot(gs[0, 2])
    common_lags = ac1["lag"].values
    diff = ac2.set_index("lag")["autocorr"] - ac1.set_index("lag")["autocorr"]
    ax_diff.axhline(0, color="grey", lw=0.8, ls="--")
    ax_diff.axvline(21, color="black", lw=1, ls=":", alpha=0.6, label="lag = 21 nt")
    ax_diff.fill_between(common_lags, diff.values, 0,
                         where=(diff.values >= 0), color=c2, alpha=0.6,
                         label=f"Higher in {label2}")
    ax_diff.fill_between(common_lags, diff.values, 0,
                         where=(diff.values < 0), color=c1, alpha=0.6,
                         label=f"Higher in {label1}")
    ax_diff.plot(common_lags, diff.values, color="black", lw=1)
    ax_diff.set_xlabel("Lag (nt)")
    ax_diff.set_ylabel(f"Autocorr difference ({label2} − {label1})")
    ax_diff.set_title("Autocorrelation difference")
    ax_diff.set_xlim(0, window)
    ax_diff.legend(fontsize=8)

    # ── Panels [1, 0] and [1, 1]: co-occurrence heatmaps ─────────────────────
    vmax = np.nanpercentile(
        np.concatenate([norm1[~np.isnan(norm1)], norm2[~np.isnan(norm2)]]), 95
    )
    for col, (norm, label) in enumerate([(norm1, label1), (norm2, label2)]):
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(norm, origin="lower", aspect="auto",
                       cmap="magma", vmin=0, vmax=vmax,
                       extent=[-window - 0.5, window + 0.5,
                               -window - 0.5, window + 0.5])
        ax.axhline(0, color="white", lw=0.6, ls="--", alpha=0.5)
        ax.axvline(0, color="white", lw=0.6, ls="--", alpha=0.5)
        # Mark ±21 nt diagonals
        for offset_nt in [-21, 21]:
            ax.axvline(offset_nt, color="cyan", lw=1, ls=":", alpha=0.7)
            ax.axhline(offset_nt, color="cyan", lw=1, ls=":", alpha=0.7)
        plt.colorbar(im, ax=ax, label="P(edit at j | edit at i)", shrink=0.8)
        ax.set_xlabel("Rel. position j")
        ax.set_ylabel("Rel. position i")
        ax.set_title(f"Co-occurrence: {label}")

    # ── Panel [1, 2]: heatmap difference (BAM2 − BAM1) ───────────────────────
    ax = fig.add_subplot(gs[1, 2])
    diff_mat = norm2 - norm1
    vlim = np.nanpercentile(np.abs(diff_mat[~np.isnan(diff_mat)]), 95)
    im = ax.imshow(diff_mat, origin="lower", aspect="auto",
                   cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                   extent=[-window - 0.5, window + 0.5,
                           -window - 0.5, window + 0.5])
    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.axvline(0, color="black", lw=0.6, ls="--", alpha=0.5)
    for offset_nt in [-21, 21]:
        ax.axvline(offset_nt, color="black", lw=1, ls=":", alpha=0.7)
        ax.axhline(offset_nt, color="black", lw=1, ls=":", alpha=0.7)
    plt.colorbar(im, ax=ax, label=f"ΔP ({label2} − {label1})", shrink=0.8)
    ax.set_xlabel("Rel. position j")
    ax.set_ylabel("Rel. position i")
    ax.set_title(f"Co-occurrence difference\n({label2} − {label1})")

    plot_path = f"{output_prefix}_periodicity.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved periodicity plots → {plot_path}", file=sys.stderr)

    # Save raw matrices
    pos_labels = [str(i - window) for i in range(2 * window + 1)]
    for norm, label, key in [(norm1, label1, "bam1"), (norm2, label2, "bam2")]:
        pd.DataFrame(norm, index=pos_labels, columns=pos_labels).to_csv(
            f"{output_prefix}_{key}_co_occurrence.csv"
        )
    print(f"  Saved co-occurrence matrices → {output_prefix}_bam*_co_occurrence.csv",
          file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# His A stratified analysis: edited vs unedited reads
# ─────────────────────────────────────────────────────────────────────────────

def collect_his_a_stratified(sites: list,
                              bam: pysam.AlignmentFile,
                              ref_fasta: pysam.FastaFile,
                              window: int,
                              min_mapq: int = 20,
                              min_baseq: int = 10) -> pd.DataFrame:
    """
    For each His site, iterate over reads spanning the His A (rel_pos=0).
    Classify each read as edited (G at His A) or unedited (A at His A).
    For every other ref=A position in the window on that read, record
    whether that position also shows a G mismatch.

    Returns a long-form DataFrame with one row per (site, read, rel_pos):
        site_id, transcript, his_rank, rel_pos, his_a_edited, ag_edit
    where ag_edit is 1 if the read shows G at that ref=A position, 0 if A.
    """
    records = []
    for site in sites:
        chrom     = site["chrom"]
        edit_pos  = site["edit_pos"]
        win_start = site["win_start"]
        win_end   = site["win_end"]

        ref_seq = ref_fasta.fetch(chrom, win_start, win_end).upper()

        for read in bam.fetch(chrom, win_start, win_end):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality < min_mapq:
                continue
            if read.query_sequence is None:
                continue

            # Build rpos → qbase map for this read, ref=A positions only
            ref_a_calls = {}   # rel_pos → query base (A or G)
            for qpos, rpos in read.get_aligned_pairs(matches_only=True):
                if rpos < win_start or rpos >= win_end:
                    continue
                ref_base = ref_seq[rpos - win_start]
                if ref_base != "A":
                    continue
                if read.query_qualities is not None:
                    if read.query_qualities[qpos] < min_baseq:
                        continue
                qbase = read.query_sequence[qpos].upper()
                if qbase in ("A", "G"):
                    rel = rpos - edit_pos
                    if -window <= rel <= window:
                        ref_a_calls[rel] = qbase

            # Need to have a call at the His A to classify this read
            if 0 not in ref_a_calls:
                continue

            his_a_edited = ref_a_calls[0] == "G"

            # Per-read overall editing efficiency: fraction of all ref=A
            # positions on this read (excluding His A and codon flanks)
            # that show G
            window_calls = {
                rp: qb for rp, qb in ref_a_calls.items()
                if rp not in (0, -1, 1)
            }
            n_ref_a    = len(window_calls)
            n_ag       = sum(1 for qb in window_calls.values() if qb == "G")
            read_edit_eff = n_ag / n_ref_a if n_ref_a > 0 else np.nan

            # Emit one summary row per read for the efficiency CDF
            records.append({
                "site_id":        f"{chrom}:{edit_pos}",
                "transcript":     site["transcript"],
                "his_rank":       site["his_rank"],
                "rel_pos":        np.nan,          # sentinel: read-level summary row
                "his_a_edited":   his_a_edited,
                "ag_edit":        np.nan,
                "read_name":      read.query_name,
                "read_edit_eff":  read_edit_eff,
            })

            # Record per-position rows (excluding His A and codon flanks)
            for rel_pos, qbase in ref_a_calls.items():
                if rel_pos in (0, -1, 1):
                    continue
                records.append({
                    "site_id":        f"{chrom}:{edit_pos}",
                    "transcript":     site["transcript"],
                    "his_rank":       site["his_rank"],
                    "rel_pos":        rel_pos,
                    "his_a_edited":   his_a_edited,
                    "ag_edit":        1 if qbase == "G" else 0,
                    "read_name":      read.query_name,
                    "read_edit_eff":  read_edit_eff,
                })

    return pd.DataFrame(records)


def compute_stratified_log2fc(df: pd.DataFrame,
                              pseudo: float = 1e-3) -> pd.DataFrame:
    """
    Two-stage transcript-normalised log2FC of editing fraction between
    His-A-edited reads and His-A-unedited reads.

    A pseudocount is added to both edited and unedited fractions before
    taking log2FC, so transcripts with zero editing in the unedited set
    (genuine signal) still contribute rather than being dropped.

    Stage 1: per-transcript mean ag_edit fraction at each rel_pos,
             separately for edited and unedited read sets.
    Stage 2: log2FC((edited + pseudo) / (unedited + pseudo)) per transcript
             per rel_pos, then mean ± SEM across transcripts.

    Returns DataFrame(rel_pos, mean_log2fc, sem_log2fc, n_transcripts).
    """
    if df.empty:
        return pd.DataFrame()

    # Per-position rows only (read-level summary rows have rel_pos=NaN)
    df = df[df["rel_pos"].notna()].copy()

    # Stage 1: per-transcript mean editing fraction by his_a_edited status
    tx_mean = (
        df.groupby(["transcript", "his_rank", "rel_pos", "his_a_edited"])["ag_edit"]
          .mean()
          .reset_index()
          .rename(columns={"ag_edit": "mean_edit"})
    )

    # Pivot so edited and unedited are columns
    tx_pivot = tx_mean.pivot_table(
        index=["transcript", "his_rank", "rel_pos"],
        columns="his_a_edited",
        values="mean_edit",
    ).reset_index()
    tx_pivot.columns.name = None
    tx_pivot = tx_pivot.rename(columns={False: "unedited", True: "edited"})

    # Keep rows where both strata are present (pseudocount handles zeros)
    tx_pivot = tx_pivot.dropna(subset=["edited", "unedited"])

    if tx_pivot.empty:
        return pd.DataFrame()

    tx_pivot["log2fc"] = np.log2(
        (tx_pivot["edited"]   + pseudo) /
        (tx_pivot["unedited"] + pseudo)
    )

    # Stage 2: mean ± SEM across transcripts at each rel_pos
    agg = (
        tx_pivot.groupby("rel_pos")
                .agg(
                    mean_log2fc=("log2fc", "mean"),
                    sem_log2fc=("log2fc", lambda x: x.sem()),
                    n_transcripts=("transcript", "nunique"),
                )
                .reset_index()
    )
    return agg


def plot_his_a_stratified(strat_dfs: dict,
                           strat_raw_dfs: dict,
                           labels: dict,
                           output_prefix: str,
                           window: int):
    """
    Two rows:
      Row 0: log2FC in editing (edited vs unedited reads) per BAM
      Row 1: CDF of per-read overall editing efficiency for both BAMs
             overlaid, split by his_a_edited status
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    c1, c2 = COLORS["bam1"], COLORS["bam2"]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "Editing at ref=A positions: His-A-edited vs His-A-unedited reads",
        fontsize=13, fontweight="bold"
    )
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    # ── Row 0: log2FC panels per BAM ─────────────────────────────────────────
    for col, (key, label, color) in enumerate([
        ("bam1", labels["bam1"], c1),
        ("bam2", labels["bam2"], c2),
    ]):
        ax = fig.add_subplot(gs[0, col])
        agg = strat_dfs[key]
        if agg.empty:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(label)
            continue

        fc     = agg.set_index("rel_pos")["mean_log2fc"]
        sem_fc = agg.set_index("rel_pos")["sem_log2fc"]
        pos    = fc.index

        ax.axhline(0, color="grey", lw=1, ls="--")
        ax.axvline(0, color="crimson", lw=1.5, ls="--", label="His A")
        ax.axvspan(-1, 1, alpha=0.08, color=color, label="His codon")
        ax.axvline( 21, color="grey", lw=1, ls=":", alpha=0.6, label="±21 nt")
        ax.axvline(-21, color="grey", lw=1, ls=":", alpha=0.6)
        ax.fill_between(pos, fc, 0,
                        where=(fc >= 0), interpolate=True,
                        color=color, alpha=0.5, label="More in His-A-edited")
        ax.fill_between(pos, fc, 0,
                        where=(fc < 0), interpolate=True,
                        color="grey", alpha=0.4, label="More in His-A-unedited")
        ax.plot(pos, fc, color="black", lw=1.2)
        ax.fill_between(pos, fc - sem_fc, fc + sem_fc, alpha=0.15, color="black")
        ax.set_xlim(-window, window)
        ax.set_xlabel("Position relative to His A")
        ax.set_ylabel("log2FC (edited / unedited reads)")
        ax.set_title(f"{label} — log2FC\n(n={int(agg['n_transcripts'].max()):,} transcripts)")
        ax.legend(fontsize=8)

    # ── Row 0 col 2: overlaid log2FC for both BAMs ────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.axhline(0, color="grey", lw=1, ls="--")
    ax.axvline(0, color="crimson", lw=1.5, ls="--", label="His A")
    ax.axvspan(-1, 1, alpha=0.08, color="grey")
    ax.axvline( 21, color="grey", lw=1, ls=":", alpha=0.6, label="±21 nt")
    ax.axvline(-21, color="grey", lw=1, ls=":", alpha=0.6)
    for key, label, color in [("bam1", labels["bam1"], c1), ("bam2", labels["bam2"], c2)]:
        agg = strat_dfs[key]
        if agg.empty:
            continue
        fc     = agg.set_index("rel_pos")["mean_log2fc"]
        sem_fc = agg.set_index("rel_pos")["sem_log2fc"]
        pos    = fc.index
        ax.plot(pos, fc, color=color, lw=2, label=label)
        ax.fill_between(pos, fc - sem_fc, fc + sem_fc, alpha=0.15, color=color)
    ax.set_xlim(-window, window)
    ax.set_xlabel("Position relative to His A")
    ax.set_ylabel("log2FC (edited / unedited reads)")
    ax.set_title("Overlay: both libraries")
    ax.legend(fontsize=9)

    # ── Row 1: per-read editing efficiency CDFs ───────────────────────────────
    # Extract read-level rows (rel_pos is NaN), deduplicate by read_name
    # so each read contributes once even if it covers multiple His sites
    ax_all   = fig.add_subplot(gs[1, 0])   # overall efficiency, both BAMs
    ax_strat = fig.add_subplot(gs[1, 1])   # split by his_a_edited, both BAMs
    ax_box   = fig.add_subplot(gs[1, 2])   # boxplot summary

    box_data   = []
    box_labels = []
    box_colors = []

    for key, label, color in [("bam1", labels["bam1"], c1), ("bam2", labels["bam2"], c2)]:
        raw = strat_raw_dfs[key]
        read_rows = (
            raw[raw["rel_pos"].isna()]
               .drop_duplicates(subset=["read_name"])
               [["read_name", "his_a_edited", "read_edit_eff"]]
               .dropna(subset=["read_edit_eff"])
        )

        # Overall CDF (all reads regardless of His A status)
        eff = np.sort(read_rows["read_edit_eff"].values)
        cdf = np.arange(1, len(eff) + 1) / len(eff)
        ax_all.plot(eff, cdf, color=color, lw=2,
                    label=f"{label} (n={len(eff):,}, med={np.median(eff):.3f})")
        ax_all.axvline(np.median(eff), color=color, lw=1, ls="--", alpha=0.7)

        # Stratified CDFs (edited vs unedited reads)
        for his_edited, ls, suffix in [(True, "-", " | His-A edited"),
                                        (False, "--", " | His-A unedited")]:
            sub = read_rows[read_rows["his_a_edited"] == his_edited]["read_edit_eff"]
            if len(sub) == 0:
                continue
            s = np.sort(sub.values)
            c = np.arange(1, len(s) + 1) / len(s)
            ax_strat.plot(s, c, color=color, lw=2, ls=ls,
                          label=f"{label}{suffix}\n(n={len(s):,})")

        # Collect for boxplot
        for his_edited, suffix in [(True, "\nedited"), (False, "\nunedited")]:
            sub = read_rows[read_rows["his_a_edited"] == his_edited]["read_edit_eff"]
            box_data.append(sub.values)
            box_labels.append(f"{label}{suffix}")
            box_colors.append(color)

    # Finish overall CDF
    ax_all.axhline(0.5, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax_all.set_xlabel("Per-read A→G editing efficiency")
    ax_all.set_ylabel("Cumulative fraction of reads")
    ax_all.set_title("Overall editing efficiency CDF\n(all reads)")
    ax_all.set_xlim(0, 1)
    ax_all.set_ylim(0, 1)
    ax_all.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax_all.legend(fontsize=8)

    # Finish stratified CDF
    ax_strat.axhline(0.5, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax_strat.set_xlabel("Per-read A→G editing efficiency")
    ax_strat.set_ylabel("Cumulative fraction of reads")
    ax_strat.set_title("Editing efficiency CDF\nby His-A editing status")
    ax_strat.set_xlim(0, 1)
    ax_strat.set_ylim(0, 1)
    ax_strat.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax_strat.legend(fontsize=7)

    # Boxplot
    bp = ax_box.boxplot(box_data, patch_artist=True, notch=True,
                         medianprops=dict(color="black", lw=2))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax_box.set_xticks(range(1, len(box_labels) + 1))
    ax_box.set_xticklabels(box_labels, fontsize=8)
    ax_box.set_ylabel("Per-read A→G editing efficiency")
    ax_box.set_title("Editing efficiency distribution\nby library and His-A status")

    plt.tight_layout()
    plot_path = f"{output_prefix}_his_a_stratified.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved His-A stratified plots → {plot_path}", file=sys.stderr)


def compute_quiet_zone(strat_raw: pd.DataFrame,
                        half_width: int = 10) -> pd.DataFrame:
    """
    For each read, determine whether it has a 'quiet zone' — zero A→G
    mismatches at any ref=A position within [-half_width, +half_width]
    including the His A at rel_pos=0 (but still excluding codon flanks ±1
    which are never ref=A for CAT/CAC).

    The His A is included because a blocking protein would prevent editing
    there too, so a truly blocked read shows no editing across the full zone.

    Works from both per-position rows and the His A call stored in
    his_a_edited on the read-level summary rows.
    Returns a DataFrame with one row per unique (read_name, site_id):
        read_name, site_id, his_a_edited, has_quiet_zone, read_edit_eff
    """
    pos_rows = strat_raw[strat_raw["rel_pos"].notna()].copy()

    # Restrict to positions within the quiet zone window (excludes ±1 already
    # since those were never appended for CAT/CAC codons)
    pos_rows = pos_rows[pos_rows["rel_pos"].abs() <= half_width]

    # For each (read_name, site_id): any G mismatch in the flanking positions?
    any_flank_edit = (
        pos_rows.groupby(["read_name", "site_id"])["ag_edit"]
                .max()
                .reset_index()
                .rename(columns={"ag_edit": "any_flank_edit"})
    )

    # Join back his_a_edited and read_edit_eff from the read-level summary rows
    read_rows = (
        strat_raw[strat_raw["rel_pos"].isna()]
                 .drop_duplicates(subset=["read_name", "site_id"])
                 [["read_name", "site_id", "his_a_edited", "read_edit_eff"]]
    )
    result = any_flank_edit.merge(read_rows, on=["read_name", "site_id"], how="left")

    # Quiet zone = no editing in flanks AND His A itself is not edited
    result["has_quiet_zone"] = (
        (result["any_flank_edit"] == 0) & (~result["his_a_edited"])
    )
    return result


def plot_quiet_zone(quiet_dfs: dict,
                    labels: dict,
                    output_prefix: str,
                    half_width: int = 10):
    """
    For each library, show:
      - Fraction of reads with a quiet zone, split by His-A editing status
      - CDF of overall editing efficiency for quiet-zone vs non-quiet-zone reads
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    c1, c2 = COLORS["bam1"], COLORS["bam2"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Quiet zone analysis: ±{half_width} nt around His A (21 nt window, His A included)\n"
        "Reads with zero A→G editing across entire zone (consistent with RBP blocking)",
        fontsize=12, fontweight="bold"
    )

    # ── Panel 0: fraction with quiet zone per library, by His-A status ────────
    ax = axes[0]
    bar_width = 0.35
    x = np.arange(2)   # his_a_edited=False, True
    x_labels = ["His-A unedited", "His-A edited"]

    for i, (key, label, color) in enumerate([
        ("bam1", labels["bam1"], c1),
        ("bam2", labels["bam2"], c2),
    ]):
        df = quiet_dfs[key]
        fracs = []
        ns    = []
        for his_edited in [False, True]:
            sub = df[df["his_a_edited"] == his_edited]
            frac = sub["has_quiet_zone"].mean() if len(sub) > 0 else 0
            fracs.append(frac)
            ns.append(len(sub))
        bars = ax.bar(x + i * bar_width, fracs, bar_width,
                      label=label, color=color, alpha=0.8, edgecolor="white")
        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"n={n:,}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Fraction of reads with quiet zone")
    ax.set_title("Fraction with quiet zone\nby His-A editing status")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9)

    # ── Panel 1: CDF of editing efficiency, quiet vs non-quiet ───────────────
    ax = axes[1]
    for key, label, color in [("bam1", labels["bam1"], c1),
                               ("bam2", labels["bam2"], c2)]:
        df = quiet_dfs[key].dropna(subset=["read_edit_eff"])
        for has_qz, ls, suffix in [(True, "-", " quiet"), (False, "--", " non-quiet")]:
            sub = df[df["has_quiet_zone"] == has_qz]["read_edit_eff"]
            if len(sub) == 0:
                continue
            s = np.sort(sub.values)
            c = np.arange(1, len(s) + 1) / len(s)
            ax.plot(s, c, color=color, lw=2, ls=ls,
                    label=f"{label}{suffix} (n={len(s):,})")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("Per-read A→G editing efficiency")
    ax.set_ylabel("Cumulative fraction of reads")
    ax.set_title("Editing efficiency CDF\nquiet zone vs non-quiet zone")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=7)

    # ── Panel 2: His-A editing rate in quiet vs non-quiet reads ──────────────
    ax = axes[2]
    bar_width = 0.35
    x = np.arange(2)
    x_labels = ["Quiet zone", "Non-quiet zone"]

    for i, (key, label, color) in enumerate([
        ("bam1", labels["bam1"], c1),
        ("bam2", labels["bam2"], c2),
    ]):
        df = quiet_dfs[key]
        fracs = []
        ns    = []
        for has_qz in [True, False]:
            sub = df[df["has_quiet_zone"] == has_qz]
            frac = sub["his_a_edited"].mean() if len(sub) > 0 else 0
            fracs.append(frac)
            ns.append(len(sub))
        bars = ax.bar(x + i * bar_width, fracs, bar_width,
                      label=label, color=color, alpha=0.8, edgecolor="white")
        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"n={n:,}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Fraction of reads with His A edited")
    ax.set_title("His-A editing rate\nby quiet zone status")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9)

    plt.tight_layout()
    plot_path = f"{output_prefix}_quiet_zone.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved quiet zone plots → {plot_path}", file=sys.stderr)

    # Save tables
    for key, label in labels.items():
        quiet_dfs[key].to_csv(f"{output_prefix}_{key}_quiet_zone.csv", index=False)
    print(f"  Saved quiet zone tables → {output_prefix}_bam*_quiet_zone.csv",
          file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare His A→G editing between two nanopore BAM files."
    )
    p.add_argument("--bam1",   required=True, help="BAM file for condition 1")
    p.add_argument("--bam2",   required=True, help="BAM file for condition 2")
    p.add_argument("--label1", default="BAM1", help="Label for condition 1 (default: BAM1)")
    p.add_argument("--label2", default="BAM2", help="Label for condition 2 (default: BAM2)")
    p.add_argument("--ref",    required=True, help="Reference FASTA (indexed)")
    p.add_argument("--gtf",    required=True, help="GTF annotation file")
    p.add_argument("--window", type=int, default=50,
                   help="Nucleotides either side of His A (default: 50)")
    p.add_argument("--min_coverage", type=int, default=10,
                   help="Minimum read depth per position (default: 10)")
    p.add_argument("--min_edit_fraction", type=float, default=0.01,
                   help="Min A→G fraction to call a site edited (default: 0.01)")
    p.add_argument("--min_mapq", type=int, default=20)
    p.add_argument("--min_baseq", type=int, default=10)
    p.add_argument("--output", default="his_edit_comparison",
                   help="Output file prefix (default: his_edit_comparison)")
    p.add_argument("--chroms", nargs="*", default=None,
                   help="Restrict to specific chromosomes/contigs")
    return p.parse_args()


def main():
    args = parse_args()
    out  = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print("=== Histidine A→G Editing Comparison ===", file=sys.stderr)

    # ── Shared reference data (only parsed once) ──────────────────────────────
    print("Opening reference and parsing GTF…", file=sys.stderr)
    ref_fasta    = pysam.FastaFile(args.ref)
    cds_by_chrom = parse_gtf_cds(args.gtf)
    if args.chroms:
        cds_by_chrom = {k: v for k, v in cds_by_chrom.items() if k in args.chroms}

    print("Scanning CDS for histidine codons…", file=sys.stderr)
    sites = find_his_positions(ref_fasta, cds_by_chrom, window=args.window)
    print(f"  Found {len(sites):,} His codon sites.", file=sys.stderr)
    if not sites:
        print("ERROR: No histidine sites found. Check chromosome names.", file=sys.stderr)
        sys.exit(1)

    dfs = {}
    for key, bam_path, label in [
        ("bam1", args.bam1, args.label1),
        ("bam2", args.bam2, args.label2),
    ]:
        print(f"\nProcessing {label} ({bam_path})…", file=sys.stderr)
        bam = pysam.AlignmentFile(bam_path, "rb")
        df  = aggregate_sites(
            sites, bam, ref_fasta,
            min_coverage=args.min_coverage,
            min_mapq=args.min_mapq,
            min_baseq=args.min_baseq,
            label=label,
        )
        bam.close()

        if df.empty:
            print(f"WARNING: No positions passed coverage filter for {label}.",
                  file=sys.stderr)
            sys.exit(0)

        # save per-position data
        csv_path = f"{out}_{key}_per_position.csv.gz"
        df.to_csv(csv_path, index=False, compression="gzip")
        print(f"  Saved per-position data → {csv_path}", file=sys.stderr)

        dfs[key] = df

    ref_fasta.close()

    # ── Summaries ─────────────────────────────────────────────────────────────
    print("\nComputing summaries…", file=sys.stderr)
    # compute_summaries adds g_rate column in-place; call before compute_log2fc_agg
    s1 = compute_summaries(dfs["bam1"], min_edit_frac=args.min_edit_fraction)
    s2 = compute_summaries(dfs["bam2"], min_edit_frac=args.min_edit_fraction)

    # Transcript-normalised log2FC aggregations (window-level and rank-level)
    # g_rate is now present on both dfs after compute_summaries
    log2fc_agg, rank_log2fc_agg = compute_log2fc_agg(dfs["bam1"], dfs["bam2"])
    s1["log2fc_agg"]      = log2fc_agg
    s1["rank_log2fc_agg"] = rank_log2fc_agg

    # Save meta-aggregation tables
    for s, key in [(s1, "bam1"), (s2, "bam2")]:
        s["rel_position_agg"].to_csv(f"{out}_{key}_meta_aggregation.csv", index=False)
        s["edited_sites"].to_csv(f"{out}_{key}_edited_sites.csv", index=False)

    # Save merged per-site editing table
    ha1 = s1["his_a_sites"][["site_id", "chrom", "edit_pos", "transcript",
                              "codon", "ag_edit_frac", "coverage"]].dropna()
    ha2 = s2["his_a_sites"][["site_id", "ag_edit_frac", "coverage"]].dropna()
    merged = ha1.merge(ha2, on="site_id",
                       suffixes=(f"_{args.label1}", f"_{args.label2}"))
    # exclude sites where either condition has zero editing — no pseudocount
    merged = merged[
        (merged[f"ag_edit_frac_{args.label1}"] > 0) &
        (merged[f"ag_edit_frac_{args.label2}"] > 0)
    ].copy()
    merged["log2fc"] = np.log2(
        merged[f"ag_edit_frac_{args.label2}"] /
        merged[f"ag_edit_frac_{args.label1}"]
    )
    merged = merged.sort_values("log2fc", ascending=False)
    merged.to_csv(f"{out}_per_site_log2fc.csv", index=False)
    print(f"  Saved per-site log2FC → {out}_per_site_log2fc.csv", file=sys.stderr)

    # ── Print summary stats ───────────────────────────────────────────────────
    for s, label in [(s1, args.label1), (s2, args.label2)]:
        ha = s["his_a_sites"]
        ed = s["edited_sites"]
        print(f"\n── {label} ───────────────────────────────", file=sys.stderr)
        print(f"  His A sites with sufficient coverage : {len(ha):,}", file=sys.stderr)
        print(f"  Sites with A→G frac ≥ {args.min_edit_fraction:.2%} : {len(ed):,}",
              file=sys.stderr)
        if len(ha) > 0:
            print(f"  Median A→G frac at His A : {ha['ag_edit_frac'].median():.4f}",
                  file=sys.stderr)
            print(f"  Mean   A→G frac at His A : {ha['ag_edit_frac'].mean():.4f}",
                  file=sys.stderr)

    if len(merged) > 0:
        print(f"\n── Comparison ({args.label1} vs {args.label2}) ──────────────",
              file=sys.stderr)
        print(f"  Shared His A sites : {len(merged):,}", file=sys.stderr)
        print(f"  Median log2FC      : {merged['log2fc'].median():.3f}", file=sys.stderr)
        print(f"  Sites up in {args.label2:10s} (log2FC > 0) : "
              f"{(merged['log2fc'] > 0).sum():,}", file=sys.stderr)
        print(f"  Sites up in {args.label1:10s} (log2FC < 0) : "
              f"{(merged['log2fc'] < 0).sum():,}", file=sys.stderr)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating periodicity plots…", file=sys.stderr)
    bam1 = pysam.AlignmentFile(args.bam1, "rb")
    bam2 = pysam.AlignmentFile(args.bam2, "rb")
    ref_fasta = pysam.FastaFile(args.ref)
    plot_periodicity(
        sites, bam1, bam2, ref_fasta,
        s1, s2, args.label1, args.label2,
        out, args.window, args.min_mapq, args.min_baseq,
    )

    print("\nCollecting read-level stratified data…", file=sys.stderr)
    strat_dfs     = {}
    strat_raw_dfs = {}
    for key, label in [("bam1", args.label1), ("bam2", args.label2)]:
        bam_fresh = pysam.AlignmentFile(
            args.bam1 if key == "bam1" else args.bam2, "rb"
        )
        print(f"  [{label}]…", file=sys.stderr)
        strat_raw = collect_his_a_stratified(
            sites, bam_fresh, ref_fasta, args.window,
            min_mapq=args.min_mapq, min_baseq=args.min_baseq,
        )
        bam_fresh.close()
        strat_raw_dfs[key] = strat_raw
        strat_dfs[key]     = compute_stratified_log2fc(strat_raw)
        strat_raw.to_csv(f"{out}_{key}_his_a_stratified.csv.gz",
                         index=False, compression="gzip")

    print("\nGenerating comparison plots…", file=sys.stderr)
    plot_comparison(s1, s2, args.label1, args.label2,
                    out, args.window, args.min_edit_fraction,
                    strat_raw_dfs=strat_raw_dfs)

    print("\nGenerating His-A stratified plots…", file=sys.stderr)
    plot_his_a_stratified(
        strat_dfs,
        strat_raw_dfs=strat_raw_dfs,
        labels={"bam1": args.label1, "bam2": args.label2},
        output_prefix=out,
        window=args.window,
    )

    print("\nGenerating quiet zone plots…", file=sys.stderr)
    quiet_dfs = {
        key: compute_quiet_zone(strat_raw_dfs[key])
        for key in ("bam1", "bam2")
    }
    plot_quiet_zone(
        quiet_dfs,
        labels={"bam1": args.label1, "bam2": args.label2},
        output_prefix=out,
    )

    bam1.close()
    bam2.close()
    ref_fasta.close()

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()