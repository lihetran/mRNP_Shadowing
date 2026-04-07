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
            m2 = re.search(r'gene_name "([^"]+)"', fields[8])
            gname = m2.group(1) if m2 else "."
            cds_by_chrom[chrom].append((start, end, strand, tid, gname))
    for chrom in cds_by_chrom:
        cds_by_chrom[chrom].sort()
    return dict(cds_by_chrom)


def parse_gtf_biotypes(gtf_path: str) -> dict:
    """
    Parse all features from the GTF that have a transcript_biotype attribute.
    Returns a dict: chrom → sorted list of (start0, end0, biotype, gene_name)
    covering every feature interval (gene, transcript, exon, CDS, etc.).
    Using all feature types ensures reads in UTRs/introns are still assigned.
    """
    idx = collections.defaultdict(list)
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            if fields[2] not in ("transcript", "gene"):
                continue
            m = re.search(r'transcript_biotype "([^"]+)"', fields[8])
            if not m:
                m = re.search(r'gene_biotype "([^"]+)"', fields[8])
            if not m:
                continue
            gn = re.search(r'gene_name "([^"]+)"', fields[8])
            chrom     = fields[0]
            start     = int(fields[3]) - 1
            end       = int(fields[4])
            biotype   = m.group(1)
            gene_name = gn.group(1) if gn else "."
            idx[chrom].append((start, end, biotype, gene_name))
    for chrom in idx:
        idx[chrom].sort()
    return dict(idx)


def overlapping_biotypes(chrom: str,
                          read_start: int,
                          read_end: int,
                          biotype_idx: dict) -> set:
    """
    Return a set of (biotype, gene_name) tuples for all intervals overlapping
    [read_start, read_end) on chrom.
    """
    intervals = biotype_idx.get(chrom, [])
    result = set()
    for (start, end, biotype, gene_name) in intervals:
        if start >= read_end:
            break
        if end > read_start:
            result.add((biotype, gene_name))
    return result


def assign_read_biotypes(bam_path: str,
                          biotype_idx: dict,
                          min_mapq: int = 0) -> dict:
    """
    Iterate every read in the BAM and assign it to all biotypes whose
    intervals overlap its alignment. Reads with no overlap get biotype
    'intergenic'.

    Ribosomal protein genes (gene_name matching RPL* or RPS*) are
    reclassified from 'protein_coding' to 'ribosomal_protein' so they
    can be distinguished from other protein-coding genes in plots.

    Returns dict: read_name → frozenset of biotypes (strings).
    """
    def _reclassify(biotype: str, gene_name: str) -> str:
        if biotype == "protein_coding" and \
           (gene_name.upper().startswith("RPL") or
            gene_name.upper().startswith("RPS")):
            return "ribosomal_protein"
        return biotype

    read_biotypes = {}
    bam = pysam.AlignmentFile(bam_path, "rb")
    for read in bam:
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if read.mapping_quality < min_mapq:
            continue
        hits = overlapping_biotypes(
            read.reference_name,
            read.reference_start,
            read.reference_end,
            biotype_idx,
        )
        if hits:
            biotypes = frozenset(
                _reclassify(bt, gn) for bt, gn in hits
            )
        else:
            biotypes = frozenset(["intergenic"])
        read_biotypes[read.query_name] = biotypes
    bam.close()
    return read_biotypes


def plot_biotype_counts(biotype_maps: dict,
                         labels: dict,
                         output_prefix: str):
    """
    For each library, plot a stacked bar of read counts by biotype.
    Reads assigned to multiple biotypes are counted once per biotype.

    biotype_maps: dict key → read_name → frozenset of biotypes
    labels:       dict key → display label
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Collect all biotypes and counts per library
    lib_counts = {}
    all_biotypes = set()
    for key, bmap in biotype_maps.items():
        counts = collections.Counter()
        for biotypes in bmap.values():
            for bt in biotypes:
                counts[bt] += 1
        lib_counts[key] = counts
        all_biotypes.update(counts.keys())

    # Sort biotypes by total count descending
    total = collections.Counter()
    for counts in lib_counts.values():
        total.update(counts)
    biotype_order = [bt for bt, _ in total.most_common()]

    # Build DataFrame for plotting
    rows = []
    for key, counts in lib_counts.items():
        for bt in biotype_order:
            rows.append({
                "library": labels[key],
                "biotype": bt,
                "count":   counts.get(bt, 0),
            })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Read counts by transcript biotype", fontsize=13, fontweight="bold")

    palette = sns.color_palette("tab20", n_colors=len(biotype_order))
    color_map = dict(zip(biotype_order, palette))

    for ax, (key, label) in zip(axes, labels.items()):
        sub = df[df["library"] == label].set_index("biotype").loc[biotype_order]
        bars = ax.barh(biotype_order[::-1],
                       sub.loc[biotype_order[::-1], "count"],
                       color=[color_map[bt] for bt in biotype_order[::-1]],
                       edgecolor="white")
        # Annotate counts
        for bar, bt in zip(bars, biotype_order[::-1]):
            n = sub.loc[bt, "count"]
            ax.text(bar.get_width() + sub["count"].max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{n:,}", va="center", fontsize=8)
        ax.set_xlabel("Number of reads\n(reads overlapping multiple biotypes counted per biotype)")
        ax.set_title(label)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )

    plt.tight_layout()
    plot_path = f"{output_prefix}_biotype_counts.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved biotype count plots → {plot_path}", file=sys.stderr)

    # Save counts table
    df.to_csv(f"{output_prefix}_biotype_counts.csv", index=False)



def plot_editing_efficiency_by_biotype(strat_raw_dfs: dict,
                                        biotype_maps: dict,
                                        labels: dict,
                                        output_prefix: str,
                                        min_reads: int = 50):
    """
    For each biotype, plot the CDF of per-read editing efficiency,
    overlaying both libraries. One panel per biotype with >= min_reads
    in at least one library.

    Uses the read-level summary rows from strat_raw (rel_pos is NaN)
    joined to biotype_maps on read_name.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    c1, c2 = COLORS["bam1"], COLORS["bam2"]

    # Build per-library DataFrame of (read_name, read_edit_eff, biotype)
    # expanding multi-biotype reads into one row per biotype
    lib_dfs = {}
    for key in ("bam1", "bam2"):
        raw = strat_raw_dfs[key]
        read_rows = (
            raw[raw["rel_pos"].isna()]
               .drop_duplicates(subset=["read_name"])
               [["read_name", "read_edit_eff"]]
               .dropna(subset=["read_edit_eff"])
        )
        bmap = biotype_maps[key]
        rows = []
        for _, row in read_rows.iterrows():
            for bt in bmap.get(row["read_name"], {"unassigned"}):
                rows.append({
                    "read_name":     row["read_name"],
                    "read_edit_eff": row["read_edit_eff"],
                    "biotype":       bt,
                })
        lib_dfs[key] = pd.DataFrame(rows)

    # Find biotypes present with enough reads in at least one library
    all_biotypes = set()
    for key, df in lib_dfs.items():
        counts = df.groupby("biotype")["read_name"].nunique()
        all_biotypes.update(counts[counts >= min_reads].index)

    if not all_biotypes:
        print("  No biotypes with sufficient reads for efficiency plot.",
              file=sys.stderr)
        return

    # Sort by total read count descending
    total_counts = collections.Counter()
    for key, df in lib_dfs.items():
        for bt, n in df.groupby("biotype")["read_name"].nunique().items():
            total_counts[bt] += n
    biotype_order = [bt for bt, _ in total_counts.most_common()
                     if bt in all_biotypes]

    n_bt  = len(biotype_order)
    ncols = min(3, n_bt)
    nrows = (n_bt + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(6 * ncols, 4 * nrows),
                              squeeze=False)
    fig.suptitle(
        "Per-read A→G editing efficiency by biotype",
        fontsize=13, fontweight="bold"
    )

    for idx, bt in enumerate(biotype_order):
        ax = axes[idx // ncols][idx % ncols]
        for key, label, color in [("bam1", labels["bam1"], c1),
                                   ("bam2", labels["bam2"], c2)]:
            sub = lib_dfs[key][lib_dfs[key]["biotype"] == bt]["read_edit_eff"]
            if len(sub) == 0:
                continue
            s      = np.sort(sub.values)
            cdf    = np.arange(1, len(s) + 1) / len(s)
            median = float(np.median(s))
            ax.plot(s, cdf, color=color, lw=2,
                    label=f"{label} (n={len(s):,}, med={median:.3f})")
            ax.axvline(median, color=color, lw=1, ls="--", alpha=0.6)
        ax.axhline(0.5, color="grey", lw=0.8, ls=":", alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("A→G editing efficiency")
        ax.set_ylabel("Cumulative fraction")
        ax.set_title(bt)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend(fontsize=7)

    # Hide unused axes
    for idx in range(n_bt, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    plot_path = f"{output_prefix}_efficiency_by_biotype.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved efficiency by biotype → {plot_path}", file=sys.stderr)



def find_codon_positions(ref_fasta: pysam.FastaFile,
                         cds_by_chrom: dict,
                         window: int,
                         target_codons: set,
                         codon_label: str = "codon") -> list:
    """
    General version of find_his_positions — finds any set of target codons.
    edit_pos is always the middle base (position 1) of the codon, which is
    the A in His (CAT/CAC) and equivalent position in control codons.
    """
    sites = []
    for chrom, intervals in cds_by_chrom.items():
        try:
            chrom_len = ref_fasta.get_reference_length(chrom)
        except KeyError:
            continue
        for (cds_start, cds_end, strand, tid, gname) in intervals:
            cds_seq = ref_fasta.fetch(chrom, cds_start, cds_end).upper()
            cds_len = cds_end - cds_start
            for i in range(0, cds_len - 2, 3):
                codon = cds_seq[i:i+3]
                if strand == "-":
                    codon = reverse_complement(codon)
                if codon in target_codons:
                    codon_ref_start = cds_start + i
                    edit_pos        = cds_start + i + 1  # middle base, both strands
                    win_start = max(0, edit_pos - window)
                    win_end   = min(chrom_len, edit_pos + window + 1)
                    sites.append({
                        "chrom":       chrom,
                        "edit_pos":    edit_pos,
                        "codon_start": codon_ref_start,
                        "strand":      strand,
                        "codon":       codon,
                        "transcript":  tid,
                        "gene_name":   gname,
                        "win_start":   win_start,
                        "win_end":     win_end,
                        "codon_label": codon_label,
                    })

    tid_counter: dict = collections.defaultdict(int)
    for site in sites:
        tid_counter[site["transcript"]] += 1
        site["his_rank"] = tid_counter[site["transcript"]]

    seen_positions = set()
    deduped = []
    for site in sites:
        key = (site["chrom"], site["edit_pos"])
        if key not in seen_positions:
            seen_positions.add(key)
            deduped.append(site)

    print(f"  [{codon_label}] {len(deduped):,} unique sites "
          f"(deduplicated from {len(sites):,}).", file=sys.stderr)
    return deduped


def find_his_positions(ref_fasta: pysam.FastaFile,
                       cds_by_chrom: dict,
                       window: int) -> list:
    return find_codon_positions(
        ref_fasta, cds_by_chrom, window,
        target_codons=HIS_CODONS,
        codon_label="His",
    )


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
        if strand == "-":
            rel_pos = -rel_pos

        counts = collections.Counter()
        for pread in pcolumn.pileups:
            if pread.is_del or pread.is_refskip:
                continue
            qbase_raw = pread.alignment.query_sequence[pread.query_position].upper()
            # Convert to transcript coordinates using XOR of read strand and gene strand:
            # needs_complement when read orientation doesn't match gene orientation
            needs_complement = pread.alignment.is_reverse != (strand == "-")
            qbase = complement_base(qbase_raw) if needs_complement else qbase_raw
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
                              group_cols: list = None,
                              pseudo: float = 1e-3) -> pd.DataFrame:
    """
    Two-stage aggregation restricted to ref=A positions only, using
    ag_edit_frac = G/(A+G) + pseudocount as the editing metric:
      1. Average (ag_edit_frac + pseudo) across all His sites within a
         transcript at each rel_pos
      2. Average those transcript means across transcripts at each rel_pos

    A pseudocount is added so that sites with zero editing still contribute
    rather than being dropped, keeping the meta-plot anchored to all sites.

    Non-A ref positions and codon flanks are excluded (see in_his_codon filter).
    group_cols: additional columns to group by before rel_pos (e.g. ["his_rank"])
    Returns DataFrame with rel_pos (+ group_cols), mean_edit_frac, sem_edit_frac, n_transcripts.
    """
    if group_cols is None:
        group_cols = []

    ref_a = df[
        df["ag_edit_frac"].notna() &
        (df["ref_base"] == "A") &
        (~df["in_his_codon"] | df["is_his_A"])
    ].copy()
    ref_a["ag_edit_frac_ps"] = ref_a["ag_edit_frac"] + pseudo

    # Stage 1: per-transcript mean at each rel_pos
    tx_mean = (
        ref_a.groupby(group_cols + ["transcript", "rel_pos"])["ag_edit_frac_ps"]
             .mean()
             .reset_index()
             .rename(columns={"ag_edit_frac_ps": "tx_mean_edit_frac"})
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


def compute_log2fc_agg(df1: pd.DataFrame, df2: pd.DataFrame,
                       pseudo: float = 1e-3) -> tuple:
    """
    Transcript-normalised log2FC of A→G editing fraction at ref=A positions only.
    A pseudocount is added so transcripts with zero editing in one condition
    still contribute rather than being excluded.

    Steps:
      1. Restrict each BAM's data to ref=A positions (excl. codon flanks)
      2. Per-transcript mean (ag_edit_frac + pseudo) at each rel_pos
      3. Merge on (transcript, his_rank, rel_pos)
      4. log2FC = log2(mean_2 / mean_1) per transcript per rel_pos
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
        ref_a["ag_edit_frac_ps"] = ref_a["ag_edit_frac"] + pseudo
        return (
            ref_a.groupby(["transcript", "his_rank", "rel_pos"])["ag_edit_frac_ps"]
                 .mean()
                 .reset_index()
        )

    tm1 = _tx_mean(df1)
    tm2 = _tx_mean(df2)

    merged = tm1.merge(tm2,
                       on=["transcript", "his_rank", "rel_pos"],
                       suffixes=("_1", "_2"))

    merged["log2fc"] = np.log2(
        merged["ag_edit_frac_ps_2"] / merged["ag_edit_frac_ps_1"]
    )

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
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
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
    ax = axes[1, 0]
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
    ax.set_title("log2 Fold-Change in editing\n(mean ± SEM across transcripts)")
    ax.legend(fontsize=8)

    # ── Panel 4: CDFs — His A editing fraction + overall read efficiency ──────
    ax = axes[1, 1]
    # His A editing fraction CDF
    for fracs, label, color in [
        (s1["edit_frac_dist"], label1, c1),
        (s2["edit_frac_dist"], label2, c2),
    ]:
        if len(fracs) > 0:
            sf = np.sort(fracs)
            cdf = np.arange(1, len(sf) + 1) / len(sf)
            median = float(np.median(sf))
            ax.plot(sf, cdf, color=color, lw=2,
                    label=f"{label} His A (med={median:.3f})")
            ax.axvline(median, color=color, lw=1, ls="--", alpha=0.7)
    # Overall read efficiency CDF (dashed)
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
            ax.plot(eff, cdf, color=color, lw=2, ls="--",
                    label=f"{label} read eff. (med={median:.3f})")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("A→G fraction")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("CDF: His A editing (solid) vs\noverall read efficiency (dashed)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=8)

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
    offset = window

    for site in sites:
        chrom    = site["chrom"]
        edit_pos = site["edit_pos"]
        win_start = site["win_start"]
        win_end   = site["win_end"]
        strand    = site["strand"]

        ref_seq = ref_fasta.fetch(chrom, win_start, win_end).upper()

        for read in bam.fetch(chrom, win_start, win_end):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality < min_mapq:
                continue
            if read.query_sequence is None:
                continue

            is_rev = read.is_reverse
            needs_complement = is_rev != (strand == "-")

            ag_rel_positions = []
            for qpos, rpos in read.get_aligned_pairs(matches_only=True):
                if rpos < win_start or rpos >= win_end:
                    continue
                ref_idx          = rpos - win_start
                ref_base_genomic = ref_seq[ref_idx] if ref_idx < len(ref_seq) else "N"
                ref_base_tx      = complement_base(ref_base_genomic) \
                                   if needs_complement else ref_base_genomic
                if ref_base_tx != "A":
                    continue
                if read.query_qualities is not None:
                    if read.query_qualities[qpos] < min_baseq:
                        continue
                qbase_raw = read.query_sequence[qpos].upper()
                qbase_tx  = complement_base(qbase_raw) \
                            if needs_complement else qbase_raw
                if qbase_tx == "G":
                    rel = rpos - edit_pos
                    if strand == "-":
                        rel = -rel
                    if -window <= rel <= window:
                        ag_rel_positions.append(rel)

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
    records   = []
    chrom_seq_cache: dict = {}   # chrom → full sequence, loaded once
    for site in sites:
        chrom     = site["chrom"]
        edit_pos  = site["edit_pos"]
        win_start = site["win_start"]
        win_end   = site["win_end"]
        strand    = site["strand"]

        if chrom not in chrom_seq_cache:
            chrom_seq_cache[chrom] = ref_fasta.fetch(chrom).upper()

        ref_seq      = chrom_seq_cache[chrom][win_start:win_end]
        full_ref_seq = chrom_seq_cache[chrom]

        for read in bam.fetch(chrom, win_start, win_end):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality < min_mapq:
                continue
            if read.query_sequence is None:
                continue

            is_rev = read.is_reverse

            # Convert genomic bases to transcript coordinates.
            # A read is in transcript orientation when:
            #   - it is forward on a plus-strand gene, OR
            #   - it is reverse on a minus-strand gene
            # In both cases (is_rev XOR gene_minus) == False → no complement needed.
            # Otherwise complement both ref and query to get transcript bases.
            needs_complement = is_rev != (strand == "-")

            ref_a_calls = {}
            for qpos, rpos in read.get_aligned_pairs(matches_only=True):
                if rpos < win_start or rpos >= win_end:
                    continue
                ref_base_genomic = ref_seq[rpos - win_start]
                ref_base_tx = complement_base(ref_base_genomic) \
                              if needs_complement else ref_base_genomic
                if ref_base_tx != "A":
                    continue
                if read.query_qualities is not None:
                    if read.query_qualities[qpos] < min_baseq:
                        continue
                qbase_raw = read.query_sequence[qpos].upper()
                qbase_tx  = complement_base(qbase_raw) \
                            if needs_complement else qbase_raw
                if qbase_tx in ("A", "G"):
                    rel = rpos - edit_pos
                    if strand == "-":
                        rel = -rel
                    if -window <= rel <= window:
                        ref_a_calls[rel] = qbase_tx

            if 0 not in ref_a_calls:
                continue

            his_a_edited = ref_a_calls[0] == "G"

            # Per-read overall editing efficiency: genome-wide, transcript-aware
            edits  = 0
            num_as = 0
            for qpos, rpos in read.get_aligned_pairs():
                if rpos is None or qpos is None:
                    continue
                if rpos >= len(full_ref_seq):
                    continue
                ref_base_genomic = full_ref_seq[rpos]
                ref_base_tx = complement_base(ref_base_genomic) \
                              if needs_complement else ref_base_genomic
                if ref_base_tx == "A":
                    num_as += 1
                    qbase_raw = read.query_sequence[qpos].upper()
                    qbase_tx  = complement_base(qbase_raw) \
                                if needs_complement else qbase_raw
                    if qbase_tx == "G":
                        edits += 1
            read_edit_eff = edits / num_as if num_as > 0 else np.nan

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



def plot_editing_efficiency_by_biotype(strat_raw_dfs: dict,
                                        biotype_maps: dict,
                                        labels: dict,
                                        output_prefix: str):
    """
    For each library, show the distribution of per-read editing efficiency
    broken down by transcript biotype.

    Two panels per library:
      - CDF of editing efficiency per biotype (overlaid)
      - Boxplot of editing efficiency per biotype
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)

    # Extract read-level rows and attach all biotypes, exploding multi-biotype reads
    lib_dfs = {}
    all_biotypes = set()
    for key in ("bam1", "bam2"):
        raw = strat_raw_dfs[key]
        read_rows = (
            raw[raw["rel_pos"].isna()]
               .drop_duplicates(subset=["read_name"])
               [["read_name", "read_edit_eff"]]
               .dropna(subset=["read_edit_eff"])
        )
        # Map each read to its biotype(s) and explode
        bmap = biotype_maps[key]
        read_rows = read_rows.copy()
        read_rows["biotype_set"] = read_rows["read_name"].map(
            lambda rn: list(bmap.get(rn, {"unassigned"}))
        )
        exploded = read_rows.explode("biotype_set").rename(
            columns={"biotype_set": "biotype"}
        )
        lib_dfs[key] = exploded
        all_biotypes.update(exploded["biotype"].unique())

    # Order biotypes by median editing efficiency across both libraries
    med_eff = {}
    for bt in all_biotypes:
        vals = []
        for df in lib_dfs.values():
            sub = df[df["biotype"] == bt]["read_edit_eff"]
            if len(sub) > 0:
                vals.extend(sub.tolist())
        med_eff[bt] = float(np.median(vals)) if vals else 0
    biotype_order = sorted(all_biotypes, key=lambda bt: med_eff[bt], reverse=True)

    palette = sns.color_palette("tab20", n_colors=len(biotype_order))
    color_map = dict(zip(biotype_order, palette))
    c1, c2 = COLORS["bam1"], COLORS["bam2"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Per-read A→G editing efficiency by transcript biotype",
                 fontsize=13, fontweight="bold")

    for col, (key, label, lib_color) in enumerate([
        ("bam1", labels["bam1"], c1),
        ("bam2", labels["bam2"], c2),
    ]):
        df = lib_dfs[key]

        # ── Top: CDF per biotype ──────────────────────────────────────────────
        ax = axes[0, col]
        for bt in biotype_order:
            sub = np.sort(df[df["biotype"] == bt]["read_edit_eff"].values)
            if len(sub) == 0:
                continue
            cdf    = np.arange(1, len(sub) + 1) / len(sub)
            median = float(np.median(sub))
            ax.plot(sub, cdf, color=color_map[bt], lw=1.5,
                    label=f"{bt} (n={len(sub):,}, med={median:.3f})")
        ax.axhline(0.5, color="grey", lw=0.8, ls=":", alpha=0.6)
        ax.set_xlabel("Per-read A→G editing efficiency")
        ax.set_ylabel("Cumulative fraction of reads")
        ax.set_title(f"{label} — editing efficiency CDF by biotype")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend(fontsize=7, loc="lower right")

        # ── Bottom: boxplot per biotype ───────────────────────────────────────
        ax = axes[1, col]
        box_data   = []
        box_colors = []
        box_labels = []
        box_ns     = []
        for bt in biotype_order:
            sub = df[df["biotype"] == bt]["read_edit_eff"].values
            if len(sub) == 0:
                continue
            box_data.append(sub)
            box_colors.append(color_map[bt])
            box_labels.append(bt)
            box_ns.append(len(sub))

        if box_data:
            bp = ax.boxplot(box_data, patch_artist=True, notch=False,
                            medianprops=dict(color="black", lw=1.5),
                            flierprops=dict(marker=".", markersize=2,
                                            alpha=0.3, color="grey"))
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_xticks(range(1, len(box_labels) + 1))
            ax.set_xticklabels(
                [f"{bt}\n(n={n:,})" for bt, n in zip(box_labels, box_ns)],
                fontsize=7, rotation=30, ha="right"
            )
        ax.set_ylabel("Per-read A→G editing efficiency")
        ax.set_title(f"{label} — editing efficiency by biotype")
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plot_path = f"{output_prefix}_efficiency_by_biotype.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved editing efficiency by biotype → {plot_path}", file=sys.stderr)

    # Save summary table
    rows = []
    for key, label in labels.items():
        df = lib_dfs[key]
        for bt in biotype_order:
            sub = df[df["biotype"] == bt]["read_edit_eff"]
            if len(sub) == 0:
                continue
            rows.append({
                "library":  label,
                "biotype":  bt,
                "n_reads":  len(sub),
                "mean":     sub.mean(),
                "median":   sub.median(),
                "std":      sub.std(),
            })
    pd.DataFrame(rows).to_csv(
        f"{output_prefix}_efficiency_by_biotype.csv", index=False
    )



# ─────────────────────────────────────────────────────────────────────────────
# Control codon specificity analysis
# ─────────────────────────────────────────────────────────────────────────────

# Default control codons: CAA (Gln), ACA (Thr), TAT (Tyr)
# All have an A at codon position 1, matching the His edit target position
CONTROL_CODONS = {
    "CAA": {"CAA"},          # Gln — same first two bases as His
    "ACA": {"ACA"},          # Thr
    "TAT": {"TAT", "TAC"},   # Tyr — two codons, both have A at position 1
}


def plot_codon_specificity(his_agg_bam1: pd.DataFrame,
                            his_agg_bam2: pd.DataFrame,
                            control_aggs: dict,
                            label1: str,
                            label2: str,
                            output_prefix: str,
                            window: int):
    """
    Two-row figure showing meta-analysis editing profiles for His and each
    control codon, one column per codon. Row 0 = BAM1, Row 1 = BAM2.
    A separate panel overlays all codons for each BAM for direct comparison.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    c1, c2 = COLORS["bam1"], COLORS["bam2"]

    codon_names  = ["His"] + list(control_aggs.keys())
    n_codons     = len(codon_names)
    codon_colors = sns.color_palette("tab10", n_colors=n_codons)
    codon_color_map = dict(zip(codon_names, codon_colors))

    # ── Figure 1: per-BAM rows, per-codon columns ─────────────────────────────
    ncols = n_codons + 1   # one col per codon + one overlay col
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 9), sharey=False)
    fig.suptitle(
        "Codon specificity: His vs control codons\n"
        "(meta A→G editing fraction at position 1 of each codon)",
        fontsize=12, fontweight="bold"
    )

    for row, (bam_label, color, his_agg) in enumerate([
        (label1, c1, his_agg_bam1),
        (label2, c2, his_agg_bam2),
    ]):
        # Individual codon panels
        for col, codon_name in enumerate(codon_names):
            ax = axes[row, col]
            if codon_name == "His":
                agg = his_agg
            else:
                agg = control_aggs[codon_name].get(bam_label, pd.DataFrame())

            ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.5)
            ax.axvline(0, color="crimson", lw=1.5, ls="--", label="Position 1 (A)")
            ax.axvspan(-1, 1, alpha=0.08, color=codon_color_map[codon_name])

            if not agg.empty:
                n = int(agg["n_transcripts"].max())
                ax.plot(agg["rel_pos"], agg["mean_edit_frac"],
                        color=codon_color_map[codon_name], lw=2,
                        label=f"{codon_name} (n={n:,})")
                ax.fill_between(
                    agg["rel_pos"],
                    agg["mean_edit_frac"] - agg["sem_edit_frac"],
                    agg["mean_edit_frac"] + agg["sem_edit_frac"],
                    alpha=0.2, color=codon_color_map[codon_name],
                )
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)

            ax.set_xlim(-window, window)
            ax.set_xlabel("Position relative to codon A")
            ax.set_ylabel("Mean A→G edit fraction")
            ax.set_title(f"{bam_label} — {codon_name}")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
            ax.legend(fontsize=8)

        # Overlay panel (last column)
        ax = axes[row, -1]
        ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(0, color="crimson", lw=1.5, ls="--", label="Position 1 (A)")
        for codon_name in codon_names:
            if codon_name == "His":
                agg = his_agg
            else:
                agg = control_aggs[codon_name].get(bam_label, pd.DataFrame())
            if agg.empty:
                continue
            lw = 2.5 if codon_name == "His" else 1.5
            ax.plot(agg["rel_pos"], agg["mean_edit_frac"],
                    color=codon_color_map[codon_name], lw=lw,
                    label=codon_name,
                    ls="-" if codon_name == "His" else "--")
        ax.set_xlim(-window, window)
        ax.set_xlabel("Position relative to codon A")
        ax.set_ylabel("Mean A→G edit fraction")
        ax.set_title(f"{bam_label} — overlay")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = f"{output_prefix}_codon_specificity.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved codon specificity plots → {plot_path}", file=sys.stderr)


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
    p.add_argument("--gene_list", default=None,
                   help="Text file with one gene/transcript ID per line. "
                        "If provided, only His sites on these genes are analysed.")
    p.add_argument("--control_codons", nargs="*", default=None,
                   help="Control codon names to run specificity analysis "
                        f"(default: {list(CONTROL_CODONS.keys())}). "
                        "Must be keys of the CONTROL_CODONS dict in the script, "
                        "or pass none to skip.")
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

    # ── Gene list filter ──────────────────────────────────────────────────────
    if args.gene_list:
        with open(args.gene_list) as fh:
            allowed_genes = {line.strip() for line in fh if line.strip()}
        print(f"  Loaded {len(allowed_genes):,} genes from {args.gene_list}.",
              file=sys.stderr)
        # Keep CDS intervals whose gene_name is in the list
        filtered = {}
        for chrom, intervals in cds_by_chrom.items():
            kept = [(s, e, st, tid, gname) for s, e, st, tid, gname in intervals
                    if gname in allowed_genes]
            if kept:
                filtered[chrom] = kept
        n_before = sum(len(v) for v in cds_by_chrom.values())
        n_after  = sum(len(v) for v in filtered.values())
        print(f"  Filtered CDS intervals: {n_before:,} → {n_after:,} "
              f"({len(filtered):,} chromosomes retained).", file=sys.stderr)
        cds_by_chrom = filtered

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

    # Save merged per-site editing table — use pseudocount so sites with
    # zero editing in one condition still contribute rather than being dropped
    pseudo = 1e-3
    ha1 = s1["his_a_sites"][["site_id", "chrom", "edit_pos", "transcript",
                              "codon", "ag_edit_frac", "coverage"]].dropna()
    ha2 = s2["his_a_sites"][["site_id", "ag_edit_frac", "coverage"]].dropna()
    merged = ha1.merge(ha2, on="site_id",
                       suffixes=(f"_{args.label1}", f"_{args.label2}"))
    merged["log2fc"] = np.log2(
        (merged[f"ag_edit_frac_{args.label2}"] + pseudo) /
        (merged[f"ag_edit_frac_{args.label1}"] + pseudo)
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

    # ── Control codon specificity analysis ───────────────────────────────────
    control_codons_to_run = args.control_codons \
                            if args.control_codons is not None \
                            else list(CONTROL_CODONS.keys())

    if control_codons_to_run:
        print("\nRunning control codon specificity analysis…", file=sys.stderr)
        # control_aggs: codon_name → {label → rel_agg DataFrame}
        control_aggs = {}
        ref_fasta_ctrl = pysam.FastaFile(args.ref)

        for codon_name in control_codons_to_run:
            if codon_name not in CONTROL_CODONS:
                print(f"  WARNING: unknown control codon '{codon_name}', skipping.",
                      file=sys.stderr)
                continue
            print(f"  [{codon_name}] Finding sites…", file=sys.stderr)
            ctrl_sites = find_codon_positions(
                ref_fasta_ctrl, cds_by_chrom, args.window,
                target_codons=CONTROL_CODONS[codon_name],
                codon_label=codon_name,
            )
            control_aggs[codon_name] = {}
            for key, bam_path, label in [
                ("bam1", args.bam1, args.label1),
                ("bam2", args.bam2, args.label2),
            ]:
                print(f"  [{codon_name}] Pileup {label}…", file=sys.stderr)
                bam_ctrl = pysam.AlignmentFile(bam_path, "rb")
                ctrl_df  = aggregate_sites(
                    ctrl_sites, bam_ctrl, ref_fasta_ctrl,
                    min_coverage=args.min_coverage,
                    min_mapq=args.min_mapq,
                    min_baseq=args.min_baseq,
                    label=f"{label}/{codon_name}",
                )
                bam_ctrl.close()
                control_aggs[codon_name][label] = \
                    transcript_normalised_agg(ctrl_df) if not ctrl_df.empty \
                    else pd.DataFrame()

        ref_fasta_ctrl.close()

        print("\nGenerating codon specificity plots…", file=sys.stderr)
        plot_codon_specificity(
            his_agg_bam1=s1["rel_position_agg"],
            his_agg_bam2=s2["rel_position_agg"],
            control_aggs=control_aggs,
            label1=args.label1,
            label2=args.label2,
            output_prefix=out,
            window=args.window,
        )
    print("\nBuilding biotype index from GTF…", file=sys.stderr)
    biotype_idx = parse_gtf_biotypes(args.gtf)

    print("Assigning reads to biotypes…", file=sys.stderr)
    biotype_maps = {}
    for key, bam_path, label in [
        ("bam1", args.bam1, args.label1),
        ("bam2", args.bam2, args.label2),
    ]:
        print(f"  [{label}]…", file=sys.stderr)
        biotype_maps[key] = assign_read_biotypes(
            bam_path, biotype_idx, min_mapq=args.min_mapq
        )

    print("Generating biotype count plots…", file=sys.stderr)
    plot_biotype_counts(
        biotype_maps,
        labels={"bam1": args.label1, "bam2": args.label2},
        output_prefix=out,
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    # print("\nGenerating periodicity plots…", file=sys.stderr)
    bam1 = pysam.AlignmentFile(args.bam1, "rb")
    bam2 = pysam.AlignmentFile(args.bam2, "rb")
    ref_fasta = pysam.FastaFile(args.ref)
    # plot_periodicity(
    #     sites, bam1, bam2, ref_fasta,
    #     s1, s2, args.label1, args.label2,
    #     out, args.window, args.min_mapq, args.min_baseq,
    # )

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
        # Attach biotypes to strat_raw for downstream breakdowns
        strat_raw["biotypes"] = strat_raw["read_name"].map(
            lambda rn: ",".join(sorted(biotype_maps[key].get(rn, {"unassigned"})))
        )
        strat_dfs[key]     = compute_stratified_log2fc(strat_raw)
        strat_raw.to_csv(f"{out}_{key}_his_a_stratified.csv.gz",
                         index=False, compression="gzip")

    print("\nGenerating comparison plots…", file=sys.stderr)
    plot_comparison(s1, s2, args.label1, args.label2,
                    out, args.window, args.min_edit_fraction,
                    strat_raw_dfs=strat_raw_dfs)

    print("\nGenerating editing efficiency by biotype plots…", file=sys.stderr)
    plot_editing_efficiency_by_biotype(
        strat_raw_dfs, biotype_maps,
        labels={"bam1": args.label1, "bam2": args.label2},
        output_prefix=out,
    )

    print("\nGenerating His-A stratified plots…", file=sys.stderr)
    plot_his_a_stratified(
        strat_dfs,
        strat_raw_dfs=strat_raw_dfs,
        labels={"bam1": args.label1, "bam2": args.label2},
        output_prefix=out,
        window=args.window,
    )

    bam1.close()
    bam2.close()
    ref_fasta.close()

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()