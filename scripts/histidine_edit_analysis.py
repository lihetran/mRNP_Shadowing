#!/usr/bin/env python3
"""
Histidine Codon A-to-I (A→G) Editing Analysis for Nanopore Reads
=================================================================
Detects A-to-G mismatches in reads around histidine codons (CAU/CAC)
from a BAM/CRAM file aligned to a reference genome/transcriptome.

Usage:
    python histidine_edit_analysis.py \
        --bam reads.bam \
        --ref reference.fa \
        --gtf annotation.gtf \
        [--window 50] \
        [--min_coverage 10] \
        [--min_edit_fraction 0.01] \
        [--output results]

Requirements:
    pip install pysam biopython pandas matplotlib seaborn
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
HIS_CODONS = {"CAT", "CAC"}   # CAU → CAT in DNA; CAC stays CAC

# ── CIGAR operation codes that consume reference ────────────────────────────
REF_CONSUMING = {
    pysam.CMATCH,        # 0 M
    pysam.CDEL,          # 2 D
    pysam.CREF_SKIP,     # 3 N
    pysam.CEQUAL,        # 7 =
    pysam.CDIFF,         # 8 X
}
# Operations that consume the query (read) sequence
QUERY_CONSUMING = {
    pysam.CMATCH,
    pysam.CINS,          # 1 I
    pysam.CSOFT_CLIP,    # 4 S
    pysam.CEQUAL,
    pysam.CDIFF,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Parse GTF for CDS features → histidine codon positions
# ─────────────────────────────────────────────────────────────────────────────

def parse_gtf_cds(gtf_path: str) -> dict:
    """
    Return a dict:  chrom → sorted list of (start0, end0, strand, transcript_id)
    for every CDS interval in the GTF.
    """
    cds_by_chrom = collections.defaultdict(list)
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "CDS":
                continue
            chrom  = fields[0]
            start  = int(fields[3]) - 1   # GTF is 1-based inclusive → 0-based
            end    = int(fields[4])
            strand = fields[6]
            # extract transcript_id
            m = re.search(r'transcript_id "([^"]+)"', fields[8])
            tid = m.group(1) if m else "."
            cds_by_chrom[chrom].append((start, end, strand, tid))
    # sort each chromosome list
    for chrom in cds_by_chrom:
        cds_by_chrom[chrom].sort()
    return dict(cds_by_chrom)


def find_his_positions(ref_fasta: pysam.FastaFile,
                       cds_by_chrom: dict,
                       window: int) -> list:
    """
    Scan every CDS for histidine codons.
    Return list of dicts with site metadata + window coords.
    """
    sites = []
    for chrom, intervals in cds_by_chrom.items():
        try:
            chrom_len = ref_fasta.get_reference_length(chrom)
        except KeyError:
            continue

        # Merge overlapping CDS intervals to avoid double-counting
        # (simple scan; for a full tool you'd group by transcript)
        for (cds_start, cds_end, strand, tid) in intervals:
            cds_seq = ref_fasta.fetch(chrom, cds_start, cds_end).upper()
            cds_len = cds_end - cds_start
            # walk codons
            for i in range(0, cds_len - 2, 3):
                codon = cds_seq[i:i+3]
                if strand == "-":
                    codon = reverse_complement(codon)
                if codon in HIS_CODONS:
                    # genomic position of the codon's first base
                    if strand == "+":
                        codon_ref_start = cds_start + i
                    else:
                        # on minus strand codon order is reversed
                        codon_ref_start = cds_end - i - 3
                    # The A that gets edited is position 1 of CAT/CAC (0-indexed in codon)
                    # CAT: C=0, A=1, T=2 → edited A at position 1
                    if strand == "+":
                        edit_pos = codon_ref_start + 1   # 0-based genomic
                    else:
                        edit_pos = codon_ref_start + 1   # complement handled in mismatch check

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
    return sites


def reverse_complement(seq: str) -> str:
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Pileup-based mismatch counting
# ─────────────────────────────────────────────────────────────────────────────

def count_mismatches_at_site(bam: pysam.AlignmentFile,
                             ref_fasta: pysam.FastaFile,
                             site: dict,
                             min_mapq: int = 20,
                             min_baseq: int = 10) -> dict:
    """
    For each position in [win_start, win_end) around a His codon,
    count A, G, C, T, and total coverage from reads.
    Returns dict: rel_pos → {"A":n,"G":n,"C":n,"T":n,"cov":n,"ref":base}
    """
    chrom     = site["chrom"]
    edit_pos  = site["edit_pos"]
    win_start = site["win_start"]
    win_end   = site["win_end"]
    strand    = site["strand"]

    pos_data = {}

    for pcolumn in bam.pileup(
        chrom, win_start, win_end,
        truncate=True,
        min_mapping_quality=min_mapq,
        min_base_quality=min_baseq,
        stepper="samtools",
        ignore_overlaps=False,
    ):
        ref_pos = pcolumn.reference_pos          # 0-based
        if ref_pos < win_start or ref_pos >= win_end:
            continue

        ref_base = ref_fasta.fetch(chrom, ref_pos, ref_pos + 1).upper()
        rel_pos  = ref_pos - edit_pos             # 0 = the edited A

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
            "ref_pos": ref_pos,
            "ref_base": ref_base if strand == "+" else complement_base(ref_base),
            "A": counts.get("A", 0),
            "G": counts.get("G", 0),
            "C": counts.get("C", 0),
            "T": counts.get("T", 0),
            "cov": total,
        }
    return pos_data


def complement_base(b: str) -> str:
    return str.maketrans("ACGTacgt", "TGCAtgca").get(b, b) if len(b) == 1 \
        else b.translate(str.maketrans("ACGTacgt", "TGCAtgca"))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Aggregate across all sites
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_sites(sites: list,
                    bam: pysam.AlignmentFile,
                    ref_fasta: pysam.FastaFile,
                    min_coverage: int,
                    min_mapq: int,
                    min_baseq: int) -> pd.DataFrame:
    """
    Collect per-position mismatch data across all His sites.
    Returns a long-form DataFrame.
    """
    records = []
    for i, site in enumerate(sites):
        if (i + 1) % 100 == 0:
            print(f"  Processing site {i+1}/{len(sites)}…", file=sys.stderr)

        pos_data = count_mismatches_at_site(
            bam, ref_fasta, site,
            min_mapq=min_mapq, min_baseq=min_baseq
        )

        for rel_pos, counts in pos_data.items():
            if counts["cov"] < min_coverage:
                continue
            ag_denom = counts["A"] + counts["G"]
            ag_edit = counts["G"] / ag_denom if counts["ref_base"] == "A" and ag_denom > 0 else np.nan
            records.append({
                "site_id":    f"{site['chrom']}:{site['edit_pos']}",
                "transcript": site["transcript"],
                "chrom":      site["chrom"],
                "edit_pos":   site["edit_pos"],
                "strand":     site["strand"],
                "codon":      site["codon"],
                "rel_pos":    rel_pos,
                "ref_base":   counts["ref_base"],
                "A":          counts["A"],
                "G":          counts["G"],
                "C":          counts["C"],
                "T":          counts["T"],
                "coverage":   counts["cov"],
                "ag_edit_frac": ag_edit,
                # flag: is this the canonical His-A site?
                "is_his_A":   rel_pos == 0 and counts["ref_base"] == "A",
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_summaries(df: pd.DataFrame, min_edit_frac: float) -> dict:
    summaries = {}

    # 4a. Per-site editing fraction at position 0 (the His A)
    his_a_df = df[df["is_his_A"]].copy()
    summaries["his_a_sites"] = his_a_df

    # 4b. Meta-analysis: mean G/cov rate by relative position (all ref bases)
    # Using G/cov rather than ag_edit_frac so every position is comparable —
    # the His A at rel=0 should stand out above the background G error rate
    df["g_rate"] = df["G"] / df["coverage"]
    rel_agg = (
        df.groupby("rel_pos")
          .agg(
              mean_g_rate=("g_rate", "mean"),
              sem_g_rate=("g_rate", lambda x: x.sem()),
              n_sites=("site_id", "nunique"),
              total_cov=("coverage", "sum"),
          )
          .reset_index()
    )
    summaries["rel_position_agg"] = rel_agg

    # 4c. Distribution of editing fractions at the His A
    summaries["edit_frac_dist"] = his_a_df["ag_edit_frac"]

    # 4d. Edited sites (above threshold)
    edited = his_a_df[his_a_df["ag_edit_frac"] >= min_edit_frac].copy()
    edited = edited.sort_values("ag_edit_frac", ascending=False)
    summaries["edited_sites"] = edited

    return summaries


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(summaries: dict, output_prefix: str, window: int):
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("A→G Editing Around Histidine Codons (Nanopore)", fontsize=14, fontweight="bold")

    # ── Plot 1: Mean G/cov mismatch rate by relative position ────────────────
    ax = axes[0, 0]
    rel = summaries["rel_position_agg"]
    ax.axvline(0, color="crimson", lw=1.5, ls="--", label="His A (edit site)")
    ax.axvspan(-3, -1, alpha=0.08, color="steelblue", label="His codon (CAT/CAC)")
    ax.plot(rel["rel_pos"], rel["mean_g_rate"], color="steelblue", lw=2)
    ax.fill_between(
        rel["rel_pos"],
        rel["mean_g_rate"] - rel["sem_g_rate"],
        rel["mean_g_rate"] + rel["sem_g_rate"],
        alpha=0.25, color="steelblue"
    )
    ax.set_xlim(-window, window)
    ax.set_xlabel("Position relative to His codon A")
    ax.set_ylabel("Mean G/coverage rate")
    ax.set_title("Meta-analysis: G mismatch rate across window")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.legend(fontsize=9)

    # ── Plot 2: CDF of A→G fraction at the His A ─────────────────────────────
    ax = axes[0, 1]
    edit_fracs = summaries["edit_frac_dist"]
    if len(edit_fracs) > 0:
        sorted_fracs = np.sort(edit_fracs)
        cdf = np.arange(1, len(sorted_fracs) + 1) / len(sorted_fracs)
        ax.plot(sorted_fracs, cdf, color="steelblue", lw=2)
        median = edit_fracs.median()
        ax.axvline(median, color="crimson", ls="--",
                   label=f"Median = {median:.3f}")
        ax.axhline(0.5, color="crimson", ls=":", lw=0.8, alpha=0.6)
        ax.set_xlabel("A→G edit fraction at His A")
        ax.set_ylabel("Cumulative fraction of sites")
        ax.set_title("CDF of editing at His A")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    # ── Plot 3: Coverage distribution ────────────────────────────────────────
    ax = axes[1, 0]
    cov = summaries["his_a_sites"]["coverage"]
    if len(cov) > 0:
        ax.hist(np.log10(cov + 1), bins=40, color="mediumpurple", edgecolor="white")
        ax.set_xlabel("log10(Coverage + 1) at His A")
        ax.set_ylabel("Number of sites")
        ax.set_title("Coverage at His A positions")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    # ── Plot 4: Top edited sites bar chart ───────────────────────────────────
    ax = axes[1, 1]
    top = summaries["edited_sites"].head(20)
    if len(top) > 0:
        labels = top["site_id"].str.replace(r"(.{15}).*", r"\1…", regex=True)
        ax.barh(range(len(top)), top["ag_edit_frac"], color="coral", edgecolor="white")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("A→G edit fraction")
        ax.set_title("Top edited His sites")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    else:
        ax.text(0.5, 0.5, "No edited sites found", ha="center", va="center",
                transform=ax.transAxes)

    plt.tight_layout()
    plot_path = f"{output_prefix}_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plots → {plot_path}")
    return plot_path


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Meta-analysis of A→G editing around histidine codons in nanopore reads."
    )
    p.add_argument("--bam",  required=True, help="Sorted, indexed BAM/CRAM file")
    p.add_argument("--ref",  required=True, help="Reference FASTA (indexed with .fai)")
    p.add_argument("--gtf",  required=True, help="GTF annotation file")
    p.add_argument("--window", type=int, default=50,
                   help="Nucleotides either side of His A to analyse (default: 50)")
    p.add_argument("--min_coverage", type=int, default=10,
                   help="Minimum read depth to include a position (default: 10)")
    p.add_argument("--min_edit_fraction", type=float, default=0.01,
                   help="Min A→G fraction to call a site 'edited' (default: 0.01)")
    p.add_argument("--min_mapq", type=int, default=20,
                   help="Minimum mapping quality (default: 20)")
    p.add_argument("--min_baseq", type=int, default=10,
                   help="Minimum base quality (default: 10)")
    p.add_argument("--output", default="his_edit_results",
                   help="Output file prefix (default: his_edit_results)")
    p.add_argument("--chroms", nargs="*", default=None,
                   help="Restrict to specific chromosomes/contigs (space-separated)")
    return p.parse_args()


def main():
    args = parse_args()
    out  = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print("=== Histidine A→G Editing Meta-Analysis ===", file=sys.stderr)

    # ── Open files ────────────────────────────────────────────────────────────
    print("Opening BAM and reference…", file=sys.stderr)
    bam       = pysam.AlignmentFile(args.bam, "rb")
    ref_fasta = pysam.FastaFile(args.ref)

    # ── Find His codon sites ──────────────────────────────────────────────────
    print("Parsing GTF for CDS features…", file=sys.stderr)
    cds_by_chrom = parse_gtf_cds(args.gtf)
    if args.chroms:
        cds_by_chrom = {k: v for k, v in cds_by_chrom.items() if k in args.chroms}

    print("Scanning CDS for histidine codons…", file=sys.stderr)
    sites = find_his_positions(ref_fasta, cds_by_chrom, window=args.window)
    print(f"  Found {len(sites):,} His codon sites.", file=sys.stderr)

    if not sites:
        print("ERROR: No histidine sites found. Check chromosome names match between GTF and FASTA.",
              file=sys.stderr)
        sys.exit(1)

    # ── Pileup and aggregate ──────────────────────────────────────────────────
    print("Aggregating mismatch counts across sites…", file=sys.stderr)
    df = aggregate_sites(
        sites, bam, ref_fasta,
        min_coverage=args.min_coverage,
        min_mapq=args.min_mapq,
        min_baseq=args.min_baseq,
    )
    bam.close()
    ref_fasta.close()

    if df.empty:
        print("WARNING: No positions passed coverage filter. Try lowering --min_coverage.",
              file=sys.stderr)
        sys.exit(0)

    # ── Save raw data ─────────────────────────────────────────────────────────
    raw_csv = f"{out}_per_position.csv.gz"
    df.to_csv(raw_csv, index=False, compression="gzip")
    print(f"  Saved per-position data → {raw_csv}", file=sys.stderr)

    # ── Summaries ─────────────────────────────────────────────────────────────
    print("Computing summaries…", file=sys.stderr)
    summaries = compute_summaries(df, min_edit_frac=args.min_edit_fraction)

    # Save edited sites table
    edited_csv = f"{out}_edited_sites.csv"
    summaries["edited_sites"].to_csv(edited_csv, index=False)
    print(f"  Saved edited sites → {edited_csv}", file=sys.stderr)

    # Save meta-analysis aggregation
    agg_csv = f"{out}_meta_aggregation.csv"
    summaries["rel_position_agg"].to_csv(agg_csv, index=False)
    print(f"  Saved meta-aggregation → {agg_csv}", file=sys.stderr)

    # Print quick stats
    his_a = summaries["his_a_sites"]
    ed    = summaries["edited_sites"]
    print("\n── Summary ──────────────────────────────", file=sys.stderr)
    print(f"  His A sites with sufficient coverage : {len(his_a):,}", file=sys.stderr)
    print(f"  Sites with A→G frac ≥ {args.min_edit_fraction:.2%} : {len(ed):,}", file=sys.stderr)
    if len(his_a) > 0:
        print(f"  Median A→G frac at His A             : {his_a['ag_edit_frac'].median():.4f}",
              file=sys.stderr)
        print(f"  Mean   A→G frac at His A             : {his_a['ag_edit_frac'].mean():.4f}",
              file=sys.stderr)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("Generating plots…", file=sys.stderr)
    plot_results(summaries, out, window=args.window)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()