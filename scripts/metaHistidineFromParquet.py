'''
May 26, 2026 LT

Remake of histidine meta analysis plots from parquet files output by
shadowingBamToParquetWithGTF2.py. Computes transcript-normalised A->G
editing meta-plots around histidine codons and produces pyx PDF figures.

inputs:
- parquet directory 1
- parquet directory 2
- ref fasta
- GTF
- label 1
- label 2
- window size
- min coverage at site
- output prefix
'''

import argparse
import sys
import re
import collections
import math
from pathlib import Path

import pysam
import pandas as pd
import numpy as np


HIS_CODONS = {"CAT", "CAC"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Helpers
# ─────────────────────────────────────────────────────────────────────────────

def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]

def complement_base(b: str) -> str:
    return b.translate(str.maketrans("ACGTacgt", "TGCAtgca"))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Parquet loading
# ─────────────────────────────────────────────────────────────────────────────

def load_parquet_chunks(parquet_dir: str, gene: dict) -> pd.DataFrame:
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


def load_all_parquet_chunks(parquet_dir: str) -> pd.DataFrame:
    """Load all chunks from a parquet directory into one DataFrame."""
    parquet_dir = Path(parquet_dir)
    chunks = sorted(parquet_dir.glob("*.parquet"))
    if not chunks:
        return pd.DataFrame()
    dfs = [pd.read_parquet(c) for c in chunks]
    return pd.concat(dfs, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. GTF parsing
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
            m      = re.search(r'transcript_id "([^"]+)"', fields[8])
            tid    = m.group(1) if m else "."
            m2     = re.search(r'gene_name "([^"]+)"', fields[8])
            gname  = m2.group(1) if m2 else "."
            cds_by_chrom[chrom].append((start, end, strand, tid, gname))
    for chrom in cds_by_chrom:
        cds_by_chrom[chrom].sort()
    return dict(cds_by_chrom)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Site finding
# ─────────────────────────────────────────────────────────────────────────────

def find_codon_positions(ref_fasta: pysam.FastaFile,
                          cds_by_chrom: dict,
                          window: int,
                          target_codons: set,
                          codon_label: str = "codon") -> list:
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
                codon = cds_seq[i:i + 3]
                if strand == "-":
                    codon = reverse_complement(codon)
                if codon in target_codons:
                    edit_pos  = cds_start + i + 1
                    win_start = max(0, edit_pos - window)
                    win_end   = min(chrom_len, edit_pos + window + 1)
                    sites.append({
                        "chrom":       chrom,
                        "edit_pos":    edit_pos,
                        "codon_start": cds_start + i,
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


# ─────────────────────────────────────────────────────────────────────────────
# 5. Read collection helpers
# ─────────────────────────────────────────────────────────────────────────────

def collect_his_site_reads_from_dataframe(sites: list,
                                           dataframe: pd.DataFrame) -> set:
    """
    Returns set of read_ids whose alignment overlaps at least one His site window.
    Uses read_start/read_end columns for fast vectorised filtering per chrom.
    """
    read_names = set()
    sites_by_chrom = collections.defaultdict(list)
    for site in sites:
        sites_by_chrom[site["chrom"]].append(site)

    for chrom, chrom_sites in sites_by_chrom.items():
        chrom_df = dataframe[dataframe["chrom"] == chrom]
        if chrom_df.empty:
            continue
        for read in chrom_df.itertuples():
            # Use read_start/read_end if available, else fall back to absolute_indices
            if hasattr(read, "read_start") and hasattr(read, "read_end"):
                read_start = read.read_start
                read_end   = read.read_end
            else:
                mapped = [p for p in read.absolute_indices if p is not None]
                if not mapped:
                    continue
                read_start = mapped[0]
                read_end   = mapped[-1]

            for site in chrom_sites:
                if read_start < site["win_end"] and read_end > site["win_start"]:
                    read_names.add(read.read_id)
                    break

    return read_names


def compute_read_edit_efficiency_from_dataframe(
        dataframe: pd.DataFrame,
        restrict_to_reads: set = None) -> np.ndarray:
    """
    Compute per-read A->G editing efficiency from parquet data.
    Uses global_edit_freq column if available (computed at parquet creation time).
    Optionally restricted to reads in restrict_to_reads set.
    """
    df = dataframe
    if restrict_to_reads is not None:
        df = df[df["read_id"].isin(restrict_to_reads)]

    if df.empty:
        return np.array([])

    if "global_edit_freq" in df.columns:
        return df["global_edit_freq"].dropna().values
    return np.array([])


# ─────────────────────────────────────────────────────────────────────────────
# 6. Pileup and aggregation
# ─────────────────────────────────────────────────────────────────────────────

def count_mismatches_at_site(ref_fasta: pysam.FastaFile,
                              dataframe: pd.DataFrame,
                              site: dict) -> dict:
    """
    Parquet-compatible pileup. Pre-fetches window sequence once.
    Returns {rel_pos: {ref_pos, ref_base, A, G, C, T, cov}}.
    """
    chrom     = site["chrom"]
    edit_pos  = site["edit_pos"]
    win_start = site["win_start"]
    win_end   = site["win_end"]
    strand    = site["strand"]

    win_seq = ref_fasta.fetch(chrom, win_start, win_end).upper()
    counts_by_rel = collections.defaultdict(lambda: collections.Counter())

    for read in dataframe.itertuples():
        for i, (query_pos, ref_pos) in enumerate(read.aligned_pairs):
            if ref_pos is None or query_pos is None:
                continue
            if not (win_start <= ref_pos < win_end):
                continue

            rel_pos   = ref_pos - edit_pos
            if strand == "-":
                rel_pos = -rel_pos
            rel_pos = int(rel_pos)

            read_base = read.read_sequence_aligned[i].upper()
            if read_base == ' ':
                continue

            counts_by_rel[rel_pos][read_base] += 1

    pos_data = {}
    for rel_pos, counts in counts_by_rel.items():
        gpos = int(edit_pos + rel_pos) if strand == "+" else int(edit_pos - rel_pos)
        ref_base_genomic = win_seq[gpos - win_start]
        ref_base         = complement_base(ref_base_genomic) \
                           if strand == "-" else ref_base_genomic
        total = sum(counts.values())
        pos_data[rel_pos] = {
            "ref_pos":  gpos,
            "ref_base": ref_base,
            "A":        counts.get("A", 0),
            "G":        counts.get("G", 0),
            "C":        counts.get("C", 0),
            "T":        counts.get("T", 0),
            "cov":      total,
        }
    return pos_data


def aggregate_sites(sites: list,
                    dataframe: pd.DataFrame,
                    ref_fasta: pysam.FastaFile,
                    min_coverage: int,
                    label: str) -> pd.DataFrame:
    records = []
    for i, site in enumerate(sites):
        if (i + 1) % 100 == 0:
            print(f"  [{label}] Processing site {i + 1}/{len(sites)}…",
                  file=sys.stderr)

        # Vectorised pre-filter using read_start/read_end if available
        if "read_start" in dataframe.columns and "read_end" in dataframe.columns:
            site_df = dataframe[
                (dataframe["chrom"]      == site["chrom"]) &
                (dataframe["read_start"] <  site["win_end"]) &
                (dataframe["read_end"]   >  site["win_start"])
            ]
        else:
            site_df = dataframe[dataframe["chrom"] == site["chrom"]]

        if site_df.empty:
            continue

        pos_data = count_mismatches_at_site(ref_fasta, site_df, site)

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


# ─────────────────────────────────────────────────────────────────────────────
# 7. Aggregation and analysis
# ─────────────────────────────────────────────────────────────────────────────

def transcript_normalised_agg(df: pd.DataFrame,
                               group_cols: list = None,
                               pseudo: float = 1e-3) -> pd.DataFrame:
    """
    Two-stage transcript-normalised aggregation restricted to ref=A positions.
    Stage 1: mean (ag_edit_frac + pseudo) per (transcript, rel_pos)
    Stage 2: grand mean + SEM across transcripts at each rel_pos
    """
    if group_cols is None:
        group_cols = []

    ref_a = df[
        df["ag_edit_frac"].notna() &
        (df["ref_base"] == "A") &
        (~df["in_his_codon"] | df["is_his_A"])
    ].copy()
    ref_a["ag_edit_frac_ps"] = ref_a["ag_edit_frac"] + pseudo

    tx_mean = (
        ref_a.groupby(group_cols + ["transcript", "rel_pos"])["ag_edit_frac_ps"]
             .mean()
             .reset_index()
             .rename(columns={"ag_edit_frac_ps": "tx_mean_edit_frac"})
    )

    agg = (
        tx_mean.groupby(group_cols + ["rel_pos"])
               .agg(
                   mean_edit_frac=("tx_mean_edit_frac", "mean"),
                   sem_edit_frac =("tx_mean_edit_frac", lambda x: x.sem()),
                   n_transcripts =("transcript", "nunique"),
               )
               .reset_index()
    )
    return agg


def compute_log2fc_agg(df1: pd.DataFrame,
                        df2: pd.DataFrame,
                        pseudo: float = 1e-3) -> tuple:
    """
    Transcript-normalised log2FC between two aggregated site DataFrames.
    Returns (log2fc_agg, rank_log2fc_agg).
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
                   mean_log2fc  =("log2fc", "mean"),
                   sem_log2fc   =("log2fc", lambda x: x.sem()),
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


def compute_summaries(df: pd.DataFrame,
                       min_edit_frac: float = 0.01) -> dict:
    his_a_df = df[df["is_his_A"]].copy()
    rel_agg  = transcript_normalised_agg(df)

    rank_agg = {}
    for rank in [1, 2, 3]:
        sub = df[df["his_rank"] == rank]
        rank_agg[rank] = transcript_normalised_agg(sub) if not sub.empty \
                         else pd.DataFrame()

    codon_agg = {}
    for codon in ["CAT", "CAC"]:
        sub = df[df["codon"] == codon]
        codon_agg[codon] = transcript_normalised_agg(sub) if not sub.empty \
                           else pd.DataFrame()

    edited = his_a_df[his_a_df["ag_edit_frac"] >= min_edit_frac].copy()
    edited = edited.sort_values("ag_edit_frac", ascending=False)

    return {
        "his_a_sites":      his_a_df,
        "rel_position_agg": rel_agg,
        "rank_agg":         rank_agg,
        "codon_agg":        codon_agg,
        "edit_frac_dist":   his_a_df["ag_edit_frac"].dropna(),
        "edited_sites":     edited,
        "read_eff_dist":    np.array([]),   # filled after efficiency computation
        "log2fc_agg":       pd.DataFrame(), # filled after compute_log2fc_agg
        "rank_log2fc_agg":  {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. Pyx plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pyx_meta_graph(c, xpos, ypos, datasets, window,
                    y_title="Edit Frac",
                    x_title="Relative Position",
                    share_xaxis=None,
                    panel_w=5, panel_h=3):
    """
    Insert one meta-analysis panel into canvas c.
    datasets: list of (rel_agg DataFrame, pyx color, linestyle) tuples.
    All datasets plotted before c.insert() so overlay works correctly.
    """
    from pyx import graph, color, style

    y_max = 0.02
    for rel_agg, col, ls in datasets:
        if isinstance(rel_agg, pd.DataFrame) and rel_agg.empty:
            continue
        frac      = rel_agg["mean_edit_frac"].values
        sem       = rel_agg["sem_edit_frac"].values
        candidate = float(np.nanmax(frac + sem)) * 1.15 if len(frac) > 0 else 0
        y_max     = max(y_max, candidate)

    x_axis = graph.axis.linear(min=-window, max=window, title=x_title) \
             if share_xaxis is None \
             else graph.axis.linkedaxis(share_xaxis.axes["x"])

    g = graph.graphxy(
        width=panel_w, height=panel_h,
        xpos=xpos, ypos=ypos,
        x=x_axis,
        y=graph.axis.linear(min=0, max=y_max, title=y_title),
    )

    # Codon span markers and His A line
    g.plot(graph.data.function("x(y)=-1", min=0, max=y_max),
           [graph.style.line([color.gray(0.8), style.linewidth.thin])])
    g.plot(graph.data.function("x(y)=1", min=0, max=y_max),
           [graph.style.line([color.gray(0.8), style.linewidth.thin])])
    g.plot(graph.data.function("x(y)=0", min=0, max=y_max),
           [graph.style.line([color.cmyk(0, 1, 1, 0),
                              style.linewidth.thick,
                              style.linestyle.dashed])])

    for rel_agg, col, ls in datasets:
        if isinstance(rel_agg, pd.DataFrame) and rel_agg.empty:
            continue
        pos  = rel_agg["rel_pos"].values
        frac = rel_agg["mean_edit_frac"].values
        sem  = rel_agg["sem_edit_frac"].values

        for pts in [list(zip(pos.tolist(), (frac - sem).tolist())),
                    list(zip(pos.tolist(), (frac + sem).tolist()))]:
            g.plot(graph.data.points(pts, x=1, y=2),
                   [graph.style.line([col, style.linewidth.thin,
                                      style.linestyle.dotted])])
        g.plot(graph.data.points(list(zip(pos.tolist(), frac.tolist())), x=1, y=2),
               [graph.style.line([col, style.linewidth.normal, ls])])

    c.insert(g)
    return g


def _pyx_log2fc_graph(c, xpos, ypos, log2fc_agg, label1, label2, window,
                       col1, col2, share_xaxis=None, panel_w=5, panel_h=3):
    """Insert a log2FC panel into canvas c."""
    from pyx import graph, color, style

    if isinstance(log2fc_agg, pd.DataFrame) and log2fc_agg.empty:
        return None

    pos    = log2fc_agg["rel_pos"].values
    fc     = log2fc_agg["mean_log2fc"].values
    sem_fc = log2fc_agg["sem_log2fc"].values

    y_abs        = max(np.nanmax(np.abs(fc)), 0.5) * 1.15
    y_min, y_max = -y_abs, y_abs

    x_axis = graph.axis.linear(min=-window, max=window,
                                title="Relative Position") \
             if share_xaxis is None \
             else graph.axis.linkedaxis(share_xaxis.axes["x"])

    g = graph.graphxy(
        width=panel_w, height=panel_h,
        xpos=xpos, ypos=ypos,
        x=x_axis,
        y=graph.axis.linear(min=y_min, max=y_max,
                            title=f"log2FC ({label2}/{label1})"),
    )

    g.plot(graph.data.function("y(x)=0", min=-window, max=window),
           [graph.style.line([color.cmyk(0, 0, 0, 1), style.linewidth.thin,
                              style.linestyle.dashed])])
    g.plot(graph.data.function("x(y)=0", min=y_min, max=y_max),
           [graph.style.line([color.cmyk(0, 1, 1, 0),
                              style.linewidth.thick,
                              style.linestyle.dashed])])

    for pts in [list(zip(pos.tolist(), (fc - sem_fc).tolist())),
                list(zip(pos.tolist(), (fc + sem_fc).tolist()))]:
        g.plot(graph.data.points(pts, x=1, y=2),
               [graph.style.line([color.gray(0.5), style.linewidth.thin,
                                  style.linestyle.dotted])])

    g.plot(graph.data.points(list(zip(pos.tolist(), fc.tolist())), x=1, y=2),
           [graph.style.line([color.cmyk(0, 0, 0, 1), style.linewidth.normal,
                              style.linestyle.solid])])

    c.insert(g)
    return g


def _pyx_cdf_graph(c, xpos, ypos, s1, s2, label1, label2,
                   col1, col2, panel_w=5, panel_h=3):
    """CDF of His A edit frac (solid) and per-read efficiency (dashed)."""
    from pyx import graph, style

    g = graph.graphxy(
        width=panel_w, height=panel_h,
        xpos=xpos, ypos=ypos,
        x=graph.axis.linear(min=0, max=1, title="A->G edit frac"),
        y=graph.axis.linear(min=0, max=1, title="Cumulative fraction"),
    )

    for s, col in [(s1, col1), (s2, col2)]:
        fracs = s["edit_frac_dist"]
        if len(fracs) > 0:
            sf  = np.sort(fracs.values if hasattr(fracs, "values") else fracs)
            cdf = np.arange(1, len(sf) + 1) / len(sf)
            g.plot(graph.data.points(list(zip(sf.tolist(), cdf.tolist())),
                                     x=1, y=2),
                   [graph.style.line([col, style.linewidth.normal,
                                      style.linestyle.solid])])

        eff = s.get("read_eff_dist", np.array([]))
        if len(eff) > 0:
            eff_s = np.sort(eff)
            cdf_e = np.arange(1, len(eff_s) + 1) / len(eff_s)
            g.plot(graph.data.points(list(zip(eff_s.tolist(), cdf_e.tolist())),
                                     x=1, y=2),
                   [graph.style.line([col, style.linewidth.normal,
                                      style.linestyle.dashed])])

    c.insert(g)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# 9. Top-level pyx plot functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_pyx(s1: dict, s2: dict,
                         label1: str, label2: str,
                         output_prefix: str, window: int):
    """2x2 comparison figure: meta BAM1, meta BAM2, log2FC, CDF."""
    from pyx import canvas, color, style, text as pyx_text

    col1    = color.cmyk(0, 0, 0, 1)
    col2    = color.cmyk(1, 0.5, 0, 0)
    panel_w = 5
    panel_h = 3
    gap     = 2.0

    c = canvas.canvas()

    g1 = _pyx_meta_graph(c, xpos=0, ypos=panel_h + gap,
                          datasets=[(s1["rel_position_agg"], col1,
                                     style.linestyle.solid)],
                          y_title="Edit Frac",
                          window=window, panel_w=panel_w, panel_h=panel_h)
    c.text(g1.xpos + g1.width / 2., g1.ypos + g1.height + 0.4, label1,
           [pyx_text.halign.center, pyx_text.size.normalsize])

    g2 = _pyx_meta_graph(c, xpos=panel_w + gap, ypos=panel_h + gap,
                          datasets=[(s2["rel_position_agg"], col2,
                                     style.linestyle.solid)],
                          y_title="Edit Frac",
                          window=window, panel_w=panel_w, panel_h=panel_h)
    c.text(g2.xpos + g2.width / 2., g2.ypos + g2.height + 0.4, label2,
           [pyx_text.halign.center, pyx_text.size.normalsize])

    _pyx_log2fc_graph(c, xpos=0, ypos=0,
                      log2fc_agg=s1["log2fc_agg"],
                      label1=label1, label2=label2,
                      window=window, col1=col1, col2=col2,
                      panel_w=panel_w, panel_h=panel_h)

    _pyx_cdf_graph(c, xpos=panel_w + gap, ypos=0,
                   s1=s1, s2=s2, label1=label1, label2=label2,
                   col1=col1, col2=col2, panel_w=panel_w, panel_h=panel_h)

    plot_path = f"{output_prefix}_comparison_pyx"
    c.writePDFfile(plot_path)
    print(f"  Saved -> {plot_path}.pdf", file=sys.stderr)


def plot_rank_comparison_pyx(s1: dict, s2: dict,
                              label1: str, label2: str,
                              output_prefix: str, window: int):
    """3-column rank figure: 1st/2nd/3rd His codon, meta overlay + log2FC."""
    from pyx import canvas, color, style, text as pyx_text

    col1    = color.cmyk(0, 0, 0, 1)
    col2    = color.cmyk(1, 0.5, 0, 0)
    panel_w = 4
    panel_h = 3
    gap     = 2.0

    c = canvas.canvas()

    for col_idx, rank in enumerate([1, 2, 3]):
        xpos        = col_idx * (panel_w + gap)
        r1          = s1["rank_agg"].get(rank, pd.DataFrame())
        r2          = s2["rank_agg"].get(rank, pd.DataFrame())
        rank_labels = {1: "1st", 2: "2nd", 3: "3rd"}
        y_title     = "Edit Frac" if col_idx == 0 else ""

        g_top = _pyx_meta_graph(
            c, xpos=xpos, ypos=panel_h + gap,
            datasets=[
                (r1, col1, style.linestyle.solid),
                (r2, col2, style.linestyle.solid),
            ],
            y_title=y_title,
            window=window, panel_w=panel_w, panel_h=panel_h,
        )
        c.text(g_top.xpos + g_top.width / 2., g_top.ypos + g_top.height + 0.4,
               f"{rank_labels[rank]} His codon",
               [pyx_text.halign.center, pyx_text.size.small])

        rank_fc = s1["rank_log2fc_agg"].get(rank, pd.DataFrame())
        _pyx_log2fc_graph(c, xpos=xpos, ypos=0,
                           log2fc_agg=rank_fc,
                           label1=label1, label2=label2,
                           window=window, col1=col1, col2=col2,
                           share_xaxis=g_top,
                           panel_w=panel_w, panel_h=panel_h)

    plot_path = f"{output_prefix}_rank_pyx"
    c.writePDFfile(plot_path)
    print(f"  Saved -> {plot_path}.pdf", file=sys.stderr)


def plot_codon_type_comparison_pyx(s1: dict, s2: dict,
                                    label1: str, label2: str,
                                    output_prefix: str, window: int):
    """CAT vs CAC: 2 rows x 3 cols (CAT, CAC, overlay)."""
    from pyx import canvas, color, style, path, text as pyx_text

    col_cat = color.cmyk(0, 0, 0, 1)
    col_cac = color.cmyk(1, 0.5, 0, 0)
    panel_w = 5
    panel_h = 3
    gap     = 2.0
    row_gap = 2.0

    c = canvas.canvas()

    for row_idx, (bam_label, s) in enumerate([(label1, s1), (label2, s2)]):
        ypos    = (1 - row_idx) * (panel_h + row_gap)
        x_title = "Relative Position" if row_idx == 1 else ""

        for col_idx, (codon, col) in enumerate([("CAT", col_cat), ("CAC", col_cac)]):
            xpos    = col_idx * (panel_w + gap)
            agg     = s["codon_agg"].get(codon, pd.DataFrame())
            y_title = "Edit Frac" if col_idx == 0 else ""
            g = _pyx_meta_graph(
                c, xpos=xpos, ypos=ypos,
                datasets=[(agg, col, style.linestyle.solid)],
                y_title=y_title, x_title=x_title,
                window=window, panel_w=panel_w, panel_h=panel_h,
            )
            c.text(g.xpos + g.width / 2., g.ypos + g.height + 0.3, codon,
                   [pyx_text.halign.center, pyx_text.size.small])

        xpos_ov = 2 * (panel_w + gap)
        g_ov = _pyx_meta_graph(
            c, xpos=xpos_ov, ypos=ypos,
            datasets=[
                (s["codon_agg"].get("CAT", pd.DataFrame()),
                 col_cat, style.linestyle.solid),
                (s["codon_agg"].get("CAC", pd.DataFrame()),
                 col_cac, style.linestyle.dashed),
            ],
            y_title="", x_title=x_title,
            window=window, panel_w=panel_w, panel_h=panel_h,
        )
        c.text(g_ov.xpos + g_ov.width / 2., g_ov.ypos + g_ov.height + 0.3,
               "CAT vs CAC",
               [pyx_text.halign.center, pyx_text.size.small])
        c.text(-0.2, ypos + panel_h / 2., bam_label,
               [pyx_text.halign.boxright, pyx_text.valign.middle,
                pyx_text.size.small])

    # Legend
    leg_x   = 2 * (panel_w + gap) + panel_w + 0.4
    leg_lw  = 0.8
    leg_dy  = 0.55
    top_y   = panel_h + row_gap
    for j, (codon, col, ls) in enumerate([
        ("CAT", col_cat, style.linestyle.solid),
        ("CAC", col_cac, style.linestyle.dashed),
    ]):
        ly = top_y + panel_h - 0.3 - j * leg_dy
        c.stroke(path.line(leg_x, ly, leg_x + leg_lw, ly),
                 [col, style.linewidth.normal, ls])
        c.text(leg_x + leg_lw + 0.15, ly, codon,
               [pyx_text.valign.middle, pyx_text.size.small])

    plot_path = f"{output_prefix}_codon_type_pyx"
    c.writePDFfile(plot_path)
    print(f"  Saved -> {plot_path}.pdf", file=sys.stderr)


def plot_codon_specificity_overlay_pyx(his_agg_bam1: pd.DataFrame,
                                        his_agg_bam2: pd.DataFrame,
                                        control_aggs: dict,
                                        label1: str, label2: str,
                                        output_prefix: str, window: int):
    """Overlay of His + control codons, one panel per BAM."""
    from pyx import canvas, color, style, path, text as pyx_text

    codon_names = ["His"] + list(control_aggs.keys())
    codon_colors_cmyk = [
        color.cmyk(0, 1, 1, 0),
        color.cmyk(1, 0.5, 0, 0),
        color.cmyk(0.1, 0.05, 0.9, 0),
        color.cmyk(0.97, 0, 0.75, 0),
        color.cmyk(0, 0.6, 0.3, 0),
    ]
    codon_color_map = {name: codon_colors_cmyk[i]
                       for i, name in enumerate(codon_names)}

    panel_w = 7
    panel_h = 3.5
    gap     = 1.5
    leg_lw  = 0.8
    leg_dy  = 0.55

    c = canvas.canvas()

    for row_idx, (bam_label, his_agg) in enumerate([
        (label1, his_agg_bam1),
        (label2, his_agg_bam2),
    ]):
        ypos    = (1 - row_idx) * (panel_h + gap)
        x_title = "Relative Position" if row_idx == 1 else ""

        overlay_datasets = []
        for codon_name in codon_names:
            agg = his_agg if codon_name == "His" \
                  else control_aggs[codon_name].get(bam_label, pd.DataFrame())
            ls  = style.linestyle.solid if codon_name == "His" \
                  else style.linestyle.dashed
            overlay_datasets.append((agg, codon_color_map[codon_name], ls))

        g = _pyx_meta_graph(
            c, xpos=0, ypos=ypos,
            datasets=overlay_datasets,
            y_title="Edit Frac", x_title=x_title,
            window=window, panel_w=panel_w, panel_h=panel_h,
        )
        c.text(g.xpos + g.width / 2., g.ypos + g.height + 0.4, bam_label,
               [pyx_text.halign.center, pyx_text.size.normalsize])

        if row_idx == 0:
            leg_x_start = g.xpos + g.width + 0.4
            leg_y_start = g.ypos + g.height - 0.2
            for j, codon_name in enumerate(codon_names):
                col = codon_color_map[codon_name]
                ls  = style.linestyle.solid if codon_name == "His" \
                      else style.linestyle.dashed
                ly  = leg_y_start - j * leg_dy
                c.stroke(path.line(leg_x_start, ly, leg_x_start + leg_lw, ly),
                         [col, style.linewidth.normal, ls])
                c.text(leg_x_start + leg_lw + 0.15, ly, codon_name,
                       [pyx_text.valign.middle, pyx_text.size.small])

    plot_path = f"{output_prefix}_codon_overlay_pyx"
    c.writePDFfile(plot_path)
    print(f"  Saved -> {plot_path}.pdf", file=sys.stderr)


CONTROL_CODONS = {
    "CAA": {"CAA"},
    "ACA": {"ACA"},
    "TAT": {"TAT", "TAC"},
}


def plot_codon_specificity_pyx(his_agg_bam1: pd.DataFrame,
                                his_agg_bam2: pd.DataFrame,
                                control_aggs: dict,
                                label1: str, label2: str,
                                output_prefix: str, window: int):
    """
    2 rows (BAM1 top, BAM2 bottom) x (n_codons + 1) columns.
    Individual codon panels + overlay column.
    y-title on leftmost column only, x-title on bottom row only.
    Codon name titles above each panel.
    """
    from pyx import canvas, color, style, text as pyx_text

    codon_names = ["His"] + list(control_aggs.keys())
    codon_colors_cmyk = [
        color.cmyk(0, 1, 1, 0),
        color.cmyk(1, 0.5, 0, 0),
        color.cmyk(0.1, 0.05, 0.9, 0),
        color.cmyk(0.97, 0, 0.75, 0),
        color.cmyk(0, 0.6, 0.3, 0),
    ]
    codon_color_map = {name: codon_colors_cmyk[i]
                       for i, name in enumerate(codon_names)}

    panel_w = 3
    panel_h = 2.5
    gap     = 2.5
    row_gap = 2.0

    c = canvas.canvas()

    for row_idx, (bam_label, his_agg) in enumerate([
        (label1, his_agg_bam1),
        (label2, his_agg_bam2),
    ]):
        ypos    = (1 - row_idx) * (panel_h + row_gap)
        x_title = "Relative Position" if row_idx == 1 else ""

        for col_idx, codon_name in enumerate(codon_names):
            xpos    = col_idx * (panel_w + gap)
            agg     = his_agg if codon_name == "His" \
                      else control_aggs[codon_name].get(bam_label, pd.DataFrame())
            col     = codon_color_map[codon_name]
            y_title = f"{bam_label} edit frac" if col_idx == 0 else ""

            g = _pyx_meta_graph(
                c, xpos=xpos, ypos=ypos,
                datasets=[(agg, col, style.linestyle.solid)],
                y_title=y_title, x_title=x_title,
                window=window, panel_w=panel_w, panel_h=panel_h,
            )
            c.text(g.xpos + g.width / 2., g.ypos + g.height + 0.3,
                   codon_name,
                   [pyx_text.halign.center, pyx_text.size.small])

        # Overlay column
        overlay_datasets = []
        for codon_name in codon_names:
            agg = his_agg if codon_name == "His" \
                  else control_aggs[codon_name].get(bam_label, pd.DataFrame())
            ls  = style.linestyle.solid if codon_name == "His" \
                  else style.linestyle.dashed
            overlay_datasets.append((agg, codon_color_map[codon_name], ls))

        xpos_overlay = len(codon_names) * (panel_w + gap)
        g_ov = _pyx_meta_graph(
            c, xpos=xpos_overlay, ypos=ypos,
            datasets=overlay_datasets,
            y_title="", x_title=x_title,
            window=window, panel_w=panel_w, panel_h=panel_h,
        )
        c.text(g_ov.xpos + g_ov.width / 2., g_ov.ypos + g_ov.height + 0.3,
               "Overlay",
               [pyx_text.halign.center, pyx_text.size.small])

    plot_path = f"{output_prefix}_codon_specificity_pyx"
    c.writePDFfile(plot_path)
    print(f"  Saved -> {plot_path}.pdf", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# 10. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Histidine meta analysis from parquet files."
    )
    p.add_argument("--parquet1",     required=True,
                   help="Parquet directory for library 1 (reference)")
    p.add_argument("--parquet2",     required=True,
                   help="Parquet directory for library 2 (query)")
    p.add_argument("--label1",       default="BAM1")
    p.add_argument("--label2",       default="BAM2")
    p.add_argument("--ref",          required=True)
    p.add_argument("--gtf",          required=True)
    p.add_argument("--output",       default="his_meta")
    p.add_argument("--window",       type=int, default=50)
    p.add_argument("--min_coverage", type=int, default=10)
    p.add_argument("--min_edit_frac", type=float, default=0.01)
    p.add_argument("--control_codons", nargs="*", default=None,
                   help="Control codon names for specificity analysis. "
                        f"Default: {list(CONTROL_CODONS.keys())}. "
                        "Pass none to skip.")
    return p.parse_args()


def main():
    args = parse_args()
    out  = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print("=== Histidine Meta Analysis (parquet) ===", file=sys.stderr)

    # ── Parse GTF and find His sites ──────────────────────────────────────────
    print("\nParsing GTF...", file=sys.stderr)
    ref_fasta    = pysam.FastaFile(args.ref)
    cds_by_chrom = parse_gtf_cds(args.gtf)
    sites        = find_his_positions(ref_fasta, cds_by_chrom, args.window)

    # ── Load parquets ─────────────────────────────────────────────────────────
    print("\nLoading parquet chunks...", file=sys.stderr)
    df_all1 = load_all_parquet_chunks(args.parquet1)
    df_all2 = load_all_parquet_chunks(args.parquet2)
    print(f"  {args.label1}: {len(df_all1):,} reads", file=sys.stderr)
    print(f"  {args.label2}: {len(df_all2):,} reads", file=sys.stderr)

    # ── Collect reads spanning His sites (for efficiency CDF) ─────────────────
    print("\nCollecting reads overlapping His sites...", file=sys.stderr)
    his_reads1 = collect_his_site_reads_from_dataframe(sites, df_all1)
    his_reads2 = collect_his_site_reads_from_dataframe(sites, df_all2)
    print(f"  {args.label1}: {len(his_reads1):,} reads", file=sys.stderr)
    print(f"  {args.label2}: {len(his_reads2):,} reads", file=sys.stderr)

    # ── Aggregate sites ───────────────────────────────────────────────────────
    print("\nAggregating sites...", file=sys.stderr)
    agg_df1 = aggregate_sites(sites, df_all1, ref_fasta,
                               args.min_coverage, args.label1)
    agg_df2 = aggregate_sites(sites, df_all2, ref_fasta,
                               args.min_coverage, args.label2)

    # ── Compute summaries ─────────────────────────────────────────────────────
    print("\nComputing summaries...", file=sys.stderr)
    s1 = compute_summaries(agg_df1, args.min_edit_frac)
    s2 = compute_summaries(agg_df2, args.min_edit_frac)

    # Fill read efficiency distributions
    s1["read_eff_dist"] = compute_read_edit_efficiency_from_dataframe(
        df_all1, restrict_to_reads=his_reads1)
    s2["read_eff_dist"] = compute_read_edit_efficiency_from_dataframe(
        df_all2, restrict_to_reads=his_reads2)

    # Fill log2FC
    log2fc_agg, rank_log2fc_agg = compute_log2fc_agg(agg_df1, agg_df2)
    s1["log2fc_agg"]      = log2fc_agg
    s1["rank_log2fc_agg"] = rank_log2fc_agg
    s2["log2fc_agg"]      = log2fc_agg   # same — it's between the two libraries
    s2["rank_log2fc_agg"] = rank_log2fc_agg

    # ── Save aggregated data ──────────────────────────────────────────────────
    agg_df1.to_csv(f"{out}_{args.label1}_agg.csv.gz", index=False,
                   compression="gzip")
    agg_df2.to_csv(f"{out}_{args.label2}_agg.csv.gz", index=False,
                   compression="gzip")

    # ── Control codon specificity ─────────────────────────────────────────────
    control_codons_to_run = args.control_codons \
                            if args.control_codons is not None \
                            else list(CONTROL_CODONS.keys())

    control_aggs = {}
    if control_codons_to_run:
        print("\nRunning control codon specificity analysis...", file=sys.stderr)
        for codon_name in control_codons_to_run:
            if codon_name not in CONTROL_CODONS:
                print(f"  WARNING: unknown control codon '{codon_name}', skipping.",
                      file=sys.stderr)
                continue
            print(f"  [{codon_name}] Finding sites...", file=sys.stderr)
            ctrl_sites = find_codon_positions(
                ref_fasta, cds_by_chrom, args.window,
                target_codons=CONTROL_CODONS[codon_name],
                codon_label=codon_name,
            )
            control_aggs[codon_name] = {}
            for df_ctrl, label in [(df_all1, args.label1),
                                    (df_all2, args.label2)]:
                print(f"  [{codon_name}] Aggregating {label}...", file=sys.stderr)
                ctrl_df = aggregate_sites(ctrl_sites, df_ctrl, ref_fasta,
                                           args.min_coverage, label)
                control_aggs[codon_name][label] = \
                    transcript_normalised_agg(ctrl_df) if not ctrl_df.empty \
                    else pd.DataFrame()

    # ── Plot ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...", file=sys.stderr)
    try:
        plot_comparison_pyx(s1, s2, args.label1, args.label2,
                            out, args.window)
        plot_rank_comparison_pyx(s1, s2, args.label1, args.label2,
                                  out, args.window)
        plot_codon_type_comparison_pyx(s1, s2, args.label1, args.label2,
                                        out, args.window)
        if control_aggs:
            plot_codon_specificity_pyx(
                his_agg_bam1=s1["rel_position_agg"],
                his_agg_bam2=s2["rel_position_agg"],
                control_aggs=control_aggs,
                label1=args.label1, label2=args.label2,
                output_prefix=out, window=args.window,
            )
            plot_codon_specificity_overlay_pyx(
                his_agg_bam1=s1["rel_position_agg"],
                his_agg_bam2=s2["rel_position_agg"],
                control_aggs=control_aggs,
                label1=args.label1, label2=args.label2,
                output_prefix=out, window=args.window,
            )
    except Exception as e:
        print(f"  WARNING: pyx plotting failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    ref_fasta.close()
    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()