'''
August 10, 2026 LT

Optimized variant of metaHistidineFromParquet.py -- same His-codon
transcript-normalised A->G editing meta-analysis, same pyx PDF figures,
same CLI, but with three optimizations applied:

  1. Parquet column projection: parquet chunks are read with columns=
     (intersected with whatever the chunk's own schema actually has, so
     it degrades gracefully on older parquet files missing e.g.
     read_start/read_end) instead of pulling in every column
     shadowingBamToParquetWithGTF2.py wrote -- this script only ever
     touches chrom/read_id/read_start/read_end/edit_string/
     absolute_indices/global_edit_freq/is_reverse, so the rest
     (ref_sequence_aligned, shadow_*, n_a_positions, ...) was pure
     load/concat overhead.

  2. Single combined CDS walk for His + every control codon:
     find_codon_positions previously re-fetched/re-walked the SAME CDS
     sequence once per codon label (His, then separately CAA, ACA, TAT
     -- 4 total passes over every gene's CDS). find_multi_codon_positions
     replaces it with one fetch+walk per gene that classifies each codon
     against a single label lookup built from all requested codon
     groups at once, producing the exact same per-label deduped site
     lists find_codon_positions would have (dedup/his_rank are computed
     per-label after the shared walk, so semantics are unchanged) -- but
     without paying for the CDS scan more than once.

  3. Streaming per-chunk aggregation instead of one giant in-memory
     library DataFrame: the original script's load_all_parquet_chunks
     pd.concat'd an entire parquet directory into one DataFrame that
     stayed resident for the whole run (this is what was getting OOM
     killed on large libraries). process_library_streaming reads one
     parquet chunk at a time, tallies every site's (His AND every
     control codon at once, same combined-pass idea as optimization 2)
     A/G/C/T counts into a small per-site accumulator, and discards the
     chunk before moving to the next one -- peak memory is bounded by
     one chunk's worth of read-level data plus the accumulators (which
     hold only aggregate counts, not per-read data), not the whole
     library. This also folds what used to be three separate full
     passes (collect_his_site_reads_from_dataframe, then aggregate_sites
     once per codon label) into that same single streaming pass per
     library, so each parquet chunk is only read off disk once.

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
from logJosh import Tee


HIS_CODONS = {"CAT", "CAC"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Helpers
# ─────────────────────────────────────────────────────────────────────────────

def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]

def complement_base(b: str) -> str:
    return b.translate(str.maketrans("ACGTacgt", "TGCAtgca"))


_TEX_SPECIAL = {
    '\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
    '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
}

def tex_escape(s):
    """Escape characters (e.g. '_') that PyX's TeX text engine treats as
    special, so library labels (--label1/--label2, manuscript sample names
    like '+3AT_rep1') render literally instead of erroring."""
    return ''.join(_TEX_SPECIAL.get(ch, ch) for ch in str(s))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Parquet loading
# ─────────────────────────────────────────────────────────────────────────────

# Every column this script actually reads out of a shadowingBamToParquetWithGTF2.py
# chunk -- anything else in the schema (ref_sequence_aligned, shadow_*,
# n_a_positions, gene_biotype, ...) is loaded/concatenated for nothing.
REQUIRED_COLUMNS = [
    "chrom", "read_id", "read_start", "read_end",
    "edit_string", "absolute_indices", "global_edit_freq", "is_reverse",
]


def _select_available_columns(parquet_path: Path, wanted: list) -> list:
    """
    Intersect `wanted` with parquet_path's own schema, so a caller asking
    for e.g. read_start/read_end/is_reverse doesn't blow up on older
    chunks that predate those columns -- pd.read_parquet(columns=...)
    raises if any requested column is absent from the file.

    Uses schema_arrow (the Arrow logical schema pandas' own column names
    come from), NOT the raw Parquet-level schema -- list-typed columns
    (e.g. absolute_indices) are stored as a nested "item" field at the
    Parquet physical-schema level, so plain .schema.names reports
    "item" for every such column (even producing duplicate "item"
    entries) instead of its real pandas column name, which would make
    this silently drop columns it should have kept.
    """
    import pyarrow.parquet as pq
    schema_names = set(pq.ParquetFile(parquet_path).schema_arrow.names)
    missing = [c for c in wanted if c not in schema_names]
    if missing:
        print(f"  NOTE: {parquet_path.name} schema is missing {missing}; "
              f"reading without them.", file=sys.stderr)
    return [c for c in wanted if c in schema_names]


def list_parquet_chunks(parquet_dir: str) -> list:
    return sorted(Path(parquet_dir).glob("*.parquet"))


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

def find_multi_codon_positions(ref_fasta: pysam.FastaFile,
                                cds_by_chrom: dict,
                                window: int,
                                codon_groups: dict) -> dict:
    """
    codon_groups: {label: set_of_codons}, e.g. {"His": {"CAT","CAC"},
    "CAA": {"CAA"}, ...} -- codon sets must be mutually disjoint (true for
    His vs. CAA/ACA/TAT). Does ONE fetch+walk per CDS interval (instead of
    one walk per label) by classifying each codon against a single
    codon->label lookup built from every group at once.

    Returns {label: deduped_sites}, each list identical to what a
    standalone find_codon_positions(..., target_codons=codon_groups[label],
    codon_label=label) call would have produced -- his_rank and the
    (chrom, edit_pos) dedup are both computed per-label, after the shared
    walk, so per-label semantics are unchanged from the one-label-at-a-time
    version.
    """
    codon_to_label = {}
    for label, codons in codon_groups.items():
        for codon in codons:
            codon_to_label[codon] = label

    sites_by_label = collections.defaultdict(list)
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
                label = codon_to_label.get(codon)
                if label is None:
                    continue
                edit_pos  = cds_start + i + 1
                win_start = max(0, edit_pos - window)
                win_end   = min(chrom_len, edit_pos + window + 1)
                sites_by_label[label].append({
                    "chrom":       chrom,
                    "edit_pos":    edit_pos,
                    "codon_start": cds_start + i,
                    "strand":      strand,
                    "codon":       codon,
                    "transcript":  tid,
                    "gene_name":   gname,
                    "win_start":   win_start,
                    "win_end":     win_end,
                    "codon_label": label,
                })

    result = {}
    for label, sites in sites_by_label.items():
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

        print(f"  [{label}] {len(deduped):,} unique sites "
              f"(deduplicated from {len(sites):,}).", file=sys.stderr)
        result[label] = deduped

    for label in codon_groups:
        result.setdefault(label, [])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5/6. Streaming per-library read collection + pileup aggregation
#
# Replaces the original script's load_all_parquet_chunks (one big
# pd.concat'd DataFrame per library) + collect_his_site_reads_from_dataframe
# + one aggregate_sites call per codon label with a single streaming pass
# per library: one parquet chunk resident in memory at a time, with a
# small per-site accumulator (just A/G/C/T counts per relative position,
# not per-read data) carried across chunks. Covers His AND every control
# codon in that same pass -- each parquet chunk is read off disk exactly
# once per library, regardless of how many codon groups are requested.
# ─────────────────────────────────────────────────────────────────────────────

def _attach_win_seqs(ref_fasta: pysam.FastaFile, sites: list) -> None:
    """
    Fetch each site's window sequence once, up front, and cache it on the
    site dict -- shared across every parquet chunk AND both libraries
    (win_seq depends only on chrom/window/ref_fasta), instead of the
    original count_mismatches_at_site's ref_fasta.fetch() once per
    (site, chunk) call.
    """
    for site in sites:
        site["win_seq"] = ref_fasta.fetch(
            site["chrom"], site["win_start"], site["win_end"]).upper()


def _site_key(site: dict):
    return (site["chrom"], site["edit_pos"], site["codon_label"])


def _tally_site_reads(dataframe_slice: pd.DataFrame,
                       site: dict,
                       counts_by_rel: dict) -> None:
    """
    Mutates counts_by_rel (a rel_pos -> collections.Counter(A/G/C/T)
    defaultdict) in place from dataframe_slice's reads -- the same
    per-read tally the original count_mismatches_at_site did in one shot,
    but split out so it can be called once per parquet CHUNK and
    accumulate across calls instead of requiring one big already-
    concatenated DataFrame.

    Uses edit_string + absolute_indices, both already sense-oriented from
    the parquet generator (see the original script's docstring for the
    full encoding rationale): '1' = A->G edit, '0' = no edit at ref=A,
    '2' = indel/non-target base, only '0'/'1' contribute counts.
    """
    edit_pos   = site["edit_pos"]
    win_start  = site["win_start"]
    win_end    = site["win_end"]
    win_seq    = site["win_seq"]
    gene_minus = (site["strand"] == "-")

    for read in dataframe_slice.itertuples():
        edit_str    = read.edit_string
        abs_indices = read.absolute_indices

        for i, ref_pos in enumerate(abs_indices):
            if ref_pos is None or (isinstance(ref_pos, float) and ref_pos != ref_pos):
                continue  # None or NaN (None becomes NaN after parquet round-trip)
            if i >= len(edit_str):
                continue
            ref_pos = int(ref_pos)
            if not (win_start <= ref_pos < win_end):
                continue

            edit_val = edit_str[i]
            if edit_val == "2":
                continue

            ref_base_genomic = win_seq[ref_pos - win_start]
            ref_base_tx = complement_base(ref_base_genomic) \
                          if gene_minus else ref_base_genomic
            if ref_base_tx != "A":
                continue

            rel_pos = ref_pos - edit_pos
            if gene_minus:
                rel_pos = -rel_pos
            rel_pos = int(rel_pos)

            qbase = "G" if edit_val == "1" else "A"
            counts_by_rel[rel_pos][qbase] += 1


def _finalize_pos_data(site: dict, counts_by_rel: dict) -> dict:
    """Same tail as the original count_mismatches_at_site: turn the
    accumulated per-rel_pos Counters into {rel_pos: {ref_pos, ref_base,
    A, G, C, T, cov}}, using the site's cached win_seq."""
    win_seq   = site["win_seq"]
    win_start = site["win_start"]
    edit_pos  = site["edit_pos"]
    strand    = site["strand"]
    gene_minus = (strand == "-")

    pos_data = {}
    for rel_pos, counts in counts_by_rel.items():
        gpos = int(edit_pos + rel_pos) if strand == "+" else int(edit_pos - rel_pos)
        ref_base_genomic = win_seq[gpos - win_start]
        ref_base = complement_base(ref_base_genomic) if gene_minus else ref_base_genomic
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


def _pos_data_to_records(site: dict, pos_data: dict, min_coverage: int) -> list:
    """Same record-building tail as the original aggregate_sites."""
    records = []
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
    return records


def process_library_streaming(parquet_dir: str,
                               sites: list,
                               ref_fasta: pysam.FastaFile,
                               min_coverage: int,
                               library_label: str,
                               columns: list = None) -> dict:
    """
    One streaming pass over parquet_dir's chunks for one library, covering
    every site in `sites` (His + every requested control codon at once --
    sites already carry their own codon_label from find_multi_codon_positions).
    Only one chunk's DataFrame is resident at a time; per-site A/G/C/T
    counts accumulate across chunks in a small dict (keyed by _site_key),
    not per-read data. Requires sites to already have "win_seq" attached
    (see _attach_win_seqs) -- fetching it here per chunk would repeat the
    same ref_fasta.fetch() once per (site, chunk) instead of once total.

    Returns {n_total, n_rev, his_reads, read_eff_dist, agg_by_label} --
    agg_by_label is {codon_label: pd.DataFrame(records)}, the same shape
    the original script's per-codon-label aggregate_sites() call returned,
    just all computed in one pass instead of one full library scan per
    codon label.
    """
    chunks = list_parquet_chunks(parquet_dir)
    if not chunks:
        return {
            "n_total": 0, "n_rev": 0, "his_reads": set(),
            "read_eff_dist": np.array([]), "agg_by_label": {},
        }

    proj_cols = _select_available_columns(chunks[0], columns) if columns else None
    has_read_span = proj_cols is None or \
                    ("read_start" in proj_cols and "read_end" in proj_cols)

    sites_by_chrom = collections.defaultdict(list)
    for site in sites:
        sites_by_chrom[site["chrom"]].append(site)

    counts_by_site = {_site_key(site): collections.defaultdict(collections.Counter)
                       for site in sites}
    his_reads  = set()
    eff_chunks = []
    n_total    = 0
    n_rev      = 0

    for ci, chunk_path in enumerate(chunks):
        print(f"  [{library_label}] chunk {ci + 1}/{len(chunks)}: "
              f"{chunk_path.name}", file=sys.stderr)
        chunk_df = pd.read_parquet(chunk_path, columns=proj_cols)

        n_total += len(chunk_df)
        if "is_reverse" in chunk_df.columns:
            n_rev += int(chunk_df["is_reverse"].sum())
        if "global_edit_freq" in chunk_df.columns:
            eff_chunks.append(chunk_df["global_edit_freq"].dropna().values)

        for chrom, chrom_chunk_df in chunk_df.groupby("chrom"):
            chrom_sites = sites_by_chrom.get(chrom)
            if not chrom_sites:
                continue
            for site in chrom_sites:
                if has_read_span:
                    site_reads = chrom_chunk_df[
                        (chrom_chunk_df["read_start"] < site["win_end"]) &
                        (chrom_chunk_df["read_end"]   > site["win_start"])
                    ]
                else:
                    site_reads = chrom_chunk_df
                if site_reads.empty:
                    continue
                _tally_site_reads(site_reads, site, counts_by_site[_site_key(site)])
                if site["codon_label"] == "His":
                    his_reads.update(site_reads["read_id"])

        del chunk_df

    records_by_label = collections.defaultdict(list)
    for site in sites:
        pos_data = _finalize_pos_data(site, counts_by_site[_site_key(site)])
        records_by_label[site["codon_label"]].extend(
            _pos_data_to_records(site, pos_data, min_coverage))

    agg_by_label = {label: pd.DataFrame(recs)
                    for label, recs in records_by_label.items()}
    read_eff_dist = np.concatenate(eff_chunks) if eff_chunks else np.array([])

    return {
        "n_total": n_total, "n_rev": n_rev,
        "his_reads": his_reads,
        "read_eff_dist": read_eff_dist,
        "agg_by_label": agg_by_label,
    }


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
# 7b. Manuscript color map
# ─────────────────────────────────────────────────────────────────────────────

def load_color_map(path: str) -> dict:
    """
    Parse a manuscript color-map TSV with columns:
        sample_name, rep, path, hex_color (no leading '#')
    Returns a dict keyed by both "name_rep" (e.g. "+3AT_rep1") and bare
    "name" (first match wins for the bare key) mapping to "#RRGGBB".
    """
    color_map = {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 4:
                continue
            name, rep, _path, hexcol = fields[0], fields[1], fields[2], fields[3]
            hexcol = "#" + hexcol.strip().lstrip("#")
            color_map.setdefault(f"{name}_{rep}" if rep else name, hexcol)
            color_map.setdefault(name, hexcol)
    return color_map


def lookup_color(color_map: dict, label: str):
    """Return the hex color for label, or None if absent/no map given."""
    if not color_map:
        return None
    return color_map.get(label)


def hex_to_pyx_color(hexcol: str):
    from pyx import color
    hexcol = hexcol.lstrip("#")
    r = int(hexcol[0:2], 16) / 255.0
    g = int(hexcol[2:4], 16) / 255.0
    b = int(hexcol[4:6], 16) / 255.0
    return color.rgb(r, g, b)


def resolve_color(hexcol, fallback_cmyk: tuple):
    """hexcol (e.g. '#1F78B4') -> pyx color, or fallback_cmyk if hexcol is None."""
    from pyx import color
    if hexcol:
        return hex_to_pyx_color(hexcol)
    return color.cmyk(*fallback_cmyk)


def grey_shades(n: int, lo: float = 0.25, hi: float = 0.7) -> list:
    """n evenly spaced pyx.color.gray values from dark (lo) to light (hi)."""
    from pyx import color
    if n <= 0:
        return []
    if n == 1:
        return [color.gray((lo + hi) / 2)]
    return [color.gray(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


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
                            title=f"log2FC ({tex_escape(label2)}/{tex_escape(label1)})"),
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
    """CDF of per-read global A->G editing efficiency for both libraries."""
    from pyx import graph, style, path, text as pyx_text

    g = graph.graphxy(
        width=panel_w, height=panel_h,
        xpos=xpos, ypos=ypos,
        x=graph.axis.linear(min=0, max=1, title="Edit Freq per Read"),
        y=graph.axis.linear(min=0, max=1, title="Cumulative fraction"),
    )

    for s, col, label in [(s1, col1, label1), (s2, col2, label2)]:
        eff = s.get("read_eff_dist", np.array([]))
        if len(eff) == 0:
            continue
        eff_s = np.sort(eff)
        cdf   = np.arange(1, len(eff_s) + 1) / len(eff_s)
        g.plot(graph.data.points(list(zip(eff_s.tolist(), cdf.tolist())),
                                 x=1, y=2),
               [graph.style.line([col, style.linewidth.normal,
                                  style.linestyle.solid])])

    c.insert(g)
    return g


def plot_cdf_pyx(s1: dict, s2: dict,
                  label1: str, label2: str,
                  output_prefix: str,
                  color1: str = None, color2: str = None):
    """
    Standalone CDF figure of per-read global A->G editing efficiency.
    One panel with both libraries overlaid and a legend.
    """
    from pyx import canvas, graph, color, style, path, text as pyx_text

    col1    = resolve_color(color1, (0, 0, 0, 1))
    col2    = resolve_color(color2, (1, 0.5, 0, 0))
    panel_w = 7
    panel_h = 5
    leg_lw  = 0.8
    leg_dy  = 0.55

    c = canvas.canvas()

    g = graph.graphxy(
        width=panel_w, height=panel_h,
        xpos=0, ypos=0,
        x=graph.axis.linear(min=0, max=1, title="A{$\\to$}G edit freq per read"),
        y=graph.axis.linear(min=0, max=1, title="Cumulative fraction"),
    )

    for s, col, label, ls in [
        (s1, col1, label1, style.linestyle.solid),
        (s2, col2, label2, style.linestyle.solid),
    ]:
        eff = s.get("read_eff_dist", np.array([]))
        if len(eff) == 0:
            continue
        eff_s = np.sort(eff)
        cdf   = np.arange(1, len(eff_s) + 1) / len(eff_s)
        g.plot(graph.data.points(list(zip(eff_s.tolist(), cdf.tolist())),
                                 x=1, y=2),
               [graph.style.line([col, style.linewidth.normal, ls])])

    c.insert(g)

    # Title
    c.text(g.xpos + g.width / 2., g.ypos + g.height + 0.4,
           "Per-read A{$\to$}G editing efficiency",
           [pyx_text.halign.center, pyx_text.size.normalsize])

    # Legend to the right of the panel
    leg_x     = g.xpos + g.width + 0.4
    leg_y_top = g.ypos + g.height - 0.3
    for j, (label, col) in enumerate([(label1, col1), (label2, col2)]):
        ly = leg_y_top - j * leg_dy
        c.stroke(path.line(leg_x, ly, leg_x + leg_lw, ly),
                 [col, style.linewidth.normal, style.linestyle.solid])
        c.text(leg_x + leg_lw + 0.15, ly, tex_escape(label),
               [pyx_text.valign.middle, pyx_text.size.small])

    plot_path = f"{output_prefix}_editing_efficiency_cdf_pyx"
    c.writePDFfile(plot_path)
    print(f"  Saved -> {plot_path}.pdf", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Top-level pyx plot functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_pyx(s1: dict, s2: dict,
                         label1: str, label2: str,
                         output_prefix: str, window: int,
                         color1: str = None, color2: str = None):
    """2x2 comparison figure: meta BAM1, meta BAM2, log2FC, CDF."""
    from pyx import canvas, color, style, text as pyx_text

    col1    = resolve_color(color1, (0, 0, 0, 1))
    col2    = resolve_color(color2, (1, 0.5, 0, 0))
    panel_w = 5
    panel_h = 3
    gap     = 2.0

    c = canvas.canvas()

    g1 = _pyx_meta_graph(c, xpos=0, ypos=panel_h + gap,
                          datasets=[(s1["rel_position_agg"], col1,
                                     style.linestyle.solid)],
                          y_title="Edit Frac",
                          window=window, panel_w=panel_w, panel_h=panel_h)
    c.text(g1.xpos + g1.width / 2., g1.ypos + g1.height + 0.4, tex_escape(label1),
           [pyx_text.halign.center, pyx_text.size.normalsize])

    g2 = _pyx_meta_graph(c, xpos=panel_w + gap, ypos=panel_h + gap,
                          datasets=[(s2["rel_position_agg"], col2,
                                     style.linestyle.solid)],
                          y_title="Edit Frac",
                          window=window, panel_w=panel_w, panel_h=panel_h)
    c.text(g2.xpos + g2.width / 2., g2.ypos + g2.height + 0.4, tex_escape(label2),
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
                              output_prefix: str, window: int,
                              color1: str = None, color2: str = None):
    """3-column rank figure: 1st/2nd/3rd His codon, meta overlay + log2FC."""
    from pyx import canvas, color, style, text as pyx_text

    col1    = resolve_color(color1, (0, 0, 0, 1))
    col2    = resolve_color(color2, (1, 0.5, 0, 0))
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
        c.text(-0.2, ypos + panel_h / 2., tex_escape(bam_label),
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
                                        output_prefix: str, window: int,
                                        color1: str = None, color2: str = None):
    """Overlay of His + control codons, one panel per BAM. His is colored
    per-sample (matching the other comparison plots); control codons are
    shades of grey so they read as background/specificity controls."""
    from pyx import canvas, color, style, path, text as pyx_text

    control_names      = list(control_aggs.keys())
    control_color_map  = dict(zip(control_names, grey_shades(len(control_names))))
    his_color_by_label = {
        label1: resolve_color(color1, (0, 0, 0, 1)),
        label2: resolve_color(color2, (1, 0.5, 0, 0)),
    }

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
        his_col = his_color_by_label[bam_label]

        overlay_datasets = [(his_agg, his_col, style.linestyle.solid)]
        for codon_name in control_names:
            agg = control_aggs[codon_name].get(bam_label, pd.DataFrame())
            overlay_datasets.append(
                (agg, control_color_map[codon_name], style.linestyle.dashed))

        g = _pyx_meta_graph(
            c, xpos=0, ypos=ypos,
            datasets=overlay_datasets,
            y_title="Edit Frac", x_title=x_title,
            window=window, panel_w=panel_w, panel_h=panel_h,
        )
        c.text(g.xpos + g.width / 2., g.ypos + g.height + 0.4, tex_escape(bam_label),
               [pyx_text.halign.center, pyx_text.size.normalsize])

        if row_idx == 0:
            leg_x_start = g.xpos + g.width + 0.4
            leg_y_start = g.ypos + g.height - 0.2
            legend_entries = [("His", his_col, style.linestyle.solid)] + [
                (name, control_color_map[name], style.linestyle.dashed)
                for name in control_names
            ]
            for j, (codon_name, col, ls) in enumerate(legend_entries):
                ly = leg_y_start - j * leg_dy
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
                                output_prefix: str, window: int,
                                color1: str = None, color2: str = None):
    """
    2 rows (BAM1 top, BAM2 bottom) x (n_codons + 1) columns.
    Individual codon panels + overlay column.
    y-title on leftmost column only, x-title on bottom row only.
    Codon name titles above each panel. His is colored per-sample; control
    codons are shades of grey.
    """
    from pyx import canvas, color, style, text as pyx_text

    codon_names        = ["His"] + list(control_aggs.keys())
    control_names      = codon_names[1:]
    control_color_map  = dict(zip(control_names, grey_shades(len(control_names))))
    his_color_by_label = {
        label1: resolve_color(color1, (0, 0, 0, 1)),
        label2: resolve_color(color2, (1, 0.5, 0, 0)),
    }

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
        his_col = his_color_by_label[bam_label]
        codon_color_map = {"His": his_col, **control_color_map}

        for col_idx, codon_name in enumerate(codon_names):
            xpos    = col_idx * (panel_w + gap)
            agg     = his_agg if codon_name == "His" \
                      else control_aggs[codon_name].get(bam_label, pd.DataFrame())
            col     = codon_color_map[codon_name]
            y_title = f"{tex_escape(bam_label)} edit frac" if col_idx == 0 else ""

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
        description="Histidine meta analysis from parquet files (optimized: "
                     "column-projected parquet loads, single combined His+"
                     "control-codon CDS walk)."
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
    p.add_argument("--window",       type=int, default=100)
    p.add_argument("--min_coverage", type=int, default=10)
    p.add_argument("--min_edit_frac", type=float, default=0.01)
    p.add_argument("--control_codons", nargs="*", default=None,
                   help="Control codon names for specificity analysis. "
                        f"Default: {list(CONTROL_CODONS.keys())}. "
                        "Pass none to skip.")
    p.add_argument("--color_map", default=None,
                   help="Optional TSV file mapping sample name/rep to a hex "
                        "color (columns: name, rep, path, hex_color, no '#'). "
                        "Colors for --label1/--label2 are looked up as "
                        "'name_rep' or bare 'name'; unmatched labels fall "
                        "back to the default black/red scheme. Control "
                        "codons are always plotted as shades of grey.")
    return p.parse_args()


def main():
    args = parse_args()
    out  = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print("=== Histidine Meta Analysis (parquet, optimized) ===", file=sys.stderr)

    # ── Resolve manuscript colors ─────────────────────────────────────────────
    color_map = load_color_map(args.color_map) if args.color_map else {}
    color1    = lookup_color(color_map, args.label1)
    color2    = lookup_color(color_map, args.label2)
    if args.color_map:
        for lbl, col in [(args.label1, color1), (args.label2, color2)]:
            if col is None:
                print(f"  WARNING: no color found for label '{lbl}' in "
                      f"{args.color_map}; using default.", file=sys.stderr)

    # ── Parse GTF, find His + control codon sites in ONE combined CDS walk ────
    print("\nParsing GTF...", file=sys.stderr)
    ref_fasta    = pysam.FastaFile(args.ref)
    cds_by_chrom = parse_gtf_cds(args.gtf)

    control_codons_to_run = args.control_codons \
                            if args.control_codons is not None \
                            else list(CONTROL_CODONS.keys())
    codon_groups = {"His": HIS_CODONS}
    valid_control_codons = []
    for codon_name in control_codons_to_run:
        if codon_name not in CONTROL_CODONS:
            print(f"  WARNING: unknown control codon '{codon_name}', skipping.",
                  file=sys.stderr)
            continue
        codon_groups[codon_name] = CONTROL_CODONS[codon_name]
        valid_control_codons.append(codon_name)

    print(f"\nFinding His + control codon sites "
          f"({', '.join(codon_groups)}, single CDS pass)...", file=sys.stderr)
    sites_by_label = find_multi_codon_positions(
        ref_fasta, cds_by_chrom, args.window, codon_groups)
    all_sites = [s for label in codon_groups for s in sites_by_label[label]]
    print(f"  {len(all_sites):,} total sites across {len(codon_groups)} "
          f"codon group(s); pre-fetching window sequences...", file=sys.stderr)
    _attach_win_seqs(ref_fasta, all_sites)

    # ── Stream each library's parquet chunks (column-projected) ───────────────
    # One chunk resident at a time -- no load_all_parquet_chunks-style
    # pd.concat of the whole directory -- covering His + every control
    # codon in the same pass instead of one full scan per codon label.
    print(f"\nStreaming {args.parquet1} ({args.label1})...", file=sys.stderr)
    result1 = process_library_streaming(args.parquet1, all_sites, ref_fasta,
                                         args.min_coverage, args.label1,
                                         columns=REQUIRED_COLUMNS)
    print(f"\nStreaming {args.parquet2} ({args.label2})...", file=sys.stderr)
    result2 = process_library_streaming(args.parquet2, all_sites, ref_fasta,
                                         args.min_coverage, args.label2,
                                         columns=REQUIRED_COLUMNS)

    for result, label in [(result1, args.label1), (result2, args.label2)]:
        n_total = result["n_total"]
        n_rev   = result["n_rev"]
        n_fwd   = n_total - n_rev
        pct_rev = 100 * n_rev / n_total if n_total > 0 else 0.0
        print(f"  {label}: {n_total:,} reads  |  "
              f"forward: {n_fwd:,} ({100-pct_rev:.1f}%)  |  "
              f"reverse (minus-strand gene): {n_rev:,} ({pct_rev:.1f}%)",
              file=sys.stderr)
        print(f"  {label}: {len(result['his_reads']):,} reads overlapping "
              f"His sites", file=sys.stderr)

    agg_df1 = result1["agg_by_label"].get("His", pd.DataFrame())
    agg_df2 = result2["agg_by_label"].get("His", pd.DataFrame())

    # ── Compute summaries ─────────────────────────────────────────────────────
    print("\nComputing summaries...", file=sys.stderr)
    s1 = compute_summaries(agg_df1, args.min_edit_frac)
    s2 = compute_summaries(agg_df2, args.min_edit_frac)

    # Fill read efficiency distributions — global_edit_freq per read
    s1["read_eff_dist"] = result1["read_eff_dist"]
    s2["read_eff_dist"] = result2["read_eff_dist"]

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

    # ── Control codon specificity (already aggregated during streaming) ──────
    control_aggs = {}
    if valid_control_codons:
        print("\nComputing control codon specificity aggregates...", file=sys.stderr)
        for codon_name in valid_control_codons:
            control_aggs[codon_name] = {}
            for result, label in [(result1, args.label1), (result2, args.label2)]:
                ctrl_df = result["agg_by_label"].get(codon_name, pd.DataFrame())
                control_aggs[codon_name][label] = \
                    transcript_normalised_agg(ctrl_df) if not ctrl_df.empty \
                    else pd.DataFrame()

    # ── Plot ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...", file=sys.stderr)
    try:
        plot_comparison_pyx(s1, s2, args.label1, args.label2,
                            out, args.window, color1=color1, color2=color2)
        plot_rank_comparison_pyx(s1, s2, args.label1, args.label2,
                                  out, args.window, color1=color1, color2=color2)
        plot_codon_type_comparison_pyx(s1, s2, args.label1, args.label2,
                                        out, args.window)
        plot_cdf_pyx(s1, s2, args.label1, args.label2, out,
                     color1=color1, color2=color2)
        if control_aggs:
            plot_codon_specificity_pyx(
                his_agg_bam1=s1["rel_position_agg"],
                his_agg_bam2=s2["rel_position_agg"],
                control_aggs=control_aggs,
                label1=args.label1, label2=args.label2,
                output_prefix=out, window=args.window,
                color1=color1, color2=color2,
            )
            plot_codon_specificity_overlay_pyx(
                his_agg_bam1=s1["rel_position_agg"],
                his_agg_bam2=s2["rel_position_agg"],
                control_aggs=control_aggs,
                label1=args.label1, label2=args.label2,
                output_prefix=out, window=args.window,
                color1=color1, color2=color2,
            )
    except Exception as e:
        print(f"  WARNING: pyx plotting failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    ref_fasta.close()
    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    Tee()
    main()
