'''
August 11, 2026 LT

Simplified variant of metaHistidineFromParquet2.py: every sample/rep gets
overlaid on ONE meta plot (transcript-normalised A->G editing fraction vs.
position relative to the nearest His codon), instead of the label1-vs-
label2 comparison suite v1/v2 build. No control codons, no CAT-vs-CAC
codon-type panel, no rank-split panels, no log2FC, and no per-read editing-
efficiency CDF -- just the one core His-codon meta curve, colored
per-library (manuscript COLOR_MAP if given, else a built-in palette) so an
arbitrary number of libraries can be compared on sight.

Reuses v2's two real optimizations since they apply regardless of library
count:
  1. Parquet column projection (schema_arrow-checked, so it degrades
     gracefully on older chunks) -- global_edit_freq is dropped from the
     projected columns entirely here, since nothing in this script needs
     per-read editing efficiency.
  2. Streaming per-chunk aggregation (process_library_streaming) -- one
     parquet chunk resident at a time, per-site A/G/C/T counts accumulated
     across chunks, instead of concatenating an entire library into one
     DataFrame.

inputs:
- libs file (line-delimited "fileName rep parquetDir", same convention as
  substitutionProfileFromParquet.py / polysomeShadowHMMQC.py)
- ref fasta
- GTF
- window size
- min coverage at site
- output prefix
- optional manuscript color-map TSV
'''

import argparse
import sys
import re
import collections
from pathlib import Path

import pysam
import pandas as pd
import numpy as np
from logJosh import Tee


HIS_CODONS = {"CAT", "CAC"}
ACGT_BASES = set("ACGT")

PALETTE   = [None]   # list of pyx colors, set in main() after pyx import
COLOR_MAP = {}        # {libraryID: "#RRGGBB"}, set in main() if --color_map given


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
    special, so library labels render literally instead of erroring."""
    return ''.join(_TEX_SPECIAL.get(ch, ch) for ch in str(s))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Parquet loading
# ─────────────────────────────────────────────────────────────────────────────

# No global_edit_freq here -- nothing in this script computes per-read
# editing efficiency, unlike v1/v2's CDF plot.
REQUIRED_COLUMNS = [
    "chrom", "read_id", "read_start", "read_end",
    "edit_string", "absolute_indices", "is_reverse",
]


def list_parquet_chunks(parquet_dir: str) -> list:
    return sorted(Path(parquet_dir).glob("*.parquet"))


def _select_available_columns(parquet_path: Path, wanted: list) -> list:
    """
    Intersect `wanted` with parquet_path's own schema, so a caller asking
    for e.g. read_start/read_end/is_reverse doesn't blow up on older
    chunks that predate those columns -- pd.read_parquet(columns=...)
    raises if any requested column is absent from the file.

    Uses schema_arrow (the Arrow logical schema pandas' own column names
    come from), NOT the raw Parquet-level schema -- list-typed columns
    (e.g. absolute_indices) are stored as a nested "item" field at the
    Parquet physical-schema level, so plain .schema.names reports "item"
    for every such column instead of its real pandas column name, which
    would make this silently drop columns it should have kept.
    """
    import pyarrow.parquet as pq
    schema_names = set(pq.ParquetFile(parquet_path).schema_arrow.names)
    missing = [c for c in wanted if c not in schema_names]
    if missing:
        print(f"  NOTE: {parquet_path.name} schema is missing {missing}; "
              f"reading without them.", file=sys.stderr)
    return [c for c in wanted if c in schema_names]


def parse_libs_file(path: str) -> list:
    """
    Line-delimited 'fileName rep parquetDir' -> [(libraryID, parquetDir), ...]
    with libraryID = 'fileName-rep', the same convention
    substitutionProfileFromParquet.py/polysomeShadowHMMQC.py use.
    """
    libs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            fileName, rep, parquetDir = parts[0], parts[1], parts[2]
            libs.append((f"{fileName}-{rep}", parquetDir))
    return libs


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
# 4. Site finding (His codons only -- no control-codon groups to merge)
# ─────────────────────────────────────────────────────────────────────────────

def find_his_positions(ref_fasta: pysam.FastaFile,
                        cds_by_chrom: dict,
                        window: int) -> list:
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
                if codon not in HIS_CODONS:
                    continue
                edit_pos  = cds_start + i + 1
                win_start = max(0, edit_pos - window)
                win_end   = min(chrom_len, edit_pos + window + 1)
                sites.append({
                    "chrom":       chrom,
                    "edit_pos":    edit_pos,
                    "strand":      strand,
                    "codon":       codon,
                    "transcript":  tid,
                    "gene_name":   gname,
                    "win_start":   win_start,
                    "win_end":     win_end,
                })

    seen_positions = set()
    deduped = []
    for site in sites:
        key = (site["chrom"], site["edit_pos"])
        if key not in seen_positions:
            seen_positions.add(key)
            deduped.append(site)

    print(f"  {len(deduped):,} unique His sites (deduplicated from "
          f"{len(sites):,}).", file=sys.stderr)
    return deduped


def _attach_win_seqs(ref_fasta: pysam.FastaFile, sites: list) -> None:
    """Fetch each site's window sequence once, up front -- shared across
    every parquet chunk AND every library, instead of re-fetching the
    same window once per (site, chunk).

    Also fetches a 1bp-padded copy (motif_win_seq/motif_win_start) so
    _site_motif_at_gpos can read the immediate flanking base on either
    side of every position in the window, including the two positions
    right at win_start/win_end-1 that "win_seq" alone has no flank for."""
    for site in sites:
        chrom = site["chrom"]
        site["win_seq"] = ref_fasta.fetch(
            chrom, site["win_start"], site["win_end"]).upper()
        chrom_len = ref_fasta.get_reference_length(chrom)
        pad_start = max(0, site["win_start"] - 1)
        pad_end   = min(chrom_len, site["win_end"] + 1)
        site["motif_win_seq"]   = ref_fasta.fetch(chrom, pad_start, pad_end).upper()
        site["motif_win_start"] = pad_start


def _site_motif_at_gpos(site: dict, gpos: int):
    """
    3nt motif (prevNt,'A',nextNt), read 5'->3' along the mRNA, at genomic
    position gpos within site's (padded) window. Only meaningful when
    gpos's transcript-sense ref base is 'A' (caller's responsibility --
    same convention as computeMotifFreqs in calculateProtectionAcrossParquets.py).
    Returns None if gpos falls outside the padded window or either flank
    isn't a plain A/C/G/T base.
    """
    pad_seq   = site["motif_win_seq"]
    pad_start = site["motif_win_start"]
    idx = gpos - pad_start
    if idx - 1 < 0 or idx + 1 >= len(pad_seq):
        return None
    prev_g, next_g = pad_seq[idx - 1], pad_seq[idx + 1]
    if site["strand"] == "-":
        # mRNA-sense 5' neighbor is the complement of the genomically-higher
        # base, and the 3' neighbor is the complement of the genomically-lower
        # base -- same convention as computeMotifFreqs' minus-strand branch.
        prevNt, nextNt = complement_base(next_g), complement_base(prev_g)
    else:
        prevNt, nextNt = prev_g, next_g
    if prevNt not in ACGT_BASES or nextNt not in ACGT_BASES:
        return None
    return prevNt + "A" + nextNt


def _site_key(site: dict):
    return (site["chrom"], site["edit_pos"])


# ─────────────────────────────────────────────────────────────────────────────
# 5. Streaming per-library pileup + aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _tally_site_reads(dataframe_slice: pd.DataFrame,
                       site: dict,
                       counts_by_rel: dict) -> None:
    """
    Mutates counts_by_rel (rel_pos -> collections.Counter(A/G/C/T)
    defaultdict) in place from dataframe_slice's reads, so this can be
    called once per parquet CHUNK and accumulate across calls instead of
    requiring one big already-concatenated DataFrame.

    edit_string encoding (transcript/sense coordinates, see
    shadowingBamToParquetWithGTF2.py): '1' = A->G edit, '0' = no edit at a
    ref=A position, '2' = indel/non-target ref base -- only '0'/'1'
    contribute counts.
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
    records = []
    for rel_pos, counts in pos_data.items():
        if counts["cov"] < min_coverage:
            continue
        ag_denom = counts["A"] + counts["G"]
        ag_edit  = counts["G"] / ag_denom \
                   if counts["ref_base"] == "A" and ag_denom > 0 else np.nan
        motif = _site_motif_at_gpos(site, counts["ref_pos"]) \
                if counts["ref_base"] == "A" else None
        records.append({
            "site_id":      f"{site['chrom']}:{site['edit_pos']}",
            "transcript":   site["transcript"],
            "chrom":        site["chrom"],
            "edit_pos":     site["edit_pos"],
            "strand":       site["strand"],
            "codon":        site["codon"],
            "rel_pos":      rel_pos,
            "ref_base":     counts["ref_base"],
            "motif":        motif,
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
                               min_coverage: int,
                               library_label: str,
                               columns: list = None) -> dict:
    """
    One streaming pass over parquet_dir's chunks: only one chunk is
    resident at a time, with per-site A/G/C/T counts accumulated across
    chunks in a small dict -- not per-read data. Requires sites to already
    have "win_seq" attached (see _attach_win_seqs).

    Returns {n_total, n_rev, agg_df}.
    """
    chunks = list_parquet_chunks(parquet_dir)
    if not chunks:
        return {"n_total": 0, "n_rev": 0, "agg_df": pd.DataFrame()}

    proj_cols = _select_available_columns(chunks[0], columns) if columns else None
    has_read_span = proj_cols is None or \
                    ("read_start" in proj_cols and "read_end" in proj_cols)

    sites_by_chrom = collections.defaultdict(list)
    for site in sites:
        sites_by_chrom[site["chrom"]].append(site)

    counts_by_site = {_site_key(site): collections.defaultdict(collections.Counter)
                       for site in sites}
    n_total = 0
    n_rev   = 0

    for ci, chunk_path in enumerate(chunks):
        print(f"  [{library_label}] chunk {ci + 1}/{len(chunks)}: "
              f"{chunk_path.name}", file=sys.stderr)
        chunk_df = pd.read_parquet(chunk_path, columns=proj_cols)

        n_total += len(chunk_df)
        if "is_reverse" in chunk_df.columns:
            n_rev += int(chunk_df["is_reverse"].sum())

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

        del chunk_df

    records = []
    for site in sites:
        pos_data = _finalize_pos_data(site, counts_by_site[_site_key(site)])
        records.extend(_pos_data_to_records(site, pos_data, min_coverage))

    return {"n_total": n_total, "n_rev": n_rev, "agg_df": pd.DataFrame(records)}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def transcript_normalised_agg(df: pd.DataFrame, pseudo: float = 1e-3) -> pd.DataFrame:
    """
    Two-stage transcript-normalised aggregation restricted to ref=A positions.
    Stage 1: mean (ag_edit_frac + pseudo) per (transcript, rel_pos)
    Stage 2: grand mean + SEM across transcripts at each rel_pos
    """
    ref_a = df[
        df["ag_edit_frac"].notna() &
        (df["ref_base"] == "A") &
        (~df["in_his_codon"] | df["is_his_A"])
    ].copy()
    ref_a["ag_edit_frac_ps"] = ref_a["ag_edit_frac"] + pseudo

    tx_mean = (
        ref_a.groupby(["transcript", "rel_pos"])["ag_edit_frac_ps"]
             .mean()
             .reset_index()
             .rename(columns={"ag_edit_frac_ps": "tx_mean_edit_frac"})
    )

    agg = (
        tx_mean.groupby(["rel_pos"])
               .agg(
                   mean_edit_frac=("tx_mean_edit_frac", "mean"),
                   sem_edit_frac =("tx_mean_edit_frac", lambda x: x.sem()),
                   n_transcripts =("transcript", "nunique"),
               )
               .reset_index()
    )
    return agg


def load_motif_baseline_csv(path: str) -> dict:
    """
    Load an editEfficiencyFromParquet.py *_context_editing.csv (columns:
    label, context, n_edited, n_total, edit_rate) into
    {label: {motif: edit_rate}}, keyed literally by that CSV's own label
    column. editEfficiencyFromParquet.py joins its labels as
    "sample_name_rep" (underscore) while this script's own libraryIDs are
    "fileName-rep" (hyphen, see parse_libs_file) -- main() handles that
    join-character mismatch at lookup time (see motif_baseline_for_label),
    rather than this loader guessing at label variants.
    """
    df = pd.read_csv(path)
    baseline = collections.defaultdict(dict)
    for row in df.itertuples(index=False):
        baseline[row.label][row.context] = float(row.edit_rate)
    return dict(baseline)


def motif_baseline_for_label(baseline_by_label: dict, label: str) -> dict:
    """
    Look up label (this script's own "fileName-rep" libraryID) against
    baseline_by_label's keys, which come from editEfficiencyFromParquet.py
    and use "fileName_rep" instead. Tries the exact label first, then
    swaps only the LAST '-' for '_' (recovering the fileName/rep split
    parse_libs_file itself created, since rep is always the final token
    joined on) -- safer than a blanket str.replace, which would also
    mangle any '-' that happens to be part of fileName itself.
    """
    if label in baseline_by_label:
        return baseline_by_label[label]
    if "-" in label:
        fileName_part, rep_part = label.rsplit("-", 1)
        alt_label = f"{fileName_part}_{rep_part}"
        if alt_label in baseline_by_label:
            return baseline_by_label[alt_label]
    return {}


def motif_normalized_agg(df: pd.DataFrame, motif_freqs: dict,
                          pseudo: float = 1e-3) -> pd.DataFrame:
    """
    Same two-stage transcript-normalised aggregation as
    transcript_normalised_agg, but on ag_edit_frac expressed relative to
    this library's own 3nt-motif baseline editing rate (motif_freqs, loaded
    from an editEfficiencyFromParquet.py *_context_editing.csv via
    load_motif_baseline_csv) instead of the raw fraction -- same
    motif-bias-correction idea as calculateProtectionAcrossParquets.py's
    weightedEdit/weightedTot, just applied per meta-plot position instead
    of pooled over UTR5/CDS/UTR3.

    ratio ~1 means that position edits at exactly its own motif's
    library-wide predicted rate; <1 means more protected than motif
    composition alone would predict; >1 means less protected. This is
    what isolates a "distance from His codon" effect from a confound
    where different rel_pos across transcripts simply have different
    flanking sequence (and thus different intrinsic editability).
    """
    ref_a = df[
        df["ag_edit_frac"].notna() &
        (df["ref_base"] == "A") &
        (~df["in_his_codon"] | df["is_his_A"]) &
        df["motif"].notna()
    ].copy()
    ref_a["motif_freq"] = ref_a["motif"].map(motif_freqs)
    ref_a = ref_a[ref_a["motif_freq"].notna() & (ref_a["motif_freq"] > 0)]
    ref_a["motif_norm_ratio"] = (ref_a["ag_edit_frac"] + pseudo) / (ref_a["motif_freq"] + pseudo)

    tx_mean = (
        ref_a.groupby(["transcript", "rel_pos"])["motif_norm_ratio"]
             .mean()
             .reset_index()
             .rename(columns={"motif_norm_ratio": "tx_mean_ratio"})
    )

    agg = (
        tx_mean.groupby(["rel_pos"])
               .agg(
                   mean_ratio    =("tx_mean_ratio", "mean"),
                   sem_ratio     =("tx_mean_ratio", lambda x: x.sem()),
                   n_transcripts =("transcript", "nunique"),
               )
               .reset_index()
    )
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 7. Manuscript color map
# ─────────────────────────────────────────────────────────────────────────────

def load_color_map(path: str) -> dict:
    """
    Parse a manuscript color-map TSV with columns:
        sample_name, rep, path, hex_color (no leading '#')
    Returns a dict keyed by "name_rep", "name-rep" (this script's own
    libraryID convention), and bare "name" (first match wins for the bare
    key) mapping to "#RRGGBB".
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
            if rep:
                color_map.setdefault(f"{name}_{rep}", hexcol)
                color_map.setdefault(f"{name}-{rep}", hexcol)
            color_map.setdefault(name, hexcol)
    return color_map


def hex_to_pyx_color(hexcol: str):
    from pyx import color
    hexcol = hexcol.lstrip("#")
    r = int(hexcol[0:2], 16) / 255.0
    g = int(hexcol[2:4], 16) / 255.0
    b = int(hexcol[4:6], 16) / 255.0
    return color.rgb(r, g, b)


def _libcolor(libID: str, i: int):
    """COLOR_MAP's manuscript hex color for libID if present, else the
    i'th color from the built-in PALETTE cycle."""
    hexcol = COLOR_MAP.get(libID)
    if hexcol:
        return hex_to_pyx_color(hexcol)
    return PALETTE[i % len(PALETTE)]


# ─────────────────────────────────────────────────────────────────────────────
# 8. Pyx plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_meta_all_libraries_pyx(rel_agg_by_lib: dict, window: int, output_prefix: str,
                                y_title="Edit Frac", x_title="Relative Position",
                                panel_w=10, panel_h=6):
    """
    One panel, every library overlaid -- one line (+ dotted SEM bounds)
    per library, colored via _libcolor (manuscript COLOR_MAP if given,
    else PALETTE), with a legend and thin reference markers at the His
    codon span (rel_pos -1/0/+1, center dashed).
    """
    from pyx import canvas, graph, color, style, text as pyx_text

    libIDs = sorted(rel_agg_by_lib)

    y_max = 0.02
    for lib in libIDs:
        rel_agg = rel_agg_by_lib[lib]
        if rel_agg.empty:
            continue
        frac = rel_agg["mean_edit_frac"].values
        sem  = rel_agg["sem_edit_frac"].values
        candidate = float(np.nanmax(frac + sem)) * 1.15 if len(frac) > 0 else 0
        y_max = max(y_max, candidate)

    g = graph.graphxy(
        width=panel_w, height=panel_h, xpos=0, ypos=0,
        x=graph.axis.linear(min=-window, max=window, title=x_title),
        y=graph.axis.linear(min=0, max=y_max, title=y_title),
        key=graph.key.key(pos="tr", hinside=0),
    )

    g.plot(graph.data.function("x(y)=-1", min=0, max=y_max, title=None),
           [graph.style.line([color.gray(0.8), style.linewidth.thin])])
    g.plot(graph.data.function("x(y)=1", min=0, max=y_max, title=None),
           [graph.style.line([color.gray(0.8), style.linewidth.thin])])
    g.plot(graph.data.function("x(y)=0", min=0, max=y_max, title=None),
           [graph.style.line([color.gray(0.5), style.linewidth.thick,
                              style.linestyle.dashed])])

    for i, lib in enumerate(libIDs):
        rel_agg = rel_agg_by_lib[lib]
        if rel_agg.empty:
            continue
        pos  = rel_agg["rel_pos"].values
        frac = rel_agg["mean_edit_frac"].values
        sem  = rel_agg["sem_edit_frac"].values
        col  = _libcolor(lib, i)
        n_tx = int(rel_agg["n_transcripts"].max())

        for pts in [list(zip(pos.tolist(), (frac - sem).tolist())),
                    list(zip(pos.tolist(), (frac + sem).tolist()))]:
            g.plot(graph.data.points(pts, x=1, y=2, title=None),
                   [graph.style.line([col, style.linewidth.thin,
                                      style.linestyle.dotted])])

        title = r"%s (n=%d tx)" % (tex_escape(lib), n_tx)
        g.plot(graph.data.points(list(zip(pos.tolist(), frac.tolist())), x=1, y=2,
                                 title=title),
               [graph.style.line([col, style.linewidth.Thick])])

    c = canvas.canvas()
    c.insert(g)
    plot_path = f"{output_prefix}_meta_all_libraries_pyx"
    c.writePDFfile(plot_path)
    print(f"  Saved -> {plot_path}.pdf", file=sys.stderr)


def plot_meta_all_libraries_motif_normalized_pyx(rel_agg_by_lib: dict, window: int,
                                                  output_prefix: str,
                                                  y_title="Edit / Motif Baseline",
                                                  x_title="Relative Position",
                                                  panel_w=10, panel_h=6):
    """
    Same layout as plot_meta_all_libraries_pyx, but for motif_normalized_agg's
    output: a ratio of observed A->G editing to this library's own 3nt-motif
    baseline rate, so the reference line that matters here is a horizontal
    y=1 (ratio of 1 = edits exactly at the motif-predicted rate) rather than
    y=0.
    """
    from pyx import canvas, graph, color, style, text as pyx_text

    libIDs = sorted(rel_agg_by_lib)

    y_max = 1.2
    y_min = 0.0
    for lib in libIDs:
        rel_agg = rel_agg_by_lib[lib]
        if rel_agg.empty:
            continue
        ratio = rel_agg["mean_ratio"].values
        sem   = rel_agg["sem_ratio"].values
        if len(ratio) > 0:
            y_max = max(y_max, float(np.nanmax(ratio + sem)) * 1.15)
            y_min = min(y_min, float(np.nanmin(ratio - sem)) * 0.9)

    g = graph.graphxy(
        width=panel_w, height=panel_h, xpos=0, ypos=0,
        x=graph.axis.linear(min=-window, max=window, title=x_title),
        y=graph.axis.linear(min=y_min, max=y_max, title=y_title),
        key=graph.key.key(pos="tr", hinside=0),
    )

    g.plot(graph.data.function("x(y)=-1", min=y_min, max=y_max, title=None),
           [graph.style.line([color.gray(0.8), style.linewidth.thin])])
    g.plot(graph.data.function("x(y)=1", min=y_min, max=y_max, title=None),
           [graph.style.line([color.gray(0.8), style.linewidth.thin])])
    g.plot(graph.data.function("x(y)=0", min=y_min, max=y_max, title=None),
           [graph.style.line([color.gray(0.5), style.linewidth.thick,
                              style.linestyle.dashed])])
    g.plot(graph.data.function("y(x)=1", min=-window, max=window, title=None),
           [graph.style.line([color.gray(0.5), style.linewidth.thick,
                              style.linestyle.dashed])])

    for i, lib in enumerate(libIDs):
        rel_agg = rel_agg_by_lib[lib]
        if rel_agg.empty:
            continue
        pos   = rel_agg["rel_pos"].values
        ratio = rel_agg["mean_ratio"].values
        sem   = rel_agg["sem_ratio"].values
        col   = _libcolor(lib, i)
        n_tx  = int(rel_agg["n_transcripts"].max())

        for pts in [list(zip(pos.tolist(), (ratio - sem).tolist())),
                    list(zip(pos.tolist(), (ratio + sem).tolist()))]:
            g.plot(graph.data.points(pts, x=1, y=2, title=None),
                   [graph.style.line([col, style.linewidth.thin,
                                      style.linestyle.dotted])])

        title = r"%s (n=%d tx)" % (tex_escape(lib), n_tx)
        g.plot(graph.data.points(list(zip(pos.tolist(), ratio.tolist())), x=1, y=2,
                                 title=title),
               [graph.style.line([col, style.linewidth.Thick])])

    c = canvas.canvas()
    c.insert(g)
    plot_path = f"{output_prefix}_meta_all_libraries_motifNorm_pyx"
    c.writePDFfile(plot_path)
    print(f"  Saved -> {plot_path}.pdf", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# 9. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Histidine meta analysis from parquet files -- every "
                     "sample/rep overlaid on one meta plot."
    )
    p.add_argument("--libs", required=True,
                   help="Line-delimited 'fileName rep parquetDir' file "
                        "(same convention as substitutionProfileFromParquet.py "
                        "/ polysomeShadowHMMQC.py).")
    p.add_argument("--ref", required=True)
    p.add_argument("--gtf", required=True)
    p.add_argument("--output",       default="his_meta_all")
    p.add_argument("--window",       type=int, default=100)
    p.add_argument("--min_coverage", type=int, default=10)
    p.add_argument("--color_map", default=None,
                   help="Optional TSV file mapping sample name/rep to a hex "
                        "color (columns: name, rep, path, hex_color, no '#'). "
                        "Colors are looked up per library as 'name_rep'/"
                        "'name-rep' or bare 'name'; unmatched libraries fall "
                        "back to a built-in palette.")
    p.add_argument("--motif_baseline_csv", default=None,
                   help="A *_context_editing.csv produced by "
                        "editEfficiencyFromParquet.py for these same "
                        "libraries (columns: label, context, n_edited, "
                        "n_total, edit_rate). Used as each library's own "
                        "3nt-motif baseline editing rate for the second, "
                        "motif-normalized meta plot. If omitted, that "
                        "second plot is skipped.")
    return p.parse_args()


def main():
    global PALETTE, COLOR_MAP
    args = parse_args()
    out  = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print("=== Histidine Meta Analysis, all libraries overlaid (parquet) ===",
          file=sys.stderr)

    from pyx import color as pyx_color
    PALETTE = [
        pyx_color.cmyk(0, 0, 0, 1),        # black
        pyx_color.cmyk(1, 0.5, 0, 0),      # blue
        pyx_color.cmyk(0, 1, 1, 0),        # red
        pyx_color.cmyk(0.4, 1, 0, 0),      # purple
        pyx_color.cmyk(0, 0.5, 1, 0),      # orange
        pyx_color.cmyk(0.7, 0, 0, 0),      # cyan
        pyx_color.cmyk(0, 0, 0, 0.5),      # grey
        pyx_color.cmyk(0.3, 0, 1, 0.2),    # olive
    ]
    COLOR_MAP = load_color_map(args.color_map) if args.color_map else {}

    libs = parse_libs_file(args.libs)
    if not libs:
        print(f"No libraries found in {args.libs}; exiting.", file=sys.stderr)
        sys.exit(1)
    if args.color_map:
        unmatched = [lib for lib, _ in libs if lib not in COLOR_MAP]
        if unmatched:
            print(f"  WARNING: no color found in {args.color_map} for "
                  f"librar{'y' if len(unmatched)==1 else 'ies'} {unmatched}; "
                  f"falling back to the default palette.", file=sys.stderr)

    motif_baseline_by_label = load_motif_baseline_csv(args.motif_baseline_csv) \
                              if args.motif_baseline_csv else None
    if args.motif_baseline_csv:
        unmatched = [lib for lib, _ in libs
                     if not motif_baseline_for_label(motif_baseline_by_label, lib)]
        if unmatched:
            print(f"  WARNING: no motif baseline found in "
                  f"{args.motif_baseline_csv} for "
                  f"librar{'y' if len(unmatched)==1 else 'ies'} {unmatched}; "
                  f"the motif-normalized plot will skip {'it' if len(unmatched)==1 else 'them'}.",
                  file=sys.stderr)

    print("\nParsing GTF and finding His codon sites...", file=sys.stderr)
    ref_fasta    = pysam.FastaFile(args.ref)
    cds_by_chrom = parse_gtf_cds(args.gtf)
    sites        = find_his_positions(ref_fasta, cds_by_chrom, args.window)
    _attach_win_seqs(ref_fasta, sites)

    rel_agg_by_lib       = {}
    rel_agg_motif_by_lib = {}
    for label, parquet_dir in libs:
        print(f"\nStreaming {parquet_dir} ({label})...", file=sys.stderr)
        result = process_library_streaming(parquet_dir, sites, args.min_coverage,
                                            label, columns=REQUIRED_COLUMNS)
        n_total, n_rev = result["n_total"], result["n_rev"]
        n_fwd   = n_total - n_rev
        pct_rev = 100 * n_rev / n_total if n_total > 0 else 0.0
        print(f"  {label}: {n_total:,} reads  |  "
              f"forward: {n_fwd:,} ({100-pct_rev:.1f}%)  |  "
              f"reverse (minus-strand gene): {n_rev:,} ({pct_rev:.1f}%)",
              file=sys.stderr)

        agg_df = result["agg_df"]
        agg_df.to_csv(f"{out}_{label}_agg.csv.gz", index=False, compression="gzip")
        rel_agg_by_lib[label] = transcript_normalised_agg(agg_df) \
                                if not agg_df.empty else pd.DataFrame()

        if motif_baseline_by_label is not None:
            motif_freqs = motif_baseline_for_label(motif_baseline_by_label, label)
            rel_agg_motif_by_lib[label] = motif_normalized_agg(agg_df, motif_freqs) \
                                          if motif_freqs and not agg_df.empty \
                                          else pd.DataFrame()

    print("\nGenerating plots...", file=sys.stderr)
    try:
        plot_meta_all_libraries_pyx(rel_agg_by_lib, args.window, out)
    except Exception as e:
        print(f"  WARNING: pyx plotting failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    if motif_baseline_by_label is not None:
        try:
            plot_meta_all_libraries_motif_normalized_pyx(rel_agg_motif_by_lib, args.window, out)
        except Exception as e:
            print(f"  WARNING: motif-normalized pyx plotting failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

    ref_fasta.close()
    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    Tee()
    main()
