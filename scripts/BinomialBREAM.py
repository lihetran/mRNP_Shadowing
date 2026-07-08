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
                cds_spanning: bool = False,
                min_edit_freq: float = 0.0) -> pd.DataFrame:
    """
    Fast vectorised pre-filter to reads overlapping this gene.

    If cds_spanning is True, only keep reads whose alignment spans the full
    CDS (read_start <= cds_genomic_start and read_end >= cds_genomic_end),
    so every read had the opportunity to be edited at every position.

    If min_edit_freq > 0, only keep reads whose global_edit_freq column
    (per-read A->G edit fraction) is >= min_edit_freq. Reads below this
    threshold are likely unedited / poorly edited molecules that carry no
    protection signal and only add noise.
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
    if min_edit_freq > 0.0 and "global_edit_freq" in df_all.columns:
        mask &= (df_all["global_edit_freq"] >= min_edit_freq)
    return df_all[mask]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Build gpos -> tx_pos map and ref frequencies
#    Uses absolute_indices + edit_string (sense-oriented, no aligned_pairs)
# ─────────────────────────────────────────────────────────────────────────────

def _full_tx_map(gene: dict, ref_fasta: pysam.FastaFile) -> dict:
    """
    Map tx_pos -> (gpos, ref_base_sense) for EVERY CDS position (all bases,
    not just A). ref_base_sense is the transcript-sense reference base
    (reverse-complemented for minus-strand genes), so 'A' marks editable sites.
    Used to emit the full nucleotide context within a shadow span.
    """
    chrom     = gene["chrom"]
    strand    = gene["strand"]
    chrom_seq = ref_fasta.fetch(chrom).upper()

    full = {}
    tx_pos = 0
    for (cs, ce) in gene["cds"]:
        if strand == "+":
            for gpos in range(cs, ce):
                full[tx_pos] = (gpos, chrom_seq[gpos])
                tx_pos += 1
        else:
            for gpos in range(ce - 1, cs - 1, -1):
                base = complement_base(chrom_seq[gpos])
                full[tx_pos] = (gpos, base)
                tx_pos += 1
    return full


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


def build_reference_freq_and_coverage(df: pd.DataFrame, gpos_to_tx: dict,
                                       gene: dict) -> tuple:
    """
    Like build_reference_freq but also returns per-tx-position coverage:
    the number of reference reads contributing an A or G call at each
    tx position (= the denominator of the background estimate).

    Returns (ref_freq, ref_cov) where:
      ref_freq = {tx_pos: p_edit}
      ref_cov  = {tx_pos: n_reads_with_AG_call}
    """
    if df.empty:
        return {}, {}

    min_gp = min(gpos_to_tx.keys(), default=None)
    max_gp = max(gpos_to_tx.keys(), default=None)
    if min_gp is None:
        return {}, {}

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
    ref_cov  = {}
    for tx, (n0, n1) in edit_counts.items():
        total = n0 + n1
        if total > 0:
            ref_freq[tx] = max(1e-6, min(1 - 1e-6, n1 / total))
            ref_cov[tx]  = total
    return ref_freq, ref_cov


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

def compute_binomial_pvals_per_read2(read_edits: dict, ref_freq: dict,
                                     ref_cov: dict,
                                     nt_window: int, min_sites: int,
                                     gene_len: int,
                                     tx_to_gpos: dict = None) -> tuple:
    """
    Sliding-window lower-tail binomial protection test per read.

    Returns (results, per_site) where:

      results  = {read_id: [(window_centre, -log10(p), window_coverage), ...]}
                 window_coverage is the MEAN background reference coverage
                 across the ref=A sites the read covers in that window.

      per_site = {read_id: [ {tx_pos, gpos, edit, ref_freq, ref_cov}, ... ]}
                 one entry per ref=A site the read covers (edit: 1 = edited G,
                 0 = unedited A). gpos is the genomic reference position if
                 tx_to_gpos is provided, else None.

    The window trace and the per-site data are kept separate (option 2) so the
    per-window aggregate isn't cluttered with variable-length site lists.
    """
    results  = {}
    per_site = {}

    for read_id, pos_dict in read_edits.items():
        trace = []
        for start in range(0, gene_len):
            ks, ps, cs = [], [], []
            for tx in range(start, start + nt_window):
                if tx in pos_dict and tx in ref_freq:
                    ks.append(pos_dict[tx])
                    ps.append(ref_freq[tx])
                    cs.append(ref_cov.get(tx, 0))

            n = len(ks)
            if n < min_sites:
                continue

            k      = sum(ks)
            c      = float(np.mean(cs)) if cs else 0.0
            p_mean = float(np.mean(ps))
            p_val  = max(scipy.stats.binom.cdf(k, n, p_mean), 1e-300)

            trace.append((start + nt_window / 2., -math.log10(p_val), c))

        if trace:
            results[read_id] = trace

            # Per-site edit status and reference position for this read.
            site_rows = []
            for tx in sorted(pos_dict.keys()):
                if tx not in ref_freq:
                    continue
                site_rows.append({
                    "tx_pos":   tx,
                    "gpos":     tx_to_gpos.get(tx) if tx_to_gpos else None,
                    "edit":     pos_dict[tx],
                    "ref_freq": ref_freq[tx],
                    "ref_cov":  ref_cov.get(tx, 0),
                })
            per_site[read_id] = site_rows

    return results, per_site


# ─────────────────────────────────────────────────────────────────────────────
# 7. Meta aggregation
# ─────────────────────────────────────────────────────────────────────────────

def get_meta_read(binomial_scores: dict) -> list:
    by_pos = collections.defaultdict(list)
    for trace in binomial_scores.values():
        for centre, v, _cov in trace:
            by_pos[centre].append(v)

    return [
        (centre,
         float(np.quantile(vals, 0.05)),
         float(np.quantile(vals, 0.50)),
         float(np.quantile(vals, 0.95)))
        for centre, vals in sorted(by_pos.items())
    ]


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.

    Given an array of raw p-values, returns an array of adjusted p-values
    (q-values) of the same length and order. A window is significant at
    FDR level q if its adjusted p-value <= q.

    Method:
      1. Rank p-values ascending: p_(1) <= p_(2) <= ... <= p_(m)
      2. Adjusted p_(i) = p_(i) * m / i
      3. Enforce monotonicity from the largest rank downward so adjusted
         values never decrease as raw p increases.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p

    order      = np.argsort(p)              # indices that sort p ascending
    ranked     = p[order]
    ranks      = np.arange(1, m + 1)
    adj_ranked = ranked * m / ranks

    # Enforce monotonicity: walk from largest to smallest, take running min
    adj_ranked = np.minimum.accumulate(adj_ranked[::-1])[::-1]
    adj_ranked = np.clip(adj_ranked, 0, 1)

    # Scatter back to original order
    adj = np.empty(m, dtype=float)
    adj[order] = adj_ranked
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# 8. Plotting
# ─────────────────────────────────────────────────────────────────────────────

def write_binomial_scores_to_df(gene_name, binomial_scores,
                                pval_threshold=0.05):
    """
    Flatten a gene's significant windows to a DataFrame — one row per
    significant (p <= threshold) window per read. Window-level (not per-site).
    Index-based trace access so it tolerates 3+/-element trace points
    (centre, -log10(p), coverage[, ...]).

    Columns: gene_name, read_id, window_centre, p_value, neg_log10p, coverage
    """
    rows = []
    for read_id, trace in binomial_scores.items():
        for point in trace:
            centre    = point[0]
            neg_log_p = point[1]
            cov       = point[2] if len(point) > 2 else None
            p_value   = 10 ** (-neg_log_p)
            if p_value > pval_threshold:
                continue
            rows.append({
                "gene_name":     gene_name,
                "read_id":       read_id,
                "window_centre": centre,
                "p_value":       p_value,
                "neg_log10p":    neg_log_p,
                "coverage":      cov,
            })
    return pd.DataFrame(rows)


def write_shadow_calls_to_df(gene_name, strand, binomial_scores,
                             read_edits, ref_cov, ref_freq, full_tx_map,
                             nt_window, pval_threshold=0.05):
    """
    Full-read shadow-call table for spot-checking, in genomic coordinates.

    For each read with at least one significant (p <= threshold) window, emits
    ONE ROW PER TRANSCRIPT POSITION across the read's ENTIRE covered extent
    (first to last covered ref=A site) — every base, not just ref=A sites, and
    not just the shadow. The shadow region is flagged with in_shadow=True, so
    scanning a read top-to-bottom shows: context -> shadow start -> shadow ->
    shadow end -> context, making the shadow boundaries obvious.

    Columns:
      gene_name, strand, read_id, tx_pos, gpos, ref_base, is_editable,
      covered, edit, ref_cov, ref_freq, in_shadow, best_p, best_neg_log10p

    Notes:
      - covered: True if the read observed this ref=A site (has an edit call).
      - edit: 1/0 at covered ref=A sites, NA at non-A positions or A sites the
        read didn't cover.
      - ref_cov / ref_freq: only defined at ref=A sites (the background model
        is A-only); NA elsewhere.
      - in_shadow: True if the position falls inside any significant window.
    """
    rows = []

    for read_id, trace in binomial_scores.items():
        pos_dict = read_edits.get(read_id, {})
        if not pos_dict:
            continue

        # best (smallest) sig p covering each transcript position
        best_p = {}
        has_sig = False
        for point in trace:
            centre    = point[0]
            neg_log_p = point[1]
            p_value   = 10 ** (-neg_log_p)
            if p_value > pval_threshold:
                continue
            has_sig = True
            lo = int(math.ceil(centre - nt_window / 2.0))
            hi = int(math.floor(centre + nt_window / 2.0 - 1e-9))
            for tx in range(lo, hi + 1):
                if tx in full_tx_map:
                    if tx not in best_p or p_value < best_p[tx]:
                        best_p[tx] = p_value

        # Only emit reads that actually have a shadow
        if not has_sig:
            continue

        # Read extent = first to last covered ref=A site (transcript order)
        covered_tx = sorted(pos_dict.keys())
        span_lo = covered_tx[0]
        span_hi = covered_tx[-1]

        # Emit every transcript position across the read's covered extent
        for tx in range(span_lo, span_hi + 1):
            if tx not in full_tx_map:
                continue
            gpos, ref_base = full_tx_map[tx]
            is_editable = (ref_base == "A")
            covered     = is_editable and (tx in pos_dict)
            p = best_p.get(tx)

            edit_val = int(pos_dict[tx]) if covered else None

            rows.append({
                "gene_name":       gene_name,
                "strand":          strand,
                "read_id":         read_id,
                "tx_pos":          tx,
                "gpos":            gpos,
                "ref_base":        ref_base,
                "is_editable":     is_editable,
                "covered":         covered,
                "edit":            edit_val,
                "ref_cov":         ref_cov.get(tx) if is_editable else None,
                "ref_freq":        ref_freq.get(tx) if is_editable else None,
                "in_shadow":       p is not None,
                "best_p":          p,
                "best_neg_log10p": (-math.log10(p) if p is not None else None),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["read_id", "gpos"]).reset_index(drop=True)
    return df


def plot_gene_pyx(gene_name, meta, binomial_scores, his_positions,
                   read_edits, label1, label2, gene_len, num_reads, pdf_path,
                   ref_cov=None):
    from pyx import canvas, graph, color, style, text as pyx_text

    col_qry  = color.cmyk(1, 0.5, 0, 0)
    col_his  = color.cmyk(0, 1, 1, 0)
    col_sig  = color.cmyk(0, 0, 0, 0.4)
    col_edit = color.cmyk(0, 0, 0, 1)
    col_no   = color.cmyk(0, 0.8, 1, 0)
    col_cov  = color.cmyk(0.7, 0, 0.7, 0.1)   # green — background coverage

    x_min, x_max = 0, gene_len
    panel_w  = 12
    tick_h   = 0.25
    read_h   = 1.2
    gap      = 0.6
    cov_h    = 1.5
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

    # ── Background coverage panel above the meta plot ─────────────────────────
    cov_ypos = meta_ypos + 3 + gap
    title_y  = cov_ypos + 3 + 0.4   # default if no coverage panel drawn

    if ref_cov:
        cov_items = sorted(ref_cov.items())
        cov_x     = [tx for tx, _ in cov_items]
        cov_y     = [n  for _, n  in cov_items]
        cov_y_max = max(cov_y) * 1.1 if cov_y else 1.0

        g_cov = graph.graphxy(
            width=panel_w, height=cov_h,
            xpos=0, ypos=cov_ypos,
            x=graph.axis.linkedaxis(g_meta.axes["x"]),
            y=graph.axis.linear(min=0, max=cov_y_max,
                                title="Ref cov"),
        )
        # His codon lines on the coverage panel too
        for hp in his_positions:
            g_cov.plot(
                graph.data.function(f"x(y)={hp}", min=0, max=cov_y_max),
                [graph.style.line([col_his, style.linewidth.thin,
                                   style.linestyle.solid])])
        # Coverage as filled impulses / line
        if len(cov_items) > 0:
            g_cov.plot(
                graph.data.points(list(zip(cov_x, cov_y)), x=1, y=2),
                [graph.style.line([col_cov, style.linewidth.normal,
                                   style.linestyle.solid])])
        c.insert(g_cov)
        title_y = g_cov.ypos + g_cov.height + 0.4

    c.text(g_meta.xpos + g_meta.width / 2., title_y,
           f"{gene_name} - {label2} binomial protection (ref: {label1})",
           [pyx_text.halign.center, pyx_text.size.normalsize])

    read_ids = list(binomial_scores.keys())[:num_reads]
    for jj, read_id in enumerate(read_ids):
        trace = binomial_scores[read_id]
        # Each trace point is (center, -log10(p), window_coverage).
        ypos  = (num_reads - 1 - jj) * (read_h + tick_h + gap)
        y_max_read = max(max(v for _, v, _ in trace) * 1.1, 2.0)

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
        # Plot p-value trace (x = center, y = -log10(p)); drop coverage col
        g_read.plot(
            graph.data.points([(t[0], t[1]) for t in trace], x=1, y=2),
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

        # Mean window coverage across this read's trace, shown in the label
        covs = [t[2] for t in trace if t[2] is not None]
        label_txt = read_id[:20]
        if covs:
            label_txt += f" (cov={np.mean(covs):.0f})"
        c.text(panel_w + 0.15, ypos + tick_h + read_h / 2.,
               label_txt, [pyx_text.valign.middle, pyx_text.size.tiny])

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
    p.add_argument("--min_query_reads", type=int, default=10,
                   help="Minimum number of query (spanning) reads a gene must "
                        "have to be analyzed (default: 10). Separate from "
                        "--min_coverage, which gates the reference background.")
    p.add_argument("--min_sites",    type=int,   default=5)
    p.add_argument("--num_reads",    type=int,   default=10)
    p.add_argument("--top_n_plots",  type=int,   default=10,
                   help="Number of genes to plot, ranked by number of "
                        "CDS-spanning query reads (default: 10)")
    p.add_argument("--gene_list",    default=None)
    p.add_argument("--require_his_codon", action="store_true",
                   help="Only process genes that contain at least one His "
                        "codon (CAT/CAC).")
    p.add_argument("--background_all_reads", action="store_true",
                   help="When --cds_spanning is set, build the reference "
                        "background frequencies from ALL overlapping "
                        "reference reads (not just spanning ones) for a more "
                        "precise per-position estimate. Query reads are still "
                        "restricted to spanning. No effect without "
                        "--cds_spanning.")
    p.add_argument("--min_edit_freq", type=float, default=0.0,
                   help="Only consider query reads whose per-read "
                        "global_edit_freq is >= this value. Filters out "
                        "unedited/poorly-edited molecules that carry no "
                        "protection signal. Applied to query reads only, "
                        "not the background estimate. Default: 0.0 (off)")
    p.add_argument("--fdr_correction", action="store_true",
                   help="Apply Benjamini-Hochberg FDR correction across all "
                        "window p-values and add a frac_sig_windows_fdr "
                        "column to the summary.")
    p.add_argument("--fdr_level", type=float, default=0.05,
                   help="FDR q-value threshold for --fdr_correction "
                        "(default: 0.05)")
    p.add_argument("--cds_spanning", action="store_true",
                   help="Only include reads whose alignment spans the full "
                        "CDS of the gene.")
    return p.parse_args()


def main():
    args = parse_args()
    out  = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    # pyarrow for incremental parquet writing (memory-flat)
    import pyarrow as pa
    import pyarrow.parquet as pq

    print("=== Binomial Shadowing Analysis (parquet) ===", file=sys.stderr)
    print(f"  Coverage gate: background >= {int(args.min_coverage)} reads, "
          f"query >= {args.min_query_reads} spanning reads", file=sys.stderr)
    if args.cds_spanning:
        print("  CDS spanning filter: ON", file=sys.stderr)
    if args.require_his_codon:
        print("  His codon filter: ON (genes need >=1 CAT/CAC)",
              file=sys.stderr)
    if args.cds_spanning and args.background_all_reads:
        print("  Background estimate: ALL overlapping reference reads "
              "(query restricted to spanning)", file=sys.stderr)
    if args.min_edit_freq > 0.0:
        print(f"  Query edit-freq filter: ON "
              f"(global_edit_freq >= {args.min_edit_freq})", file=sys.stderr)

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
    print(f"\nPass 1: scanning {len(gene_names):,} genes "
          f"(summary + ranking + results parquet)...", file=sys.stderr)

    # Diagnostics for spanning filter
    max_spanning_ref = 0
    max_spanning_qry = 0
    genes_with_any_spanning = 0

    # Pass 1: compute summary for every passing gene, record spanning counts,
    # and write per-window results incrementally. Keeps memory flat by writing
    # each gene's rows straight to parquet and discarding heavy objects.
    gene_span_counts = {}   # gname -> n_spanning_qry (ranking metric)

    # For optional global BH FDR correction
    global_pvals = []       # flat list of all window p-values
    gene_pval_slices = {}   # gname -> (start_idx, end_idx) into global_pvals

    # Incremental results-parquet writer (created lazily on first non-empty
    # gene DataFrame so the schema is fixed from real data).
    results_parquet = f"{out}_results.parquet"
    results_writer  = None

    for i, gname in enumerate(gene_names):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(gene_names)} scanned, "
                  f"{n_pass} passing...", file=sys.stderr)

        gene = genes[gname]
        gene_len = cds_length(gene)

        # His codon filter — skip genes with no His codons if requested
        his_positions = find_his_codon_tx_positions(ref_fasta, gene)
        if args.require_his_codon and len(his_positions) == 0:
            continue

        # Reference background: ALWAYS all overlapping reads (never spanning),
        # so the reference distribution has maximum per-position support.
        # The edit-freq filter is NOT applied to the background.
        df_ref_bg = get_gene_df(df_all_ref, gene, cds_spanning=False)

        # Query: reads we make protection calls on — respect --cds_spanning
        # and the edit-freq filter.
        df_qry = get_gene_df(df_all_qry, gene, cds_spanning=args.cds_spanning,
                             min_edit_freq=args.min_edit_freq)

        # Also track spanning reference reads for diagnostics only.
        df_ref = get_gene_df(df_all_ref, gene, cds_spanning=args.cds_spanning)

        if len(df_ref_bg) > 0 and len(df_qry) > 0:
            genes_with_any_spanning += 1
        max_spanning_ref = max(max_spanning_ref, len(df_ref))
        max_spanning_qry = max(max_spanning_qry, len(df_qry))

        # Coverage gate (two independent thresholds):
        #   - reference background must have >= min_coverage reads
        #     (all overlapping reads; does NOT require CDS spanning)
        #   - query must have >= min_query_reads spanning reads
        if len(df_ref_bg) < args.min_coverage:
            continue
        if len(df_qry) < args.min_query_reads:
            continue
        n_pass += 1

        gpos_to_tx = _gpos_to_tx_map(gene, ref_fasta)
        if not gpos_to_tx:
            continue
        tx_to_gpos = {tx: gp for gp, tx in gpos_to_tx.items()}

        ref_freq, ref_cov = build_reference_freq_and_coverage(
            df_ref_bg, gpos_to_tx, gene)
        if not ref_freq:
            continue

        read_edits = collect_read_edits(df_qry, gpos_to_tx, gene)
        if not read_edits:
            continue

        binomial_scores, per_site = compute_binomial_pvals_per_read2(
            read_edits, ref_freq, ref_cov,
            nt_window=args.window,
            min_sites=args.min_sites,
            gene_len=gene_len,
            tx_to_gpos=tx_to_gpos,
        )
        if not binomial_scores:
            continue

        all_scores = [point[1] for trace in binomial_scores.values()
                      for point in trace]
        sig_line   = -math.log10(0.05)
        frac_sig   = (sum(1 for v in all_scores if v >= sig_line)
                      / len(all_scores)) if all_scores else 0.0

        n_spanning_qry = len(df_qry)
        gene_span_counts[gname] = n_spanning_qry

        # For optional global BH FDR: record this gene's window p-values and
        # the index range they occupy in the global pool.
        if args.fdr_correction:
            gene_pvals = [10 ** (-v) for v in all_scores]
            start_idx  = len(global_pvals)
            global_pvals.extend(gene_pvals)
            gene_pval_slices[gname] = (start_idx, len(global_pvals))

        summary_rows.append({
            "gene":              gname,
            "n_reads":           len(binomial_scores),
            "n_his_codons":      len(his_positions),
            "gene_len":          gene_len,
            "n_ref_a_sites":     len(ref_freq),
            "n_reads_ref_bg":    len(df_ref_bg),
            "n_reads_ref_span":  len(df_ref),
            "n_reads_qry":       n_spanning_qry,
            "median_neg_log10p": float(np.median(all_scores)),
            "frac_sig_windows":  frac_sig,
        })

    #     # Write this gene's significant-window rows straight to parquet.
    #     gene_df = write_binomial_scores_to_df(gname, binomial_scores)
    #     if not gene_df.empty:
    #         table = pa.Table.from_pandas(gene_df, preserve_index=False)
    #         if results_writer is None:
    #             results_writer = pq.ParquetWriter(results_parquet,
    #                                               table.schema)
    #         results_writer.write_table(table)
    #
    #     # Free heavy objects immediately — not needed until pass 2
    #     del binomial_scores, per_site, read_edits, ref_freq, gpos_to_tx, \
    #         ref_cov, gene_df
    #
    # # Close the incremental results writer
    # if results_writer is not None:
    #     results_writer.close()
    #     print(f"\n  Results -> {results_parquet}", file=sys.stderr)
    # else:
    #     print("\n  No significant windows written to results parquet.",
    #           file=sys.stderr)

    # ── Pass 2: recompute and plot only the top-N genes by spanning reads ─────
    shadow_call_frames = []   # per-gene shadow-call tables (spot-checking)
    if gene_span_counts:
        top_genes = sorted(gene_span_counts.keys(),
                           key=lambda g: gene_span_counts[g],
                           reverse=True)

        print(f"\nPass 2: plotting top {len(top_genes)} genes "
              f"by CDS-spanning reads...", file=sys.stderr)

        for gname in top_genes:
            gene     = genes[gname]
            gene_len = cds_length(gene)
            print(f"    {gname}: {gene_span_counts[gname]:,} spanning reads",
                  file=sys.stderr)

            # Reference background: always all overlapping reads (matches
            # pass 1). Query: spanning + edit-freq filter.
            df_ref_bg = get_gene_df(df_all_ref, gene, cds_spanning=False)
            df_qry = get_gene_df(df_all_qry, gene,
                                 cds_spanning=args.cds_spanning,
                                 min_edit_freq=args.min_edit_freq)

            gpos_to_tx = _gpos_to_tx_map(gene, ref_fasta)
            tx_to_gpos_map = {tx: gp for gp, tx in gpos_to_tx.items()}
            full_tx_map = _full_tx_map(gene, ref_fasta)
            ref_freq, ref_cov = build_reference_freq_and_coverage(
                df_ref_bg, gpos_to_tx, gene)
            read_edits = collect_read_edits(df_qry, gpos_to_tx, gene)

            binomial_scores, _per_site = compute_binomial_pvals_per_read2(
                read_edits, ref_freq, ref_cov,
                nt_window=args.window,
                min_sites=args.min_sites,
                gene_len=gene_len,
                tx_to_gpos=tx_to_gpos_map,
            )
            if not binomial_scores:
                continue

            meta = get_meta_read(binomial_scores)
            his_positions = find_his_codon_tx_positions(ref_fasta, gene)
            safe_name = re.sub(r"[^\w\-]", "_", gname)

            print(f"{gname}: {len(his_positions)} His codon(s) "
                  f"at tx positions {his_positions[:10]}"
                  f"{'...' if len(his_positions) > 10 else ''}",
                  file=sys.stderr)

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
                    ref_cov=ref_cov,
                )
            except Exception as e:
                print(f"WARNING: pyx plot failed for {gname}: {e}",
                      file=sys.stderr)

            # Full-nucleotide shadow-call table (spot-checking, genomic coords)
            sc_df = write_shadow_calls_to_df(
                gname, gene["strand"], binomial_scores,
                read_edits, ref_cov, ref_freq, full_tx_map,
                nt_window=args.window, pval_threshold=0.05,
            )
            if not sc_df.empty:
                shadow_call_frames.append(sc_df)

    ref_fasta.close()

    # Write the combined shadow-call table for the plotted genes
    if shadow_call_frames:
        shadow_calls_df = pd.concat(shadow_call_frames, ignore_index=True)
        shadow_calls_path = f"{out}_shadow_calls.parquet"
        shadow_calls_df.to_parquet(shadow_calls_path, index=False)
        print(f"\n  Shadow calls -> {shadow_calls_path} "
              f"({len(shadow_calls_df):,} site rows)", file=sys.stderr)

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

    summary_df = pd.DataFrame(summary_rows)

    # ── Optional global Benjamini-Hochberg FDR correction ─────────────────────
    if args.fdr_correction and global_pvals:
        print(f"\nApplying Benjamini-Hochberg FDR correction across "
              f"{len(global_pvals):,} windows (q = {args.fdr_level})...",
              file=sys.stderr)
        adj = benjamini_hochberg(np.array(global_pvals))

        frac_fdr = {}
        for gname, (s, e) in gene_pval_slices.items():
            gene_adj = adj[s:e]
            frac_fdr[gname] = (float((gene_adj <= args.fdr_level).mean())
                               if len(gene_adj) else 0.0)

        summary_df["frac_sig_windows_fdr"] = summary_df["gene"].map(
            frac_fdr).fillna(0.0)

        n_sig_total = int((adj <= args.fdr_level).sum())
        print(f"  {n_sig_total:,}/{len(adj):,} windows significant at "
              f"FDR q <= {args.fdr_level}.", file=sys.stderr)

    summary_df = summary_df.sort_values("median_neg_log10p", ascending=False)
    summary_csv = f"{out}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n  Summary -> {summary_csv}", file=sys.stderr)
    print(f"  Gene PDFs -> {pdf_dir}/", file=sys.stderr)
    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    Tee()
    main()