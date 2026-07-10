'''
July 9, 2026

For each high-coverage gene, position a window on a covered A position on the transcript and
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
        --parquet1 reference_chunks/ --label1 "ribosome-less" \
        --parquet2 query_chunks/     --label2 "3-AT" \
        --ref reference.fa \
        --gtf annotation.gtf \
        --output output_prefix \
        [--window 30] \
        [--min_coverage 50] \
        [--min_sites 5] \
        [--num_reads 10] \

'''

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

def complement_base(b: str) -> str:
    return b.translate(str.maketrans("ACGTacgt", "TGCAtgca"))

def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]

def parse_gtf(gtf_path: str) -> dict:
    """
    Parse a GTF carrying only `exon` and `CDS` features (plus optional `gene`).

    UTRs are derived as exon - CDS in genomic space, then assigned to 5'/3' by
    strand. Each gene is pinned to a single transcript (the first encountered)
    so isoforms are never blended into one coordinate space.

    Returns {gene_name: {
        chrom, strand, transcript, gene_name,
        cds, exons, utr5, utr3,          # (start, end) lists, transcript order
        gene_start, gene_end,
        cds_genomic_start, cds_genomic_end,
    }}
    """
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

            m_gn  = re.search(r'gene_name "([^"]+)"', fields[8])
            m_tid = re.search(r'transcript_id "([^"]+)"', fields[8])
            gname = m_gn.group(1)  if m_gn  else None
            tid   = m_tid.group(1) if m_tid else "."
            if gname is None:
                continue

            if feature == "gene":
                gene_extents[gname] = (start, end)
                continue

            if feature not in ("CDS", "exon"):
                continue

            if gname not in genes:
                genes[gname] = {
                    "chrom": chrom, "strand": strand,
                    "transcript": tid, "gene_name": gname,
                    "cds": [], "exons": [],
                }
            elif genes[gname]["transcript"] != tid:
                continue                 # pin to first transcript only

            key = "cds" if feature == "CDS" else "exons"
            genes[gname][key].append((start, end))

    drop = []
    for gname, g in genes.items():
        if not g["cds"]:
            drop.append(gname)           # non-coding: no CDS to anchor on
            continue

        if gname in gene_extents:
            g["gene_start"], g["gene_end"] = gene_extents[gname]
        else:
            spans = g["exons"] or g["cds"]
            g["gene_start"] = min(s for s, e in spans)
            g["gene_end"]   = max(e for s, e in spans)

        g["cds_genomic_start"] = min(s for s, e in g["cds"])
        g["cds_genomic_end"]   = max(e for s, e in g["cds"])

        # UTRs = exon - CDS, split by genomic side, then assigned by strand
        utr = _subtract_intervals(g["exons"], g["cds"])
        left  = [iv for iv in utr if iv[1] <= g["cds_genomic_start"]]
        right = [iv for iv in utr if iv[0] >= g["cds_genomic_end"]]

        if g["strand"] == "+":
            g["utr5"], g["utr3"] = left, right
        else:
            g["utr5"], g["utr3"] = right, left

        # Sort every segment list into transcript order
        rev = (g["strand"] == "-")
        for key in ("cds", "exons", "utr5", "utr3"):
            g[key].sort(key=lambda x: x[0], reverse=rev)

    for gname in drop:
        del genes[gname]

    return genes

def cds_length(gene: dict) -> int:
    return sum(ce - cs for cs, ce in gene["cds"])

def _subtract_intervals(exons, cds):
    """
    exons, cds: lists of (start, end) genomic half-open intervals, unsorted.
    Returns the parts of exons not covered by any cds interval, sorted by start.
    """
    if not cds:
        return sorted(exons)

    cds = sorted(cds)
    out = []
    for (es, ee) in sorted(exons):
        cur = es
        for (cs, ce) in cds:
            if ce <= cur or cs >= ee:
                continue                  # no overlap with the remaining piece
            if cs > cur:
                out.append((cur, cs))     # piece before this CDS block
            cur = max(cur, ce)
            if cur >= ee:
                break
        if cur < ee:
            out.append((cur, ee))         # trailing piece
    return out


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
    mask = ((df_all["chrom"] == gene["chrom"]) & (df_all["gene_strand"] == gene["strand"]))
    if "read_start" in df_all.columns and "read_end" in df_all.columns:
        if cds_spanning:
            cds_start = gene.get("cds_genomic_start", gene["gene_start"])
            cds_end = gene.get("cds_genomic_end",   gene["gene_end"])
            mask &= ((df_all["read_start"] <= cds_start) & (df_all["read_end"]   >= cds_end))
        else:
            mask &= ((df_all["read_start"] < gene["gene_end"]) & (df_all["read_end"] > gene["gene_start"]))
    if min_edit_freq > 0.0 and "global_edit_freq" in df_all.columns:
        mask &= (df_all["global_edit_freq"] >= min_edit_freq)
    return df_all[mask]

def _full_tx_map(gene: dict, ref_fasta: pysam.FastaFile,
                 include_utrs: bool = True) -> dict:
    """
    Map tx_pos -> (gpos, ref_base_sense) for EVERY transcript position (all
    bases, not just A). ref_base_sense is the transcript-sense reference base
    (complemented for minus-strand genes), so 'A' marks editable sites.

    Coordinates are CDS-relative: the first CDS base is tx_pos 0, 5'UTR
    positions are negative, 3'UTR positions are >= cds_length(gene).
    """
    chrom_seq = ref_fasta.fetch(gene["chrom"]).upper()
    strand    = gene["strand"]

    def _walk(segments, tx_start):
        """Yield (tx_pos, gpos, sense_base) in transcript order."""
        tx = tx_start
        for (cs, ce) in segments:
            rng = range(cs, ce) if strand == "+" else range(ce - 1, cs - 1, -1)
            for gpos in rng:
                base = chrom_seq[gpos]
                if strand == "-":
                    base = complement_base(base)
                yield tx, gpos, base
                tx += 1

    full = {}
    for tx, gpos, base in _walk(gene["cds"], 0):
        full[tx] = (gpos, base)

    if include_utrs:
        cds_len = cds_length(gene)
        for tx, gpos, base in _walk(gene.get("utr3", []), cds_len):
            full[tx] = (gpos, base)

        # 5'UTR: walk from 0 in transcript order, then shift so it ends at -1
        u5 = list(_walk(gene.get("utr5", []), 0))
        n5 = len(u5)
        for tx, gpos, base in u5:
            full[tx - n5] = (gpos, base)

    return full


def _gpos_to_tx_map(gene, ref_fasta, include_utrs=True):
    chrom_seq = ref_fasta.fetch(gene["chrom"]).upper()
    strand    = gene["strand"]
    want      = "A" if strand == "+" else "T"

    def _walk(segments, tx_start):
        """Yield (gpos, tx_pos) in transcript order, starting at tx_start."""
        tx = tx_start
        for (cs, ce) in segments:
            rng = range(cs, ce) if strand == "+" else range(ce - 1, cs - 1, -1)
            for gpos in rng:
                yield gpos, tx
                tx += 1

    out = {}
    for gpos, tx in _walk(gene["cds"], 0):
        if chrom_seq[gpos] == want:
            out[gpos] = tx

    if include_utrs:
        cds_len = cds_length(gene)
        for gpos, tx in _walk(gene["utr3"], cds_len):
            if chrom_seq[gpos] == want:
                out[gpos] = tx
        # 5'UTR: walk in transcript order, then offset so it ends at -1
        u5 = list(_walk(gene["utr5"], 0))
        n5 = len(u5)
        for gpos, tx in u5:
            if chrom_seq[gpos] == want:
                out[gpos] = tx - n5

    return out

def classify_tx(tx_pos, cds_len):
    if tx_pos < 0:            return "UTR5"
    if tx_pos < cds_len:      return "CDS"
    return "UTR3"

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
    gene_end = gene["gene_end"]
    min_gp = min(gpos_to_tx.keys(), default=None)
    max_gp = max(gpos_to_tx.keys(), default=None)
    if min_gp is None:
        return {}

    # Pre-filter to reads spanning ref=A positions
    if "read_start" in df.columns and "read_end" in df.columns:
        sub = df[(df["read_start"] <= max_gp) & (df["read_end"] >= min_gp)]
    else:
        sub = df

    edit_counts = collections.defaultdict(lambda: [0, 0])

    for read in sub.itertuples():
        edit_str = read.edit_string
        abs_indices = read.absolute_indices
        n_edit = len(edit_str)

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
        edit_str = read.edit_string
        abs_indices = read.absolute_indices
        n_edit = len(edit_str)

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

def passes_coverage(df_ref: pd.DataFrame, df_qry: pd.DataFrame,
                    min_coverage: float) -> bool:
    """
    Use read count as a fast coverage proxy. Both libraries must have at
    least min_coverage reads overlapping the gene.
    """
    return len(df_ref) >= min_coverage and len(df_qry) >= min_coverage

def compute_binomial_pvals_per_read2(read_edits: dict, ref_freq: dict,
                                     ref_cov: dict,
                                     nt_window: int, min_sites: int,
                                     gene_len: int,
                                     tx_to_gpos: dict = None) -> tuple:
    """
    Site-anchored lower-tail binomial protection test per read.

    One window per covered ref=A site, centered on that site. The trace point
    is plotted AT the site, so trace x-values are real transcript positions.

    Returns (results, per_site) where:
      results  = {read_id: [(tx_pos, -log10(p), window_coverage), ...]}
      per_site = {read_id: [ {tx_pos, gpos, edit, ref_freq, ref_cov}, ... ]}
    """
    results = {}
    per_site = {}
    half = nt_window // 2

    for read_id, pos_dict in read_edits.items():
        sites = sorted(pos_dict.keys())
        trace = []

        for site in sites:
            ks, ps, cs = [], [], []
            for tx in range(site - half, site + half):
                if tx in pos_dict and tx in ref_freq:
                    ks.append(pos_dict[tx])
                    ps.append(ref_freq[tx])
                    cs.append(ref_cov.get(tx, 0))

            n = len(ks)
            if n < min_sites:
                continue

            k = sum(ks)
            c = float(np.mean(cs)) if cs else 0.0
            p_mean = float(np.mean(ps))
            p_val = max(scipy.stats.binom.cdf(k, n, p_mean), 1e-300)

            trace.append((site, -math.log10(p_val), c))

        if trace:
            results[read_id] = trace

            site_rows = []
            for tx in sites:
                if tx not in ref_freq:
                    continue
                site_rows.append({
                    "tx_pos": tx,
                    "gpos": tx_to_gpos.get(tx) if tx_to_gpos else None,
                    "edit": pos_dict[tx],
                    "ref_freq": ref_freq[tx],
                    "ref_cov": ref_cov.get(tx, 0),
                })
            per_site[read_id] = site_rows

    return results, per_site

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

def write_shadow_calls_to_df(gene, df_qry, binomial_scores, read_edits,
                             ref_cov, gpos_to_tx, tx_to_gpos,
                             pval_threshold=0.05):
    """
    Shadow calls in read-level format, mirroring the source parquet schema.

    One row per read with >= 1 significant site. All original read columns are
    carried through; shadow information is added as parallel arrays plus a
    shadow_string aligned index-for-index with edit_string.

    Site-anchored: each trace point IS a ref=A site, so a site is "in shadow"
    when its own window p-value <= pval_threshold.

    Added columns:
      shadow_gene        gene whose background model produced these calls
      shadow_string      per aligned position: '1' in shadow, '0' not, '2' indel
      shadow_tx_pos      CDS-relative positions of significant sites (5'UTR < 0)
      shadow_gpos        genomic positions of those sites
      shadow_region      UTR5 / CDS / UTR3 per site
      shadow_pval        p-value at each site
      shadow_neg_log10p  -log10 of the above
      shadow_edit        1 = edited (G), 0 = unedited (A)
      shadow_ref_cov     reference reads backing the background at each site
      n_shadow_sites     total significant sites on the read
      n_sites_utr5/cds/utr3   breakdown by region
      min_pval           strongest p-value on the read
    """
    sig_by_read = {}
    for read_id, trace in binomial_scores.items():
        sig = {}
        for tx, nlp, _cov in trace:
            p = 10 ** (-nlp)
            if p <= pval_threshold:
                sig[tx] = p
        if sig:
            sig_by_read[read_id] = sig

    if not sig_by_read:
        return pd.DataFrame()

    sub = (df_qry[df_qry["read_id"].isin(sig_by_read)]
           .copy()
           .reset_index(drop=True))
    if sub.empty:
        return pd.DataFrame()

    cds_len = cds_length(gene)

    shadow_strings, tx_list, gpos_list, region_list = [], [], [], []
    pval_list, nlp_list, edit_list, cov_list = [], [], [], []
    n_sites, min_pvals = [], []
    n_utr5, n_cds, n_utr3 = [], [], []

    for read in sub.itertuples():
        sig      = sig_by_read[read.read_id]
        pos_dict = read_edits.get(read.read_id, {})
        edit_str = read.edit_string

        # shadow_string: index-for-index with edit_string
        chars = []
        for i, gpos in enumerate(read.absolute_indices):
            if i >= len(edit_str) or edit_str[i] == "2":
                chars.append("2")
                continue
            try:
                gp = int(gpos)
            except (TypeError, ValueError):
                chars.append("2")
                continue
            tx = gpos_to_tx.get(gp)
            chars.append("1" if (tx is not None and tx in sig) else "0")
        shadow_strings.append("".join(chars))

        sites   = sorted(sig.keys())
        pvals   = [sig[t] for t in sites]
        regions = [classify_tx(t, cds_len) for t in sites]
        counts  = collections.Counter(regions)

        tx_list.append(sites)
        gpos_list.append([tx_to_gpos.get(t) for t in sites])
        region_list.append(regions)
        pval_list.append(pvals)
        nlp_list.append([-math.log10(p) for p in pvals])
        edit_list.append([int(pos_dict[t]) for t in sites])
        cov_list.append([ref_cov.get(t) for t in sites])

        n_sites.append(len(sites))
        min_pvals.append(min(pvals))
        n_utr5.append(counts["UTR5"])
        n_cds.append(counts["CDS"])
        n_utr3.append(counts["UTR3"])

    sub["shadow_gene"]       = gene["gene_name"]
    sub["shadow_string"]     = shadow_strings
    sub["shadow_tx_pos"]     = tx_list
    sub["shadow_gpos"]       = gpos_list
    sub["shadow_region"]     = region_list
    sub["shadow_pval"]       = pval_list
    sub["shadow_neg_log10p"] = nlp_list
    sub["shadow_edit"]       = edit_list
    sub["shadow_ref_cov"]    = cov_list
    sub["n_shadow_sites"]    = n_sites
    sub["n_sites_utr5"]      = n_utr5
    sub["n_sites_cds"]       = n_cds
    sub["n_sites_utr3"]      = n_utr3
    sub["min_pval"]          = min_pvals

    return sub


def plot_gene_pyx(gene_name, meta, binomial_scores, his_positions,
                  read_edits, label1, label2, tx_lo, tx_hi, num_reads,
                  pdf_path, ref_cov=None):
    from pyx import canvas, graph, color, style, text as pyx_text

    col_qry  = color.cmyk(1, 0.5, 0, 0)
    col_his  = color.cmyk(0, 1, 1, 0)
    col_sig  = color.cmyk(0, 0, 0, 0.4)
    col_edit = color.cmyk(0, 0, 0, 1)
    col_no   = color.cmyk(0, 0.8, 1, 0)
    col_cov  = color.cmyk(0.7, 0, 0.7, 0.1)   # green — background coverage

    x_min, x_max = tx_lo, tx_hi
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

def parse_args():
    p = argparse.ArgumentParser(
        description="Binomial test for editing protection."
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
          f"query >= {args.min_query_reads} reads", file=sys.stderr)
    if args.cds_spanning:
        print("  CDS spanning filter: ON (query only)", file=sys.stderr)
    if args.require_his_codon:
        print("  His codon filter: ON (genes need >=1 CAT/CAC)",
              file=sys.stderr)
    if args.min_edit_freq > 0.0:
        print(f"  Query edit-freq filter: ON "
              f"(global_edit_freq >= {args.min_edit_freq})", file=sys.stderr)

    print("\nParsing GTF...", file=sys.stderr)
    genes = parse_gtf(args.gtf)
    print(f"  {len(genes):,} genes.", file=sys.stderr)

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
          f"(summary + ranking)...", file=sys.stderr)

    # Diagnostics
    max_bg  = 0
    max_qry = 0

    gene_span_counts = {}   # gname -> n_query_reads (ranking metric)

    for i, gname in enumerate(gene_names):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(gene_names)} scanned, "
                  f"{n_pass} passing...", file=sys.stderr)

        gene = genes[gname]
        gene_len = cds_length(gene)

        his_positions = find_his_codon_tx_positions(ref_fasta, gene)
        if args.require_his_codon and len(his_positions) == 0:
            continue

        # Background: ALWAYS all overlapping reference reads (never spanning),
        # never edit-freq filtered — maximum per-position support.
        df_ref_bg = get_gene_df(df_all_ref, gene, cds_spanning=False)

        # Query: the reads we make protection calls on.
        df_qry = get_gene_df(df_all_qry, gene, cds_spanning=args.cds_spanning,
                             min_edit_freq=args.min_edit_freq)

        max_bg  = max(max_bg,  len(df_ref_bg))
        max_qry = max(max_qry, len(df_qry))

        # Two independent coverage thresholds
        if len(df_ref_bg) < args.min_coverage:
            continue
        if len(df_qry) < args.min_query_reads:
            continue

        gpos_to_tx = _gpos_to_tx_map(gene, ref_fasta)
        tx_lo = min(gpos_to_tx.values())
        tx_hi = max(gpos_to_tx.values())
        if not gpos_to_tx:
            continue

        ref_freq, ref_cov = build_reference_freq_and_coverage(
            df_ref_bg, gpos_to_tx, gene)
        if not ref_freq:
            continue

        read_edits = collect_read_edits(df_qry, gpos_to_tx, gene)
        if not read_edits:
            continue

        binomial_scores, _ = compute_binomial_pvals_per_read2(
            read_edits, ref_freq, ref_cov,
            nt_window=args.window,
            min_sites=args.min_sites,
            gene_len=gene_len,
        )
        if not binomial_scores:
            continue

        n_pass += 1

        sig_line = -math.log10(0.05)
        by_region = {"UTR5": [], "CDS": [], "UTR3": []}
        for trace in binomial_scores.values():
            for point in trace:
                by_region[classify_tx(point[0], gene_len)].append(point[1])

        all_scores = [v for vals in by_region.values() for v in vals]
        frac_sig = (sum(1 for v in all_scores if v >= sig_line)
                    / len(all_scores)) if all_scores else 0.0

        row = {
            "gene": gname,
            "n_reads": len(binomial_scores),
            "n_his_codons": len(his_positions),
            "gene_len": gene_len,
            "n_ref_a_sites": len(ref_freq),
            "n_reads_ref_bg": len(df_ref_bg),
            "n_reads_qry": len(df_qry),
            "median_neg_log10p": float(np.median(all_scores)),
            "frac_sig_windows": frac_sig,
        }
        for region in ("UTR5", "CDS", "UTR3"):
            vals = by_region[region]
            key = region.lower()
            row[f"n_sites_{key}"] = len(vals)
            row[f"frac_sig_{key}"] = (
                sum(1 for v in vals if v >= sig_line) / len(vals)
                if vals else float("nan"))

        gene_span_counts[gname] = len(df_qry)
        summary_rows.append(row)
        del binomial_scores, read_edits, ref_freq, ref_cov, gpos_to_tx


    # ── Pass 2: recompute, plot, and write shadow calls for passing genes ────
    shadow_calls_path = f"{out}_shadow_calls.parquet"
    shadow_writer     = None

    if gene_span_counts:
        ranked = sorted(gene_span_counts.keys(),
                        key=lambda g: gene_span_counts[g], reverse=True)

        print(f"\nPass 2: plotting {len(ranked)} genes "
              f"(ranked by query reads)...", file=sys.stderr)

        for gname in ranked:
            gene     = genes[gname]
            gene_len = cds_length(gene)
            print(f"    {gname}: {gene_span_counts[gname]:,} query reads",
                  file=sys.stderr)

            df_ref_bg = get_gene_df(df_all_ref, gene, cds_spanning=False)
            df_qry = get_gene_df(df_all_qry, gene,
                                 cds_spanning=args.cds_spanning,
                                 min_edit_freq=args.min_edit_freq)

            gpos_to_tx  = _gpos_to_tx_map(gene, ref_fasta)
            tx_lo = min(gpos_to_tx.values())
            tx_hi = max(gpos_to_tx.values())
            full_tx_map = _full_tx_map(gene, ref_fasta)
            ref_freq, ref_cov = build_reference_freq_and_coverage(
                df_ref_bg, gpos_to_tx, gene)
            read_edits = collect_read_edits(df_qry, gpos_to_tx, gene)

            binomial_scores, _ = compute_binomial_pvals_per_read2(
                read_edits, ref_freq, ref_cov,
                nt_window=args.window,
                min_sites=args.min_sites,
                gene_len=gene_len,
            )
            if not binomial_scores:
                continue

            meta          = get_meta_read(binomial_scores)
            his_positions = find_his_codon_tx_positions(ref_fasta, gene)
            safe_name     = re.sub(r"[^\w\-]", "_", gname)

            try:
                plot_gene_pyx(
                    gene_name=gname, meta=meta,
                    binomial_scores=binomial_scores,
                    his_positions=his_positions, read_edits=read_edits,
                    label1=args.label1, label2=args.label2,
                    tx_lo=tx_lo, tx_hi=tx_hi,
                    num_reads=args.num_reads,
                    pdf_path=str(pdf_dir / safe_name),
                    ref_cov=ref_cov,
                )
            except Exception as e:
                print(f"WARNING: pyx plot failed for {gname}: {e}",
                      file=sys.stderr)

            # Shadow calls: write incrementally, don't accumulate
            tx_to_gpos = {tx: gp for gp, tx in gpos_to_tx.items()}

            sc_df = write_shadow_calls_to_df(
                gene, df_qry, binomial_scores, read_edits,
                ref_cov, gpos_to_tx, tx_to_gpos,
                pval_threshold=0.05,
            )
            if not sc_df.empty:
                table = pa.Table.from_pandas(sc_df, preserve_index=False)
                if shadow_writer is None:
                    shadow_writer = pq.ParquetWriter(shadow_calls_path,
                                                     table.schema)
                shadow_writer.write_table(table)

            del binomial_scores, read_edits, ref_freq, ref_cov, \
                gpos_to_tx, full_tx_map, sc_df

    if shadow_writer is not None:
        shadow_writer.close()
        print(f"\n  Shadow calls -> {shadow_calls_path}", file=sys.stderr)

    ref_fasta.close()

    print(f"\n  {n_pass:,}/{len(gene_names):,} genes passed all filters.",
          file=sys.stderr)
    print(f"  [diagnostics] max reads in any gene: "
          f"background={max_bg:,}  query={max_qry:,}", file=sys.stderr)
    if n_pass == 0:
        print(f"  HINT: no gene cleared --min_coverage={int(args.min_coverage)} "
              f"(background, max seen {max_bg:,}) and "
              f"--min_query_reads={args.min_query_reads} "
              f"(query, max seen {max_qry:,}).", file=sys.stderr)

    if not summary_rows:
        print("ERROR: No genes produced output.", file=sys.stderr)
        sys.exit(1)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("median_neg_log10p", ascending=False)
    summary_csv = f"{out}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n  Summary -> {summary_csv}", file=sys.stderr)
    print(f"  Gene PDFs -> {pdf_dir}/", file=sys.stderr)
    print("\nDone.", file=sys.stderr)

if __name__ == "__main__":
    Tee()
    main()