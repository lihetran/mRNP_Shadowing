'''
July 10, LT

In order to call ribosome footprints, we are going to build a Bernoulli Naive Bayesian Classifier. To do this, we will estimate whether positions or windows of a translated molecule are more likely to have come from a
ribosome-less (phenol-extracted, A) population of molecules or an unedited (mock tadA, B) population of molecules.

Step 1: For each gene, build reference frequency table.
    - do Laplace, plus 1 smoothing
    - we're going to have 2 frequency tables, one for ribosome-less/TadA and one for mock Tad

Step 2: Estimate class priors
    - P(A) = 0.8
    - P(B) = 0.2
    - It would be cool to do gradient descent to optimize these initial priors, for now these numbers are based on 30 nt RPFs and ~150 nt spacing on RNAs

Step 3: For each read, compute the log-likelihood of the read coming from class A or class B
    - P(A|C) or P(B|C) where C is new read
    - do this in a sliding window so that the question is "which class is this window of the read more likely to have come from?"
Step 4: Call the read as class A or class B based on the log-likelihood
    - we should have some probability attached to calls that'll let us know how confident we are about said call

Step 5: Output the read calls to a parquet file for downstream analysis



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


def get_gene_df(df_all: pd.DataFrame, gene: dict,
                cds_spanning: bool = False,
                min_edit_freq: float = 0.0) -> pd.DataFrame:
    """
    Built with Claude
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
    Built with Claude
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
    if tx_pos < 0: return "UTR5"
    if tx_pos < cds_len: return "CDS"
    return "UTR3"


def build_reference_freq_and_coverage(df: pd.DataFrame, gpos_to_tx: dict,
                                       gene: dict) -> tuple:
    """
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

def build_frequency_df(df: pd.DataFrame, gpos_to_tx: dict, alpha=1, beta=1):
    '''
    Very similar to build reference freq and coverage but going to apply Laplace smoothing
    '''
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
    ref_cov = {}
    for tx, (n0, n1) in edit_counts.items():
        total = n0 + n1
        if total > 0:
            # Laplace / Beta(alpha, beta) smoothing:
            #   alpha = pseudocount of 1s, beta = pseudocount of 0s
            ref_freq[tx] = (n1 + alpha) / (total + alpha + beta)
            ref_cov[tx] = total  # coverage stays the RAW count
    return ref_freq, ref_cov

def train(A_df, B_df, alpha=1, beta=1, gpos_to_tx=None):
    pA, covA = build_frequency_df(A_df, gpos_to_tx, alpha, beta)
    pB, covB = build_frequency_df(B_df, gpos_to_tx, alpha, beta)

    w1, w0 = {}, {}
    for tx in pA.keys() & pB.keys():  # positions BOTH populations saw
        a, b = pA[tx], pB[tx]
        w1[tx] = math.log(a) - math.log(b)  # weight when the bit is 1
        w0[tx] = math.log(1 - a) - math.log(1 - b)  # weight when the bit is 0

    nA, nB = len(A_df), len(B_df)
    prior_log_odds = math.log(nA / nB) if nA and nB else 0.0

    return {"w1": w1, "w0": w0, "prior_log_odds": prior_log_odds, "pA": pA, "pB": pB, "covA": covA, "covB": covB}

def classify_read(model, edit_string, absolute_indices, gpos_to_tx, use_prior=True):
    log_odds = model["prior_log_odds"] if use_prior else 0.0
    n_edit = len(edit_string)
    for i, ref_pos in enumerate(absolute_indices):
        if ref_pos is None or (isinstance(ref_pos, float) and ref_pos != ref_pos):
            continue
        ref_pos = int(ref_pos)
        if ref_pos not in gpos_to_tx or i >= n_edit:
            continue
        ev = edit_string[i]
        if ev == "2":                               # no-info base, skip
            continue
        tx = gpos_to_tx[ref_pos]
        if tx not in model["w1"]:                   # position unseen in training
            continue
        log_odds += model["w1"][tx] if ev == "1" else model["w0"][tx]
    p_A = 1.0 / (1.0 + math.exp(-log_odds))
    return {"P_A": p_A, "P_B": 1 - p_A, "label": "A" if p_A >= 0.5 else "B"}

def classify_positions(model, read_id, chrom, edit_string, absolute_indices,
                       gpos_to_tx, use_prior=True):
    base = model["prior_log_odds"] if use_prior else 0.0
    n_edit = len(edit_string)
    positions, txs, labels, A, B = [], [], [], [], []

    for i, ref_pos in enumerate(absolute_indices):
        if ref_pos is None or (isinstance(ref_pos, float) and ref_pos != ref_pos):
            continue
        ref_pos = int(ref_pos)
        if ref_pos not in gpos_to_tx or i >= n_edit:
            continue
        ev = edit_string[i]
        if ev == "2":
            continue
        tx = gpos_to_tx[ref_pos]
        if tx not in model["w1"]:
            continue

        log_odds = base + (model["w1"][tx] if ev == "1" else model["w0"][tx])
        p_A = 1.0 / (1.0 + math.exp(-log_odds))

        positions.append(ref_pos)
        txs.append(tx)
        A.append(p_A)
        B.append(1.0 - p_A)
        labels.append("A" if p_A >= 0.5 else "B")

    yield {"read_id": read_id, "chrom": chrom,
           "absolute_indices": positions, "tx": txs,
           "labels": labels, "P_A": A, "P_B": B}
    # return events

import math, bisect

def classify_positions_windowed(model, read_id, chrom, edit_string, absolute_indices,
                                gpos_to_tx, window=30, coord="tx",
                                use_prior=True):
    base = model["prior_log_odds"] if use_prior else 0.0
    n_edit = len(edit_string)

    # pass 1: collect scored sites (read order) with per-site contribution
    sites = []                                        # (tx, ref_pos, contrib)
    for i, ref_pos in enumerate(absolute_indices):
        if ref_pos is None or (isinstance(ref_pos, float) and ref_pos != ref_pos):
            continue
        ref_pos = int(ref_pos)
        if ref_pos not in gpos_to_tx or i >= n_edit:
            continue
        ev = edit_string[i]
        if ev == "2":
            continue
        tx = gpos_to_tx[ref_pos]
        if tx not in model["w1"]:
            continue
        contrib = model["w1"][tx] if ev == "1" else model["w0"][tx]
        sites.append((tx, ref_pos, contrib))

    if not sites:
        yield {"read_id": read_id, "chrom": chrom, "absolute_indices": [],
               "tx": [], "labels": [], "P_A": [], "P_B": [], "win_n": []}
        return

    # sorted-by-coordinate copy for prefix sums + window lookup
    key = (lambda s: s[0]) if coord == "tx" else (lambda s: s[1])
    ordered = sorted(sites, key=key)
    scoords = [key(s) for s in ordered]
    prefix  = [0.0]
    for s in ordered:
        prefix.append(prefix[-1] + s[2])

    half = window / 2.0
    positions, txs, labels, A, B, win_n = [], [], [], [], [], []
    for tx, ref_pos, _ in sites:                      # emit in read order
        center = tx if coord == "tx" else ref_pos
        l = bisect.bisect_left(scoords,  center - half)
        r = bisect.bisect_right(scoords, center + half)
        log_odds = base + (prefix[r] - prefix[l])
        p_A = 1.0 / (1.0 + math.exp(-log_odds))
        positions.append(ref_pos); txs.append(tx)
        A.append(p_A); B.append(1.0 - p_A)
        labels.append("A" if p_A >= 0.5 else "B")
        win_n.append(r - l)

    yield {"read_id": read_id, "chrom": chrom, "absolute_indices": positions,
           "tx": txs, "labels": labels, "P_A": A, "P_B": B, "win_n": win_n}

# def read_contributions(model, edit_string, absolute_indices, gpos_to_tx):
#     '''
#     Returns a list of contribution events. Scores each position
#     '''
#     events = []
#     n_edit = len(edit_string)
#     for i, ref_pos in enumerate(absolute_indices):
#         if ref_pos is None or (isinstance(ref_pos, float) and ref_pos != ref_pos):
#             continue
#         ref_pos = int(ref_pos)
#         if ref_pos not in gpos_to_tx or i >= n_edit:
#             continue
#         ev = edit_string[i]
#         if ev == "2":
#             continue
#         tx = gpos_to_tx[ref_pos]
#         if tx not in model["w1"]:
#             continue
#         contrib = model["w1"][tx] if ev == "1" else model["w0"][tx]
#         events.append({"i": i, "ref_pos": ref_pos, "tx": tx, "contrib": contrib})
#     return events
#
# def sliding_classify(model, edit_string, absolute_indices, gpos_to_tx,
#                      window, step=1, coord="ref_pos",
#                      min_positions=1, use_prior=False):
#     '''
#     Classify events in read by window. Manually annotates events in a window
#     '''
#     import bisect, math
#     events = read_contributions(model, edit_string, absolute_indices, gpos_to_tx)
#     if not events:
#         return []
#     events.sort(key=lambda e: e[coord])
#     coords = [e[coord] for e in events]
#     prefix = [0.0]
#     for e in events:
#         prefix.append(prefix[-1] + e["contrib"])
#     lo_c, hi_c = coords[0], coords[-1]
#     base = model["prior_log_odds"] if use_prior else 0.0
#
#     results = []
#     start = lo_c
#     while start <= hi_c:
#         end = start + window
#         l = bisect.bisect_left(coords, start)
#         r = bisect.bisect_left(coords, end)
#         n = r - l
#         if n >= min_positions:
#             log_odds = base + (prefix[r] - prefix[l])
#             p_A = 1.0 / (1.0 + math.exp(-log_odds))
#             results.append({"start": start, "end": end, "center": start + window/2,
#                             "P_A": p_A, "P_B": 1 - p_A,
#                             "label": "A" if p_A >= 0.5 else "B", "n": n})
#         start += step
#     return results

def plot_pb_by_tx_pyx(gene_name, df, his_positions, tx_lo, tx_hi, pdf_path,
                      label1="A", label2="B", ref_cov=None,
                      tx_col="tx", pb_col="P_B", edit_col="edits",
                      pct=(5, 95), num_reads=10):
    from pyx import canvas, graph, color, style, text as pyx_text
    from collections import defaultdict
    import numpy as np

    col_qry  = color.cmyk(1, 0.5, 0, 0)
    col_his  = color.cmyk(0, 1, 1, 0)
    col_sig  = color.cmyk(0, 0, 0, 0.4)
    col_edit = color.cmyk(0, 0, 0, 1)
    col_no   = color.cmyk(0, 0.8, 1, 0)
    col_cov  = color.cmyk(0.7, 0, 0.7, 0.1)   # green — background coverage

    pdf_path = str(pdf_path)

    x_min, x_max = tx_lo, tx_hi
    panel_w  = 12
    tick_h   = 0.25
    read_h   = 1.2
    gap      = 0.6
    cov_h    = 1.5

    # aggregate per-tx distribution of P_B across all reads (meta)
    pb_by_tx = defaultdict(list)
    for tx_list, pb_list in zip(df[tx_col], df[pb_col]):
        for t, pb in zip(tx_list, pb_list):
            pb_by_tx[int(t)].append(pb)
    items    = sorted(pb_by_tx.items())
    xs       = [t for t, _ in items]
    mean_pb  = [float(np.mean(v))               for _, v in items]
    lo_pb    = [float(np.percentile(v, pct[0])) for _, v in items]
    hi_pb    = [float(np.percentile(v, pct[1])) for _, v in items]
    cov_self = {t: len(v) for t, v in items}

    c         = canvas.canvas()
    meta_ypos = num_reads * (read_h + tick_h + gap) + gap * 2

    g_meta = graph.graphxy(
        width=panel_w, height=3, xpos=0, ypos=meta_ypos,
        x=graph.axis.linear(min=x_min, max=x_max,
                            title="Position Along Transcript (nt)"),
        y=graph.axis.linear(min=0, max=1, title="P(Shadow)"),
    )
    for hp in his_positions:
        g_meta.plot(
            graph.data.function(f"x(y)={hp}", min=0, max=1),
            [graph.style.line([col_his, style.linewidth.thin,
                               style.linestyle.solid])])
    g_meta.plot(
        graph.data.function("y(x)=0.5", min=x_min, max=x_max),
        [graph.style.line([col_sig, style.linewidth.thin,
                           style.linestyle.dashed])])
    for yb in (lo_pb, hi_pb):
        if xs:
            g_meta.plot(graph.data.points(list(zip(xs, yb)), x=1, y=2),
                        [graph.style.line([col_qry, style.linewidth.thin,
                                           style.linestyle.dotted])])
    if xs:
        g_meta.plot(graph.data.points(list(zip(xs, mean_pb)), x=1, y=2),
                    [graph.style.line([col_qry, style.linewidth.normal,
                                       style.linestyle.solid])])
    c.insert(g_meta)

    # ── Background coverage panel above the meta plot ─────────────────────────
    cov_ypos = meta_ypos + 3 + gap
    title_y  = cov_ypos + 3 + 0.4   # default if no coverage panel drawn

    cov_source = ref_cov if ref_cov else cov_self
    if cov_source:
        cov_items = sorted(cov_source.items())
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
        for hp in his_positions:
            g_cov.plot(
                graph.data.function(f"x(y)={hp}", min=0, max=cov_y_max),
                [graph.style.line([col_his, style.linewidth.thin,
                                   style.linestyle.solid])])
        if len(cov_items) > 0:
            g_cov.plot(
                graph.data.points(list(zip(cov_x, cov_y)), x=1, y=2),
                [graph.style.line([col_cov, style.linewidth.normal,
                                   style.linestyle.solid])])
        c.insert(g_cov)
        title_y = g_cov.ypos + g_cov.height + 0.4

    c.text(g_meta.xpos + g_meta.width / 2., title_y,
           f"{gene_name} - {label2}",
           [pyx_text.halign.center, pyx_text.size.normalsize])

    n_show = min(num_reads, len(df))
    for jj in range(n_show):
        row = df.iloc[jj]
        tx_r = list(row[tx_col]); pb_r = list(row[pb_col])
        if not tx_r:
            continue
        order = np.argsort(tx_r)
        tx_s = [tx_r[k] for k in order]; pb_s = [pb_r[k] for k in order]

        ypos = (num_reads - 1 - jj) * (read_h + tick_h + gap)

        g_read = graph.graphxy(
            width=panel_w, height=read_h, xpos=0, ypos=ypos + tick_h,
            x=graph.axis.linkedaxis(g_meta.axes["x"]),
            y=graph.axis.linear(min=0, max=1, title=""),
        )
        for hp in his_positions:
            g_read.plot(
                graph.data.function(f"x(y)={hp}", min=0, max=1),
                [graph.style.line([col_his, style.linewidth.thin,
                                   style.linestyle.solid])])
        g_read.plot(
            graph.data.function("y(x)=0.5", min=x_min, max=x_max),
            [graph.style.line([col_sig, style.linewidth.thin,
                               style.linestyle.dashed])])
        g_read.plot(
            graph.data.points(list(zip(tx_s, pb_s)), x=1, y=2),
            [graph.style.line([col_qry, style.linewidth.thin,
                               style.linestyle.solid])])
        c.insert(g_read)

        g_ticks = graph.graphxy(
            width=panel_w, height=tick_h, xpos=0, ypos=ypos,
            x=graph.axis.linkedaxis(g_meta.axes["x"]),
            y=graph.axis.linear(min=0, max=1),
        )
        edit_r = list(row[edit_col]) if edit_col in df.columns else None
        for k in order:
            v = int(edit_r[k]) if edit_r is not None else (0 if pb_r[k] >= 0.5 else 1)
            col = col_edit if v == 1 else col_no
            g_ticks.plot(
                graph.data.function(f"x(y)={tx_r[k]}", min=0, max=1),
                [graph.style.line([col, style.linewidth.thin,
                                   style.linestyle.solid])])
        c.insert(g_ticks)

        label_txt = str(row["read_id"])[:20] if "read_id" in df.columns else f"read {jj}"
        c.text(panel_w + 0.15, ypos + tick_h + read_h / 2.,
               label_txt, [pyx_text.valign.middle, pyx_text.size.tiny])

    c.writePDFfile(pdf_path)

def plot_signed_log_pyx(gene_name, df, his_positions, tx_lo, tx_hi, pdf_path,
                        label1="A", label2="B", ref_cov=None,
                        tx_col="tx", pa_col="P_A", pb_col="P_B", edit_col="edits",
                        pct=(5, 95), num_reads=10, eps=1e-6):
    from pyx import canvas, graph, color, style, text as pyx_text
    from collections import defaultdict
    import numpy as np, math

    col_qry  = color.cmyk(1, 0.5, 0, 0)       # blue  — P_A trace (flipped, +)
    col_pb   = color.cmyk(0.4, 1, 0, 0)       # purple — P_B trace (unflipped, -)
    col_his  = color.cmyk(0, 1, 1, 0)
    col_sig  = color.cmyk(0, 0, 0, 0.4)
    col_edit = color.cmyk(0, 0, 0, 1)
    col_no   = color.cmyk(0, 0.8, 1, 0)
    col_cov  = color.cmyk(0.7, 0, 0.7, 0.1)

    pdf_path = str(pdf_path)
    x_min, x_max = tx_lo, tx_hi
    panel_w, tick_h, read_h, gap, cov_h = 12, 0.25, 1.2, 0.6, 1.5

    def yA(pa): return -math.log10(max(pa, eps))
    def yB(pb): return  math.log10(max(pb, eps))

    ya_by, yb_by = defaultdict(list), defaultdict(list)
    for tx_l, pa_l, pb_l in zip(df[tx_col], df[pa_col], df[pb_col]):
        for t, pa, pb in zip(tx_l, pa_l, pb_l):
            ya_by[int(t)].append(yA(pa)); yb_by[int(t)].append(yB(pb))
    items   = sorted(ya_by); xs = list(items)
    ya_mean = [float(np.mean(ya_by[t]))               for t in items]
    ya_lo   = [float(np.percentile(ya_by[t], pct[0])) for t in items]
    ya_hi   = [float(np.percentile(ya_by[t], pct[1])) for t in items]
    yb_mean = [float(np.mean(yb_by[t]))               for t in items]
    yb_lo   = [float(np.percentile(yb_by[t], pct[0])) for t in items]
    yb_hi   = [float(np.percentile(yb_by[t], pct[1])) for t in items]
    cov_self = {t: len(ya_by[t]) for t in items}

    allv  = [abs(v) for v in ya_hi + yb_lo + ya_mean + yb_mean] or [1.0]
    y_lim = max(max(allv) * 1.1, 1.0)

    def draw_his(g):
        for hp in his_positions:
            g.plot(graph.data.function(f"x(y)={hp}", min=-y_lim, max=y_lim),
                   [graph.style.line([col_his, style.linewidth.thin, style.linestyle.solid])])
    def draw_zero(g):
        g.plot(graph.data.function("y(x)=0", min=x_min, max=x_max),
               [graph.style.line([col_sig, style.linewidth.thin, style.linestyle.dashed])])

    c = canvas.canvas()
    meta_ypos = num_reads * (read_h + tick_h + gap) + gap * 2

    g_meta = graph.graphxy(
        width=panel_w, height=3, xpos=0, ypos=meta_ypos,
        x=graph.axis.linear(min=x_min, max=x_max, title="Position Along Transcript (nt)"),
        y=graph.axis.linear(min=-y_lim, max=y_lim, title="log10 P"))
    draw_his(g_meta); draw_zero(g_meta)
    if xs:
        for yb in (ya_lo, ya_hi):
            g_meta.plot(graph.data.points(list(zip(xs, yb)), x=1, y=2),
                        [graph.style.line([col_qry, style.linewidth.thin, style.linestyle.dotted])])
        for yb in (yb_lo, yb_hi):
            g_meta.plot(graph.data.points(list(zip(xs, yb)), x=1, y=2),
                        [graph.style.line([col_pb, style.linewidth.thin, style.linestyle.dotted])])
        g_meta.plot(graph.data.points(list(zip(xs, ya_mean)), x=1, y=2),
                    [graph.style.line([col_qry, style.linewidth.normal, style.linestyle.solid])])
        g_meta.plot(graph.data.points(list(zip(xs, yb_mean)), x=1, y=2),
                    [graph.style.line([col_pb, style.linewidth.normal, style.linestyle.solid])])
    c.insert(g_meta)
    # directional labels, horizontal, inside the top-left / bottom-left corners
    lab_x = g_meta.xpos + 0.2
    c.text(lab_x, meta_ypos + 3 - 0.25, "log P(protected)",
           [pyx_text.halign.left, pyx_text.valign.top, pyx_text.size.small])
    c.text(lab_x, meta_ypos + 0.25, "log P(unprotected)",
           [pyx_text.halign.left, pyx_text.valign.bottom, pyx_text.size.small])

    cov_ypos = meta_ypos + 3 + gap
    title_y  = cov_ypos + 3 + 0.4
    cov_source = ref_cov if ref_cov else cov_self
    if cov_source:
        cov_items = sorted(cov_source.items())
        cov_x = [tx for tx, _ in cov_items]; cov_y = [n for _, n in cov_items]
        cov_y_max = max(cov_y) * 1.1 if cov_y else 1.0
        g_cov = graph.graphxy(width=panel_w, height=cov_h, xpos=0, ypos=cov_ypos,
            x=graph.axis.linkedaxis(g_meta.axes["x"]),
            y=graph.axis.linear(min=0, max=cov_y_max, title="Ref cov"))
        for hp in his_positions:
            g_cov.plot(graph.data.function(f"x(y)={hp}", min=0, max=cov_y_max),
                       [graph.style.line([col_his, style.linewidth.thin, style.linestyle.solid])])
        if cov_items:
            g_cov.plot(graph.data.points(list(zip(cov_x, cov_y)), x=1, y=2),
                       [graph.style.line([col_cov, style.linewidth.normal, style.linestyle.solid])])
        c.insert(g_cov)
        title_y = g_cov.ypos + g_cov.height + 0.4

    c.text(g_meta.xpos + g_meta.width / 2., title_y, f"{gene_name}: {label2}",
           [pyx_text.halign.center, pyx_text.size.normalsize])

    n_show = min(num_reads, len(df))
    for jj in range(n_show):
        row = df.iloc[jj]
        tx_r = list(row[tx_col]); pa_r = list(row[pa_col]); pb_r = list(row[pb_col])
        if not tx_r:
            continue
        order = np.argsort(tx_r)
        tx_s = [tx_r[k] for k in order]
        yaS  = [yA(pa_r[k]) for k in order]
        ybS  = [yB(pb_r[k]) for k in order]

        ypos = (num_reads - 1 - jj) * (read_h + tick_h + gap)
        g_read = graph.graphxy(width=panel_w, height=read_h, xpos=0, ypos=ypos + tick_h,
            x=graph.axis.linkedaxis(g_meta.axes["x"]),
            y=graph.axis.linear(min=-y_lim, max=y_lim, title=""))
        draw_his(g_read); draw_zero(g_read)
        g_read.plot(graph.data.points(list(zip(tx_s, yaS)), x=1, y=2),
                    [graph.style.line([col_qry, style.linewidth.thin, style.linestyle.solid])])
        g_read.plot(graph.data.points(list(zip(tx_s, ybS)), x=1, y=2),
                    [graph.style.line([col_pb, style.linewidth.thin, style.linestyle.solid])])
        c.insert(g_read)

        g_ticks = graph.graphxy(width=panel_w, height=tick_h, xpos=0, ypos=ypos,
            x=graph.axis.linkedaxis(g_meta.axes["x"]),
            y=graph.axis.linear(min=0, max=1))
        edit_r = list(row[edit_col]) if edit_col in df.columns else None
        for k in order:
            v = int(edit_r[k]) if edit_r is not None else (0 if pa_r[k] < 0.5 else 1)
            col = col_edit if v == 1 else col_no
            g_ticks.plot(graph.data.function(f"x(y)={tx_r[k]}", min=0, max=1),
                         [graph.style.line([col, style.linewidth.thin, style.linestyle.solid])])
        c.insert(g_ticks)

        label_txt = str(row["read_id"])[:20] if "read_id" in df.columns else f"read {jj}"
        c.text(panel_w + 0.15, ypos + tick_h + read_h / 2., label_txt,
               [pyx_text.valign.middle, pyx_text.size.tiny])

    c.writePDFfile(pdf_path)

def parse_args():
    p = argparse.ArgumentParser(
        description="Bernoulli Naive Bayes Shadow Classifier."
    )
    p.add_argument("--parquet1", required=True)
    p.add_argument("--parquet2", required=True)
    p.add_argument("--parquet3", required=True)
    p.add_argument("--label1", default="ribosome-less")
    p.add_argument("--label2", default="mock TadA")
    p.add_argument("--label3", default="query library")
    p.add_argument("--ref", required=True)
    p.add_argument("--gtf", required=True)
    p.add_argument("--output", default="bernoulliShadow")
    p.add_argument("--window", type=int,   default=30)
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
    print(f"label3={args.label3!r}  output={args.output!r}")
    out = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    # train_df1 = load_all_parquet_chunks(args.parquet1)
    # train_df2 = load_all_parquet_chunks(args.parquet2)
    # query_df = load_all_parquet_chunks(args.parquet3)
    train_df1 = pd.read_parquet(args.parquet1)
    train_df2 = pd.read_parquet(args.parquet2)
    query_df = pd.read_parquet(args.parquet3)

    print(f"Loaded {len(train_df1):,} reads from {args.label1}.", file=sys.stderr)
    print(f"Loaded {len(train_df2):,} reads from {args.label2}.", file=sys.stderr)
    print(f"Loaded {len(query_df):,} reads from {args.label3}.", file=sys.stderr)

    if args.cds_spanning:
        print("CDS spanning filter: ON (query only)", file=sys.stderr)
    if args.require_his_codon:
        print("His codon filter: ON (genes need >=1 CAT/CAC)",
              file=sys.stderr)
    if args.min_edit_freq > 0.0:
        print(f"  Query edit-freq filter: ON "
              f"(global_edit_freq >= {args.min_edit_freq})", file=sys.stderr)

    print("\nParsing GTF...", file=sys.stderr)
    genes = parse_gtf(args.gtf)
    print(f"{len(genes):,} genes.", file=sys.stderr)

    ref_fasta = pysam.FastaFile(args.ref)

    pdf_dir = Path(f"{out}_gene_pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    n_pass = 0

    gene_names = list(genes.keys())
    print(f"\nPass 1: scanning {len(gene_names):,} genes "
          f"(summary + ranking)...", file=sys.stderr)

    # Diagnostics
    max_bg = 0
    max_qry = 0

    gene_span_counts = {}  # gname -> n_query_reads (ranking metric)
    model_dict = {} # dictionary to hold model information per passing gene
    for i, gname in enumerate(gene_names):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(gene_names)} scanned, "
                  f"{n_pass} passing...", file=sys.stderr)

        gene = genes[gname]
        gene_len = cds_length(gene)

        his_positions = find_his_codon_tx_positions(ref_fasta, gene)
        if args.require_his_codon and len(his_positions) == 0:
            continue

        # Background: ALWAYS all overlapping reference reads (never spanning),
        # never edit-freq filtered — maximum per-position support.
        t1 = get_gene_df(train_df1, gene, cds_spanning=False)
        t2 = get_gene_df(train_df2, gene, cds_spanning=False)

        # Query: the reads we make protection calls on.
        df_qry = get_gene_df(query_df, gene, cds_spanning=args.cds_spanning,
                             min_edit_freq=args.min_edit_freq)

        max_bg = max(max_bg, len(t1))
        max_qry = max(max_qry, len(df_qry))

        # Two independent coverage thresholds
        if len(t1) < args.min_coverage:
            continue
        if len(df_qry) < args.min_query_reads:
            continue

        gpos_to_tx = _gpos_to_tx_map(gene, ref_fasta)
        tx_lo = min(gpos_to_tx.values())
        tx_hi = max(gpos_to_tx.values())

        # Train the model on the first two libraries
        model_dict[gname] = train(t1, t2, gpos_to_tx=gpos_to_tx)
        print("Trained gene model: ", gname, file=sys.stderr)
        # Classify
        rows = []
        for _, row in df_qry.iterrows():
            # rows.extend(classify_positions(
            #     model_dict[gname], row['read_id'], row['chrom'],
            #     row['edit_string'], row['absolute_indices'], gpos_to_tx, use_prior=True
            # ))
            rows.extend(classify_positions_windowed(
                model_dict[gname], row['read_id'], row['chrom'],
                row['edit_string'], row['absolute_indices'], gpos_to_tx,
                window=args.window, coord="tx", use_prior=True
            ))
        df = pd.DataFrame(rows)
        df.to_parquet(f"{out}_{gname}_calls.parquet", index=False)
        # plot_pb_by_tx_pyx(
        #     gname, df, his_positions, tx_lo, tx_hi,
        #     pdf_path=pdf_dir / f"{gname}.pdf",
        #     label1="P(Shadow)", label2=args.label3,
        #     ref_cov=model_dict[gname]["covA"]
        # )
        plot_signed_log_pyx(
            gname, df, his_positions, tx_lo, tx_hi,
            pdf_path=pdf_dir / f"{gname}.pdf",
            label1="P(Protection)", label2=args.label3,
            ref_cov=model_dict[gname]["covA"]
        )
        # print(calls_dict)
        # convert to
if __name__ == "__main__":
    Tee()
    main()




