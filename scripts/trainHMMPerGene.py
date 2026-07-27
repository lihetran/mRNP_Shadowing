'''
July 20, 2026 LT

The purpose of this script is to minimize the memory overhead of reading in and training my HMM every time I run the model.
This script will take in shadow parquet files for a ribosome-less control and mock TadA libraries, and train a separate HMM for each gene. The trained HMMs will be stored in a dictionary and saved to a pickle file for later use.

inputs:
    -gtf: gtf
    -ref: ref
    -min_coverage: number of overlapping reads in both libraries per gene
    -parquet1: phenol-extracted, ribosome-less
    -parquet2: mock TadA
    -output: output pickle file to store the trained HMMs


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
import pyarrow.parquet as pq
from logJosh import Tee
import pickle

from runHMMPerGene import _forward_backward_hsmm, _duration_pmf_default
# NOTE: _gpos_to_tx_map is NOT imported -- this file already defines its own
# copy below (matching this codebase's existing per-script duplication
# convention); importing it too would just create a confusing shadow.

def complement_base(b: str) -> str:
    return b.translate(str.maketrans("ACGTacgt", "TGCAtgca"))

def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]

def _available_columns(parquet_dir: str, wanted: list) -> list:
    """Which of `wanted` columns actually exist in this dir's parquet schema
    (checked against just the first chunk -- every chunk in one library
    shares one schema). Some optional columns (e.g. global_edit_freq) aren't
    always present."""
    parquet_dir = Path(parquet_dir)
    chunks = sorted(parquet_dir.glob("*.parquet"))
    if not chunks:
        return wanted
    have = set(pq.read_schema(chunks[0]).names)
    return [c for c in wanted if c in have]


def load_all_parquet_chunks(parquet_dir: str, columns=None) -> pd.DataFrame:
    """
    columns, if given, is passed straight to pd.read_parquet so pyarrow
    never converts the unwanted columns to pandas objects in the first
    place. Trimming columns AFTER a full-column read doesn't help peak
    memory -- the Arrow-to-pandas conversion for big object columns
    (read_sequence, aligned_pairs, ...) is what actually blows up RSS, and
    that conversion has already happened by the time you slice columns off
    the result (confirmed: on a real 725,847-read library, full-column
    read+concat peaked at 51.9GB RSS vs. 3.67GB with columns= passed here).
    """
    parquet_dir = Path(parquet_dir)
    chunks = sorted(parquet_dir.glob("*.parquet"))
    if not chunks:
        return pd.DataFrame()
    dfs = [pd.read_parquet(c, columns=columns) for c in chunks]
    df  = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df):,} reads from {len(chunks)} chunk(s).",
          file=sys.stderr)
    return df

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

def train_transitions(ref_df, gpos_to_tx, alpha=1.0, beta=1.0,
                      edit_col="edit_string", ai_col="absolute_indices"):
    c = {0: [0, 0], 1: [0, 0]}          # c[prev][cur]
    first = [0, 0]
    for row in ref_df.itertuples():
        es = getattr(row, edit_col); ai = getattr(row, ai_col)
        seq = []
        for i, ref_pos in enumerate(ai):
            if ref_pos is None or (isinstance(ref_pos, float) and ref_pos != ref_pos):
                continue
            ref_pos = int(ref_pos)
            if ref_pos not in gpos_to_tx or i >= len(es):
                continue
            ev = es[i]
            if ev == "2":
                continue
            seq.append((gpos_to_tx[ref_pos], int(ev)))
        if not seq:
            continue
        seq.sort(key=lambda p: p[0])
        bits = [b for _t, b in seq]
        first[bits[0]] += 1
        for prev, cur in zip(bits, bits[1:]):
            c[prev][cur] += 1
    def smooth(n1, n0):
        return (n1 + alpha) / (n0 + n1 + alpha + beta)
    return {"p1_given0": smooth(c[0][1], c[0][0]),
            "p1_given1": smooth(c[1][1], c[1][0]),
            "pi1":       smooth(first[1], first[0])}

def train(A_df, B_df, alpha=1, beta=1, gpos_to_tx=None, block = 30):
    pA, covA = build_frequency_df(A_df, gpos_to_tx, alpha, beta)
    pB, covB = build_frequency_df(B_df, gpos_to_tx, alpha, beta)

    w1, w0 = {}, {}
    for tx in pA.keys() & pB.keys():  # positions BOTH populations saw
        a, b = pA[tx], pB[tx]
        w1[tx] = math.log(a) - math.log(b)  # weight when the bit is 1
        w0[tx] = math.log(1 - a) - math.log(1 - b)  # weight when the bit is 0

    # nA, nB = len(A_df), len(B_df)
    # prior_log_odds = math.log(nA / nB) if nA and nB else 0.0
    # prior_A = 0.8 # based off of 30 nt RPF and 150 nt spacing
    prior_A = 1.0 - (block / 150)

    transA = train_transitions(A_df, gpos_to_tx, alpha, beta)
    transB = train_transitions(B_df, gpos_to_tx, alpha, beta)


    return {"pA": pA, "pB": pB, "covA": covA, "covB": covB,
            "w1": w1, "w0": w0,
            "prior_A": prior_A,
            "prior_log_odds": math.log(prior_A / (1 - prior_A)), "transA": transA, "transB": transB}


# ─────────────────────────────────────────────────────────────────────────
# Baum-Welch for the explicit-duration (HMM3) model's transition parameters
# (D_B, a_AB). Emissions (pA/pB, from train() above) stay fixed -- see the
# design discussion this came out of: neither ribosome-less nor mock-TadA
# contains real occupancy-switching to learn transitions from (ribosome-less
# is always state A; mock-TadA is unedited, i.e. no signal either way), so
# this fits transitions from the real experimental query reads instead --
# the only population that actually contains a molecule switching between
# genuinely edited and genuinely protected stretches.
#
# Fit ONE shared D_B/a_AB pooled across every gene, not per-gene: most genes
# don't clear enough query reads to stably support a whole duration
# histogram (Dmax free parameters) on their own.
# ─────────────────────────────────────────────────────────────────────────

def collect_hsmm_training_reads(query_df, model_dict, genes, ref_fasta,
                                 min_sites=2):
    """
    Generator: for every gene with an already-trained model (pA/pB, from
    train()), stream that gene's query reads and yield each read's (coords,
    eA, eB) triplet -- the per-read input _forward_backward_hsmm needs.
    Pooled across every gene, since D_B/a_AB are fit as ONE shared
    distribution (see module notes above).

    This used to build and return one big list holding every read across
    every gene at once -- on a real full-scale query library that's a
    genuine OOM risk (confirmed: a real run was killed at 63GB RSS). It's a
    generator now specifically so train_hsmm_durations can re-stream this
    fresh each EM iteration instead of ever holding the whole read set in
    memory simultaneously. Only the columns actually used below are kept
    from query_df, since the raw parquet also carries several large
    per-read columns (aligned_pairs, read_sequence, ref_sequence_aligned,
    ...) this function never touches -- dropping them here keeps this
    function's own working set small regardless of what the caller does
    with its own reference to query_df.

    min_sites: skip reads with fewer than this many scored sites -- a read
    with 0-1 sites carries no information about how state changes with
    position, and just adds noise/cost to the E-step.
    """
    needed_cols = ["chrom", "gene_strand", "read_start", "read_end",
                   "edit_string", "absolute_indices"]
    if "global_edit_freq" in query_df.columns:
        needed_cols.append("global_edit_freq")
    query_df = query_df[needed_cols]

    for gname, gmodel in model_dict.items():
        gene = genes.get(gname)
        if gene is None:
            continue

        df_g = get_gene_df(query_df, gene, cds_spanning=False)
        if df_g.empty:
            continue

        gpos_to_tx = _gpos_to_tx_map(gene, ref_fasta)

        for row in df_g.itertuples():
            edit_str = row.edit_string
            abs_idx  = row.absolute_indices
            n_edit   = len(edit_str)

            sites = []
            for i, ref_pos in enumerate(abs_idx):
                if ref_pos is None or (isinstance(ref_pos, float) and ref_pos != ref_pos):
                    continue
                ref_pos = int(ref_pos)
                if ref_pos not in gpos_to_tx or i >= n_edit:
                    continue
                ev = edit_str[i]
                if ev == "2":
                    continue
                tx = gpos_to_tx[ref_pos]
                if tx not in gmodel["pA"] or tx not in gmodel["pB"]:
                    continue
                sites.append((tx, int(ev), gmodel["pA"][tx], gmodel["pB"][tx]))

            if len(sites) < min_sites:
                continue

            sites.sort(key=lambda s: s[0])
            coords = np.array([s[0] for s in sites], dtype=float)
            bits   = [s[1] for s in sites]
            pA_l   = [s[2] for s in sites]
            pB_l   = [s[3] for s in sites]
            eA = np.array([pA_l[i] if bits[i] else 1 - pA_l[i]
                           for i in range(len(sites))])
            eB = np.array([pB_l[i] if bits[i] else 1 - pB_l[i]
                           for i in range(len(sites))])
            yield (coords, eA, eB)


def train_hsmm_durations(reads, pi_A, D_B_init=None, a_AB_init=None,
                          max_iters=20, tol=1e-4, verbose=True):
    """
    Baum-Welch: fit a shared protected-state duration pmf D_B and entry rate
    a_AB by EM over a pool of reads (coords, eA, eB). Each iteration:
      E-step: run _forward_backward_hsmm on every read with the CURRENT
              D_B/a_AB, accumulating each read's expected_len (duration
              counts) and a_AB sufficient statistics, plus total
              log-likelihood for a convergence check.
      M-step: normalize the pooled duration counts into the new D_B;
              new a_AB = pooled A->B starts / pooled A occupancy.

    reads: EITHER a plain list of (coords, eA, eB) tuples (fine for small
    synthetic runs -- see test_hsmm.py, where materializing everything is
    simplest), OR a zero-argument callable that returns a fresh
    generator/iterable of the same tuples each time it's called -- use this
    for real-scale training, passing e.g.
        lambda: collect_hsmm_training_reads(query_df, model_dict, genes, ref_fasta)
    Since collect_hsmm_training_reads is a generator, each E-step re-streams
    reads from query_df instead of ever holding the full read set in memory
    at once, which is exactly what OOM'd a real training run at 63GB RSS.

    Returns (D_B, a_AB, history) where history is the total log-likelihood
    per iteration -- should increase monotonically iteration to iteration
    (that's the EM guarantee); if it doesn't, something's wrong upstream.
    """
    D_B  = D_B_init  if D_B_init  is not None else _duration_pmf_default()
    a_AB = a_AB_init if a_AB_init is not None else 0.01
    Dmax = len(D_B)

    history = []
    prev_ll = None

    for it in range(max_iters):
        total_len       = collections.defaultdict(float)
        total_A_occ     = 0.0
        total_AB_starts = 0.0
        total_ll        = 0.0

        current_reads = reads() if callable(reads) else reads
        for coords, eA, eB in current_reads:
            _post_B, expected_len, loglik, extra = _forward_backward_hsmm(
                coords, eA, eB, pi_A, a_AB, D_B)
            total_ll += loglik
            for d, w in expected_len.items():
                total_len[d] += w
            total_A_occ     += extra["expected_A_occupancy"]
            total_AB_starts += extra["expected_AB_starts"]

        history.append(total_ll)
        if verbose:
            print(f"  iter {it}: total logL = {total_ll:.2f}  "
                  f"(a_AB={a_AB:.5f})", file=sys.stderr)

        if prev_ll is not None and abs(total_ll - prev_ll) < tol * abs(prev_ll):
            if verbose:
                print(f"  converged after {it} iterations.", file=sys.stderr)
            break
        prev_ll = total_ll

        # ---- M-step ----
        len_sum = sum(total_len.values())
        if len_sum > 0:
            D_B = np.array([total_len.get(d, 1e-6) for d in range(1, Dmax + 1)])
            D_B = D_B / D_B.sum()
        if total_A_occ > 0:
            a_AB = total_AB_starts / total_A_occ

    return D_B, a_AB, history

def parse_args():
    p = argparse.ArgumentParser(
        description="Train HMM models for each gene."
    )
    p.add_argument("--parquet1", required=True)
    p.add_argument("--parquet2", required=True)
    p.add_argument("--label1", default="ribosome-less")
    p.add_argument("--label2", default="mock TadA")
    p.add_argument("--ref", required=True)
    p.add_argument("--gtf", required=True)
    p.add_argument("--min_coverage", type=float, default=100.0)
    p.add_argument("--output", default="gene_models.pickle")

    return p.parse_args()

def main():
    args = parse_args()

    out = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    # Only the columns get_gene_df/train() actually touch. This MUST be
    # passed as columns= into pd.read_parquet (not sliced off afterward) --
    # the raw parquet also carries several large per-read columns this
    # script never uses (read_sequence, read_sequence_aligned,
    # ref_sequence_aligned, aligned_pairs, gene_biotype, transcript_id,
    # gene_name, barcode, bar_seq), and pyarrow's Arrow-to-pandas conversion
    # for those columns is what actually blows up peak RSS -- that
    # conversion has already happened by the time a post-load
    # df[needed_cols] slice runs. Confirmed on a real 725,847-read library:
    # full-column read+concat peaked at 51.9GB RSS vs. 3.67GB with columns=
    # passed at read time.
    needed_cols = ["chrom", "gene_strand", "read_start", "read_end",
                   "edit_string", "absolute_indices", "global_edit_freq"]
    cols1 = _available_columns(args.parquet1, needed_cols)
    cols2 = _available_columns(args.parquet2, needed_cols)
    train_df1 = load_all_parquet_chunks(args.parquet1, columns=cols1)
    train_df2 = load_all_parquet_chunks(args.parquet2, columns=cols2)

    print(f"Loaded {len(train_df1):,} reads from {args.label1}.", file=sys.stderr)
    print(f"Loaded {len(train_df2):,} reads from {args.label2}.", file=sys.stderr)

    print("\nParsing GTF...", file=sys.stderr)
    genes = parse_gtf(args.gtf)
    print(f"{len(genes):,} genes.", file=sys.stderr)

    ref_fasta = pysam.FastaFile(args.ref)
    gene_names = list(genes.keys())
    print(f"\nPass 1: scanning {len(gene_names):,} genes "
          f"(summary + ranking)...", file=sys.stderr)

    model_dict = {}  # dictionary to hold model information per passing gene
    n_pass = 0
    for i, gname in enumerate(gene_names):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(gene_names)} scanned, "
                  f"{n_pass} passing...", file=sys.stderr)

        gene = genes[gname]
        gene_len = cds_length(gene)

        # Background: ALWAYS all overlapping reference reads (never spanning),
        # never edit-freq filtered — maximum per-position support.
        t1 = get_gene_df(train_df1, gene, cds_spanning=False)
        t2 = get_gene_df(train_df2, gene, cds_spanning=False)

        # Two independent coverage thresholds
        if len(t1) < args.min_coverage or len(t2) < args.min_coverage:  # both need to pass
            continue

        gpos_to_tx = _gpos_to_tx_map(gene, ref_fasta)
        tx_to_gpos = {tx: gp for gp, tx in gpos_to_tx.items()}

        # Train the model on the first two libraries
        model_dict[gname] = train(t1, t2, gpos_to_tx=gpos_to_tx, block = 50)
        print("Trained gene model: ", gname, file=sys.stderr)

        n_pass += 1

    # write to pickle
    with open(args.output, "wb") as f:
        pickle.dump(model_dict, f)
    print(f"stored model in {args.output}")
if __name__ == "__main__":
    Tee()
    main()