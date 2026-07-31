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
import scipy.stats
from logJosh import Tee
import json
import pickle

HIS_CODONS = {"CAT", "CAC"}
def complement_base(b: str) -> str:
    return b.translate(str.maketrans("ACGTacgt", "TGCAtgca"))

def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]

_TEX_ESCAPE = {
    "\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "$": r"\$",
    "#": r"\#", "&": r"\&", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}

def tex_escape(s) -> str:
    """
    PyX's canvas.text() runs its argument through TeX, where _ % $ # & { } ~ ^
    \\ are all special characters -- an unescaped one (e.g. a --label or
    gene_name containing "_") crashes with "Missing $ inserted." deep in
    pyx.text. Everything placed in a plot title/label here is user- or
    data-supplied (CLI --label, GTF gene_name, read_id), so escape it first.
    """
    return "".join(_TEX_ESCAPE.get(ch, ch) for ch in str(s))

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
    never converts the unwanted columns (read_sequence, aligned_pairs, ...)
    to pandas objects in the first place -- that Arrow-to-pandas conversion,
    not the final DataFrame's resident size, is what actually blows up peak
    RSS on a real full-scale library (confirmed in trainHMMPerGene.py: same
    bug, 51.9GB peak with all columns vs. 3.67GB with columns= passed here).
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

def load_model_from_pickle(pickle_path: str) -> dict:
    with open(pickle_path, "rb") as f:
        return pickle.load(f)

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

def _gene_introns(gene: dict) -> list:
    """
    Genomic-order (start, end) intron intervals implied by gaps between the
    gene's annotated exons. gene["exons"] is sorted in TRANSCRIPT order
    (reversed for minus strand, see parse_gtf), so this re-sorts genomically
    first -- introns are a genomic concept, independent of strand. Empty
    for single-exon (intron-less) genes.
    """
    exons_sorted = sorted(gene["exons"], key=lambda x: x[0])
    introns = []
    for (_s1, e1), (s2, _e2) in zip(exons_sorted, exons_sorted[1:]):
        if s2 > e1:
            introns.append((e1, s2))
    return introns


def _read_has_unexplained_gap(absolute_indices, edit_string, introns: list, max_gap_nt: int) -> bool:
    """
    True if this read's alignment contains a reference-skipped stretch
    (a deletion or intron, in CIGAR terms) spanning more than max_gap_nt
    that does NOT overlap one of the gene's annotated introns.

    IMPORTANT: a real intron/deletion does NOT show up as a jump or a None
    in absolute_indices -- pysam's get_aligned_pairs() (see
    shadowingBamToParquetWithGTF2.py's get_absolute_positions) still
    reports the real, individually-incrementing reference position for
    every base spanned by a skip, it just has no corresponding read base.
    The only place that's recorded is edit_string=='2' at those same
    positions (shadowingBamToParquetWithGTF2.py sets edit=2 whenever
    read_pos is None, i.e. a skip, alongside the correct ref_pos). So gap
    detection has to key off runs of edit_string=='2' with a non-null
    absolute_indices value, not off jumps in the ref values themselves --
    confirmed on real data (an RPS13 read whose 539 intron positions were
    all present, correctly incrementing, and all edit_string=='2').

    (edit_string=='2' also marks INSERTIONS, i.e. ref_pos is None -- those
    already show up as None in absolute_indices and are skipped by the
    null-check below, so they don't get mistaken for a reference gap.)
    """
    run_vals = []

    def _flush():
        if not run_vals:
            return False
        lo, hi = min(run_vals), max(run_vals) + 1
        return hi - lo > max_gap_nt and not any(lo < ihi and hi > ilo for ilo, ihi in introns)

    for i, x in enumerate(absolute_indices):
        is_null = x is None or (isinstance(x, float) and x != x)
        is_gap_site = (not is_null) and i < len(edit_string) and edit_string[i] == "2"
        if is_gap_site:
            run_vals.append(float(x))
            continue
        if _flush():
            return True
        run_vals = []
    return _flush()


def get_gene_df(df_all: pd.DataFrame, gene: dict,
                cds_spanning: bool = False,
                min_edit_freq: float = 0.0,
                drop_unexplained_gaps: bool = False,
                max_gap_nt: int = 20) -> pd.DataFrame:
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

    If drop_unexplained_gaps is True, also drop reads with a gap (see
    _read_has_unexplained_gap) larger than max_gap_nt between consecutive
    anchored positions that doesn't overlap one of the gene's annotated
    introns -- a likely chimeric/mis-aligned read rather than real
    splicing. Intron-less genes have zero annotated introns, so ANY such
    gap drops the read there. Applied last, on the already-narrowed subset,
    since it's a per-row Python-level check rather than a vectorised one.
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

    sub = df_all[mask]
    if (drop_unexplained_gaps and "absolute_indices" in sub.columns
            and "edit_string" in sub.columns and not sub.empty):
        introns = _gene_introns(gene)
        keep = ~sub.apply(
            lambda row: _read_has_unexplained_gap(
                row["absolute_indices"], row["edit_string"], introns, max_gap_nt),
            axis=1)
        sub = sub[keep]
    return sub

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

def classify_positions_hmm2(model, read_id, chrom, edit_string,
                           absolute_indices, gpos_to_tx,
                           mean_block_nt=None, coord="tx", use_prior=True):
    """
    Two-state HMM over the scored (Ref=A) sites of one read.
      state A = unprotected (ribosome absent, TadA edits at pA[tx])
      state B = protected   (ribosome bound, TadA blocked, pB[tx] ~ 0)
    Emissions are the trained rates, unchanged. Transitions encode
    'protection comes in contiguous ~mean_block_nt runs at ~occupancy'.
    """
    n_edit = len(edit_string)

    # ---- pass 1: collect scored sites (read order) -----------------------
    sites = []                       # (tx, ref_pos, bit, pA_i, pB_i)
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
        if tx not in model["pA"] or tx not in model["pB"]:     # need BOTH
            continue
        sites.append((tx, ref_pos, int(ev), model["pA"][tx], model["pB"][tx]))

    if not sites:
        yield {"read_id": read_id, "chrom": chrom, "absolute_indices": [], "tx": [],
               "labels": [], "P_A": [], "P_B": [], "logL_A": [], "logL_B": [],
               "surprise": [], "edits": [], "win_n": []}
        return

    key = (lambda s: s[0]) if coord == "tx" else (lambda s: s[1])

    # keep an explicit READ-ORDER INDEX alongside each site, so duplicate
    # coordinates can never collapse into one another
    indexed = list(enumerate(sites))                     # (read_idx, site)
    ordered = sorted(indexed, key=lambda p: key(p[1]))   # sorted by coordinate
    coords  = [key(s) for _ri, s in ordered]
    n = len(ordered)

    # ---- transitions from biology ---------------------------------------
    if mean_block_nt is None:
        mean_block_nt = model.get("rpf_len_nt", 30)
    occupancy = (1.0 - model["prior_A"]) if use_prior else 0.5
    a_BA = 1.0 / mean_block_nt                     # exit protected, per nt
    a_AB = a_BA * occupancy / (1.0 - occupancy)    # enter protected, per nt
    T  = np.array([[1 - a_AB, a_AB],
                   [a_BA,     1 - a_BA]])
    pi = np.array([1 - occupancy, occupancy])

    # gap correction: transition across d nt = T^d
    Tstep = [np.linalg.matrix_power(T, max(1, int(coords[k] - coords[k-1])))
             for k in range(1, n)]

    def emis(k):
        _tx, _rp, bit, pA_i, pB_i = ordered[k][1]
        return np.array([pA_i if bit else 1 - pA_i,
                         pB_i if bit else 1 - pB_i])

    # ---- forward-backward, scaled, written by Claude ---------------------------
    alpha = np.zeros((n, 2)); c = np.zeros(n)
    v = pi * emis(0); c[0] = v.sum(); alpha[0] = v / c[0]
    for k in range(1, n):
        v = (alpha[k-1] @ Tstep[k-1]) * emis(k)
        c[k] = v.sum(); alpha[k] = v / c[k]
    beta = np.zeros((n, 2)); beta[-1] = 1.0
    for k in range(n-2, -1, -1):
        beta[k] = (Tstep[k] @ (emis(k+1) * beta[k+1])) / c[k+1]
    post = alpha * beta
    post /= post.sum(1, keepdims=True)

    # ---- map back to READ ORDER by index, not by coordinate --------------
    pB_by_readidx  = {ri: float(post[j, 1]) for j, (ri, _s) in enumerate(ordered)}
    sur_by_readidx = {ri: -math.log10(max(float(c[j]), 1e-300))
                      for j, (ri, _s) in enumerate(ordered)}

    positions, txs, labels = [], [], []
    A, B, lA, lB, sur, eds, win_n = [], [], [], [], [], [], []
    for ri, s in enumerate(sites):                  # read order
        tx, ref_pos, bit, pA_i, pB_i = s
        pB_post = pB_by_readidx[ri]
        positions.append(ref_pos); txs.append(tx)
        A.append(1.0 - pB_post);  B.append(pB_post)
        lA.append(math.log10(pA_i if bit else 1 - pA_i))   # per-site emission
        lB.append(math.log10(pB_i if bit else 1 - pB_i))
        sur.append(sur_by_readidx[ri])                     # -log10 P(bit | past)
        labels.append("B" if pB_post >= 0.5 else "A")
        eds.append(bit); win_n.append(n)

    yield {"read_id": read_id, "chrom": chrom, "absolute_indices": positions,
           "tx": txs, "labels": labels, "P_A": A, "P_B": B,
           "logL_A": lA, "logL_B": lB, "surprise": sur,
           "edits": eds, "win_n": win_n}


# ─────────────────────────────────────────────────────────────────────────
# HMM3: explicit-duration (semi-Markov) version of the above.
#
# State A (unprotected) is unchanged: geometric dwell, rate a_AB to exit.
# State B (protected) no longer has a constant per-nt exit rate -- instead
# it has an explicit nt-length duration pmf D_B, so a candidate protected
# run competes against "does this length look like a ribosome footprint"
# rather than being explained equally well by any length (which is what a
# memoryless geometric does, and why it can't tell a real footprint apart
# from an arbitrarily-sized secondary-structure block).
#
# Segment length is measured on the observed-site grid (distance in nt from
# the last site before a run to the last site inside it), matching the same
# granularity classify_positions_hmm2 already accepts for its Tstep/gap
# correction -- the exact boundary between two consecutive scored sites is
# unidentifiable from data anyway, since nothing is observed in between.
#
# THIS IS A FIRST DRAFT, not yet validated against synthetic data with a
# known duration distribution -- do that before trusting it on real reads.
# In particular the segment-posterior rescaling (un-scaling alphaA/betaA
# back to true probabilities via the recorded log_c cumulative sums) is the
# trickiest part of this math and the most likely place for a subtle bug.
# ─────────────────────────────────────────────────────────────────────────

def _duration_pmf_default(mean_nt=50.0, sd_nt=6.0, dmax=120):
    """
    Discretized Gaussian duration pmf, D_B[d-1] = P(protected segment length
    == d nt) for d = 1..dmax. Placeholder so classify_positions_hmm3 is
    runnable/sanity-checkable before a Baum-Welch training loop exists to
    re-estimate D_B from real query data.
    """
    d = np.arange(1, dmax + 1, dtype=float)
    w = np.exp(-0.5 * ((d - mean_nt) / sd_nt) ** 2)
    return w / w.sum()


def _duration_pmf_bimodal(mean1, sd1, weight1, mean2, sd2, dmax):
    """
    Discretized two-component Gaussian mixture duration pmf -- for testing
    whether protected-segment lengths look like a mixture of single-
    ribosome and collided/disome footprints (see the ~50nt real-data
    finding) rather than one unimodal shape. weight1 is the mixture weight
    on the (mean1, sd1) component; the second component gets (1 - weight1).
    """
    d = np.arange(1, dmax + 1, dtype=float)
    dens1 = np.exp(-0.5 * ((d - mean1) / sd1) ** 2) / (sd1 * math.sqrt(2 * math.pi))
    dens2 = np.exp(-0.5 * ((d - mean2) / sd2) ** 2) / (sd2 * math.sqrt(2 * math.pi))
    mix = weight1 * dens1 + (1.0 - weight1) * dens2
    return mix / mix.sum()


def get_duration_pmf(mode="gaussian", dmax=120, **kwargs):
    if mode == "gaussian":
        return _duration_pmf_default(
            mean_nt=kwargs.get("mean_nt", 50.0),
            sd_nt=kwargs.get("sd_nt", 6.0),
            dmax=dmax)
    elif mode == "bimodal":
        return _duration_pmf_bimodal(
            mean1=kwargs.get("mean1", 50.0),
            sd1=kwargs.get("sd1", 4.0),
            weight1=kwargs.get("weight1", 0.6),
            mean2=kwargs.get("mean2", 80.0),
            sd2=kwargs.get("sd2", 6.0),
            dmax=dmax)
    else:
        raise ValueError(f"Unknown mode: {mode}")



def _duration_to_hazard(D_B):
    """
    Convert a duration pmf D_B[d-1] = P(length == d), d = 1..Dmax, into
    per-duration hazard rates: hazard[d-1] = P(segment ends at length d |
    it already reached length d) = D_B(d) / P(length >= d). The last entry
    is forced to 1.0 -- a segment can't run past Dmax by construction.
    """
    Dmax = len(D_B)
    tail = np.cumsum(D_B[::-1])[::-1]        # tail[i] = P(length >= i+1)
    hazard = np.array([D_B[i] / tail[i] if tail[i] > 1e-300 else 1.0
                       for i in range(Dmax)])
    hazard[-1] = 1.0
    return hazard


def plot_duration_hazard(D_B, output_path, title=None, a_AB=None):
    """
    Quick look at a duration pmf D_B and its implied hazard curve
    (_duration_to_hazard) side by side -- for eyeballing whether a candidate
    D_B (placeholder Gaussian/bimodal now, Baum-Welch-fit later) has the
    shape you expect before trusting it inside classify_positions_hmm3.
    Matplotlib rather than PyX/TeX -- this is a throwaway diagnostic, not a
    per-gene report figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D_B = np.asarray(D_B, dtype=float)
    hazard = _duration_to_hazard(D_B)
    d = np.arange(1, len(D_B) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    ax1.plot(d, D_B, color="tab:blue")
    ax1.set_ylabel("P(length = d)")
    ax1.set_title(title or "Duration pmf D_B")

    ax2.plot(d, hazard, color="tab:red")
    ax2.set_ylabel("hazard(d) = P(exit | length >= d)")
    ax2.set_xlabel("protected segment length (nt)")
    ax2.set_ylim(0, 1.05)

    if a_AB is not None:
        fig.suptitle(f"a_AB (A→B entry rate) = {a_AB:.4g}", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote duration/hazard plot to {output_path}", file=sys.stderr)


def _forward_backward_hsmm(coords, eA, eB, pi_A, a_AB, D_B):
    """
    Explicit-duration (semi-Markov) forward-backward for the 2-state model
    described in the module notes above, via the standard "expanded state"
    equivalence: any discrete duration distribution can be represented
    EXACTLY as an ordinary Markov chain by giving state B one sub-state per
    possible elapsed duration (B_1..B_Dmax), each advancing to the next or
    exiting to A with a duration-specific hazard rate. That reduces the
    whole thing to a plain forward-backward over a bigger state space -- the
    same recursion classify_positions_hmm2 already uses, just more states --
    instead of custom segment-indexed math.

    (An earlier version of this function used hand-derived segment-indexed
    recursions directly on the sparse site list. It had a real bug -- P_B
    blowing up past 1, log-likelihood off by ~9 nats -- caught by
    cross-checking against exactly this expanded-chain construction used as
    a brute-force reference. Since the reference checked out and is far
    simpler to reason about, it's now the actual implementation rather than
    just a validation tool.)

    Runs at full nucleotide resolution across [coords[0], coords[-1]], not
    skipping between sites the way the 2-state Tstep/gap trick does, since
    duration bookkeeping needs single-nt granularity. Fine computationally
    for one gene's span (at most a few kb).

    coords, eA, eB: as before, per SCORED SITE (sparse)
    D_B: duration pmf, D_B[d-1] = P(protected segment length == d nt)

    Returns:
      post_B       -- np.array, P(site k is state B | all observations)
      expected_len -- dict {d: expected number of length-d segments implied
                      by this read} -- this read's contribution to a
                      Baum-Welch M-step's duration histogram: the standard
                      xi-statistic for the B_d -> A transition
      loglik       -- natural-log likelihood of this read under the model
      extra        -- dict with the two sufficient statistics an M-step for
                      a_AB needs, pooled the same way as expected_len:
                        "expected_A_occupancy" = expected nt spent in state A
                        "expected_AB_starts"   = expected number of A->B_1
                                                 (new block) events
    """
    Dmax   = len(D_B)
    hazard = _duration_to_hazard(D_B)
    S = 1 + Dmax    # state 0 = A, states 1..Dmax = B_1..B_Dmax
    ## Build Transition matrix T[s_from, s_to] = P(next_state | current_state)
    T = np.zeros((S, S))
    T[0, 0] = 1 - a_AB
    T[0, 1] = a_AB
    for d in range(1, Dmax):            # B_d (1-indexed) is row/col d
        T[d, d + 1] = 1 - hazard[d - 1]
        T[d, 0]     = hazard[d - 1]
    T[Dmax, 0] = 1.0                    # forced exit at max duration
    ## Initial state distribution: pi[s] = P(state at t=0)
    pi = np.zeros(S)
    pi[0] = pi_A
    pi[1] = 1.0 - pi_A

    L = int(coords[-1] - coords[0]) + 1
    site_at = {int(c - coords[0]): i for i, c in enumerate(coords)}

    def emis_vec(t):
        e = np.ones(S)
        i = site_at.get(t)
        if i is not None:
            e[0]  = eA[i]
            e[1:] = eB[i]
        return e

    alpha = np.zeros((L, S)); c = np.zeros(L)
    v = pi * emis_vec(0); c[0] = v.sum(); alpha[0] = v / c[0]
    for t in range(1, L):
        v = (alpha[t - 1] @ T) * emis_vec(t)
        c[t] = v.sum(); alpha[t] = v / c[t]

    beta = np.zeros((L, S)); beta[-1] = 1.0
    for t in range(L - 2, -1, -1):
        beta[t] = (T @ (emis_vec(t + 1) * beta[t + 1])) / c[t + 1]

    post = alpha * beta
    post /= post.sum(axis=1, keepdims=True)

    post_B = np.array([post[int(c_ - coords[0]), 1:].sum() for c_ in coords])
    loglik = float(np.sum(np.log(c)))

    # expected_len[d]: expected number of "B_d -> A" exits across the whole
    # read -- the standard Baum-Welch xi-statistic for that one transition,
    # summed over every nt position. This is exactly what an M-step would
    # accumulate across many reads to re-estimate D_B.
    expected_len = collections.defaultdict(float)
    expected_AB_starts = 0.0
    for t in range(L - 1):
        e_next = emis_vec(t + 1)
        for d in range(1, Dmax + 1):
            if alpha[t, d] <= 0.0:
                continue
            xi = alpha[t, d] * T[d, 0] * e_next[0] * beta[t + 1, 0] / c[t + 1]
            if xi > 1e-12:
                expected_len[d] += xi
        if alpha[t, 0] > 0.0:
            expected_AB_starts += alpha[t, 0] * T[0, 1] * e_next[1] * beta[t + 1, 1] / c[t + 1]

    # expected_A_occupancy: expected total nt spent in state A, i.e.
    # sum_t gamma_t(A) -- gamma is just post[:,0] here (already normalized).
    expected_A_occupancy = float(post[:, 0].sum())

    extra = {"expected_A_occupancy": expected_A_occupancy,
             "expected_AB_starts": expected_AB_starts}

    return post_B, dict(expected_len), loglik, extra


def classify_positions_hsmm(model, read_id, chrom, edit_string,
                            absolute_indices, gpos_to_tx,
                            coord="tx", D_B=None, a_AB=None):
    """
    Explicit-duration (semi-Markov) counterpart to classify_positions_hmm2.
    Same output schema (one dict per read, yielded) so this is a drop-in
    for write_shadow_calls_to_df / the plotting functions -- plus one extra
    key, "expected_seg_len", this read's contribution to the segment-length
    histogram a future Baum-Welch training loop would accumulate across
    many reads to re-estimate D_B.

    D_B, a_AB: pulled from `model` if not passed explicitly
    (model["D_B"], model["a_AB"]) -- these are exactly the two things a
    training loop would fit from real query data. Falls back to a
    placeholder Gaussian D_B and the same a_AB formula
    classify_positions_hmm2 uses, so this runs before that training loop
    exists.

    NOTE: "surprise" (per-site -log10 P(bit | past)) doesn't have as clean
    a per-site decomposition here as it did under hmm2's per-step scaling
    factors, since a single segment can span many sites at once. Left as
    NaN for now -- needs its own derivation, not a guessed placeholder.
    """
    n_edit = len(edit_string)

    sites = []
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
        if tx not in model["pA"] or tx not in model["pB"]:
            continue
        sites.append((tx, ref_pos, int(ev), model["pA"][tx], model["pB"][tx]))

    if not sites:
        yield {"read_id": read_id, "chrom": chrom, "absolute_indices": [], "tx": [],
               "labels": [], "P_A": [], "P_B": [], "logL_A": [], "logL_B": [],
               "surprise": [], "edits": [], "win_n": [], "expected_seg_len": {}}
        return

    key = (lambda s: s[0]) if coord == "tx" else (lambda s: s[1])
    indexed = list(enumerate(sites))
    ordered = sorted(indexed, key=lambda p: key(p[1]))
    coords  = np.array([key(s) for _ri, s in ordered], dtype=float)
    n = len(ordered)

    if D_B is None:
        D_B = model.get("D_B")
        if D_B is None:
            D_B = _duration_pmf_default(mean_nt=model.get("rpf_len_nt", 50))
    if a_AB is None:
        a_AB = model.get("a_AB")
        if a_AB is None:
            occupancy     = 1.0 - model["prior_A"]
            mean_block_nt = model.get("rpf_len_nt", 50)
            a_BA          = 1.0 / mean_block_nt
            a_AB          = a_BA * occupancy / (1.0 - occupancy)

    pi_A = model["prior_A"]

    bits    = [s[2] for _ri, s in ordered]
    pA_list = [s[3] for _ri, s in ordered]
    pB_list = [s[4] for _ri, s in ordered]
    eA = np.array([pA_list[i] if bits[i] else 1 - pA_list[i] for i in range(n)])
    eB = np.array([pB_list[i] if bits[i] else 1 - pB_list[i] for i in range(n)])

    post_B, expected_len, _loglik, _extra = _forward_backward_hsmm(
        coords, eA, eB, pi_A, a_AB, D_B)

    pB_by_readidx = {ri: float(post_B[j]) for j, (ri, _s) in enumerate(ordered)}

    positions, txs, labels = [], [], []
    A, B, lA, lB, sur, eds, win_n = [], [], [], [], [], [], []
    for ri, s in enumerate(sites):
        tx, ref_pos, bit, pA_i, pB_i = s
        pB_post = pB_by_readidx[ri]
        positions.append(ref_pos); txs.append(tx)
        A.append(1.0 - pB_post);  B.append(pB_post)
        lA.append(math.log10(pA_i if bit else 1 - pA_i))
        lB.append(math.log10(pB_i if bit else 1 - pB_i))
        sur.append(float("nan"))               # see docstring note above
        labels.append("B" if pB_post >= 0.5 else "A")
        eds.append(bit); win_n.append(n)

    yield {"read_id": read_id, "chrom": chrom, "absolute_indices": positions,
           "tx": txs, "labels": labels, "P_A": A, "P_B": B,
           "logL_A": lA, "logL_B": lB, "surprise": sur,
           "edits": eds, "win_n": win_n, "expected_seg_len": expected_len}


def write_shadow_calls_to_df(gene, df_qry, records, read_edits,
                             ref_cov, gpos_to_tx, tx_to_gpos,
                             min_win_n=1,
                             cds_start_tx=0, cds_end_tx=None):
    """
    Per-read, per-scored-site records in read-level format, mirroring the
    source parquet schema. NOT thresholded on P_B -- every scored site for
    every read (subject only to the min_win_n read-level quality gate)
    gets a row here, so downstream consumers apply whatever P_B cutoff they
    want themselves (see e.g. polysomeShadowHMMQC.py's PROB_CUTOFFS sweep).

    This used to hard-filter to P_B>=prob_threshold at write time (dropping
    both individual sub-threshold sites AND entire reads with zero sites
    clearing it), which (a) discarded the continuous posterior signal
    needed for e.g. a per-position ribo-seq correlation track, and (b) made
    "reads observed for this gene" undercounted in anything computed from
    this file (a read with zero above-threshold sites simply never
    appeared), which quietly biased any per-read rate (false-positive rate
    on a ribosome-less control, e.g.) computed downstream. Un-thresholding
    fixes both at the source instead of working around them per-consumer.

    Serves all three models (window / Markov / HMM) via the shared P_B schema.
    """
    qry_by_id = {row.read_id: row for row in df_qry.itertuples()}

    def region_of(txp):
        if txp < cds_start_tx:
            return "UTR5"
        if cds_end_tx is not None and txp >= cds_end_tx:
            return "UTR3"
        return "CDS"

    out_rows = []
    for rec in records:
        rid = rec["read_id"]

        site_idx = [k for k in range(len(rec["tx"]))
                    if rec["win_n"][k] >= min_win_n]
        if not site_idx:
            continue

        src = qry_by_id.get(rid)
        if src is None:
            continue
        src = src._asdict()

        tx_pos  = [rec["tx"][k]                   for k in site_idx]
        gpos    = [tx_to_gpos.get(rec["tx"][k])   for k in site_idx]
        regions = [region_of(t)                   for t in tx_pos]
        P_B     = [rec["P_B"][k]                  for k in site_idx]
        P_A     = [rec["P_A"][k]                  for k in site_idx]
        edit    = [rec["edits"][k]                for k in site_idx]
        rcov    = [ref_cov.get(rec["tx"][k])      for k in site_idx]

        row = dict(src)
        row.pop("Index", None)
        row.update({
            "shadow_gene":    gene,
            "shadow_tx_pos":  tx_pos,
            "shadow_gpos":    gpos,
            "shadow_region":  regions,
            "shadow_P_B":     P_B,
            "shadow_P_A":     P_A,
            "shadow_edit":    edit,
            "shadow_ref_cov": rcov,
            "n_scored_sites": len(site_idx),
            "n_sites_utr5":   regions.count("UTR5"),
            "n_sites_cds":    regions.count("CDS"),
            "n_sites_utr3":   regions.count("UTR3"),
            "max_P_B":        max(P_B),
        })
        out_rows.append(row)

    return pd.DataFrame(out_rows)

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
           f"{tex_escape(gene_name)} - {tex_escape(label2)}",
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
               tex_escape(label_txt), [pyx_text.valign.middle, pyx_text.size.tiny])

    c.writePDFfile(pdf_path)


def plot_signed_log_pyx(gene_name, df, his_positions, tx_lo, tx_hi, pdf_path,
                        label1="A", label2="B", label3="sample",
                        ref_cov_A=None, ref_cov_B=None,
                        tx_col="tx", pa_col="P_A", pb_col="P_B", edit_col="edits",
                        pct=(5, 95), num_reads=10, eps=1e-6):
    from pyx import canvas, graph, color, style, text as pyx_text
    from collections import defaultdict
    import numpy as np, math

    def nice_parter(lim):
        """explicit tick spacing so PyX never has to SEARCH for a partition"""
        import math
        from pyx import graph
        if not (lim > 0) or math.isinf(lim) or math.isnan(lim):
            return None
        raw = lim / 2.0
        mag = 10.0 ** math.floor(math.log10(raw))
        for m in (1, 2, 2.5, 5, 10):
            if raw <= m * mag:
                return graph.axis.parter.linear(tickdists=[m * mag])
        return None

    def safe_lim(v, floor=0.2):
        """guard against nan / inf / zero collapsing the axis"""
        import math
        try:
            v = float(v)
        except Exception:
            return floor
        if math.isnan(v) or math.isinf(v) or v <= 0:
            return floor
        return max(v, floor)

    col_qry  = color.cmyk(1, 0.5, 0, 0)       # blue  — P_A trace (flipped, +)
    col_pb   = color.cmyk(0.4, 1, 0, 0)       # purple — P_B trace (unflipped, -)
    col_his  = color.cmyk(0, 1, 1, 0)
    col_sig  = color.cmyk(0, 0, 0, 0.4)
    col_edit = color.cmyk(0, 0, 0, 1)
    col_no   = color.cmyk(0, 0.8, 1, 0)
    col_covA = color.cmyk(0.7, 0, 0.7, 0.1)   # green — ref A coverage
    col_covB = color.cmyk(0, 0.5, 1, 0.1)     # orange — ref B coverage

    pdf_path = str(pdf_path)
    x_min, x_max = tx_lo, tx_hi
    panel_w, tick_h, read_h, gap, cov_h = 12, 0.25, 1.2, 0.6, 1.5

    # the transform: log-scale the less-likely probability, sign per the rule.
    # NO branching needed -- the likely class collapses to ~0 on its own.
    def yA(pa): return -math.log10(max(pa, eps))     # P_A less likely -> flip -> positive
    def yB(pb): return  math.log10(max(pb, eps))     # P_B less likely -> keep -> negative

    # aggregate per-tx distributions of both transformed traces across reads
    ya_by, yb_by = defaultdict(list), defaultdict(list)
    for tx_l, pa_l, pb_l in zip(df[tx_col], df[pa_col], df[pb_col]):
        for t, pa, pb in zip(tx_l, pa_l, pb_l):
            ya_by[int(t)].append(yA(pa)); yb_by[int(t)].append(yB(pb))

    items   = sorted(ya_by)
    xs      = list(items)
    ya_mean = [float(np.mean(ya_by[t]))               for t in items]
    ya_lo   = [float(np.percentile(ya_by[t], pct[0])) for t in items]
    ya_hi   = [float(np.percentile(ya_by[t], pct[1])) for t in items]
    yb_mean = [float(np.mean(yb_by[t]))               for t in items]
    yb_lo   = [float(np.percentile(yb_by[t], pct[0])) for t in items]
    yb_hi   = [float(np.percentile(yb_by[t], pct[1])) for t in items]
    cov_self = {t: len(ya_by[t]) for t in items}

    # INDEPENDENT limits: blue caps ~+3 (evidence asymmetry), purple floors at -6 (eps)
    hi_vals  = [abs(v) for v in ya_hi + ya_mean] or [1.0]
    lo_vals  = [abs(v) for v in yb_lo + yb_mean] or [1.0]
    y_lim_hi = safe_lim(max(hi_vals) * 1.1)
    y_lim_lo = safe_lim(max(lo_vals) * 1.1)

    def draw_his(g):
        for hp in his_positions:
            g.plot(graph.data.function(f"x(y)={hp}", min=-y_lim_lo, max=y_lim_hi),
                   [graph.style.line([col_his, style.linewidth.thin, style.linestyle.solid])])
    def draw_zero(g):
        g.plot(graph.data.function("y(x)=0", min=x_min, max=x_max),
               [graph.style.line([col_sig, style.linewidth.thin, style.linestyle.dashed])])

    c = canvas.canvas()
    meta_ypos = num_reads * (read_h + tick_h + gap) + gap * 2

    # ── meta panel: both traces, mean + percentile bounds ────────────────────
    g_meta = graph.graphxy(
        width=panel_w, height=3, xpos=0, ypos=meta_ypos,
        x=graph.axis.linear(min=x_min, max=x_max, title="Position Along Transcript (nt)"),
        y=graph.axis.linear(min=-y_lim_lo, max=y_lim_hi, title="log10 P(Protection)",
                            parter=nice_parter(y_lim_lo + y_lim_hi)),
    )
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

    # ── coverage panel: both reference tracks ────────────────────────────────
    cov_ypos = meta_ypos + 3 + gap
    title_y  = cov_ypos + 3 + 0.4

    cov_tracks = []
    if ref_cov_A:
        cov_tracks.append((ref_cov_A, col_covA, label1))
    if ref_cov_B:
        cov_tracks.append((ref_cov_B, col_covB, label2))
    if not cov_tracks and cov_self:
        cov_tracks.append((cov_self, col_covA, "query"))

    if cov_tracks:
        cov_y_max = safe_lim(max(max(d.values()) for d, _, _ in cov_tracks) * 1.1, floor=1.0)
        g_cov = graph.graphxy(width=panel_w, height=cov_h, xpos=0, ypos=cov_ypos,
                              x=graph.axis.linkedaxis(g_meta.axes["x"]),
                              y=graph.axis.linear(min=0, max=cov_y_max, title="Ref cov",
                                                  parter=nice_parter(cov_y_max)))

        for hp in his_positions:
            g_cov.plot(graph.data.function(f"x(y)={hp}", min=0, max=cov_y_max),
                       [graph.style.line([col_his, style.linewidth.thin, style.linestyle.solid])])
        for d, col, _lab in cov_tracks:
            g_cov.plot(graph.data.points(sorted(d.items()), x=1, y=2),
                       [graph.style.line([col, style.linewidth.normal, style.linestyle.solid])])
        c.insert(g_cov)

        # inline legend, top-left inside the coverage panel
        for kk, (_d, col, lab) in enumerate(cov_tracks):
            c.text(g_cov.xpos + 0.2, g_cov.ypos + cov_h - 0.18 - kk * 0.28, tex_escape(lab),
                   [pyx_text.halign.left, pyx_text.valign.top, pyx_text.size.tiny, col])
        title_y = g_cov.ypos + g_cov.height + 0.4

    c.text(g_meta.xpos + g_meta.width / 2., title_y,
           f"{tex_escape(gene_name)} - {tex_escape(label3)}",
           [pyx_text.halign.center, pyx_text.size.normalsize])

    # ── individual read panels: upper trace / edit ticks / lower trace ───────
    n_show = min(num_reads, len(df))
    for jj in range(n_show):
        row = df.iloc[jj]
        tx_r = list(row[tx_col]); pa_r = list(row[pa_col]); pb_r = list(row[pb_col])
        if not tx_r:
            continue
        order = np.argsort(tx_r)
        tx_s  = [tx_r[k] for k in order]
        yaS   = [yA(pa_r[k]) for k in order]
        ybS   = [yB(pb_r[k]) for k in order]

        ypos   = (num_reads - 1 - jj) * (read_h + tick_h + gap)
        half_h = read_h / 2.0

        # lower half: unprotected trace (negative), below the ticks
        g_lo = graph.graphxy(width=panel_w, height=half_h, xpos=0, ypos=ypos,
                             x=graph.axis.linkedaxis(g_meta.axes["x"]),
                             y=graph.axis.linear(min=-y_lim_lo, max=0, title="",
                                                 parter=nice_parter(y_lim_lo)))
        for hp in his_positions:
            g_lo.plot(graph.data.function(f"x(y)={hp}", min=-y_lim_lo, max=0),
                      [graph.style.line([col_his, style.linewidth.thin, style.linestyle.solid])])
        g_lo.plot(graph.data.points(list(zip(tx_s, ybS)), x=1, y=2),
                  [graph.style.line([col_pb, style.linewidth.thin, style.linestyle.solid])])
        c.insert(g_lo)

        # upper half: protected trace (positive), above the ticks
        g_hi = graph.graphxy(width=panel_w, height=half_h, xpos=0,
                             ypos=ypos + half_h + tick_h,
                             x=graph.axis.linkedaxis(g_meta.axes["x"]),
                             y=graph.axis.linear(min=0, max=y_lim_hi, title="",
                                                 parter=nice_parter(y_lim_hi)))
        for hp in his_positions:
            g_hi.plot(graph.data.function(f"x(y)={hp}", min=0, max=y_lim_hi),
                      [graph.style.line([col_his, style.linewidth.thin, style.linestyle.solid])])
        g_hi.plot(graph.data.points(list(zip(tx_s, yaS)), x=1, y=2),
                  [graph.style.line([col_qry, style.linewidth.thin, style.linestyle.solid])])
        c.insert(g_hi)

        # centre strip: edit ticks, no y-axis labels
        g_ticks = graph.graphxy(width=panel_w, height=tick_h, xpos=0, ypos=ypos + half_h,
            x=graph.axis.linkedaxis(g_meta.axes["x"]),
            y=graph.axis.linear(min=0, max=1, parter=None, painter=None))
        edit_r = list(row[edit_col]) if edit_col in df.columns else None
        for k in order:
            v = int(edit_r[k]) if edit_r is not None else (0 if pa_r[k] < 0.5 else 1)
            col = col_edit if v == 1 else col_no
            g_ticks.plot(graph.data.function(f"x(y)={tx_r[k]}", min=0, max=1),
                         [graph.style.line([col, style.linewidth.thin, style.linestyle.solid])])
        c.insert(g_ticks)

        label_txt = str(row["read_id"])[:20] if "read_id" in df.columns else f"read {jj}"
        c.text(panel_w + 0.15, ypos + half_h + tick_h / 2., tex_escape(label_txt),
               [pyx_text.valign.middle, pyx_text.size.tiny])

    # print(f"y_lim_hi={y_lim_hi!r}  y_lim_lo={y_lim_lo!r}  "
    #       f"cov_y_max={cov_y_max if cov_tracks else None!r}")
    # c.writePDFfile(pdf_path)
    c.writePDFfile(pdf_path)

def parse_args():
    p = argparse.ArgumentParser(
        description="Shadow Hidden Markov Model."
    )
    p.add_argument("--parquet", required=True)
    p.add_argument("--label", default="sample")
    p.add_argument("--model", required=True, default=None, help="Trained model stored as pickle file")
    p.add_argument("--ref", required=True)
    p.add_argument("--gtf", required=True)
    p.add_argument("--output", default="HMM")
    p.add_argument("--window", type=int,   default=30)
    p.add_argument("--min_query_reads", type=int, default=10,
                   help="Minimum number of query (spanning) reads a gene must "
                        "have to be analyzed (default: 10). Separate from "
                        "--min_coverage, which gates the reference background.")
    p.add_argument("--num_reads",    type=int,   default=10)
    p.add_argument("--top_n_plots",  type=int,   default=10,
                   help="Number of genes to plot, ranked by number of "
                        "CDS-spanning query reads (default: 10)")
    p.add_argument("--require_his_codon", action="store_true",
                   help="Only process genes that contain at least one His "
                        "codon (CAT/CAC).")
    p.add_argument("--min_edit_freq", type=float, default=0.0,
                   help="Only consider query reads whose per-read "
                        "global_edit_freq is >= this value. Filters out "
                        "unedited/poorly-edited molecules that carry no "
                        "protection signal. Applied to query reads only, "
                        "not the background estimate. Default: 0.0 (off)")

    p.add_argument("--cds_spanning", action="store_true",
                   help="Only include reads whose alignment spans the full "
                        "CDS of the gene.")
    p.add_argument("--drop_unexplained_gaps", action="store_true",
                   help="Drop reads with an alignment gap (see --max_gap_nt) "
                        "that doesn't overlap one of the gene's annotated "
                        "introns -- likely a chimeric/mis-aligned read "
                        "rather than real splicing. Intron-less genes have "
                        "no annotated introns, so any such gap drops the read.")
    p.add_argument("--max_gap_nt", type=int, default=20,
                   help="Minimum gap size (nt) between consecutive anchored "
                        "positions to be considered for --drop_unexplained_gaps "
                        "(default: 20).")
    return p.parse_args()


def main():
    args = parse_args()
    out = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    needed_cols = ["chrom", "gene_strand", "read_start", "read_end",
                   "edit_string", "absolute_indices", "global_edit_freq",
                   "read_id"]
    query_df = load_all_parquet_chunks(
        args.parquet, columns=_available_columns(args.parquet, needed_cols))
    print(f"Loaded {len(query_df):,} reads from {args.label}.", file=sys.stderr)
    if args.model is not None:
        model_dict = load_model_from_pickle(args.model)
    else:
        # exit
        raise ValueError("No model provided. Please specify a trained model using --model.")

    if args.cds_spanning:
        print("CDS spanning filter: ON (query only)", file=sys.stderr)
    if args.require_his_codon:
        print("His codon filter: ON (genes need >=1 CAT/CAC)",
              file=sys.stderr)
    if args.min_edit_freq > 0.0:
        print(f"Query edit-freq filter: ON "
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

    shadow_call_frames = []
    shadow_calls_path = f"{out}_shadow_calls.parquet"
    ## Build D_B from the placeholder, then run the Hidden semi-Markov model
    D_B = get_duration_pmf(mode="gaussian", dmax=150)
    ## plot duration distribution
    plot_duration_hazard(D_B, f"{out}_duration_distribution.pdf", title=f"Duration distribution")
    for i, gname in enumerate(gene_names):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(gene_names)} scanned, "
                  f"{n_pass} passing...", file=sys.stderr)

        gene = genes[gname]
        gene_len = cds_length(gene)

        his_positions = find_his_codon_tx_positions(ref_fasta, gene)
        if args.require_his_codon and len(his_positions) == 0:
            continue

        # Query: the reads we make protection calls on.
        df_qry = get_gene_df(query_df, gene, cds_spanning=args.cds_spanning,
                             min_edit_freq=args.min_edit_freq,
                             drop_unexplained_gaps=args.drop_unexplained_gaps,
                             max_gap_nt=args.max_gap_nt)
        if len(df_qry) < args.min_query_reads:
            continue

        if gname not in model_dict:
            print(f"  {gname} skipped: no model found for this gene.", file=sys.stderr)
            continue

        n_pass += 1
        gpos_to_tx = _gpos_to_tx_map(gene, ref_fasta)
        tx_to_gpos = {tx: gp for gp, tx in gpos_to_tx.items()}
        tx_lo = min(gpos_to_tx.values())
        tx_hi = max(gpos_to_tx.values())

        rows = []
        for _, row in df_qry.iterrows():
            ## two-state HMM classification of each read, yielding a list of dicts
            # rows.extend(classify_positions_hmm2(
            #     model_dict[gname], row['read_id'], row['chrom'],
            #     row['edit_string'], row['absolute_indices'], gpos_to_tx,
            # mean_block_nt=50, coord="tx", use_prior=True)
            # )

            ## Hidden semi-Markov model classification of each read, yielding a list of dicts
            rows.extend(classify_positions_hsmm(
                model_dict[gname], row['read_id'], row['chrom'],
                row['edit_string'], row['absolute_indices'], gpos_to_tx, coord="tx", D_B=D_B)
            )

        df = pd.DataFrame(rows)
        # df.to_parquet(f"{out}_{gname}_calls.parquet", index=False)
        plot_pb_by_tx_pyx(
            gname, df, his_positions, tx_lo, tx_hi,
            pdf_path=pdf_dir / f"{gname}.pdf",
            label1="P(Shadow)", label2=args.label,
            ref_cov=model_dict[gname]["covA"]
        )
        plot_signed_log_pyx(
            gname, df, his_positions, tx_lo, tx_hi,
            pdf_path=pdf_dir / f"{gname}_log.pdf",
            label1="ribosome-less", label2="Mock TadA",
            label3=args.label,
            ref_cov_A=model_dict[gname]["covA"],
            ref_cov_B=model_dict[gname]["covB"]
        )
        gene_calls = write_shadow_calls_to_df(
            gname, df_qry, rows,  # `rows` = the classify records
            read_edits=None,
            ref_cov=model_dict[gname]["covA"],
            gpos_to_tx=gpos_to_tx, tx_to_gpos=tx_to_gpos,
            cds_start_tx=0, cds_end_tx=gene_len,  # your CDS bounds in tx coords
        )
        if not gene_calls.empty:
            shadow_call_frames.append(gene_calls)
    # print(calls_dict)
    # convert to
    if shadow_call_frames:
        shadow_df = pd.concat(shadow_call_frames, ignore_index=True)
        shadow_df.to_parquet(shadow_calls_path, index=False)
        print(f"wrote {len(shadow_df)} shadow-call reads across "
              f"{shadow_df['shadow_gene'].nunique()} genes -> {shadow_calls_path}",
              file=sys.stderr)
if __name__ == "__main__":
    Tee()
    main()