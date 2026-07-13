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

def train():
    pass

def classify():
    pass