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
import pandas as pd
from logJosh import Tee
import pickle

def complement_base(b: str) -> str:
    return b.translate(str.maketrans("ACGTacgt", "TGCAtgca"))

def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]

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

def train(A_df, B_df, alpha=1, beta=1, gpos_to_tx=None):
    pA, covA = build_frequency_df(A_df, gpos_to_tx, alpha, beta)
    pB, covB = build_frequency_df(B_df, gpos_to_tx, alpha, beta)

    w1, w0 = {}, {}
    for tx in pA.keys() & pB.keys():  # positions BOTH populations saw
        a, b = pA[tx], pB[tx]
        w1[tx] = math.log(a) - math.log(b)  # weight when the bit is 1
        w0[tx] = math.log(1 - a) - math.log(1 - b)  # weight when the bit is 0

    # nA, nB = len(A_df), len(B_df)
    # prior_log_odds = math.log(nA / nB) if nA and nB else 0.0
    prior_A = 0.8 # based off of 30 nt RPF and 150 nt spacing

    transA = train_transitions(A_df, gpos_to_tx, alpha, beta)
    transB = train_transitions(B_df, gpos_to_tx, alpha, beta)


    return {"pA": pA, "pB": pB, "covA": covA, "covB": covB,
            "w1": w1, "w0": w0,
            "prior_A": prior_A,
            "prior_log_odds": math.log(prior_A / (1 - prior_A)), "transA": transA, "transB": transB}

def parse_args():
    p = argparse.ArgumentParser(
        description="Bernoulli Naive Bayes Shadow Classifier."
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
    train_df1 = load_all_parquet_chunks(args.parquet1)
    train_df2 = load_all_parquet_chunks(args.parquet2)

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
        model_dict[gname] = train(t1, t2, gpos_to_tx=gpos_to_tx)
        print("Trained gene model: ", gname, file=sys.stderr)

        n_pass += 1

    # write to pickle
    with open(args.output, "wb") as f:
        pickle.dump(model_dict, f)
    print(f"stored model in {args.output}")
if __name__ == "__main__":
    Tee()
    main()