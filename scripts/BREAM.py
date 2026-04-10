'''
April 10, 2026 LT

This script will modify and adapt the algorithm in oligoShadowingJointProbability.py to call ribosome footprints on RNAs treated by TadA.
I will first compute the reference frequency of edits from a control library to then compute the probability of observing the same edit pattern on a different library.
Assuming similar edit frequencies between the two libraries, the biggest changes in editing should be from ribosome footprints or secondary structures.

inputs:
    --bam1: probably no drug control or phenol-extracted RNA \
    --bam2: probably with some ribosome stalling drug like 3-AT \
    --label1: name for bam1 \
    --label2: name for bam2 \
    --ref reference.fa \
    --gtf annotation.gtf \
    --output output_prefix \
    [--min_coverage 50] \
    [--min_mapq 20] \
    [--min_baseq 10] \

'''

import argparse
import sys
import re
import collections
from pathlib import Path

import pysam
import numpy as np
import pandas as pd


# ── Histidine codons (DNA, transcript coordinates) ──────────────────────────
HIS_CODONS = {"CAT", "CAC"}

# ─────────────────────────────────────────────────────────────────────────────
# 1. GTF parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_gtf(gtf_path: str) -> dict:
    """
    Parse GTF to get gene body extent and CDS intervals per gene.
    Returns:
        gene_name → {
            "chrom":      str,
            "strand":     "+" | "-",
            "gene_start": int (0-based),
            "gene_end":   int,
            "transcript": str,
            "gene_name":  str,
            "cds":        sorted list of (start0, end0)
        }
    gene_start/gene_end span the full gene body (from gene feature or
    min/max of CDS intervals as fallback).
    """
    genes = {}
    gene_extents = {}   # gene_name → (start, end)

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
                        "chrom":      chrom,
                        "strand":     strand,
                        "transcript": tid,
                        "gene_name":  gname,
                        "cds":        [],
                    }
                genes[gname]["cds"].append((start, end))

    # Attach gene body extent; fall back to min/max of CDS if no gene feature
    for gname, g in genes.items():
        if gname in gene_extents:
            g["gene_start"], g["gene_end"] = gene_extents[gname]
        else:
            all_starts = [s for s, e in g["cds"]]
            all_ends   = [e for s, e in g["cds"]]
            g["gene_start"] = min(all_starts)
            g["gene_end"]   = max(all_ends)

    # Sort CDS intervals in transcript order
    for g in genes.values():
        g["cds"].sort(key=lambda x: x[0],
                      reverse=(g["strand"] == "-"))
    return genes


# ─────────────────────────────────────────────────────────────────────────────
# 2. Coverage filtering
# ─────────────────────────────────────────────────────────────────────────────

def mean_cds_coverage(bam: pysam.AlignmentFile,
                      gene: dict,
                      min_mapq: int) -> float:
    """Mean per-position depth across all CDS intervals of a gene."""
    total_depth = 0
    total_bases = 0
    for (start, end) in gene["cds"]:
        for col in bam.pileup(
            gene["chrom"], start, end,
            truncate=True,
            min_mapping_quality=min_mapq,
            stepper="samtools",
        ):
            total_depth += col.nsegments
            total_bases += 1
    return total_depth / total_bases if total_bases > 0 else 0.0


def filter_high_coverage_genes(genes: dict,
                                bam1_path: str,
                                bam2_path: str,
                                min_coverage: float,
                                min_mapq: int) -> list:
    """
    Return list of gene_names where BOTH BAMs have mean CDS coverage
    >= min_coverage.
    """
    passing = []
    bam1 = pysam.AlignmentFile(bam1_path, "rb")
    bam2 = pysam.AlignmentFile(bam2_path, "rb")

    for i, (gname, gene) in enumerate(genes.items()):
        if (i + 1) % 100 == 0:
            print(f"  Checking coverage {i+1}/{len(genes)}…", file=sys.stderr)
        cov1 = mean_cds_coverage(bam1, gene, min_mapq)
        cov2 = mean_cds_coverage(bam2, gene, min_mapq)
        if cov1 >= min_coverage and cov2 >= min_coverage:
            passing.append(gname)

    bam1.close()
    bam2.close()
    print(f"  {len(passing):,} / {len(genes):,} genes pass coverage filter.",
          file=sys.stderr)
    return passing


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build per-position editing matrix for a gene
# ─────────────────────────────────────────────────────────────────────────────

def complement_base(b: str) -> str:
    return b.translate(str.maketrans("ACGTacgt", "TGCAtgca"))


def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]

def build_editing_matrix(bam: pysam.AlignmentFile,
                          ref_fasta: pysam.FastaFile,
                          gene: dict,
                          min_mapq: int,
                          min_baseq: int) -> pd.DataFrame:
    """
    For every position across the full gene body (gene_start to gene_end)
    in transcript order, compute:
        - ref_base (transcript coordinates)
        - A, G, C, T counts (transcript coordinates)
        - coverage
        - ag_edit_frac = G/(A+G) where ref_base == A, else NaN
        - in_cds: bool — whether position falls inside a CDS interval

    Strand handling: uses read.is_reverse XOR (strand=="-") to convert
    each read to transcript coordinates before counting.
    """
    chrom      = gene["chrom"]
    strand     = gene["strand"]
    gene_start = gene["gene_start"]
    gene_end   = gene["gene_end"]

    # Build a set of CDS genomic positions for fast lookup
    cds_positions = set()
    for (cs, ce) in gene["cds"]:
        cds_positions.update(range(cs, ce))

    records = []
    tx_pos  = 0

    # Walk all genomic positions across the gene body in transcript order
    if strand == "+":
        gpos_range = range(gene_start, gene_end)
    else:
        gpos_range = range(gene_end - 1, gene_start - 1, -1)

    for gpos in gpos_range:
        ref_base_genomic = ref_fasta.fetch(chrom, gpos, gpos + 1).upper()
        ref_base_tx = complement_base(ref_base_genomic) \
                      if strand == "-" else ref_base_genomic

        counts = collections.Counter()
        for col in bam.pileup(
            chrom, gpos, gpos + 1,
            truncate=True,
            min_mapping_quality=min_mapq,
            min_base_quality=min_baseq,
            stepper="samtools",
        ):
            if col.reference_pos != gpos:
                continue
            for pread in col.pileups:
                if pread.is_del or pread.is_refskip:
                    continue
                qbase_raw = pread.alignment.query_sequence[
                    pread.query_position
                ].upper()
                needs_complement = pread.alignment.is_reverse != (strand == "-")
                qbase = complement_base(qbase_raw) if needs_complement \
                        else qbase_raw
                counts[qbase] += 1

        cov      = sum(counts.values())
        ag_denom = counts["A"] + counts["G"]
        ag_frac  = counts["G"] / ag_denom \
                   if ref_base_tx == "A" and ag_denom > 0 else np.nan

        records.append({
            "tx_pos":       tx_pos,
            "gpos":         gpos,
            "ref_base":     ref_base_tx,
            "in_cds":       gpos in cds_positions,
            "A":            counts["A"],
            "G":            counts["G"],
            "C":            counts["C"],
            "T":            counts["T"],
            "coverage":     cov,
            "ag_edit_frac": ag_frac,
        })
        tx_pos += 1

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Find histidine codon positions in transcript coordinates
# ─────────────────────────────────────────────────────────────────────────────

def find_his_codon_tx_positions(ref_fasta: pysam.FastaFile,
                                 gene: dict) -> list:
    """
    Returns list of transcript positions (0-based) that are the middle base
    (A) of each His codon in the CDS, in transcript order.
    """
    his_positions = []
    tx_pos = 0
    chrom  = gene["chrom"]
    strand = gene["strand"]

    # Concatenate CDS sequence in transcript order
    tx_seq = ""
    for (cds_start, cds_end) in gene["cds"]:
        seg = ref_fasta.fetch(chrom, cds_start, cds_end).upper()
        if strand == "-":
            seg = reverse_complement(seg)
        tx_seq += seg

    # Walk codons
    for i in range(0, len(tx_seq) - 2, 3):
        codon = tx_seq[i:i+3]
        if codon in HIS_CODONS:
            his_positions.append(i + 1)   # middle base of codon

    return his_positions

def computeJointProbabilitiesPerRead(editing_df1, editing_df2):
    '''
    Computes the joint probability of observing an edit from the reference frequency computed in build_editing_matrix

    '''
    pass