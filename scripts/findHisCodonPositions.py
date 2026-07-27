'''
July 27, 2026 LT

For a lot of my scripts, I've been recomputing Histidine codon positions in transcript space for every gene, which is wasteful. This module provides a function to compute and cache these positions for all genes in a GTF, so that they can be reused across scripts.

input: GTF file, reference FASTA file
output: pickle file containing a dict of {gene_name: [his_codon_positions_in_transcript_space, position in genomic space]}
'''

import re
import sys
import math
import collections
from pathlib import Path

import pysam
import numpy as np
import pandas as pd
from logJosh import Tee
import pickle

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

def find_his_codon_positions(ref_fasta, gene):
    """
    Locate every His codon (CAT/CAC) in a gene's CDS, in both coordinate
    spaces at once. Position = the codon's middle base (both His codons
    have A there -- the A->G-editable position the rest of this codebase
    cares about), matching the find_his_codon_tx_positions convention
    duplicated across runHMMPerGene.py/trainHMMPerGene.py/etc.

    Returns (tx_positions, gpos_positions): two parallel lists --
    transcript-space (CDS-relative, 0 = first CDS base, transcript
    orientation) and genomic (0-indexed chrom coordinate) for the same
    codon positions, in CDS order.
    """
    chrom, strand = gene["chrom"], gene["strand"]
    tx_seq = ""
    tx_to_gpos = []
    for (cs, ce) in gene["cds"]:
        seg = ref_fasta.fetch(chrom, cs, ce).upper()
        rng = range(cs, ce)
        if strand == "-":
            seg = reverse_complement(seg)
            rng = range(ce - 1, cs - 1, -1)
        tx_seq += seg
        tx_to_gpos.extend(rng)

    tx_positions, gpos_positions, seen = [], [], set()
    for i in range(0, len(tx_seq) - 2, 3):
        if tx_seq[i:i+3] in HIS_CODONS:
            p = i + 1
            if p not in seen:
                seen.add(p)
                tx_positions.append(p)
                gpos_positions.append(tx_to_gpos[p])
    return tx_positions, gpos_positions

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

def main(args):
    gtf_path = args[0]
    fasta_path = args[1]
    out_path = args[2]

    genes = parse_gtf(gtf_path)
    ref_fasta = pysam.FastaFile(fasta_path)
    print(f"{len(genes):,} genes parsed from {gtf_path}.", file=sys.stderr)

    his_positions = {}
    for gname, gene in genes.items():
        tx_pos, gpos = find_his_codon_positions(ref_fasta, gene)
        his_positions[gname] = [tx_pos, gpos]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(his_positions, f)

    n_with_his = sum(1 for tx_pos, _ in his_positions.values() if tx_pos)
    print(f"{n_with_his:,}/{len(his_positions):,} genes have >=1 His codon.",
          file=sys.stderr)
    print(f"Wrote His codon positions to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    Tee()
    main(sys.argv[1:])