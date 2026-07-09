'''
July 9, 2026 LT

Same as metaHistidineFromParquet.py but trying to optimize it for large datasets. I'm going to try and run the analysis on individual parquets,
store that as a parquet, and then combine the parquets at the end. This should reduce memory usage and allow us to run on larger datasets.

inputs:
- parquet directory 1
- parquet directory 2
- ref fasta
- GTF
- label 1
- label 2
- window size
- min coverage at site
- output prefix
'''

import argparse
import sys
import re
import collections
import math
from pathlib import Path

import pysam
import pandas as pd
import numpy as np
from logJosh import Tee


HIS_CODONS = {"CAT", "CAC"}

def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]

def complement_base(b: str) -> str:
    return b.translate(str.maketrans("ACGTacgt", "TGCAtgca"))

def parse_gtf_cds(gtf_path: str) -> dict:
    '''
    Returns a dict of chrom
    '''
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