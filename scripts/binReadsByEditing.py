'''
November 3, 2025 LT
This script will bin TadA edited reads by shadows at user-defined regions.

First, I will iterate through each read in the BAM file. For each read, I will check if it overlaps with any of the user-defined regions. 
If it does, I will convert the read to a binary string of 1's and 0's where 1's are TadA edits and 0's are not edited. 

If there are no 1s in the user-defined region, the read will be binned as 'shadowed'.

inputs:
- BAM file of aligned reads
- TSV file of user-defined regions

outputs:
-barchart of proportion of shadowed reads per barcode
-bam file of shadowed reads only
'''
import pysam
import pandas as pd
import matplotlib.pyplot as plt 
import sys
from collections import defaultdict

def load_regions(tsv_file):
    regions = []
    df = pd.read_csv(tsv_file, sep='\t', header=None)
    for index, row in df.iterrows():
        regions.append((row[0], row[1], row[2]))  # (chromosome, start, end)
    return regions

def is_shadowed(read, regions, reference_sequence):
    for chrom, start, end in regions:
        if read.reference_name == chrom and not (read.reference_end < start or read.reference_start > end):
            # Check for TadA edits in the region
            aligned_seq = read.get_aligned_pairs(matches_only=True)
            for query_pos, ref_pos in aligned_seq:
                if ref_pos is not None and start <= ref_pos < end:
                    read_base = read.query_sequence[query_pos]
                    ref_base = reference_sequence[ref_pos]
                    if (ref_base == 'A' and read_base == 'G'):
                        return False
                    else: 
                        continue
            return True
    return False
                    
def bin_reads_by_editing(bam_file, tsv_file, output_bam, output_chart):
    regions = load_regions(tsv_file)
    reference_sequence = None
    shadows = 0
    total_reads = 0
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        for read in bam:
            if reference_sequence is None:
                reference_sequence = bam.get_reference_sequence(read.reference_name)

            if is_shadowed(read, regions, reference_sequence):
                # Bin the read as shadowed
                shadows += 1
            total_reads += 1
    if total_reads > 0:
        print(f"Proportion of shadowed reads: {shadows / total_reads:.2%}")
    else:
        print("No reads found.")