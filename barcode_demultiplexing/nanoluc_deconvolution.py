'''
August 7, 2024 LT

This script will create barcodes from the sequence of threonine codons found in different versions of nanoluc.
For a bam file, it will iterate through each read and extract the threonine barcode sequence and compare to the reference threonine barcode. 
If the barcode is a match, the read will be written to a new bam file with a custom tag.
'''

import os
import seaborn
import sys
import subprocess
import pandas as pd
import numpy as np

import pickle

from pathlib import Path
from icecream import ic
from datetime import datetime

from pprint import pprint

from tqdm.auto import tqdm

from Bio import SeqIO
import pysam

import seaborn as sea
import matplotlib.pyplot as plt
import argparse

from typing import Dict, List, Tuple

# get Thr coordinates
aa_dict = {'TTT':'F', 'TTC':'F', 'TTA':'L', 'TTG':'L',
              'TCT':'S', 'TCC':'S', 'TCA':'S', 'TCG':'S',
              'TAT':'Y', 'TAC':'Y', 'TAA':'*', 'TAG':'*',
              'TGT':'C', 'TGC':'C', 'TGA':'*', 'TGG':'W',
              'CTT':'L', 'CTC':'L', 'CTA':'L', 'CTG':'L',
              'CCT':'P', 'CCC':'P', 'CCA':'P', 'CCG':'P',
              'CAT':'H', 'CAC':'H', 'CAA':'Q', 'CAG':'Q',
              'CGT':'R', 'CGC':'R', 'CGA':'R', 'CGG':'R',
              'ATT':'I', 'ATC':'I', 'ATA':'I', 'ATG':'M',
              'ACT':'T', 'ACC':'T', 'ACA':'T', 'ACG':'T',
              'AAT':'N', 'AAC':'N', 'AAA':'K', 'AAG':'K',
              'AGT':'S', 'AGC':'S', 'AGA':'R', 'AGG':'R',
              'GTT':'V', 'GTC':'V', 'GTA':'V', 'GTG':'V',
              'GCT':'A', 'GCC':'A', 'GCA':'A', 'GCG':'A',
              'GAT':'D', 'GAC':'D', 'GAA':'E', 'GAG':'E',
              'GGT':'G', 'GGC':'G', 'GGA':'G', 'GGG':'G',
              'NNN': 'T'}

def findCodonInORF(ref_seq, aa):
    '''
    Find position of codon in ORF in reference sequence
    '''
    start = 0
    stop = 0
    codonPositions = []
    # Find start codon
    for i in range(0, len(ref_seq)-3, 3):
        if aa_dict[ref_seq[i:i+3]] == 'M':
            start = i
            break
    # print(start, ref_seq[start:start+3])
    # Find stop codon
    for i in range(start, len(ref_seq)-3, 3):
        if ref_seq[i:i+3] in aa_dict.keys():
            if aa_dict[ref_seq[i:i+3]] == '*':
                stop = i+3
                break
    # print(stop, ref_seq[stop-3:stop])  
    orf = ref_seq[start:stop]
    # Find the position of the codon of interest
    for i in range(start, stop, 3):
        if ref_seq[i:i+3] in aa_dict.keys():
            if aa_dict[ref_seq[i:i+3]] == aa:
                codonPositions.append(i+1)


    return start, stop, codonPositions, orf

def getThrBarcode(ref_seq, codonPositions):
    '''
    Get the barcode sequence for the codon positions in the reference sequence
    '''
    barcodes = []
    for i in codonPositions:
        barcodes.append(ref_seq[i-1:i+2])
    return ''.join(i for i in barcodes)

def extract_codons_from_alignment(codonPositions, query_seq, start):
    '''
    Extract codons from the alignment
    '''
    codons = []
    # adjust codon positions to query sequence
    codonPositions = [i-start for i in codonPositions]
    for i in codonPositions:
        codons.append(query_seq[i-1:i+2])
    return ''.join(i for i in codons)
    
def extract_seq_with_ref_positions_and_aligned_pairs(aligned_pairs, start, end, query_seq) -> str:
    """
    Extract the query positions from a list of aligned pairs that fall within a given range.
    This replaces the cs_tag based extraction!

    Fixed by Liam Tran 6/24/24

    Args:
        aligned_pairs (list): List of aligned pairs.
        start (int): Start of the range.
        end (int): End of the range.

    Returns:
        The query sequence that aligned between the start and end params
        with deletions denoted with '.'s and insertions denoted with '-'s.
    """
    result = []
    for query_pos, ref_pos in aligned_pairs:
        if ref_pos is None: # insertion in query
            if result:  # if result list is not empty, append None
                result.append(query_pos) # use this if you want to keep insertions
                
        elif start <= ref_pos <= end:
            result.append(query_pos)

        elif ref_pos > end:
            break  # exit loop if position is beyond the end of the range

    region_query_seq_list = [query_seq[i] if i else "." for i in result]

    region_query_sequence = "".join(region_query_seq_list)
    
    return region_query_sequence

def extract_codon_from_aligned_pairs(aligned_pairs, start, end, query_seq):
    '''
    Works like extract_seq_with_ref_positions_and_aligned_pairs but extracts codons. Only retains bases that align to the reference.
    '''

    result = []
    for q, r in aligned_pairs:
        if r is not None:
            if start <= r <= end:
                result.append(q)
            elif r > end:
                break

    codon_list = [query_seq[i] if i else "." for i in result]
    codon = "".join(codon_list)

    middle_base = codon[1]

    return codon, middle_base

def get_barcodes(barcodes_fasta='',
                 rev_compliment=False) -> dict:
    barcodes_fasta = Path(barcodes_fasta)
    # First lets load the barcodes fasta:
    print("Starting barcode load...", end="")
    if not rev_compliment:
        barcodes = {rec.id: str(rec.seq) for rec in SeqIO.parse(barcodes_fasta, "fasta")}
    else:
        barcodes = {rec.id: str(rec.seq.reverse_complement()) for rec in SeqIO.parse(barcodes_fasta, "fasta")}
    print("Done!")
    return barcodes

def pick_best_barcode(input_seq: str,
                      barcodes_dict: dict,
                      better_than_second_best_by: int = 0):
    import stringdist
    dist_dict = {}
    seq_upper = input_seq.upper()
    for name, bar in barcodes_dict.items():
        dist_dict[name] = stringdist.levenshtein(seq_upper, bar.upper())
    closest = min(dist_dict, key=dist_dict.get)
    closest_dist = dist_dict[closest]
    second_best = sorted(dist_dict, key=dist_dict.get)[1]
    second_best_dist = dist_dict[second_best]

    if second_best_dist - closest_dist > better_than_second_best_by:
        return closest, closest_dist
    else:
        return "NotDistinct", closest_dist
    
def check_nanoluc_version(query_seq, codonPositions, aligned_pairs, barcode_dict):
    '''
    Check the version of nanoluc by comparing aligned bases between threonine codons in the reference and query sequence. 
    '''
    import stringdist

    nanoluc_version = 'NotDistinct'
    ref_string = ''
    
    extracted_seq = ''
    mid_bar = ''
    r_mid_bar = ''

    # check for mismatches
    for q, r in aligned_pairs:
        if r in codonPositions:
            
            extracted_codon, e_mid = extract_codon_from_aligned_pairs(aligned_pairs, r-1, r+1, query_seq)
            # ic(extracted_codon, result)

            extracted_seq += extracted_codon
            # ref_string += ref_codon

            # r_mid_bar += r_mid
            mid_bar += e_mid

    
    # check if the extracted sequence is a barcode
    nanoluc_version, dist = pick_best_barcode(mid_bar, barcode_dict)
    

    return nanoluc_version, dist, extracted_seq, mid_bar
                    
def main():
    parser = argparse.ArgumentParser(description='Extract nanoluc barcode from bam file')
    parser.add_argument('--bam', type=str, help='Input bam file')
    parser.add_argument('--nanoluc_versions', type=str, help='Reference fasta, NOT the one used for alignment!!! This is the nanoluc reference with threonine codons not masked')
    parser.add_argument('--barcode', type=str, help='Barcode fasta file')
    parser.add_argument('--ref', type=str, help='Reference fasta file used for alignment')
    parser.add_argument('--output', type=str, help='Output bam file')

    args = parser.parse_args()

    # Load the reference sequences
    ref_degenerate = ''
    ref_999 = ''

    for record in SeqIO.parse(args.ref, "fasta"):
        if record.id == "nanoluc_999":
            ref_999 = record.seq
        else:
            ref_degenerate = record.seq

    # Load the nanoluc versions
    nanoluc_versions = {}
    
    for record in SeqIO.parse(args.nanoluc_versions, "fasta"):
        nanoluc_versions[record.id] = str(record.seq)

    # ic(nanoluc_versions)

    # Find the codon positions in the reference sequence
    deg_start, deg_stop, deg_codonPositions, deg_orf = findCodonInORF(ref_degenerate.upper(), 'T')
    Arich_start, Arich_stop, Arich_codonPositions, Arich_orf = findCodonInORF(nanoluc_versions['nanoluc_996'].upper(), aa='T')
    noThr_start, noThr_stop, noThr_codonPositions, noThr_orf = findCodonInORF(nanoluc_versions['nanoluc_997'].upper(), aa='T')
    twoThr_start, twoThr_stop, twoThr_codonPositions, twoThr_orf = findCodonInORF(nanoluc_versions['nanoluc_998'].upper(), aa='T')
    # hairpin_start, hairpin_stop, hairpin_codonPositions, hairpin_orf = findCodonInORF(nanoluc_versions['999'].upper(), aa='T')


    # Get the barcode sequence for the codon positions in the reference sequence
    barcode_dict = get_barcodes(args.barcode)
    ic(barcode_dict)

    # get the barcode for each version
    # deg_barcode = getThrBarcode(ref_degenerate, deg_codonPositions)
    Arich_barcode = getThrBarcode(nanoluc_versions['nanoluc_996'].upper(), Arich_codonPositions)
    noThr_barcode = getThrBarcode(nanoluc_versions['nanoluc_997'].upper(), noThr_codonPositions)
    twoThr_barcode = getThrBarcode(nanoluc_versions['nanoluc_998'].upper(), twoThr_codonPositions)
    # hairpin_barcode = getThrBarcode(nanoluc_versions['999'].upper(), hairpin_codonPositions)

    # Load the bam file
    bam = pysam.AlignmentFile(args.bam, "rb")

    # hairpin_coords
    hairpin_start = 40 # hairpin_start = 48, added some padding
    hairpin_stop = 100 # hairpin_stop = 96, added some padding

    # Create a new bam file
    with pysam.AlignmentFile(args.output, "wb", header=bam.header) as output_bam:
        all_nanoluc_ct = 0
        hairpin_ct = 0
        # Iterate through each read aligning to degenerate nanoluc
        for read in tqdm(bam.fetch('nanoluc_degenerate', start=deg_start, stop=deg_stop), desc = 'Deconvoluting nanoluc versions'):
            if not read.is_unmapped:
                # Get the aligned pairs
                aligned_pairs = read.get_aligned_pairs()
                query_seq = read.query_sequence
                # Check the version of nanoluc
                nanoluc_version, dist, extracted_seq, mid_bar = check_nanoluc_version(query_seq, deg_codonPositions, aligned_pairs, barcode_dict)
                # Add the version to the read
                read.set_tag('bN', nanoluc_version)
                all_nanoluc_ct += 1
                output_bam.write(read)

        # # Iterate through each read aligning to hairpin nanoluc
        # for read in tqdm(bam.fetch('nanoluc_999', start=hairpin_start, stop=hairpin_stop), desc='Extracting hairpin nanoluc'):
        #     if not read.is_unmapped:
        #         # Get the aligned pairs
        #         aligned_pairs = read.get_aligned_pairs()
        #         read.set_tag('bN', '999')
        #         output_bam.write(read)
        #         hairpin_ct += 1
        
        # updating to check for presence of deletion in reads aligning to hairpin
        ab = dict()
        for read in tqdm(bam.fetch('nanoluc_999', start=hairpin_start, stop=hairpin_stop), desc='Extracting hairpin nanoluc'):
            if not read.is_unmapped:
                # Get the aligned pairs
                aligned_pairs = read.get_aligned_pairs()
                ab[read.query_name] = aligned_pairs

        # deletion range
        del_start = 48
        del_stop = 96
        deletion = []
        deletion_reads = []
        for x in range(del_start, del_stop+1):
            deletion.append(x)

        for read in ab:
            aligned_pairs = ab[read]
            
            for q, r in aligned_pairs:
                if r in deletion and q is None:
                    deletion_reads.append(read)
                    
        for read in tqdm(bam.fetch('nanoluc_999', start=hairpin_start, stop=hairpin_stop), desc='Extracting hairpin nanoluc'):
            if not read.is_unmapped:
                if read.query_name not in deletion_reads:
                    read.set_tag('bN', '999')
                    output_bam.write(read)
                    hairpin_ct += 1
                    
                
                

    output_bam.close()
    bam.close()
    # sort and index the output bam file
    subprocess.run(['samtools', 'sort', args.output, '-o', args.output])
    subprocess.run(['samtools', 'index', args.output])

    print(f'All nanoluc reads: {all_nanoluc_ct}')
    print(f'Hairpin nanoluc reads: {hairpin_ct}')

if __name__ == '__main__':
    main()




