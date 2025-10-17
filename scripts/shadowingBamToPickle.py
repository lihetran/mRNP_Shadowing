'''
July 2, 2025
LT

I'm finding that creating an intermediate file for downstream shadowing analyses has been useful. Pickles have been great for storing dictionaries.

    This script creates a pickle file from a bam that contains a dictionary with read names as keys and values are binary edits strings, barcodes, barcode sequences, read sequences,
    aligned read sequences, aligned reference sequences, aligned pairs, and absolute indices for every base in the read.

input: bam file, this file will usually be a merged bam file that is the result of mapBased_barcodeSplitting_Hamming.py
    reference fasta file
    output_dir - directory to save the pickle file
output: pickle file with the same name as the bam file, but with a .pickle extension
'''
import pysam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from icecream import ic
from Bio import SeqIO
import sys
import pickle


def get_binary_edit_string(bam, ref_sequence, chrom):
    '''
    For every read in a bam, this function will extract a binary string of A-G mismatches between the reference sequence and the read sequence.

    input: bam - path to the bam file
              ref_sequence - reference sequence
              aligned_pairs - list of aligned pairs
              start - start position of the region to extract
              end - end position of the region to extract
    output: dictionary with keys as the read names and values as the binary strings
    '''
    edit_dict = {}
    for read in pysam.AlignmentFile(bam, "rb").fetch(chrom):
        if not read.is_unmapped:
            read_seq = read.query_sequence.upper()
            ref_seq = ref_sequence.upper()
            aligned_pairs = read.get_aligned_pairs()
            edits = []
            read_string = []
            ref_string = []

            for read_pos, ref_pos in aligned_pairs:
                if ref_pos is not None and read_pos is not None:
                    if ref_seq[ref_pos] == 'A' and read_seq[read_pos] == 'G':
                        edits.append(1)
                        read_string.append(read_seq[read_pos])
                        ref_string.append(ref_seq[ref_pos])
                    else:
                        edits.append(0)
                        read_string.append(read_seq[read_pos])
                        ref_string.append(ref_seq[ref_pos])
            # convert the list of edits to a binary string
            edit_string = ''.join(str(i) for i in edits)
            # read_string = ''.join(str(i) for i in read_string)
            # ref_string = ''.join(str(i) for i in ref_string)
            # print(read_string)
            # print(edit_string)
            # print(ref_string)
            # yield read.query_name, edit_string
            edit_dict[read.query_name] = edit_string

    return edit_dict

def get_absolute_positions(read):
    '''need to calculate absolute position of the read in the reference sequence, will do this by getting the start of 
    the alignment (4th field in the sam file) and the CIGAR string (5th field in the sam file) to get the absolute positions.'''
    # get the start position of the read
    start = read.query_alignment_start
    # get the CIGAR string
    cigar_string = read.cigarstring
    # get the aligned positions
    aligned_positions = []
    ref_pos = start
    for i in range(len(cigar_string)):
        if cigar_string[i].isdigit():
            continue
        else:
            length = int(cigar_string[:i])
            if cigar_string[i] == 'M':  # match or mismatch
                aligned_positions.extend(range(ref_pos, ref_pos + length))
                ref_pos += length
            elif cigar_string[i] == 'I':  # insertion
                aligned_positions.extend([None] * length)  # None for insertion
            elif cigar_string[i] == 'D':  # deletion
                ref_pos += length  # skip these positions in the reference
            cigar_string = cigar_string[i + 1:]
            break
    # continue processing the remaining CIGAR string
    while cigar_string:
        for i in range(len(cigar_string)):
            if cigar_string[i].isdigit():
                continue
            else:
                length = int(cigar_string[:i])
                if cigar_string[i] == 'M':  # match or mismatch
                    aligned_positions.extend(range(ref_pos, ref_pos + length))
                    ref_pos += length
                elif cigar_string[i] == 'I':  # insertion
                    aligned_positions.extend([None] * length)  # None for insertion
                elif cigar_string[i] == 'D':  # deletion
                    ref_pos += length  # skip these positions in the reference
                cigar_string = cigar_string[i + 1:]
                break
    # print(f"Read {read.query_name} aligned positions: {aligned_positions}")
    

    return aligned_positions

def get_read_dict(bam, ref_sequence, chrom):
    '''
    For every read in a bam, this function will extract a binary string of A-G mismatches between the reference sequence and the read sequence.

    input: bam - path to the bam file
              ref_sequence - reference sequence
              aligned_pairs - list of aligned pairs
              start - start position of the region to extract
              end - end position of the region to extract
    output: dictionary with keys as the read names and values as the binary strings

    # need to calculate absolute position of the read in the reference sequence, will do this by getting the start of the alignment (4th field in the sam file) and the CIGAR string (5th field in the sam file) to get the aligned positions.
    '''
    edit_dict = {}
    for read in pysam.AlignmentFile(bam, "rb").fetch(chrom):
        read_dict = {}
        if not read.is_unmapped:
            barcode = read.get_tag('cI')
            bar_seq = read.get_tag('cS')
            read_seq = read.query_sequence.upper()
            ref_seq = ref_sequence.upper()
            aligned_pairs = read.get_aligned_pairs()
            edits = []
            read_string = []
            ref_string = []
            absolute_indices = get_absolute_positions(read)
            for read_pos, ref_pos in aligned_pairs:
                if ref_pos is not None and read_pos is not None:
                    if ref_seq[ref_pos] == 'A' and read_seq[read_pos] == 'G':
                        edits.append(1)
                        read_string.append(read_seq[read_pos])
                        ref_string.append(ref_seq[ref_pos])
                    else:
                        edits.append(0)
                        read_string.append(read_seq[read_pos])
                        ref_string.append(ref_seq[ref_pos])
                elif ref_pos is None: # if insertion in read
                    edits.append(2)
                    read_string.append(read_seq[read_pos])
                    ref_string.append(' ')
                elif read_pos is None: # if deletion in read
                    edits.append(2)
                    read_string.append(' ')
                    ref_string.append(ref_seq[ref_pos])
            # convert the list of edits to a binary string
            edit_string = ''.join(str(i) for i in edits)
            read_string = ''.join(str(i) for i in read_string)
            ref_string = ''.join(str(i) for i in ref_string)
            # print(ref_sequence)
            # print(read_string)
            # print("".join(str(i) for i in absolute_indices))

            # print(read_string)
            # print(edit_string)
            # print(ref_string)
            # yield read.query_name, edit_string
            read_dict = {
                'edit_string': edit_string,
                'barcode': barcode,
                'bar_seq': bar_seq,
                'read_sequence': read.query_sequence,
                'read_sequence_aligned': read_string,
                'ref_sequence_aligned': ref_string,
                'aligned_pairs': aligned_pairs,
                'absolute_indices': absolute_indices
            }
            edit_dict[read.query_name] = read_dict

    return edit_dict

def main(args):
    bam_file = Path(args[0])
    ref_fasta = Path(args[1])
    output_dir = Path(args[2])
    
    # Load the reference sequence
    ref_dict = SeqIO.to_dict(SeqIO.parse(ref_fasta, 'fasta'))
    
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # make a pickle per chromosome
    for chrom in ref_dict.keys():
        print(f"Processing chromosome {chrom}...")
        edit_dict = get_read_dict(bam_file, ref_dict[chrom].seq, chrom)

        # make a pickle file for each chromosome
        pickle_file = output_dir / f"{Path(bam_file).stem}_{chrom}.pickle"
        print(f"Saving edits to {pickle_file}...")

        with open(pickle_file, 'wb') as f:
            pickle.dump(edit_dict, f)
        print(f"Saved {len(edit_dict)} reads to {pickle_file}")
    
    print("All done!")

if __name__=='__main__':
    if len(sys.argv) != 4:
        print("Usage: python shadowingBamToPickle.py <bam_file> <reference_fasta> <output_dir>")
        sys.exit(1)
    main(sys.argv[1:])
