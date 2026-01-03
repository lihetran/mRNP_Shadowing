'''
This script will compute editing efficiencies per gene type from shadowing sham files
'''

from ShamReader import shamReader
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
import sys
import pysam
import os
from Bio import SeqIO, Seq, SeqRecord
import pandas as pd


def ParseCigar(read_seq, ref_seq, cigar, ref_start, ref_end, q_st, q_en, feature_string):
    import re
    parsed_cigar = re.findall(rf'(\d+)([MDNSIX])', cigar)
    parsed_cigar = [(int(num), char) for num, char in parsed_cigar]
    ref_seq = ref_seq[int(ref_start):int(ref_end)].upper()
    ref_pos = 0
    read_seq = read_seq[int(q_st):int(q_en)].upper()
    read_pos = 0

    top_line = ""
    middle_line = ""
    bottom_line = ""
    feature_line = ""

    for length, code in parsed_cigar:
        if code == "M":  # Map (Read & Ref Match)
            read_map_piece = read_seq[read_pos:read_pos + length]
            ref_map_piece = ref_seq[ref_pos:ref_pos + length]
            feature_map_piece = feature_string[read_pos:read_pos + length]
            perfect_matches = ""
            for index, char in enumerate(read_map_piece):
                try:
                    if char == ref_map_piece[index]:
                        perfect_matches += "|"
                    else:
                        perfect_matches += "•"
                except IndexError:
                    perfect_matches += " "
            top_line += read_map_piece
            middle_line += perfect_matches
            bottom_line += ref_map_piece
            feature_line += feature_map_piece
            ref_pos += length
            read_pos += length

        elif code == "I":  # Insert (Gap in Ref)
            top_line += read_seq[read_pos:read_pos + length]
            feature_line += feature_string[read_pos:read_pos + length]
            middle_line += " " * length
            bottom_line += " " * length
            read_pos += length

        elif code == "D" or code == "N":  # Delete (Gap in Read)
            top_line += " " * length
            middle_line += " " * length
            feature_line += " " * length
            bottom_line += ref_seq[ref_pos:ref_pos + length]
            ref_pos += length

    return top_line, middle_line, bottom_line, feature_line

def reverseComplement(seq):

    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    reverse_complement = "".join(complement.get(base, base) for base in reversed(seq.upper()))

    return reverse_complement

def getEfficiency(top_line, bottom_line, strand):
    inosines = 0
    numSites = bottom_line.count('A')
     
    if strand == '+': #reads are reverse complemented and mutated from A-G
        for i in range(len(top_line)):
            if top_line[i] == 'G' and bottom_line[i] == 'A':  
                inosines += 1
                
    elif strand == '-': #reads are not reverse complemented and mutated from C-T
        for i in range(len(top_line)):
            if top_line[i] == 'C' and bottom_line[i] == 'T':
                inosines += 1

    if numSites == 0: #don't include if no A's in alignment
        return None
    else:
        return inosines/numSites * 100
    
def getEfficiency2(editString, refString, strand):
    

    if strand == '+':
        numSites = refString.count('A')
        inosines = editString.count('1')

    elif strand == '-':
        numSites = refString.count('T')
        inosines = editString.count('1')

    if numSites == 0:
        return None
    else:
        return inosines/numSites * 100

def main(args):
    sham = args[0]
    genome = args[1]
    outputPrefix = args[2]

    ##### get reference genome #####
    print("Reading reference genome...")
    refDict = {}
    for record in SeqIO.parse(genome, 'fasta'):
        name, sequence = record.id, str(record.seq)
        refDict[name] = sequence

    geneTypes = {'protein_coding': [],
                 'rRNA': []}
    
    shamFile = shamReader(sham)
    print("Parsing sham file...")
    for samInfo, annotString, geneType, editString, coords in shamFile.readSham():
        if geneType not in geneTypes:
            continue
        flag = samInfo[1]
        # strand info
        if flag != str(16):
            strand='+'
        else:
            strand='-'
        # parse alignment
        if strand == '+':
            query = samInfo[9]
            ref = refDict[samInfo[2]]
            top, middle, bottom, feature = ParseCigar(samInfo[9], ref, samInfo[5], coords[0], coords[1], coords[2], coords[3], annotString)
            efficiency = getEfficiency(top, bottom, strand) 
            if efficiency is not None:
                if geneType == 'protein_coding':
                    geneTypes['protein_coding'].append(efficiency) 
                elif geneType == 'rRNA':
                    geneTypes['rRNA'].append(efficiency)

    ##### plot results #####
    print("Plotting results...")
    ##### compute mean efficiencies #####
    meanEfficiencies = {}
    for geneType, efficiencies in geneTypes.items():
        meanEfficiencies[geneType] = np.mean(efficiencies)

    ##### create boxplot #####
    data = [geneTypes['protein_coding'], geneTypes['rRNA']]
    labels = ['protein_coding', 'rRNA']
    plt.figure(figsize=(8,6))
    sns.boxplot(data=data)
    plt.xticks(ticks=[0,1], labels=labels, fontsize=14)
    plt.ylabel('Editing Efficiency (%)', fontsize=14)
    plt.title('Editing Efficiencies by Gene Type', fontsize=16)
    ##### add mean efficiency lines #####
    # for i, geneType in enumerate(labels):
    #     plt.axhline(y=meanEfficiencies[geneType], color='r', linestyle='--', label=f'Mean {geneType}: {meanEfficiencies[geneType]:.2f}%')
    # plt.legend()
    plt.savefig(f"{outputPrefix}_editing_efficiencies_per_gene_type.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main(sys.argv[1:])