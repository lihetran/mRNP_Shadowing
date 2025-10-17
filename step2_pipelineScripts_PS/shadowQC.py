'''
Script to output QC metrics on nanopore polysome shadowing data. 
Meant to determine whether or not the data is good enough to be used for further analysis "at a glance".

usage: python shadowQC.py <mock sam file> <mock fastq file> <treated sam file> <treated fastq file> <genome fasta file> <output file>

output: length distribution of reads, editing efficiency for tad/mock libraries, number of reads, % reads that map to rRNA, mRNA, standards, etc.
'''

import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
import sys
import pysam
import os

from FastqReader import FastQreader
from FastaReader import FastAreader

from substitution_errors import init_dict_substitutions, getSubstitutionErrors

'''
This method is adapted from Marcus Viscardi's cigar string parsing code.
'''
def ParseCigar(read_seq, ref_seq, strand: bool, cigar, ref_start, ref_end, q_st, q_en):
    import re
    parsed_cigar = re.findall(rf'(\d+)([MDNSIX])', cigar)
    parsed_cigar = [(int(num), char) for num, char in parsed_cigar]
    ref_seq = ref_seq[ref_start: ref_end].upper()
    ref_pos = 0
    read_seq = read_seq[q_st: q_en].upper()
    read_pos = 0

    top_line = ""
    middle_line = ""
    bottom_line = ""

    for length, code in parsed_cigar:
        if code == "M":  # Map (Read & Ref Match)
            read_map_piece = read_seq[read_pos:read_pos + length]
            ref_map_piece = ref_seq[ref_pos:ref_pos + length]
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
            ref_pos += length
            read_pos += length

        elif code == "I":  # Insert (Gap in Ref)
            top_line += read_seq[read_pos:read_pos + length]
            middle_line += " " * length
            bottom_line += " " * length
            read_pos += length

        elif code == "D" or code == "N":  # Delete (Gap in Read)
            top_line += " " * length
            middle_line += " " * length
            bottom_line += ref_seq[ref_pos:ref_pos + length]
            ref_pos += length
    
    inosines = 0
    numSites = bottom_line.count('A')
     
    if strand == True: #reads are reverse complemented and mutated from A-G
        for i in range(len(top_line)):
            if top_line[i] == 'G' and bottom_line[i] == 'A':  
                inosines += 1
                
    elif strand == False: #reads are not reverse complemented and mutated from C-T
        for i in range(len(top_line)):
            if top_line[i] == 'C' and bottom_line[i] == 'T':
                inosines += 1

    if numSites == 0: #don't include if no A's in alignment
        return None
    else:
        return inosines/numSites * 100



def reverseComplement(seq):

    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    reverse_complement = "".join(complement.get(base, base) for base in reversed(seq.upper()))

    return reverse_complement
    

def main(args):
    
    #### reverse complement mock cDNA reads ########
    mockFastq = FastQreader(args[1])
    mockReadDict = {}
    
    print(args[1])
    for record in mockFastq.readFastq():
        header = record[0]
        seq = record[1]
        strand = record[2]
        qual = record[3]

        rc = reverseComplement(seq.upper())
        #print(header.split()[0][1:])
        mockReadDict[header.split()[0][1:]] = rc 

    #### reverse complement tad cDNA reads ########
    tadFastq = FastQreader(args[3])
    tadReadDict = {}
    # print(args[3])
    for record in tadFastq.readFastq():
        header = record[0]
        seq = record[1]
        strand = record[2]
        qual = record[3]

        rc = reverseComplement(seq.upper())
        tadReadDict[header.split()[0][1:]] = rc #rc?

    #### get reference sequences ########
    fasta = FastAreader(args[4])
    refDict = {}
    # print(args[4])
    for header, seq in fasta.readFasta():
        h = header.split()[0][1:]
        refDict[h] = seq
               
    # print(args[0])
    mockBam = pysam.AlignmentFile(args[0], "rb")
    mockLengths = []
    efficienciesPerRead = []
    mockSubs = init_dict_substitutions()
    totalMockSubs = 0
    
    # Don't include these chromosomes in the analysis because reads aligned here are unlikely to be edited
    # need to implement a way to get the rRNA, protein-coding counts from the bam file, featureCounts?
    # badChroms = ['cerENO2', 'I', 'V']
    badChroms = ['I', 'V']
    mock_rRNA_counts = 0  
    std_counts = 0
    for read in mockBam:
        # if read.reference_name == 'I' or read.reference_name == 'V':
        #     mock_rRNA_counts += 1
        # # elif read.reference_name == 'cerENO2':
        # #     std_counts += 1
        if read.reference_name not in badChroms:
            
            originalSeq = mockReadDict[read.query_name]
            originalRef = refDict[read.reference_name]

            mockLengths.append(read.query_length)
            e = ParseCigar(originalSeq, originalRef, True, read.cigarstring, 
                           read.reference_start, read.reference_end, read.query_alignment_start, read.query_alignment_end)
            #print(e)
            efficienciesPerRead.append(e)
            if e is not None:
                mockReadDict[read.query_name] = (e, read.query_length)
            readSubs = init_dict_substitutions()
            readSubs = getSubstitutionErrors(originalSeq, originalRef, read.cigarstring, read.reference_start, read.reference_end, read.query_alignment_start, read.query_alignment_end)
            for sub in readSubs:
                mockSubs[sub] += readSubs[sub]

    mockEff = [x for x in efficienciesPerRead if x != None]
    for sub in mockSubs:
        totalMockSubs += mockSubs[sub]


    # print(args[2])
    tadBam = pysam.AlignmentFile(args[2], "rb")
    tadLengths = []
    efficienciesPerRead = []
    tadSubs = init_dict_substitutions()
    totalTadSubs = 0
    # Don't include these chromosomes in the analysis because reads aligned here are unlikely to be edited
    # badChroms = ['cerENO2', 'I', 'V']  
    badChroms = ['I', 'V']
    for read in tadBam:
        
        if read.reference_name not in badChroms:
            
            originalSeq = tadReadDict[read.query_name]
            originalRef = refDict[read.reference_name]

            tadLengths.append(read.query_length)
            e = ParseCigar(originalSeq, originalRef, True, read.cigarstring, 
                           read.reference_start, read.reference_end, read.query_alignment_start, read.query_alignment_end)
            #print(e)
            efficienciesPerRead.append(e)
            if e is not None:
                tadReadDict[read.query_name] = (e, read.query_length)
            readSubs = init_dict_substitutions()
            readSubs = getSubstitutionErrors(originalSeq, originalRef, read.cigarstring, read.reference_start, read.reference_end, read.query_alignment_start, read.query_alignment_end)
            for sub in readSubs:
                tadSubs[sub] += readSubs[sub]

    tadEff = [x for x in efficienciesPerRead if x != None]
    for sub in tadSubs:
        totalTadSubs += tadSubs[sub]
    
    

    figureHeight = 11
    figureWidth = 4.5
    plt.figure(figsize=(figureWidth, figureHeight))

    firstPanelHeight = 2 / figureHeight
    firstPanelWidth = 2 / figureWidth

    secondPanelHeight = 2 / figureHeight
    secondPanelWidth = 2 / figureWidth

    thirdPanelHeight = 2 / figureHeight
    thirdPanelWidth = 3.5 / figureWidth

    topPanelHeight = 1 / figureHeight
    topPanelWidth = 4 / figureWidth

    topPanel = plt.axes([0.05, 0.9, topPanelWidth, topPanelHeight])
    topPanel.tick_params(bottom=False, labelbottom=False,
                   left=False, labelleft=False,
                   right=False, labelright=False,
                   top=False, labeltop=False)
    topPanel.text(0.05, 0.7, "Mean Tad8A.20 A-G mismatches/# A's: " + '%.2f' % np.mean(tadEff) + '%', fontsize=10)
    topPanel.text(0.05, 0.5, "Median Tad8A.20 A-G mismatches/# A's: " + '%.2f' % np.median(tadEff) + '%', fontsize=10)
    topPanel.text(0.05, 0.3, "Mean Mock A-G mismatches/# A's: " + '%.2f' % np.mean(mockEff) + '%', fontsize=10)
    topPanel.text(0.05, 0.1, "Median Mock A-G mismatches/# A's: " + '%.2f' % np.median(mockEff) + '%', fontsize=10)

    panel1 = plt.axes([0.25, 0.65, firstPanelWidth, firstPanelHeight])
    panel1 = sns.histplot(mockEff, color='blue', alpha = 0.5,label='Mock')
    panel1 = sns.histplot(tadEff, color='red', alpha=0.5, label='Treated')
    panel1.legend()
    panel1.set_xlabel("Editing Efficiency (%)")
    panel1.set_ylabel("Count")
    panel1.set_title("Editing Efficiency per Read")

    panel2 = plt.axes([0.25, 0.35, secondPanelWidth, secondPanelHeight])
    m = [math.log(x,2) for x in mockLengths if x != 0]
    m_counts, bins = np.histogram(m)
    m_pdf = m_counts / sum(m_counts)
    m_cdf = np.cumsum(m_pdf)
    panel2.plot(bins[:-1], m_cdf, color='blue', alpha=0.5, label='Mock')
    # panel2 = sns.histplot(m, color='blue', alpha=0.5, label='Mock')

    t = [math.log(x,2) for x in tadLengths if x != 0]
    t_counts, bins = np.histogram(t)
    t_pdf = t_counts / sum(t_counts)
    t_cdf = np.cumsum(t_pdf)
    panel2.plot(bins[:-1], t_cdf, color='red', alpha=0.5, label='Treated')
    # panel2 = sns.histplot(t, color='red', alpha=0.5, label='Treated')

    panel2.legend()
    panel2.set_xlabel("Read Length (log2-scale)")
    panel2.set_ylabel("CDF")
    panel2.set_title("Read Lengths")

    panel3 = plt.axes([0.15, 0.05, thirdPanelWidth, thirdPanelHeight])
    for r in mockSubs:
        mock, = panel3.plot(r, mockSubs[r]/totalMockSubs, marker='o', color='blue', label = 'Mock')
    for r in tadSubs:
        tad, = panel3.plot(r, tadSubs[r]/totalTadSubs, marker='o', color='red', label = 'Treated')
    # for r in mockSubs:
    #     mock, = panel3.plot(r, mockSubs[r], marker='o', color='blue', label = 'Mock')
    # for r in tadSubs:
    #     tad, = panel3.plot(r, tadSubs[r], marker='o', color='red', label = 'Treated')
    panel3.set_xlabel("Substitution Type (Genomic Base - Read Base)")
    panel3.set_ylabel("# of Substitutions")
    panel3.set_title("Substitution Errors")
    panel3.legend(handles=[mock, tad])

    # print(args[-1])
    plt.savefig(args[-1])
    

if __name__ == "__main__":
    main(sys.argv[1:])

    




