'''
Script to output QC metrics on nanopore polysome shadowing data. 
Meant to determine whether or not the data is good enough to be used for further analysis "at a glance".

usage: python shadowQC.py <mock sham file> <treated sham file> <genome fasta> <output file>

output: length distribution of reads, editing efficiency for tad/mock libraries, number of reads, % reads that map to rRNA, mRNA, standards, etc.
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

from substitution_errors import init_dict_substitutions, getSubstitutionErrors

def editsPerGeneRegion(modList, feature_line, ref_line):
    """Returns a list of the number of edits in each gene region for a given feature string"""
    mods = [int(x) for x in modList]

    editsPerCDS = 0
    num_CDS_As = 0

    editsPerUTR = 0
    num_UTR_As = 0

    #### get numA's per gene region ####
    for i in range(len(feature_line)):
        if feature_line[i] == 'C' and ref_line[i] == 'A':
            num_CDS_As += 1
        elif feature_line[i] == 'U' and ref_line[i] == 'A':
            num_UTR_As += 1

    #### get num edits per gene region ####
    for i in range(len(feature_line)):
        if feature_line[i] == 'C' and mods[i] == 1:
            editsPerCDS += 1
        elif feature_line[i] == 'U' and mods[i] == 1:
            editsPerUTR += 1
    
    return ([editsPerCDS, num_CDS_As], [editsPerUTR, num_UTR_As])
    

def plot_edits(editDict):
    plot_dict = {'CDS': 0, 'UTR': 0}
    
    # get total number of edits per region
    for read in editDict:
        plot_dict['CDS'] += editDict[read][0][0]
        plot_dict['UTR'] += editDict[read][1][0]
    # get total number of As per region
    regionDict = {'CDS': 0, 'UTR': 0}
    for read in editDict:
        regionDict['CDS'] += editDict[read][0][1]
        regionDict['UTR'] += editDict[read][1][1]
        
    # normalize number of edits by number of A's in each region
    for k in plot_dict:
        plot_dict[k] = plot_dict[k]/regionDict[k]
    
    return plot_dict
    
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

    mockSham = shamReader(args[0])
    treatedSham = shamReader(args[1])
    genome = args[2]

    ##### get reference genome #####
    print("Reading reference genome...")
    refDict = {}
    for record in SeqIO.parse(genome, 'fasta'):
        name, sequence = record.id, str(record.seq)
        refDict[name] = sequence

    ##### get mock stats #####
    mockReadLengths = []
    mockEffs = []
    mockGeneTypes = {'protein-coding': 0, 'rRNA': 0, 'ncRNA': 0, 'other': 0}
    mockReadDict = {}
    mockEditDict = {}
    mockSubs = init_dict_substitutions()
    totalMockSubs = 0
    print("Reading mock sham file...")
    for mockInfo, mockAnnot, mockGene, mockEdit, coords in mockSham.readSham():
        print(mockInfo[0])
        print(mockAnnot)
        print(mockGene)
        print(mockEdit)
        print(coords)
        # read lengths
        mockReadLengths.append(len(mockInfo[9]))
        flag = mockInfo[1]
        # strand info
        if flag != str(16):
            strand='+'
        else:
            strand='-'
        # parse alignment
        if strand == '+':
            mockQuery = mockInfo[9]
            mockRef = refDict[mockInfo[2]]
            top, middle, bottom, feature = ParseCigar(mockInfo[9], mockRef, mockInfo[5], coords[0], coords[1], coords[2], coords[3], mockAnnot)
            # e = getEfficiency2(mockEdit, mockRef, strand)
            e = getEfficiency(top, bottom, strand)
            if mockGene == 'protein_coding' and mockInfo[2] != 'cerENO2':
                
                mockEffs.append(e)
                readSubs = init_dict_substitutions()
                readSubs = getSubstitutionErrors(mockQuery, mockRef, mockInfo[5], coords[0], coords[1], coords[2], coords[3])
                for sub in readSubs:
                    mockSubs[sub] += readSubs[sub]

            # if mockInfo[2] != 'cerENO2':


            # mockEditDict[mockInfo[0]] = editsPerGeneRegion(mockEdit, feature, bottom)
                
            if e != None:
                mockReadDict[mockInfo[0]] = (e, len(mockQuery))
            # gene types
            
            if mockGene == 'protein_coding' and mockInfo[2] != 'cerENO2':
                mockGeneTypes['protein-coding'] += 1
            elif mockGene == 'rRNA':
                mockGeneTypes['rRNA'] += 1
            elif mockGene == 'ncRNA':
                mockGeneTypes['ncRNA'] += 1
            # elif mockInfo[2] == 'cerENO2':
            #     mockGeneTypes['standards'] += 1
            else:
                mockGeneTypes['other'] += 1
            # read lengths
            mockReadLengths.append(len(mockInfo[9]))
            # substitution errors
            
            
    mockEff = [x for x in mockEffs if x != None]
    for sub in mockSubs:
        totalMockSubs += mockSubs[sub]

    ##### get treated stats #####
    tadReadLengths = []
    tadEffs = []
    tadGeneTypes = {'protein-coding': 0, 'rRNA': 0, 'ncRNA': 0, 'other': 0}
    tadEditDict = {}
    tadReadDict = {}
    tadSubs = init_dict_substitutions()
    totalTadSubs = 0
    print("Reading treated sham file...")
    for tadInfo, tadAnnot, tadGene, tadEdit, coords in treatedSham.readSham():
        # read lengths
        tadReadLengths.append(len(tadInfo[9]))
        flag = tadInfo[1]
        # strand info
        if flag != str(16):
            strand='+'
        else:
            strand='-'
        if strand == '+':
            # parse alignment
            tadQuery = tadInfo[9]
            tadRef = refDict[tadInfo[2]]
            top, middle, bottom, feature = ParseCigar(tadInfo[9], tadRef, tadInfo[5], coords[0], coords[1], coords[2], coords[3], tadAnnot)
            # e = getEfficiency2(tadEdit, tadRef, strand)
            e = getEfficiency(top, bottom, strand) 
            if tadGene == 'protein_coding' and tadInfo[2] != 'cerENO2':
                # tadEffs.append(getEfficiency(top, bottom, strand))
                tadEffs.append(e)
                readSubs = init_dict_substitutions()
                readSubs = getSubstitutionErrors(tadQuery, tadRef, tadInfo[5], coords[0], coords[1], coords[2], coords[3])
                for sub in readSubs:
                    tadSubs[sub] += readSubs[sub]
                tadEditDict[tadInfo[0]] = editsPerGeneRegion(tadEdit, feature, bottom)
            # print(len(feature) == len(tadEdit))
        
            if e != None:
                tadReadDict[mockInfo[0]] = (e, len(tadQuery))
            # gene types
           
            if tadGene == 'protein_coding' and mockInfo[2] != 'cerENO2':
                tadGeneTypes['protein-coding'] += 1
            elif tadGene == 'rRNA':
                tadGeneTypes['rRNA'] += 1
            elif tadGene == 'ncRNA':
                tadGeneTypes['ncRNA'] += 1
            # elif mockInfo[2] == 'cerENO2':
            #     tadGeneTypes['standards'] += 1
            else:
                tadGeneTypes['other'] += 1
            # read lengths
            tadReadLengths.append(len(tadInfo[9]))
            # substitution errors
            
    tadEff = [x for x in tadEffs if x != None]
    
    for sub in tadSubs:
        totalTadSubs += tadSubs[sub]
    
    figureHeight = 11
    # figureWidth = 4.5
    figureWidth = 8.5

    plt.figure(figsize=(figureWidth, figureHeight))

    firstPanelHeight = 2 / figureHeight
    firstPanelWidth = 2.5 / figureWidth

    secondPanelHeight = 2 / figureHeight
    secondPanelWidth = 2.5 / figureWidth

    thirdPanelHeight = 2 / figureHeight
    thirdPanelWidth = 3.5 / figureWidth

    fourthPanelHeight = 2 / figureHeight
    fourthPanelWidth = 3 / figureWidth

    fifthPanelHeight = 4 / figureHeight
    fifthPanelWidth = 2 / figureWidth

    topPanelHeight = 1 / figureHeight
    topPanelWidth = 5 / figureWidth

    topPanel = plt.axes([0.2, 0.9, topPanelWidth, topPanelHeight])
    topPanel.tick_params(bottom=False, labelbottom=False,
                   left=False, labelleft=False,
                   right=False, labelright=False,
                   top=False, labeltop=False)
    topPanel.text(0.1, 0.7, "Mean Tad8A.20 A-G mismatches/# A's: " + '%.2f' % np.mean(tadEff) + '%', fontsize=12)
    topPanel.text(0.1, 0.5, "Median Tad8A.20 A-G mismatches/# A's: " + '%.2f' % np.median(tadEff) + '%', fontsize=12)
    topPanel.text(0.1, 0.3, "Mean Mock A-G mismatches/# A's: " + '%.2f' % np.mean(mockEff) + '%', fontsize=12)
    topPanel.text(0.1, 0.1, "Median Mock A-G mismatches/# A's: " + '%.2f' % np.median(mockEff) + '%', fontsize=12)

    panel1 = plt.axes([0.15, 0.65, firstPanelWidth, firstPanelHeight])
    panel1 = sns.histplot(mockEff, color='blue', alpha = 0.5,label='Mock')
    panel1 = sns.histplot(tadEff, color='red', alpha=0.5, label='Treated')
    panel1.legend()
    panel1.set_xlabel("Editing Efficiency (%)")
    panel1.set_ylabel("Count")
    panel1.set_title("Editing Efficiency per Read")

    panel2 = plt.axes([0.15, 0.35, secondPanelWidth, secondPanelHeight])
    m = [math.log(x,2) for x in mockReadLengths if x != 0]
    m_counts, bins = np.histogram(m)
    m_pdf = m_counts / sum(m_counts)
    m_cdf = np.cumsum(m_pdf)
    panel2.plot(bins[:-1], m_cdf, color='blue', alpha=0.5, label='Mock')
    # panel2 = sns.histplot(m, color='blue', alpha=0.5, label='Mock')

    t = [math.log(x,2) for x in tadReadLengths if x != 0]
    t_counts, bins = np.histogram(t)
    t_pdf = t_counts / sum(t_counts)
    t_cdf = np.cumsum(t_pdf)
    panel2.plot(bins[:-1], t_cdf, color='red', label='Treated')
    # panel2 = sns.histplot(t, color='red', alpha=0.5, label='Treated')

    panel2.legend()
    panel2.set_xlabel("Read Length (log2-scale)")
    panel2.set_ylabel("CDF")
    panel2.set_title("Read Lengths")

    panel3 = plt.axes([0.1, 0.05, thirdPanelWidth, thirdPanelHeight])
    for r in mockSubs:
        mock, = panel3.plot(r, mockSubs[r]/totalMockSubs, marker='o', color='blue', label = 'Mock')
    for r in tadSubs:
        tad, = panel3.plot(r, tadSubs[r]/totalTadSubs, marker='o', color='red', label = 'Treated')
    panel3.set_xlabel("Substitution Type (Genomic Base - Read Base)")
    panel3.set_ylabel("Fraction of Substitutions")
    panel3.set_title("Substitution Errors")
    panel3.legend(handles=[mock, tad])

    panel4 = plt.axes([0.6, 0.65, fourthPanelWidth, fourthPanelHeight])
    
    df1 = pd.DataFrame.from_dict(mockGeneTypes, orient='index', columns=['counts'])
    df1.index.name = 'gene type'
    df1['condition'] = 'mock'
    df2 = pd.DataFrame.from_dict(tadGeneTypes, orient='index', columns=['counts'])
    df2.index.name = 'gene type'
    df2['condition'] = 'treated'
    df = pd.concat([df1, df2])
    df.reset_index(inplace=True)
    
    palette = {'mock': 'blue', 'treated': 'red'}
    panel4 = sns.barplot(data=df, x='gene type', y='counts', hue='condition', palette=palette)
    panel4.legend()
    panel4.set_xlabel("Gene Type")
    panel4.set_ylabel("Count")
    panel4.set_title("Gene Types")

    panel5 = plt.axes([0.65, 0.15, fifthPanelWidth, fifthPanelHeight])
    tadPlotDict = plot_edits(tadEditDict)
    
    panel5 = sns.barplot(x=list(tadPlotDict.keys()), y=list(tadPlotDict.values()), palette="dark")

    panel5.set_xlabel("Gene Region")
    panel5.set_ylabel("# of Edits per Gene Region / # of A's per Gene Region")
    panel5.set_title("Editing per Gene Region in Treated Library")
    
    # print(args[-1])
    plt.savefig(args[-1])
    

if __name__ == "__main__":
    main(sys.argv[1:])

    










