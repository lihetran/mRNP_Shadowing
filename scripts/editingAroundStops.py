'''
September 2, 2025   Liam Tran

This script will examine A-I editing around stop codons. I'll be using gene annotations generated from JA's script assignLongReadsToGenes.py. That script generates a ".longjam" file
where gene features are described by position where C is CDS, U is UTR, T is stop codons, S is starts, a is ambiguous, and no gene is ng. I want to get a distribution of editing +/-50 nts
around the stop codon.

inputs:
    -longjam file
    -bam file
    -reference genome (fasta)

output:
    - A-I editing events around stop codons, .png
'''

import numpy as np
import pysam
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import pickle
import sys


def parsePickleFile(pickleFile):
    """
    dataDict is of format:
    {readID:{readID, chrom, edit_string,barcode,query_string,ref_string,aligned_pairs:...}}
    Will sort reads according to chrom, and output a tally of the number
    of reads per chrom. Will also compute the total size (in nts) of each
    chrom, to get a sense of coverage.

    Will output a dictionary of format:
    {chrom:{readID:{position:edit}}}
    """
    ##start by unpickling the input file
    with open(pickleFile,'rb') as f:
        dataDict=pickle.load(f)
    ## 
    aa=collections.defaultdict(lambda:collections.defaultdict(int))
    for readID,subDict in dataDict.items():
        chrom=subDict['chrom']
        queryLength=len(subDict['read_sequence'])
        aa[chrom]['ct']+=1
        aa[chrom]['length']+=queryLength
        ##
    for k,v in aa.items():
        print('Chromosome %s had %s sequences totaling %s kilo nts.'%(
            k,v['ct'],v['length']/1000))
    ##
    bb=collections.defaultdict(lambda:collections.defaultdict(dict))
    ##
    stops = 0
    non_edited_stops = []
    for readID,subDict in dataDict.items():
        chrom=subDict['chrom']
        #print(chrom,readID)
        editString=subDict['edit_string']
        #print(editString,len(editString))
        readSeq=subDict['read_sequence_aligned']
        #print(readSeq,len(readSeq))
        refSeq=subDict['ref_sequence_aligned']
        #print(refSeq,len(refSeq))
        alignedPairs=subDict['aligned_pairs']
        #print(alignedPairs,len(alignedPairs))
        feature_string=subDict['feature_sequence_aligned']
        if 'T' in feature_string:
            stops += 1
        #sys.exit()
        assert len(editString)==len(readSeq)==len(refSeq)==len(feature_string)
        for ii in range(len(alignedPairs)-1):
            entry=alignedPairs[ii]
            idx=entry[0]
            
            if idx!=None:
                seq=refSeq[ii]
                edit=editString[ii]
                feature=feature_string[ii]
    #             if seq=='A' and edit!='2':
    #                 bb[chrom][readID][idx]=(int(edit), feature) # just A positions
                if seq == 'A' and edit != '2':
                    bb[chrom][readID][idx]=(int(edit), feature) # all A positions
                elif feature == 'T' and edit != '2':
                    bb[chrom][readID][idx]=(int(edit), feature)
                    if '1' not in editString[ii:ii+3] and '2' not in editString[ii:ii+3]:
                        non_edited_stops.append(readID)

    return bb, stops, non_edited_stops

def parse_non_edited_stops(pickleFile, non_edited_stops):
    """
    Get edit information for reads with stop codons but no editing within 3 nts of the stop codon.
    """
    ##start by unpickling the input file
    with open(pickleFile,'rb') as f:
        dataDict=pickle.load(f)

    ## 
    cc=collections.defaultdict(lambda:collections.defaultdict(dict))
    for readID, subDict in dataDict.items():
        if readID in non_edited_stops:
            chrom=subDict['chrom']
            editString=subDict['edit_string']
            readSeq=subDict['read_sequence_aligned']
            refSeq=subDict['ref_sequence_aligned']
            alignedPairs=subDict['aligned_pairs']
            feature_string=subDict['feature_sequence_aligned']
            assert len(editString)==len(readSeq)==len(refSeq)==len(feature_string)
            for ii in range(len(alignedPairs)-1):
                entry=alignedPairs[ii]
                idx=entry[0]
                
                if idx!=None:
                    seq=refSeq[ii]
                    edit=editString[ii]
                    feature=feature_string[ii]
                    if seq == 'A' and edit != '2':
                        cc[chrom][readID][idx]=(int(edit), feature)
                    elif feature == 'T' and edit != '2':
                        cc[chrom][readID][idx]=(int(edit), feature)
    return cc

def extract_features_from_longjam(file):
    

    read_feature_dict = {}
    nextLine = ''
    print("Extracting features from longjam file...")
    with open(file, 'r') as f:
        for line in f:
            if not line.startswith('@'):
                read_id = line.strip().split('\t')[0]
                nextLine = next(f)
                featureString = nextLine.strip().split('\t')[0]
                read_feature_dict[read_id] = featureString

    return read_feature_dict


def mkStopCodonMatrix(data):
    '''
    Create a matrix of A-I editing events around stop codons. I'll do this by creating a matrix that represents a window of 100nt where the stop codon is in the center.
    Stop codons are identified by the 'gene_features' key in the editDict. 'T' denotes the location of the stop codon in the string. 

    data is {chrom:{readID:{position:(edit, featureChar)}}}. The positions only correspond to A positions

    To do: create a separate matrix that stores the number of reads at each position
    '''
    mat1 = np.zeros((0, 101)) # start with empty matrix for counting edits
    mat2 = np.zeros((0, 101)) # start with empty matrix for counting reads
    for chrom, subDict in data.items():
        for readID, editDict in subDict.items():
            positions = sorted(editDict.keys()) # A positions
            features = [editDict[pos][1] for pos in positions] # features at A positions
            edits = [editDict[pos][0] for pos in positions] # edits at A positions
            row1 = np.zeros(101)
            row2 = np.zeros(101)
            if 'T' in features:
                stopCodonPos = features.index('T') # This doesn't work, it gives the location of the stop codon in the features list, not the actual  reference position
                # get the position 
                row1[50] = edits[stopCodonPos]  
                row2[50] = 1
                for j in range(-50, 51):
                    if 0 <= stopCodonPos + j < len(features):
                        row1[50 + j] = edits[stopCodonPos + j] if edits[stopCodonPos + j] != 2 else 0
                        row2[50 + j] = 1
                mat1 = np.vstack((mat1, row1))
                mat2 = np.vstack((mat2, row2))
    return mat1, mat2

def mkStopCodonMatrix2(data, outfile):
    mat1 = np.zeros((0, 101)) # start with empty matrix for counting edits
    mat2 = np.zeros((0, 101)) # start with empty matrix for counting reads
    with open(outfile+'_mat.txt', 'w') as f:
        
        for chrom, subDict in data.items():
            for readID, editDict in subDict.items():
                readInfo = sorted(editDict.items()) # list of (position, (edit, featureChar))
                positions = [item[0] for item in readInfo]
                features = [item[1][1] for item in readInfo]
                edits = [item[1][0] for item in readInfo]
                row1 = np.zeros(101)
                row2 = np.zeros(101)
                if 'T' in features:
                    stopCodonIdx = features.index('T') # index in the features list
                    stopCodonPos = positions[stopCodonIdx] # actual reference position of the stop codon
                    editVal = edits[stopCodonIdx]
                    row1[50] = editVal 
                    row2[50] = 1 
                    for pos, edit, feature in zip(positions, edits, features):
                        offset = pos - stopCodonPos
                        if -50 <= offset <= 50:
                            if offset != 0:  # only update if not already set
                                row1[50 + offset] = edit
                                row2[50 + offset] = 1

                    f.write(f"ReadID: {readID}\n")
                    f.write(f"row1: {row1}\n")
                    f.write(f"row2: {row2}\n\n")

                    mat1 = np.vstack((mat1, row1))
                    mat2 = np.vstack((mat2, row2))
                
    return mat1, mat2


def countReadsWithStops(featureDict):
    count = 0
    for readID, featureString in featureDict.items():
        if 'T' in featureString:
            count += 1
    return count


def collapseForMetaPlot(stopCodonMatrix, readCountMatrix):
    '''
    Collapse the stop codon matrix to get the total number of A-I editing events at each position relative to the stop codon. Normalize each position by total number of items at that position in the stop codon matrix.
    '''
    collapsed1 = np.zeros(stopCodonMatrix.shape[1])
    collapsed2 = np.zeros(readCountMatrix.shape[1])
    for i in range(stopCodonMatrix.shape[1]):
        collapsed1[i] = np.sum(stopCodonMatrix[:, i] == 1)  # count A-I edits
        collapsed2[i] = np.sum(readCountMatrix[:, i] == 1)  # count reads
    # collapsed /= np.sum(stopCodonMatrix != 0, axis=0)  # normalize by number of reads at each position
    return collapsed1, collapsed2

def normalize(collapsed1, collapsed2):
    '''
    Normalize the collapsed matrices by the total number of reads.
    '''
    # normalized_mat = collapsed1 / collapsed2
    normalized_mat = np.divide(collapsed1, collapsed2)
    return normalized_mat

def plotMeanEditingInGeneRegion(normalized_mat, out_file):
    '''
    Plot the mean A-I editing events in the gene region. The CDS is the first 50 positions in the matrix and the UTR are the positions 65-100
    '''
    import matplotlib.pyplot as plt

    figureHeight = 8
    figureWidth = 4

    panelHeight = 6 / figureHeight
    panelWidth = 3 / figureWidth

    CDS_array = normalized_mat[0:50]
    UTR_array = normalized_mat[65:100]

    CDS_mean = np.mean(CDS_array)
    UTR_mean = np.mean(UTR_array)
    plt.figure(figsize=(figureWidth, figureHeight))

    panel = plt.axes([0.17, 0.15, panelWidth, panelHeight])
    panel.bar(['CDS', 'UTR'], [CDS_mean, UTR_mean], color=['grey', 'lightgrey'])
    panel.set_ylabel('Mean A-I Editing Events')
    panel.set_title('Mean A-I Editing Events in Gene Regions')
    plt.savefig(out_file + '_mean.png', dpi=300)
    plt.savefig(out_file + '_mean.svg')
    plt.close()

def plotEditingAroundStops(normalized_mat, out_file):
    import matplotlib.pyplot as plt

    figureHeight = 6
    figureWidth = 10

    panelHeight = 4 / figureHeight
    panelWidth = 8 / figureWidth

    plt.figure(figsize=(figureWidth, figureHeight))
    panel = plt.axes([0.1, 0.15, panelWidth, panelHeight])
    panel.plot(np.arange(0, 101, 1), normalized_mat)
    panel.set_xlabel('Position relative to stop codon')
    panel.set_xticks(np.arange(0, 101, 10))
    panel.set_xticklabels(np.arange(-50, 51, 10))
    panel.set_ylabel('Edits')
    panel.set_title('A-I Editing Events Around Stop Codons')
    plt.savefig(out_file + '.png', dpi=300)
    plt.savefig(out_file + '.svg')
    plt.close()


def main(args):
    pickleFile = args[0]
    out_file = args[1]
    data, stops, non_edited_stops = parsePickleFile(pickleFile)
    print(f"Number of reads with stop codons: {stops}")
    mat1, mat2 = mkStopCodonMatrix2(data, out_file)
    print(mat1.shape)
    print('Collapsing stop codon matrix...')
    collapsed1, collapsed2 = collapseForMetaPlot(mat1, mat2)
    print(collapsed1)
    print(collapsed2)
    normalized_mat = normalize(collapsed1, collapsed2)
    print(normalized_mat)

    plotEditingAroundStops(normalized_mat, out_file)
    plotMeanEditingInGeneRegion(normalized_mat, out_file)

    print(f"Number of reads with stop codons but no editing within 3 nts: {len(non_edited_stops)}")
    #write non_edited_stops to a file
    # with open(out_file + '_non_edited_stops.txt', 'w') as f:
    #     for readID in non_edited_stops:
    #         f.write(readID + '\n')
    print("Processing non-edited stops...")
    data2 = parse_non_edited_stops(pickleFile, non_edited_stops)
    mat1b, mat2b = mkStopCodonMatrix2(data2, out_file + '_binned')
    print(mat1b.shape)
    print('Collapsing stop codon matrix for non-edited stops...')
    collapsed1b, collapsed2b = collapseForMetaPlot(mat1b, mat2b)
    print(collapsed1b)
    print(collapsed2b)
    normalized_mat_b = normalize(collapsed1b, collapsed2b)
    print(normalized_mat_b)
    plotEditingAroundStops(normalized_mat_b, out_file + '_binned')
    plotMeanEditingInGeneRegion(normalized_mat_b, out_file + '_binned')

    # longjamFile = args[1]
    # featureDict = extract_features_from_longjam(longjamFile)
    # stopCodonCount = countReadsWithStops(featureDict)
    # print(f"Number of reads with stop codons: {stopCodonCount}")
    # data, ct = addGeneFeaturesToData(data, featureDict)
    # stopCodonMatrix, stops = mkStopCodonMatrix(data, ct)
    # print(f"Stop codon matrix shape: {stopCodonMatrix.shape}")
    
    # print(collapsed)

if __name__=='__main__':
    main(sys.argv[1:])   
