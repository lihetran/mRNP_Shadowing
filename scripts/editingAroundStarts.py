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
    starts = 0
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
        if 'S' in feature_string:
            starts += 1
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
                
                bb[chrom][readID][idx]=(int(edit), feature) # all positions

    return bb, starts

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

def addGeneFeaturesToData(data, featureDict):
    print("Adding gene features to data...")
    ct = 0
    copy = data.copy()
    for chrom, subDict in copy.items():
        for readID in subDict.keys():
            if readID in featureDict:
                data[chrom][readID]['gene_features'] = featureDict[readID]
                ct += 1
    print("Added gene features for %d reads." % ct)
    return data, ct

def mkStartCodonMatrix(data, ct):
    '''
    Create a matrix of A-I editing events around start codons. I'll do this by creating a matrix that represents a window of 100nt where the start codon is in the center.
    Start codons are identified by the 'gene_features' key in the editDict. 'S' denotes the location of the start codon in the string. 
    '''


    # mat = np.zeros((len(data), 101)) # 101 columns for -50 to +50 around start codon
    # create empty numpy array
    mat = np.zeros((ct, 101))
    print("Creating start codon matrix of size %s..." % (mat.shape,))

    starts = 0
    for chrom, subDict in data.items():

        for i, (readID, editDict) in enumerate(subDict.items()):
            if 'gene_features' in editDict:
                feature_string = editDict['gene_features']
                for j, char in enumerate(feature_string):

                    if char == 'S':
                        starts += 1
                        # get position of start codon
                        startCodonPos = j
                        # set window around start codon
                        window_st = startCodonPos - 50
                        window_en = startCodonPos + 51
                        # print(f"Read {readID} has start codon at position {startCodonPos} with window {window_st} to {window_en}")
                        # get editing events in window
                        for k in range(window_st, window_en):
                            # if k in editDict and window_st > 0 and window_en < len(feature_string):
                            if k in editDict:
                                # adjust indexing for matrix
                                mat[i, k - window_st] = editDict[k]
    # remove empty rows
    # find indices of empty rows
    mat = mat[~np.all(mat == 0, axis=1)]
    return mat, starts

def mkStartCodonMatrix2(data):
    '''
    Create a matrix of A-I editing events around start codons. I'll do this by creating a matrix that represents a window of 100nt where the start codon is in the center.
    Start codons are identified by the 'gene_features' key in the editDict. 'S' denotes the location of the start codon in the string. 

    data is {chrom:{readID:{position:(edit, featureChar)}}}. The positions only correspond to A positions

    To do: create a separate matrix that stores the number of reads at each position
    '''
    mat1 = np.zeros((0, 101)) # start with empty matrix for counting edits
    mat2 = np.zeros((0, 101)) # start with empty matrix for counting reads
    for chrom, subDict in data.items():
        for readID, editDict in subDict.items():
            positions = sorted(editDict.keys())
            features = [editDict[pos][1] for pos in positions]
            edits = [editDict[pos][0] for pos in positions]
            row1 = np.zeros(101)
            row2 = np.zeros(101)
            if 'S' in features:
                startCodonPos = features.index('S')
                row1[50] = edits[startCodonPos]
                row2[50] = 1
                for j in range(-50, 51):
                    if 0 <= startCodonPos + j < len(features):
                        row1[50 + j] = edits[startCodonPos + j] if edits[startCodonPos + j] != 2 else 0
                        row2[50 + j] = 1
                mat1 = np.vstack((mat1, row1))
                mat2 = np.vstack((mat2, row2))
    return mat1, mat2

def countReadsWithStops(featureDict):
    count = 0
    for readID, featureString in featureDict.items():
        if 'T' in featureString:
            count += 1
    return count


def collapseForMetaPlot(startCodonMatrix, readCountMatrix):
    '''
    Collapse the start codon matrix to get the total number of A-I editing events at each position relative to the start codon. Normalize each position by total number of items at that position in the start codon matrix.
    '''
    collapsed1 = np.zeros(startCodonMatrix.shape[1])
    collapsed2 = np.zeros(readCountMatrix.shape[1])
    for i in range(startCodonMatrix.shape[1]):
        collapsed1[i] = np.sum(startCodonMatrix[:, i] == 1)  # count A-I edits
        collapsed2[i] = np.sum(readCountMatrix[:, i] == 1)  # count reads
    # collapsed /= np.sum(startCodonMatrix != 0, axis=0)  # normalize by number of reads at each position
    return collapsed1, collapsed2

def normalize(collapsed1, collapsed2):
    '''
    Normalize the collapsed matrices by the total number of reads.
    '''
    normalized_mat = collapsed1 / collapsed2
    return normalized_mat

def plotEditingAroundStarts(normalized_mat, out_file):
    import matplotlib.pyplot as plt

    figureHeight = 6
    figureWidth = 10

    panelHeight = 4 / figureHeight
    panelWidth = 8 / figureWidth

    plt.figure(figsize=(figureWidth, figureHeight))
    panel = plt.axes([0.1, 0.15, panelWidth, panelHeight])
    panel.plot(np.arange(0, 101, 1), normalized_mat)
    panel.set_xlabel('Position relative to start codon')
    panel.set_xticks(np.arange(0, 101, 10))
    panel.set_xticklabels(np.arange(-50, 51, 10))
    panel.set_ylabel('Edits')
    panel.set_title('A-I Editing Events Around Start Codons')
    plt.savefig(out_file + '.png', dpi=300)
    plt.savefig(out_file + '.svg')
    plt.close()


def main(args):
    pickleFile = args[0]
    out_file = args[1]
    data, stops = parsePickleFile(pickleFile)
    print(f"Number of reads with start codons: {stops}")
    mat1, mat2 = mkStartCodonMatrix2(data)
    print(mat1.shape)
    print('Collapsing start codon matrix...')
    collapsed1, collapsed2 = collapseForMetaPlot(mat1, mat2)
    print(collapsed1)
    print(collapsed2)
    normalized_mat = normalize(collapsed1, collapsed2)
    print(normalized_mat)

    plotEditingAroundStarts(normalized_mat, out_file)

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
