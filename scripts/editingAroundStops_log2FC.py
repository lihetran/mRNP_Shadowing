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
    edited_stops = []
    ##
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
                    elif '1' in editString[ii:ii+3] and '2' not in editString[ii:ii+3]:
                        edited_stops.append(readID)

    return bb, stops, non_edited_stops, edited_stops

def parse_stops(pickleFile, edited_stops, non_edited_stops):
    """
    Get edit information for reads with stop codons.
    """
    ##start by unpickling the input file
    with open(pickleFile,'rb') as f:
        dataDict=pickle.load(f)

    ## 
    aa=collections.defaultdict(lambda:collections.defaultdict(dict))
    bb=collections.defaultdict(lambda:collections.defaultdict(dict))
    ##
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
                        aa[chrom][readID][idx]=(int(edit), feature)
                    elif feature == 'T' and edit != '2':
                        aa[chrom][readID][idx]=(int(edit), feature)

        elif readID in edited_stops:
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
                        bb[chrom][readID][idx]=(int(edit), feature)
                    elif feature == 'T' and edit != '2':
                        bb[chrom][readID][idx]=(int(edit), feature)
    return aa, bb

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

def log2FC(normalized_mat1, normalized_mat2, window_size=5):
    '''
    Compute the log2 fold change between two normalized matrices.
    '''
    with np.errstate(divide='ignore', invalid='ignore'):
        log2fc = np.log2(np.divide(normalized_mat1, normalized_mat2))
        # log2fc[np.isinf(log2fc)] = 10  # replace inf with really large number
        log2fc[np.isinf(log2fc)] = 0  # replace inf with 0
        log2fc = np.nan_to_num(log2fc)  # replace nan with 0

    # smooth the log2fc using a rolling average with window size 5 except at 50:53
    # log2fc = pd.Series(log2fc).rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()
    log2fc1 = pd.Series(log2fc[:50]).rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()
    log2fc2 = pd.Series(log2fc[53:]).rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()
    log2fc = np.concatenate((log2fc1, log2fc[50:53], log2fc2))
    log2fc = np.nan_to_num(log2fc)  # replace nan with 0 again after smoothing
    return log2fc

def plotLog2FC(log2fc, output_file):
    figureHeight=4
    figureWidth=7
    panelHeight = 3 / figureHeight
    panelWidth = 5 / figureWidth

    plt.figure(figsize=(figureWidth, figureHeight))
    panel = plt.axes([0.15, 0.15, panelWidth, panelHeight])
    # panel.plot(np.arange(0, 101, 1), log2fc)
    panel.plot(np.arange(0, 50, 1), log2fc[:50], color='blue')
    panel.plot(np.arange(53, 101, 1), log2fc[53:], color='blue')
    # grey out positions 50, 51, 52
    panel.axvspan(50, 52, color='grey')
    # add horizontal line at y=0
    panel.axhline(0, color='black', linestyle='--')
    # add vertical lines for 20 nt footprint on either side of stop codon
    panel.axvline(41, color='red', linestyle='--')
    panel.axvline(61, color='red', linestyle='--')
    panel.set_xlabel('Position relative to stop codon')
    panel.set_xticks(np.arange(0, 101, 10))
    panel.set_xticklabels(np.arange(-50, 51, 10))
    panel.set_ylabel('Log2 Fold Change')
    panel.set_title('Log2 Fold Change of A-I Editing Events Around Stop Codons')
    plt.savefig(output_file + '.png', dpi=300)
    plt.savefig(output_file + '.svg')
    plt.close()

def main(args):
    if len(args) != 3:
        print("Usage: python editingAroundStops_log2FC.py <pickle_file> <output_prefix> <window_size>")
        sys.exit(1)
    pickle_file = args[0]
    output_prefix = args[1]
    window_size = int(args[2])

    dataDict, stops, non_edited_stops, edited_stops = parsePickleFile(pickle_file)
    print(f"Total reads with stop codons: {stops}")
    print(f"Reads with non-edited stop codons: {len(non_edited_stops)}")
    print(f"Reads with edited stop codons: {len(edited_stops)}")

    non_edited_data, edited_data = parse_stops(pickle_file, edited_stops, non_edited_stops) 
    non_edited_mat1, non_edited_mat2 = mkStopCodonMatrix2(non_edited_data, output_prefix + '_non_edited')
    edited_mat1, edited_mat2 = mkStopCodonMatrix2(edited_data, output_prefix + '_edited')
    collapsed_non_edited1, collapsed_non_edited2 = collapseForMetaPlot(non_edited_mat1, non_edited_mat2)
    collapsed_edited1, collapsed_edited2 = collapseForMetaPlot(edited_mat1, edited_mat2)
    normalized_non_edited = normalize(collapsed_non_edited1, collapsed_non_edited2)
    normalized_edited = normalize(collapsed_edited1, collapsed_edited2)
    log2fc = log2FC(normalized_non_edited, normalized_edited, window_size=window_size)
    plotLog2FC(log2fc, output_prefix + '_log2FC_' + str(window_size))
    print("All done!")

if __name__=='__main__':
    main(sys.argv[1:])


