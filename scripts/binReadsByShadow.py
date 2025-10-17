'''
August 20, 2025 Liam Tran

This script will bin reads from a shadow pickle by the absence of edits in a window. 

The dictionary in the pickle has the form {read.query_name: read_dict = {
                'edit_string': edit_string,
                'barcode': barcode,
                'bar_seq': bar_seq,
                'read_sequence': read.query_sequence,
                'read_sequence_aligned': read_string,
                'ref_sequence_aligned': ref_string,
                'aligned_pairs': aligned_pairs,
                'absolute_indices': absolute_indices
            }}

I will use absolute indices to determine the presence of edits in a window. This is a binary string where 1's indicate the presence of an edit and 0's indicate the absence of an edit.

input: shadow pickle
       TSV of window location
output: plot of binned reads with grey areas indicating no edits and gold areas indicating edits
'''

import sys, common, pickle, collections, math, numpy, scipy.stats
from pyx import *
from icecream import ic

def parsePickleFile(pickleFile):
    """
    dataDict is of format:
    {readID:{edit_string,barcode,query_string,ref_string,aligned_pairs:...}}
    Will sort reads according to barcode, and output a tally of the number
    of reads per barcode. Will also compute the total size (in nts) of each
    barcode, to get a sense of coverage.
    
    Will output a dictionary of format:
    {bc:{readID:{position:edit}}}
    """
    ##start by unpickling the input file
    with open(pickleFile,'rb') as f:
        dataDict=pickle.load(f)
    ##
    aa=collections.defaultdict(lambda:collections.defaultdict(int))
    for readID,subDict in dataDict.items():
        bc=subDict['barcode']
        queryLength=len(subDict['read_sequence'])
        aa[bc]['ct']+=1
        aa[bc]['length']+=queryLength
    ##
    for k,v in aa.items():
        print('Barcode %s had %s sequences totaling %s kilo nts.'%(\
            k,v['ct'],v['length']/1000))
    ##
    bb=collections.defaultdict(lambda:collections.defaultdict(dict))
    ##
    for readID,subDict in dataDict.items():
        barCode=subDict['barcode']
        #print(barCode,readID)
        editString=subDict['edit_string']
        #print(editString,len(editString))
        readSeq=subDict['read_sequence_aligned']
        #print(readSeq,len(readSeq))
        refSeq=subDict['ref_sequence_aligned']
        #print(refSeq,len(refSeq))
        alignedPairs=subDict['aligned_pairs']
        #print(alignedPairs,len(alignedPairs))
        #sys.exit()
        for ii in range(len(alignedPairs)-1):
            entry=alignedPairs[ii]
            idx=entry[0]
            absIdx=entry[1]
            if idx!=None and absIdx!=None:
                seq=refSeq[ii]
                edit=editString[ii]
                if seq=='A' and edit!='2' and absIdx<=695:
                    ##695 is where the RT primer binds--anything past this
                    ##is artifact.
                    bb[barCode][readID][absIdx]=int(edit)
    ##
    return bb

def getBoundaryLocations(boundaryFile):
    aa=[]
    with open(boundaryFile,'r') as f:
        for line in f:
            line=line.strip().split()
            aa.append([entry for entry in map(int,line)])
    return aa

def bin_reads_by_shadow(read_dict, boundary_locations):
    """
    Bin reads from a shadow pickle by the absence of edits in a window.
    """
    binned_reads = collections.defaultdict(lambda: collections.defaultdict(list))
    start, end  = boundary_locations[0][0], boundary_locations[-1][-1]
    
    print(f"Start: {start}, End: {end}")
    for barCode, readDict in read_dict.items():
        for readID, editDict in readDict.items():
            for absIdx, edit in editDict.items():
                window_edits = []
                
                if start <= absIdx < end:
                    window_edits.append(edit)
                if 1 not in window_edits:
                    binned_reads[barCode][(start, end)].append((readID, absIdx, edit))
                if len(window_edits) > 1:
                    print(window_edits)
    return binned_reads

def plot_binned_reads(binned_reads):
    """
    Plot the binned reads with grey areas indicating no edits and gold areas indicating edits. This will be like a custom IGV track.
    """
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(10, 6))
    pass

def main(args):
    pickleFile = args[0]
    boundaryFile = args[1]
    outFile = args[2]

    read_dict = parsePickleFile(pickleFile)
    boundary_locations = getBoundaryLocations(boundaryFile)
    print(f"Boundary locations: {boundary_locations}")
    binned_reads = bin_reads_by_shadow(read_dict, boundary_locations)
    # print(binned_reads)

if __name__ == '__main__':
    main(sys.argv[1:])