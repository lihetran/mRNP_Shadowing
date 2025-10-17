'''
September 12, 2025 LT

This script will plot the A-I editing in the CDS vs the UTR of mRNAs from a nanopore sequencing experiment.

Input:
    -pickle file from "shadowingBamToPickle2.py"
Output:
    -svg/png barplot of A-I editing in CDS vs UTR
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
    {chrom:{readID:{position:(edit, feature)}}}
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
                if feature != 'T':
                    if seq == 'A' and edit != '2':
                        bb[chrom][readID][idx]=(int(edit), feature) # all A positions
                else:
                    # skip 14 nts after stop codon
                    ii+=14 if ii+14<len(alignedPairs) else len(alignedPairs)-1

    return bb

def tallyEditsPerFeature(dataDict):
    '''
    Tally the number of edits per feature (CDS vs UTR) for each chromosome.
    dataDict is of format:
    {chrom:{readID:{position:(edit, feature)}}}

    Will output a dictionary of format:
    {feature:{'total':int,'edited':int}}}
    '''
    aa = {'CDS':{'total':0,'edited':0}, 'UTR':{'total':0,'edited':0}}
    for chrom,subDict in dataDict.items():
        for readID,subSubDict in subDict.items():
            for position,(edit,feature) in subSubDict.items():
                if feature=='C':
                    aa['CDS']['total']+=1
                    if edit==1:
                        aa['CDS']['edited']+=1
                elif feature=='U':
                    aa['UTR']['total']+=1
                    if edit==1:
                        aa['UTR']['edited']+=1

    return aa

def plotEditsPerFeature(tallyDict,outputPrefix):
    '''
    Plot the number of edits per feature (CDS vs UTR) for each chromosome.
    tallyDict is of format:
    {feature:{'total':int,'edited':int}}}
    '''
    ##convert to dataframe for plotting
    tally_df = pd.DataFrame.from_dict(tallyDict, orient='index')
    tally_df.reset_index(inplace=True)
    tally_df.rename(columns={'index':'feature'}, inplace=True)
    df = tally_df[['feature', 'total', 'edited']]
    ##compute fraction edited
    df['fraction_edited']=df['edited']/df['total']
    print(df)
    ##plot
    figureHeight=10
    figureWidth=6

    panelHeight=8 / figureHeight
    panelWidth=4 / figureWidth

    plt.figure(figsize=(figureWidth,figureHeight))
    panel=plt.axes([0.15,0.1,panelWidth,panelHeight])
    sns.barplot(data=df,x='feature',y='fraction_edited', hue='feature',ax=panel, palette = 'Greys')
    panel.set_ylabel('Fraction Edited', fontsize=14)
    panel.set_xlabel('Gene Feature', fontsize=14)
    panel.set_title('A-I Editing in CDS vs UTR', fontsize=16)
    plt.savefig(outputPrefix+'.png',dpi=300)
    plt.savefig(outputPrefix+'.svg')
    plt.close()

def main(args):
    if len(args)!=2:
        print('Usage: python plotEditingInCDSvsUTR.py <input_pickle_file> <output_prefix>')
        sys.exit(1)
    input_pickle_file=args[0]
    output_prefix=args[1]
    ##parse the pickle file
    dataDict=parsePickleFile(input_pickle_file)
    ##tally edits per feature
    tallyDict=tallyEditsPerFeature(dataDict)
    ##plot edits per feature
    plotEditsPerFeature(tallyDict,output_prefix)

if __name__=='__main__':
    main(sys.argv[1:])

    