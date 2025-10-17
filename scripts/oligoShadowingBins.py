"""
Joshua Arribere, June 24, 2025

Script to analyze oligo shadowing data.

Input: pickle2 file - from Liam, contains a dictionary of read_info, with keys
    being read IDs, and values being a dictionary of:
    {edit_string:...,barcode:...,query_string:...,ref_string:...,
        aligned_pairs: ...}
    This info is for the aligned region, not the raw read.
    Align, aligned_pairs needs to have the absolute index w.r.t. the ref.
    oligoBounds.txt - line and space-delimited file of oligo bounds, e.g.
        150 184
        ...

Output: graph of the distribution of number of edits within the oligo annealing
    regions.

"""
import sys, common, pickle, collections, numpy
from logJosh import Tee
from math import log
from pyx import *

def getEditFreq(dataDict,oligoSites):
    """
    dataDict is of format:
    {readID:{edit_string,barcode,query_string,ref_string,aligned_pairs:...}
    Will sort reads according to barcode, and output a tally of the number
    of reads per barcode. Will also compute the total size (in nts) of each
    barcode, to get a sense of coverage.
    Will determine whether each A-containing position is edited or not.
    Will make an intermediate dict of format
    {bc:readID:{position:edited}}
    Will then look up reads that span oligoSites=[[x1,x2],[x3,x4],...]
    and make an object that is:
    {bc:readID:[[score1,ct1],[score2,ct2],...]
    where scorei is the total number of edits in the xi,xi+1 window, and cti
    is the total number of scored sites in that window
    """
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
    bb=collections.defaultdict(lambda:collections.defaultdict(lambda:\
        collections.defaultdict(int)))
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
    #bb is of format:
    #{bc:{readID:{position:0/1}}}
    ##
    cc={}
    for bc,v in bb.items():
        cc[bc]={}
        for readID,subDict in bb[bc].items():
            cc[bc][readID]=[]
            for oligo in oligoSites:
                leftBound=oligo[0]
                rightBound=oligo[1]
                ct=0
                score=0
                for position,edit in subDict.items():
                    if leftBound<=position<=rightBound:
                        ct+=1
                        if edit==1:
                            score+=1
                cc[bc][readID].append([score,ct])
    ##
    return cc

def getTheData(seqEditFreqs,ii,bcs):
    """
    seqEditFreqs=
    {bc:readID:[[editCt1,totalCt1],...]}
    bcs is the barcodes in a particular order
    ii is which editCt/totalCt to extract
    """
    aa=[]
    maxVal=[]
    for bc in bcs:
        temp=[]
        for readID,valList in seqEditFreqs[bc].items():
            temp.append(valList[ii])
            maxVal.append(valList[ii][-1])
        aa.append(temp)
    ##
    maxVal=max(maxVal)
    bb=[]
    for entry in aa:
        temp=[]
        for entry2 in entry:
            if entry2[-1]==maxVal:
                temp.append(entry2[0])
        bb.append(temp)
    ##bb is not a list of lists where the inner most list is just the number
    ##of edits for all reads that were editable across the whole oligo region.
    ##
    cc=[]
    for entry in bb:
        temp=[]
        ct=0
        k=len(entry)
        for ii in range(maxVal+1):
            ct+=entry.count(ii)
            temp.append((ii,ct/k))
        cc.append(temp)
    ##
    return cc

def plotSeqEditFreqs(seqEditFreqs,outPrefix,oligoSites):
    """
    seqEditFreqs={bc:{readID:[[numEdits1,numScored1],...]}}
    For each bc, will plot a CDF of the number of reads w/ numEdits1 for
    the range [0,numScored1]
    """
    bcs=['bc3','bc4','bc5','bc6']
    colors=[color.cmyk(0,0,0,1),
        color.cmyk(1,0.5,0,0),
        color.cmyk(0.1,0.05,0.9,0),
        color.cmyk(0.97,0,0.75,0)]
    ##
    c=canvas.canvas()
    for ii in range(len(oligoSites)):
        g=graph.graphxy(width=2,height=2,xpos=ii*6,
            key=graph.key.key(pos='tr',hinside=0),
            x=graph.axis.linear(title='Num Oligo%s Edits'%(ii+1)),
            y=graph.axis.linear(min=0,max=1,
                title='CDF'))
        ##
        theData=getTheData(seqEditFreqs,ii,bcs)
        ##
        for jj in range(len(bcs)):
            g.plot(graph.data.points(theData[jj],x=1,y=2,
                title=bcs[jj]),
                [graph.style.line([style.linestyle.solid,
                    colors[jj]])])
        ##
        c.insert(g)
        ##
    ##
    c.writePDFfile(outPrefix)

def main(args):
    inFile,oligoFile,outPrefix=args[0:]
    ##
    ##parse the oligos
    oligoSites=[]
    with open(oligoFile,'r') as f:
        for line in f:
            line=line.strip().split()
            oligoSites.append([entry for entry in map(int,line)])
    ##
    with open(inFile,'rb') as f:
        dataDict=pickle.load(f)
    ##
    seqEditFreqs=getEditFreq(dataDict,oligoSites)
    ##
    plotSeqEditFreqs(seqEditFreqs,outPrefix,oligoSites)
    ##

if __name__=='__main__':
    Tee()
    main(sys.argv[1:])
