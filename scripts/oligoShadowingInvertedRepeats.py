"""
Joshua Arribere, June 10, 2025

Script to analyze oligo shadowing data.

Input: pickle2 file - from Liam, contains a dictionary of read_info, with keys
    being read IDs, and values being a dictionary of:
    {edit_string:...,barcode:...,query_string:...,ref_string:...,
        aligned_pairs: ...}
    This info is for the aligned region, not the raw read.
    Align, aligned_pairs needs to have the absolute index w.r.t. the ref.

Output: graph of sequence:edit freq for all 3mers about a single adenosine.

run as python3 oligoShadowingInvertedRepeats.py pickle2 outPrefix
"""
import sys, common, pickle, collections, numpy
from logJosh import Tee
from math import log
from pyx import *

def getEditFreq(dataDict):
    """
    dataDict is of format:
    {readID:{edit_string,barcode,query_string,ref_string,aligned_pairs:...}
    Will sort reads according to barcode, and output a tally of the number
    of reads per barcode. Will also compute the total size (in nts) of each
    barcode, to get a sense of coverage.
    Will return a dict of
    {bc:position:freq}
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
    bb=collections.defaultdict(lambda:collections.defaultdict(list))
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
                    bb[barCode][absIdx].append(int(edit))
    ##
    #bb is of format:
    #{bc:{3mer:{position:[0,1,1,0...1]}}}
    ##
    cc={}
    for bc,v in bb.items():
        cc[bc]={}
        for position,valList in v.items():
            cc[bc][int(position)]=sum(valList)/len(valList)
    ##
    return cc

def getTheData(seqEditFreqs):
    """
    seqEditFreqs={position:freq}
    Will return ordered list
    """
    ##
    ps=[entry for entry in seqEditFreqs.keys()]
    ps.sort()
    aa=[]
    for p in ps:
        aa.append((p,seqEditFreqs[p]))
    ##
    return aa

def plotSeqEditFreqs(seqEditFreqs,outPrefix,invertedRepeats):
    """
    seqEditFreqs={position:freq}
    Will plot:
        x-axis: position along transcript
        y-axis: freq
        Will show each bc as a different symbol/color
        Will stack plots to get to 16 plots
    """
    nts=['A','T','G','C']
    seqs=[nt1+'A'+nt2 for nt1 in nts for nt2 in nts]
    bcs=['bc3','bc4','bc5','bc6']
    colors=[color.cmyk(0,0,0,1),
        color.cmyk(1,0.5,0,0),
        color.cmyk(0.1,0.05,0.9,0),
        color.cmyk(0.97,0,0.75,0)]
    ##
    c=canvas.canvas()
    ##
    g=graph.graphxy(width=8,height=2,
        x=graph.axis.linear(min=0,max=700,
            title='Position Along Transcript'),
        y=graph.axis.linear(min=0,max=1,
            title='Edit Freq'))
    ##
    for site in invertedRepeats:
        leftBound=site[0]
        rightBound=site[1]
        c.fill(path.rect(g.width*leftBound/700,0,
            g.width*(rightBound-leftBound)/700,2),
            [deco.filled([color.cmyk(0,0,0,0.2),
            color.transparency(0.2)])])
    ##
    theData=getTheData(seqEditFreqs)
    g.plot(graph.data.points(theData,x=1,y=2),
        [graph.style.line([style.linestyle.solid,
            color.cmyk.black])])
    ##
    c.insert(g)
    ##
    ##
    c.writePDFfile(outPrefix)

def main(args):
    inFile,outPrefix=args[0:]
    ##
    with open(inFile,'rb') as f:
        dataDict=pickle.load(f)
    ##
    seqEditFreqs=getEditFreq(dataDict)
    ##
    ##will now look for short inverted repeats in the reference seq.
    refSeq='GCGGtTgtCtgTTCCgCTCTAGAAATAATTTTGTTTAACTTTAAtAAGGAGgTATACATATGgcatgcCAcCATCATCAcCATCATACAACCACTGGaTCGtcaGGaGTaTTtACAtTaGAAGATTTtGTaGGGGAtTGGaGACAGACAGCaGGaTAtAAtCTGGAtCAAGTatTaGAACAGGGAGGaGTGTCatcaTTGTTTCAGAATtTaGGGGTGTCaGTAACaCCGATaCAAAGGATaGTaCTGtcaGGaGAAAATGGGCTGAAGATaGAtATaCATGTaATaATaCCGTATGAAGGaCTGtcaGGaGAtCAAATGGGaCAGATaGAAAAAATaTTTAAGGTGGTGTAtCCaGTGGATGATCATCAtTTTAAGGTGATaCTGCAtTATGGaACACTGGTAATaGAtGGGGTaACGCCGAAtATGATaGAtTATTTtGGACGGCCGTATGAAGGaATaGCaGTGTTtGAtGGaAAAAAGATaACaGTAACAGGGACaCTGTGGAAtGGaAAtAAAATaATaGAtGAGaGaCTGATaAAtCCaGAtGGaTCaCTGCTGTTtaGAGTAACaATaAAtGGAGTGACaGGaTGGCGGCTGTGtGAAaGaATaCTGGCGTAAGATggaCTTTTTCCCTCTGCCAAAAATTATGGGGACATCATGAAGCCCCTTGAGCATCTGACTTCTGGCTAATAAtGGcggTTTgTcTTCgTTGCNNNNCCGCAATACGTAACTGAACGAAGTACAGG'
    refSeq=refSeq.upper()
    k=5
    l=4
    invertedRepeatSites=[]
    for ii in range(0,len(refSeq)-2*k-l):
        leftStem=refSeq[ii:ii+k]
        for jj in range(ii+k,ii+k+l):
            rightStem=refSeq[jj:jj+k]
            if leftStem==common.revCompl(rightStem):#then it's an inverted repeat.
                invertedRepeatSites.append([ii,jj+k])
                #print(ii,jj,jj-ii-k,leftStem,rightStem)
    #print(invertedRepeatSites)
    plotSeqEditFreqs(seqEditFreqs['bc3'],outPrefix,invertedRepeatSites)
    ##
    ##

if __name__=='__main__':
    Tee()
    main(sys.argv[1:])
