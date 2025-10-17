"""
Joshua Arribere, June 27, 2025

Script to analyze oligo shadowing data.

Input: pickle2 file - from Liam, contains a dictionary of read_info, with keys
    being read IDs, and values being a dictionary of:
    {edit_string:...,barcode:...,query_string:...,ref_string:...,
        aligned_pairs: ...}
    This info is for the aligned region, not the raw read.
    Align, aligned_pairs needs to have the absolute index w.r.t. the ref.
    bounds - in the format
        xLeft,xRight
    oligoLocations - line and space-delimited of the format:
        x1 x2
        ...
        xn xn+1
        Will shade these locations on the plot

Output: stacked bar graphs of edit and non-edit frequency for all barcodes.

run as python3 oligoShadowingStackedBarPlots.py pickle2File 0,695 outPrefix
"""
import sys, common, pickle, collections, numpy
from logJosh import Tee
from math import log
from pyx import *

def parseOligos(oligoFile):
    aa=[]
    with open(oligoFile,'r') as f:
        for line in f:
            aa.append([int(entry) for entry in line.strip().split()])
    return aa

def getEditFreq(dataDict):
    """
    dataDict is of format:
    {readID:{edit_string,barcode,query_string,ref_string,aligned_pairs:...}
    Will sort reads according to barcode, and output a tally of the number
    of reads per barcode. Will also compute the total size (in nts) of each
    barcode, to get a sense of coverage.
    
    Will output a dictionary of: {barcode:position:editFreq}
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
    #{bc:{position:[0,1,1,0...1]}}
    ##
    cc={}
    for bc,v in bb.items():
        cc[bc]={}
        for position,editList in v.items():
            cc[bc][position]=sum(editList)/len(editList)
    ##
    return cc

def getTheData(bcs,seqEditFreqs):
    """
    bcs is a list of the barcodes in a specific order
    seqEditFreqs={bc:position:freq}
    """
    ##first get all the positions
    aa=[]
    for bc in bcs:
        aa.append(list([entry for entry in seqEditFreqs[bc].items()]))
    ##
    return aa

def getTheData2(seq,bcs,seqEditFreqs):
    """
    seq is the current seq of interest
    bcs is the barcodes in a particular order
    seqEditFreqs={bc:seq:[(position,freq,ct),...]}
    """
    ##first get all the positions
    positions=[]
    for bc in bcs:
        for entry in seqEditFreqs[bc][seq]:
            positions.append(entry[0])
    positions=list(set(positions))
    positions.sort()
    ##
    aa=[]
    for bc in bcs:
        temp=[]
        for entry in seqEditFreqs[bc][seq]:
            temp.append(entry[:2])
        temp2=[entry[1] for entry in temp]
        ##
        avg1=numpy.average(temp2)
        std1=numpy.std(temp2)
        ##
        temp3=[]
        for entry in temp:
            temp3.append((entry[0],(entry[1]-avg1)/std1))
        aa.append(temp3)
    ##
    return aa

def plotEditFreqs(seqEditFreqs,oligoLocations,outPrefix):
    """
    seqEditFreqs={bc:position:freq}
    Will plot each bc
    Will shade oligoLocations=[(x1,x2),...(xn,xn+1)]
    """
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
    ##plot the oligoLocations
    for oligo in oligoLocations:
        leftBound=oligo[0]
        rightBound=oligo[1]
        c.fill(path.rect(g.width*leftBound/700,0,
            g.width*(rightBound-leftBound)/700,2),
            [deco.filled([color.cmyk(0,0,0,0.2),
                color.transparency(0.2)])])
    ##
    theData=getTheData(bcs,seqEditFreqs)
    for jj in range(len(bcs)):
        g.plot(graph.data.points(sorted(theData[jj]),x=1,y=2),
            [graph.style.line([style.linestyle.solid,
                colors[jj]])])
    ##
    c.insert(g)
    ##
    c.writePDFfile(outPrefix)

def plotStackedBarGraph(seqEditFreqs,oligoLocations,refSeq,\
    bounds,outPrefix):
    """
    seqEditFreqs={bc:position:freq}
    oligoLocations=[(x1,x2),...(xn,xn+1)]
    refSeq='AGTACGTAC...GTACGACGTAC' or whatever the seq is
    bounds=[leftBound,rightBound]
    
    For each bc in seqEditFreqs, this function will make a
    stacked bar plot of the freq at each position within
    bounds, inclusive. It will also shade the oligoLocations
    if they overlap with bounds. Will also put the nts of
    refSeq along the bottom as axis labels.
    """
    bcs=['bc3','bc4','bc5','bc6']
    ##flip the order of barcodes, so that as we go through
    ##them we can just add them vertically until we reach
    ##the first one
    bcs.reverse()
    ##
    colors=[color.cmyk(0,0,0,1),
        color.cmyk(1,0.5,0,0),
        color.cmyk(0.1,0.05,0.9,0),
        color.cmyk(0.97,0,0.75,0)]
    ##also need to reverse the order of colors
    colors.reverse()
    ##initialize the canvas
    c=canvas.canvas()
    ##initialize these variables to make downstream
    ##more readable
    leftBound=bounds[0]
    rightBound=bounds[1]
    graphWidth=8
    ntWidth=graphWidth/(rightBound-leftBound)
    ##next variable controls spacing in between nts. Multiply
    ##this number by two to get the spacing between consecurive
    ##nts
    visualOffset=0.1
    ##initialize the ticks for use in all the graphs
    myTicks=[graph.axis.tick.tick(ii,label=refSeq[ii]) \
        for ii in range(leftBound,rightBound+1)]
    ##
    for ii in range(len(bcs)):
        bc=bcs[ii]
        ##initialize the graph
        g=graph.graphxy(width=graphWidth,height=2,ypos=ii*3.5,
            x=graph.axis.linear(min=leftBound,max=rightBound,
                title='Position Along Transcript',
                manualticks=myTicks,parter=None),
            y=graph.axis.linear(min=0,max=1,
                title='Edit Freq %s'%(bc)))
        ##
        ##first draw the location of the oligos
        for oligo in oligoLocations:
            if oligo[1]<=leftBound:
                pass##then oligo not in the field of view
            elif oligo[0]>=rightBound:
                pass##then oligo not in the field of view
            else:
                leftBoxEnd=max([leftBound,oligo[0]-0.5])
                rightBoxEnd=min([rightBound,oligo[1]+0.5])
                c.fill(path.rect(\
                    g.width*(leftBoxEnd-leftBound)/(rightBound-leftBound),\
                    g.ypos,\
                    g.width*(rightBoxEnd-leftBoxEnd)/(rightBound-leftBound),\
                    g.height),
                    [deco.filled([color.cmyk(0,0,0,0.2),
                        color.transparency(0.2)])])
                ##
        ##now plot the data
        for position,freq in seqEditFreqs[bc].items():
            if leftBound<position<rightBound+1:
                positionOffset=position-leftBound
                ##draw the edited (red) part
                c.fill(path.rect(\
                    g.width*(positionOffset-0.5+visualOffset)/(rightBound-leftBound),\
                    g.ypos,\
                    g.width*(1-2*visualOffset)/(rightBound-leftBound),\
                    freq*g.height),\
                    [deco.filled([color.cmyk(0,0.8,1,0)])])
                ##now draw the unedited (grey) part
                c.fill(path.rect(\
                    g.width*(positionOffset-0.5+visualOffset)/(rightBound-leftBound),\
                    g.ypos+freq*g.height,\
                    g.width*(1-2*visualOffset)/(rightBound-leftBound),\
                    (1-freq)*g.height),\
                    [deco.filled([color.cmyk(0,0,0,0.8)])])
        ##
        ##
        c.insert(g)
    ##
    c.writePDFfile(outPrefix)




def main(args):
    inFile,bounds,oligoFile,outPrefix=args[0:]
    ##
    ##
    oligoLocations=parseOligos(oligoFile)
    ##
    with open(inFile,'rb') as f:
        dataDict=pickle.load(f)
    ##
    seqEditFreqs=getEditFreq(dataDict)
    ##
    #plotEditFreqs(seqEditFreqs,oligoLocations,outPrefix)
    ##
    refSeq='GCGGtTgtCtgTTCCgCTCTAGAAATAATTTTGTTTAACTTTAAtAAGGAGgTATACATATGgcatgcCAcCATCATCAcCATCATACAACCACTGGaTCGtcaGGaGTaTTtACAtTaGAAGATTTtGTaGGGGAtTGGaGACAGACAGCaGGaTAtAAtCTGGAtCAAGTatTaGAACAGGGAGGaGTGTCatcaTTGTTTCAGAATtTaGGGGTGTCaGTAACaCCGATaCAAAGGATaGTaCTGtcaGGaGAAAATGGGCTGAAGATaGAtATaCATGTaATaATaCCGTATGAAGGaCTGtcaGGaGAtCAAATGGGaCAGATaGAAAAAATaTTTAAGGTGGTGTAtCCaGTGGATGATCATCAtTTTAAGGTGATaCTGCAtTATGGaACACTGGTAATaGAtGGGGTaACGCCGAAtATGATaGAtTATTTtGGACGGCCGTATGAAGGaATaGCaGTGTTtGAtGGaAAAAAGATaACaGTAACAGGGACaCTGTGGAAtGGaAAtAAAATaATaGAtGAGaGaCTGATaAAtCCaGAtGGaTCaCTGCTGTTtaGAGTAACaATaAAtGGAGTGACaGGaTGGCGGCTGTGtGAAaGaATaCTGGCGTAAGATggaCTTTTTCCCTCTGCCAAAAATTATGGGGACATCATGAAGCCCCTTGAGCATCTGACTTCTGGCTAATAAtGGcggTTTgTcTTCgTTGCNNNNCCGCAATACGTAACTGAACGAAGTACAGG'
    refSeq=refSeq.upper()
    ##
    bounds=[entry for entry in map(int,bounds.split(','))]
    plotStackedBarGraph(seqEditFreqs,oligoLocations,refSeq,\
        bounds,outPrefix+'.stackedBarGraph')
    ##

if __name__=='__main__':
    Tee()
    main(sys.argv[1:])
