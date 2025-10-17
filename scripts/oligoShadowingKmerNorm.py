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
    Graph of raw edit frequency. Graph of zscore-normalized edit freq.
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
    Will identify the frequency of edits for each of the 16nt combos before
    and after an A (NAN), additonally sorted by position. Will output that as 
    a dictionary of: {barcode:{seq:[(position,edited,all)]}} where
    edited,all are the counts of edited at that site, and all reads spanning
    that site.
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
        collections.defaultdict(list)))
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
                seq=refSeq[ii-1:ii+2]
                edit=editString[ii]
                if ' ' not in seq and edit!='2' and absIdx<=695:
                    ##695 is where the RT primer binds--anything past this
                    ##is artifact.
                    if refSeq[ii]=='A' and len(seq)==3:
                        bb[barCode][seq][absIdx].append(int(edit))
    ##
    #bb is of format:
    #{bc:{3mer:{position:[0,1,1,0...1]}}}
    ##
    cc={}
    for bc,v in bb.items():
        cc[bc]={}
        for seq,subDict in bb[bc].items():
            cc[bc][seq]=[]
            for position,valList in subDict.items():
                cc[bc][seq].append((position,
                    sum(valList)/len(valList),
                    len(valList)))
    ##
    return cc

def getTheData(seqEditFreqs):
    """
    seqEditFreqs={seq:[(position,freq,ct),...]}
    Will return an ordered list of [(position,freq),...]
    """
    aa=[]
    for k,v in seqEditFreqs.items():
        for entry in v:
            aa.append(entry)
    ##
    return sorted(aa)

def getTheData2(seqEditFreqs):
    """
    seqEditFreqs={seq:[(position,freq,ct),...]}
    Will z-score normalize all the freqs of the same seq,
    and then order all the positions and return as a single
    list
    """
    aa=[]
    for seq,valList in seqEditFreqs.items():
        ##
        freqList=[entry[1] for entry in valList]
        freqMean=numpy.average(freqList)
        freqStd=numpy.std(freqList)
        ##
        for entry in valList:
            aa.append((entry[0],(entry[1]-freqMean)/freqStd))
    ##
    return sorted(aa)

def plotEditFreqPerSeq(seqEditFreqs,outPrefix):
    """
    seqEditFreqs={seq:[(position,freq,ct),...]}
    Will plot x-axis as each of the seqs, and y-axis as the freqs.
        Will order seqs by the median edit freq across all entries
        in the [(position,freq,ct),...] list
    """
    ##first get all the medians
    aa={}
    for seq,valList in seqEditFreqs.items():
        aa[seq]=[entry[1] for entry in valList]
        aa[seq]=numpy.median(aa[seq])
    medianList=[entry for entry in aa.values()]
    medianList.sort()
    ##now order the data for plotting.
    ##
    xAxis=[]
    plotVals=[]
    medVals=[]
    offset=0.15
    ii=0
    for median in medianList:
        ii+=1
        for seq,val in aa.items():
            if median==val:
                xAxis.append((ii,seq))
                plotVals+=[(ii-offset,entry[1]) for entry in seqEditFreqs[seq]]
                medVals.append((ii+offset,median))
        ##
    ##use the xAxis above to create x-axis labels
    myTicks=[graph.axis.tick.tick(entry[0],label=entry[1], \
         labelattrs=[trafo.rotate(90)]) for entry in xAxis]
    ##initialize the graph
    g=graph.graphxy(width=5,height=5,
        x=graph.axis.linear(min=0,max=len(xAxis)+1,
            manualticks=myTicks,parter=None),
        y=graph.axis.linear(min=0,max=1,title='Edit Frequency'))
    ##plot the raw values
    g.plot(graph.data.points(plotVals,x=1,y=2),
        [graph.style.symbol(graph.style.symbol.circle,
            symbolattrs=[color.cmyk.black,deco.filled],
            size=0.05)])
    ##plot the medians
    g.plot(graph.data.points(medVals,x=1,y=2),
        [graph.style.symbol(graph.style.symbol.square,
            symbolattrs=[color.cmyk(0,0.8,1,0),deco.filled],
            size=0.1)])
    ##
    g.writePDFfile(outPrefix)

def plotSeqEditFreqs(seqEditFreqs,outPrefix):
    """
    seqEditFreqs={seq:[(position,freq,ct),...]}
    Will make one plot of overall freq, sorted by position, and
    irrespective of sequence. Will make a second plot where the
    freqs are all z-transformed w/in the distribution of each
    seq's freqs.
    """
    ##
    c=canvas.canvas()
    ##initialize second graph (z-score)
    ##since want to link its axis to the first graph
    gZ=graph.graphxy(width=8,height=2,
        x=graph.axis.linear(min=0,max=700,
            title='Position Along Transcript'),
        y=graph.axis.linear(max=2,title='Z-score'))
    ##first graph
    g=graph.graphxy(width=8,height=2,ypos=2.1,
        x=graph.axis.linkedaxis(gZ.axes["x"]),
        y=graph.axis.linear(min=0,max=1,
            title='Edit Freq'))
    ##
    theData=getTheData(seqEditFreqs)
    g.plot(graph.data.points(theData,x=1,y=2),
        [graph.style.line([style.linestyle.solid,
            color.cmyk(0,0,0,1)])])
    ##
    c.insert(g)
    ##proceed w/ plotting on the second graph
    theData2=getTheData2(seqEditFreqs)
    gZ.plot(graph.data.points(theData2,x=1,y=2),
        [graph.style.line([style.linestyle.solid,
            color.cmyk(0,0,0,1)])])
    ##
    c.insert(gZ)
    ##
    c.writePDFfile(outPrefix)

def main(args):
    inFile,outPrefix=args[0:]
    ##
    with open(inFile,'rb') as f:
        dataDict=pickle.load(f)
    ##
    seqEditFreqs=getEditFreq(dataDict)
    ##plot the edit frequency per 3mer
    plotEditFreqPerSeq(seqEditFreqs['bc3'],outPrefix)
    ##plot the overall edit frequency, as well as the z-score
    ##normalized edit freq
    plotSeqEditFreqs(seqEditFreqs['bc3'],outPrefix+'.positionEdit')
    ##

if __name__=='__main__':
    Tee()
    main(sys.argv[1:])
