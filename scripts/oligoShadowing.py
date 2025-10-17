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

def getTheData(seq,bcs,seqEditFreqs):
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
        aa.append(temp)
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

def plotSeqEditFreqs(seqEditFreqs,outPrefix):
    """
    seqEditFreqs={bc:{seq:[(position,freq,ct),...]}}
    Will make plots for each seq:
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
    ii=0
    for seq in seqs:
        if ii==0:
            g=graph.graphxy(width=8,height=1,ypos=ii*1.5,
                x=graph.axis.linear(min=0,max=700,
                    title='Position Along Transcript'),
                y=graph.axis.linear(min=0,max=1,
                    title=seq))
            ##
            g.plot([graph.data.function("x(y)=150"),
                graph.data.function("x(y)=184"),
                graph.data.function("x(y)=446"),
                graph.data.function("x(y)=476")],
                [graph.style.line([style.linestyle.solid])])
            ##
            theData=getTheData(seq,bcs,seqEditFreqs)
            for jj in range(len(bcs)):
                g.plot(graph.data.points(theData[jj],x=1,y=2),
                    [graph.style.symbol(graph.style.symbol.triangle,
                        symbolattrs=[colors[jj],deco.filled],
                        size=0.05)])
            ##
            c.insert(g)
            ##
        else:
            g2=graph.graphxy(width=8,height=1,ypos=ii*1.5,
                x=graph.axis.linkedaxis(g.axes["x"]),
                y=graph.axis.linear(min=0,max=1,
                    title=seq))
            ##
            g2.plot([graph.data.function("x(y)=150"),
                graph.data.function("x(y)=184"),
                graph.data.function("x(y)=446"),
                graph.data.function("x(y)=476")],
                [graph.style.line([style.linestyle.solid])])
            ##
            theData=getTheData(seq,bcs,seqEditFreqs)
            for jj in range(len(bcs)):
                g2.plot(graph.data.points(theData[jj],x=1,y=2),
                    [graph.style.symbol(graph.style.symbol.triangle,
                        symbolattrs=[colors[jj],deco.filled],
                        size=0.05)])
            ##
            c.insert(g2)
            ##
        ii+=1
    ##
    c.writePDFfile(outPrefix)

def plotSeqEditFreqs2(seqEditFreqs,outPrefix,sitesToPlot):
    """
    seqEditFreqs={bc:{seq:[(position,freq,ct),...]}}
    Will make plots for each seq:
        x-axis: position along transcript
        y-axis: freq, z-score normalized
        Will only plot bc3
        Will stack plots to get to 16 plots
    """
    nts=['A','T','G','C']
    seqs=[nt1+'A'+nt2 for nt1 in nts for nt2 in nts]
    bcs=['bc3']
    colors=[color.cmyk(0,0,0,1),
        color.cmyk(1,0.5,0,0),
        color.cmyk(0.1,0.05,0.9,0),
        color.cmyk(0.97,0,0.75,0)]
    ##
    c=canvas.canvas()
    ii=0
    allData=[]
    for seq in seqs:
        if ii==0:
            g=graph.graphxy(width=8,height=1,ypos=ii*1.5,
                x=graph.axis.linear(min=0,max=700,
                    title='Position Along Transcript'),
                y=graph.axis.linear(min=-4,max=2.5,
                    title='Z-score'))
            ##
            for site in sitesToPlot:
                #g.plot([graph.data.function("x(y)=%s"%(site[0])),
                #    graph.data.function("x(y)=%s"%(site[1]))],
                #    [graph.style.line([style.linestyle.solid])])
                ##
                g.fill(path.rect(g.width*site[0]/700,0,
                    g.width*(site[1]-site[0])/700,g.height),
                    [color.rgb.black,style.linewidth(0),
                    color.transparency(0.8)])
            ##
            theData=getTheData2(seq,bcs,seqEditFreqs)
            for jj in range(len(bcs)):
                g.plot(graph.data.points(theData[jj],x=1,y=2),
                    [graph.style.symbol(graph.style.symbol.circle,
                        symbolattrs=[colors[jj],deco.filled],
                        size=0.05)])
            ##
            allData+=theData[jj]
            ##
            c.insert(g)
            ##
        else:
            #g2=graph.graphxy(width=8,height=1,ypos=ii*1.5,
            #    x=graph.axis.linkedaxis(g.axes["x"]),
            #    y=graph.axis.linear(min=-4,max=2.5,
            #        title=seq))
            ##
            #g2.plot([graph.data.function("x(y)=150"),
            #    graph.data.function("x(y)=184"),
            #    graph.data.function("x(y)=446"),
            #    graph.data.function("x(y)=476")],
            #    [graph.style.line([style.linestyle.solid])])
            ##
            theData=getTheData2(seq,bcs,seqEditFreqs)
            for jj in range(len(bcs)):
                g.plot(graph.data.points(theData[jj],x=1,y=2),
                    [graph.style.symbol(graph.style.symbol.circle,
                        symbolattrs=[colors[jj],deco.filled],
                        size=0.05)])
            ##
            allData+=theData[jj]
            ##
            #c.insert(g2)
            ##
        ii+=1
    ##
    c.writePDFfile(outPrefix)
    ##
    temp=[entry[0] for entry in allData]
    temp.sort()
    temp2=[]
    for idx in temp:
        for entry in allData:
            if entry[0]==idx:
                temp2.append((entry[0],round(float(entry[1]),2)))
    print(temp2)

def computeMI(temp):
    """
    temp={p1:p2:[list of 00,01,10,11]}
    """
    aa={}
    for p1,v in temp.items():
        aa[p1]={}
        for p2,valList in v.items():
            ##pseudocounting
            #valList+=['00','01','10','11']
            ##
            valListLength=len(valList)
            p00=valList.count('00')/valListLength
            p01=valList.count('01')/valListLength
            p10=valList.count('10')/valListLength
            p11=valList.count('11')/valListLength
            ##
            p0_=p00+p01
            p1_=p10+p11
            p_0=p00+p10
            p_1=p01+p11
            ##
            if p00*p01*p10*p11*p0_*p1_*p_0*p_1!=0:
                MI=p00*log(p00/(p0_*p_0),2)+\
                   p01*log(p01/(p0_*p_1),2)+\
                   p10*log(p10/(p1_*p_0),2)+\
                   p11*log(p11/(p1_*p_1),2)
            else:
                MI='na'
            ##
            """
            if p1==64 and p2==69:
                print(p00,p01,p10,p11)
                print(p00/(p0_*p_0))
                print(p01/(p0_*p_1))
                print(p10/(p1_*p_0))
                print(p11/(p1_*p_1))
            """
            ##
            aa[p1][p2]=MI
    ##
    return aa

def mkHeatMap(temp2,outPrefix):
    """
    temp2={p1:{p2:MI}}
    Will plot a heatmap.
    'na' may be an a value, which will appear as a grey box.
    """
    c=canvas.canvas()
    ##
    theVals=[]
    thePositions=[]
    for p1,v in temp2.items():
        thePositions.append(p1)
        for p2,val in v.items():
            thePositions.append(p2)
            theVals.append(val)
    theVals=[entry for entry in theVals if entry!='na']
    theMax=max(theVals)
    print('Max and Min mutual information:',theMax,min(theVals))
    thePositions=list(set(thePositions))
    thePositions.sort()
    ##
    for ii in range(len(thePositions)):
        p1=thePositions[ii]
        c.text(ii+0.5,0,p1,[text.halign.boxcenter,text.valign.top])
        for jj in range(len(thePositions)):
            p2=thePositions[jj]
            c.text(-0.5,jj+1+0.5,p2,[text.halign.boxright,text.valign.middle])
            if p1 in temp2:
                if p2 in temp2[p1]:
                    MI=temp2[p1][p2]
                else:
                    MI='na'
                if MI!='na':
                    theFactor=round(MI/theMax,2)
                    if theFactor<=0:
                        theFactor=0
                    c.fill(path.rect(ii,jj+1,1,1),
                        [deco.filled([
                        color.cmyk(theFactor,0.5*theFactor,0,0)])])
                else:
                    c.fill(path.rect(ii,jj+1,1,1),
                        [deco.filled([
                        color.cmyk(0,0,0,0.2)])])
    ##
    ##now add boxes for oligo1 and oligo2
    oligoPositions=[(150,184),(446,476)]
    for oligoPosition in oligoPositions:
        leftBound=oligoPosition[0]
        rightBound=oligoPosition[1]
        for position in thePositions:
            if leftBound>=position:
                temp1=position
            if rightBound>=position:
                temp2=position
        ##
        temp1=thePositions.index(temp1)
        temp2=thePositions.index(temp2)
        temp3=temp2-temp1
        ##
        c.stroke(path.rect(temp1+1,temp1+2,
            temp3,temp3),
            [color.cmyk(0,0,0,1)])
    ##
    c.writePDFfile(outPrefix)

def getMIs(dataDict,outPrefix):
    """
    dataDict is of format:
    {readID:{edit_string,barcode,query_string,ref_string,aligned_pairs:...}
    Will sort reads according to barcode, and output a tally of the number
    of reads per barcode. Will also compute the total size (in nts) of each
    barcode, to get a sense of coverage.
    For each edit site, will determine whether that site was indeed edited or
    not (via lookup). Will then loop over every other edit site for the same
    molecule, and determine whether the second site was also edited (or not).
    Will then store this information (as 1/1, 1/0, 0/1, or 0/0).
    Once that's been done for all molecules, will then compute mutal
    information for all pairs of sites.
    Will return a final dict of format:
    {bc:position1:position2:(MI,n)} where n is number of molecules used for
    that MI calculation.
    """
    ##
    print('Now computing MI...')
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
                    if len(seq)>=3:
                        if seq[1]=='A':
                            bb[barCode][readID][absIdx]=int(edit)
    ##
    #bb is of format:
    #{bc:{readID:{position:0/1}}}
    ##
    #cc={}
    ##initially wanted to do this as a function and return dictionaries, but
    ##memory requirements were too high.
    for bc,v in bb.items():
        if len(v)>=100:##
            print('Working on %s...'%(bc))
            #cc[bc]=collections.defaultdict(lambda:collections.defaultdict(list))
            temp=collections.defaultdict(lambda:collections.defaultdict(list))
            cntr=0
            for readID,v2 in v.items():
                if cntr%100==0:
                    print(readID,cntr)
                cntr+=1
                for p1,edit1 in v2.items():
                    pass
                    for p2,edit2 in v2.items():
                        #cc[bc][p1][p2].append('%s%s'%(edit1,edit2))
                        temp[p1][p2].append('%s%s'%(edit1,edit2))
                if cntr==1000:
                    print(cntr,bc)
                    break
            ##now compute MI
            temp2=computeMI(temp)
            ##
            ##now plot the info
            print('Making MI heatmap for bc %s...'%(bc))
            mkHeatMap(temp2,outPrefix+'.%s'%(bc))
    ##
    ##
    #return cc
    pass

def main(args):
    inFile,outPrefix=args[0:]
    ##
    with open(inFile,'rb') as f:
        dataDict=pickle.load(f)
    ##
    #seqEditFreqs=getEditFreq(dataDict)
    ##
    #plotSeqEditFreqs(seqEditFreqs,outPrefix)
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
    ##now just plot the bc3 sample (no oligos), and z-score normalize
    ##around each 3mers' editing freq
    #plotSeqEditFreqs2(seqEditFreqs,outPrefix+'.zScore',invertedRepeatSites)
    ##
    ##will now look for mutual information of editing
    ##
    getMIs(dataDict,outPrefix+'.MIs')
    ##

if __name__=='__main__':
    Tee()
    main(sys.argv[1:])
