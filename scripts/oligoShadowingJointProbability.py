"""
Joshua Arribere, June 24, 2025

Script to compute the probability of observing a sequence of edits and
    non edits for each molecule/read.

    To do this, it'll first compute a reference probability of editing
    from a reference bc (input). Will then use joint probability across
    a window (user input) of nts, and then display some number (10?) of
    reads, with position on the x-axis and y-axis as -logProb. Will
    also plot a summary metric...a metaRead plot, where x-axis is again
    position, and y-axis is...fraction of reads w/ a significant p-val?
    Not 100% on these metrics, so look at the code to ensure I didn't
    change my mind while I wrote the code.

Input: pickle2 - a pickled file from Liam
    refBc - probably bc3, a "no oligo" library
    window - probably 30, but whatever you want. This is in nts, not in
        number of sites (e.g., 30nts might have 4 As, or 10 As).
    numReads - will plot this many reads as examples
    oligoFile - line and space-delimited file of oligo boundaries.

Output: might output some pickle files to make processing faster during
    writing this code.
    Plots of metaRead significant hits, with individual reads tiled 
        above that.

run as python3 oligoShadowingJointProbability.py pickle2 refBc window 
    numReads outPrefix
"""
import sys, common, pickle, collections, math, numpy, scipy.stats
import sklearn
from logJosh import Tee
from pyx import *

def getOligoLocations(oligoFile):
    aa=[]
    with open(oligoFile,'r') as f:
        for line in f:
            line=line.strip().split()
            aa.append([entry for entry in map(int,line)])
    return aa

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

def getReferenceFreq(refDict):
    """
    refDict is of the format
    {readID:{position:edit}}
    For every position, will compute a frequency of edit and a frequency of 
    nonedit, and output it as:
    {position:(freq0,freq1)}
    """
    aa=collections.defaultdict(list)
    for readID,positionDict in refDict.items():
        for position,editStatus in positionDict.items():
            aa[position].append(editStatus)
    ##aa is of the format {position:[0,1,1,0,0,0,1,1,1,],...}
    bb={}
    for position,editList in aa.items():
        freq0=editList.count(0)/len(editList)
        freq1=editList.count(1)/len(editList)
        bb[position]=(freq0,freq1)
    ##
    return bb

def computeJointProbabilitiesPerRead(editFreqPerRead,referenceFreq,ntWindow):
    """
    editFreqPerRead=
    {bc:readID:position:edit/not}
    referenceFreq={position:(freq0,freq1)
    For every readID, will scan from 0 through 695 and if there's at least three
    sites in that window, will compute the joint probability. Will return:
    {bc:readID:[(position,prob),...]
    where positions are ordered
    """
    print('#####\nComputing joint probabilities for every read...\n#####')
    ##
    aa=collections.defaultdict(lambda:collections.defaultdict(list))
    ##
    N=1000
    print('Limiting to %s reads per sample...'%(N))
    cntr=0
    for bc,readDict in editFreqPerRead.items():
        for readID,positionDict in readDict.items():
            cntr+=1
            for ii in range(0,695,1):
                temp=[]
                for jj in range(ii,ii+ntWindow):
                    if jj in positionDict:
                        editVal=positionDict[jj]
                        temp.append(referenceFreq[jj][editVal]) # refFreq[prob of edit or non edit]
                if len(temp)>=5:
                    jointProb=math.prod(temp)
                    aa[bc][readID].append((ii+ntWindow/2.,jointProb))
            if cntr==N:
                print('Done w/ barcode %s after %s reads.'%(bc,cntr))
                cntr=0
                break
        if cntr!=0:
            print('Done w/ barcode %s after %s reads.'%(bc,cntr))
            cntr=0
    ##
    return aa

def getPoints(jointProbabilitiesPerRead,bcList,position):
    """
    jointProbabilitiesPerRead is of the format:
    {bc:readID:[(x,prob),...]}
    yesOligo is a list of barcodes
    position is an x value.
    Will return a list of all probs that correspond to all x values
    where x==position
    """
    aa=[]
    for bc,readDict in jointProbabilitiesPerRead.items():
        if bc in bcList:
            for readID,positionList in readDict.items():
                for entry in positionList:
                    if entry[0]==position:
                        aa.append(entry[1])
    return aa

def getPointsToPlot(allPositives,allNegatives):
    """
    given two lists of positive and negative values, will loop through every
    100th value and determine the TPR and FPR at each of those values. Will
    return a list of [(FPR1,TPR1),...] for each of those 100 values
    """
    ##
    allVals=allPositives+allNegatives
    ##
    N=100
    cutoffs=numpy.percentile(allVals,\
        [i*100/N for i in range(1,N)])
    ##cutoffs is a list of scores at every integer-valued percentile
    aa=[]
    for cutoff in cutoffs:
        TPR=(scipy.stats.percentileofscore(allPositives,cutoff)/100)
        FPR=(scipy.stats.percentileofscore(allNegatives,cutoff)/100)
        aa.append((FPR,TPR))
    ##
    return aa

def getAUC(aPs,aNs):
    """
    Given a series of positive score values (allPositives, aPs) and 
    negative score values (allNegatives, aNs), will compute AUC.
    Will return AUC as a float.
    """
    ##note: although aPs are "positives", they are 0 below. That's
    ##b/c the p-value is LOWER for the positives than the negatives.
    ##alternatively, could -log10 the scores.
    auc=sklearn.metrics.roc_auc_score(\
        [0]*len(aPs)+[1]*len(aNs),\
        aPs+aNs)
    ##
    return auc

def getOptimal(aPs,aNs):
    """
    Given a series of positive score values (allPositives, aPs) and 
    negative score values (allNegatives, aNs), will compute the optimal
    cutoff to maximize the difference between TPR and FPR.
    Will return optimal as a float.
    """
    ##note: although aPs are "positives", they are 0 below. That's
    ##b/c the p-value is LOWER for the positives than the negatives.
    ##alternatively, could -log10 the scores.
    fpr,tpr,thresholds=sklearn.metrics.roc_curve(\
        [0]*len(aPs)+[1]*len(aNs),\
        aPs+aNs)
    ##
    optimal_idx = numpy.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    ##
    return optimal_threshold

def getClassifications(jointProbabilitiesPerRead,cutoffs,bcs):
    """
    jointProbabilitiesPerRead is of the format:
    {bc:readID:[(x,prob),...]}
    bcs is an ordering of the barcodes
    cutoffs is [(position,AUC,cutoff),...]
    Based on whether readIDs have probs <=cutoff at each position,
    will classify reads as having a shadow / not at each position.
    Will return a list of format:
    [(ct,ct1,ct2,ct12),...]
    where:
        ct - no shadows at either position
        ct1 - shadow at the first position
        ct2 - shadow at the second position
        ct12 - shadow at both positions
    actually, instead of cts, will be frequencies
    """
    ##
    aa=[]
    ##
    for bc in bcs:
        temp2=collections.defaultdict(int)
        for readID,positionList in jointProbabilitiesPerRead[bc].items():
            shadows=['na','na']
            for entry in positionList:
                position=entry[0]
                prob=entry[1]
                ##
                for ii in range(2):
                    site=cutoffs[ii]
                    if position==site[0]:
                        if prob<=site[2]:
                            shadows[ii]=1
                        else:
                            shadows[ii]=0
            ##
            if 'na' not in shadows:
                temp2['%s%s'%(shadows[0],shadows[1])]+=1
        ##
        k=sum(temp2.values())
        aa.append([temp2['00']/k,\
            temp2['01']/k,
            temp2['10']/k,
            temp2['11']/k])
    ##
    return aa

def mkROCCurves(jointProbabilitiesPerRead,oligoLocations,outPrefix):
    """
    jointProbabilitiesPerRead is of the format:
    {bc:readID:[(x,prob),...]}
    oligoLocations is of the format: [[x1,x2],...[xn,xn+1]]
    Will loop through every location in each oligo location, and compute
    the TPR and FPR for every 100th (maybe?) p-value cutoff. Then plot that as
    a ROC curve. Will array these out for each oligo, so that we have a ROC curve
    for every position.
    
    Will also plot the AUC and optimal cutoff for each position in the oligo,
    on a separate plot.
    """
    ##initialize object to store AUC and optimal cutoff for later
    ##plotting
    aa=[]
    ##bb will store the AUC/cutoff for the central position of each oligo
    bb=[]
    ##
    for ii in range(len(oligoLocations)):
        oligo=oligoLocations[ii]
        ##create new list item in the meta object
        aa.append([])
        ##
        leftBound=oligo[0]
        rightBound=oligo[1]
        ##
        c=canvas.canvas()
        outPrefix2=outPrefix+'.oligo%s'%(ii+1)
        if ii==0:
            noOligo=['bc3','bc4']
            yesOligo=['bc5','bc6']
        elif ii==1:
            noOligo=['bc3','bc5']
            yesOligo=['bc4','bc6']
        ##
        cntr=0
        for position in range(leftBound,rightBound):
            ##
            allPositives=getPoints(jointProbabilitiesPerRead,yesOligo,position)
            allNegatives=getPoints(jointProbabilitiesPerRead,noOligo,position)
            pointsToPlot=getPointsToPlot(allPositives,allNegatives)
            ##now use sklearn to get the AUC and optimal cutoff
            aa[-1].append([position,getAUC(allPositives,allNegatives),
                getOptimal(allPositives,allNegatives)])
            if position==leftBound+(rightBound-leftBound)/2:
                print('Central position of oligo:',aa[-1][-1])
                bb.append(aa[-1][-1])
            #print(aa[-1][-1])
            ##
            g=graph.graphxy(width=2,height=2,xpos=cntr*4,ypos=ii*4,
                x=graph.axis.linear(min=0,max=1,title='FPR'),
                y=graph.axis.linear(min=0,max=1,title='TPR'))
            g.text(g.xpos+g.width/2.,\
                    g.ypos+g.height+.5,\
                    '%s,%s'%(ii+1,position))
            ##
            g.plot(graph.data.points(pointsToPlot,x=1,y=2),
                [graph.style.line([style.linestyle.solid,
                    color.cmyk.black])])
            ##
            g.plot(graph.data.function("y(x)=x",min=0,max=1),
                [graph.style.line([style.linestyle.dotted,
                    color.cmyk.black])])
            ##
            c.insert(g)
            ##
            cntr+=1
        ##
        print('Writing ROC curve plots for oligo%s...'%(ii+1))
        print(outPrefix2)
        ##
        c.writePDFfile(outPrefix2)
    ##
    print('Done w/ ROC curves...')
    ##
    print('#####\nPlotting AUC...\n#####')
    ##aa is a list of length equal to the number of oligos sites.
    ##Each entry in aa is a list of [(position,AUC,cutoff),...]
    ##where AUC is the AUC for that position, and cutoff is the 
    ##optimal cutoff for that position.
    ##Will plot the AUC as a bar graph and the optimal value
    ##as a scatter plot.
    for ii in range(len(aa)):
        c=canvas.canvas()
        ##
        entry=aa[ii]
        ##first the AUC graph
        g=graph.graphxy(width=4,height=2,
            x=graph.axis.bar(title='Position'),
            y=graph.axis.linear(min=0.8,max=1,
                title='AUC'))
        ##
        g.plot(graph.data.points(entry,xname=1,y=2),
            [graph.style.bar([color.cmyk.white])])
        ##
        c.insert(g)
        ##now the optimal cutoff graph
        g2=graph.graphxy(width=4,height=2,ypos=g.height+2,
            x=graph.axis.linear(title='Position'),
            y=graph.axis.log(title='Optimal Cutoff'))
        ##
        g2.plot(graph.data.points(entry,x=1,y=3),
            [graph.style.symbol(graph.style.symbol.circle,
                symbolattrs=[color.cmyk.black,deco.filled],
                size=0.01)])
        ##
        c.insert(g2)
        ##
        c.writePDFfile(outPrefix+'.aucAndCutoff.%s'%(ii+1))
    ##
    ##jointProbabilitiesPerRead is of the format:$
    ##{bc:readID:[(x,prob),...]}
    ##Will now go through and classify reads in each library
    print('Classifying reads...')
    ##set the barcode order
    bcs=['bc3','bc4','bc5','bc6']
    colors=[color.cmyk(0,0,0,1),
        color.cmyk(1,0.5,0,0),
        color.cmyk(0.1,0.05,0.9,0),
        color.cmyk(0.97,0,0.75,0)]
    ##
    classifications=getClassifications(jointProbabilitiesPerRead,\
        bb,bcs)
    ##classifications is a list of lists
    ##[[freq00,freq01,freq10,freq11],...]
    c=canvas.canvas()
    boxWidth=4
    for ii in range(len(bcs)):
        c.text(-0.5,-ii,bcs[ii],[text.halign.boxright,text.valign.middle])
        ##
        groups=classifications[ii]
        #print(bcs[ii],groups)
        runTotal=0.
        for jj in range(len(groups)):
            val=groups[jj]
            c.fill(path.rect(runTotal*boxWidth,-ii-0.4,\
                val*boxWidth,0.8),\
                [colors[jj]])
            runTotal+=val
    ##
    c.writePDFfile(outPrefix+'.classifications')


def getMetaReadPerBarcode(jointProbabilitiesPerRead):
    """
    jointProbabilitiesPerRead=
    {bc:readID:[(x,prob),...]}
    For all readIDs w/in one bc, will get summary metrics of the probs for each
    x, returning a dict of the format:
    {bc:[(x,quartile25,median,quartile75),...]}
    """
    print('#####\nMaking metaRead of joint probabilities...\n#####')
    ##
    aa=collections.defaultdict(lambda:collections.defaultdict(list))
    for bc,readDict in jointProbabilitiesPerRead.items():
        for readID,valList in readDict.items():
            for entry in valList:
                aa[bc][entry[0]].append(entry[1])
    ##
    ##aa is of the format:
    ##{bc:position:[value list of probs]}
    bb={}
    for bc,positionDict in aa.items():
        ##initialize the dict
        bb[bc]=[]
        ##
        tempList=[entry for entry in positionDict.keys()]
        tempList.sort()
        ##
        for position in tempList:
            freqVals=positionDict[position]
            bb[bc].append((position,\
                -math.log(numpy.quantile(freqVals,0.05),10),\
                -math.log(numpy.quantile(freqVals,0.5),10),\
                -math.log(numpy.quantile(freqVals,0.95),10)))
    ##
    print('Did top/bottom deciles rather than quartiles...')
    return bb

def mkMetaPlot(metaReadJointProb,oligoLocations,outPrefix):
    """
    oligoLocations=[(x1,x2),...(xn,xn+1)]
    metaReadJointProb=
    {bc:[(x,prob25,prob50,prob75)...]
    """
    print('Plotting meta read plot for each barcode...')
    print('Drawing oligo locations: ',oligoLocations)
    ##
    colors=[color.cmyk(0,0,0,1),
        color.cmyk(1,0.5,0,0),
        color.cmyk(0.1,0.05,0.9,0),
        color.cmyk(0.97,0,0.75,0)]
    bcs=['bc3','bc4','bc5','bc6']
    bcs.reverse()
    colors.reverse()
    ##
    c=canvas.canvas()
    ##
    for ii in range(len(bcs)):
        bc=bcs[ii]
        ##
        g=graph.graphxy(width=8,height=2,ypos=ii*3.5,
            x=graph.axis.linear(min=0,max=700,
                title='Position Along Txt (nt)'),
            y=graph.axis.linear(min=0,max=15,title='-log10 Prob'))
        ##
        for oligo in oligoLocations:
            leftBound=oligo[0]
            rightBound=oligo[1]
            c.fill(path.rect(g.width*leftBound/700,g.ypos,
                g.width*(rightBound-leftBound)/700,2),
                [deco.filled([color.cmyk(0,0,0,0.2),
                    color.transparency(0.2)])])
        ##
        ##plot the median
        g.plot(graph.data.points(metaReadJointProb[bc],x=1,y=3,
            title=bc),[graph.style.line([style.linestyle.solid,
                colors[ii]])])
        ##plot the quartiles
        g.plot(graph.data.points(metaReadJointProb[bc],x=1,y=2,
            title=bc),[graph.style.line([style.linestyle.dotted,
                colors[ii]])])
        g.plot(graph.data.points(metaReadJointProb[bc],x=1,y=4,
            title=bc),[graph.style.line([style.linestyle.dotted,
                colors[ii]])])
        ##
        c.insert(g)
    ##
    c.writePDFfile(outPrefix)

def getEditData(inDict,edit):
    """
    inDict={position:edit/not}
    Will return a list of graph.data.function objections w/ all positions
    matching edit.
    """
    aa=[]
    for k,v in inDict.items():
        if v==edit:
            aa.append(k)
    bb=[]
    for entry in aa:
        bb.append(graph.data.function("x(y)=%s"%(entry),title=None))
    return bb

def mkIndividualReadPlots(jointProbabilitiesPerRead,oligoLocations,\
        numberOfReadsToPlot,outPrefix,editFreqPerRead):
    """
    jointProbabilitiesPerRead is of the format
    {bc:readID:[(x,prob),...]
    oligoLocations=[[x1,x2],...,[xn,xn+1]]
    numberOfReadsToPlot is the number of reads that this function will plot
    for each barcode.
    editFreqPerRead is of the format
    {bc:readID:{position:edit/not}}
    Will plot numberOfReadsToPlot reads for each barcode from
    jointProbabilitiesPerRead. Will also draw tick marks for edits/not
    from editFreqPerRead
    """
    print('#####\nPlotting %s individual reads for each barcode...'%(\
        numberOfReadsToPlot))
    ##
    colors=[color.cmyk(0,0,0,1),
        color.cmyk(1,0.5,0,0),
        color.cmyk(0.1,0.05,0.9,0),
        color.cmyk(0.97,0,0.75,0)]
    bcs=['bc3','bc4','bc5','bc6']
    ##
    c=canvas.canvas()
    for ii in range(len(bcs)):
        ##
        bc=bcs[ii]
        ##
        readIDs=[readID for readID in jointProbabilitiesPerRead[bc].keys()]
        ##
        for jj in range(numberOfReadsToPlot):
            ##
            readID=readIDs[jj]
            ##
            g=graph.graphxy(width=8,height=1,xpos=ii*10,ypos=jj*3,
                x=graph.axis.linear(min=0,max=700,
                    title=readID),
                y=graph.axis.log(min=10**-15,title='log Prob'))
            ##
            for oligo in oligoLocations:
                g.plot([graph.data.function("x(y)=%s"%(oligo[0]),title=None),
                    graph.data.function("x(y)=%s"%(oligo[1]),title=None)],
                        [graph.style.line([style.linestyle.dotted])])
            ##
            g.plot(graph.data.points(jointProbabilitiesPerRead[bc][readID],\
                x=1,y=2),
                [graph.style.line([style.linestyle.solid,
                    colors[ii]])])
            ##now attempt to draw ticks for every edit and non-edit
            g2=graph.graphxy(width=8,height=0.25,xpos=ii*10,ypos=jj*3+g.height,
                x=graph.axis.linkedaxis(g.axes["x"]),
                y=graph.axis.linear(min=0,max=1))
            ##
            g2.plot(getEditData(editFreqPerRead[bc][readID],1),
                [graph.style.line([style.linestyle.solid,color.cmyk.black])])
            g2.plot(getEditData(editFreqPerRead[bc][readID],0),
                [graph.style.line([style.linestyle.solid,color.cmyk(0,0.8,1,0)])])
            ##
            c.insert(g)
            c.insert(g2)
    ##
    c.writePDFfile(outPrefix)


def main(args):
    ##
    pickleFile,referenceBarcode,ntWindow,numberOfReadsToPlot,\
        oligoFile,outPrefix=args[0:]
    ##
    oligoLocations=getOligoLocations(oligoFile)
    ##extract the edit information from the pickle file
    editFreqPerRead=parsePickleFile(pickleFile)
    ##get the reference frequencies of the reference barcode
    referenceFreq=getReferenceFreq(editFreqPerRead[referenceBarcode])
    ##referenceFreq is of the format {position:(freq0,freq1),...}
    ##
    ##Will use those reference frequencies to compute the joint probability
    ##of observing a given sequence of edits.
    jointProbabilitiesPerRead=computeJointProbabilitiesPerRead(editFreqPerRead,\
        referenceFreq,int(ntWindow))
    ##
    ##data should now be in jointProbabilitiesPerRead in the format:
    ##{bc:readID:[(x,prob),...]} where x is the x-coordinate of the window centered
    ##at that position, and prob is the joint probability of all edits in that
    ##window.
    ##
    ##One next thing I'll do is evaluate the behavior of Joint Probability as a
    ##classifier for "shadowed v not". To do this, I'll loop through every position
    ##within each oligo-annealing site. And then array out all p-values, and loop
    ##through them (maybe subseting to make the plot less painful).
    mkROCCurves(jointProbabilitiesPerRead,oligoLocations,outPrefix+'.ROCs')
    ##
    metaReadJointProb=getMetaReadPerBarcode(jointProbabilitiesPerRead)
    ##
    mkMetaPlot(metaReadJointProb,oligoLocations,outPrefix+'.meta')
    ##
    mkIndividualReadPlots(jointProbabilitiesPerRead,oligoLocations,\
        int(numberOfReadsToPlot),outPrefix+'.reads',editFreqPerRead)

if __name__=='__main__':
    Tee()
    main(sys.argv[1:])
