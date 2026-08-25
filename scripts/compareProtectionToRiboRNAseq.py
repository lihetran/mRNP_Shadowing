"""
Joshua Arribere, May 1, 2026

Script to analyze RNA counts and protection in a PS library and compare them
    to Ribo-seq and RNA-seq values.

Input: RiboseqRPKM.txt - of the format
        gene\trpkm
    RNAseqRPKM.txt - same, for RNA-seq
    protection.txt - text file as output from calculateProtectionAcrossParquets.py:
        txtID\treadID\tlibrary\tUTR5editCt\tUTR5totCt\tCDSeditCt\tCDStotCt\t
        UTR3editCt\tUTR3totCt\t[UTR5weightedEdit\tUTR5weightedTot\t...]
        (library, added by calculateProtectionAcrossParquets.py's
        inFileParquet.txt support, splits the TE-vs-protection panels
        below into one figure per library, colored with that library's
        manuscript color; the optional trailing weightedEdit/weightedTot
        columns, from that script's computeMotifFreqs, feed the weighted
        panels)
    N - length filter for protection regions. If a region has fewer
        than this number of editable sites, that read will be ignored.
    readCutoff - minimum number of reads in PS data
    color_map.txt (optional) - manuscript color TSV 'name rep path hex_color'
        (no leading '#'). Labels are looked up as 'name-rep'/'name_rep' or
        bare 'name'; unmatched libraries cycle through a small default
        palette.

Output: Will produce scatter plots of:
    RNA-seq v PS read count
    Ribo-seq v PS read count
    Ribo-seq/RNA-seq v average protection, 3 panels (5'UTR/CDS/3'UTR) --
        a SEPARATE figure per library (colored with that library's
        manuscript color), each with a motif-bias-weighted version and a
        raw (unweighted) version (see mkTEProtectionPanels)
Per-gene single-histogram plots (mkSingleGeneHists) are currently disabled
    (not called from main) but kept defined for future use.

run as python3 compareProtectionToRiboRNAseq.py RiboseqRPKM.txt
    RNAseqRPKM.txt protection.txt N readCutoff outPrefix [color_map.txt]
"""
import sys, common, collections, scipy.stats, numpy
import diptest
import matplotlib.pyplot as plt
from logJosh import Tee
from pyx import *


def load_color_map(path: str) -> dict:
    """
    Parse a manuscript color-map TSV with columns:
        sample_name, rep, path, hex_color (no leading '#')
    Returns a dict keyed by "name_rep", "name-rep" (this pipeline's
    libraryID convention, see calculateProtectionAcrossParquets.py's
    parse_parquet_libs_file), and bare "name" (first match wins for the
    bare key) mapping to "#RRGGBB".
    """
    color_map = {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 4:
                continue
            name, rep, _path, hexcol = fields[0], fields[1], fields[2], fields[3]
            hexcol = "#" + hexcol.strip().lstrip("#")
            if rep:
                color_map.setdefault(f"{name}_{rep}", hexcol)
                color_map.setdefault(f"{name}-{rep}", hexcol)
            color_map.setdefault(name, hexcol)
    return color_map

def parseRPKM(inFile):
    aa={}
    with open(inFile,'r') as f:
        for line in f:
            line=line.strip().split('\t')
            aa[line[0]]=float(line[1])
    return aa

def _addPSRead(d,txtID,UTR5,CDS,UTR3,N,UTR5w=None,CDSw=None,UTR3w=None):
    """
    Add one read's UTR5/CDS/UTR3 [editCt,totCt] pairs into d (a psDict, see
    parsePS), appending an edit frequency to a region's list only if that
    region has at least N editable bases.

    UTR5w/CDSw/UTR3w, if given, are the matching [weightedEditSum,
    weightedTot] pairs from calculateProtectionAcrossParquets.py's
    motif-bias-weighted columns (see its computeMotifFreqs). A weighted
    ratio is appended to '<region>_w' under the SAME N gate as the raw
    ratio (so the two lists cover the same reads and stay directly
    comparable), plus a weightedTot>0 check since some reads have zero
    weighted-eligible positions (all their edits landed on a motif never
    observed edited in this library, so 1/motifFreq was undefined).
    """
    if txtID not in d:
        ##{txtID:{'ct':total_readID,'UTR5':[list],'CDS':[list],'UTR3':[list]}
        d[txtID]={'ct':0,'UTR5':[],'CDS':[],'UTR3':[],
                  'UTR5_w':[],'CDS_w':[],'UTR3_w':[]}
    ##
    d[txtID]['ct']+=1
    ##
    if UTR5[1]>=N:
        d[txtID]['UTR5'].append(UTR5[0]/UTR5[1])
        if UTR5w is not None and UTR5w[1]>0:
            d[txtID]['UTR5_w'].append(UTR5w[0]/UTR5w[1])
    if CDS[1]>=N:
        d[txtID]['CDS'].append(CDS[0]/CDS[1])
        if CDSw is not None and CDSw[1]>0:
            d[txtID]['CDS_w'].append(CDSw[0]/CDSw[1])
    if UTR3[1]>=N:
        d[txtID]['UTR3'].append(UTR3[0]/UTR3[1])
        if UTR3w is not None and UTR3w[1]>0:
            d[txtID]['UTR3_w'].append(UTR3w[0]/UTR3w[1])

def parsePS(protectionFile,N):
    """
    Will parse a file of the format:
    txtID\treadID\tlibrary\tUTR5editCt\tUTR5totCt\tCDSeditCt\tCDStotCt\t
        UTR3editCt\tUTR3totCt\t[UTR5weightedEdit\tUTR5weightedTot\t
        CDSweightedEdit\tCDSweightedTot\tUTR3weightedEdit\tUTR3weightedTot]
    (the weighted columns are optional -- older protection.txt files
    without them still parse fine, just with empty '_w' lists throughout).

    Returns (psDict,psDictByLib):
    psDict pools every library together, of the format
    {txtID:{'ct':total_readID,'UTR5':[list],'CDS':[list],'UTR3':[list],
             'UTR5_w':[list],'CDS_w':[list],'UTR3_w':[list]}
    for each list, that will be a list of all the (motif-bias-weighted, for
    the '_w' keys) edit frequencies for all reads with at least N editable
    bases in that region -- used for each panel's overall Spearman R.
    psDictByLib is the same, but split out per library:
    {library:{txtID:{...same inner structure...}}} -- used to color
    individual scatter points by library (see mkTEProtectionPanels).
    Will also strip '_' from txtIDs
    """
    aa={}
    byLib={}
    with open(protectionFile,'r') as f:
        header=f.readline()##skip the first line
        hasWeighted='weightedEdit' in header
        ##
        for line in f:
            ##
            line=line.strip().split('\t')
            ##
            txtID=line[0].split('_')[0]
            readID=line[1]##won't actually do anything with this
            library=line[2]
            UTR5=[int(line[3]),int(line[4])]
            CDS=[int(line[5]),int(line[6])]
            UTR3=[int(line[7]),int(line[8])]
            ##
            if hasWeighted and len(line)>=15:
                UTR5w=[float(line[9]),int(line[10])]
                CDSw=[float(line[11]),int(line[12])]
                UTR3w=[float(line[13]),int(line[14])]
            else:
                UTR5w=CDSw=UTR3w=None
            ##
            _addPSRead(aa,txtID,UTR5,CDS,UTR3,N,UTR5w,CDSw,UTR3w)
            _addPSRead(byLib.setdefault(library,{}),txtID,UTR5,CDS,UTR3,N,UTR5w,CDSw,UTR3w)
    ##
    ##print(aa)
    print('%s transcripts were found in the protection file.'%(len(aa)))
    print('%s libraries were found in the protection file: %s'%(len(byLib),sorted(byLib.keys())))
    if not hasWeighted:
        print('  No motif-bias-weighted columns found in %s; weighted comparisons will be empty.'%(protectionFile))
    ##
    return aa,byLib

def mkPlot(xDict,xTitle,yDict,yTitle,outPrefix):
    """
    Will plot a scatter plot of xDict,yDict, log-scaled.
    """
    xVals,yVals=[],[]
    for k,xVal in xDict.items():
        if k in yDict:
            yVal=yDict[k]
            if xVal>0 and yVal>0:
                xVals.append(xVal)
                yVals.append(yVal)
    print('\nPlotting %s points on "%s" vs. "%s".'%(len(xVals),xTitle,yTitle))
    #
    g=graph.graphxy(width=4,height=4,
                    x=graph.axis.log(title=xTitle),
                    y=graph.axis.log(title=yTitle))
    ##
    g.plot(graph.data.points([entry for entry in zip(xVals,yVals)],x=1,y=2),
           [graph.style.symbol(graph.style.symbol.circle,
                               symbolattrs=[color.cmyk.black,deco.filled],
                               size=0.05)])
    ##
    R,pval=scipy.stats.spearmanr(xVals,yVals)
    print(f"SpearmanR: {R:.3f}\nPval: {pval:.4g}")
    g.text(g.width/2.,g.height+0.1,f"SpearmanR: {R:.4g}\nPval: {pval:.4g}",
           [text.valign.bottom,text.halign.boxcenter])
    ##
    g.writeSVGfile(outPrefix)

def mkPlotXLogYLinear(xDict,xTitle,yDict,yTitle,outPrefix,noDomain=False):
    """
    Will plot a scatter plot of xDict,yDict, log-scaled.
    """
    xVals,yVals=[],[]
    for k,xVal in xDict.items():
        if k in yDict:
            yVal=yDict[k]
            if xVal>0 and yVal>0:
                xVals.append(xVal)
                yVals.append(yVal)
    print('\nPlotting %s points on "%s" vs. "%s".'%(len(xVals),xTitle,yTitle))
    #
    if not noDomain:
        g=graph.graphxy(width=4,height=4,
                    x=graph.axis.log(min=0.1,max=5,title=xTitle),
                    y=graph.axis.linear(title=yTitle))
    else:
        g=graph.graphxy(width=4,height=4,
                    x=graph.axis.log(title=xTitle),
                    y=graph.axis.linear(title=yTitle))
    ##
    g.plot(graph.data.points([entry for entry in zip(xVals,yVals)],x=1,y=2),
           [graph.style.symbol(graph.style.symbol.circle,
                               symbolattrs=[color.cmyk.black,deco.filled],
                               size=0.05)])
    ##
    R,pval=scipy.stats.spearmanr(xVals,yVals)
    print(f"SpearmanR: {R:.3f}\nPval: {pval:.4g}")
    g.text(g.width/2.,g.height+0.1,f"SpearmanR: {R:.4g}\nPval: {pval:.4g}",
           [text.valign.bottom,text.halign.boxcenter])
    ##
    g.writeSVGfile(outPrefix)

DEFAULT_SCATTER_PALETTE=[
    color.cmyk(0,0,0,1),      # black
    color.cmyk(1,0.5,0,0),    # blue
    color.cmyk(0,1,1,0),      # red
    color.cmyk(1,0,1,0),      # green
    color.cmyk(0,0.6,1,0),    # orange
    color.cmyk(1,1,0,0),      # purple
]

def _hex_to_pyx_color(hexcol):
    hexcol=hexcol.lstrip('#')
    r,g,b=int(hexcol[0:2],16)/255.,int(hexcol[2:4],16)/255.,int(hexcol[4:6],16)/255.
    return color.rgb(r,g,b)

def _resolve_scatter_color(label,idx,color_map):
    """Look up label's manuscript color; fall back to DEFAULT_SCATTER_PALETTE[idx]."""
    hexcol=color_map.get(label) if color_map else None
    if hexcol:
        return _hex_to_pyx_color(hexcol)
    return DEFAULT_SCATTER_PALETTE[idx%len(DEFAULT_SCATTER_PALETTE)]

def mkTEProtectionPanels(teDict,regionPanels,outPrefix,variantLabel='',libraryLabel=None):
    """
    One figure, 3 side-by-side panels, for ONE sample: TE (Ribo-seq/
    RNA-seq) on a log x-axis vs. average per-transcript protection (edit
    frequency) on a linear y-axis, for 5'UTR, CDS, and 3'UTR respectively.

    regionPanels is a list of (yTitle,repSeries) tuples, one per region.
    repSeries is a list of (repLabel,repColor,yDict) triples -- normally
    every replicate of the SAME sample (e.g. [('rep1',col1,d1),
    ('rep2',col2,d2)]) -- each already resolved to its OWN manuscript
    color by the caller (color_map.txt defines a color per sample AND per
    rep, see resolve_color/_resolve_scatter_color), so this function just
    draws whatever color it's given rather than deriving one. Each rep's
    own Spearman R/pval is computed and annotated SEPARATELY, in that same
    rep's color (not pooled across reps, since pooling would hide a
    rep-to-rep discrepancy that's often exactly what you want to see).
    libraryLabel, if given, is drawn as a title above the figure (typically
    the sample name, shared by all its reps).

    variantLabel (e.g. 'weighted'/'unweighted'), if given, is appended to
    each panel's y-axis title and to outPrefix, so the motif-bias-weighted
    and raw versions of this figure (see calculateProtectionAcrossParquets.py's
    computeMotifFreqs) land in clearly-named, separate files.

    panelW/gap are both in cm: gap must clear each panel's own y-axis
    title+tick-label width (~1.5-2cm), or the next panel's y-axis
    overlaps this panel's plot area -- verified empirically via
    graph.graphxy.bbox(), gap=1 overlaps, gap>=2 does not.

    Spearman R annotations are drawn via c.text(g.xpos+..., g.ypos+...)
    rather than g.text(...): graphxy.text() inserts a textbox at (x,y) in
    ITS OWN local frame WITHOUT applying the graph's xpos/ypos offset
    (only .plot()/axis rendering go through that transform), so calling
    g.text() directly on a panel placed at xpos>0 draws the text back at
    the canvas origin instead of above that panel -- confirmed via
    pdftotext -bbox-layout, where two panels' annotations both landed at
    x~0 instead of at their own panel's position.
    """
    titleSuffix=' (%s)'%(variantLabel) if variantLabel else ''
    panelW,panelH,gap=4,4,2.5
    c=canvas.canvas()
    for i,(yTitleBase,repSeries) in enumerate(regionPanels):
        yTitle=yTitleBase+titleSuffix
        g=graph.graphxy(width=panelW,height=panelH,
                        xpos=i*(panelW+gap),ypos=0,
                        x=graph.axis.log(min=0.1,max=5,title='TE (Ribo-seq/RNA-seq)'),
                        y=graph.axis.linear(title=yTitle))
        statLines=[]##(repLabel,R,pval,repColor) tuples, deferred so each renders in its own color
        for repLabel,repColor,yDict in repSeries:
            xVals,yVals=[],[]
            for k,xVal in teDict.items():
                if k in yDict:
                    yVal=yDict[k]
                    if xVal>0 and yVal>0:
                        xVals.append(xVal)
                        yVals.append(yVal)
            print('\nPlotting %s points on "TE (Ribo-seq/RNA-seq)" vs. "%s" (%s).'
                  %(len(xVals),yTitle,repLabel))
            g.plot(graph.data.points([entry for entry in zip(xVals,yVals)],x=1,y=2),
                   [graph.style.symbol(graph.style.symbol.circle,
                                       symbolattrs=[repColor,deco.filled],size=0.05)])
            if len(xVals)>=2:
                R,pval=scipy.stats.spearmanr(xVals,yVals)
                print(f"SpearmanR: {R:.3f}\nPval: {pval:.4g}")
                statLines.append((repLabel,R,pval,repColor))
        c.insert(g)
        for si,(repLabel,R,pval,repColor) in enumerate(statLines):
            c.text(g.xpos+g.width/2.,g.ypos+g.height+0.1+si*0.4,
                   f"{repLabel}: R={R:.4g}, p={pval:.4g}",
                   [repColor,text.valign.bottom,text.halign.boxcenter])
    if libraryLabel:
        c.text(0,panelH+1.2,libraryLabel+titleSuffix,[text.halign.left,text.size.Large])
    outFile=outPrefix+('_%s'%(variantLabel) if variantLabel else '')
    c.writeSVGfile(outFile)
    print('TE vs. protection panels (5\'UTR/CDS/3\'UTR)%s for %s saved to %s.svg'
          %(titleSuffix,libraryLabel or '',outFile))

def makeCtDict(psDict):
    aa=dict((k,v['ct']) for k,v in psDict.items())
    total=sum(aa.values())/1000000
    bb=dict((k,v/total) for k,v in aa.items())
    return bb

def getTE(riboSeqDict,rnaSeqDict):
    aa={}
    for k,v in riboSeqDict.items():
        if k in rnaSeqDict:
            rna=rnaSeqDict[k]
            if rna>0:
                aa[k]=v/rna
    return aa

def getProtection(psDict,string,readCutoff):
    """
    psDict is of the format:
    {txtID:{'ct':total_readID,'UTR5':[freqList],'CDS':[freqList],'UTR3':[freqList]}
    string is one of 'UTR5','CDS','UTR3'
    Will return dict of {txtID:avgFreq} if len(freqList)>=readCutoff
    """
    aa={}
    for txtID,v in psDict.items():
        if len(v[string])>=readCutoff:
            aa[txtID]=numpy.average(v[string])
    return aa

def getDispersion(psDict,readCutoff):
    """
    psDict is of the format:
    {txtID:{'ct':total_readID,'UTR5':[freqList],'CDS':[freqList],'UTR3':[freqList]}
    For each txtID, will CDS, will calculate the stDev / mean for freqList
    """
    aa={}
    for txtID,v in psDict.items():
        freqList=v['CDS']
        if len(freqList)>=readCutoff:
            stdev=numpy.std(freqList)
            mean=numpy.mean(freqList)
            aa[txtID]=stdev
    return aa

def mkHist(theVals,xTitle,outPrefix,nBins=50,header='Histogram'):
    """
    theVals is a list of values
    """
    ##
    plt.hist(theVals,bins=nBins,edgecolor='black')
    ##
    plt.xlabel(xTitle)
    plt.ylabel('Frequency')
    plt.title(header)
    plt.savefig(outPrefix+'.svg')
    ##
    plt.clf()

def mkSingleGeneHists(psDict,xTitle,outPrefix):
    """
    psDict is of the format:
    {txtID:{'ct':total_readID,'UTR5':[freqList],'CDS':[freqList],'UTR3':[freqList]}
    Will loop through psDict and plot genes w/ a large number of reads
    """
    cntr=0
    print('\nPerforming Hartigan Dip Test...\n')
    for txtID,v in psDict.items():
        theVals=v['CDS']
        ##
        if len(theVals)<=10:
            continue
        ##
        dip,pval=diptest.diptest(numpy.array(theVals))
        ##
        #if len(theVals)>=200:
        if pval<=1e-3:
            theAvg=numpy.average(theVals)
            ##
            print('Plotting histogram of %s'%(txtID))
            print('Pval: %s'%(pval))
            #mkHist(theVals,xTitle,
            #       outPrefix+'_%s_readCt%s_Avg%s'%(txtID,len(theVals),f"{theAvg:.3f}"),
            #       nBins=50,header='%s (%s, Avg: %s)'%(txtID,len(theVals),f"{theAvg:.3f}"))
            mkHist(theVals,xTitle,
                   outPrefix+'_%s_readCt%s_Avg%s'%(txtID,len(theVals),f"{theAvg:.3f}"),
                   nBins=50,header='%s (%s, Avg: %s, Pval: %s)'%(txtID,len(theVals),f"{theAvg:.3f}",f"{pval:.4f}"))
            cntr+=1
    print('Plotted %s gene-specific histograms.'%(cntr))


def main(args):
    ##
    riboSeqFile=args[0]
    rnaSeqFile=args[1]
    protectionFile=args[2]
    N=args[3]
    readCutoff=args[4]
    outPrefix=args[5]
    colorMapPath=args[6] if len(args)>6 else None
    ##
    color_map=load_color_map(colorMapPath) if colorMapPath else {}
    ##
    riboSeqDict=parseRPKM(riboSeqFile)
    print('%s transcripts were found in the Ribo-seq file.'%(len(riboSeqDict)))
    rnaSeqDict=parseRPKM(rnaSeqFile)
    print('%s transcripts were found in the RNA-seq file.'%(len(rnaSeqDict)))
    ##
    N=int(N)
    print('Only considering PS regions with at least %s editable sites.'%(N))
    readCutoff=int(readCutoff)
    print('Only considering gene with at least %s PS reads.'%(readCutoff))
    ##
    psDict,psDictByLib=parsePS(protectionFile,N)
    ##psDict is of the format:
    ##{txtID:{'ct':total_readID,'UTR5':[freqList],'CDS':[freqList],'UTR3':[freqList],
    ##         'UTR5_w':[freqList],'CDS_w':[freqList],'UTR3_w':[freqList]}
    ##psDictByLib is the same, split out per library -- see mkTEProtectionPanels
    if colorMapPath:
        unmatched=[lib for lib in psDictByLib if lib not in color_map]
        if unmatched:
            print('  WARNING: no color found in %s for librar%s %s; '
                  'falling back to the default palette.'
                  %(colorMapPath,'y' if len(unmatched)==1 else 'ies',unmatched))
    psCtDict=makeCtDict(psDict)
    ##
    outPrefix+='_EditPos%s_ReadCut%s'%(N,readCutoff)
    ##
    ##now plot RNA-seq v PS read count
    mkPlot(rnaSeqDict,'RNA-seq Count (RPKM)',psCtDict,'PS Read Count (RPM)',
           outPrefix+'_RNA_PSCt')
    ##now plot Ribo-seq v PS read count
    mkPlot(riboSeqDict,'Ribo-seq Count (RPKM)',psCtDict,'PS Read Count (RPM)',
           outPrefix+'_Ribo_PSCt')
    ##now plot Ribo-seq/RNA-seq v average protection, 3 panels (5'UTR/CDS/3'UTR) per figure --
    ##a SEPARATE figure per SAMPLE, with every replicate of that sample overlaid on the same
    ##panels, each rep resolved to its OWN manuscript color (color_map.txt defines a color per
    ##sample and per rep, see resolve_color/_resolve_scatter_color) and its own Spearman R/pval
    ##reported separately (not pooled). Each sample gets a motif-bias-weighted version and a
    ##raw (unweighted) one.
    teDict=getTE(riboSeqDict,rnaSeqDict)
    cds=getProtection(psDict,'CDS',readCutoff)
    ##group library labels (e.g. '-3AT-rep1','-3AT-rep2') by sample name, stripping the
    ##trailing '-repN' (this pipeline's libraryID convention is always 'fileName-rep', see
    ##calculateProtectionAcrossParquets.py's parse_parquet_libs_file, so the LAST '-' is
    ##always the rep separator, even when the sample name itself starts with '-' or '+').
    samplesToLibs=collections.defaultdict(list)##sampleName -> [(repToken,label),...]
    for label in psDictByLib:
        sampleName,repToken=label.rsplit('-',1) if '-' in label else (label,'')
        samplesToLibs[sampleName].append((repToken,label))
    ##stable per-label index, so labels missing from color_map still get distinct (not
    ##colliding) fallback colors from DEFAULT_SCATTER_PALETTE
    labelIdx={label:i for i,label in enumerate(sorted(psDictByLib.keys()))}
    ##
    for sampleName in sorted(samplesToLibs.keys()):
        repsForSample=sorted(samplesToLibs[sampleName])
        regionPanelsWeighted=[]
        regionPanelsUnweighted=[]
        for yTitleBase,region in [("5'UTR Protection",'UTR5'),('CDS Protection','CDS'),
                                  ("3'UTR Protection",'UTR3')]:
            repSeriesW=[(repToken,_resolve_scatter_color(label,labelIdx[label],color_map),
                         getProtection(psDictByLib[label],region+'_w',readCutoff))
                        for repToken,label in repsForSample]
            repSeriesUnw=[(repToken,_resolve_scatter_color(label,labelIdx[label],color_map),
                           getProtection(psDictByLib[label],region,readCutoff))
                          for repToken,label in repsForSample]
            regionPanelsWeighted.append((yTitleBase,repSeriesW))
            regionPanelsUnweighted.append((yTitleBase,repSeriesUnw))
        safeLabel=sampleName.replace('/','_').replace(' ','_')
        sampleOutPrefix=outPrefix+'_'+safeLabel+'_TE_protectionPanels'
        mkTEProtectionPanels(teDict,regionPanelsWeighted,sampleOutPrefix,
                             variantLabel='weighted',libraryLabel=sampleName)
        mkTEProtectionPanels(teDict,regionPanelsUnweighted,sampleOutPrefix,
                             variantLabel='unweighted',libraryLabel=sampleName)
    ##plot dispersion of protection v read counts
    dispersionDict=getDispersion(psDict,readCutoff)
    mkPlotXLogYLinear(psCtDict,'PS Read Count (RPM)',dispersionDict,'Protection Variability Across Molecules',
                      outPrefix+'_PSCt_ProtectDispersion',noDomain=True)
    ##plot a histogram of the CDS protection values
    print('Plotting histogram of CDS Protection values.')
    mkHist([entry for entry in cds.values()],'PS CDS Protection',outPrefix+'_histCDSprot')
    ##plot histograms of individual protection values -- disabled for now (kept defined,
    ##just not called).
    #mkSingleGeneHists(psDict,'PS CDS Protection',outPrefix+'_singleGeneCDSProt')


if __name__=='__main__':
    Tee()
    main(sys.argv[1:])