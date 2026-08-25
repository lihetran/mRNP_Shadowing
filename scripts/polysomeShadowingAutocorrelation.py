"""
Joshua Arribere, July 27, 2026

Script to analyze shadowing data for periodicities via autocorrelation.

Input: inFileParquet.txt - a line-delimited file of format:
        fileName    rep    parquetDir
    (same inFileParquet.txt convention used by calculateProtectionAcrossParquets.py /
    polysomeShadowHMMQC.py). Each row becomes its own library, labeled 'fileName-rep';
    parquetDir may be a directory of *.parquet chunk files (globbed and read as one
    library) or a single parquet file.
    gtfFile
    color_map.txt (optional) - manuscript color TSV 'name rep path hex_color' (no
        leading '#'). Labels are looked up as 'name-rep'/'name_rep' or bare 'name';
        unmatched libraries fall back to common.colors(idx).

Output: Graphs of autocorrelation outputs

run as python3 polysomeShadowingAutocorrelation.py inFileParquet.txt gtfFile outPrefix
    [color_map.txt]
"""
import sys, common, metaStartStop, collections, math, statistics
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import t as tdist
from pyx import *
from logJosh import Tee


def load_color_map(path: str) -> dict:
    """
    Parse a manuscript color-map TSV with columns:
        sample_name, rep, path, hex_color (no leading '#')
    Returns a dict keyed by "name_rep", "name-rep" (this script's own
    libraryID convention, see parse_inFileParquet_libs), and bare "name"
    (first match wins for the bare key) mapping to "#RRGGBB".
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


def hex_to_pyx_color(hexcol: str):
    hexcol = hexcol.lstrip("#")
    r = int(hexcol[0:2], 16) / 255.0
    g = int(hexcol[2:4], 16) / 255.0
    b = int(hexcol[4:6], 16) / 255.0
    return color.rgb(r, g, b)


def resolve_color(color_map, label, idx):
    """Look up label's manuscript color; fall back to common.colors(idx)."""
    hexcol = color_map.get(label) if color_map else None
    if hexcol:
        return hex_to_pyx_color(hexcol)
    return common.colors(idx)


def parse_inFileParquet_libs(path: str) -> list:
    """
    Parse a line-delimited inFileParquet.txt file of format:
        fileName    rep    parquetDir
    (same convention as calculateProtectionAcrossParquets.py's
    parse_parquet_libs_file / polysomeShadowHMMQC.py), and return a list of
    (libraryID, parquetFiles) tuples with libraryID = 'fileName-rep' and
    parquetFiles = the sorted list of every *.parquet file found under
    parquetDir if it's a directory, or [parquetDir] itself if it's already
    a single file.
    """
    libs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            fileName, rep, parquetPath = parts[0], parts[1], parts[2]
            p = Path(parquetPath)
            if p.is_dir():
                parquetFiles = [str(x) for x in sorted(p.glob("*.parquet"))]
            else:
                parquetFiles = [str(p)]
            libs.append((f"{fileName}-{rep}", parquetFiles))
    return libs

def fillGapsForRead(sites):
    """
    sites is a list of (relStart,relStop,absIndx,refNt,edit) tuples for a single read,
    sorted by increasing relStart.

    For every position where refNt!='A', replaces edit with the edit value of the nearest
    (by relStart distance) position where refNt=='A'. Ties (equidistant A's on either side
    with differing edit values) are broken in favor of the upstream (lower relStart) A.

    Returns a new list of (relStart,relStop,absIndx,refNt,degapped_edit) tuples, or None if
    the read contains no refNt=='A' position at all (nothing to borrow an edit value from).
    """
    n=len(sites)
    if not any(s[3]=='A' for s in sites):
        return None##no informative site anywhere in this read
    ##
    leftA=[None]*n##index of the nearest A at or before position i
    lastSeen=None
    for i in range(n):
        if sites[i][3]=='A':
            lastSeen=i
        leftA[i]=lastSeen
    ##
    rightA=[None]*n##index of the nearest A at or after position i
    lastSeen=None
    for i in range(n-1,-1,-1):
        if sites[i][3]=='A':
            lastSeen=i
        rightA[i]=lastSeen
    ##
    degapped=[]
    for i,(relStart,relStop,absIndx,refNt,edit) in enumerate(sites):
        if refNt=='A':
            degapped.append((relStart,relStop,absIndx,refNt,edit))
            continue
        li,ri=leftA[i],rightA[i]
        if li is None:
            chosen=ri
        elif ri is None:
            chosen=li
        else:
            leftDist=relStart-sites[li][0]
            rightDist=sites[ri][0]-relStart
            chosen=li if leftDist<=rightDist else ri##ties favor the upstream (lower relStart) A
        degapped.append((relStart,relStop,absIndx,refNt,sites[chosen][4]))
    return degapped

def parseParquetsFillInGaps(inFiles,gtfDict,maxReadsPerLib=200000):
    """
    inFiles is an inFileParquet.txt file, line-delimited with each line
    containing:
    fileName    rep    parquetDir
    (same convention as calculateProtectionAcrossParquets.py /
    polysomeShadowHMMQC.py -- see parse_inFileParquet_libs). Each row
    becomes its own libNamei, labeled 'fileName-rep'; parquetDir may be a
    directory of *.parquet chunk files (globbed into that libNamei's list
    of libFiles) or a single parquet file.

    gtfDict is of the format:
    {strand:{chr:{absIndx:(txtName,relStart,relStop)]}}}
    
    This function will do the following:
    (1) For each libNamei, loop through the parquet files, and identify individual reads.
        Reads will be the rows of the parquetFiles, which will have columns:
        ['chrom', 'gene_strand', 'is_reverse', 'transcript_id', 'gene_name',
                   'gene_biotype', 'read_id', 'read_start', 'read_end', 'edit_string',
                   'barcode', 'bar_seq', 'read_sequence', 'read_sequence_aligned',
                   'ref_sequence_aligned', 'aligned_pairs', 'absolute_indices',
                   'global_edit_freq', 'n_a_positions']
        Obtain the sequence of nucleotides, their absIndx, and their edit status by zipping
            the iterabiles: 'ref_sequence_aligned','absolute_indices', and 'edit_string'.
            Restrict to positions with a 1/0 in edit_string. Restrict to positions for which
            the corresponding position from gtfDict contains one and exactly one 
            (txtName,relStart,relStop). Create an object for each read_id of the format:
            [(relStart,relStop,absIndx,refNt,edit),...]. This list will be ordered according
            to relStart such that relStart is increasing left>right.
        Create an overall object of the format:
            {libNamei:{read_id:[(relStart,relStop,absIndx,refNt,edit),...]}}
        During this looping, if more than maxReadsPerLib read_ids is reached, then move on to
        the next libNamei. maxReadsPerLib may be None, in which case no limits on read counts
        should be imposed.
    (2) Now fill in the gaps. "edits" of a value of 1 can only happen when refNt==A. "edits" of
        a value of 0 can happen when refNt==A or refNt!=A, but are only informative when
        refNt==A. The script will loop through all refNt!=A, and revise edit according to the
        closest refNt that is an A (in relStart distance). This is equivalent to looking at all
        the places where refNt is an A, and then transferring that label to the nearby non-A
        refNts. If there is a tie (meaning, that a nt is equidistance from two A nts of opposite
        edit values), then take the value of the upstream (lower relStart value) A.
    (3) Return an object of the format:
        {libNamei:{read_id:[(relStart,relStop,absIndx,refNt,degapped_edit),...]}}
        where degapped_edit is the output of (2)
    Note: Can easily subset each parquet file to its first 1000 rows before parsing, for
    faster piloting -- remove that subsetting for a full run.
    """
    ##parse inFiles to get the (libName,libFiles) pairs, one per libNamei
    libFilesList=parse_inFileParquet_libs(inFiles)
    ##
    #print('Subsetting to 1000 reads per library for piloting purposes...')
    editStrings={}
    for libName,libFiles in libFilesList:
        print('\nParsing library %s (%d parquet files)...'%(libName,len(libFiles)))
        readSites={}
        for libFile in libFiles:
            if maxReadsPerLib is not None and len(readSites)>=maxReadsPerLib:
                break##already reached the per-library read cap -- skip remaining files
            print('Parsing %s...'%(libFile))
            df=pd.read_parquet(libFile)
            #df=df[:1000]
            ##
            for row in df.itertuples(index=False):
                if maxReadsPerLib is not None and len(readSites)>=maxReadsPerLib:
                    print('Reached maxReadsPerLib=%d for %s -- moving on.'%(maxReadsPerLib,libName))
                    break##reached the per-library read cap -- move on to the next libNamei
                transcript_id=row.transcript_id
                strand=row.gene_strand
                chrom=row.chrom
                read_id=row.read_id
                refSeq=row.ref_sequence_aligned
                absIndices=row.absolute_indices
                editStr=row.edit_string
                ##
                sites=[]##(relStart,relStop,absIndx,refNt,edit) tuples surviving all filters
                for refNt,absIdx,edit in zip(refSeq,absIndices,editStr):
                    if pd.isna(absIdx):
                        continue##no genomic anchor at this position (e.g., an insertion)
                    edit=int(edit)
                    if edit not in (0,1):
                        continue##filters out anything other than a 0/1 edit call
                    absIdx=int(absIdx)
                    entries=gtfDict.get(strand,{}).get(chrom,{}).get(absIdx,[])
                    if len(entries)!=1:
                        continue##ambiguous (or unannotated) position -- filtered out
                    txtName,relStart,relStop=entries[0]
                    if txtName!=transcript_id:
                        continue##annotated transcript at this position doesn't match the read's
                    sites.append((relStart,relStop,absIdx,refNt,edit))
                ##
                if not sites:
                    continue##this read didn't contribute any usable positions
                sites.sort(key=lambda s:s[0])##order by increasing relStart
                readSites[read_id]=sites
        ##
        ##(2)/(3) fill in the gaps for each read, dropping reads with no 'A' anywhere
        degappedReads={}
        for read_id,sites in readSites.items():
            degapped=fillGapsForRead(sites)
            if degapped is None:
                continue##no refNt=='A' anywhere in this read -- drop it entirely
            degappedReads[read_id]=degapped
        ##
        editStrings[libName]=degappedReads
    ##
    return editStrings

def computeAutocorrelation(editString,minOffset=0,maxOffset=None):
    """
    editString is a list of 1/0 values. For that editString, this function will compute
    the autocorrelation by:
    (1) looping through offsets for a range of [minOffset,len(editString)-20], inclusive
        (further capped at maxOffset, if provided).
    (2) For each offset, create a copy of the editString, and shift its index by offset.
    (3) Run Pearson's correlation between the starting editString and the offset
        editString.
    (4) Return:
        (a) (offsets,pearsonRs,pearsonPvals) -- three parallel numpy arrays, one entry
            per offset
        (b) len(editString)-20
    This was created after reading https://stats.stackexchange.com/questions/533217/interpretation-of-the-autocorrelation-of-a-binary-process

    Performance note: rather than calling scipy.stats.pearsonr separately for each offset
    (which recomputes each window's mean/variance from scratch, and carries substantial
    per-call overhead), this uses prefix sums of editString (and editString**2) so each
    offset's mean/variance are O(1) instead of O(len(editString)), and computes the
    p-value directly via the t-distribution -- the same formula scipy.stats.pearsonr uses
    internally, verified to agree with it to within floating-point precision (~1e-15) and
    exactly on NaN (zero-variance) cases, across thousands of random trials. This is not
    an approximation: results are numerically identical to the original implementation,
    just much faster (roughly 30x, empirically, for CDS-length reads). minOffset/maxOffset
    narrow the offsets that get computed AT ALL (rather than computing everything and
    discarding some afterward), which matters because the remaining per-offset cost (the
    cross-term dot product) still scales with the window length -- for a read much longer
    than maxOffset, this avoids the bulk of the work entirely rather than just the output.
    """
    arr=np.asarray(editString,dtype=np.float64)
    n=len(arr)
    firstOffset=max(0,minOffset)
    lastOffset=n-20
    if maxOffset is not None:
        lastOffset=min(lastOffset,maxOffset)
    numOffsets=lastOffset-firstOffset+1
    if numOffsets<=0:##editString too short (or minOffset>maxOffset) -- nothing to compute
        return (np.empty(0,dtype=np.int64),np.empty(0,dtype=np.float64),
            np.empty(0,dtype=np.float64)),n-20
    ##
    cumsum=np.concatenate(([0.0],np.cumsum(arr)))##cumsum[k] = sum of arr[:k]
    cumsumsq=np.concatenate(([0.0],np.cumsum(arr*arr)))
    ##
    offsets=np.arange(firstOffset,lastOffset+1,dtype=np.int64)
    pearsonRs=np.empty(numOffsets,dtype=np.float64)
    dofs=np.empty(numOffsets,dtype=np.float64)
    for idx,offset in enumerate(offsets):
        length=n-offset
        original=arr[:length]
        shifted=arr[offset:n]
        sumAB=(original*shifted).sum()##the one part that's still O(length) per offset
        meanOriginal=cumsum[length]/length
        meanShifted=(cumsum[n]-cumsum[offset])/length
        cov=sumAB/length-meanOriginal*meanShifted
        varOriginal=cumsumsq[length]/length-meanOriginal*meanOriginal
        varShifted=(cumsumsq[n]-cumsumsq[offset])/length-meanShifted*meanShifted
        denom=math.sqrt(varOriginal*varShifted)
        pearsonRs[idx]=cov/denom if denom>0 else float('nan')
        dofs[idx]=length-2
    ##
    with np.errstate(invalid='ignore',divide='ignore'):##NaN/inf here mirror scipy's own
        ##behavior for zero-variance/perfectly-correlated inputs, so are expected, not bugs
        tStats=pearsonRs*np.sqrt(dofs/(1-pearsonRs*pearsonRs))
        pearsonPvals=2*tdist.sf(np.abs(tStats),dofs)
    return (offsets,pearsonRs,pearsonPvals),n-20

def computeAutoCorrForLibs(editStrings,minOffset=10,maxOffset=1000,filterCDSLength=300):
    """
    editStrings is of the format
    {libNamei:{read_id:[(relStart,relStop,absIndx,refNt,degapped_edit),...]}}
    This function will do the following:
    (1) For each read_id, split its list of tuples into three regions:
        TL, CDS, and UTR regions according to:
        (a) TL: relStart<0
        (b) CDS: relStart>=0 and relStop<=-2
        (c) UTR: relStop>2
        If filterCDSLength is provided (default val 300), then only keep reads with at
        least filterCDSLength length of CDS nucleotides. That means that there would
        need to be at least filterCDSLength entries with a relStart>=0 and a 
        relStop<=-2
    (2) For each of the TL, CDS, and UTR list of tuples, compute the autocorrelation.
        Do this per the instructions in the header of computeAutocorrelation function.
        This function will return two objects:
            (a) (offsets,pearsonRs,pearsonPvals) -- three parallel numpy arrays
            (b) len(editString)-20
        If minOffset is provided, then start the offset at this value. If maxOffset is
        provided, then cap the offset at this value. Both are passed straight through to
        computeAutocorrelation, which only computes offsets in [minOffset,maxOffset] in
        the first place (rather than computing the full range and discarding the rest).
    (3) Return an object of format:
        {libNamei:{read_id:{'TL':[len(editString)-20,offsets,pearsonRs,pearsonPvals],
                            'CDS':...
                            'UTR':...}}}
        where offsets/pearsonRs/pearsonPvals are the numpy arrays from (2)(a). (Storing
        these as numpy arrays rather than a Python list of (offset,pearsonR,pearsonPval)
        tuples cuts the memory footprint of this -- potentially very large, one entry per
        read per offset per region -- structure by roughly 7x.)
    Note: Some of TL/CDS/UTR may be too short to execute the autocorrelation analysis.
    """
    autoCorrResults={}
    ##
    for libName,readDict in editStrings.items():
        libResults={}
        for read_id,sites in readDict.items():
            ##(1) split this read's sites into TL/CDS/UTR editStrings
            regionEdits={'TL':[],'CDS':[],'UTR':[]}
            for relStart,relStop,_,_,edit in sites:
                if relStart<0:
                    regionEdits['TL'].append(edit)
                if relStart>=0 and relStop<=-2:
                    regionEdits['CDS'].append(edit)
                if relStop>2:
                    regionEdits['UTR'].append(edit)
            if len(regionEdits['CDS'])<filterCDSLength:
                continue##not enough CDS nucleotides -- drop this read entirely
            ##(2) compute the autocorrelation for each region
            readResult={}
            for region,editString in regionEdits.items():
                (offsets,pearsonRs,pearsonPvals),lengthMetric=computeAutocorrelation(editString,
                    minOffset=minOffset if minOffset is not None else 0,maxOffset=maxOffset)
                readResult[region]=[lengthMetric,offsets,pearsonRs,pearsonPvals]
            libResults[read_id]=readResult
        autoCorrResults[libName]=libResults
    ##
    return autoCorrResults

def computeMeanSEM(values):
    """
    values is a list of numbers. Returns (mean,sem). sem is 0.0 if fewer than 2 values
    (sample standard deviation, hence SEM, is undefined for n<2).
    """
    n=len(values)
    mean=sum(values)/n
    if n<2:
        return mean,0.0
    return mean,statistics.stdev(values)/math.sqrt(n)

def computeMeanSEMSeries(offsetDict,minPointFilter=1):
    """
    offsetDict is of the format {offset:[(pearsonR,pearsonPval),...]}.

    Returns a list of (offset,mean,sem) triples, one per offset (sorted increasing),
    computed from that offset's pearsonR values. Excludes offset==0 (the log x-axis can't
    show x=0), NaN pearsonR values (computeAutocorrelation returns NaN for zero-variance/
    constant inputs, e.g. an editString of all 0s), and any offset left with fewer than
    minPointFilter non-NaN pearsonR values contributing to its mean/sem.
    """
    series=[]
    for offset in sorted(offsetDict.keys()):
        if offset==0:
            continue
        values=[pearsonR for pearsonR,_ in offsetDict[offset] if not math.isnan(pearsonR)]
        if len(values)<minPointFilter:
            continue
        mean,sem=computeMeanSEM(values)
        series.append((offset,mean,sem))
    return series

def aggregateAutoCorrByOffset(autoCorrResults):
    """
    autoCorrResults is of the format:
    {libNamei:{read_id:{'TL':[len-20,offsets,pearsonRs,pearsonPvals],'CDS':...,'UTR':...}}}
    where offsets/pearsonRs/pearsonPvals are parallel numpy arrays (see computeAutoCorrForLibs).

    Aggregates across all read_ids within each libNamei, separately for each of TL/CDS/UTR.
    Returns {libNamei:{'TL':{offset:[(pearsonR,pearsonPval),...]},'CDS':...,'UTR':...}}
    """
    agg={}
    for libName,readDict in autoCorrResults.items():
        libAgg={'TL':collections.defaultdict(list),'CDS':collections.defaultdict(list),
            'UTR':collections.defaultdict(list)}
        for regionResults in readDict.values():
            for region in ('TL','CDS','UTR'):
                _,offsets,pearsonRs,pearsonPvals=regionResults[region]
                for offset,pearsonR,pearsonPval in zip(offsets,pearsonRs,pearsonPvals):
                    libAgg[region][int(offset)].append((pearsonR,pearsonPval))
        agg[libName]={region:dict(offsetDict) for region,offsetDict in libAgg.items()}
    return agg

def computeSignificantAgg(agg):
    """
    agg is of the format {libNamei:{'TL':{offset:[(pearsonR,pearsonPval),...]},'CDS':...,'UTR':...}}

    Per (libNamei,region), restricts to entries with pearsonPval<=0.01, Bonferroni-corrects
    the surviving pvals by N -- the total number of such entries across ALL offsets in that
    (libNamei,region), not just within a single offset -- then keeps only entries whose
    corrected pval is also <=0.01.

    Returns an object of the same format as agg, containing only the significant entries.
    """
    sigAgg={}
    for libName,regionsDict in agg.items():
        sigAgg[libName]={}
        for region,offsetDict in regionsDict.items():
            ##first pass: tally N, the number of entries with pearsonPval<=0.01 across
            ##every offset in this (libNamei,region)
            prelimByOffset={}
            numPassed=0
            for offset,pairs in offsetDict.items():
                prelim=[(pearsonR,pearsonPval) for pearsonR,pearsonPval in pairs if pearsonPval<=0.01]
                prelimByOffset[offset]=prelim
                numPassed+=len(prelim)
            ##second pass: Bonferroni-correct using that single, shared N
            sigOffsetDict={}
            for offset,prelim in prelimByOffset.items():
                if not prelim:
                    continue
                significant=[(pearsonR,pearsonPval*numPassed) for pearsonR,pearsonPval in prelim
                    if pearsonPval*numPassed<=0.01]
                if significant:
                    sigOffsetDict[offset]=significant
            sigAgg[libName][region]=sigOffsetDict
    return sigAgg

def buildAutoCorrCanvas(agg,libNames,minPointFilter=1,color_map=None):
    """
    Builds a single pyx canvas containing, for each libNamei (columns, in libNames order),
    a column of TL (top)/CDS (middle)/UTR (bottom) plots of pearsonR vs offset -- mean as a
    solid line, +/-1 SEM as a semi-transparent shaded band -- each with a linked, log-scaled
    site-count plot directly above it. Y-axes are linked across libNamei within each region
    row (and, separately, within each site-count row); each site-count plot's x-axis is
    linked to the plot beneath it. Each Pearson R plot also draws a dotted reference line
    at y=0.

    color_map (library label -> hex, no leading '#' needed), if given, colors each
    libNamei's column with its manuscript color (see resolve_color); a library missing
    from color_map falls back to common.colors(idx).

    minPointFilter is passed through to computeMeanSEMSeries: offsets with fewer than
    minPointFilter contributing pearsonR values are excluded from the mean+/-SEM plot (but
    still appear, at their true count, in the site-count plot above it).

    Axis ranges are fixed rather than auto-scaled from each individual plot's own data:
    Pearson R to its natural [-1,1] range, and the offset/site-count ranges to the min/max
    across the whole agg passed in. This is deliberate, not cosmetic -- it avoids pyx's
    "zero axis range" crash on a region/library with no (or only one distinct value of)
    data, e.g. a library with zero significant sites anywhere in a region.

    Building the shaded band requires each box graph's axes to be finalized (via .finish())
    before its data can be converted to physical coordinates (via .pos()) for the fill
    polygon -- but .finish()-ing a graph locks its (possibly cross-column-linked) axes
    against further data, so ALL graphs' data must be registered via .plot() first, in one
    pass, before ANY of them is .finish()-ed/filled/inserted, in a second pass.
    """
    regionsBottomUp=['UTR','CDS','TL']##build order; UTR ends up at the bottom, TL at the top
    boxSize=8##mean+/-SEM plots are square
    countHeight=2##countHeight:boxSize is a 1:4 ratio, per instructions
    subGap=0.3
    rowGap=1.5
    colGap=2
    ##
    ##a fixed, data-independent range for each row's axes avoids pyx's "zero axis range"
    ##crash on graphs/rows with no (or only one distinct value of) data -- e.g. a library
    ##with zero significant sites anywhere in a region
    maxCount=2
    allOffsets=[]
    for libName in libNames:
        for region in ('TL','CDS','UTR'):
            for offset,pairs in agg.get(libName,{}).get(region,{}).items():
                if offset!=0:
                    maxCount=max(maxCount,len(pairs))
                    allOffsets.append(offset)
    minOffset,maxOffset=(min(allOffsets),max(allOffsets)) if allOffsets else (1,2)
    if minOffset==maxOffset:
        maxOffset=minOffset+1
    ##
    c=canvas.canvas()
    firstBoxAxis={}##region -> the first column's y-axis object, for cross-column linking
    firstCountAxis={}##region -> the first column's site-count y-axis object
    records=[]##(gBox,series,plotColor,gCount) tuples, deferred to the second pass
    columnTops=[]##(xpos,topYpos,libName), for the column labels, added at the very end
    ##
    ##first pass: build every graph and register all of its data, without finishing any of them
    for ii,libName in enumerate(libNames):
        xpos=ii*(boxSize+colGap)
        ypos=0
        plotColor=resolve_color(color_map,libName,ii)
        for region in regionsBottomUp:
            offsetDict=agg.get(libName,{}).get(region,{})
            series=computeMeanSEMSeries(offsetDict,minPointFilter)
            countPoints=[(offset,len(offsetDict[offset]))
                for offset in sorted(offsetDict.keys()) if offset!=0]
            ##
            isFirstCol=region not in firstBoxAxis
            ##pearsonR is mathematically bounded to [-1,1], so a fixed range is both safe
            ##(never a zero-range axis, even with no data) and the more natural choice
            yAxisBox=graph.axis.linear(min=-1,max=1,title='%s Pearson R'%region) if isFirstCol \
                else graph.axis.linkedaxis(firstBoxAxis[region])
            gBox=graph.graphxy(width=boxSize,height=boxSize,xpos=xpos,ypos=ypos,
                x=graph.axis.log(min=minOffset,max=maxOffset,title='Offset'),y=yAxisBox)
            if isFirstCol:
                firstBoxAxis[region]=gBox.axes['y']
            ##reference line at y=0
            gBox.plot(graph.data.points([(minOffset,0),(maxOffset,0)],x=1,y=2),
                [graph.style.line([color.grey(0.5),style.linestyle.dotted])])
            if series:
                upperPoints=[(offset,mean+sem) for offset,mean,sem in series]
                lowerPoints=[(offset,mean-sem) for offset,mean,sem in series]
                meanPoints=[(offset,mean) for offset,mean,_ in series]
                ##register the band's extent for axis auto-ranging, without drawing it yet
                gBox.plot(graph.data.points(upperPoints,x=1,y=2),
                    [graph.style.line([color.transparency(1)])])
                gBox.plot(graph.data.points(lowerPoints,x=1,y=2),
                    [graph.style.line([color.transparency(1)])])
                ##the mean line itself, drawn now (can't plot() after .finish())
                gBox.plot(graph.data.points(meanPoints,x=1,y=2),
                    [graph.style.line([plotColor,style.linewidth.Thick])])
            ##
            countYpos=ypos+boxSize+subGap
            isFirstCountCol=region not in firstCountAxis
            yAxisCount=graph.axis.log(min=1,max=maxCount,title='N Sites') if isFirstCountCol \
                else graph.axis.linkedaxis(firstCountAxis[region])
            gCount=graph.graphxy(width=boxSize,height=countHeight,xpos=xpos,ypos=countYpos,
                x=graph.axis.linkedaxis(gBox.axes['x']),y=yAxisCount)
            if isFirstCountCol:
                firstCountAxis[region]=gCount.axes['y']
            if countPoints:
                gCount.plot(graph.data.points(countPoints,x=1,y=2),
                    [graph.style.symbol(graph.style.symbol.circle,
                        symbolattrs=[plotColor,deco.filled],size=0.04)])
            ##
            records.append((gBox,series,plotColor,gCount))
            ypos=countYpos+countHeight+rowGap##advance the cursor to the next region up
        columnTops.append((xpos,ypos,libName))
    ##
    ##second pass: every graph now has all its data (including cross-column-linked axes),
    ##so it's safe to finish each box plot, fill in its shaded band, and insert everything
    for gBox,series,plotColor,gCount in records:
        gBox.finish()
        if series:
            bandPoints=[gBox.pos(offset,mean+sem) for offset,mean,sem in series]+ \
                [gBox.pos(offset,mean-sem) for offset,mean,sem in reversed(series)]
            bandPath=path.path(path.moveto(*bandPoints[0]),
                *[path.lineto(*p) for p in bandPoints[1:]],path.closepath())
            gBox.fill(bandPath,[plotColor,color.transparency(0.5)])
        c.insert(gBox)
        c.insert(gCount)
    ##
    for xpos,topYpos,libName in columnTops:
        c.text(xpos+boxSize/2.,topYpos,libName,[text.halign.boxcenter])
    return c

def buildOverlaidAutoCorrCanvas(agg,libNames,minPointFilter=1,color_map=None):
    """
    Like buildAutoCorrCanvas, but instead of one column per libNamei, every
    library is overlaid on the SAME three region panels (TL/CDS/UTR, left
    to right) -- one Pearson-R-vs-offset mean+/-SEM line/band per library,
    all sharing one set of axes per region, so library-to-library
    differences are directly comparable at a glance instead of requiring
    side-by-side columns. A linked, log-scaled site-count panel (also
    overlaid per library) sits above each region panel, and a color
    legend identifying each library sits to the right of the UTR panel.

    color_map (library label -> hex, no leading '#' needed) resolves each
    library's color the same way as buildAutoCorrCanvas (see
    resolve_color); a library missing from color_map falls back to
    common.colors(idx). minPointFilter is passed through to
    computeMeanSEMSeries exactly as in buildAutoCorrCanvas.

    Axis ranges are fixed (not auto-scaled) for the same "zero axis range"
    crash-avoidance reason as buildAutoCorrCanvas. The two-pass
    plot-then-finish-then-fill discipline for the shaded bands is also the
    same as buildAutoCorrCanvas -- see its docstring for why.

    Panel gap is 2.5 (not colGap=2, as buildAutoCorrCanvas uses for its
    columns): each region panel here has ITS OWN y-axis title/tick labels
    (columns in buildAutoCorrCanvas link/share theirs, so its panels can
    sit closer), and gap<2 was empirically found to let a panel's y-axis
    overlap the previous panel's plot area (see compareProtectionToRiboRNAseq.py's
    mkTEProtectionPanels docstring, verified via graph.graphxy.bbox()).
    """
    regionsOrder=['TL','CDS','UTR']##left to right
    boxSize=8
    countHeight=2
    subGap=0.3
    gap=2.5
    ##
    ##fixed, data-independent axis ranges -- see buildAutoCorrCanvas
    maxCount=2
    allOffsets=[]
    for libName in libNames:
        for region in regionsOrder:
            for offset,pairs in agg.get(libName,{}).get(region,{}).items():
                if offset!=0:
                    maxCount=max(maxCount,len(pairs))
                    allOffsets.append(offset)
    minOffset,maxOffset=(min(allOffsets),max(allOffsets)) if allOffsets else (1,2)
    if minOffset==maxOffset:
        maxOffset=minOffset+1
    ##
    c=canvas.canvas()
    records=[]##(gBox,perLibSeries) tuples, deferred to the second pass
    ##
    ##first pass: build each region's panels and register every library's data on them,
    ##without finishing any of them (see buildAutoCorrCanvas for why finish() must wait)
    for ri,region in enumerate(regionsOrder):
        xpos=ri*(boxSize+gap)
        ypos=0
        gBox=graph.graphxy(width=boxSize,height=boxSize,xpos=xpos,ypos=ypos,
            x=graph.axis.log(min=minOffset,max=maxOffset,title='Offset'),
            y=graph.axis.linear(min=-1,max=1,title='%s Pearson R'%region))
        ##reference line at y=0
        gBox.plot(graph.data.points([(minOffset,0),(maxOffset,0)],x=1,y=2),
            [graph.style.line([color.grey(0.5),style.linestyle.dotted])])
        ##
        countYpos=ypos+boxSize+subGap
        gCount=graph.graphxy(width=boxSize,height=countHeight,xpos=xpos,ypos=countYpos,
            x=graph.axis.linkedaxis(gBox.axes['x']),
            y=graph.axis.log(min=1,max=maxCount,title='N Sites'))
        ##
        perLibSeries=[]
        for li,libName in enumerate(libNames):
            plotColor=resolve_color(color_map,libName,li)
            offsetDict=agg.get(libName,{}).get(region,{})
            series=computeMeanSEMSeries(offsetDict,minPointFilter)
            countPoints=[(offset,len(offsetDict[offset]))
                for offset in sorted(offsetDict.keys()) if offset!=0]
            if series:
                upperPoints=[(offset,mean+sem) for offset,mean,sem in series]
                lowerPoints=[(offset,mean-sem) for offset,mean,sem in series]
                meanPoints=[(offset,mean) for offset,mean,_ in series]
                ##register the band's extent for axis auto-ranging, without drawing it yet
                gBox.plot(graph.data.points(upperPoints,x=1,y=2),
                    [graph.style.line([color.transparency(1)])])
                gBox.plot(graph.data.points(lowerPoints,x=1,y=2),
                    [graph.style.line([color.transparency(1)])])
                ##the mean line itself, drawn now (can't plot() after .finish())
                gBox.plot(graph.data.points(meanPoints,x=1,y=2),
                    [graph.style.line([plotColor,style.linewidth.Thick])])
            if countPoints:
                gCount.plot(graph.data.points(countPoints,x=1,y=2),
                    [graph.style.symbol(graph.style.symbol.circle,
                        symbolattrs=[plotColor,deco.filled],size=0.04)])
            perLibSeries.append((series,plotColor))
        records.append((gBox,perLibSeries,gCount))
    ##
    ##second pass: every panel now has all its data, so it's safe to finish, fill in each
    ##library's shaded band, and insert everything
    for gBox,perLibSeries,gCount in records:
        gBox.finish()
        for series,plotColor in perLibSeries:
            if not series:
                continue
            bandPoints=[gBox.pos(offset,mean+sem) for offset,mean,sem in series]+ \
                [gBox.pos(offset,mean-sem) for offset,mean,sem in reversed(series)]
            bandPath=path.path(path.moveto(*bandPoints[0]),
                *[path.lineto(*p) for p in bandPoints[1:]],path.closepath())
            gBox.fill(bandPath,[plotColor,color.transparency(0.5)])
        c.insert(gBox)
        c.insert(gCount)
    ##
    ##legend, to the right of the last (UTR) panel
    legX=len(regionsOrder)*(boxSize+gap)-gap+0.3
    for li,libName in enumerate(libNames):
        plotColor=resolve_color(color_map,libName,li)
        ly=boxSize-0.3-li*0.4
        c.stroke(path.line(legX,ly,legX+0.6,ly),[plotColor,style.linewidth.Thick])
        c.text(legX+0.75,ly,libName,[text.valign.middle,text.size.small])
    return c

def plotAutoCorrResults(autoCorrResults,outPrefix,minPointFilter=30,color_map=None):
    """
    autoCorrResults is of the format:
    {libNamei:{read_id:{'TL':[len(editString)-20,offsets,pearsonRs,pearsonPvals],
                    'CDS':...
                    'UTR':...}}}
    where offsets/pearsonRs/pearsonPvals are parallel numpy arrays (see computeAutoCorrForLibs).

    (1) Aggregates (pearsonR,pearsonPval) values across all read_ids within each libNamei,
        separately for each of TL/CDS/UTR, into {offset:[(pearsonR,pearsonPval),...]}
        (see aggregateAutoCorrByOffset).
    (2) For each libNamei, builds one plot per region:
         - x-axis: offset, log-scaled (offset==0 is excluded so the log axis is defined)
         - y-axis: pearsonR, fixed to its natural range [-1,1], with a dotted reference
           line at y=0
         Instead of plotting every read's pearsonR (too many points), plots the mean
         pearsonR across reads as a solid line, with a semi-transparent shaded band
         spanning +/-1 SEM above/below it. Each plot is square.
         Vertically arrays the TL (top)/CDS (middle)/UTR (bottom) plots for a single
         libNamei in one column, horizontally arrays the columns for the various libNamei,
         and links the y-axes across libNamei within each region row (see
         buildAutoCorrCanvas).
         For a given offset, if the number of points contributing to the mean and stdev
         falls below minPointFilter, filter out that offset.
    (3) Above each plot from (2), adds a plot showing the number of sites (reads)
        contributing to each offset, log-scaled on the y-axis, sized to a 1:4
        height:width ratio relative to the plot below it, with its x-axis linked to
        that plot's x-axis. As with (2), these site-count plots are horizontally
        arrayed with their y-axes linked across libNamei.
    (4) Repeats (2) and (3) using only statistically significant sites: per
        (libNamei,region), restricts to sites with pearsonPval<=0.01, then
        Bonferroni-corrects by the total number of such sites across ALL offsets in
        that (libNamei,region) (not per-offset), keeping only sites whose corrected
        pval is also <=0.01 (see computeSignificantAgg).
    (5) Saves (2)+(3) as outPrefix+'.autocorr.all.svg', and (4) as
        outPrefix+'.autocorr.significant.svg'.
    (6) Also saves an OVERLAID version of (2)+(3)/(4): the same three region panels, but
        with every libNamei's mean+/-SEM line/band drawn on the SAME shared axes per
        region instead of one column per library (see buildOverlaidAutoCorrCanvas), as
        outPrefix+'.autocorr.overlaid.all.svg' and
        outPrefix+'.autocorr.overlaid.significant.svg'.
    """
    libNames=sorted(autoCorrResults.keys())
    ##(1) aggregate (pearsonR,pearsonPval) across read_ids, per libNamei/region/offset
    agg=aggregateAutoCorrByOffset(autoCorrResults)
    ##
    ##(2)+(3) mean+/-SEM + site-count plots using all sites
    allCanvas=buildAutoCorrCanvas(agg,libNames,minPointFilter,color_map)
    allCanvas.writeSVGfile(outPrefix+'.autocorr.all.svg')
    overlaidAllCanvas=buildOverlaidAutoCorrCanvas(agg,libNames,minPointFilter,color_map)
    overlaidAllCanvas.writeSVGfile(outPrefix+'.autocorr.overlaid.all.svg')
    ##
    ##(4) repeat (2)+(3), restricted to sites whose Bonferroni-corrected pval is significant
    sigAgg=computeSignificantAgg(agg)
    sigCanvas=buildAutoCorrCanvas(sigAgg,libNames,minPointFilter,color_map)
    sigCanvas.writeSVGfile(outPrefix+'.autocorr.significant.svg')
    overlaidSigCanvas=buildOverlaidAutoCorrCanvas(sigAgg,libNames,minPointFilter,color_map)
    overlaidSigCanvas.writeSVGfile(outPrefix+'.autocorr.overlaid.significant.svg')

def main(args):
    inFiles=args[0]
    gtfFile=args[1]
    outPrefix=args[2]
    colorMapPath=args[3] if len(args)>3 else None
    ##
    color_map=load_color_map(colorMapPath) if colorMapPath else {}
    ##
    gtfDict=metaStartStop.parseGTF(gtfFile)
    ##gtfDict is of the format:
    ##{strand:{chr:{absIndx:(txtName,relStart,relStop)]}}}
    ##
    ##now parse the parquetFiles into per-read 0/1 edit strings, with gaps filled in
    ##(TL/CDS/UTR splitting happens next, in computeAutoCorrForLibs).
    print('Parsing parquets and filling in gaps...')
    editStrings=parseParquetsFillInGaps(inFiles,gtfDict)
    print('Done parsing parquets and filling in gaps.\n')
    ##
    if colorMapPath:
        unmatched=[lib for lib in editStrings if lib not in color_map]
        if unmatched:
            print('  WARNING: no color found in %s for librar%s %s; '
                  'falling back to the default palette.'
                  %(colorMapPath,'y' if len(unmatched)==1 else 'ies',unmatched))
    ##
    print('\nPerforming autocorrelation analysis...')
    autoCorrResults=computeAutoCorrForLibs(editStrings)
    print('...done.')
    ##
    ##autoCorrResults is of the format:
    ##{libNamei:{read_id:{'TL':[len(editString)-20,offsets,pearsonRs,pearsonPvals],
    ##                            'CDS':...
    ##                            'UTR':...}}}
    ##where offsets/pearsonRs/pearsonPvals are parallel numpy arrays
    ##
    ##Now plot the data.
    print('Plotting results...')
    plotAutoCorrResults(autoCorrResults,outPrefix,color_map=color_map)
    print('...done.')

if __name__=='__main__':
    Tee()
    main(sys.argv[1:])