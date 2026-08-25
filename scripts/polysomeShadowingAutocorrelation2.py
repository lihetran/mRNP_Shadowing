"""
Joshua Arribere, July 27, 2026 (revised for streaming, low-memory processing)

Script to analyze shadowing data for periodicities via autocorrelation.

Input: inFileParquet.txt - a line-delimited file of format:
        fileName    rep    parquetDir
    (same inFileParquet.txt convention used by calculateProtectionAcrossParquets.py /
    polysomeShadowHMMQC.py / the sibling polysomeShadowingAutocorrelation.py). Each row
    becomes its own library, labeled 'fileName-rep'; parquetDir may be a directory of
    *.parquet chunk files (globbed and read as one library) or a single parquet file.
    gtfFile
    color_map.txt (optional) - manuscript color TSV 'name rep path hex_color' (no
        leading '#'). Labels are looked up as 'name-rep'/'name_rep' or bare 'name';
        unmatched libraries fall back to common.colors(idx).

Output: Graphs of autocorrelation outputs

run as python3 polysomeShadowingAutocorrelation2.py inFileParquet.txt gtfFile outPrefix
    [color_map.txt]

------------------------------------------------------------------------------------
Architecture note (why this file looks different from a "parse everything, then
process everything" script):

Each read is pushed all the way through the pipeline -- extract its sites, fill gaps,
split into TL/CDS/UTR, filter, autocorrelate, and fold the result into a running
per-offset aggregate -- as soon as it's read off disk, in processOneReadSites() (driven
by the file-reading loop in streamAutoCorrForLibs()). Nothing about an individual read
is kept in memory once that happens. This matters because the old design (parse ALL
reads into one big dict, THEN gap-fill ALL of them into another big dict, THEN
autocorrelate ALL of them into a third big dict, THEN aggregate) held every read's data
in memory multiple times over, with peak memory scaling with the number of reads in a
library (which can be very large). Here, peak memory scales with the number of DISTINCT
OFFSETS being tracked per (library,region) -- typically a few hundred to a couple
thousand -- regardless of how many millions of reads are streamed through.

Memory note: foldReadIntoAggregate's per-(region,offset) accumulator used to be a plain
list that grew by one (pearsonR,pearsonPval) tuple per read reaching that offset -- which
actually made peak memory scale with reads x offsets, not offsets alone, despite the
paragraph above. It's now two much smaller structures per (region,offset): a running
(nTotal,nValid,sumR,sumRsq) accumulator ("stats") that mean/SEM are computed from
directly (see computeMeanSEMFromStats) without ever storing individual values, and a
list ("sig") that only retains pairs with pearsonPval<=SIG_PVAL_THRESHOLD -- the same
raw threshold computeSignificantAgg's Bonferroni correction already restricts to, so
prefiltering here discards (rather than stores-then-discards) the large majority of
non-significant pairs. This is what makes the paragraph above actually true.

The functions below are deliberately kept small and single-purpose (one pipeline stage
each) so each can be tested/troubleshot in isolation, e.g. from a REPL:
    sites = extractSitesForRow(someRow, gtfDict)
    degapped = fillGapsForRead(sites)
    regionEdits = splitSitesIntoRegions(degapped)
    ...
------------------------------------------------------------------------------------
"""
import sys, common, metaStartStop, collections, math, statistics
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import t as tdist
from pyx import *
from logJosh import Tee


def parse_inFileParquet_libs(path: str) -> list:
    """
    Parse a line-delimited inFileParquet.txt file of format:
        fileName    rep    parquetDir
    (same convention as calculateProtectionAcrossParquets.py's
    parse_parquet_libs_file / polysomeShadowHMMQC.py / the sibling
    polysomeShadowingAutocorrelation.py's parse_inFileParquet_libs), and
    return a list of (libraryID, parquetFiles) tuples with libraryID =
    'fileName-rep' and parquetFiles = the sorted list of every *.parquet
    file found under parquetDir if it's a directory, or [parquetDir]
    itself if it's already a single file.
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


def load_color_map(path: str) -> dict:
    """
    Parse a manuscript color-map TSV with columns:
        sample_name, rep, path, hex_color (no leading '#')
    Returns a dict keyed by "name_rep", "name-rep" (this script's own
    libraryID convention -- see parseLibFilesList, "%s-%s"-style labels
    like the sibling polysomeShadowingAutocorrelation.py), and bare "name"
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


##Only these columns of the parquet files are ever touched. Restricting pd.read_parquet
##to just these avoids loading/deserializing unused (and often large) columns like
##'aligned_pairs' or 'read_sequence' -- a large time/memory savings on its own, on top of
##the streaming design.
NEEDED_PARQUET_COLUMNS=['chrom','gene_strand','transcript_id','read_id',
    'ref_sequence_aligned','absolute_indices','edit_string']

##Raw per-entry significance threshold applied in TWO places that must never drift apart:
##(1) foldReadIntoAggregate's fold-time prefilter, which only ever retains a
##    (pearsonR,pearsonPval) pair in the 'sig' accumulator if pearsonPval<=this, and
##(2) computeSignificantAgg's final Bonferroni-corrected cutoff.
##Using one named constant for both means "significant" always means the same thing
##whether or not you consider a pair for the correction, or accept it after correction.
SIG_PVAL_THRESHOLD=0.001

##############################################################################
## Stage 1: turn one parquet row into a single read's (relStart,relStop,absIndx,refNt,
## edit) site list.
##############################################################################

def extractSitesForRow(row,gtfDict):
    """
    row is one row of a parquet DataFrame (as yielded by df.itertuples(index=False)),
    expected to have (at least) the columns: transcript_id, gene_strand, chrom,
    ref_sequence_aligned, absolute_indices, edit_string.

    gtfDict is of the format {strand:{chr:{absIndx:[(txtName,relStart,relStop)]}}}.

    Zips ref_sequence_aligned / absolute_indices / edit_string together to walk this
    read's aligned positions one nucleotide at a time. For each position, keeps it only
    if ALL of the following hold:
        - absIdx is not NaN (i.e. the position has a genomic anchor -- not an insertion)
        - edit is 0 or 1 (not some other gap/ambiguity code)
        - gtfDict has exactly one (txtName,relStart,relStop) annotated at that absIdx
          (an ambiguous or unannotated position is dropped)
        - that annotation's txtName matches this read's own transcript_id (guards
          against overlapping/nested gene annotations mapping this read to the wrong
          transcript's coordinate frame)

    Returns a list of (relStart,relStop,absIndx,refNt,edit) tuples for the surviving
    positions, sorted by increasing relStart. Returns an empty list if nothing survives.
    """
    transcript_id=row.transcript_id
    strand=row.gene_strand
    chrom=row.chrom
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
    sites.sort(key=lambda s:s[0])##order by increasing relStart
    return sites

##############################################################################
## Stage 2: fill in the gaps at non-'A' positions using the nearest 'A'.
##############################################################################

def fillGapsForRead(sites,method,windowNt):
    """
    sites is a list of (relStart,relStop,absIndx,refNt,edit) tuples for a single read,
    sorted by increasing relStart.

    This function will fill in values for the non-A nts according to one of two methods:
    If method is 'gap_fill':
        For every position where refNt!='A', replaces edit with the edit value of the nearest
        (by relStart distance) position where refNt=='A'. Ties (equidistant A's on either side
        with differing edit values) are broken in favor of the upstream (lower relStart) A.
    If method is 'window':
        For every position, whether refNt is A or not, define a window centered at that position.
        The window will be defined by [relStart-windowNt//2,relStart+windowNt//2], for relStart
        of a given position. For all positions in that window, add up all 1s, and divide by the
        total number of refNt==A sites. This is equivalent to the fraction of sites for which
        refNt is A and is 1. If there are no refNt=='A' sites in that window, falls back to the
        value of the single nearest refNt=='A' site instead (the same nearest-neighbor lookup
        'gap_fill' uses, including its upstream tie-break) -- deliberately NOT nan. Every
        position downstream (computeAutocorrelation, via splitSitesIntoRegions) is compared to
        another position purely by LIST INDEX, which stands in for nucleotide distance; dropping
        or NaN-ing out a position would silently shift every later position's effective offset,
        so every position here must get a real, usable value, even in this rare edge case.

    Returns a new list of (relStart,relStop,absIndx,refNt,degapped_edit) tuples, or None if
    the read contains no refNt=='A' position at all (nothing to borrow an edit value from).
    """
    n=len(sites)
    if not any(s[3]=='A' for s in sites):
        return None##no informative site anywhere in this read
    ##
    ##nearest refNt=='A' site at-or-before / at-or-after each position -- the primary
    ##mechanism for 'gap_fill', and the empty-window fallback for 'window' (see above)
    leftA=[None]*n
    lastSeen=None
    for i in range(n):
        if sites[i][3]=='A':
            lastSeen=i
        leftA[i]=lastSeen
    rightA=[None]*n
    lastSeen=None
    for i in range(n-1,-1,-1):
        if sites[i][3]=='A':
            lastSeen=i
        rightA[i]=lastSeen
    ##
    def nearestAValue(i,relStart):
        li,ri=leftA[i],rightA[i]
        if li is None:
            chosen=ri
        elif ri is None:
            chosen=li
        else:
            leftDist=relStart-sites[li][0]
            rightDist=sites[ri][0]-relStart
            chosen=li if leftDist<=rightDist else ri##ties favor the upstream (lower relStart) A
        return sites[chosen][4]
    ##
    if method=='gap_fill':
        degapped=[]
        for i,(relStart,relStop,absIndx,refNt,edit) in enumerate(sites):
            if refNt=='A':
                degapped.append((relStart,relStop,absIndx,refNt,edit))
            else:
                degapped.append((relStart,relStop,absIndx,refNt,nearestAValue(i,relStart)))
        return degapped
    ##
    elif method=='window':
        ##the refNt=='A' subsequence -- sites is sorted by relStart, so this is too
        aRelStarts=[s[0] for s in sites if s[3]=='A']
        aEdits=[s[4] for s in sites if s[3]=='A']
        prefixSumA=[0]##prefixSumA[k] = sum of the edit values of the first k A-sites
        for e in aEdits:
            prefixSumA.append(prefixSumA[-1]+e)
        ##
        half=windowNt//2
        degapped=[]
        aLeft=0##first A-site index with relStart >= the current window's lower bound
        aRight=0##first A-site index with relStart > the current window's upper bound
        for i,(relStart,relStop,absIndx,refNt,edit) in enumerate(sites):
            lo=relStart-half
            hi=relStart+half
            ##relStart only increases as we go, so lo/hi are non-decreasing too --
            ##aLeft/aRight only ever need to move forward (a classic sliding window),
            ##making this O(n) overall rather than re-scanning the window each time
            while aLeft<len(aRelStarts) and aRelStarts[aLeft]<lo:
                aLeft+=1
            while aRight<len(aRelStarts) and aRelStarts[aRight]<=hi:
                aRight+=1
            countA=aRight-aLeft
            if countA==0:
                windowEdit=nearestAValue(i,relStart)##no A in this window -- fall back (see above)
            else:
                windowEdit=(prefixSumA[aRight]-prefixSumA[aLeft])/countA
            degapped.append((relStart,relStop,absIndx,refNt,windowEdit))
        return degapped
    ##
    else:
        raise ValueError("fillGapsForRead: method must be 'gap_fill' or 'window', got %r"%(method,))

##############################################################################
## Stage 3: split one (gap-filled) read's sites into TL/CDS/UTR edit-value lists.
##############################################################################

def splitSitesIntoRegions(sites):
    """
    sites is a list of (relStart,relStop,absIndx,refNt,edit) tuples for a single read
    (normally already gap-filled -- see fillGapsForRead).

    Splits the read's edit values into three regions by relStart/relStop:
        TL:  relStart<0
        CDS: relStart>=0 and relStop<=-2
        UTR: relStop>2
    (a position with relStop in [-1,2] -- immediately around the stop codon -- falls in
    none of the three, and is simply dropped)

    Returns {'TL':[edit,...],'CDS':[edit,...],'UTR':[edit,...]}: just the edit values (0/1
    from the 'gap_fill' method, or fractional in [0,1] from the 'window' method -- see
    fillGapsForRead), in increasing relStart order, ready to feed into computeAutocorrelation.

    JA: Note: this is cheaper b/c it just keeps track of the 1/0 values. That's all that's
    needed for downstream analyses, provided the order of the 1/0 is not perturbed. (This
    is also why fillGapsForRead never drops or NaNs out a position, even in its 'window'
    method's rare empty-window case: computeAutocorrelation compares positions purely by
    LIST INDEX as a stand-in for nucleotide distance, so removing one here would silently
    shift every later position's effective offset.)
    """
    regionEdits={'TL':[],'CDS':[],'UTR':[]}
    for relStart,relStop,_,_,edit in sites:
        if relStart<0:
            regionEdits['TL'].append(edit)
        if relStart>=0 and relStop<=-2:
            regionEdits['CDS'].append(edit)
        if relStop>2:
            regionEdits['UTR'].append(edit)
    return regionEdits

##############################################################################
## Stage 4: autocorrelation for one region's edit-value list.
##############################################################################

def computeAutocorrelation(editString,minOffset=0,maxOffset=None):
    """
    editString is a list of numeric edit values -- 0/1 from the 'gap_fill' gap-filling
    method, or fractional values in [0,1] from the 'window' method (see
    fillGapsForRead); the math below works identically either way. For that editString,
    this function will compute the autocorrelation by:
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
    ##JA: this is just initialization and processing of parameters
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
    ##JA: this is the start of the computations. The basic logic of this approach is to pre-compute
    ##some of the summary parameters (mean,var) so that they don't have to be re-calculated every
    ##time.
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
        ##this E[X^2]-E[X]^2 formula is prone to floating-point catastrophic cancellation
        ##for near-zero variance (common with 'window'-method fractional edit values,
        ##whose true variance can be many orders of magnitude smaller than float64's
        ##~1e-16 relative noise floor here): the subtraction can round to a tiny NEGATIVE
        ##number (making sqrt() below raise) or a tiny but spuriously POSITIVE number
        ##(silently producing a nonsensical |r|>1 "correlation" from noise, not signal).
        ##varEps is comfortably above that noise floor and comfortably below any real
        ##variance this data can have (minimum ~1/windowNt or ~1/readLength), so treating
        ##anything below it as exactly-zero variance is safe in both directions.
        varEps=1e-9
        varOriginal=cumsumsq[length]/length-meanOriginal*meanOriginal
        varShifted=(cumsumsq[n]-cumsumsq[offset])/length-meanShifted*meanShifted
        if varOriginal<varEps or varShifted<varEps:
            pearsonRs[idx]=float('nan')
        else:
            pearsonRs[idx]=cov/math.sqrt(varOriginal*varShifted)
        dofs[idx]=length-2
    ##
    with np.errstate(invalid='ignore',divide='ignore'):##NaN/inf here mirror scipy's own
        ##behavior for zero-variance/perfectly-correlated inputs, so are expected, not bugs
        tStats=pearsonRs*np.sqrt(dofs/(1-pearsonRs*pearsonRs))
        pearsonPvals=2*tdist.sf(np.abs(tStats),dofs)
    return (offsets,pearsonRs,pearsonPvals),n-20

##############################################################################
## Stage 5: fold one read's per-region autocorrelation directly into the running,
## cross-read aggregate -- this is the step that makes the whole pipeline streaming.
##############################################################################

def foldReadIntoAggregate(libAgg,regionEdits,minOffset,maxOffset):
    """
    Computes the autocorrelation for each of a single read's TL/CDS/UTR edit-value lists
    (regionEdits, from splitSitesIntoRegions), and immediately folds the resulting
    (pearsonR,pearsonPval) pairs into libAgg -- instead of retaining this read's own
    results anywhere afterward. This is what keeps peak memory proportional to the number
    of distinct offsets tracked, not the number of reads processed (see the module
    docstring's memory note).

    libAgg is of the format:
        {'TL':{'stats':{offset:[nTotal,nValid,sumR,sumRsq]},
               'sig':{offset:[(pearsonR,pearsonPval),...]}},
         'CDS':...,'UTR':...}
    with 'stats'/'sig' each a collections.defaultdict (so a not-yet-seen offset is
    created on first use), and is mutated in place; this function returns nothing.

    'stats' is a running-sums accumulator, updated in O(1) per (region,offset) regardless
    of how many reads have already contributed to it: nTotal counts every read that
    reached this offset (matching the site-count plot's original semantics, which counted
    every entry including NaN ones); nValid/sumR/sumRsq accumulate only the non-NaN
    pearsonR values (computeAutocorrelation returns NaN for zero-variance/constant
    inputs), from which mean/SEM are computed directly later (see
    computeMeanSEMFromStats) without ever storing each individual value.

    'sig' retains actual (pearsonR,pearsonPval) pairs, but ONLY those with
    pearsonPval<=SIG_PVAL_THRESHOLD -- the same raw threshold computeSignificantAgg's
    Bonferroni correction already restricts its candidates to, so prefiltering here means
    the (typically large) majority of non-significant pairs are never retained in memory
    at all, rather than stored and filtered out only when plotting.
    """
    for region,editString in regionEdits.items():
        ##region is TL/CDS/UTR
        ##editString is a list of 0/1 (gap_fill) or fractional [0,1] (window) edit values
        (offsets,pearsonRs,pearsonPvals),_=computeAutocorrelation(editString,
            minOffset=minOffset,maxOffset=maxOffset)
        statsDict=libAgg[region]['stats']
        sigDict=libAgg[region]['sig']
        for offset,pearsonR,pearsonPval in zip(offsets,pearsonRs,pearsonPvals):
            offset=int(offset)
            entry=statsDict[offset]##[nTotal,nValid,sumR,sumRsq], created fresh on first use
            entry[0]+=1
            if not math.isnan(pearsonR):
                entry[1]+=1
                entry[2]+=pearsonR
                entry[3]+=pearsonR*pearsonR
            if pearsonPval<=SIG_PVAL_THRESHOLD:
                sigDict[offset].append((pearsonR,pearsonPval))

##############################################################################
## Per-read orchestrator: stages 2-5 for one read's already-extracted sites.
## (Stage 1, extractSitesForRow, is kept separate -- see streamAutoCorrForLibs -- because
## the caller needs its result, specifically whether it's empty, to decide whether this
## row counts towards maxReadsPerLib.)
##############################################################################

def processReadSites(sites,libAgg,filterCDSLength,minOffset,maxOffset,method,windowNt):
    """
    Runs the rest of the per-read pipeline for one read's already-extracted, non-empty
    sites list (see extractSitesForRow): fills gaps, splits into TL/CDS/UTR, filters by
    CDS length, and -- if it survives -- folds its autocorrelation into libAgg.

    Returns True if the read was kept and folded into libAgg, False if it was dropped
    along the way (no refNt=='A' anywhere, or too little CDS). Nothing about this read is
    retained beyond this function call either way.
    """
    degapped=fillGapsForRead(sites,method,windowNt)
    if degapped is None:
        return False##no refNt=='A' anywhere in this read -- nothing informative to keep
    ##
    regionEdits=splitSitesIntoRegions(degapped)
    if len(regionEdits['CDS'])<filterCDSLength:
        return False##not enough CDS nucleotides -- drop this read entirely
    ##
    foldReadIntoAggregate(libAgg,regionEdits,minOffset,maxOffset)
    return True

##############################################################################
## Library/file-level driver: this replaces the old parseParquetsFillInGaps ->
## computeAutoCorrForLibs -> aggregateAutoCorrByOffset three-stage pipeline.
##############################################################################

def parseLibFilesList(inFiles):
    """
    inFiles is an inFileParquet.txt file, line-delimited with each line
    containing:
    fileName    rep    parquetDir
    (same convention as calculateProtectionAcrossParquets.py /
    polysomeShadowHMMQC.py / the sibling polysomeShadowingAutocorrelation.py
    -- see parse_inFileParquet_libs). Each row becomes its own libName,
    labeled 'fileName-rep'; parquetDir may be a directory of *.parquet
    chunk files (globbed into that libName's list of libFiles) or a
    single parquet file.

    Returns a list of (libName,libFiles) tuples, one per non-blank line.
    """
    return parse_inFileParquet_libs(inFiles)

def streamAutoCorrForLibs(inFiles,gtfDict,maxReadsPerLib=100000,filterCDSLength=300,
        minOffset=10,maxOffset=1000,method='gap_fill',windowNt=30):
    """
    Main entry point for turning parquet files into the aggregated autocorrelation object
    that plotAutoCorrResults consumes, WITHOUT ever holding more than one read's data in
    memory at a time: for each row of each parquet file, extracts that read's sites
    (extractSitesForRow), and -- if it has any -- immediately runs it through the rest of
    the pipeline and folds it into a small running aggregate (processReadSites /
    foldReadIntoAggregate). See the module docstring for why this matters.

    inFiles/gtfDict: as in the original, batch-oriented parseParquetsFillInGaps.
    maxReadsPerLib: stop processing a library's parquet files once this many reads have
        had at least one usable site extracted (matching the ORIGINAL script's counting
        point, i.e. as soon as a read would have been added to its readSites dict) --
        NOT only reads that additionally survive gap-filling/CDS-length filtering below.
        A duplicate read_id (already seen for this library) is skipped and does not count
        towards this limit either (see the note on read_id de-duplication below).
        None disables the cap.
    filterCDSLength/minOffset/maxOffset: passed straight through to processReadSites ->
        foldReadIntoAggregate -> computeAutocorrelation; see their docstrings.

    Returns agg, of the format:
        {libNamei:{'TL':{'stats':{offset:[nTotal,nValid,sumR,sumRsq]},
                          'sig':{offset:[(pearsonR,pearsonPval),...]}},
                   'CDS':...,'UTR':...}}
    (see foldReadIntoAggregate for what 'stats'/'sig' hold and why). Downstream,
    normalizeAllSitesAgg/normalizeSignificantAgg convert this into the flat
    {offset:(mean,sem,count)}-shaped summaries buildAutoCorrCanvas/plotAutoCorrResults
    actually draw from.

    Note on read_id de-duplication: the original script stored each read's sites in a
    dict keyed by read_id, so if the same read_id appeared in more than one row (e.g. a
    multi-mapped read with more than one alignment kept), only its LAST occurrence was
    ultimately used. Streaming can't replicate "last occurrence wins" without buffering
    every read first -- which is exactly what this rewrite is avoiding -- so instead this
    keeps a per-library set of read_ids already folded into libAgg, and skips (does not
    re-fold, does not double-count) any later occurrence of the same read_id. This is
    "first occurrence wins" rather than "last occurrence wins". If your data can contain
    multiple alignments per read_id and that distinction matters to you, flag it and this
    can be revisited.
    """
    libFilesList=parseLibFilesList(inFiles)
    ##libFilesList is a list of tuples of [(libName,libFiles),...]
    ##where libFiles is a list of the filenames associated with that libName.
    ##libFiles will be at least one file long.
    ##
    agg={}
    for libName,libFiles in libFilesList:
        print('\nParsing library %s (%d parquet files)...'%(libName,len(libFiles)))
        ##libAgg holds ONLY the running per-offset aggregate for this library -- never
        ##any individual read's own data -- which is the crux of the memory savings here
        ##(see foldReadIntoAggregate for what 'stats'/'sig' hold)
        libAgg={region:{'stats':collections.defaultdict(lambda:[0,0,0.0,0.0]),
                        'sig':collections.defaultdict(list)}
                for region in ('TL','CDS','UTR')}
        seenReadIds=set()##for de-duplication -- see the docstring note above
        readCount=0##reads with >=1 usable site (matches the original maxReadsPerLib point)
        keptCount=0##reads that made it all the way through to contribute to libAgg
        ##
        for libFile in libFiles:
            ##Note: JA changed the next line to ensure that we have at least keptCount reads
            ##per library. That way if a bunch of reads get filtered out without contributing,
            ##that doesn't decrease the total number of reads per lib.
            if maxReadsPerLib is not None and keptCount>=maxReadsPerLib:
                break##already reached the per-library read cap -- skip remaining files
            print('Parsing %s...'%(libFile))
            df=pd.read_parquet(libFile,columns=NEEDED_PARQUET_COLUMNS)
            ##
            for row in df.itertuples(index=False):
                ##Note: JA changed the next line to ensure that we have at least keptCount reads
                ##per library. That way if a bunch of reads get filtered out without contributing,
                ##that doesn't decrease the total number of reads per lib.
                if maxReadsPerLib is not None and keptCount>=maxReadsPerLib:
                    print('Reached maxReadsPerLib=%d for %s -- moving on.'%(
                        maxReadsPerLib,libName))
                    break##reached the per-library read cap -- move on to the next libNamei
                ##
                sites=extractSitesForRow(row,gtfDict)
                if not sites:
                    continue##this read didn't contribute any usable positions
                if row.read_id in seenReadIds:
                    continue##already folded this read_id in -- see de-duplication note
                seenReadIds.add(row.read_id)
                readCount+=1
                ##
                if processReadSites(sites,libAgg,filterCDSLength,minOffset,maxOffset,method,windowNt):
                    keptCount+=1
        ##
        print('%s: %d reads had usable sites, %d were kept after gap-filling/CDS-length '
            'filtering.'%(libName,readCount,keptCount))
        ##convert the per-offset defaultdicts to plain dicts before handing this off --
        ##keeps the returned object's type contract identical to a plain nested dict
        agg[libName]={region:{'stats':dict(regionData['stats']),'sig':dict(regionData['sig'])}
            for region,regionData in libAgg.items()}
    ##
    return agg

##############################################################################
## Everything below here (mean/SEM computation, significance filtering, and plotting)
## is unchanged from the batch version -- it all operates on agg, the aggregated
## structure that streamAutoCorrForLibs now produces directly.
##############################################################################

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

def computeMeanSEMFromStats(nValid,sumR,sumRsq):
    """
    Computes (mean,sem) directly from running sums -- nValid non-NaN pearsonR
    contributions, their sum, and their sum-of-squares (see foldReadIntoAggregate's
    'stats' accumulator) -- instead of from a stored list of individual values. Returns
    (nan,0.0) if nValid==0; sem is 0.0 if nValid<2 (sample variance is undefined for
    n<2, matching computeMeanSEM's convention).

    Uses the sum/sumSq variance formula rather than Welford's online algorithm:
    pearsonR is bounded to [-1,1], so unlike the edit-value variance elsewhere in this
    file (see computeAutocorrelation's varEps comment), catastrophic cancellation here
    is not a practical concern at float64 precision.
    """
    if nValid==0:
        return float('nan'),0.0
    mean=sumR/nValid
    if nValid<2:
        return mean,0.0
    ##max(...,0.0) guards against a tiny negative from float roundoff when the true
    ##variance is near zero
    variance=max(0.0,(sumRsq-nValid*mean*mean)/(nValid-1))
    return mean,math.sqrt(variance/nValid)

def summarizeStatsOffsets(statsDict,minPointFilter=1):
    """
    Normalizes a 'stats' running-sums accumulator (see foldReadIntoAggregate) --
    {offset:[nTotal,nValid,sumR,sumRsq]} -- into the {'series':[(offset,mean,sem),...],
    'counts':{offset:nTotal}} shape buildAutoCorrCanvas/buildAutoCorrCanvasTogether
    expect, computed directly from the running sums rather than from any stored
    per-read values.

    'counts' uses nTotal (every read that reached that offset, including NaN-pearsonR
    ones) for the site-count plot, matching the site-count plot's original semantics;
    'series' (the mean+/-SEM plot) excludes NaN contributions and any offset with fewer
    than minPointFilter of them, exactly as computeMeanSEMSeries did on the old
    list-of-pairs representation.
    """
    counts={offset:entry[0] for offset,entry in statsDict.items()}
    series=[]
    for offset in sorted(statsDict.keys()):
        if offset==0:
            continue
        nTotal,nValid,sumR,sumRsq=statsDict[offset]
        if nValid<minPointFilter:
            continue
        mean,sem=computeMeanSEMFromStats(nValid,sumR,sumRsq)
        series.append((offset,mean,sem))
    return {'series':series,'counts':counts}

def summarizePairsOffsets(offsetDict,minPointFilter=1):
    """
    Normalizes a plain {offset:[(pearsonR,pearsonPval),...]} mapping (e.g.
    computeSignificantAgg's output) into the same {'series':...,'counts':...} shape
    summarizeStatsOffsets produces, so buildAutoCorrCanvas/buildAutoCorrCanvasTogether
    can consume either "all sites" or "significant sites" data uniformly. 'counts' is
    simply len(pairs) per offset, matching the site-count plot's original behavior for
    the significant-sites figure.
    """
    counts={offset:len(pairs) for offset,pairs in offsetDict.items()}
    series=computeMeanSEMSeries(offsetDict,minPointFilter)
    return {'series':series,'counts':counts}

def normalizeAllSitesAgg(agg,minPointFilter=1):
    """
    Converts agg (see streamAutoCorrForLibs -- {libName:{region:{'stats':...,'sig':...}}})
    into {libName:{region:{'series':...,'counts':...}}} for the "all sites" figure,
    using ONLY the 'stats' running-sums accumulator (see summarizeStatsOffsets) -- the
    'sig' pre-filtered candidate list is not used here at all; it exists purely for
    computeSignificantAgg.
    """
    return {libName:{region:summarizeStatsOffsets(regionData['stats'],minPointFilter)
                for region,regionData in regionsDict.items()}
            for libName,regionsDict in agg.items()}

def normalizeSignificantAgg(sigAgg,minPointFilter=1):
    """
    Converts sigAgg (see computeSignificantAgg -- {libName:{region:{offset:
    [(pearsonR,pearsonPval),...]}}}, already Bonferroni-corrected) into the same
    {libName:{region:{'series':...,'counts':...}}} shape as normalizeAllSitesAgg, via
    summarizePairsOffsets, so buildAutoCorrCanvas/buildAutoCorrCanvasTogether can
    consume either figure uniformly.
    """
    return {libName:{region:summarizePairsOffsets(offsetDict,minPointFilter)
                for region,offsetDict in regionsDict.items()}
            for libName,regionsDict in sigAgg.items()}

def computeSignificantAgg(agg):
    """
    agg is of the format {libNamei:{region:{'stats':...,'sig':{offset:[(pearsonR,
    pearsonPval),...]}}}} (see streamAutoCorrForLibs/foldReadIntoAggregate) -- 'sig' is
    already pre-filtered at fold time to only entries with
    pearsonPval<=SIG_PVAL_THRESHOLD, so the per-offset lists here only ever contain
    candidates for significance, not every read that touched that offset.

    Per (libNamei,region), Bonferroni-corrects those pre-filtered pvals by N -- the
    total number of pre-filtered entries across ALL offsets in that (libNamei,region),
    not just within a single offset -- then keeps only entries whose corrected pval is
    also <=SIG_PVAL_THRESHOLD.

    Returns {libNamei:{region:{offset:[(pearsonR,pearsonPval),...]}}} -- just the
    'sig'-shaped part, restricted to entries surviving Bonferroni correction -- which is
    what normalizeSignificantAgg expects. This is a DIFFERENT shape than agg's own
    per-region value (which additionally carries 'stats'); the "all sites" figure uses
    agg[lib][region]['stats'] directly, via normalizeAllSitesAgg, not this function.
    """
    sigAgg={}
    for libName,regionsDict in agg.items():
        sigAgg[libName]={}
        for region,regionData in regionsDict.items():
            offsetDict=regionData['sig']
            ##N, the number of (already pre-filtered) entries across every offset in
            ##this (libNamei,region) -- no need to re-filter by pearsonPval<=
            ##SIG_PVAL_THRESHOLD here, since offsetDict only ever contains such entries
            numPassed=sum(len(pairs) for pairs in offsetDict.values())
            sigOffsetDict={}
            for offset,pairs in offsetDict.items():
                significant=[(pearsonR,pearsonPval*numPassed) for pearsonR,pearsonPval in pairs
                    if pearsonPval*numPassed<=SIG_PVAL_THRESHOLD]
                if significant:
                    sigOffsetDict[offset]=significant
            sigAgg[libName][region]=sigOffsetDict
    return sigAgg

def buildAutoCorrCanvas(normalized,libNames,color_map=None):
    """
    Builds a single pyx canvas containing, for each libNamei (columns, in libNames order),
    a column of TL (top)/CDS (middle)/UTR (bottom) plots of pearsonR vs offset -- mean as a
    solid line, +/-1 SEM as a semi-transparent shaded band -- each with a linked, log-scaled
    site-count plot directly above it. Y-axes are linked across libNamei within each region
    row (and, separately, within each site-count row); each site-count plot's x-axis is
    linked to the plot beneath it. Each Pearson R plot also draws a dotted reference line
    at y=0.

    normalized is {libName:{region:{'series':[(offset,mean,sem),...],
    'counts':{offset:count}}}} (see normalizeAllSitesAgg/normalizeSignificantAgg) --
    already summarized down to just what this function draws, uniformly regardless of
    whether it came from the "all sites" running-sums accumulator or the "significant
    sites" Bonferroni-corrected pairs (minPointFilter, previously a parameter here, is
    now applied during that normalization step instead).

    color_map (library label -> hex, no leading '#' needed), if given, colors each
    libNamei's column with its manuscript color (see resolve_color); a library missing
    from color_map falls back to common.colors(idx).

    Axis ranges are fixed rather than auto-scaled from each individual plot's own data:
    Pearson R to its natural [-1,1] range, and the offset/site-count ranges to the min/max
    across the whole normalized structure passed in. This is deliberate, not cosmetic --
    it avoids pyx's "zero axis range" crash on a region/library with no (or only one
    distinct value of) data, e.g. a library with zero significant sites anywhere in a
    region.

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
            for offset,count in normalized.get(libName,{}).get(region,{}).get('counts',{}).items():
                if offset!=0:
                    maxCount=max(maxCount,count)
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
            regionInfo=normalized.get(libName,{}).get(region,{})
            series=regionInfo.get('series',[])
            counts=regionInfo.get('counts',{})
            countPoints=[(offset,counts[offset])
                for offset in sorted(counts.keys()) if offset!=0]
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

def plotAutoCorrResults(agg,outPrefix,minPointFilter=30,color_map=None):
    """
    agg is of the format:
    {libNamei:{'TL':{'stats':{offset:[nTotal,nValid,sumR,sumRsq]},
                      'sig':{offset:[(pearsonR,pearsonPval),...]}},'CDS':...,'UTR':...}}
    -- i.e. the object streamAutoCorrForLibs returns (already aggregated across reads;
    there is no per-read structure left by this point). normalizeAllSitesAgg/
    normalizeSignificantAgg (see below) flatten this into the {'series':...,'counts':...}
    shape buildAutoCorrCanvas actually draws from.

    (1) For each libNamei, builds one plot per region:
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
    (2) Above each plot from (1), adds a plot showing the number of sites (reads)
        contributing to each offset, log-scaled on the y-axis, sized to a 1:4
        height:width ratio relative to the plot below it, with its x-axis linked to
        that plot's x-axis. As with (1), these site-count plots are horizontally
        arrayed with their y-axes linked across libNamei.
    (3) Repeats (1) and (2) using only statistically significant sites: per
        (libNamei,region), restricts to sites with pearsonPval<=0.01, then
        Bonferroni-corrects by the total number of such sites across ALL offsets in
        that (libNamei,region) (not per-offset), keeping only sites whose corrected
        pval is also <=0.01 (see computeSignificantAgg).
    (4) Saves (1)+(2) as outPrefix+'.autocorr.all.pdf', and (3) as
        outPrefix+'.autocorr.significant.pdf'.
    """
    libNames=sorted(agg.keys())
    ##
    ##(1)+(2) mean+/-SEM + site-count plots using all sites
    allNormalized=normalizeAllSitesAgg(agg,minPointFilter)
    allCanvas=buildAutoCorrCanvas(allNormalized,libNames,color_map)
    allCanvas.writePDFfile(outPrefix+'.autocorr.all.pdf')
    ##
    ##(3) repeat (1)+(2), restricted to sites whose Bonferroni-corrected pval is significant
    sigAgg=computeSignificantAgg(agg)
    sigNormalized=normalizeSignificantAgg(sigAgg,minPointFilter)
    sigCanvas=buildAutoCorrCanvas(sigNormalized,libNames,color_map)
    sigCanvas.writePDFfile(outPrefix+'.autocorr.significant.pdf')

def buildAutoCorrCanvasTogether(normalized,libNames,color_map=None):
    """
    Like buildAutoCorrCanvas, but instead of one column of TL/CDS/UTR plots per libNamei,
    builds exactly THREE plots (TL, CDS, UTR, left to right), each one OVERLAYING every
    libNamei's mean+/-SEM pearsonR-vs-offset band (one color per library, via
    resolve_color) on a single shared set of axes, with a color-keyed legend below. Each
    region's plot still has its own linked, log-scaled site-count plot directly above it
    (all libraries' counts overlaid there too, same color per library).

    normalized is {libName:{region:{'series':...,'counts':...}}} -- the same normalized
    shape buildAutoCorrCanvas consumes (see normalizeAllSitesAgg/normalizeSignificantAgg);
    minPointFilter, previously a parameter here, is now applied during that normalization
    step instead.

    color_map (library label -> hex, no leading '#' needed), if given, colors each
    library the same way as buildAutoCorrCanvas (see resolve_color); a library missing
    from color_map falls back to common.colors(idx).

    Axis ranges are fixed for the same reason as buildAutoCorrCanvas (Pearson R to its
    natural [-1,1]; offset/site-count to the min/max across the whole normalized
    structure passed in) -- avoids pyx's "zero axis range" crash on a region with no data
    anywhere.

    Building the shaded bands again requires each region's box graph to be .finish()-ed
    (so .pos() gives valid physical coordinates) before every library's data has been
    registered on it -- so, as in buildAutoCorrCanvas, ALL libraries' data is plotted onto
    a region's graphs first, and only then is that graph finished/filled, in two passes.
    """
    regions=['TL','CDS','UTR']##left-to-right order, per the header
    boxSize=8##mean+/-SEM plots are square
    countHeight=2##countHeight:boxSize is a 1:4 ratio, matching buildAutoCorrCanvas
    subGap=0.3
    colGap=2
    ##
    ##a fixed, data-independent range for every axis (see buildAutoCorrCanvas for why)
    maxCount=2
    allOffsets=[]
    for libName in libNames:
        for region in regions:
            for offset,count in normalized.get(libName,{}).get(region,{}).get('counts',{}).items():
                if offset!=0:
                    maxCount=max(maxCount,count)
                    allOffsets.append(offset)
    minOffset,maxOffset=(min(allOffsets),max(allOffsets)) if allOffsets else (1,2)
    if minOffset==maxOffset:
        maxOffset=minOffset+1
    ##
    c=canvas.canvas()
    ##first pass: one gBox+gCount pair PER REGION (not per library), with every library's
    ##data registered onto that same shared pair of graphs, but nothing finished yet
    records=[]##(region,xpos,gBox,gCount,perLibSeries)
    for ri,region in enumerate(regions):
        xpos=ri*(boxSize+colGap)
        gBox=graph.graphxy(width=boxSize,height=boxSize,xpos=xpos,ypos=0,
            x=graph.axis.log(min=minOffset,max=maxOffset,title='Offset'),
            y=graph.axis.linear(min=-1,max=1,title='%s Pearson R'%region))
        ##reference line at y=0
        gBox.plot(graph.data.points([(minOffset,0),(maxOffset,0)],x=1,y=2),
            [graph.style.line([color.grey(0.5),style.linestyle.dotted])])
        ##
        countYpos=boxSize+subGap
        gCount=graph.graphxy(width=boxSize,height=countHeight,xpos=xpos,ypos=countYpos,
            x=graph.axis.linkedaxis(gBox.axes['x']),
            y=graph.axis.log(min=1,max=maxCount,title='N Sites'))
        ##
        perLibSeries=[]##(plotColor,series), deferred to the second pass for the fill
        for ii,libName in enumerate(libNames):
            plotColor=resolve_color(color_map,libName,ii)
            regionInfo=normalized.get(libName,{}).get(region,{})
            series=regionInfo.get('series',[])
            counts=regionInfo.get('counts',{})
            countPoints=[(offset,counts[offset])
                for offset in sorted(counts.keys()) if offset!=0]
            if series:
                upperPoints=[(offset,mean+sem) for offset,mean,sem in series]
                lowerPoints=[(offset,mean-sem) for offset,mean,sem in series]
                meanPoints=[(offset,mean) for offset,mean,_ in series]
                ##register the band's extent for axis auto-ranging, without drawing it yet
                gBox.plot(graph.data.points(upperPoints,x=1,y=2),
                    [graph.style.line([color.transparency(1)])])
                gBox.plot(graph.data.points(lowerPoints,x=1,y=2),
                    [graph.style.line([color.transparency(1)])])
                ##this library's mean line, drawn now (can't plot() after .finish())
                gBox.plot(graph.data.points(meanPoints,x=1,y=2),
                    [graph.style.line([plotColor,style.linewidth.Thick])])
            if countPoints:
                gCount.plot(graph.data.points(countPoints,x=1,y=2),
                    [graph.style.symbol(graph.style.symbol.circle,
                        symbolattrs=[plotColor,deco.filled],size=0.04)])
            perLibSeries.append((plotColor,series))
        ##
        records.append((region,xpos,gBox,gCount,perLibSeries))
    ##
    ##second pass: every region's graphs now have all libraries' data, so it's safe to
    ##finish each box plot and fill in every library's shaded band
    for region,xpos,gBox,gCount,perLibSeries in records:
        gBox.finish()
        for plotColor,series in perLibSeries:
            if series:
                bandPoints=[gBox.pos(offset,mean+sem) for offset,mean,sem in series]+ \
                    [gBox.pos(offset,mean-sem) for offset,mean,sem in reversed(series)]
                bandPath=path.path(path.moveto(*bandPoints[0]),
                    *[path.lineto(*p) for p in bandPoints[1:]],path.closepath())
                gBox.fill(bandPath,[plotColor,color.transparency(0.5)])
        c.insert(gBox)
        c.insert(gCount)
        c.text(xpos+boxSize/2.,boxSize+countHeight+subGap+0.4,region,[text.halign.boxcenter])
    ##
    ##a color-keyed legend, since overlaid bands otherwise have no library label
    legendYpos=-1.8##clears the "Offset" axis titles beneath the box plots
    for ii,libName in enumerate(libNames):
        plotColor=resolve_color(color_map,libName,ii)
        xpos=ii*2.5
        c.fill(path.rect(xpos,legendYpos,0.3,0.3),[plotColor])
        c.text(xpos+0.4,legendYpos+0.15,libName,[text.halign.left,text.valign.middle])
    ##
    return c

def plotAutoCorrResultsTogether(agg,outPrefix,minPointFilter=30,color_map=None):
    """
    plotAutoCorrResultsTogether will make a plot just like plotAutoCorrResults, but instead
    of one plot per each library's TL/CDS/UTR, this function will combine all libraries'
    TLs onto one plot, all libraries' CDSs onto one plot, and all libraries' UTRs onto one
    plot.

    Otherwise mirrors plotAutoCorrResults exactly: mean+/-SEM band per library (now
    overlaid on shared axes, one color per library, with a legend -- see
    buildAutoCorrCanvasTogether), a linked log-scaled site-count plot above each region's
    plot, the same minPointFilter behavior, and the same "all sites" vs. "significant
    sites" pair of outputs (significance via computeSignificantAgg, unchanged), saved as
    outPrefix+'.autocorrTogether.all.pdf' and outPrefix+'.autocorrTogether.significant.pdf'.
    """
    libNames=sorted(agg.keys())
    ##
    ##all sites
    allNormalized=normalizeAllSitesAgg(agg,minPointFilter)
    allCanvas=buildAutoCorrCanvasTogether(allNormalized,libNames,color_map)
    allCanvas.writePDFfile(outPrefix+'.autocorrTogether.all.pdf')
    ##
    ##restricted to sites whose Bonferroni-corrected pval is significant
    sigAgg=computeSignificantAgg(agg)
    sigNormalized=normalizeSignificantAgg(sigAgg,minPointFilter)
    sigCanvas=buildAutoCorrCanvasTogether(sigNormalized,libNames,color_map)
    sigCanvas.writePDFfile(outPrefix+'.autocorrTogether.significant.pdf')

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
    ##stream every read through parsing/gap-filling/region-splitting/autocorrelation and
    ##straight into the aggregate -- see streamAutoCorrForLibs and the module docstring
    print('Streaming parquet files and computing autocorrelation per read...')
    agg=streamAutoCorrForLibs(inFiles,gtfDict,method='gap_fill')
    print('...done.')
    ##
    if colorMapPath:
        unmatched=[lib for lib in agg if lib not in color_map]
        if unmatched:
            print('  WARNING: no color found in %s for librar%s %s; '
                  'falling back to the default palette.'
                  %(colorMapPath,'y' if len(unmatched)==1 else 'ies',unmatched))
    ##
    ##agg is of the format:
    ##{libNamei:{'TL':{offset:[(pearsonR,pearsonPval),...]},'CDS':...,'UTR':...}}
    ##
    print('Plotting results...')
    plotAutoCorrResults(agg,outPrefix,color_map=color_map)
    print('...done.')
    ##
    print('Plotting results all together...')
    plotAutoCorrResultsTogether(agg,outPrefix,color_map=color_map)
    print('...done.')
    ##
    print('Now doing windowing approach...')
    agg=streamAutoCorrForLibs(inFiles,gtfDict,method='window',windowNt=30)
    print('Plotting results...')
    plotAutoCorrResults(agg,outPrefix+'.windowed',color_map=color_map)
    plotAutoCorrResultsTogether(agg,outPrefix+'.windowed',color_map=color_map)
    print('Done.')

if __name__=='__main__':
    Tee()
    main(sys.argv[1:])
