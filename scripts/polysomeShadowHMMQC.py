"""
Liam Tran, July 20, 2026

Script to analyze HMM calls from PS data. Based off of JA's polysomeShadowBinomialBreamQC.py

Input: inFile.gtf - gtf-formatted file containing genome annotations.
        Will use this to build a dict of format:
            {strand:{chr:[absIndx:(txtName,relStart,relStop)]}}
            where relStart is the index relative to the start codon,
            and relStop is the index relative to the stop codon.
            Will also use this to derive cds lengths.
    inFilesParquet.txt - a line-delimited text file of format:
        fileNamei repi parquetFile
        for some number (i) of files, where fileNamei is an identifier for the file,
        and parquetFile is the actual parquetFile. Will compare all fileNamei with
        the same identifier and different repi during reproducibility analyses.
        These parquetFiles are from HMM calls from bayesianShadowClassifier.py. They probably contain all
        the same information from his prior parquetFiles, and also contain shadowcall info.

Output: graphs of the following:
     - relationship between probability cutoff and length, number, and location of shadow calls


run as python3 metaStartStop.py inFile.gtf inFilesParquet.txt outPrefix
"""

import sys, common, polysomeShadowQC, collections
from logJosh import Tee
import pandas as pd
from pyx import *

def lookupRelPositions(gtfDict,strand,chrom,absIdx):
    """
    Given gtfDict of format {strand:{chr:{absIndx:[(txtName,relStart,relStop)]}}},
    returns (relStart,relStop) for absIdx if exactly one transcript is annotated at
    that position. If zero or multiple (overlapping) transcripts are annotated there,
    the site is ambiguous and (None,None) is returned.
    """
    entries=gtfDict.get(strand,{}).get(chrom,{}).get(absIdx,[])
    if len(entries)!=1:
        return None,None
    _,relStart,relStop=entries[0]
    return relStart,relStop

def extractShadows(df,gtfDict,probCutOff,N):
    """
    This function will execute the bulk of Part 2, per the instructions in the header of
    plotPvalVersusLengthNumberAndLocation
    """
    shadowDict=collections.defaultdict(list)
    ##
    halfN=N//2
    for row in df.itertuples(index=False):
        shadowGpos=row.shadow_gpos
        shadowCutOff=row.shadow_P_B
        numShadowSites=len(shadowGpos)
        if numShadowSites==0:
            continue##no candidate shadow sites on this read at any cutoff
        ##
        absIndices=row.absolute_indices
        refSeq=row.ref_sequence_aligned
        numPos=len(absIndices)
        ##
        ##map each genomic position in the read to its index within absIndices/refSeq,
        ##so that shadow_gpos entries (a subset of absolute_indices) can be located
        ##within the full aligned read for sequence extraction and N/2 padding.
        gposToReadIdx={gpos:idx for idx,gpos in enumerate(absIndices)}
        ##
        ##scan the (sparse) shadow sites for consecutive runs above the cutoff
        i=0
        while i<numShadowSites:
            if float(shadowCutOff[i])>probCutOff:
                j=i
                while j<numShadowSites and float(shadowCutOff[j])>probCutOff:
                    j+=1
                ##[i,j) is the contiguous run of significant shadow sites
                firstReadIdx=gposToReadIdx.get(shadowGpos[i])
                lastReadIdx=gposToReadIdx.get(shadowGpos[j-1])
                ##
                if firstReadIdx is None or lastReadIdx is None:
                    i=j
                    continue##couldn't locate this shadow site within the read, skip it
                ##
                startI=max(0,firstReadIdx-halfN)
                endI=min(numPos,lastReadIdx+halfN+1)
                ##
                shadowSeq=''.join(refSeq[startI:endI])
                startIdx=absIndices[startI]
                endIdx=absIndices[endI-1]
                ##
                startRelStart,startRelStop=lookupRelPositions(gtfDict,row.gene_strand,
                    row.chrom,startIdx)
                endRelStart,endRelStop=lookupRelPositions(gtfDict,row.gene_strand,
                    row.chrom,endIdx)
                ##
                shadowDict[row.read_id].append({
                    'shadowSeq':shadowSeq,
                    'startIdx':startIdx,
                    'endIdx':endIdx,
                    'startIdxRelStartOfCDS':startRelStart,
                    'startIdxRelStopOfCDS':startRelStop,
                    'endIdxRelStartOfCDS':endRelStart,
                    'endIdxRelStopOfCDS':endRelStop})
                ##
                i=j
            else:
                i+=1
    ##
    return dict(shadowDict)


def gatherLengthAndCts(tempDict):
    """
    This function will execute the tabulation of shadow sizes and counts, per Part 3
    """
    lengthAndCts = collections.defaultdict(int)
    ##
    for shadowList in tempDict.values():
        for shadow in shadowList:
            length = len(shadow['shadowSeq'])
            lengthAndCts[length] += 1
    ##
    return dict(lengthAndCts)


def gatherPositionCts(tempDict):
    """
    This function will execute the tabulation of shadows as they occur in
    TL/CDS/UTR, per Part 3
    """
    positionCts = {'TL': 0, 'CDS': 0, 'UTR3': 0}
    ##
    for shadowList in tempDict.values():
        for shadow in shadowList:
            startRelStart = shadow['startIdxRelStartOfCDS']
            endRelStart = shadow['endIdxRelStartOfCDS']
            startRelStop = shadow['startIdxRelStopOfCDS']
            endRelStop = shadow['endIdxRelStopOfCDS']
            ##
            if None in (startRelStart, endRelStart, startRelStop, endRelStop):
                continue  ##couldn't place this shadow in the gtfDict, skip it
            ##
            if endRelStart <= -25:
                positionCts['TL'] += 1
            elif startRelStart >= 25 and endRelStop <= -25:
                positionCts['CDS'] += 1
            elif startRelStop >= 25:
                positionCts['UTR3'] += 1
    ##
    return positionCts


def mkLengthAndCountPlot(shadowDict, outPrefix):
    """
    This function will execute Part 4: for each pvalCutoff, plot the number of shadow
    calls of each length, as a vertical array of plots (linked x-axes, x in [30,100]),
    one libraryID per line (same color across graphs), key top right and just outside
    of the array of graphs. The y-axis is log-scaled; lengths with zero shadow calls
    are floored to a small positive value so they can still be plotted on that scale.
    """
    FLOOR = 0.5  ##stand-in for zero counts on the log-scaled y-axis
    lengths = list(range(30, 101))
    ##
    pvalCutoffs = sorted(shadowDict.keys())  ##low (bottom) to high (top)
    libraryIDs = sorted({libraryID for cutoff in shadowDict for libraryID in shadowDict[cutoff]})
    ##
    ##find a common y-axis max across every cutoff/library/length so all graphs share a scale
    maxCt = 0
    for cutoff in pvalCutoffs:
        for libraryID in libraryIDs:
            lengthAndCts = shadowDict[cutoff].get(libraryID, {}).get('lengthAndCts', {})
            for length, ct in lengthAndCts.items():
                if 30 <= length <= 100:
                    maxCt = max(maxCt, ct)
    ymax = maxCt / 0.9 if maxCt > 0 else 1
    ##
    c = canvas.canvas()
    bottomGraph = None
    for ii, cutoff in enumerate(pvalCutoffs):
        isTopGraph = (ii == len(pvalCutoffs) - 1)
        ##
        if bottomGraph is None:
            xAxis = graph.axis.linear(min=30, max=100, title='Shadow length (nt)')
        else:
            xAxis = graph.axis.linkedaxis(bottomGraph.axes['x'], painter=None)
        ##
        graphKwargs = {'width': 12, 'height': 3, 'ypos': ii * 3.5,
                       'x': xAxis,
                       'y': graph.axis.log(min=FLOOR, max=ymax, title=r'Ct, P$_B$ $>$ %s' % cutoff)}
        if isTopGraph:
            graphKwargs['key'] = graph.key.key(pos='tr', hinside=0)
        g = graph.graphxy(**graphKwargs)
        ##
        for libIdx, libraryID in enumerate(libraryIDs):
            lengthAndCts = shadowDict[cutoff].get(libraryID, {}).get('lengthAndCts', {})
            ##every length in the plotted range gets a point; missing/zero-count
            ##lengths are floored so the line stays visible on the log-scaled axis
            lineData = [(length, lengthAndCts.get(length, 0) or FLOOR) for length in lengths]
            g.plot(graph.data.points(lineData, x=1, y=2, title=libraryID),
                   [graph.style.line([common.colors(libIdx)])])
        c.insert(g)
        if bottomGraph is None:
            bottomGraph = g
    ##
    c.writePDFfile(outPrefix)


def mkPositionCountPlot(shadowDict, outPrefix):
    """
    This function will execute Part 5: for each pvalCutoff, plot a nested bar graph
    (TL/CDS/UTR3 bars placed side-by-side within each libraryID's group), vertically
    arrayed as in Part 4. The y-axis is log-scaled; regions with zero shadow calls
    are floored to a small positive value so they can still be plotted on that scale.
    """
    FLOOR = 0.5  ##stand-in for zero counts on the log-scaled y-axis
    regions = ['TL', 'CDS', 'UTR3']
    regionColors = [common.colors(0), common.colors(1), common.colors(2)]
    ##
    pvalCutoffs = sorted(shadowDict.keys())
    libraryIDs = sorted({libraryID for cutoff in shadowDict for libraryID in shadowDict[cutoff]})
    ##
    ##find a common y-axis max across cutoffs/libraries/regions
    maxCt = 0
    for cutoff in pvalCutoffs:
        for libraryID in libraryIDs:
            positionCts = shadowDict[cutoff].get(libraryID, {}).get('positionCts', {})
            for region in regions:
                maxCt = max(maxCt, positionCts.get(region, 0))
    ymax = maxCt / 0.9 if maxCt > 0 else 1
    ##
    c = canvas.canvas()
    bottomGraph = None
    for ii, cutoff in enumerate(pvalCutoffs):
        isTopGraph = (ii == len(pvalCutoffs) - 1)
        ##
        if bottomGraph is None:
            xAxis = graph.axis.nestedbar(title='Library')
        else:
            xAxis = graph.axis.linkedaxis(bottomGraph.axes['x'], painter=None)
        ##
        graphKwargs = {'width': 12, 'height': 3, 'ypos': ii * 3.5,
                       'x': xAxis,
                       'y': graph.axis.log(min=FLOOR, max=ymax, title='Ct, -log10(p)>%s' % cutoff)}
        if isTopGraph:
            graphKwargs['key'] = graph.key.key(pos='tr', hinside=0)
        g = graph.graphxy(**graphKwargs)
        ##
        for regionIdx, region in enumerate(regions):
            barData = []
            for libraryID in libraryIDs:
                positionCts = shadowDict[cutoff].get(libraryID, {}).get('positionCts', {})
                ##zero-count regions are floored so they still render on the log-scaled axis
                barData.append(((libraryID, region), positionCts.get(region, 0) or FLOOR))
            g.plot(graph.data.points(barData, xname=1, y=2, title=region),
                   [graph.style.bar([regionColors[regionIdx]])])
        c.insert(g)
        ##
        if bottomGraph is None:
            bottomGraph = g
    ##
    c.writePDFfile(outPrefix)


def plotPvalVersusLengthNumberAndLocation(gtfDict, parquetFiles, outPrefix, N=30):
    """
    gtfDict is of the format:
    {strand:{chr:{absIndx:[(txtName,relStart,relStop)]}}}

    parquetFiles is of the format:
    fileNamei repi parquetFile
    Will first make a libraryID=fileNamei-repi

    N=30(default) is the window that was used in the shadow calling for pval computation

    Each parquetFile will have the following fields:
    'chrom', 'gene_strand', 'is_reverse', 'transcript_id', 'gene_name',
       'gene_biotype', 'read_id', 'read_start', 'read_end', 'edit_string',
       'barcode', 'bar_seq', 'read_sequence', 'read_sequence_aligned',
       'ref_sequence_aligned', 'aligned_pairs', 'absolute_indices',
       'global_edit_freq', 'n_a_positions', 'shadow_string', 'shadow_tx_pos',
       'shadow_gpos', 'shadow_pval', 'shadow_neg_log10p', 'shadow_edit',
       'shadow_ref_cov', 'n_shadow_sites', 'min_pval'

    Each row of the parquetFile is a read, with each position in that read having some
        shadowCall/statistics. This function will mostly pay attention to 'shadow_neg_log10p',
        which is the negative log_10 of 'shadow_pval'.

        'shadow_gpos' and 'shadow_neg_log10p' are lists of values that contain a shadow_neg_log10p
        value that has passed initial significance threshold. The goal of this function is to explore
        effects of additional significance thresholds. The 'shadow_gpos' values are values that correspond
        to 'absolute_indices', absolute genomic indices.

    The overall goal of this function is to explore the effect different p-value cutoffs (by
        restricting to positions that have shadow_neg_log10p>theCutoff) and analyzing the
        resulting calls. This will happen in several parts:

    Part 1: load the parquetFiles into memory

    Part 2: Combine overlapping shadows according to various pval cutoffs
    For each entry in a hard-coded list of neg_log10p values,
    this function will loop through every read_id in every library. If a consecutive
    group of bases exist after zipping shadow_gpos and shadow_neg_log10p
    with shadow_neg_log10p all above neg_log10p_cutoff, then that consecutive group of bases
    is a contiguous shadow. The nucleotide sequence of the shadow is the sequence of nucleotides
    from ref_sequence_aligned of these bases, which can be obtained by zipping 'absolute_indices' and
    'ref_sequence_aligned', including -N/2 bases upstream from the first nucleotide
    with a significant pval, and +N/2 bases downstream from the last nucleotide with a significant pval.

    At this point, for every pval cutoff there will be an object of the format:
    {libraryID:{read_id:[shadowList]}}
    where each entry in shadowList will be of the format:
    {'shadowSeq': sequence of the nts involved in the call, which will be at least N nts long,
    'startIdx':first nt of the call in genomic space,
     'endIdx':last nt of the call in genomic space,
      'startIdxRelStartOfCDS':first nt of the call in relStart space,
       'startIdxRelStopOfCDS':first nt of the call in relStop space,
       'endIdxRelStartOfCDS':last nt of the call in relStart space,
       'endIdxRelStopOfCDS':last nt of the call in relStop space}
    To obtain the information for startIdxRelStartOfCDS, startIdxRelStopOfCDS, endIdxRelStartOfCDS,
        and endIdxRelStopOfCDS, look up the relevant positions in the gtfDict.

    Within the for loop containing each pval cutoff, the script will then gather summary data
    in Part 3.

    Part 3: Gather the following summary data about every shadowList entry:
     - the number and length of every shadow call, in format:
        lengthAndCts={libraryID:{length:ct}}
     - the number of shadow calls in each region of the gene:
        positionCts={libraryID:{'TL':ct1,'CDS':ct2,'UTR3':ct3}}
        where: TL is endIdxRelStartOfCDS in range (-inf,-25]
                CDS is startIdxRelStartOfCDS>=25 and endIdxRelStopOfCDS<=-25
                UTR3 is startIdxRelStopOfCDS in range [+25,+inf)

    At this point there will be one large object--a dataframe or a nested dict--with information:
        {pvalCutoff:{'lengthAndCts':lengthAndCts,'positionCts':positionCts}}
    These data will be plotted in Part 4 & Part 5

    Part 4: Plot the length and cts
    For each pvalCutoff, plot the number of shadow calls of each length. Do this as a vertical array
        of plots with linked x-axes. The x-axes contains the length, and ranges from [30,100].
        The y-axis contains the counts. Keep all y-axes on the same scale. Array the plots vertically
        from low pval cutoff (less significant) on the bottom to higher pval cutoff (more significant)
        near the top. Within each graph, have a different libraryID as a single line, and keep those
        lines the same colors across all graphs. Show the key to the top right and just outside the
        array of graphs.

    Part 5: Plot where shadow calls are
    For each pvalCutoff, for each libraryID, plot the number of shadow calls in each of TL/CDS/UTR.
        Do this as a stacked bar graph where each bar is a different libraryID, and TL/CDS/UTR
        are different stacks on the same bar. Do formatting as with Part 4, but now the x-axes will
        be libraryID rather than length. As with Part 4, vertically array different p-value cutoffs
        and keep the y-axes of the individal plots on the same scale.

    """
    ##list pvalue cutoffs
    probCutOffs = [0.5, 0.6, 0.7, 0.8, 0.9]
    ##initialize dictionary
    ##will be of the format:
    ##{pvalCutoff:{libraryID:{readID:[shadowList]}}}
    shadowDict = collections.defaultdict(lambda: collections.defaultdict(dict))
    ##
    with open(parquetFiles, 'r') as f:
        for line in f:
            line = line.strip().split()
            fileName, repName, parquetFile = line[0:]
            libraryID = '%s-%s' % (fileName, repName)
            ##
            ##Part 1: Load files into memory
            print('Analyzing parquetFile %s...' % (parquetFile))
            df = pd.read_parquet(parquetFile)
            ##
            ##Part 2: Combine overlapping shadows according to various pval cutoffs
            for cutoff in probCutOffs:
                print('Working on probCutOff: %s...' % (cutoff))
                tempDict = extractShadows(df, gtfDict, cutoff, N)
                ##
                ##Part 3: Gather summary data about every shadowList entry:
                ##lengthAndCts
                lengthAndCts = gatherLengthAndCts(tempDict)
                ##positionCts
                positionCts = gatherPositionCts(tempDict)
                ##
                ##now save that data
                shadowDict[cutoff][libraryID] = {'lengthAndCts': lengthAndCts,
                                                  'positionCts': positionCts}
    ##
    ##Part 1-3 complete, now do Part 4
    mkLengthAndCountPlot(shadowDict, outPrefix + '.lengthAndCts')
    ##
    ##Part 4 complete, now do Part 5
    mkPositionCountPlot(shadowDict, outPrefix + '.positionCts')

def main(args):
    gtfFile, parquetFiles, outPrefix = args[0:]
    ##
    ##first parse the gtf file to a dictionary
    gtfDict, cdsLengths = polysomeShadowQC.parseGTF(gtfFile)
    ##gtfDict is of the format:
    ##{strand:{chr:{absIndx:[(txtName,relStart,relStop)]}}}
    ##
    ##now analyze the effect of p-value cutoffs on the footprints
    plotPvalVersusLengthNumberAndLocation(gtfDict, parquetFiles, outPrefix, 50)

if __name__ == '__main__':
    Tee()
    main(sys.argv[1:])