"""
Joshua Arribere, July 1, 2026

Script to analyze:
 - per library/parquet: edit bias per 3nt motifs
 - per library: aligned read length distribution
 - relationship between minimum read count and edit frequency reproducibility
 - relationship between minimum A count and edit frequency reproducibility
 - reproducibility of edit frequencies across replicates at gene level
 - metaStart/Stop distribution of edit frequencies
 - relationship between edit frequency and cds length
 - correlation between edit frequency and Ribo-seq/RNA-seq
 - interedit distance, and the same for random shufflings of edit strings

Input: inFile.gtf - gtf-formatted file containing genome annotations.
        Will use this to build a dict of format:
            {strand:{chr:[absIndx:(txtName,relStart,relStop)]}}
            where relStart is the index relative to the start codon,
            and relStop is the index relative to the stop codon.
            Will also use this to derive cds lengths.
    N - the minimum number of reads per transcript_id to be included in the analysis.
    inFilesParquet.txt - a line-delimited text file of format:
        fileNameiRepj fileij1.parquet fileij2.parquet ...fileijN.parquet
        for some number (N) of filesij1-ijN. fileNameiRepj is a single token with NO
        separator between the fileNamei and Repj portions -- the replicate portion is
        identified by the last occurrence of the literal substring "Rep" within the
        token (e.g. "SampleARep1" -> fileNamei="SampleA", repj="Rep1"). If "Rep" does not
        occur anywhere in the token, the whole token is fileNamei and there is no
        replicate. Will compare all fileNamei with the same identifier and different
        Repj during reproducibility analyses. Unlike some prior formats, the parquet
        files are named explicitly on the line -- nothing is globbed from a directory.
    inFile.Ribo.txt - of the format:
        YORFi\tValuei
    inFile.RNA.txt - of the format:
        YORFi\tValuei
    colors.txt - OPTIONAL. A line-delimited text file of format:
        libNamei\thexCodei
        for some number of libNamei, with no header row. libNamei can be written as EITHER
        the raw token used in inFilesParquet.txt (e.g. "minus3ATPolyRep1") OR the dashed
        label most plots actually display (e.g. "minus3ATPoly-Rep1") -- assignColors treats
        those as equivalent. A bare fileName with no Rep suffix at all (e.g. "minus3ATPoly")
        is also honored by readCountAndEditFreqRepro's two replicate-pooling plots; if no
        such bare entry is given but every one of that fileName's Rep-suffixed entries agrees
        on the same hexCode, that shared color is used for the pooled plot too. Libraries not
        covered by any of the above (or if this file is omitted entirely) are auto-assigned
        distinct colors instead (see assignColors/distinctColors).

Output: Graphs saved in outPrefix.[identifier].pdf

run as python3 polysomeShadowQC.py inFile.gtf N minA inFilesParquet.txt inFile.Ribo.txt inFile.RNA.txt outPrefix [colors.txt]
"""
import collections, sys, math, itertools, pickle, random
import matplotlib
from scipy.stats import spearmanr, linregress
from logJosh import Tee
import pandas as pd
from pyx import *

def parseInFilesParquet(parquetFiles):
    """
    parquetFiles is a line-delimited text file where each line is of format:
        fileNameiRepj fileij1.parquet fileij2.parquet ... fileijN.parquet
    fileNameiRepj is a single token with NO separator between the fileNamei and Repj
    portions. The replicate portion is identified by the LAST occurrence of the literal
    substring "Rep" within the token: everything before it is fileNamei, and "Rep" plus
    everything after it is repj (e.g. "SampleARep1" -> fileName="SampleA", rep="Rep1").
    If "Rep" does not occur anywhere in the token, the whole token is fileName and
    rep=None -- that entry has no replicate, so it will not be grouped with any other
    entry during reproducibility comparisons that key off of (fileName,rep).

    Caveat (by design, per JA): this is a bare substring match, not a "Rep" *marker* --
    a fileName that happens to contain "Rep" as part of an ordinary word (e.g.
    "ControlNoReplicate", "ReportedSample") will be mis-split there too. Avoid "Rep" in
    fileName portions except as an intentional replicate marker.

    Unlike older formats that named a directory to glob('*.parquet') within, the files
    following the identifier token are the parquet files themselves, used directly (in
    the order listed).

    Returns a list of (fileName,rep,parquetFilePaths) tuples, one per non-blank line. A
    line naming an identifier but no parquet files is skipped (with a warning), since
    there would be nothing to analyze for it.
    """
    entries=[]
    with open(parquetFiles,'r') as f:
        for line in f:
            parts=line.strip().split()
            if not parts:
                continue##skips blank lines
            token=parts[0]
            filePaths=parts[1:]
            if not filePaths:
                print('No parquet files listed for %s; skipping.'%(token))
                continue
            repIdx=token.rfind('Rep')
            if repIdx==-1:
                fileName,rep=token,None
            else:
                fileName,rep=token[:repIdx],token[repIdx:]
            entries.append((fileName,rep,filePaths))
    return entries

def labelForEntry(fileName,rep):
    """
    Builds the display label for a (fileName,rep) pair, as returned by
    parseInFilesParquet: 'fileName-rep' if there's a replicate, else just fileName.
    """
    return '%s-%s'%(fileName,rep) if rep is not None else fileName

def analyzeMotifs(parquetFiles,outPrefix):
    """
    parquetFiles is a file of the format described in parseInFilesParquet's docstring.
    This function will loop through all reads in all parquet files and analyze the frequency of edits
    at the central position of each A-containing motif. Will save a pdf of the results.

    Returns a dict of format {libName:{seq:freq}}, where libName is the fileName-rep label
    (see labelForEntry) and seq is a 3nt motif (e.g. 'CAT'). Every libName's inner dict has
    the same set of seq keys -- the union of motifs observed across ALL libraries -- so a
    motif never observed within a given library is present there with freq=0.0 rather than
    being absent (matching how the bar charts above already zero-pad missing motifs).

    Function written by Claude, inspected and edited by JA.
    """
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

    # {label: {motif: [edit_ints]}}
    motifDict = collections.defaultdict(lambda: collections.defaultdict(list))

    for fileName,rep,pFiles in parseInFilesParquet(parquetFiles):
        label = labelForEntry(fileName,rep)
        ##
        for pFileIndex,pFile in enumerate(pFiles):
            ##JA added the next two lines to expedite testing. Remove for full analysis.
            #if pFileIndex!=0:
            #    continue
            ##
            print('\nAnalyzing %s for motif analysis...' % pFile)
            df = pd.read_parquet(pFile,columns=['gene_strand','ref_sequence_aligned','edit_string'])
            ##
            #df=df[:1000]##JA added this line to subset the data for testing purposes. Remove for full analysis.
            ##
            for row in df.itertuples(index=False):
                strand = row.gene_strand
                refSeq = row.ref_sequence_aligned
                editStr = row.edit_string

                for i, (refNt, edit) in enumerate(zip(refSeq, editStr)):
                    ##
                    edit = int(edit)
                    ##
                    if edit not in [0, 1]:
                        continue##skips 2, which is a gap
                    ##
                    hasA = (refNt == 'A')##Liam should have made stranded parquets
                    ##
                    if edit==1 and not hasA:
                        ##JA added this check since we've been changing strand formatting.
                        print('Error: found an "edit" at a non-A position in motif analysis. This is unexpected and might indicate an issue with the input data.')
                        print(strand)
                        print(refSeq)
                        print(editStr)
                        print('Exiting...')
                        sys.exit()
                    ##
                    if not hasA or i == 0 or i == len(refSeq) - 1:##avoid the first and last and non-A positions.
                        continue
                    ##
                    prevNt = refSeq[i - 1]
                    prevEdit=int(editStr[i - 1])
                    nextNt = refSeq[i + 1]
                    nextEdit=int(editStr[i + 1])
                    ##
                    if prevEdit not in [0, 1] or nextEdit not in [0, 1]:
                        continue##skips 2, which is a gap
                    ##
                    motif = prevNt + 'A' + nextNt
                    ##
                    if all(nt in 'ACGT' for nt in motif):
                        motifDict[label][motif].append(edit)

    allMotifs = sorted({m for ld in motifDict.values() for m in ld})
    libColors = assignColors(motifDict.keys())

    c = canvas.canvas()
    for libIdx, label in enumerate(sorted(motifDict.keys())):
        ld = motifDict[label]
        barData = [(motif, sum(ld[motif]) / len(ld[motif]) if ld[motif] else 0)
                   for motif in allMotifs]

        g = graph.graphxy(
            width=16, height=6,
            ypos=libIdx * 7,
            x=graph.axis.bar(),
            y=graph.axis.linear(min=0, max=1, title='Edit Frequency (%s)' % label)
        )
        g.plot(graph.data.points(barData, xname=1, y=2),
               [graph.style.bar([libColors[label]])])
        c.insert(g)

    c.writePDFfile(outPrefix + '.motifs.pdf')
    print('Motif analysis completed and saved to %s.motifs.pdf' % outPrefix)
    ##
    ##now plot all the motifs together for comparison across libraries.
    ##do this as a nested bar graph: each group of bars is a motif,
    ##each bar within a group is a different library.
    maxFreq = max(
        sum(motifDict[label][motif]) / len(motifDict[label][motif])
        for label in motifDict
        for motif in motifDict[label]
        if motifDict[label][motif]
    )
    ymax = maxFreq / 0.9
    c = canvas.canvas()
    g = graph.graphxy(
        width=16, height=6,
        x=graph.axis.nestedbar(),
        y=graph.axis.linear(min=0, max=ymax, title='Edit Frequency'),
        key=graph.key.key(pos='mr', hinside=0)
    )
    for libIdx, label in enumerate(sorted(motifDict.keys())):
        barData = [((motif, label),
                    sum(motifDict[label][motif]) / len(motifDict[label][motif]) if motifDict[label][motif] else 0)
                   for motif in allMotifs]
        g.plot(graph.data.points(barData, xname=1, y=2, title=label),
               [graph.style.bar([libColors[label]])])
    c.insert(g)
    c.writePDFfile(outPrefix + '.motifs_comparison.pdf')
    print('Motif comparison across libraries completed and saved to %s.motifs_comparison.pdf' % outPrefix)
    ##
    ##now repeat, but normalizing by the mean edit frequency across all motifs for each library.
    ##y-axis is log2(motifFreq / overallFreq) = fold-over-edit.
    import math
    overallFreq = {}
    for label in motifDict:
        allEdits = [e for motif in motifDict[label] for e in motifDict[label][motif]]
        overallFreq[label] = sum(allEdits) / len(allEdits) if allEdits else 0
    ##build log2 fold-over-edit values
    log2Data = {}
    for label in motifDict:
        log2Data[label] = {}
        for motif in allMotifs:
            edits = motifDict[label][motif]
            motifFreq = sum(edits) / len(edits) if edits else 0
            if motifFreq > 0 and overallFreq[label] > 0:
                log2Data[label][motif] = math.log2(motifFreq / overallFreq[label])
            else:
                log2Data[label][motif] = 0
    ##scale y-axis so the extreme bars sit at ~90% of the axis range
    allLog2Vals = [v for ld in log2Data.values() for v in ld.values()]
    ymax = max(allLog2Vals) / 0.9
    ymin = min(allLog2Vals) / 0.9 if min(allLog2Vals) < 0 else 0
    c = canvas.canvas()
    g = graph.graphxy(
        width=16, height=6,
        x=graph.axis.nestedbar(),
        y=graph.axis.linear(min=ymin, max=ymax, title='log2(Fold-over-edit)'),
        key=graph.key.key(pos='mr', hinside=0)
    )
    for libIdx, label in enumerate(sorted(motifDict.keys())):
        barData = [((motif, label), log2Data[label][motif]) for motif in allMotifs]
        g.plot(graph.data.points(barData, xname=1, y=2, title=label),
               [graph.style.bar([libColors[label]])])
    c.insert(g)
    c.writePDFfile(outPrefix + '.motifs_normalized.pdf')
    print('Normalized motif analysis completed and saved to %s.motifs_normalized.pdf' % outPrefix)
    ##
    ##build the {libName:{seq:freq}} dict to return, zero-padded to allMotifs so every
    ##library has the same set of motif keys.
    freqDict = {}
    for label in motifDict:
        ld = motifDict[label]
        freqDict[label] = {motif: (sum(ld[motif]) / len(ld[motif]) if ld[motif] else 0.0)
                            for motif in allMotifs}
    return freqDict

def readCountAndEditFreqRepro(parquetFiles,N,outPrefix):
    """
    parquetFiles is a file of the format described in parseInFilesParquet's docstring.
    This function will loop through all reads in all parquet files and first build a dict or dataframe
    containing the following information on each read:
        txtID
        global_edit_freq (taken directly from the parquet file)
        n_a_positions (taken directly from the parquet file)
        fileNamei
        repi
    For parquet files with the same fileNamei, this function will then analyze the correlation (spearmanR)
    of global_edit_freq between replicates (repi) for different cutoffs of minimum reads per transcript_id
    and minimum A's per transcript_id. It will make plots for each, where the x-axis is the minimum read
    number or minimum A number, and the y-axis is the spearmanR between replicates.
    If a given library does not have multiple replicates, it will be skipped in this analysis.
    Will save a pdf of the results, with different libraries plotted in different colors but on the same graph.

    Function written by Claude, inspected and edited by JA
    """
    ##first, extract the relevant information from the parquet files:
    ##txtID, global_edit_freq, n_a_positions, fileNamei, repi
    ##save those in a dataframe for easy analysis.
    rows = []
    for fileName,rep,pFiles in parseInFilesParquet(parquetFiles):
        for parquetFileIndex, pFile in enumerate(pFiles):
            ##
            #if parquetFileIndex != 0:
            #    continue##JA added this line to expedite testing. Remove for full analysis.
            ##
            print('\nAnalyzing %s for reproducibility analysis...' % pFile)
            df = pd.read_parquet(pFile,columns=['transcript_id','global_edit_freq','n_a_positions'])
            ##
            for row in df.itertuples(index=False):
                rows.append({
                    'txtID': row.transcript_id,
                    'global_edit_freq': row.global_edit_freq,
                    'n_a_positions': row.n_a_positions,
                    'fileName': fileName,
                    'rep': rep
                })
    ##
    allData = pd.DataFrame(rows)
    ##aggregate to transcript level: count reads, sum A positions, mean edit freq
    txData = allData.groupby(['fileName', 'rep', 'txtID']).agg(
        readCt=('global_edit_freq', 'count'),
        totalA=('n_a_positions', 'sum'),
        meanEditFreq=('global_edit_freq', 'mean')
    ).reset_index()
    ##
    ##now compute the spearmanR between replicates for different cutoffs of minimum reads per
    ##transcript_id and minimum A's per transcript_id.
    readCutoffs=list(range(1,101))+list(range(101,251,5))+list(range(251,1001,20))
    aCutoffs=list(range(1,101))+list(range(101,251,5))+list(range(251,1001,20))
    ##reproDict: {fileName: {'readCt': {cutoff: spearmanR}, 'totalA': {cutoff: spearmanR}}}
    reproDict = {}
    for fn in txData['fileName'].unique():##fn is abbreviation for filename
        fnData = txData[txData['fileName'] == fn]
        reps = fnData['rep'].unique()
        ##
        if len(reps) < 2:
            print('%s has only one replicate; skipping reproducibility analysis.' % fn)
            continue
        ##
        reproDict[fn] = {'readCt': {}, 'totalA': {}}
        repPairs = list(itertools.combinations(reps, 2))
        ##for readCt: transcript-level frames indexed by txtID (readCt and meanEditFreq already aggregated)
        repFrames  = {r: fnData[fnData['rep'] == r].set_index('txtID') for r in reps}
        ##for aCount: raw per-read data per rep, so we can re-aggregate after filtering by n_a_positions
        fnRawData  = allData[allData['fileName'] == fn]
        repRawData = {r: fnRawData[fnRawData['rep'] == r] for r in reps}
        ##readCt sweep: for each rep, keep transcripts with >= cutoff reads and use their
        ##per-transcript avg edit freq (meanEditFreq). Compare between reps on shared transcripts.
        for cutoff in readCutoffs:
            pairRs = []
            for r1, r2 in repPairs:
                r1tx = repFrames[r1][repFrames[r1]['readCt'] >= cutoff]['meanEditFreq']
                r2tx = repFrames[r2][repFrames[r2]['readCt'] >= cutoff]['meanEditFreq']
                common2 = r1tx.index.intersection(r2tx.index)
                if len(common2) > 2:
                    rval, _ = spearmanr(r1tx.loc[common2], r2tx.loc[common2])
                    pairRs.append(rval)
            reproDict[fn]['readCt'][cutoff] = (sum(pairRs) / len(pairRs)
                                               if pairRs else float('nan'))
        ##aCount sweep: for each rep, filter reads to those with n_a_positions >= cutoff,
        ##then compute avg global_edit_freq per transcript from those reads.
        ##Compare between reps on shared transcripts.
        for cutoff in aCutoffs:
            pairRs = []
            for r1, r2 in repPairs:
                r1tx = (repRawData[r1][repRawData[r1]['n_a_positions'] >= cutoff]
                        .groupby('txtID')['global_edit_freq'].mean())
                r2tx = (repRawData[r2][repRawData[r2]['n_a_positions'] >= cutoff]
                        .groupby('txtID')['global_edit_freq'].mean())
                common2 = r1tx.index.intersection(r2tx.index)
                if len(common2) > 2:
                    rval, _ = spearmanr(r1tx.loc[common2], r2tx.loc[common2])
                    pairRs.append(rval)
            reproDict[fn]['totalA'][cutoff] = (sum(pairRs) / len(pairRs)
                                               if pairRs else float('nan'))
    ##
    ##now plot the data
    g1 = graph.graphxy(
        width=8, height=8,
        x=graph.axis.log(title='Minimum Read Count per Transcript'),
        y=graph.axis.linear(min=0, max=1, title='SpearmanR (Edit Frequency)'),
        key=graph.key.key(pos='br', hinside=1)
    )
    g2 = graph.graphxy(
        width=8, height=8,
        xpos=g1.width * 1.1,
        x=graph.axis.log(title='Minimum A Count per Read'),
        y=graph.axis.linkedaxis(g1.axes['y'])
    )
    libColors = assignColors(reproDict.keys())
    for fn in sorted(reproDict.keys()):
        for g, cutoffType, cutoffs in [(g1, 'readCt', readCutoffs), (g2, 'totalA', aCutoffs)]:
            data = [(c, reproDict[fn][cutoffType][c])
                    for c in cutoffs
                    if not math.isnan(reproDict[fn][cutoffType][c])]
            if data:
                g.plot(graph.data.points(data, x=1, y=2, title=fn),
                       [graph.style.line([libColors[fn]]),
                        graph.style.symbol(graph.style.symbol.circle, size=0.1,
                                           symbolattrs=[libColors[fn]])])
    c = canvas.canvas()
    c.insert(g1)
    c.insert(g2)
    c.writePDFfile(outPrefix + '.readCountEditFreqRepro.pdf')
    print('Reproducibility analysis saved to %s.readCountEditFreqRepro.pdf' % outPrefix)
    ##
    ##Now analyze the reproducibility of various a_count_cutoffs for transcript_ids with at least N
    ##(N=user input) reads. Basically re-create the right graph from above, but with txtIDs filtered to
    ##those with at least N reads.
    reproDictN = {}
    for fn in txData['fileName'].unique():
        fnData = txData[txData['fileName'] == fn]
        reps   = fnData['rep'].unique()
        if len(reps) < 2:
            continue
        reproDictN[fn] = {}
        repPairsN   = list(itertools.combinations(reps, 2))
        repFramesN  = {r: fnData[fnData['rep'] == r].set_index('txtID') for r in reps}
        fnRawDataN  = allData[allData['fileName'] == fn]
        repRawDataN = {r: fnRawDataN[fnRawDataN['rep'] == r] for r in reps}
        for cutoff in aCutoffs:
            pairRs = []
            for r1, r2 in repPairsN:
                ##find transcripts with >= N reads in both reps
                validTxts = (repFramesN[r1][repFramesN[r1]['readCt'] >= N].index
                             .intersection(repFramesN[r2][repFramesN[r2]['readCt'] >= N].index))
                ##filter reads: n_a_positions >= cutoff AND txtID in that valid set
                r1tx = (repRawDataN[r1][(repRawDataN[r1]['n_a_positions'] >= cutoff) &
                                        (repRawDataN[r1]['txtID'].isin(validTxts))]
                        .groupby('txtID')['global_edit_freq'].mean())
                r2tx = (repRawDataN[r2][(repRawDataN[r2]['n_a_positions'] >= cutoff) &
                                        (repRawDataN[r2]['txtID'].isin(validTxts))]
                        .groupby('txtID')['global_edit_freq'].mean())
                common2 = r1tx.index.intersection(r2tx.index)
                if len(common2) > 2:
                    rval, _ = spearmanr(r1tx.loc[common2], r2tx.loc[common2])
                    pairRs.append(rval)
            reproDictN[fn][cutoff] = (sum(pairRs) / len(pairRs) if pairRs else float('nan'))
    g3 = graph.graphxy(
        width=8, height=8,
        x=graph.axis.log(title='Minimum A Count per Read'),
        y=graph.axis.linear(min=0, max=1,
                            title='SpearmanR (Edit Freq, min %d reads)' % N),
        key=graph.key.key(pos='br', hinside=1)
    )
    libColorsN = assignColors(reproDictN.keys())
    for fn in sorted(reproDictN.keys()):
        data = [(c, reproDictN[fn][c])
                for c in aCutoffs
                if not math.isnan(reproDictN[fn][c])]
        if data:
            g3.plot(graph.data.points(data, x=1, y=2, title=fn),
                    [graph.style.line([libColorsN[fn]]),
                     graph.style.symbol(graph.style.symbol.circle, size=0.1,
                                        symbolattrs=[libColorsN[fn]])])
    c3 = canvas.canvas()
    c3.insert(g3)
    c3.writePDFfile(outPrefix + '.aCountRepro_minReads%d.pdf' % N)
    print('A count reproducibility (min %d reads) saved to %s.aCountRepro_minReads%d.pdf'
          % (N, outPrefix, N))
    ##
    ##now restrict to transcripts with at least N reads in each library.
    ##make a heatmap of the spearmanR of the average edit frequency between all pairs of libraries.
    ##compute per-library avg edit freq per transcript, filtered to >= N reads
    perLibFreqs = {}
    for (fn, rep), group in txData.groupby(['fileName', 'rep']):
        label = '%s-%s' % (fn, rep)
        perLibFreqs[label] = group[group['readCt'] >= N].set_index('txtID')['meanEditFreq']
    ##compute all pairwise SpearmanR values
    libLabels = sorted(perLibFreqs.keys())
    nLibs = len(libLabels)
    spearmanMatrix = {}
    for lib1 in libLabels:
        spearmanMatrix[lib1] = {}
        for lib2 in libLabels:
            common2 = perLibFreqs[lib1].index.intersection(perLibFreqs[lib2].index)
            if len(common2) > 2:
                rval, _ = spearmanr(perLibFreqs[lib1].loc[common2],
                                    perLibFreqs[lib2].loc[common2])
            else:
                rval = float('nan')
            spearmanMatrix[lib1][lib2] = rval
    ##draw heatmap: cividis, dark (min SpearmanR, least correlated) to yellow (max SpearmanR,
    ##most correlated). scale to actual data range so the full color range is always used.
    ##the diagonal (a library compared to itself) is always SpearmanR=1 by construction and
    ##isn't informative, so it's excluded from the color scale (which instead tops out at the
    ##next-highest, off-diagonal value) and censored (drawn gray, like a NaN cell, with no
    ##value printed) rather than shown as a spurious "perfect correlation".
    allRvals = [spearmanMatrix[l1][l2]
                for l1 in libLabels for l2 in libLabels
                if l1 != l2 and not math.isnan(spearmanMatrix[l1][l2])]
    rMin = min(allRvals)
    rMax = max(allRvals)
    def r_to_color(r):
        if math.isnan(r):
            return color.rgb(0.7, 0.7, 0.7)
        t = (max(rMin, min(rMax, r)) - rMin) / (rMax - rMin) if rMax > rMin else 0.0
        return cividisColor(t)
    cellSize = 1.5
    cHeat = canvas.canvas()
    for i, lib1 in enumerate(libLabels):
        for j, lib2 in enumerate(libLabels):
            rval = spearmanMatrix[lib1][lib2]
            x = j * cellSize
            y = (nLibs - 1 - i) * cellSize
            rect = path.rect(x, y, cellSize, cellSize)
            if lib1 == lib2:
                ##censored self-comparison -- always r=1, not informative.
                cHeat.fill(rect, [color.rgb(0.7, 0.7, 0.7)])
                cHeat.stroke(rect, [style.linewidth(0.01)])
                continue
            cHeat.fill(rect, [r_to_color(rval)])
            cHeat.stroke(rect, [style.linewidth(0.01)])
            if not math.isnan(rval):
                cHeat.text(x + cellSize / 2, y + cellSize / 2, '%.2f' % rval,
                           [text.halign.center, text.valign.middle])
    ##x-axis labels (below the grid, rotated 45°)
    for j, lib in enumerate(libLabels):
        cHeat.text(j * cellSize + cellSize / 2, -0.2, lib,
                   [text.halign.right, text.valign.middle, trafo.rotate(45)])
    ##y-axis labels (left of the grid)
    for i, lib in enumerate(libLabels):
        cHeat.text(-0.1, (nLibs - 1 - i) * cellSize + cellSize / 2, lib,
                   [text.halign.right, text.valign.middle])
    ##colorbar: 100 thin strips spanning actual [rMin, rMax]
    barX   = nLibs * cellSize + 0.8
    barW   = 0.5
    barH   = nLibs * cellSize
    nSteps = 100
    stepH  = barH / nSteps
    for k in range(nSteps):
        t = k / (nSteps - 1)
        cHeat.fill(path.rect(barX, k * stepH, barW, stepH + 0.01), [cividisColor(t)])
    cHeat.stroke(path.rect(barX, 0, barW, barH), [style.linewidth(0.02)])
    ##tick marks at 5 evenly spaced values across the actual data range
    for rTick in [rMin + i * (rMax - rMin) / 4 for i in range(5)]:
        yTick = ((rTick - rMin) / (rMax - rMin)) * barH
        cHeat.stroke(path.line(barX + barW, yTick, barX + barW + 0.2, yTick),
                     [style.linewidth(0.02)])
        cHeat.text(barX + barW + 0.3, yTick, '%.2f' % rTick,
                   [text.halign.left, text.valign.middle])
    ##colorbar title above the bar
    cHeat.text(barX + barW / 2, barH + 0.3, 'SpearmanR',
               [text.halign.center, text.valign.bottom])
    cHeat.writePDFfile(outPrefix + '.editFreqReproHeatmap.pdf')
    print('Heatmap saved to %s.editFreqReproHeatmap.pdf' % outPrefix)

def parseGTF(gtfFile):
    """
    Parse a gtf file to get dict of format:
    {strand:{chr:{absIndx:(txtName,relStart,relStop)]}}}
    where relStart is the index relative to the start codon,
    and relStop is the index relative to the stop codon.

    Will also derive cds lengths.
    """
    ##
    print('\nrelStart=0 is the A of the ATG.')
    print('relStop=0 is the T of the TAA/TAG/TGA.\n')
    ##
    gtfDict = {'+': {}, '-': {}}
    ##
    with open(gtfFile, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            chrom, source, feature_type, start, end, score, strand, frame, attributes = fields
            if feature_type != 'CDS':
                continue
            #if the script made it thus far, it's found a line that's a CDS.

            # Extract txtID from attributes
            transcript_id = None
            for attr in attributes.split(';'):
                if attr.strip().startswith('transcript_id'):
                    transcript_id = attr.split('"')[1]
                    break
            
            if transcript_id is None:
                continue
            
            ##now check that transcript_biotype is "protein_coding"
            biotype = None
            for attr in attributes.split(';'):
                if attr.strip().startswith('transcript_biotype'):
                    biotype = attr.split('"')[1]
                    break

            if biotype != "protein_coding":
                continue

            start, end = int(start), int(end)
            #abs_index = (start + end) // 2  # Approximate center of CDS
            
            if chrom not in gtfDict[strand]:
                gtfDict[strand][chrom] = {}
            
            if transcript_id not in gtfDict[strand][chrom]:
                gtfDict[strand][chrom][transcript_id]=[]
            gtfDict[strand][chrom][transcript_id].append((start,end))
    
    ##gtfDict is now of format {strand:{chr:{txtID:[(start,end)]}}}
    ##where there may be multiple (start,end) pairs for a given txtID
    ##if there are multiple exons.
    #print(gtfDict['+'])
    txtCounter=0
    positionCounter=0
    bb={'+':{},'-':{}}
    for strand in gtfDict:
        for chrom in gtfDict[strand]:
            bb[strand][chrom]=collections.defaultdict(list)
            for txtID in gtfDict[strand][chrom]:
                txtCounter+=1
                ##
                if txtCounter%1000==0:
                    print('Placed %d transcripts from gtf file in a genomic position dict.'%(txtCounter))
                ##
                relStart=0
                relStop=-sum([end-start+1 for start,end in gtfDict[strand][chrom][txtID]])
                ##add these two lines to start 100nts upstream of the start codon.
                relStart-=100
                relStop-=100
                #print(relStart,relStop)
                #print(gtfDict[strand][chrom][txtID])
                exons=gtfDict[strand][chrom][txtID]
                exons.sort(key=lambda x:x[0]) #sort by start position, which will ensure that the first exon is the one with the start codon, and the last exon is the
                ##these next lines were for restricting to multi-exon transcripts.
                #if len(exons)==1:
                #    counter-=1
                #    continue
                ##
                if strand=='+':
                    for ii,exon in enumerate(exons):
                        start, end = exon
                        if ii==0:##then it's the first exon
                            start-=100
                        if ii==len(exons)-1:##then it's the last exon
                            end+=100

                        for ii in range(start,end+1):
                            ##
                            bb[strand][chrom][ii-1].append([txtID,relStart,relStop])
                            ##leave for debugging.
                            #if len(exons)==1 and chrom!='Mito':
                            #    print(strand,chrom,ii,txtID,relStart,relStop,len(exons))
                            ##
                            relStart+=1
                            relStop+=1
                            positionCounter+=1
                elif strand=='-':
                    for ii,exon in enumerate(reversed(exons)):
                        start, end = exon
                        if ii==0:
                            end+=100
                        if ii==len(exons)-1:
                            start-=100
                        for ii in range(end,start-1,-1):
                            ##
                            bb[strand][chrom][ii-1].append([txtID,relStart,relStop])
                            ##leave for debugging.
                            #if len(exons)==1 and chrom!='Mito':
                            #    print(strand,chrom,ii,txtID,relStart,relStop,len(exons))
                            ##
                            relStart+=1
                            relStop+=1
                            positionCounter+=1
    ##
    ##print(bb)
    ##bb is now of format {strand:{chr:{absIndx:[(txtID,relStart,relStop)]}}}
    print('\nPlaced %d transcripts from gtf file into genomic position dict.'%(txtCounter))
    print('Placed %s positions.\n'%(positionCounter))

    ##bb is now of format {strand:{chr:{absIndx:[(txtID,relStart,relStop)]}}}
    ##now derive cds lengths, and check that the distance between relStart and relStop is a constant.
    ##if the distance between relStart and relStop is not a constant, then print an error and exit.
    cc=collections.defaultdict(list)
    for strand in bb:
        for chrom in bb[strand]:
            for absIndx in bb[strand][chrom]:
                for txtID,relStart,relStop in bb[strand][chrom][absIndx]:
                    cc[txtID].append(relStop-relStart)
    ##
    cdsLengths={}
    for txtID in cc:
        if len(set(cc[txtID]))>1:
            print('Error: transcript %s has multiple cds lengths: %s'%(txtID,cc[txtID]))
            sys.exit()
        else:
            cdsLengths[txtID]=cc[txtID][0]
    ##
    return bb, cdsLengths

def prepData(dataDict):
    """
    dataDict is of format {relPos:freq} where relPos is the position relative to the start/stop codon, and freq is the mean edit frequency at that position.
    Will convert this to a format that's easier to plot. Will be a list of tuples of format (relPos,freq) ordered from most negative relPos to most positive relPos.
    """
    dataList = sorted([(relPos, freq) for relPos, freq in dataDict.items()], key=lambda x: x[0])
    return dataList

def distinctColors(n):
    """
    Returns a list of n pyx colors, evenly spaced around the HSB color wheel. Unlike
    common.colors(), whose small hardcoded palette starts repeating colors after only
    8 entries (its 9th entry is already a duplicate of its 1st), this always produces
    n visually distinct colors, however many labels/libraries there are.
    """
    if n<=0:
        return []
    return [color.hsb(i/n,0.9,0.85) for i in range(n)]

##optionally populated in main() from a user-supplied colors.txt (see parseColorsFile);
##assignColors() consults this so every plot in the script colors a given library the same
##way, rather than colors being auto-assigned fresh (and potentially differently) per plot.
COLOR_MAP={}

def parseColorsFile(colorsFile):
    """
    colorsFile is an optional tab-delimited text file of format:
        libNamei\thexCodei
    one per line (e.g. "SampleA-Rep1\t#1b9e77"), with no header row. libNamei should match
    whatever label that library is plotted under in a given figure (e.g. the fileName-rep
    labels from labelForEntry for most plots, or the bare fileName for the reproducibility
    plots in readCountAndEditFreqRepro, which aggregate across replicates).

    Returns {libName:hexCode}, or {} if colorsFile is None.
    """
    if colorsFile is None:
        return {}
    colorMap={}
    with open(colorsFile,'r') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue##skips blank lines
            libName,hexCode=line.split('\t')
            colorMap[libName]=hexCode
    return colorMap

def hexToColor(hexCode):
    """
    hexCode is a string like '#1b9e77' or '1b9e77'. Returns the equivalent pyx color.rgb.
    """
    hexCode=hexCode.strip().lstrip('#')
    r=int(hexCode[0:2],16)/255
    g=int(hexCode[2:4],16)/255
    b=int(hexCode[4:6],16)/255
    return color.rgb(r,g,b)

def _splitRepToken(token):
    """
    Splits a colors.txt key, or a plot label, into (fileName,repToken) the same way
    parseInFilesParquet splits inFilesParquet.txt tokens (rfind of the literal substring
    "Rep"), but also tolerates an optional '-' immediately before "Rep". That means both the
    raw inFilesParquet.txt token ("minus3ATPolyRep1") AND the dashed label labelForEntry
    builds from it ("minus3ATPoly-Rep1") normalize to the same (fileName,repToken) pair,
    which is what lets assignColors treat a colors.txt entry written in either convention as
    equivalent. Returns (token,None) if "Rep" never occurs (same caveat as
    parseInFilesParquet applies here too: a fileName that happens to contain "Rep" as an
    ordinary substring will be mis-split the same way).
    """
    repIdx=token.rfind('Rep')
    if repIdx==-1:
        return token,None
    fileName=token[:repIdx]
    if fileName.endswith('-'):
        fileName=fileName[:-1]
    return fileName,token[repIdx:]

def assignColors(labels):
    """
    labels is an iterable of the library/line labels being plotted together on one figure.
    Returns {label:pyxColor}, resolving each label against the module-level COLOR_MAP
    (populated in main() from an optional user-supplied colors.txt -- see parseColorsFile) in
    order of preference:
    1. An exact string match.
    2. A match after normalizing both the label and every COLOR_MAP key via _splitRepToken --
       this is what lets colors.txt be written using the same raw tokens already in
       inFilesParquet.txt (e.g. "minus3ATPolyRep1"), even though most plots actually use the
       dashed label labelForEntry derives from that token (e.g. "minus3ATPoly-Rep1").
    3. For a bare-fileName label with no Rep suffix at all (as used by
       readCountAndEditFreqRepro's replicate-pooling plots, which group by fileName only): the
       shared color of that fileName's COLOR_MAP entries, but only if every one of them agrees
       on the same hexCode -- if they disagree there's no unambiguous single color for the
       pooled label, so it's left for auto-assignment instead.
    Any label none of the above resolves is assigned a distinct color from distinctColors, in
    sorted order, so the auto-assigned palette stays consistent across plots regardless of
    which labels COLOR_MAP happens to cover.
    """
    labels=list(labels)
    ##index every COLOR_MAP entry by its normalized (fileName,repToken) key, and separately by
    ##bare fileName (tracking every hexCode seen for that fileName, to detect ambiguity below).
    byNormalizedKey={}
    hexesByFileName=collections.defaultdict(set)
    for key,hexCode in COLOR_MAP.items():
        fileName,repToken=_splitRepToken(key)
        byNormalizedKey[(fileName,repToken)]=hexCode
        hexesByFileName[fileName].add(hexCode)
    hexByFileNameIfUnambiguous={fn:next(iter(hexes)) for fn,hexes in hexesByFileName.items() if len(hexes)==1}
    ##
    assigned={}
    remaining=[]
    for label in labels:
        if label in COLOR_MAP:
            assigned[label]=hexToColor(COLOR_MAP[label])
            continue
        fileName,repToken=_splitRepToken(label)
        normalizedHex=byNormalizedKey.get((fileName,repToken))
        if normalizedHex is not None:
            assigned[label]=hexToColor(normalizedHex)
        elif repToken is None and fileName in hexByFileNameIfUnambiguous:
            assigned[label]=hexToColor(hexByFileNameIfUnambiguous[fileName])
        else:
            remaining.append(label)
    remaining=sorted(remaining)
    autoColors=distinctColors(len(remaining))
    for i,label in enumerate(remaining):
        assigned[label]=autoColors[i]
    return assigned

##matplotlib's cividis colormap, used by cividisColor() below to color the heatmaps in
##readCountAndEditFreqRepro/analyzeRiboRNA -- perceptually uniform, dark blue-black at the
##low end, bright yellow at the high end.
_CIVIDIS=matplotlib.colormaps['cividis']

def cividisColor(t):
    """
    t is a float, nominally in [0,1] (values outside that range are clamped). Returns the
    pyx color.rgb at that position along matplotlib's 'cividis' colormap: dark blue-black at
    t=0, bright yellow at t=1.
    """
    t=max(0.0,min(1.0,t))
    r,g,b,_a=_CIVIDIS(t)
    return color.rgb(r,g,b)

def axisRange(vals,isLog=False,fallback=(-1,1)):
    """
    vals is an iterable of numeric values a plot's axis needs to cover. Returns a padded
    (vMin,vMax) that fits all of them -- 5% padding for a linear axis, a 1.1x multiplicative
    pad for a log axis (isLog=True), with a fallback if vals is empty and special-casing for
    the degenerate case where every value is identical (otherwise vMin==vMax gives pyx a
    zero-width axis range, which it refuses to render).
    """
    if not vals:
        return fallback
    vMin, vMax = min(vals), max(vals)
    if vMin == vMax:
        if isLog:
            vMin, vMax = vMin * 0.9, vMax / 0.9
        else:
            pad = abs(vMin) * 0.1 if vMin != 0 else 1.0
            vMin, vMax = vMin - pad, vMax + pad
    elif isLog:
        vMin, vMax = vMin / 1.1, vMax * 1.1
    else:
        pad = (vMax - vMin) * 0.05
        vMin, vMax = vMin - pad, vMax + pad
    return vMin, vMax

def mkPlot(libMetaDicts,outPrefix,norm=False):
    """
    libMetaDicts is of the format:
    {label: {'starts': {relPos: [editFreqs]}, 'stops': {relPos: [editFreqs]}}}
    Plots each library as a separate colored line on paired start/stop codon graphs. A key
    is shown to the right of, and outside, the stop codon graph.

    if norm=True, then the edit frequencies are normalized to the mean edit frequency across all positions for that
    library.

    The y-axis range is computed from the actual data (padded, see axisRange) rather than
    assumed to be [0,1]-ish -- editFreqs isn't always a true 0/1-bounded frequency (e.g.
    metaStartStopAnalysisNormalizeForMotifBias's motif-bias-weighted values can be >1), so a
    fixed [-0.1,1.1]/[-1.5,1.5] range would silently clip most of the data in that case.
    """
    ##convert each library's data to plottable lists
    plotDicts  = {}
    plot2Dicts = {}
    for label, metaDict in libMetaDicts.items():
        plotDicts[label]  = collections.defaultdict(dict)
        plot2Dicts[label] = collections.defaultdict(dict)
        for key in metaDict:
            avgIntCt = []
            for relPos in metaDict[key]:
                editInts = metaDict[key][relPos]
                if len(editInts) > 0:
                    freq = sum(editInts) / len(editInts)
                    plotDicts[label][key][relPos]  = freq
                    plot2Dicts[label][key][relPos] = len(editInts)
                    if relPos in range(-100, 101):
                        avgIntCt.append(len(editInts))
            if avgIntCt:
                print('\nAverage edit ints per position for %s %s: %f'
                      % (label, key, sum(avgIntCt) / len(avgIntCt)))
    ##if norm, normalize each library's freqs to its mean edit freq across all positions,
    ##then convert to log2 fold change
    if norm:
        for label in plotDicts:
            allFreqs = list(plotDicts[label]['starts'].values()) + list(plotDicts[label]['stops'].values())
            meanFreq = sum(allFreqs) / len(allFreqs) if allFreqs else 1.0
            for key in plotDicts[label]:
                plotDicts[label][key] = {
                    pos: math.log2(f / meanFreq) if f > 0 and meanFreq > 0 else float('nan')
                    for pos, f in plotDicts[label][key].items()
                }
    ##build graphs
    yTitle = 'Log2 Normalized Edit Frequency' if norm else 'Average Edit Frequency'
    ##pool every plotted freq (both starts and stops, every library) to size the (linked)
    ##y-axis, since a fixed range can't be trusted to cover the data (see docstring above).
    ##Restricted to relPos in [-100,100] -- the x-axis range actually shown below -- so a
    ##position that's in plotDicts but off-screen (e.g. from a much longer flank/CDS) can't
    ##blow up the y-axis for points nobody will see.
    allFreqVals = [v for label in plotDicts for key in ('starts','stops')
                   for relPos,v in plotDicts[label][key].items()
                   if -100<=relPos<=100 and not math.isnan(v)]
    yMin, yMax = axisRange(allFreqVals, fallback=(-1.5,1.5) if norm else (-0.1,1.1))
    start = graph.graphxy(width=8, height=8,
                          x=graph.axis.linear(min=-100, max=100,
                                              title='Position Relative to Start Codon'),
                          y=graph.axis.linear(min=yMin, max=yMax, title=yTitle))
    start2 = graph.graphxy(width=8, height=2, ypos=start.height + 0.5,
                           x=graph.axis.linkedaxis(start.axes["x"]),
                           y=graph.axis.log(title='Position Count'))
    stop = graph.graphxy(width=8, height=8, xpos=start.width * 1.1,
                         x=graph.axis.linear(min=-100, max=100,
                                             title='Position Relative to Stop Codon'),
                         y=graph.axis.linkedaxis(start.axes["y"]),
                         key=graph.key.key(pos='tr', hinside=0))
    stop2 = graph.graphxy(width=8, height=2, xpos=start.width * 1.1, ypos=start.height + 0.5,
                          x=graph.axis.linkedaxis(stop.axes["x"]),
                          y=graph.axis.linkedaxis(start2.axes["y"]))
    ##plot one colored line per library -- assignColors uses COLOR_MAP (if the user supplied a
    ##colors.txt) for any label it covers, falling back to distinctColors (rather than
    ##common.colors()'s small palette, which starts repeating colors after only 8 entries)
    ##for the rest.
    libColors = assignColors(libMetaDicts.keys())
    for label in sorted(libMetaDicts.keys()):
        startData  = prepData(plotDicts[label]['starts'])
        start2Data = prepData(plot2Dicts[label]['starts'])
        stopData   = prepData(plotDicts[label]['stops'])
        stop2Data  = prepData(plot2Dicts[label]['stops'])
        start.plot(graph.data.points(startData,  x=1, y=2),
                   [graph.style.line([libColors[label]])])
        start2.plot(graph.data.points(start2Data, x=1, y=2),
                    [graph.style.line([libColors[label]])])
        stop.plot(graph.data.points(stopData,   x=1, y=2, title=label),
                  [graph.style.line([libColors[label]])])
        stop2.plot(graph.data.points(stop2Data,  x=1, y=2),
                   [graph.style.line([libColors[label]])])
    c = canvas.canvas()
    c.insert(start)
    c.insert(start2)
    c.insert(stop)
    c.insert(stop2)
    c.writePDFfile(outPrefix)

def processMeta(metaDict,minCt=0):
    """
    metaDict is of the format:
    {starts/stops:{txtID:{position:[editInts]}}}}
    readCt is of the format:
    {txtID:set(reads)} where reads is a list of reads that all map to txtID and
    survived filters
    Will process metaDict to the format: {start/stop:{relPos:[editInts]}} 
    where relPos is the position relative to the start/stop codon, and editInts 
    is the average of the editInts for transcript_ids that overlap with that position.
    Every transcript_id is weighted the same, as long as there is at least N
    reads per transcript_id
    """
    metaDict2={}
    passed=set()
    for startOrStop in metaDict:
        ##doing this to avoid limitations around pickling lambda
        if startOrStop not in metaDict2:
            metaDict2[startOrStop]=collections.defaultdict(list)
        ##
        for txtID in metaDict[startOrStop]:
            for pos in metaDict[startOrStop][txtID]:
                editInts=metaDict[startOrStop][txtID][pos]
                if len(editInts)>=minCt:
                    avgEdit=sum(editInts)/len(editInts)
                    metaDict2[startOrStop][pos].append(avgEdit)
                    passed.add(txtID)
    ##
    print('\n%s transcripts passed read count cutoffs.'%(len(passed)))
    ##
    return dict(metaDict2)

def extractPerReadSites(parquetFiles,gtfDict):
    """
    parquetFiles is a file of the format described in parseInFilesParquet's docstring.
    gtfDict is of the format {strand:{chr:{absIndx:[(txtName,relStart,relStop)]}}}

    metaStartStopAnalysis, metaStartStopAnalysisNormalizeForMotifBias, and extractEditStrings
    (via intereditDistanceAnalyzer/intereditDistanceAnalyzerWithRandomizations) each used to
    independently re-read every parquet file and re-walk every read's nucleotides from scratch,
    despite needing near-identical raw material (each read's uniquely-assignable positions).
    This function does that walk once, and its output is shared by all of them.

    For every read, walks its ref_sequence_aligned/edit_string/absolute_indices together and,
    for every position with a real reference nt (skipping gap/insertion columns, where
    ref_sequence_aligned is a space and absolute_indices has no genomic anchor) that maps
    uniquely (exactly one hit) into gtfDict, records (relStart,relStop,edit,refNt,txtID) --
    where edit is the int-cast edit call (0, 1, or 2 -- callers each decide whether/how to
    filter on it, since that differs between consumers below) and txtID is gtfDict's own
    transcript assignment for that position (which the read's own transcript_id metadata is
    NOT checked against here, matching prior behavior -- some consumers cross-check it
    themselves, e.g. for neighbor lookups, others never did and still don't).

    Reads whose chrom isn't recognized in gtfDict for their strand are dropped (this used to
    print a warning independently from up to 3 different call sites per such read; now it
    prints once here instead, since this replaces all 3 re-reads). Unlike the meta-start/stop
    functions' own prior behavior, reads with an empty/RDN-containing transcript_id are NOT
    dropped here -- extractEditStrings never filtered on that, so that decision is left to
    each consumer instead of being baked into the shared extraction.

    Returns a dict of format {libName:[(transcript_id,sites),...]}, one (transcript_id,sites)
    entry per read that had a recognized chrom, in the order reads were encountered across
    that library's parquet file(s); sites is the (relStart,relStop,edit,refNt,txtID) list
    described above (possibly empty), in the read's own column order.
    """
    NEEDED_COLUMNS=['chrom','gene_strand','transcript_id','ref_sequence_aligned','edit_string','absolute_indices']
    perLibReads={}
    for fileName,rep,pFiles in parseInFilesParquet(parquetFiles):
        label=labelForEntry(fileName,rep)
        libReads=[]
        for parquetFile in pFiles:
            print('\nAnalyzing %s...'%(parquetFile))
            df=pd.read_parquet(parquetFile,columns=NEEDED_COLUMNS)
            print('Parquet File has %d rows'%(len(df)))
            for row in df.itertuples(index=False):
                chrom=row.chrom
                strand=row.gene_strand
                transcript_id=row.transcript_id
                chromDict=gtfDict[strand].get(chrom)
                if chromDict is None:
                    print('Chromosome %s not found in gtfDict for strand %s...'%(chrom,strand))
                    continue
                sites=[]
                for refNt,edit,absIdx in zip(row.ref_sequence_aligned,row.edit_string,row.absolute_indices):
                    if refNt not in 'ACGT':
                        continue##gap/insertion column -- no genomic anchor here
                    absIdx=int(absIdx)
                    hits=chromDict.get(absIdx)
                    if hits is None or len(hits)!=1:
                        continue##ambiguous (or unannotated) position -- filtered out
                    txtID,relStart,relStop=hits[0]
                    sites.append((relStart,relStop,int(edit),refNt,txtID))
                libReads.append((transcript_id,sites))
        perLibReads[label]=libReads
    return perLibReads

def plotReadLengthDistribution(perReadSites,outPrefix):
    """
    Given perReadSites, which is of the format:
    {libName:[(transcript_id,sites),...]}, as returned by extractPerReadSites
    this script will plot a CDF of read length distribution (technically just the
    len(sites)) for each libName.

    len(sites) is the number of positions in that read that mapped uniquely into
    gtfDict (see extractPerReadSites), not the raw sequenced/aligned read length --
    ambiguous/unassignable positions were already dropped upstream of perReadSites.

    Each library is plotted as its own line (colors from distinctColors, so libraries
    beyond common.colors()'s 8-color palette don't repeat), all on one graph, x-axis
    read length and y-axis cumulative fraction of that library's reads at or below
    that length. Saves outPrefix+'.pdf'; does not return anything.
    """
    ##one sorted length list per library -- sorting turns the CDF into a simple
    ##(length,(i+1)/n) walk below, no separate binning/counting step needed.
    lengthDict={}
    for label,libReads in perReadSites.items():
        lengthDict[label]=sorted(len(sites) for transcript_id,sites in libReads)
    ##
    allLengths=[length for lengths in lengthDict.values() for length in lengths]
    xMin,xMax=axisRange(allLengths,fallback=(0,1))
    ##
    g=graph.graphxy(width=8,height=8,
                    x=graph.axis.log(min=max(1,xMin),max=xMax,title='Read Length (nt uniquely assigned)'),
                    y=graph.axis.linear(min=0,max=1,title='Cumulative Fraction of Reads'),
                    key=graph.key.key(pos='tr',hinside=0))
    libColors=assignColors(lengthDict.keys())
    for label in sorted(lengthDict.keys()):
        lengths=lengthDict[label]
        if not lengths:
            print('%s has no reads with any uniquely-assigned positions; skipping.'%(label))
            continue
        n=len(lengths)
        cdfData=[(length,(i+1)/n) for i,length in enumerate(lengths)]
        g.plot(graph.data.points(cdfData,x=1,y=2,title=label),
               [graph.style.line([libColors[label]])])
    c=canvas.canvas()
    c.insert(g)
    c.writePDFfile(outPrefix+'.pdf')
    print('Read length distribution saved to %s.pdf'%(outPrefix))

def metaStartStopAnalysis(perReadSites,N,outPrefix):
    """
    perReadSites is of format {libName:[(transcript_id,sites),...]}, as returned by
    extractPerReadSites -- sites is [(relStart,relStop,edit,refNt,txtID),...] for every
    position in that read that mapped uniquely into gtfDict, regardless of nt identity.

    This function will not filter by N (read counts) for the metaStartStop analysis.
    The function will plot metaStartStop for each library and replicate, and will save as
    a pair of plots, with left plot being the start codon and right plot being the stop
    codon. Different libraries will be different lines on the plot.

    After making the metaStartStop plots, the function will also compute and return a dictionary
    of the format:
    {txtID:{'TL':[(editFreq,numAs),...],
        'CDS':[(editFreq,numAs),...],
        'UTR':[(editFreq,numAs),...],}}
    where each of the lists contain tuples of (editFreq,numAs) for each read that is uniquely
    assignable to that transcript_id (filtering out positions that overlap with multiple transcript_ids).
    The editFreq is the average edit frequency across the TL/CDS/UTR region, defined as:
        [-infinity,-25) for TL, [25,cdsLength-25] for CDS, and [cdsLength+25,infinity) for UTR in relStart
        coords.
    numAs is the number of A's in the same region.
    """
    ##
    metaStartStops=[]
    editFreqs=[]
    ##
    for label,libReads in perReadSites.items():
        ##
        metaDict={'starts':collections.defaultdict(lambda:collections.defaultdict(list)),
                'stops':collections.defaultdict(lambda:collections.defaultdict(list))}
        editFreqDict=collections.defaultdict(lambda:collections.defaultdict(list))
        ##
        for transcript_id,sites in libReads:
            ##skip reads whose own transcript_id is missing/rRNA, same restriction as before.
            if not transcript_id or 'RDN' in transcript_id:
                continue
            ##
            TL,CDS,UTR=[0,0],[0,0],[0,0]
            ##
            for relStart,relStop,edit,refNt,txtID in sites:
                ##
                hasA=(refNt=='A')
                ##
                if hasA and edit in (0,1):##restriction for uniquely assignable already applied upstream
                    ##
                    metaDict['starts'][transcript_id][relStart].append(edit)
                    metaDict['stops'][transcript_id][relStop].append(edit)
                    ##
                    if relStart<-25:
                        TL[0]+=edit
                        TL[1]+=1
                    elif relStart>=25 and relStop<=-25:
                        CDS[0]+=edit
                        CDS[1]+=1
                    elif relStop>25:
                        UTR[0]+=edit
                        UTR[1]+=1
                    ##
                    ##the following lines are some nt identity checks to make sure the positioning is consistent with known nt
                    ##composition of start/stop codons
                    if relStart==0 and refNt!='A':
                        print('Error: relStart=0 but refNt is not A: %s %s %s'%(transcript_id,relStart,refNt))
                    elif relStart==1 and refNt!='T':
                        print('Error: relStart=1 but refNt is not T: %s %s %s'%(transcript_id,relStart,refNt))
                    elif relStart==2 and refNt!='G':
                        if transcript_id.startswith('Q'):##filters out mitochondrial transcript, which can
                            ##start translation with ATA instead of ATG, so this is a special case that we
                            ##can ignore.
                            continue
                        print('Error: relStart=2 but refNt is not G: %s %s %s'%(transcript_id,relStart,refNt))
                    elif relStop==0 and refNt!='T':
                        print('Error: relStop=0 but refNt is not T: %s %s %s'%(transcript_id,relStop,refNt))
            ##
            editFreqDict[transcript_id]['TL'].append(TL)
            editFreqDict[transcript_id]['CDS'].append(CDS)
            editFreqDict[transcript_id]['UTR'].append(UTR)
        ##
        metaStartStops.append((label,metaDict))
        editFreqs.append((label,editFreqDict))
        ##
    ##process each library's metaDict for plotting.
    ##
    libMetaDictsProcessed = {}
    for label, libMetaDict in metaStartStops:
        libMetaDictsProcessed[label] = processMeta(libMetaDict,minCt=N)
    ##libsMetaDictsProcessed is now of the format:
    ##{label: {'starts': {relPos: [editFreqs]}, 'stops': {relPos: [editFreqs]}}}
    ##
    print('Plotting meta-edit distribution about start/stop codons...')
    mkPlot(libMetaDictsProcessed, outPrefix + '.metaStartStop.pdf')
    mkPlot(libMetaDictsProcessed, outPrefix + '.metaStartStopNormalized.pdf',norm=True)
    ##build the final editFreqDict across all libraries.
    ##each read contributed a [editSum, count] pair per region; convert to (editFreq, numAs) tuples.
    output=[]
    for label, libEditFreqDict in editFreqs:
        editFreqDict = collections.defaultdict(lambda: {'TL': [], 'CDS': [], 'UTR': []})
        for txtID in libEditFreqDict:
            for region in ['TL', 'CDS', 'UTR']:
                for editSum, count in libEditFreqDict[txtID][region]:
                    if count > 0:
                        editFreqDict[txtID][region].append((editSum / count, count))
        output.append((label, dict(editFreqDict)))
    ##
    return output

def metaStartStopAnalysisByStopCodon(perReadSites,N,outPrefix):
    """
    This function is similar to metaStartStopAnalysis, except that for every library, it will
    split the reads/transcript_ids up according to the stop codon identity. The stop codon
    will be one of ['TAA','TAG','TGA']. It will do this by restricting to reads that span
    the stop codon, and whose sequence matches one of the expected stop codons.
    It will plot the various stop codons for every library as different linestyles. It will plot
    everything together on the same plots (metaStart, metaStop), and save that as a pdf.
    It will also create a second batch of plots, where it will vertically array rows of 
    plots where each row is a different library, and the first column is the metaStart 
    and the second column is the metaStop. Linestyles will again differentiate stop codons.

    The function will return nothing, and in that sense is more similar to
    metaStartStopAnalysisNormalizeForMotifBias, except that it will not normalize for motif bias.
    """
    STOP_CODONS=['TAA','TAG','TGA']
    LINESTYLES={'TAA':style.linestyle.solid,'TAG':style.linestyle.dashed,'TGA':style.linestyle.dotted}
    ##
    ##metaByCodon: {label:{stopCodon:{'starts':{txtID:{relPos:[edit]}},'stops':{txtID:{relPos:[edit]}}}}}
    metaByCodon=collections.defaultdict(
        lambda:{codon:{'starts':collections.defaultdict(lambda:collections.defaultdict(list)),
                       'stops':collections.defaultdict(lambda:collections.defaultdict(list))}
                for codon in STOP_CODONS})
    ##
    for label,libReads in perReadSites.items():
        for transcript_id,sites in libReads:
            ##skip reads whose own transcript_id is missing/rRNA, same restriction as before.
            if not transcript_id or 'RDN' in transcript_id:
                continue
            ##
            ##determine this read's stop codon identity from the 3nts uniquely assigned to
            ##this transcript_id at relStop 0/1/2 (relStop=0 is the T of the TAA/TAG/TGA, per
            ##parseGTF) -- same cross-check against the read's own transcript_id as the
            ##relStart-neighbor lookup in metaStartStopAnalysisNormalizeForMotifBias.
            relStopToNt={relStop:refNt for relStart,relStop,edit,refNt,txtID in sites
                         if txtID==transcript_id and relStop in (0,1,2)}
            if not all(pos in relStopToNt for pos in (0,1,2)):
                continue##doesn't span the stop codon uniquely -- can't classify this read
            stopCodon=relStopToNt[0]+relStopToNt[1]+relStopToNt[2]
            if stopCodon not in STOP_CODONS:
                continue##sequence there doesn't match an expected stop codon
            ##
            metaDict=metaByCodon[label][stopCodon]
            for relStart,relStop,edit,refNt,txtID in sites:
                ##
                hasA=(refNt=='A')
                ##
                if hasA and edit in (0,1):##restriction for uniquely assignable already applied upstream
                    ##
                    metaDict['starts'][transcript_id][relStart].append(edit)
                    metaDict['stops'][transcript_id][relStop].append(edit)
                    ##
                    ##the following lines are some nt identity checks to make sure the positioning is consistent with known nt
                    ##composition of start/stop codons
                    if relStart==0 and refNt!='A':
                        print('Error: relStart=0 but refNt is not A: %s %s %s'%(transcript_id,relStart,refNt))
                    elif relStart==1 and refNt!='T':
                        print('Error: relStart=1 but refNt is not T: %s %s %s'%(transcript_id,relStart,refNt))
                    elif relStart==2 and refNt!='G':
                        if transcript_id.startswith('Q'):##filters out mitochondrial transcript, which can
                            ##start translation with ATA instead of ATG, so this is a special case that we
                            ##can ignore.
                            continue
                        print('Error: relStart=2 but refNt is not G: %s %s %s'%(transcript_id,relStart,refNt))
                    elif relStop==0 and refNt!='T':
                        print('Error: relStop=0 but refNt is not T: %s %s %s'%(transcript_id,relStop,refNt))
    ##
    ##process each (label,stopCodon) metaDict the same way mkPlot's callers do: average the
    ##per-transcript edit ints at each relPos, weighting every transcript equally.
    freqByLabelCodon={}
    for label in metaByCodon:
        freqByLabelCodon[label]={}
        for codon in STOP_CODONS:
            processed=processMeta(metaByCodon[label][codon])
            freqByLabelCodon[label][codon]={
                region:{relPos:sum(vals)/len(vals) for relPos,vals in processed.get(region,{}).items() if vals}
                for region in ('starts','stops')}
    ##
    ##pool all plotted freq values (restricted to the [-100,100] window actually shown, same
    ##rationale as mkPlot) to size a single y-axis shared across every row/column.
    allFreqVals=[freq for label in freqByLabelCodon for codon in STOP_CODONS
                 for region in ('starts','stops')
                 for relPos,freq in freqByLabelCodon[label][codon][region].items()
                 if -100<=relPos<=100]
    yMin,yMax=axisRange(allFreqVals,fallback=(-0.1,1.1))
    ##
    labels=sorted(freqByLabelCodon.keys())
    ##
    ##file 1: everything together -- one metaStart graph and one metaStop graph, every
    ##(label,stopCodon) combination its own line, colored by library (as in mkPlot) and
    ##linestyled by stop codon.
    libColors=assignColors(labels)
    start=graph.graphxy(width=8,height=8,
                        x=graph.axis.linear(min=-100,max=100,title='Position Relative to Start Codon'),
                        y=graph.axis.linear(min=yMin,max=yMax,title='Average Edit Frequency'))
    stop=graph.graphxy(width=8,height=8,xpos=start.width*1.1,
                       x=graph.axis.linear(min=-100,max=100,title='Position Relative to Stop Codon'),
                       y=graph.axis.linkedaxis(start.axes['y']),
                       key=graph.key.key(pos='tr',hinside=0))
    for label in labels:
        for codon in STOP_CODONS:
            startData=prepData(freqByLabelCodon[label][codon]['starts'])
            stopData =prepData(freqByLabelCodon[label][codon]['stops'])
            if not startData and not stopData:
                continue##this library/stop-codon combo had no qualifying reads
            title='%s %s'%(label,codon)
            if startData:
                start.plot(graph.data.points(startData,x=1,y=2),
                          [graph.style.line([libColors[label],LINESTYLES[codon]])])
            if stopData:
                stop.plot(graph.data.points(stopData,x=1,y=2,title=title),
                         [graph.style.line([libColors[label],LINESTYLES[codon]])])
    cCombined=canvas.canvas()
    cCombined.insert(start)
    cCombined.insert(stop)
    cCombined.writePDFfile(outPrefix+'.metaStartStopByStopCodon.pdf')
    print('Meta-edit distribution by stop codon saved to %s.metaStartStopByStopCodon.pdf'%(outPrefix))
    ##
    ##file 2: vertically arrayed rows (one per library), first column metaStart, second
    ##column metaStop, with stop codons differentiated by linestyle within each subplot.
    graphWidth,graphHeight=8,4
    xSpacing=graphWidth*1.15
    ySpacing=graphHeight*1.3
    noLabelPainter=graph.axis.painter.regular(labelattrs=None,titleattrs=None)
    columns=[('starts','Position Relative to Start Codon'),('stops','Position Relative to Stop Codon')]
    refYAxis=None
    c=canvas.canvas()
    for rowIdx,label in enumerate(labels):
        rowY=(len(labels)-1-rowIdx)*ySpacing##top row (label sorted first) drawn highest
        for colIdx,(region,xTitle) in enumerate(columns):
            isFirst=(rowIdx==0 and colIdx==0)
            yAxis=(graph.axis.linear(min=yMin,max=yMax,title='Average Edit Frequency') if isFirst
                   else graph.axis.linkedaxis(refYAxis,painter=noLabelPainter))
            g=graph.graphxy(width=graphWidth,height=graphHeight,
                            xpos=colIdx*xSpacing,ypos=rowY,
                            x=graph.axis.linear(min=-100,max=100,title=xTitle),
                            y=yAxis,
                            key=(graph.key.key(pos='tr',hinside=0) if rowIdx==0 and colIdx==1 else None))
            if isFirst:
                refYAxis=g.axes['y']
            for codon in STOP_CODONS:
                data=prepData(freqByLabelCodon[label][codon][region])
                if not data:
                    continue##this library/stop-codon combo had no qualifying reads
                g.plot(graph.data.points(data,x=1,y=2,title=codon),
                       [graph.style.line([color.rgb.black,LINESTYLES[codon]])])
            c.insert(g)
        ##row label to the left of the start-codon column
        c.text(-0.5,rowY+graphHeight/2,label,[text.halign.right,text.valign.middle])
    c.writePDFfile(outPrefix+'.metaStartStopByStopCodon.byLibrary.pdf')
    print('Meta-edit distribution by stop codon (by library) saved to %s.metaStartStopByStopCodon.byLibrary.pdf'%(outPrefix))

def metaStartStopAnalysisByCDSLength(perReadSites,N,cdsLengths,outPrefix):
    """
    This function is similar to metaStartStopAnalysis, except that for every library, it will
    split the transcript_ids by cdsLengths. It will split transcript_ids into quartiles based on 
    cdsLengths, and will plot the metaStartStop for each quartile separately, as different linestyles,
    as with the stop codons in metaStartStopAnalysisByStopCodon. 
    It will plot everything together on the same plots (metaStart, metaStop). It will also create a second file
    where the plots are vertically arrayed, with each row being a different library, and the first 
    column being the metaStart and the second column being the metaStop. Linestyles will again differentiate
    cds Lengths. The function will return nothing.

    The distribution of cdsLengths will also be included on the key for the plots.
    """
    NUM_QUARTILES=4
    LINESTYLES=[style.linestyle.solid,style.linestyle.dashed,style.linestyle.dotted,style.linestyle.dashdotted]
    ##
    ##assign every annotated transcript_id a quartile (0-3) based on where its cdsLength falls
    ##among all annotated cdsLengths -- quartile boundaries are the 25/50/75th percentiles of
    ##sorted(cdsLengths.values()).
    sortedLengths=sorted(cdsLengths.values())
    nLengths=len(sortedLengths)
    cutoffs=[sortedLengths[min(nLengths-1,int(p*nLengths))] for p in (0.25,0.5,0.75)]
    def quartileOf(length):
        for q,cutoff in enumerate(cutoffs):
            if length<=cutoff:
                return q
        return len(cutoffs)
    txtToQuartile={txtID:quartileOf(length) for txtID,length in cdsLengths.items()}
    ##
    ##label each quartile with the actual length range of the transcripts assigned to it, so
    ##the key shows the cdsLength distribution underlying each linestyle.
    quartileLengths=[[] for _ in range(NUM_QUARTILES)]
    for txtID,q in txtToQuartile.items():
        quartileLengths[q].append(cdsLengths[txtID])
    quartileKeyLabels=['Q%d (%d-%dnt)'%(q+1,min(lens),max(lens)) if lens else 'Q%d (no data)'%(q+1)
                       for q,lens in enumerate(quartileLengths)]
    ##
    ##metaByQuartile: {label:{quartile:{'starts':{txtID:{relPos:[edit]}},'stops':{txtID:{relPos:[edit]}}}}}
    metaByQuartile=collections.defaultdict(
        lambda:{q:{'starts':collections.defaultdict(lambda:collections.defaultdict(list)),
                  'stops':collections.defaultdict(lambda:collections.defaultdict(list))}
                for q in range(NUM_QUARTILES)})
    ##
    for label,libReads in perReadSites.items():
        for transcript_id,sites in libReads:
            ##skip reads whose own transcript_id is missing/rRNA, same restriction as before.
            if not transcript_id or 'RDN' in transcript_id:
                continue
            quartile=txtToQuartile.get(transcript_id)
            if quartile is None:
                continue##not an annotated protein-coding transcript -- can't classify by cds length
            ##
            metaDict=metaByQuartile[label][quartile]
            for relStart,relStop,edit,refNt,txtID in sites:
                ##
                hasA=(refNt=='A')
                ##
                if hasA and edit in (0,1):##restriction for uniquely assignable already applied upstream
                    ##
                    metaDict['starts'][transcript_id][relStart].append(edit)
                    metaDict['stops'][transcript_id][relStop].append(edit)
                    ##
                    ##the following lines are some nt identity checks to make sure the positioning is consistent with known nt
                    ##composition of start/stop codons
                    if relStart==0 and refNt!='A':
                        print('Error: relStart=0 but refNt is not A: %s %s %s'%(transcript_id,relStart,refNt))
                    elif relStart==1 and refNt!='T':
                        print('Error: relStart=1 but refNt is not T: %s %s %s'%(transcript_id,relStart,refNt))
                    elif relStart==2 and refNt!='G':
                        if transcript_id.startswith('Q'):##filters out mitochondrial transcript, which can
                            ##start translation with ATA instead of ATG, so this is a special case that we
                            ##can ignore.
                            continue
                        print('Error: relStart=2 but refNt is not G: %s %s %s'%(transcript_id,relStart,refNt))
                    elif relStop==0 and refNt!='T':
                        print('Error: relStop=0 but refNt is not T: %s %s %s'%(transcript_id,relStop,refNt))
    ##
    ##process each (label,quartile) metaDict the same way mkPlot's callers do: average the
    ##per-transcript edit ints at each relPos, weighting every transcript equally.
    freqByLabelQuartile={}
    for label in metaByQuartile:
        freqByLabelQuartile[label]={}
        for q in range(NUM_QUARTILES):
            processed=processMeta(metaByQuartile[label][q])
            freqByLabelQuartile[label][q]={
                region:{relPos:sum(vals)/len(vals) for relPos,vals in processed.get(region,{}).items() if vals}
                for region in ('starts','stops')}
    ##
    labels=sorted(freqByLabelQuartile.keys())
    ##
    ##pool all plotted freq values (restricted to the [-100,100] window actually shown, same
    ##rationale as mkPlot) to size a single y-axis shared across both output files below.
    allFreqVals=[freq for label in freqByLabelQuartile for q in range(NUM_QUARTILES)
                 for region in ('starts','stops')
                 for relPos,freq in freqByLabelQuartile[label][q][region].items()
                 if -100<=relPos<=100]
    yMin,yMax=axisRange(allFreqVals,fallback=(-0.1,1.1))
    ##
    ##file 1: everything together -- one metaStart graph and one metaStop graph, every
    ##(label,quartile) combination its own line, colored by label (as in mkPlot) and
    ##linestyled by quartile (as in metaStartStopAnalysisByStopCodon).
    libColors=assignColors(labels)
    start=graph.graphxy(width=8,height=8,
                        x=graph.axis.linear(min=-100,max=100,title='Position Relative to Start Codon'),
                        y=graph.axis.linear(min=yMin,max=yMax,title='Average Edit Frequency'))
    stop=graph.graphxy(width=8,height=8,xpos=start.width*1.1,
                       x=graph.axis.linear(min=-100,max=100,title='Position Relative to Stop Codon'),
                       y=graph.axis.linkedaxis(start.axes['y']),
                       key=graph.key.key(pos='tr',hinside=0))
    for label in labels:
        for q in range(NUM_QUARTILES):
            startData=prepData(freqByLabelQuartile[label][q]['starts'])
            stopData =prepData(freqByLabelQuartile[label][q]['stops'])
            if not startData and not stopData:
                continue##this library/quartile combo had no qualifying reads
            title='%s %s'%(label,quartileKeyLabels[q])
            if startData:
                start.plot(graph.data.points(startData,x=1,y=2),
                          [graph.style.line([libColors[label],LINESTYLES[q]])])
            if stopData:
                stop.plot(graph.data.points(stopData,x=1,y=2,title=title),
                         [graph.style.line([libColors[label],LINESTYLES[q]])])
    c=canvas.canvas()
    c.insert(start)
    c.insert(stop)
    c.writePDFfile(outPrefix+'.metaStartStopByCDSLength.pdf')
    print('Meta-edit distribution by CDS length saved to %s.metaStartStopByCDSLength.pdf'%(outPrefix))
    ##
    ##file 2: vertically arrayed rows (one per library), first column metaStart, second
    ##column metaStop, with quartiles differentiated by linestyle within each subplot --
    ##same grid layout as metaStartStopAnalysisByStopCodon.
    graphWidth,graphHeight=8,4
    xSpacing=graphWidth*1.15
    ySpacing=graphHeight*1.3
    noLabelPainter=graph.axis.painter.regular(labelattrs=None,titleattrs=None)
    columns=[('starts','Position Relative to Start Codon'),('stops','Position Relative to Stop Codon')]
    refYAxis=None
    c2=canvas.canvas()
    for rowIdx,label in enumerate(labels):
        rowY=(len(labels)-1-rowIdx)*ySpacing##top row (label sorted first) drawn highest
        for colIdx,(region,xTitle) in enumerate(columns):
            isFirst=(rowIdx==0 and colIdx==0)
            yAxis=(graph.axis.linear(min=yMin,max=yMax,title='Average Edit Frequency') if isFirst
                   else graph.axis.linkedaxis(refYAxis,painter=noLabelPainter))
            g=graph.graphxy(width=graphWidth,height=graphHeight,
                            xpos=colIdx*xSpacing,ypos=rowY,
                            x=graph.axis.linear(min=-100,max=100,title=xTitle),
                            y=yAxis,
                            key=(graph.key.key(pos='tr',hinside=0) if rowIdx==0 and colIdx==1 else None))
            if isFirst:
                refYAxis=g.axes['y']
            for q in range(NUM_QUARTILES):
                data=prepData(freqByLabelQuartile[label][q][region])
                if not data:
                    continue##this library/quartile combo had no qualifying reads
                g.plot(graph.data.points(data,x=1,y=2,title=quartileKeyLabels[q]),
                       [graph.style.line([color.rgb.black,LINESTYLES[q]])])
            c2.insert(g)
        ##row label to the left of the start-codon column
        c2.text(-0.5,rowY+graphHeight/2,label,[text.halign.right,text.valign.middle])
    c2.writePDFfile(outPrefix+'.metaStartStopByCDSLength.byLibrary.pdf')
    print('Meta-edit distribution by CDS length (by library) saved to %s.metaStartStopByCDSLength.byLibrary.pdf'%(outPrefix))

def metaStartStopAnalysisNormalizeForMotifBias(perReadSites,N,motifDict,outPrefix):
    """
    perReadSites is of format {libName:[(transcript_id,sites),...]}, as returned by
    extractPerReadSites -- sites is [(relStart,relStop,edit,refNt,txtID),...] for every
    position in that read that mapped uniquely into gtfDict, regardless of nt identity.
    motifDict is of the format {libName:{motif:freq}} where freq is the edit frequency
    of that motif in that libName.

    This function will not filter by N (read counts) for the metaStartStop analysis.
    The function will plot metaStartStop for each library and replicate in parquetFiles,
    and will save as a pair of plots, with left plot being the start codon and right plot
    being the stop codon. Different libraries will be different lines on the plot.

    This function will differ from metaStartStopAnalysis in that it will normalize the edits at
    any given location by the motif frequency for that library, to account for motif bias in the editing.
    To do this, it will first determine the motif about a given edited position, and then divide the
    edit frequency at that position by the motif frequency for that library. Thus if a motif is typically
    rarely edited, and it is seen to be edited, it will count for more. If a motif is commonly edited,
    and it is seen to be edited, it will count for less. This is a way to normalize for motif bias in the editing.
    Note that the motif will differ depending on the read. Meaning that every position relative to a
    start/stop codon will have a different motif, depending on the read that is being analyzed.

    The motif is the 3nt (prevNt,'A',nextNt) surrounding a given A, where prevNt/nextNt are the bases
    at relStart-1/relStart+1 for that same transcript_id -- looked up via gtfDict, not just the read's
    adjacent alignment columns (which can be offset from relStart-1/+1 by an insertion, or by the read's
    own alignment landing on an unrelated/ambiguous position there). Unedited (edit=0) A's are recorded
    as a plain 0 (no motif-based weighting is meaningful for a non-edit); an edited (edit=1) A is recorded
    as 1/motifFreq[label][motif]. Two situations make that impossible to compute, and in both, the edited
    position is simply dropped (not recorded at all, for either 'starts' or 'stops'), rather than falling
    back to an unweighted value:
    - relStart-1 or relStart+1 isn't uniquely assignable to this transcript_id in this read (e.g. the
      read ends there, or that position is ambiguous) -- the motif can't be determined.
    - motifFreq[label] doesn't have this motif, or has it at 0.0 (never observed edited in analyzeMotifs)
      -- 1/motifFreq would be undefined/infinite.
    """
    ##
    metaStartStops=[]
    ##
    for label,libReads in perReadSites.items():
        libMotifFreqs=motifDict.get(label,{})
        ##
        metaDict={'starts':collections.defaultdict(lambda:collections.defaultdict(list)),
                'stops':collections.defaultdict(lambda:collections.defaultdict(list))}
        ##
        skippedNoNeighbor=0
        skippedNoMotifFreq=0
        ##
        for transcript_id,sites in libReads:
            if not transcript_id or 'RDN' in transcript_id:
                continue
            ##
            ##build {relStart:refNt} for every position in this read that maps uniquely to
            ##this transcript_id, so relStart-1/relStart+1 neighbors of an edited A (found in
            ##the loop below) can be looked up directly.
            relStartToNt={relStart:refNt for relStart,relStop,edit,refNt,txtID in sites
                          if txtID==transcript_id}
            ##
            for relStart,relStop,edit,refNt,txtID in sites:
                ##
                hasA=(refNt=='A')
                ##
                if hasA and edit in (0,1):##restriction for uniquely assignable already applied upstream
                    ##
                    if edit==0:
                        weightedEdit=0.0
                    else:
                        prevNt=relStartToNt.get(relStart-1)
                        nextNt=relStartToNt.get(relStart+1)
                        if prevNt is None or nextNt is None:
                            skippedNoNeighbor+=1
                            continue##can't determine the motif -- drop this position
                        motif=prevNt+'A'+nextNt
                        motifFreq=libMotifFreqs.get(motif)
                        if not motifFreq:##missing, or exactly 0.0
                            skippedNoMotifFreq+=1
                            continue##1/motifFreq would be undefined -- drop this position
                        weightedEdit=1.0/motifFreq
                    ##
                    metaDict['starts'][transcript_id][relStart].append(weightedEdit)
                    metaDict['stops'][transcript_id][relStop].append(weightedEdit)
                    ##
                    ##the following lines are some nt identity checks to make sure the positioning is consistent with known nt
                    ##composition of start/stop codons
                    if relStart==0 and refNt!='A':
                        print('Error: relStart=0 but refNt is not A: %s %s %s'%(transcript_id,relStart,refNt))
                    elif relStart==1 and refNt!='T':
                        print('Error: relStart=1 but refNt is not T: %s %s %s'%(transcript_id,relStart,refNt))
                    elif relStart==2 and refNt!='G':
                        if transcript_id.startswith('Q'):##filters out mitochondrial transcript, which can
                            ##start translation with ATA instead of ATG, so this is a special case that we
                            ##can ignore.
                            continue
                        print('Error: relStart=2 but refNt is not G: %s %s %s'%(transcript_id,relStart,refNt))
                    elif relStop==0 and refNt!='T':
                        print('Error: relStop=0 but refNt is not T: %s %s %s'%(transcript_id,relStop,refNt))
        ##
        print('%s: %d edited A positions dropped (no relStart-adjacent neighbor), '
              '%d edited A positions dropped (no/zero motif frequency).'
              %(label,skippedNoNeighbor,skippedNoMotifFreq))
        metaStartStops.append((label,metaDict))
        ##
    ##process each library's metaDict for plotting.
    ##
    libMetaDictsProcessed = {}
    for label, libMetaDict in metaStartStops:
        libMetaDictsProcessed[label] = processMeta(libMetaDict)
    ##libsMetaDictsProcessed is now of the format:
    ##{label: {'starts': {relPos: [weightedEdits]}, 'stops': {relPos: [weightedEdits]}}}
    ##
    print('Plotting motif-bias-normalized meta-edit distribution about start/stop codons...')
    mkPlot(libMetaDictsProcessed, outPrefix + '.metaStartStop.pdf')
    mkPlot(libMetaDictsProcessed, outPrefix + '.metaStartStopNormalized.pdf',norm=True)

def metaStartStopAnalysisByCDSLengthNormaliedForMotifBias(perReadSites,N,cdsLengths,motifDict,outPrefix):
    """
    This function will sort transcript_ids by quartiles of cdsLengths. It will count edits using the same
    motif-based-normalization as metaStartStopAnalysisNormalizeForMotifBias. 
    It will plot the metaStartStop for each quartile separately, as different linestyles, on one plot, and then
    create a second file where the plots are vertically arrayed, with each row being a different library, 
    and the first column being the metaStart and the second column being the metaStop. 
    Linestyles will again differentiate cds length quartiles.
    """
    NUM_QUARTILES=4
    LINESTYLES=[style.linestyle.solid,style.linestyle.dashed,style.linestyle.dotted,style.linestyle.dashdotted]
    ##
    ##assign every annotated transcript_id a quartile (0-3) based on where its cdsLength falls
    ##among all annotated cdsLengths -- quartile boundaries are the 25/50/75th percentiles of
    ##sorted(cdsLengths.values()), same as metaStartStopAnalysisByCDSLength.
    sortedLengths=sorted(cdsLengths.values())
    nLengths=len(sortedLengths)
    cutoffs=[sortedLengths[min(nLengths-1,int(p*nLengths))] for p in (0.25,0.5,0.75)]
    def quartileOf(length):
        for q,cutoff in enumerate(cutoffs):
            if length<=cutoff:
                return q
        return len(cutoffs)
    txtToQuartile={txtID:quartileOf(length) for txtID,length in cdsLengths.items()}
    ##
    ##label each quartile with the actual length range of the transcripts assigned to it, so
    ##the key shows the cdsLength distribution underlying each linestyle.
    quartileLengths=[[] for _ in range(NUM_QUARTILES)]
    for txtID,q in txtToQuartile.items():
        quartileLengths[q].append(cdsLengths[txtID])
    quartileKeyLabels=['Q%d (%d-%dnt)'%(q+1,min(lens),max(lens)) if lens else 'Q%d (no data)'%(q+1)
                       for q,lens in enumerate(quartileLengths)]
    ##
    ##metaByQuartile: {label:{quartile:{'starts':{txtID:{relPos:[weightedEdit]}},'stops':{txtID:{relPos:[weightedEdit]}}}}}
    metaByQuartile=collections.defaultdict(
        lambda:{q:{'starts':collections.defaultdict(lambda:collections.defaultdict(list)),
                  'stops':collections.defaultdict(lambda:collections.defaultdict(list))}
                for q in range(NUM_QUARTILES)})
    ##
    for label,libReads in perReadSites.items():
        libMotifFreqs=motifDict.get(label,{})
        ##
        skippedNoNeighbor=0
        skippedNoMotifFreq=0
        ##
        for transcript_id,sites in libReads:
            ##skip reads whose own transcript_id is missing/rRNA, same restriction as before.
            if not transcript_id or 'RDN' in transcript_id:
                continue
            quartile=txtToQuartile.get(transcript_id)
            if quartile is None:
                continue##not an annotated protein-coding transcript -- can't classify by cds length
            ##
            ##build {relStart:refNt} for every position in this read that maps uniquely to
            ##this transcript_id, so relStart-1/relStart+1 neighbors of an edited A (found in
            ##the loop below) can be looked up directly.
            relStartToNt={relStart:refNt for relStart,relStop,edit,refNt,txtID in sites
                          if txtID==transcript_id}
            ##
            metaDict=metaByQuartile[label][quartile]
            for relStart,relStop,edit,refNt,txtID in sites:
                ##
                hasA=(refNt=='A')
                ##
                if hasA and edit in (0,1):##restriction for uniquely assignable already applied upstream
                    ##
                    if edit==0:
                        weightedEdit=0.0
                    else:
                        prevNt=relStartToNt.get(relStart-1)
                        nextNt=relStartToNt.get(relStart+1)
                        if prevNt is None or nextNt is None:
                            skippedNoNeighbor+=1
                            continue##can't determine the motif -- drop this position
                        motif=prevNt+'A'+nextNt
                        motifFreq=libMotifFreqs.get(motif)
                        if not motifFreq:##missing, or exactly 0.0
                            skippedNoMotifFreq+=1
                            continue##1/motifFreq would be undefined -- drop this position
                        weightedEdit=1.0/motifFreq
                    ##
                    metaDict['starts'][transcript_id][relStart].append(weightedEdit)
                    metaDict['stops'][transcript_id][relStop].append(weightedEdit)
                    ##
                    ##the following lines are some nt identity checks to make sure the positioning is consistent with known nt
                    ##composition of start/stop codons
                    if relStart==0 and refNt!='A':
                        print('Error: relStart=0 but refNt is not A: %s %s %s'%(transcript_id,relStart,refNt))
                    elif relStart==1 and refNt!='T':
                        print('Error: relStart=1 but refNt is not T: %s %s %s'%(transcript_id,relStart,refNt))
                    elif relStart==2 and refNt!='G':
                        if transcript_id.startswith('Q'):##filters out mitochondrial transcript, which can
                            ##start translation with ATA instead of ATG, so this is a special case that we
                            ##can ignore.
                            continue
                        print('Error: relStart=2 but refNt is not G: %s %s %s'%(transcript_id,relStart,refNt))
                    elif relStop==0 and refNt!='T':
                        print('Error: relStop=0 but refNt is not T: %s %s %s'%(transcript_id,relStop,refNt))
        ##
        print('%s: %d edited A positions dropped (no relStart-adjacent neighbor), '
              '%d edited A positions dropped (no/zero motif frequency).'
              %(label,skippedNoNeighbor,skippedNoMotifFreq))
    ##
    ##process each (label,quartile) metaDict the same way mkPlot's callers do: average the
    ##per-transcript weighted edits at each relPos, weighting every transcript equally.
    freqByLabelQuartile={}
    for label in metaByQuartile:
        freqByLabelQuartile[label]={}
        for q in range(NUM_QUARTILES):
            processed=processMeta(metaByQuartile[label][q])
            freqByLabelQuartile[label][q]={
                region:{relPos:sum(vals)/len(vals) for relPos,vals in processed.get(region,{}).items() if vals}
                for region in ('starts','stops')}
    ##
    labels=sorted(freqByLabelQuartile.keys())
    ##
    ##pool all plotted freq values (restricted to the [-100,100] window actually shown, same
    ##rationale as mkPlot) to size a single y-axis shared across both output files below.
    allFreqVals=[freq for label in freqByLabelQuartile for q in range(NUM_QUARTILES)
                 for region in ('starts','stops')
                 for relPos,freq in freqByLabelQuartile[label][q][region].items()
                 if -100<=relPos<=100]
    yMin,yMax=axisRange(allFreqVals,fallback=(-0.1,1.1))
    ##
    ##file 1: everything together -- one metaStart graph and one metaStop graph, every
    ##(label,quartile) combination its own line, colored by label (as in mkPlot) and
    ##linestyled by quartile (as in metaStartStopAnalysisByCDSLength).
    libColors=assignColors(labels)
    start=graph.graphxy(width=8,height=8,
                        x=graph.axis.linear(min=-100,max=100,title='Position Relative to Start Codon'),
                        y=graph.axis.linear(min=yMin,max=yMax,title='Average Normalized Edit Frequency'))
    stop=graph.graphxy(width=8,height=8,xpos=start.width*1.1,
                       x=graph.axis.linear(min=-100,max=100,title='Position Relative to Stop Codon'),
                       y=graph.axis.linkedaxis(start.axes['y']),
                       key=graph.key.key(pos='tr',hinside=0))
    for label in labels:
        for q in range(NUM_QUARTILES):
            startData=prepData(freqByLabelQuartile[label][q]['starts'])
            stopData =prepData(freqByLabelQuartile[label][q]['stops'])
            if not startData and not stopData:
                continue##this library/quartile combo had no qualifying reads
            title='%s %s'%(label,quartileKeyLabels[q])
            if startData:
                start.plot(graph.data.points(startData,x=1,y=2),
                          [graph.style.line([libColors[label],LINESTYLES[q]])])
            if stopData:
                stop.plot(graph.data.points(stopData,x=1,y=2,title=title),
                         [graph.style.line([libColors[label],LINESTYLES[q]])])
    c=canvas.canvas()
    c.insert(start)
    c.insert(stop)
    c.writePDFfile(outPrefix+'.metaStartStopByCDSLengthNormalizedForMotifBias.pdf')
    print('Motif-bias-normalized meta-edit distribution by CDS length saved to %s.metaStartStopByCDSLengthNormalizedForMotifBias.pdf'%(outPrefix))
    ##
    ##file 2: vertically arrayed rows (one per library), first column metaStart, second
    ##column metaStop, with quartiles differentiated by linestyle within each subplot --
    ##same grid layout as metaStartStopAnalysisByCDSLength.
    graphWidth,graphHeight=8,4
    xSpacing=graphWidth*1.15
    ySpacing=graphHeight*1.3
    noLabelPainter=graph.axis.painter.regular(labelattrs=None,titleattrs=None)
    columns=[('starts','Position Relative to Start Codon'),('stops','Position Relative to Stop Codon')]
    refYAxis=None
    c2=canvas.canvas()
    for rowIdx,label in enumerate(labels):
        rowY=(len(labels)-1-rowIdx)*ySpacing##top row (label sorted first) drawn highest
        for colIdx,(region,xTitle) in enumerate(columns):
            isFirst=(rowIdx==0 and colIdx==0)
            yAxis=(graph.axis.linear(min=yMin,max=yMax,title='Average Normalized Edit Frequency') if isFirst
                   else graph.axis.linkedaxis(refYAxis,painter=noLabelPainter))
            g=graph.graphxy(width=graphWidth,height=graphHeight,
                            xpos=colIdx*xSpacing,ypos=rowY,
                            x=graph.axis.linear(min=-100,max=100,title=xTitle),
                            y=yAxis,
                            key=(graph.key.key(pos='tr',hinside=0) if rowIdx==0 and colIdx==1 else None))
            if isFirst:
                refYAxis=g.axes['y']
            for q in range(NUM_QUARTILES):
                data=prepData(freqByLabelQuartile[label][q][region])
                if not data:
                    continue##this library/quartile combo had no qualifying reads
                g.plot(graph.data.points(data,x=1,y=2,title=quartileKeyLabels[q]),
                       [graph.style.line([color.rgb.black,LINESTYLES[q]])])
            c2.insert(g)
        ##row label to the left of the start-codon column
        c2.text(-0.5,rowY+graphHeight/2,label,[text.halign.right,text.valign.middle])
    c2.writePDFfile(outPrefix+'.metaStartStopByCDSLengthNormalizedForMotifBias.byLibrary.pdf')
    print('Motif-bias-normalized meta-edit distribution by CDS length (by library) saved to %s.metaStartStopByCDSLengthNormalizedForMotifBias.byLibrary.pdf'%(outPrefix))

def analyzeRiboRNA(editFreqDict2,riboSeqFile,rnaSeqFile,N,outPrefix):
    """
    editFreqDict is of the format:
    [(label, {txtID: {'TL': [(editFreq,numAs),...
      'CDS': [(editFreq,numAs),...], 'UTR': [(editFreq,numAs),...]}})]
    riboSeqFile and rnaSeqFile are files containing each line:
    txtID\tRPKM
    where txtID is the transcript_id and RPKM is the RPKM value for that transcript_id in that library.
    
    For each txtID, will compute the Ribo-seq RPKM / RNA-seq RPKM ratio.

    For each label in editFreqDict, and each of the three regions (TL, CDS, UTR), will 
    compute the average edit frequency across all reads for that region if there are at least N reads,
    and plot it against the Ribo/RNA ratio.

    These plots will be scatter plots, arrayed left-to-right for TL, CDS, and UTR, with a linear regression line
    fit to the data, a SpearmanR value, and a p-value for the correlation in the title of the plot.

    Each row of the array of plots will be a different label from editFreqDict.

    The plots will be saved to one PDF file with the prefix outPrefix+'.riboRNA.editFreq.pdf'.

    This function will also create a heatmap of the rho (spearmanr) values from each of the scatterplots,
    with the rows being the labels and the columns being the regions (TL, CDS, UTR).

    This function will also output a dict of format:
    {label:{'txtID':cdsAvgEditFreq}} where cdsAvgEditFreq is the average edit frequency across all 
    reads for that transcript_id in the CDS region, if there are at least N reads in that region.

    Written by Claude. Proofread, edited by JA.
    """
    ##As passed, the txtID keys of editFreqDict have an extraneous '_mRNA' suffix. Remove it.
    editFreqDict = []
    for entry in editFreqDict2:
        label, txDict = entry
        txDict = {k.rstrip('_mRNA'): v for k, v in txDict.items()}
        editFreqDict.append((label, txDict))
    ##build output dict: {label: {txtID: cdsAvgEditFreq}} for transcripts with >= N CDS reads
    cdsEditDict = {}
    for label, txDict in editFreqDict:
        cdsEditDict[label] = {}
        for txtID, regionDict in txDict.items():
            reads = regionDict.get('CDS', [])
            if len(reads) >= N:
                cdsEditDict[label][txtID] = sum(ef for ef, na in reads) / len(reads)
    ##
    ##parse ribo and rna files: txtID -> RPKM
    riboDict, rnaDict = {}, {}
    ##
    with open(riboSeqFile) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try: riboDict[parts[0]] = float(parts[1])
                except ValueError: pass
    ##
    with open(rnaSeqFile) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try: rnaDict[parts[0]] = float(parts[1])
                except ValueError: pass
    ##
    ##compute log2(Ribo/RNA) for each txtID with positive RPKM in both files
    ratioDict = {}
    for txtID in riboDict:
        if txtID in rnaDict and riboDict[txtID] > 0 and rnaDict[txtID] > 0:
            ratioDict[txtID] = math.log2(riboDict[txtID] / rnaDict[txtID])
    ##
    ##build grid: columns = TL/CDS/UTR, rows = labels
    regions = ['TL', 'CDS', 'UTR']
    graphWidth, graphHeight = 6.0, 6.0
    xSpacing = graphWidth * 1.15
    ySpacing = graphHeight * 1.3
    nRows = len(editFreqDict)
    ##painter for CDS/UTR y-axes: suppress tick labels and title, keep ticks and line
    noLabelPainter = graph.axis.painter.regular(
        labelattrs=None, titleattrs=None
    )
    rhoMatrix = {label: {region: float('nan') for region in regions}
                 for label, _ in editFreqDict}
    c = canvas.canvas()
    hasPlots = False
    for rowIdx, (label, txDict) in enumerate(editFreqDict):
        refYAxis = None##this row's leftmost-plotted column's y-axis object, so later
        ##columns can link to it (see below) rather than each picking their own tick spacing.
        for colIdx, region in enumerate(regions):
            ##collect per-txtID avg edit freq for transcripts with >= N reads in this region
            xData, yData = [], []
            for txtID, regionDict in txDict.items():
                reads = regionDict.get(region, [])
                if len(reads) < N:
                    continue
                if txtID not in ratioDict:
                    continue
                avgEditFreq = sum(ef for ef, na in reads) / len(reads)
                xData.append(ratioDict[txtID])
                yData.append(avgEditFreq)
            if len(xData) < 3:
                print('Skipping %s %s: only %d transcripts with data' % (label, region, len(xData)))
                continue
            rho, pval = spearmanr(xData, yData)
            rhoMatrix[label][region] = rho
            slope, intercept, _, _, _ = linregress(xData, yData)
            xMin, xMax = min(xData), max(xData)
            regLine = [(xMin, slope * xMin + intercept), (xMax, slope * xMax + intercept)]
            ##link this row's y-axis across TL/CDS/UTR: the first column plotted in this row
            ##gets the real (labeled) axis, and every later column in the row reuses that SAME
            ##axis object via linkedaxis (just swapping in noLabelPainter) rather than building
            ##its own independent graph.axis.linear -- two separately-constructed axes that
            ##happen to share min=0,max=1 can still pick different tick spacings from each
            ##other (pyx's automatic tick partitioning depends on whether labels are drawn),
            ##which is what was producing 0.1 ticks on TL vs 0.125 ticks on CDS/UTR.
            yAxis = (graph.axis.linear(min=0, max=1, title='Avg Edit Freq') if refYAxis is None
                     else graph.axis.linkedaxis(refYAxis, painter=noLabelPainter))
            g = graph.graphxy(
                width=graphWidth, height=graphHeight,
                xpos=colIdx * xSpacing,
                ypos=rowIdx * ySpacing,
                x=graph.axis.linear(title='log2(Ribo/RNA)'),
                y=yAxis,
            )
            if refYAxis is None:
                refYAxis = g.axes['y']
            g.plot(graph.data.points(list(zip(xData, yData)), x=1, y=2),
                   [graph.style.symbol(graph.style.symbol.circle, size=0.05,
                                       symbolattrs=[color.rgb.black])])
            g.plot(graph.data.points(regLine, x=1, y=2),
                   [graph.style.line([color.rgb.red])])
            c.insert(g)
            hasPlots = True
            ##rho and p-value in top-left inside the plot area
            textX   = colIdx * xSpacing + 0.3
            textTop = rowIdx * ySpacing + graphHeight - 0.3
            c.text(textX, textTop,       r'$\rho = %.2f$' % rho,  [text.halign.left, text.valign.top])
            c.text(textX, textTop - 0.5, r'$p = %.2g$'   % pval,  [text.halign.left, text.valign.top])
    ##column labels above the top row
    topY = (nRows - 1) * ySpacing + graphHeight
    for colIdx, region in enumerate(regions):
        c.text(colIdx * xSpacing + graphWidth / 2, topY + 0.4, region,
               [text.halign.center, text.valign.bottom])
    ##row labels to the left of each TL plot
    for rowIdx, (label, _) in enumerate(editFreqDict):
        c.text(-2.5, rowIdx * ySpacing + graphHeight / 2, label,
               [text.halign.right, text.valign.middle])
    if hasPlots:
        c.writePDFfile(outPrefix + '.riboRNA.editFreq.pdf')
        print('RiboRNA analysis saved to %s.riboRNA.editFreq.pdf' % outPrefix)
    else:
        print('RiboRNA analysis: no cells had >= 3 transcripts with matching ratio data; no PDF written.')
    ##heatmap of rho values: rows = labels, columns = regions
    allRhoVals = [rhoMatrix[lbl][reg] for lbl, _ in editFreqDict for reg in regions
                  if not math.isnan(rhoMatrix[lbl][reg])]
    if allRhoVals:
        rMin = min(allRhoVals)
        rMax = max(allRhoVals)
        ##cividis, but inverted relative to the reproducibility heatmap above: rho values here
        ##are typically negative, so the MOST NEGATIVE value (rMin) gets yellow and the value
        ##closest to rMax (usually nearest zero) gets dark -- same data-range scaling as
        ##before, just with the min/max ends of the color scale swapped.
        def rho_to_color(r):
            if math.isnan(r):
                return color.rgb(0.7, 0.7, 0.7)
            t = (max(rMin, min(rMax, r)) - rMin) / (rMax - rMin) if rMax > rMin else 0.0
            return cividisColor(1.0 - t)
        cellW, cellH = 2.0, 1.5
        nLabels = len(editFreqDict)
        cHeat = canvas.canvas()
        for rowIdx, (label, _) in enumerate(editFreqDict):
            for colIdx, region in enumerate(regions):
                rval = rhoMatrix[label][region]
                x = colIdx * cellW
                y = (nLabels - 1 - rowIdx) * cellH
                rect = path.rect(x, y, cellW, cellH)
                cHeat.fill(rect, [rho_to_color(rval)])
                cHeat.stroke(rect, [style.linewidth(0.01)])
                if not math.isnan(rval):
                    cHeat.text(x + cellW/2, y + cellH/2, '%.2f' % rval,
                               [text.halign.center, text.valign.middle])
        ##column headers (TL, CDS, UTR) below the grid
        for colIdx, region in enumerate(regions):
            cHeat.text(colIdx * cellW + cellW/2, -0.2, region,
                       [text.halign.center, text.valign.top])
        ##row labels left of the grid
        for rowIdx, (label, _) in enumerate(editFreqDict):
            cHeat.text(-0.1, (nLabels - 1 - rowIdx) * cellH + cellH/2, label,
                       [text.halign.right, text.valign.middle])
        ##colorbar
        barX   = len(regions) * cellW + 0.8
        barW   = 0.5
        barH   = nLabels * cellH
        nSteps = 100
        stepH  = barH / nSteps
        for k in range(nSteps):
            t = k / (nSteps - 1)
            cHeat.fill(path.rect(barX, k * stepH, barW, stepH + 0.01), [cividisColor(1.0 - t)])
        cHeat.stroke(path.rect(barX, 0, barW, barH), [style.linewidth(0.02)])
        for rTick in [rMin + i * (rMax - rMin) / 4 for i in range(5)]:
            yTick = ((rTick - rMin) / (rMax - rMin)) * barH
            cHeat.stroke(path.line(barX + barW, yTick, barX + barW + 0.2, yTick),
                         [style.linewidth(0.02)])
            cHeat.text(barX + barW + 0.3, yTick, '%.2f' % rTick,
                       [text.halign.left, text.valign.middle])
        cHeat.text(barX + barW/2, barH + 0.3, r'$\rho$',
                   [text.halign.center, text.valign.bottom])
        cHeat.writePDFfile(outPrefix + '.riboRNA.rhoHeatmap.pdf')
        print('Rho heatmap saved to %s.riboRNA.rhoHeatmap.pdf' % outPrefix)
    ##
    return cdsEditDict

def cdsLengthVersusEditFreq(cdsEditDict,cdsLengths2,outPrefix):
    """
    cdsEditDict is of the format:
    {label:{'txtID':cdsAvgEditFreq}} 
    cdsLengths is of the format:
    {'txtID':cdsLength}
    For each label in cdsEditDict, will plot cdsAvgEditFreq (y-axis) vs cdsLength (x-axis) 
    for each transcript_id in that label.
    Will place the spearmanr rho in the top right of the interior of the plot, and the p-value just below that.
    Will save as a pdf with the name outPrefix+'.cdsLengthVsEditFreq.pdf'.

    Written by Claude. Proofread, edited by JA.
    """
    ##first, strip of '_mRNA' suffix from txtIDs in cdsLengths
    cdsLengths={k.rstrip('_mRNA'): -v for k, v in cdsLengths2.items()}
    ##
    graphWidth, graphHeight = 8.0, 6.0
    xSpacing = graphWidth * 1.15 + 1.0
    c = canvas.canvas()
    hasPlots = False
    for colIdx, label in enumerate(sorted(cdsEditDict.keys())):
        txFreqs = cdsEditDict[label]
        xData, yData = [], []
        for txtID, avgEditFreq in txFreqs.items():
            if txtID in cdsLengths:
                xData.append(cdsLengths[txtID])
                yData.append(avgEditFreq)
        if len(xData) < 3:
            print('Skipping %s: only %d transcripts with CDS length data' % (label, len(xData)))
            continue
        rho, pval = spearmanr(xData, yData)
        g = graph.graphxy(
            width=graphWidth, height=graphHeight,
            xpos=colIdx * xSpacing,
            x=graph.axis.log(title='CDS Length (nt)'),
            y=graph.axis.linear(min=0, max=1, title='%s  CDS Avg Edit Freq' % label),
        )
        g.plot(graph.data.points(list(zip(xData, yData)), x=1, y=2),
               [graph.style.symbol(graph.style.symbol.circle, size=0.05,
                                   symbolattrs=[color.rgb.black])])
        c.insert(g)
        hasPlots = True
        ##rho and p-value in top-right inside the plot area
        textX   = colIdx * xSpacing + graphWidth - 0.3
        textTop = graphHeight - 0.3
        c.text(textX, textTop,       r'$\rho = %.2f$' % rho,  [text.halign.right, text.valign.top])
        c.text(textX, textTop - 0.5, r'$p = %.2g$'   % pval,  [text.halign.right, text.valign.top])
    if hasPlots:
        c.writePDFfile(outPrefix + '.cdsLengthVsEditFreq.pdf')
        print('CDS length vs edit freq saved to %s.cdsLengthVsEditFreq.pdf' % outPrefix)
    else:
        print('cdsLengthVersusEditFreq: no labels had >= 3 transcripts with length data; no PDF written.')

def extractEditStrings(libReads):
    """
    libReads is of format [(transcript_id,sites),...] for a single library, i.e.
    perReadSites[label] where perReadSites is extractPerReadSites's return value. sites is
    [(relStart,relStop,edit,refNt,txtID),...] for every position in that read that mapped
    uniquely into gtfDict, regardless of nt identity.

    Loops through every read and returns:
    {'TL':{transcript_id:[editString,...]},
    'CDS':{transcript_id:[editString,...]},
    'UTR':{transcript_id:[editString,...]}}
    where each editString is a string of consecutive 0/1-containing entries of the edit_string
    extracted from a single read that overlaps with TL, CDS, or UTR (a read spanning more than
    one region contributes an entry to each). Those regions are defined as before: 5'TL is
    (-inf,-25], CDS is [25,-25], and 3'UTR is [25,inf). Only recover edit positions with values
    0 or 1 (ignore 2).

    Reads are grouped here by their own transcript_id so that downstream weighting
    (computeIntereditDistances/convertToFrequency) can give every transcript equal weight
    regardless of how many reads it had. Reads with a missing/empty or RDN-containing
    transcript_id are dropped entirely -- matching metaStartStopAnalysis's restriction --
    since grouping them together would create one bogus shared "transcript" out of every
    un-annotated read, badly distorting that transcript-equal weighting.

    Written by Claude. Proofread and edited by JA.
    """
    editStringsByTranscript = {'TL': collections.defaultdict(list),
                                'CDS': collections.defaultdict(list),
                                'UTR': collections.defaultdict(list)}
    for transcript_id,sites in libReads:
        if not transcript_id or 'RDN' in transcript_id:
            continue##matches metaStartStopAnalysis's restriction; see docstring above.
        regionStrs = {'TL': '', 'CDS': '', 'UTR': ''}
        for relStart,relStop,edit,refNt,txtID in sites:
            if edit not in (0, 1):
                continue##skips 2, which is a gap
            if relStart < -25:
                regionStrs['TL'] += str(edit)
            elif relStart >= 25 and relStop <= -25:
                regionStrs['CDS'] += str(edit)
            elif relStop > 25:
                regionStrs['UTR'] += str(edit)
        ##
        for region in regionStrs:
            if regionStrs[region]:
                editStringsByTranscript[region][transcript_id].append(regionStrs[region])
    ##
    return {region: dict(byTxt) for region, byTxt in editStringsByTranscript.items()}

def computeIntereditDistances(editStringsByTranscript):
    """
    editStringsByTranscript is of the format returned by extractEditStrings:
    {'TL':{transcript_id:[editString,...]},
    'CDS':{transcript_id:[editString,...]},
    'UTR':{transcript_id:[editString,...]}}

    For each of 5'TL, CDS, and 3'UTR, and for each transcript_id, pools that transcript's own
    reads' edit strings into one raw {distance:ct} tally (summed across all of that
    transcript's reads -- this is the per-transcript raw count that convertToFrequency will
    later normalize and average across transcripts, so every transcript ends up weighted
    equally regardless of its read count). If an edit sequence begins with or ends with 0, do
    not count the distance to the next 1. This avoids spurious distances from the edges of
    edit strings at TL/CDS/UTR boundaries.

    Returns an object of the format:
    {'TL':{transcript_id:distDictTL},
    'CDS':{transcript_id:distDictCDS},
    'UTR':{transcript_id:distDictUTR}}
    A transcript_id whose reads generate no distances at all (e.g. too few edits) is omitted
    from that region's dict entirely, rather than included with an empty distDict.

    Written by Claude. Proofread and edited by JA.
    """
    distsByTranscript = {}
    for region, byTxt in editStringsByTranscript.items():
        distsByTranscript[region] = {}
        for txtID, editStrings in byTxt.items():
            distDict = collections.defaultdict(int)
            for editString in editStrings:
                ##only distances between actual 1's are counted, so edges (which are 0)
                ##never contribute a spurious distance.
                onePositions = [i for i, ch in enumerate(editString) if ch == '1']
                for prevPos, nextPos in zip(onePositions, onePositions[1:]):
                    distDict[nextPos - prevPos] += 1
            if distDict:
                distsByTranscript[region][txtID] = dict(distDict)
    ##
    return distsByTranscript

def convertToFrequency(distsByTranscript):
    """
    distsByTranscript is of the format returned by computeIntereditDistances:
    {'TL':{transcript_id:distDictTL},
    'CDS':{transcript_id:distDictCDS},
    'UTR':{transcript_id:distDictUTR}}
    Each distDict is a dictionary of the format {distance:ct}, pooled across all of that
    transcript_id's own reads.

    For each region, first normalizes every transcript's own distDict to a per-transcript
    frequency distribution (ct/totalCounts, so it sums to 1 across that transcript's own
    distances), then averages those per-transcript frequencies across transcripts -- every
    transcript contributes equally to a region's frequency at a given distance, regardless of
    how many reads or edits it had. This mirrors processMeta/mkPlot's transcript-equal
    weighting for the metaStartStop analyses: a transcript that doesn't have a given distance
    at all simply doesn't contribute to that distance's average (rather than contributing a 0),
    the same way a transcript missing data at a given relPos is absent from that position's
    list rather than zero-filled.

    Returns an object of the format:
    {'TL':distDictFreqTL,
    'CDS':distDictFreqCDS,
    'UTR':distDictFreqUTR}

    Written by Claude. Proofread and edited by JA.
    """
    distDictFreqs = {}
    for region, byTxt in distsByTranscript.items():
        perDistanceFreqs = collections.defaultdict(list)
        for txtID, distDict in byTxt.items():
            totalCounts = sum(distDict.values())
            for distance, ct in distDict.items():
                perDistanceFreqs[distance].append(ct / totalCounts)
        distDictFreqs[region] = {distance: sum(freqs) / len(freqs)
                                  for distance, freqs in perDistanceFreqs.items()}
    ##
    return distDictFreqs

def theShuffler(editStringsByTranscript):
    """
    editStringsByTranscript is of the format returned by extractEditStrings:
    {'TL':{transcript_id:[editString,...]},
    'CDS':{transcript_id:[editString,...]},
    'UTR':{transcript_id:[editString,...]}}
    Shuffles the characters of every individual edit string, preserving the
    region/transcript_id grouping (transcript_id membership and read count are properties of
    the real data, not of any one shuffle, so they stay fixed across randomizations) -- this
    lets the shuffled output run through the same transcript-equal-weighted
    computeIntereditDistances/convertToFrequency pipeline as the observed data. Returns an
    object of the same format as input.

    Written by Claude, proofread and edited by JA
    """
    shuffledByTranscript = {}
    for region, byTxt in editStringsByTranscript.items():
        shuffledByTranscript[region] = {}
        for txtID, editStrings in byTxt.items():
            shuffledStrings = []
            for editString in editStrings:
                chars = list(editString)
                random.shuffle(chars)
                shuffledStrings.append(''.join(chars))
            shuffledByTranscript[region][txtID] = shuffledStrings
    ##
    return shuffledByTranscript

def computeZscoreAndPvals(observedDistanceFrequencies,randomizationMetaData):
    """
    observedDistanceFrequencies is of the format:
    {'TL':distDictFreqTL,
    'CDS':distDictFreqCDS,
    'UTR':distDictFreqUTR}
    Each distDictFreq is of the format {distance:freq}
    randomizationMetaData is a list of objects of the same format as 
    observedDistanceFrequencies.

    For each region (TL/CDS/UTR), for each distance, this function will compute the 
    distribution of freqs from randomizationMetaData. The function will compute a
    mean and standard deviation for that distribution, and use that to calculate
    a z-score for the corresponding value from the same region from 
    observedDistanceFrequencies. The function will also create an empirical p-value,
    which is the fraction of times the region/distance/freq value exceeds that seen in
    the randomizations.

    The function will return two objects. The first is a zscore object of the format:
    {'TL':{distance:zScore}
    'CDS':{distance:zScore}
    'UTR':{distance:zScore}}
    The second object is a pval object of the format:
    {'TL':{distance:pval}
    'CDS':{distance:pval}
    'UTR':{distance:pval}}

    For the empirical pval, let's say the value is >0.5, meaning that the observed value
    exceeds more than 50% of the randomizations. In that case, subtract the p-value from 1
    (1-frac), take the log10, and report it as a positive value. If the empirical pval is
    <=0.5, take the log10, and the value will be negative.
    
    Code written by Claude, proofread and edited by JA
    """
    zScores = {}
    pVals = {}
    nRand = len(randomizationMetaData)
    for region, distFreqDict in observedDistanceFrequencies.items():
        zScores[region] = {}
        pVals[region] = {}
        for distance, obsFreq in distFreqDict.items():
            ##missing distance entries in a given randomization mean that distance never occurred, i.e. freq=0
            randFreqs = [randDict[region].get(distance, 0) for randDict in randomizationMetaData]
            meanFreq = sum(randFreqs) / nRand
            variance = sum((rf - meanFreq) ** 2 for rf in randFreqs) / nRand
            stdFreq = math.sqrt(variance)
            zScores[region][distance] = (obsFreq - meanFreq) / stdFreq if stdFreq > 0 else float('nan')
            ##
            ##fraction of randomizations that the observed value exceeds.
            ##add-one (Laplace) smoothing keeps this strictly within (0,1), since with many possible
            ##distances (e.g. in the CDS) it's common for a distance to never occur in any randomization,
            ##which would otherwise make fracExceeds exactly 0 or 1 and blow up the log10 below to +/-inf.
            fracExceeds = (sum(1 for rf in randFreqs if obsFreq > rf) + 1) / (nRand + 2)
            if fracExceeds > 0.5:
                pVals[region][distance] = -math.log10(1 - fracExceeds)
            else:
                pVals[region][distance] = math.log10(fracExceeds)
    ##
    return zScores, pVals

def intereditDistanceAnalyzer(perReadSites,outPrefix):
    """
    perReadSites is of format {libName:[(transcript_id,sites),...]}, as returned by
    extractPerReadSites.

    For each library, this function will calculate the distance between edits in the CDS, and
    between edits in the 5'TL, and 3'UTR.

    Will plot the frequency of interedit distances of each size. Will use linear-scale for x-axis
    and a log-scale for y-axis.

    This function is derived from intereditDistanceAnalyzerWithRandomizations, but removed
    the randomizations.
    """
    print('Performing interedit analysis...')
    metaData=[]
    ##first, extract edit string for regions of a read that cover a uniquely assignable CDS,
    ##and regions of a read that cover a uniquely assignable 5'TL, and 3'UTR. Keep all in
    ##separate objections.
    for label,libReads in perReadSites.items():
        ##
        print('Analyzing %s...'%(label))
        editStringLists=extractEditStrings(libReads)
        ##editStringLists is of the format:
        ##{'TL':{transcript_id:[editString,...]},
        ##'CDS':{transcript_id:[editString,...]},
        ##'UTR':{transcript_id:[editString,...]}}
        ##
        ##now compute the interedit distances for those strings, per transcript_id.
        observedDistanceLists=computeIntereditDistances(editStringLists)
        ##observedDistanceLists is of the format:
        ##{'TL':{transcript_id:distDictTL},
        ##'CDS':{transcript_id:distDictCDS},
        ##'UTR':{transcript_id:distDictUTR}}
        ##
        ##now convert to frequency of interedit distances -- every transcript_id is weighted
        ##equally regardless of its read count (see convertToFrequency's docstring).
        observedDistanceFrequencies=convertToFrequency(observedDistanceLists)
        ##
        ##
        metaData.append((label,observedDistanceFrequencies))
    ##
    ##now plot the data. For each of TL/CDS/UTR, create a square frequency plot, using a
    ##linear-scale x-axis and a log-scale y-axis.
    ##array the plots left>right TL>CDS>UTR.
    ##plot all the labels on the same plot, as different colors.
    ##place the key to the far right of the right-most plot, outside the plot.
    ##link the y-axis across TL/CDS/UTR such that the y-axis is only displayed on the left-most plot.
    regions = ['TL', 'CDS', 'UTR']
    graphSize = 6.0
    xSpacing = graphSize * 1.15
    noLabelPainter = graph.axis.painter.regular(labelattrs=None, titleattrs=None)
    c = canvas.canvas()
    metaData = sorted(metaData, key=lambda x: x[0])
    ##one color per library, via assignColors (same scheme as mkPlot/metaStartStopAnalysis,
    ##rather than common.colors(), whose small palette starts repeating colors after 8 entries)
    libColors = assignColors(label for label, obsFreqs in metaData)
    ##
    xAxisMin, xAxisMax = 10, 150
    ##
    ##pre-compute the y-value range, pooled across ALL regions/labels (not just the left-most
    ##column), since the y-axis is linked across TL/CDS/UTR. This also guards against pyx's
    ##"zero axis range" error when the data happens to be a single repeated value (or empty).
    allFreqVals = []
    for label, obsFreqs in metaData:
        for region in regions:
            allFreqVals.extend(obsFreqs[region].values())

    freqMin, freqMax = axisRange(allFreqVals, isLog=True, fallback=(0.001, 1))
    ##
    refYAxis = None##TL column's y-axis object, so CDS/UTR columns can link to it
    for colIdx, region in enumerate(regions):
        isFirstCol = (colIdx == 0)
        isLastCol  = (colIdx == len(regions) - 1)
        ##
        freqYAxis = (graph.axis.log(min=freqMin, max=freqMax, title='Frequency') if isFirstCol
                     else graph.axis.linkedaxis(refYAxis, painter=noLabelPainter))
        freqKwargs = dict(
            width=graphSize, height=graphSize,
            xpos=colIdx * xSpacing, ypos=0,
            x=graph.axis.linear(min=xAxisMin, max=xAxisMax, title='Interedit Distance (%s)' % region),
            y=freqYAxis
        )
        if isLastCol:
            ##key for the whole figure lives on the right-most (UTR) plot, outside the plot area
            freqKwargs['key'] = graph.key.key(pos='mr', hinside=0)
        freqGraph = graph.graphxy(**freqKwargs)
        if isFirstCol:
            refYAxis = freqGraph.axes['y']
        ##
        for label, observedDistanceFrequencies in metaData:
            freqData = sorted(observedDistanceFrequencies[region].items())
            if freqData:
                freqGraph.plot(graph.data.points(freqData, x=1, y=2, title=label),
                                [graph.style.line([libColors[label]]),
                                 graph.style.symbol(graph.style.symbol.circle, size=0.07,
                                                    symbolattrs=[libColors[label]])])
        ##
        c.insert(freqGraph)
    c.writePDFfile(outPrefix + '.intereditDistanceFrequency.pdf')
    print('Interedit distance analysis saved to %s.intereditDistanceFrequency.pdf' % outPrefix)


def intereditDistanceAnalyzerWithRandomizations(perReadSites,outPrefix,randomizations=1000):
    """
    perReadSites is of format {libName:[(transcript_id,sites),...]}, as returned by
    extractPerReadSites.

    For each library, this function will calculate the distance between edits in the CDS, and
    between edits in the 5'TL, and 3'UTR.

    Will plot the frequency of interedit distances of each size.

    Will also extract and shuffle edit strings, and then recalculate a control distribution of
    interedit distances using these random shuffles. It will do this 1000 (or a similarly high
    number) of times. It will then plot the z-score for over/under representation of the observed
    frequency relative to this control frequency. It will also plot the empirical p-value, which
    is the number of randomizations that exceed the observed frequency value.
    """
    print('Performing interedit analysis...')
    print('This will perform randomizations, which may take a long time (hours?).')
    print('As such, ensure you\'re running on "screen" in case you get disconnected.')
    metaData=[]
    ##first, extract edit string for regions of a read that cover a uniquely assignable CDS,
    ##and regions of a read that cover a uniquely assignable 5'TL, and 3'UTR. Keep all in
    ##separate objections.
    for label,libReads in perReadSites.items():
        ##
        print('Analyzing %s...'%(label))
        editStringLists=extractEditStrings(libReads)
        ##editStringLists is of the format:
        ##{'TL':{transcript_id:[editString,...]},
        ##'CDS':{transcript_id:[editString,...]},
        ##'UTR':{transcript_id:[editString,...]}}
        ##
        ##now compute the interedit distances for those strings, per transcript_id.
        observedDistanceLists=computeIntereditDistances(editStringLists)
        ##observedDistanceLists is of the format:
        ##{'TL':{transcript_id:distDictTL},
        ##'CDS':{transcript_id:distDictCDS},
        ##'UTR':{transcript_id:distDictUTR}}
        ##
        ##now convert to frequency of interedit distances -- every transcript_id is weighted
        ##equally regardless of its read count (see convertToFrequency's docstring).
        observedDistanceFrequencies=convertToFrequency(observedDistanceLists)
        ##
        ##
        ##now commence randomizations
        ##initialize an object to keep track of the randomizations
        randomizationMetaData=[]
        ##
        for ii in range(randomizations):
            ##
            if ii % 10 == 0:
                print('Working on randomization %s with parquetFiles associated with label %s...'%(ii,label))
            ##
            ##now for every edit string within each of TL, CDS, UTR, shuffle it.
            ##
            shuffledStrings=theShuffler(editStringLists)
            ##
            ##now recompute interedit distances as before.
            ##
            randomizedDistanceLists=computeIntereditDistances(shuffledStrings)
            ##
            ##now recompute interedit frequencies as before.
            ##
            randomizedDistanceFrequencies=convertToFrequency(randomizedDistanceLists)
            ##
            ##append to randomizationMetaData
            randomizationMetaData.append(randomizedDistanceFrequencies)
        ##
        ##compute z-score, empirical p-value
        zscores,pvals=computeZscoreAndPvals(observedDistanceFrequencies,randomizationMetaData)
        ##
        metaData.append((label,observedDistanceFrequencies,zscores,pvals))
    ##
    ##now plot the data. For each of TL/CDS/UTR, create three plots, stacked on top of each other.
    ##For the top plot, show the raw frequencies from observedDistanceFrequencies.
    ##For the middle plot, show the zscore. For the bottom plot, show the pvalue.
    ##array the plots left>right TL>CDS>UTR.
    ##plot all the labels on the same plot, as different colors.
    ##place the key to the far right of the right-most plot, outside the plot.
    ##link the y-axes for the three plots (frequency, zscore, pval) across TL/CDS/UTR such that
    ##the y-axis is only displayed on the left-most plot.
    regions = ['TL', 'CDS', 'UTR']
    graphWidth, graphHeight = 6.0, 3.0
    gap = 0.6
    rowHeight = graphHeight + gap
    xSpacing = graphWidth * 1.15
    noLabelPainter = graph.axis.painter.regular(labelattrs=None, titleattrs=None)
    c = canvas.canvas()
    metaData = sorted(metaData, key=lambda x: x[0])
    libColors = assignColors(label for label, obsFreqs, zscores, pvals in metaData)
    ##
    xAxisMin, xAxisMax = 10, 150
    ##
    ##pre-compute the y-value range for each row, pooled across ALL regions/labels (not just
    ##the left-most column), since the y-axes are linked across TL/CDS/UTR. This also guards
    ##against pyx's "zero axis range" error when the data for a row happens to be a single
    ##repeated value (or empty).
    ##The z-score range is further restricted to distances within [xAxisMin,xAxisMax], since
    ##that's the only portion of the data that will actually be visible on the plot.
    allFreqVals, allZscoreVals, allPvalVals = [], [], []
    for label, obsFreqs, zscores, pvals in metaData:
        for region in regions:
            allFreqVals.extend(obsFreqs[region].values())
            allZscoreVals.extend(z for d, z in zscores[region].items()
                                  if not math.isnan(z) and xAxisMin <= d <= xAxisMax)
            allPvalVals.extend(p for p in pvals[region].values()
                                if not math.isnan(p) and not math.isinf(p))

    freqMin, freqMax = axisRange(allFreqVals, isLog=True, fallback=(0.001, 1))
    _, zscoreMax = axisRange(allZscoreVals)
    zscoreMin = 0
    pvalMin, pvalMax = axisRange(allPvalVals)
    ##
    refYAxes = {}##row name -> TL column's y-axis object, so CDS/UTR columns can link to it
    for colIdx, region in enumerate(regions):
        isFirstCol = (colIdx == 0)
        isLastCol  = (colIdx == len(regions) - 1)
        ##
        ##bottom row: signed log10(empirical p-value)
        pvalYAxis = (graph.axis.linear(min=pvalMin, max=pvalMax,
                                        title='Signed log10(Empirical p-value)') if isFirstCol
                     else graph.axis.linkedaxis(refYAxes['pval'], painter=noLabelPainter))
        pvalGraph = graph.graphxy(
            width=graphWidth, height=graphHeight,
            xpos=colIdx * xSpacing, ypos=0,
            x=graph.axis.log(min=xAxisMin, max=xAxisMax, title='Interedit Distance (%s)' % region),
            y=pvalYAxis
        )
        if isFirstCol:
            refYAxes['pval'] = pvalGraph.axes['y']
        ##
        ##middle row: z-score
        zscoreYAxis = (graph.axis.linear(min=zscoreMin, max=zscoreMax, title='Z-score') if isFirstCol
                       else graph.axis.linkedaxis(refYAxes['zscore'], painter=noLabelPainter))
        zscoreGraph = graph.graphxy(
            width=graphWidth, height=graphHeight,
            xpos=colIdx * xSpacing, ypos=rowHeight,
            x=graph.axis.linkedaxis(pvalGraph.axes['x']),
            y=zscoreYAxis
        )
        if isFirstCol:
            refYAxes['zscore'] = zscoreGraph.axes['y']
        ##
        ##top row: raw interedit distance frequency
        freqYAxis = (graph.axis.log(min=freqMin, max=freqMax, title='Frequency') if isFirstCol
                     else graph.axis.linkedaxis(refYAxes['freq'], painter=noLabelPainter))
        freqKwargs = dict(
            width=graphWidth, height=graphHeight,
            xpos=colIdx * xSpacing, ypos=2 * rowHeight,
            x=graph.axis.linkedaxis(pvalGraph.axes['x']),
            y=freqYAxis
        )
        if isLastCol:
            ##key for the whole figure lives on the top-right (UTR) plot, outside the plot area
            freqKwargs['key'] = graph.key.key(pos='mr', hinside=0)
        freqGraph = graph.graphxy(**freqKwargs)
        if isFirstCol:
            refYAxes['freq'] = freqGraph.axes['y']
        ##
        for label, observedDistanceFrequencies, zscores, pvals in metaData:
            freqData   = sorted(observedDistanceFrequencies[region].items())
            zscoreData = sorted((d, z) for d, z in zscores[region].items() if not math.isnan(z))
            pvalData   = sorted((d, p) for d, p in pvals[region].items()
                                 if not math.isnan(p) and not math.isinf(p))
            ##
            if freqData:
                freqGraph.plot(graph.data.points(freqData, x=1, y=2, title=label),
                                [graph.style.line([libColors[label]]),
                                 graph.style.symbol(graph.style.symbol.circle, size=0.07,
                                                    symbolattrs=[libColors[label]])])
            if zscoreData:
                zscoreGraph.plot(graph.data.points(zscoreData, x=1, y=2),
                                  [graph.style.line([libColors[label]]),
                                   graph.style.symbol(graph.style.symbol.circle, size=0.07,
                                                      symbolattrs=[libColors[label]])])
            if pvalData:
                pvalGraph.plot(graph.data.points(pvalData, x=1, y=2),
                                [graph.style.line([libColors[label]]),
                                 graph.style.symbol(graph.style.symbol.circle, size=0.07,
                                                    symbolattrs=[libColors[label]])])
        ##
        c.insert(pvalGraph)
        c.insert(zscoreGraph)
        c.insert(freqGraph)
    c.writePDFfile(outPrefix + '.intereditDistanceZscorePval.pdf')
    print('Interedit distance analysis with randomizations saved to %s.randos.intereditDistance.pdf' % outPrefix)

def plotEditFreqReproAcrossReplicates(editFreqDict,N,outPrefix):
    """
    editFreqDict is of the format:
    [(label, {txtID: {'TL': [(editFreq,numAs),...
      'CDS': [(editFreq,numAs),...], 'UTR': [(editFreq,numAs),...]}})]
    Using only the CDS data, this function will plot the reproducibility of edit frequencies across replicates.
    To do so, it will compute an average edit frequency for each transcript_id in the CDS region, containing at
    least N tuples (reads). It will then obtain the spearmanr correlation coefficient and p-value for the
    average edit frequencies between all libraries. It will then plot the heatmap of those correlation
    coefficients, and the heatmap of the p-values, side-by-side. The heatmaps will be saved to a PDF file
    with the prefix outPrefix+'.editFreqReproCDSOnly.pdf'.

    Both heatmaps use cividis, and -- like readCountAndEditFreqRepro's editFreqReproHeatmap --
    censor the diagonal (a library compared to itself, always rho=1/p~0 and uninformative)
    rather than let it dominate the color scale. The SpearmanR heatmap colors by the raw rho
    value (high rho = yellow, matching the "yellow = more correlated" convention used
    elsewhere); the p-value heatmap colors by -log10(p) so a grid of very small p-values is
    actually visually distinguishable (small p = yellow, i.e. more significant = more
    correlated = yellow, same convention), while still printing the raw p-value as the cell
    text.
    """
    ##build per-library, per-transcript average CDS edit frequency, restricted to
    ##transcript_ids with >= N reads in the CDS region (same restriction/computation as
    ##analyzeRiboRNA's cdsEditDict).
    cdsEditDict = {}
    for label, txDict in editFreqDict:
        cdsEditDict[label] = {}
        for txtID, regionDict in txDict.items():
            reads = regionDict.get('CDS', [])
            if len(reads) >= N:
                cdsEditDict[label][txtID] = sum(ef for ef, na in reads) / len(reads)
    ##
    ##compute pairwise SpearmanR (and its p-value) between every pair of libraries, using
    ##whatever transcript_ids both libraries have a qualifying CDS average for.
    libLabels = sorted(cdsEditDict.keys())
    nLibs = len(libLabels)
    spearmanMatrix = {}
    pValMatrix = {}
    for lib1 in libLabels:
        spearmanMatrix[lib1] = {}
        pValMatrix[lib1] = {}
        for lib2 in libLabels:
            common2 = sorted(set(cdsEditDict[lib1]) & set(cdsEditDict[lib2]))
            if len(common2) > 2:
                x = [cdsEditDict[lib1][t] for t in common2]
                y = [cdsEditDict[lib2][t] for t in common2]
                rval, pval = spearmanr(x, y)
            else:
                rval, pval = float('nan'), float('nan')
            spearmanMatrix[lib1][lib2] = rval
            pValMatrix[lib1][lib2] = pval
    ##
    ##-log10(p) for the p-value heatmap's color scale (see docstring); p<=0 (float underflow
    ##on an extremely significant result) is floored to the smallest representable positive
    ##float rather than producing an undefined/infinite log.
    TINY = sys.float_info.min
    def negLog10(p):
        return float('nan') if math.isnan(p) else -math.log10(max(p, TINY))
    ##
    ##color scales, excluding the diagonal (censored below -- see docstring).
    allRvals = [spearmanMatrix[l1][l2] for l1 in libLabels for l2 in libLabels
                if l1 != l2 and not math.isnan(spearmanMatrix[l1][l2])]
    allNegLogPVals = [negLog10(pValMatrix[l1][l2]) for l1 in libLabels for l2 in libLabels
                       if l1 != l2 and not math.isnan(pValMatrix[l1][l2])]
    rMin, rMax = (min(allRvals), max(allRvals)) if allRvals else (0.0, 1.0)
    pMin, pMax = (min(allNegLogPVals), max(allNegLogPVals)) if allNegLogPVals else (0.0, 1.0)
    ##
    def scaledColor(colorVal, vMin, vMax):
        if math.isnan(colorVal):
            return color.rgb(0.7, 0.7, 0.7)
        t = (max(vMin, min(vMax, colorVal)) - vMin) / (vMax - vMin) if vMax > vMin else 0.0
        return cividisColor(t)
    ##
    cellSize = 1.5
    gapBetweenHeatmaps = 3.0
    cHeat = canvas.canvas()
    ##
    def drawHeatmap(matrix, xOffset, vMin, vMax, valueToColorVal, cellText, colorbarTitle):
        """
        Draws one libLabels x libLabels heatmap (plus its own colorbar) into cHeat, with its
        left edge at xOffset. matrix is {lib1:{lib2:rawValue}}; valueToColorVal maps a
        rawValue to whatever's actually used for the vMin/vMax color scale (identity for the
        SpearmanR heatmap, negLog10 for the p-value heatmap); cellText formats a rawValue for
        display in a cell. Returns the x-coordinate just past this heatmap's colorbar, so the
        caller can offset the next heatmap past it.
        """
        for i, lib1 in enumerate(libLabels):
            for j, lib2 in enumerate(libLabels):
                rawVal = matrix[lib1][lib2]
                x = xOffset + j * cellSize
                y = (nLibs - 1 - i) * cellSize
                rect = path.rect(x, y, cellSize, cellSize)
                if lib1 == lib2:
                    ##censored self-comparison -- always rho=1/p~0, not informative.
                    cHeat.fill(rect, [color.rgb(0.7, 0.7, 0.7)])
                    cHeat.stroke(rect, [style.linewidth(0.01)])
                    continue
                cHeat.fill(rect, [scaledColor(valueToColorVal(rawVal), vMin, vMax)])
                cHeat.stroke(rect, [style.linewidth(0.01)])
                if not math.isnan(rawVal):
                    cHeat.text(x + cellSize / 2, y + cellSize / 2, cellText(rawVal),
                               [text.halign.center, text.valign.middle])
        ##x-axis labels (below the grid, rotated 45°)
        for j, lib in enumerate(libLabels):
            cHeat.text(xOffset + j * cellSize + cellSize / 2, -0.2, lib,
                       [text.halign.right, text.valign.middle, trafo.rotate(45)])
        ##y-axis labels (left of the grid) -- only once, on the left-most heatmap.
        if xOffset == 0:
            for i, lib in enumerate(libLabels):
                cHeat.text(-0.1, (nLibs - 1 - i) * cellSize + cellSize / 2, lib,
                           [text.halign.right, text.valign.middle])
        ##colorbar: 100 thin strips spanning [vMin,vMax]
        barX   = xOffset + nLibs * cellSize + 0.8
        barW   = 0.5
        barH   = nLibs * cellSize
        nSteps = 100
        stepH  = barH / nSteps
        for k in range(nSteps):
            t = k / (nSteps - 1)
            cHeat.fill(path.rect(barX, k * stepH, barW, stepH + 0.01), [cividisColor(t)])
        cHeat.stroke(path.rect(barX, 0, barW, barH), [style.linewidth(0.02)])
        ##tick marks at 5 evenly spaced values across the actual data range
        for i in range(5):
            vTick = vMin + i * (vMax - vMin) / 4 if vMax > vMin else vMin
            yTick = ((vTick - vMin) / (vMax - vMin)) * barH if vMax > vMin else 0.0
            cHeat.stroke(path.line(barX + barW, yTick, barX + barW + 0.2, yTick),
                         [style.linewidth(0.02)])
            cHeat.text(barX + barW + 0.3, yTick, '%.2f' % vTick,
                       [text.halign.left, text.valign.middle])
        cHeat.text(barX + barW / 2, barH + 0.3, colorbarTitle,
                   [text.halign.center, text.valign.bottom])
        return barX + barW
    ##
    rightEdge = drawHeatmap(spearmanMatrix, 0.0, rMin, rMax,
                             valueToColorVal=lambda r: r,
                             cellText=lambda r: '%.2f' % r,
                             colorbarTitle='SpearmanR')
    drawHeatmap(pValMatrix, rightEdge + gapBetweenHeatmaps, pMin, pMax,
                valueToColorVal=negLog10,
                cellText=lambda p: '%.2g' % p,
                colorbarTitle='-log10(p)')
    ##
    cHeat.writePDFfile(outPrefix + '.editFreqReproCDSOnly.pdf')
    print('CDS-only edit frequency reproducibility heatmap saved to %s.editFreqReproCDSOnly.pdf' % outPrefix)

def main(args):
    gtfFile,N,parquetFiles,riboSeqFile,rnaSeqFile,outPrefix=args[0],args[1],args[2],args[3],args[4],args[5]
    colorsFile=args[6] if len(args)>6 else None
    """
    Quick note: lines with ###### are commented out because they are resource-intensive and not necessary
    to run every time. If you want to run this script for the first time, find and delete the ######
    """
    global COLOR_MAP
    COLOR_MAP=parseColorsFile(colorsFile)
    ##first analyze edit bias around motifs
    motifDict=analyzeMotifs(parquetFiles,outPrefix)
    ##motifDict is of the format:
    ##{libName:{3merSeq:freq}}
    ##
    ##will now analyze reproducibility of edit frequencies across replicates.
    ##will examine different numbers of minimum reads per transcript_id, and 
    ##different numbers of minimum A's per transcript_id.
    readCountAndEditFreqRepro(parquetFiles,int(N),outPrefix)
    ##
    ## parse gtf to get dict of format:
    #{strand:{chr:{absIndx:[(txtName,relStart,relStop)]}}}
    gtfDict,cdsLengths=parseGTF(gtfFile)
    ##
    ##extract every read's per-position data ONCE here, shared by metaStartStopAnalysis,
    ##metaStartStopAnalysisNormalizeForMotifBias, and intereditDistanceAnalyzer below --
    ##they used to each independently re-read and re-walk every parquet file from scratch.
    perReadSites=extractPerReadSites(parquetFiles,gtfDict)
    ##
    ##now plot the read length distribution from perReadSites
    plotReadLengthDistribution(perReadSites,outPrefix+'.readLengthDistribution')
    ##
    editFreqDict=metaStartStopAnalysis(perReadSites,int(N),outPrefix)
    ##now plot reproducibility of edit frequencies across replicates, using the editFreqDict returned by metaStartStopAnalysis
    plotEditFreqReproAcrossReplicates(editFreqDict,int(N),outPrefix)
    ##
    metaStartStopAnalysisByStopCodon(perReadSites,int(N),outPrefix+'.byStopCodon')
    metaStartStopAnalysisByCDSLength(perReadSites,int(N),cdsLengths,outPrefix+'.byCDSLength')
    metaStartStopAnalysisNormalizeForMotifBias(perReadSites,int(N),motifDict,outPrefix+'.normForMotifs')
    metaStartStopAnalysisByCDSLengthNormaliedForMotifBias(perReadSites,int(N),cdsLengths,motifDict,outPrefix+'.byCDSLength.normForMotifs')
    ##editFreqDict is of the format:
    ##[(label, {txtID: {'TL': [(editFreq,numAs),...
    ##  'CDS': [(editFreq,numAs),...], 'UTR': [(editFreq,numAs),...]}})]
    ##will now save editFreqDict to a pickle file for later use.
    #with open(outPrefix+'.editFreqDict.pkl','wb') as f:
    #    pickle.dump(editFreqDict,f)
    ##now unpickle the editFreqDict and analyze the edit frequencies in ribo-seq and rna-seq data.
    #with open(outPrefix+'.editFreqDict.pkl','rb') as f:
    #    editFreqDict=pickle.load(f)
    ##editFreqDict is of the format:
    ##[(label, {txtID: {'TL': [(editFreq,numAs),...
    ##  'CDS': [(editFreq,numAs),...], 'UTR': [(editFreq,numAs),...]}})]
    ##
    ##analyze the relationship between edit frequency and ribo/rna-seq data
    cdsEditDict=analyzeRiboRNA(editFreqDict,riboSeqFile,rnaSeqFile,int(N),outPrefix)
    ##
    ##cdsEditDict is of the format:
    ##{label:{'txtID':cdsAvgEditFreq}} where cdsAvgEditFreq is the average edit frequency across all reads for
    ##that transcript_id in the CDS region, if there are at least N reads in that region.
    ##will now analyze the relationship between cdsAvgEditFreq and cdsLengths
    cdsLengthVersusEditFreq(cdsEditDict,cdsLengths,outPrefix)
    ##
    ##Now analyze the interedit distance
    intereditDistanceAnalyzer(perReadSites,outPrefix)
    #intereditDistanceAnalyzerWithRandomizations(perReadSites,outPrefix,randomizations=1000)

if __name__=='__main__':
    Tee()
    main(sys.argv[1:])