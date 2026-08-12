"""
LT, August 2026

Quick screen for candidate LUTI (Long Undecoded Transcript Isoform) reads:
reads whose aligned footprint (absolute_indices) touches a transcript's
CDS region (+/-100nt flank) genomically upstream (5') of the CDS of the
gene the read's poly-A-proximal (3') end actually belongs to, per the
GTF-derived position map from metaStartStop.parseGTF. That's the signature
of a read that starts upstream of its own gene and reads through into a
neighboring transcript -- consistent with an alternative-TSS LUTI. Since
LUTIs are 5' extended, overlaps that instead occur downstream of the read's
own CDS (e.g. stop-codon readthrough into the next gene) are deliberately
excluded, using genomic-coordinate comparisons (strand-aware) rather than
assuming a particular iteration order for absolute_indices.

Note: the read's own gene is determined here from whichever CDS its 3'-most
touched position falls in, NOT from the parquet's transcript_id column.
transcript_id is assigned by shadowingBamToParquetWithGTF2.py via whichever
transcript has the most total nucleotide overlap with the read -- for a
genuine readthrough transcript that starts deep inside gene A and extends
past its stop codon into gene B, a slim overlap majority can land that
column on gene B even though the read's own 5' end sits inside gene A. That
previously caused this script to misclassify A-reads-through-into-B (a 3'
readthrough of A) as "B extends upstream into A" (a false-positive LUTI
call for B). Anchoring on the 3' end instead avoids that.

Note: metaStartStop.parseGTF only indexes protein_coding CDS regions
(+/-100nt), so this will miss LUTI extensions that stay entirely within a
neighbor's 5'/3' UTR without reaching its CDS. It's a first-pass filter, not
an exhaustive one.

Note: transcript_ids that mostly overlap a gene on the opposite strand
(likely dubious antisense ORFs, e.g. YAL037C-B sitting almost entirely on
top of CDC19/YAL038W) are excluded outright -- see
findAntisenseOverlappingTxtIDs -- both as a read's "own" gene and as an
"other" hit.

Note: transcript_ids in or near the rDNA repeat array (e.g. YLR162W-A/RRT15,
~900nt past the last rRNA gene on chrXII) are also excluded outright -- see
findRDNProximalTxtIDs -- for the same reason: that locus is unreliable to
align to uniquely, and in one real run this single transcript accounted for
71% of all flagged reads, which was an alignment artifact, not a LUTI.

Note: only positions inside an unbroken matched (CIGAR M) run of
>=MIN_MATCH_RUN_LENGTH nt count as "touched" -- see realMatchPositions.
absolute_indices treats a real deletion/intron and a fabricated alignment
gap identically (both just leave read_pos None), so an aligner that bridges
two unrelated loci with a bogus gap -- confirmed on real data from a
--splice_aware run, see doradoAlignToParquetPipeline.py's docstring -- can
otherwise manufacture a spurious upstream "hit" out of a handful of short
coincidental matches scattered through an otherwise non-matching stretch.

Note: a read whose 5' leader was soft-clipped away by the aligner never
touches any CDS in aligned_pairs at all -- that leader isn't a fabricated
gap, it's just absent -- so findSpannedTranscripts can't see it regardless
of MIN_MATCH_RUN_LENGTH. If the parquet was built by
shadowingBamToGTFWithParquet_LUTI.py instead of (or in addition to)
shadowingBamToParquetWithGTF2.py, its luti_rescue_hit/
luti_rescue_other_transcript_ids columns (that script's own realignment of
the clipped leader against the genome, already checked against the same
CDS+/-100nt map) are folded in as a second, independent source of "other"
hits for the same read -- see the source_signal field below. Reads read
via plain pd.read_parquet from an older parquet without those columns
degrade gracefully (row.get returns None) and are scored on
findSpannedTranscripts alone, same as before.

Input: inFile.gtf - gtf-formatted file containing genome annotations.
       parquetDir - a directory of parquet files (from
           shadowingBamToParquetWithGTF2.py, or shadowingBamToGTFWithParquet_LUTI.py
           for the additional rescue signal above). Files are streamed one
           at a time rather than all loaded at once.
       optional: N - the minimum number of total reads (LUTI-spanning or
           not) a transcript_id must have across all parquet files to be
           kept -- coverage filter so downstream analysis isn't run on
           transcripts too lowly-expressed to trust a call for. Default 10.

Output: outPrefix.lutiCandidates.pkl - pickled dict of format
            {transcript_id:[{read_id,chrom,gene_strand,gene_name,
                              parquet_transcript_id,other_transcript_ids,
                              other_gene_names,source_signal,source_parquet}]},
            restricted to transcripts passing the N coverage filter.
            transcript_id (the dict key) is the 3'-anchored own gene
            determined here (falling back to the parquet's own transcript_id
            when only a rescue hit -- no aligned-footprint CDS touch --
            anchors the read, see main()); parquet_transcript_id (in each
            read's record) is the original shadowingBamToParquetWithGTF2.py
            assignment, kept for comparison/debugging. gene_name/
            other_gene_names are looked up from the GTF (falling back to the
            transcript_id itself for unnamed/dubious ORFs). source_signal is
            'overlap', 'rescue', or 'overlap+rescue' depending on which
            signal(s) produced this read's other_transcript_ids.
        outPrefix.lutiCandidates.txt - tab-delimited flattening of the same
            data (one row per candidate read) for manual inspection.

run as python3 findLUTICandidateReads.py inFile.gtf outPrefix parquetDir [N]
"""
import sys, os, glob, collections
import pandas as pd
import common
from logJosh import Tee
import metaStartStop

##minimum fraction of a transcript's own CDS length that must be covered by
##an opposite-strand transcript's CDS for it to be considered a dubious
##antisense ORF (see findAntisenseOverlappingTxtIDs).
ANTISENSE_OVERLAP_FRACTION = 0.5

##how many nt away from the nearest rRNA gene still counts as "in/near the
##rDNA repeat array" (see findRDNProximalTxtIDs).
RDN_PROXIMITY_MARGIN = 2000

##minimum length of an unbroken matched (CIGAR M) run for its positions to
##count as real overlap (see realMatchPositions).
MIN_MATCH_RUN_LENGTH = 20


def realMatchPositions(alignedPairs, minRunLength=MIN_MATCH_RUN_LENGTH):
    """
    Return the set of reference positions covered by unbroken matched runs
    of at least minRunLength nt in alignedPairs (the parquet's aligned_pairs
    column: a list of [read_pos, ref_pos] pairs, one per CIGAR-consumed
    position, same source as absolute_indices).

    This is stricter than "any position absolute_indices lists": a real
    deletion/intron leaves read_pos None for every position it skips, but so
    does a FABRICATED alignment gap -- pysam's get_aligned_pairs represents
    both identically, so absolute_indices alone can't tell a genuine
    continuous overlap from minimap2 having bridged two unrelated loci with
    a bogus gap (confirmed on real data: a --splice_aware alignment run
    produced exactly this for a read whose true alignment, per a separately
    generated bam, was a clean 309bp block entirely within its own gene --
    the --splice_aware version instead had a fabricated 936bp gap and called
    the read's non-matching leader a match by scattering five separate
    7-16nt "matches" through it, each individually indistinguishable from a
    real hit by absolute_indices' any-position check but too short in
    aggregate to mean anything -- go over doradoAlignToParquetPipeline.py's
    module docstring for the full story).

    minRunLength=20 discards those scattered short matches (real biological
    alignment blocks are essentially never that fragmented) while keeping
    genuine matched stretches intact, including ones that individually
    survive a splice-mode alignment's small (1-3nt) indel noise, since only
    the *unbroken run* needs to clear the threshold, not the whole read.
    """
    positions = set()
    run = []
    for readPos, refPos in alignedPairs:
        if not (pd.isna(readPos) or pd.isna(refPos)):
            run.append(int(refPos))
        else:
            if len(run) >= minRunLength:
                positions.update(run)
            run = []
    if len(run) >= minRunLength:
        positions.update(run)
    return positions


def findRDNProximalTxtIDs(gtfFile, marginNt=RDN_PROXIMITY_MARGIN):
    """
    Return the set of protein_coding transcript_ids whose CDS lies within
    marginNt of any rRNA-biotype gene on the same chromosome -- i.e. in or
    immediately next to the rDNA repeat array (chrXII in yeast, ~100-200
    tandem copies genome-wide but collapsed to ~2 in the reference).

    That locus is notoriously hard to align to uniquely, and genes merely
    adjacent to it get swept up in the same artifacts even though their own
    systematic name has no "RDN" in it -- e.g. YLR162W-A/RRT15 sits ~900nt
    past the last annotated rRNA gene on chrXII, but was still responsible
    for 71% of all "LUTI candidate" reads in one real run of this script,
    which is an alignment artifact, not biology. The existing 'RDN' not in
    transcript_id convention used elsewhere in this codebase only catches
    genes literally named RDN*, so it misses cases like this one.
    """
    rrnaRanges = {}##chrom -> [(start,end)]
    cdsRanges  = {}##chrom -> {txtID:[start,end]}
    with open(gtfFile, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            chrom, feature_type, attributes = fields[0], fields[2], fields[8]
            start, end = int(fields[3]), int(fields[4])

            if feature_type == 'gene' and 'gene_biotype "rRNA"' in attributes:
                rrnaRanges.setdefault(chrom, []).append((start, end))
            elif feature_type == 'CDS':
                transcript_id = None
                biotype = None
                for attr in attributes.split(';'):
                    attr = attr.strip()
                    if attr.startswith('transcript_id'):
                        transcript_id = attr.split('"')[1]
                    elif attr.startswith('transcript_biotype'):
                        biotype = attr.split('"')[1]
                if transcript_id is None or biotype != 'protein_coding':
                    continue
                byChrom = cdsRanges.setdefault(chrom, {})
                if transcript_id not in byChrom:
                    byChrom[transcript_id] = [start, end]
                else:
                    byChrom[transcript_id][0] = min(byChrom[transcript_id][0], start)
                    byChrom[transcript_id][1] = max(byChrom[transcript_id][1], end)

    excluded = set()
    for chrom, ranges in rrnaRanges.items():
        if chrom not in cdsRanges:
            continue
        loBound = min(s for s, e in ranges) - marginNt
        hiBound = max(e for s, e in ranges) + marginNt
        for txtID, (start, end) in cdsRanges[chrom].items():
            if start <= hiBound and end >= loBound:
                excluded.add(txtID)
    return excluded


def buildGeneNameLookup(gtfFile):
    """
    Parse gtfFile's transcript lines and return a {transcript_id:gene_name}
    dict, so the output can show readable gene names (e.g. CDC19) alongside
    the systematic transcript_ids (e.g. YAL038W_mRNA). Falls back to the
    transcript_id itself when a transcript has no gene_name attribute (e.g.
    unnamed/dubious ORFs like YAL037C-A).
    """
    lookup = {}
    with open(gtfFile, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != 'transcript':
                continue

            transcript_id = None
            gene_name = None
            for attr in fields[8].split(';'):
                attr = attr.strip()
                if attr.startswith('transcript_id'):
                    transcript_id = attr.split('"')[1]
                elif attr.startswith('gene_name'):
                    gene_name = attr.split('"')[1]

            if transcript_id:
                lookup[transcript_id] = gene_name if gene_name else transcript_id
    return lookup


def findAntisenseOverlappingTxtIDs(gtfFile, minOverlapFrac=ANTISENSE_OVERLAP_FRACTION):
    """
    Parse gtfFile's CDS lines (protein_coding only, same filter as
    metaStartStop.parseGTF) and return the set of transcript_ids whose CDS
    span overlaps >=minOverlapFrac of its own length with a CDS on the
    OPPOSITE strand of the same chromosome.

    Yeast annotations are full of short "dubious" antisense ORFs (e.g.
    YAL037C-B, which sits almost entirely on top of CDC19/YAL038W on the
    other strand) that are really just ORF-finder artifacts, not
    independently transcribed genes. Of an overlapping opposite-strand pair,
    only the SHORTER transcript is flagged as dubious and returned here --
    the longer one (presumably the real gene, e.g. CDC19) is left alone
    even though a big fraction of its own length may be covered too. A read
    whose "own" gene resolves to a flagged transcript_id -- or whose
    "other" hit is one -- isn't a real LUTI signal, it's just an antisense
    read of the real gene underneath (or noise from the dubious annotation
    itself).
    """
    spans = {}##chrom -> strand -> txtID -> [start,end] (running min/max)
    with open(gtfFile, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != 'CDS':
                continue
            chrom, strand, attributes = fields[0], fields[6], fields[8]
            start, end = int(fields[3]), int(fields[4])

            transcript_id = None
            biotype = None
            for attr in attributes.split(';'):
                attr = attr.strip()
                if attr.startswith('transcript_id'):
                    transcript_id = attr.split('"')[1]
                elif attr.startswith('transcript_biotype'):
                    biotype = attr.split('"')[1]
            if transcript_id is None or biotype != 'protein_coding':
                continue

            byStrand = spans.setdefault(chrom, {}).setdefault(strand, {})
            if transcript_id not in byStrand:
                byStrand[transcript_id] = [start, end]
            else:
                byStrand[transcript_id][0] = min(byStrand[transcript_id][0], start)
                byStrand[transcript_id][1] = max(byStrand[transcript_id][1], end)

    excluded = set()
    for chrom, byStrand in spans.items():
        plusTxts  = byStrand.get('+', {})
        minusTxts = byStrand.get('-', {})
        for txtID, (start, end) in plusTxts.items():
            length = end - start + 1
            for otherTxtID, (oStart, oEnd) in minusTxts.items():
                overlap = min(end, oEnd) - max(start, oStart) + 1
                if overlap <= 0:
                    continue
                otherLength = oEnd - oStart + 1
                ##only the SHORTER of the pair is treated as the dubious
                ##antisense ORF -- a long, real gene shouldn't be excluded
                ##just because a short antisense ORF happens to cover a big
                ##chunk of it (e.g. CDC19 shouldn't be dropped just because
                ##YAL037C-B covers ~64% of CDC19's length).
                shorterTxtID, shorterLength = (
                    (txtID, length) if length <= otherLength else (otherTxtID, otherLength))
                if overlap / shorterLength >= minOverlapFrac:
                    excluded.add(shorterTxtID)
    return excluded


def findSpannedTranscripts(row, gtfDict, bothStrands=False):
    """
    Return (ownTxtID, otherTxtIDs): ownTxtID is the transcript_id whose
    CDS+/-100nt region (per gtfDict) contains the read's 3'-most
    (poly-A-proximal) touched position -- used as the read's "own" gene
    instead of the parquet's transcript_id column (see module docstring).
    otherTxtIDs is the set of *other* transcript_ids whose CDS+/-100nt
    region overlaps a position genomically upstream (5') of every position
    where the read touches ownTxtID's own CDS region. Returns None if the
    read never touches any CDS region on its own gene_strand in gtfDict --
    nothing to anchor "own"/"upstream" to.

    LUTIs are 5' extended, so an "other" transcript hit on the downstream
    (3') side (e.g. stop-codon readthrough into the next gene) doesn't
    count. "Upstream" is defined directly from genomic coordinates rather
    than from absolute_indices' iteration order, since which end of that
    list is 5' depends on whether it was flipped for minus-strand reads --
    lower absIdx is upstream on the + strand, higher absIdx is upstream on
    the - strand.

    Only positions from realMatchPositions (unbroken matched runs of
    >=MIN_MATCH_RUN_LENGTH nt) count as "touched" -- not every position
    absolute_indices lists -- so a fabricated alignment gap (or the
    scattered short coincidental matches minimap2 sometimes chains through
    one) can't manufacture a spurious upstream hit. See that function's
    docstring for why absolute_indices alone isn't trustworthy here.

    bothStrands: if True, also check the opposite strand's gtfDict entries
    for each position when looking for "other" hits (e.g. to catch LUTI
    overlap with a convergent/divergent neighbor on the other strand).
    ownTxtID is always determined from the read's own gene_strand only.
    Default False matches the same-strand-only convention used elsewhere
    in metaStartStop.py.
    """
    chrom  = row['chrom']
    strand = row['gene_strand']

    if strand not in gtfDict or chrom not in gtfDict[strand]:
        return None

    matchedAbsIdxs = realMatchPositions(row['aligned_pairs'])
    if not matchedAbsIdxs:
        return None

    ##own-gene hits: always same-strand as the read, regardless of bothStrands.
    ownHitsByAbsIdx = {}##absIdx -> set of txtIDs found there, own strand only
    for absIdx in matchedAbsIdxs:
        if chrom in gtfDict[strand] and absIdx in gtfDict[strand][chrom]:
            for txtID, relStart, relStop in gtfDict[strand][chrom][absIdx]:
                ownHitsByAbsIdx.setdefault(absIdx, set()).add(txtID)

    if not ownHitsByAbsIdx:
        return None

    ##anchor "own" on the 3'-most (poly-A-proximal) touched position: lower
    ##absIdx is 3' on the - strand, higher absIdx is 3' on the + strand.
    threePrimeAbsIdx = max(ownHitsByAbsIdx) if strand == '+' else min(ownHitsByAbsIdx)
    ownTxtID = sorted(ownHitsByAbsIdx[threePrimeAbsIdx])[0]

    strandsToCheck = ['+', '-'] if bothStrands else [strand]

    ownAbsIdxs = []
    otherHits  = {}##absIdx -> set of other txtIDs found there
    for absIdx in matchedAbsIdxs:
        for checkStrand in strandsToCheck:
            if chrom not in gtfDict[checkStrand] or absIdx not in gtfDict[checkStrand][chrom]:
                continue
            for txtID, relStart, relStop in gtfDict[checkStrand][chrom][absIdx]:
                if txtID == ownTxtID:
                    ownAbsIdxs.append(absIdx)
                else:
                    otherHits.setdefault(absIdx, set()).add(txtID)

    if strand == '+':
        boundary = min(ownAbsIdxs)
        upstreamAbsIdxs = [absIdx for absIdx in otherHits if absIdx < boundary]
    else:
        boundary = max(ownAbsIdxs)
        upstreamAbsIdxs = [absIdx for absIdx in otherHits if absIdx > boundary]

    otherTxtIDs = set()
    for absIdx in upstreamAbsIdxs:
        otherTxtIDs.update(otherHits[absIdx])

    return ownTxtID, otherTxtIDs


def main(args):
    if len(args) < 3 or len(args) > 4:
        print("Usage: python3 findLUTICandidateReads.py inFile.gtf outPrefix parquetDir [N]")
        sys.exit(1)

    gtfFile, outPrefix, parquetDir = args[0], args[1], args[2]

    ##N is the minimum number of total reads a transcript_id needs (across
    ##all parquet files) to be considered well-covered enough to trust a
    ##LUTI call for it. Defaults to 10.
    n = 10
    if len(args) == 4:
        try:
            n = int(args[3])
        except ValueError:
            print("Error: N must be an integer")
            sys.exit(1)

    parquetFiles = sorted(glob.glob(os.path.join(parquetDir, "*.parquet")))
    if not parquetFiles:
        print("Error: no .parquet files found in %s" % parquetDir)
        sys.exit(1)

    gtfDict = metaStartStop.parseGTF(gtfFile)
    ##gtfDict is of the format:
    ##{strand:{chr:{absIndx:[(txtID,relStart,relStop)]}}}

    antisenseOverlappingTxtIDs = findAntisenseOverlappingTxtIDs(gtfFile)
    print('Excluding %d transcript_ids that mostly overlap a gene on the opposite strand '
          '(likely dubious antisense ORFs).' % len(antisenseOverlappingTxtIDs))

    rdnProximalTxtIDs = findRDNProximalTxtIDs(gtfFile)
    print('Excluding %d transcript_ids in/near the rDNA repeat array '
          '(unreliable alignment region).' % len(rdnProximalTxtIDs))

    excludedTxtIDs = antisenseOverlappingTxtIDs | rdnProximalTxtIDs

    geneNameLookup = buildGeneNameLookup(gtfFile)

    ##candidatesByTranscript is of format:
    ##{transcript_id:[{read_id,chrom,gene_strand,gene_name,parquet_transcript_id,
    ##                  other_transcript_ids,other_gene_names,source_parquet}]}
    ##transcript_id here is the 3'-anchored own gene from findSpannedTranscripts,
    ##not necessarily the parquet's own transcript_id column (kept per-read
    ##as parquet_transcript_id for comparison).
    candidatesByTranscript = collections.defaultdict(list)
    ##readCountByTranscript tracks every read seen per (3'-anchored) own
    ##transcript_id, to apply the coverage filter below.
    readCountByTranscript = collections.defaultdict(set)

    for parquetFile in parquetFiles:
        print('\nAnalyzing %s...' % parquetFile)
        ##read one parquet file at a time, rather than loading the whole dir.
        df = pd.read_parquet(parquetFile)
        print('Parquet file has %d rows' % len(df))

        for index, row in df.iterrows():
            result = findSpannedTranscripts(row, gtfDict)
            ownTxtID, overlapOtherTxtIDs = result if result is not None else (None, set())

            ##Fold in shadowingBamToGTFWithParquet_LUTI.py's rescue signal, if
            ##present (row.get degrades to None on an older parquet that
            ##predates these columns, so this is a no-op there). That script
            ##only sets luti_rescue_hit when the clipped 5' leader's rescue
            ##realignment already cleared the same CDS+/-100nt upstream check
            ##findSpannedTranscripts does -- see this module's docstring.
            rescueOtherTxtIDs = set()
            if row.get('luti_rescue_hit'):
                rescueOtherTxtIDs = {t for t in
                                      str(row.get('luti_rescue_other_transcript_ids') or '').split(',')
                                      if t}
                if ownTxtID is None and rescueOtherTxtIDs:
                    ##findSpannedTranscripts couldn't anchor an own gene from
                    ##the aligned footprint at all (e.g. the clipped leader
                    ##left too little else to touch a CDS) -- fall back to
                    ##the parquet's whole-read best-overlap transcript_id.
                    ownTxtID = row['transcript_id']

            if not ownTxtID or ownTxtID in excludedTxtIDs:
                continue##no usable own gene, or it's a dubious antisense ORF/rDNA-proximal

            overlapOtherTxtIDs = overlapOtherTxtIDs - excludedTxtIDs - {ownTxtID}
            rescueOtherTxtIDs  = rescueOtherTxtIDs  - excludedTxtIDs - {ownTxtID}
            otherTxtIDs = overlapOtherTxtIDs | rescueOtherTxtIDs

            readCountByTranscript[ownTxtID].add(row['read_id'])

            if otherTxtIDs:
                sourceSignal = '+'.join(filter(None, [
                    'overlap' if overlapOtherTxtIDs else None,
                    'rescue'  if rescueOtherTxtIDs  else None,
                ]))
                candidatesByTranscript[ownTxtID].append({
                    'read_id':               row['read_id'],
                    'chrom':                 row['chrom'],
                    'gene_strand':           row['gene_strand'],
                    'gene_name':             geneNameLookup.get(ownTxtID, ownTxtID),
                    'parquet_transcript_id': row['transcript_id'],
                    'other_transcript_ids':  ','.join(sorted(otherTxtIDs)),
                    'other_gene_names':      ','.join(sorted(
                        geneNameLookup.get(t, t) for t in otherTxtIDs)),
                    'source_signal':         sourceSignal,
                    'source_parquet':        parquetFile,
                })

    ##apply the coverage filter: only keep transcripts with >=N total reads.
    wellCovered = {transcript_id: reads for transcript_id, reads in candidatesByTranscript.items()
                   if len(readCountByTranscript[transcript_id]) >= n}
    droppedForCoverage = len(candidatesByTranscript) - len(wellCovered)
    if droppedForCoverage:
        print('\nDropped %d transcript(s) with fewer than %d total reads.' % (droppedForCoverage, n))

    totalReads = sum(len(reads) for reads in wellCovered.values())
    print('\nFound %d candidate reads across %d transcripts spanning >1 transcript (>=%d reads each).'
          % (totalReads, len(wellCovered), n))

    pklFile = outPrefix + '.lutiCandidates.pkl'
    common.rePickle(wellCovered, pklFile)
    print('Wrote candidate reads (by transcript_id) to %s' % pklFile)

    ##flatten wellCovered back out to one row per read, for manual inspection.
    flatRows = []
    for transcript_id, reads in wellCovered.items():
        for read in reads:
            flatRows.append({'transcript_id': transcript_id, **read})

    txtFile = outPrefix + '.lutiCandidates.txt'
    pd.DataFrame(flatRows).to_csv(txtFile, sep='\t', index=False)
    print('Wrote %d candidate reads to %s' % (len(flatRows), txtFile))


if __name__ == '__main__':
    Tee()
    main(sys.argv[1:])
