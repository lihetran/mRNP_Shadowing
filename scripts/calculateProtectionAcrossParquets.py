"""
Joshua Arribere, May 1, 2026

Script to calculate the degree of protection across UTR5/CDS/UTR3 for every read
    in a set of parquet files. Will restrict to protein-coding genes.

Input: inFile.gtf - gtf file containing annotations.
    inFileParquet.txt - line-delimited 'fileName rep parquetFile', same
        convention as polysomeShadowHMMQC.py / substitutionProfileFromParquet.py.
        Each row becomes its own library, labeled 'fileName-rep'.
    color_map.txt (optional) - manuscript color TSV 'name rep path hex_color'
        (no leading '#'). Labels are looked up as 'name-rep'/'name_rep' or
        bare 'name'; unmatched libraries just get a printed warning (kept
        for parity with the other parquet-based scripts, ready for a future
        plotting step to consume).

Output: Will create a file containing the following:
    txtID\treadID\tlibrary\tUTR5editCt\tUTR5totCt\tCDSeditCt\tCDStotCt\t
        UTR3editCt\tUTR3totCt\tUTR5weightedEdit\tUTR5weightedTot\t
        CDSweightedEdit\tCDSweightedTot\tUTR3weightedEdit\tUTR3weightedTot
    where UTR5/CDS/UTR3 editCt/totCt are the raw degree of protection and
    totCt is the respective number of editable As in that region.
    weightedEdit/weightedTot are the same, but each edited A's contribution
    is divided by that library's own baseline editing frequency for the A's
    3nt motif (see computeMotifFreqs), so an edit at a motif this library
    rarely edits counts for more and an edit at a motif it commonly edits
    counts for less -- weightedEdit/weightedTot is ~1 when a
    region/read edits at exactly its motif-composition-predicted rate,
    <1 when it's more protected than that baseline, and >1 when it's less
    protected. Same motif-bias-correction idea as
    polysomeShadowQC.py's metaStartStopAnalysisNormalizeForMotifBias.

run as python3 calculateProtectionAcrossParquets.py inFile.gtf outPrefix
    inFileParquet.txt [color_map.txt]
"""
import sys, common, metaStartStop, collections
import pandas as pd
from logJosh import Tee

COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}


def load_color_map(path: str) -> dict:
    """
    Parse a manuscript color-map TSV with columns:
        sample_name, rep, path, hex_color (no leading '#')
    Returns a dict keyed by "name_rep", "name-rep" (this script's own
    libraryID convention, see parse_parquet_libs_file), and bare "name"
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


def parse_parquet_libs_file(path: str) -> list:
    """
    Parse a line-delimited file of format:
        fileNamei repi parquetFilei
    (same inFileParquet.txt convention used by polysomeShadowHMMQC.py /
    substitutionProfileFromParquet.py), and return a list of
    (libraryID, parquetFile) tuples with libraryID = 'fileName-rep'.
    """
    libs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            fileName, rep, parquetFile = parts[0], parts[1], parts[2]
            libs.append((f"{fileName}-{rep}", parquetFile))
    return libs


def computeMotifFreqs(df):
    """
    For every A-equivalent position in df (refNt=='A' on '+' strand genes,
    refNt=='T' on '-' strand genes -- the same per-position strand check
    main() uses below) with a determinable edit call (0 or 1, not an
    alignment gap '2'), find the 3nt motif (prevNt,'A',nextNt) read 5'->3'
    along the mRNA and tally how often that motif is edited.

    ref_sequence_aligned here is in genomic (reference/'+' strand)
    orientation (see main()'s strand check), not sense-flipped, so on a
    '-' strand gene the mRNA-5' neighbor of an edited position is the
    complement of the base one index HIGHER (i+1) in ref_sequence_aligned,
    and the mRNA-3' neighbor is the complement of the base one index LOWER
    (i-1) -- reading the flanks in reverse and complementing them, same as
    reverse-complementing that whole 3nt genomic window.

    Returns {motif: freq}, freq = fraction of that motif's positions
    (pooled across every read in df) observed edited. Mirrors
    polysomeShadowQC.py's analyzeMotifs, adapted for this script's own
    strand-aware hasA convention.
    """
    counts = collections.defaultdict(lambda: [0, 0])   # motif -> [nEdited, nTotal]
    for row in df.itertuples(index=False):
        strand  = row.gene_strand
        refSeq  = row.ref_sequence_aligned
        editStr = row.edit_string
        n = len(refSeq)
        for i in range(1, n - 1):
            refNt = refSeq[i]
            if strand == '+':
                if refNt != 'A':
                    continue
                prevNt, nextNt = refSeq[i - 1], refSeq[i + 1]
            elif strand == '-':
                if refNt != 'T':
                    continue
                prevNt, nextNt = COMPLEMENT.get(refSeq[i + 1], 'N'), COMPLEMENT.get(refSeq[i - 1], 'N')
            else:
                continue
            if prevNt not in 'ACGT' or nextNt not in 'ACGT':
                continue
            edit = int(editStr[i])
            if edit not in (0, 1):
                continue
            motif = prevNt + 'A' + nextNt
            counts[motif][1] += 1
            if edit == 1:
                counts[motif][0] += 1
    return {motif: (nEd / nTot if nTot else 0.0) for motif, (nEd, nTot) in counts.items()}


def main(args):
    ##
    gtfFile=args[0]
    outPrefix=args[1]
    libsFile=args[2]
    colorMapPath=args[3] if len(args)>3 else None
    ##
    color_map=load_color_map(colorMapPath) if colorMapPath else {}
    libs=parse_parquet_libs_file(libsFile)
    if not libs:
        print('No libraries found in %s; exiting.'%libsFile, file=sys.stderr)
        sys.exit(1)
    if colorMapPath:
        unmatched=[label for label,_ in libs if label not in color_map]
        if unmatched:
            print('  WARNING: no color found in %s for librar%s %s; '
                  'falling back to the default.'
                  %(colorMapPath, 'y' if len(unmatched)==1 else 'ies', unmatched),
                  file=sys.stderr)
    ##
    gtfDict=metaStartStop.parseGTF(gtfFile)
    ##gtfDict is of the format:
    ##{strand:{chr:{absIndx:(txtName,relStart,relStop)]}}}
    ##
    ##now parse the parquet files and write output
    with open(outPrefix+'.txt','w') as f:
        f.write('txtID\treadID\tlibrary\tUTR5editCt\tUTR5totCt\tCDSeditCt\tCDStotCt\tUTR3editCt\tUTR3totCt\t'
                 'UTR5weightedEdit\tUTR5weightedTot\tCDSweightedEdit\tCDSweightedTot\tUTR3weightedEdit\tUTR3weightedTot')
        for label,parquetFile in libs:
            ##read parquetFile into dataframe
            df=pd.read_parquet(parquetFile)
            ##
            print('Parquet File %s (%s) has %d rows'%(parquetFile,label,len(df)))
            ##
            #df=df[:1000]#useful for subsetting the data during troubleshooting.
            ##
            ##this library's own baseline editing frequency per 3nt motif, used below
            ##to weight each edit by how surprising it is for its own motif (see
            ##computeMotifFreqs).
            motifFreqs=computeMotifFreqs(df)
            print('  %s: computed baseline editing frequency for %d distinct motifs.'
                  %(label,len(motifFreqs)))
            ##
            for index,row in df.iterrows():
                ##
                chrom=row['chrom']
                strand=row['gene_strand']
                transcript_id=row['transcript_id']
                read_id=row['read_id']
                ##
                if chrom in gtfDict[strand] and transcript_id and 'RDN' not in transcript_id:
                    ##
                    ##This will keep track of the edit counts.
                    ct=0
                    ##first position is the count of edited sites, second is count of all
                    ##possible editable sites whether edited or not; third/fourth are the
                    ##motif-bias-weighted equivalents (see computeMotifFreqs) -- weighted
                    ##edit sum and weighted total, which only counts positions where a
                    ##weight could actually be computed (see below).
                    temp={'UTR5':[0,0,0.0,0],
                          'CDS':[0,0,0.0,0],
                        'UTR3':[0,0,0.0,0]}
                    ##
                    refSeqFull=row['ref_sequence_aligned']
                    nPos=len(refSeqFull)
                    for index2,(refNt,readNt,edit,absIdx) in enumerate(zip(row['ref_sequence_aligned'],row['read_sequence_aligned'],row['edit_string'],row['absolute_indices'])):
                        ##
                        hasA=False
                        if strand=='+':
                            if refNt=='A':
                                hasA=True
                        elif strand=='-':
                            if refNt=='T':
                                hasA=True
                        ##
                        if hasA:
                            absIdx=int(absIdx)
                            edit=int(edit)
                            if edit in [0,1] and absIdx in gtfDict[strand][chrom] and len(gtfDict[strand][chrom][absIdx])==1:##restriction for uniquely assignable
                                ##
                                ##determine this position's 3nt motif (mRNA-sense, see
                                ##computeMotifFreqs) so we can look up its baseline editing
                                ##frequency below.
                                motif=None
                                if 0<index2<nPos-1:
                                    if strand=='+':
                                        prevNt,nextNt=refSeqFull[index2-1],refSeqFull[index2+1]
                                    else:##strand=='-'
                                        prevNt=COMPLEMENT.get(refSeqFull[index2+1],'N')
                                        nextNt=COMPLEMENT.get(refSeqFull[index2-1],'N')
                                    if prevNt in 'ACGT' and nextNt in 'ACGT':
                                        motif=prevNt+'A'+nextNt
                                ##
                                for txtID,relStart,relStop in gtfDict[strand][chrom][absIdx]:
                                    theKey=None
                                    if relStart<=-25:
                                        theKey='UTR5'
                                    elif relStart>=25 and relStop<=-25:
                                        theKey='CDS'
                                    elif relStop>=25:
                                        theKey='UTR3'
                                    ##
                                    if theKey:
                                        temp[theKey][1]+=1
                                        if edit==1:
                                            temp[theKey][0]+=1
                                        ct+=1
                                        ##
                                        ##motif-bias-weighted contribution: unedited positions
                                        ##always contribute a known weight of 0.0; an edited
                                        ##position only contributes if its motif could be
                                        ##determined AND that motif has a nonzero baseline
                                        ##frequency in this library (else 1/motifFreq is
                                        ##undefined) -- otherwise it's dropped from both the
                                        ##weighted sum and the weighted total, same as
                                        ##polysomeShadowQC.py's
                                        ##metaStartStopAnalysisNormalizeForMotifBias.
                                        if edit==0:
                                            temp[theKey][3]+=1
                                        elif motif is not None:
                                            motifFreq=motifFreqs.get(motif)
                                            if motifFreq:
                                                temp[theKey][2]+=1.0/motifFreq
                                                temp[theKey][3]+=1
                    ##
                    if ct>0:##then at least one editable site was found, anywhere
                        f.write(f"""\n{transcript_id}\t{read_id}\t{label}\t"""
                                f"""{temp['UTR5'][0]}\t{temp['UTR5'][1]}\t"""
                                f"""{temp['CDS'][0]}\t{temp['CDS'][1]}\t"""
                                f"""{temp['UTR3'][0]}\t{temp['UTR3'][1]}\t"""
                                f"""{temp['UTR5'][2]}\t{temp['UTR5'][3]}\t"""
                                f"""{temp['CDS'][2]}\t{temp['CDS'][3]}\t"""
                                f"""{temp['UTR3'][2]}\t{temp['UTR3'][3]}""")

if __name__=='__main__':
    Tee()
    main(sys.argv[1:])