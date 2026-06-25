"""
Joshua Arribere, June 17, 2026
(updated June 2026: single-library directory input; top-N expressed transcripts)

Script to parse out the most abundant transcripts from a directory of parquet
    files belonging to a single library.

Input: inFile.gtf - gtf-formatted annotation file
    parquetDir - a directory containing the parquet files for ONE library.
        All *.parquet files directly inside this directory are treated as the
        same library.
    optional: N - the number of top-expressed transcripts to extract
        (default: 10).

Output: a single parquet file containing the reads that span the entire CDS
    (as determined by the GTF) for the N most abundant transcripts. A
    'library' column is added, set to the directory name. Abundance is measured
    by the number of full-CDS-spanning reads per transcript.

run as python3 extractHighlyExpressedTranscripts.py inFile.gtf parquetDir
    outPrefix [N]
"""
import sys, os, glob, common, metaStartStop
from logJosh import Tee
import pandas as pd


def readSpansCDS(row, gtfDict):
    """Return True if this read spans the entire CDS for its transcript."""
    chrom         = row['chrom']
    strand        = row['gene_strand']
    transcript_id = row['transcript_id']

    if not (strand in gtfDict and chrom in gtfDict[strand]
            and transcript_id and 'RDN' not in transcript_id):
        return False

    spansStart = False
    spansStop  = False
    for index2, (refNt, readNt, edit, absIdx) in enumerate(zip(
            row['ref_sequence_aligned'],
            row['read_sequence_aligned'],
            row['edit_string'],
            row['absolute_indices'])):
        ##skip if absIdx is nan
        if pd.isna(absIdx):
            continue
        ##
        absIdx = int(absIdx)
        edit   = int(edit)##still restrict to 0 or 1, otherwise 2 is soft-clipped or unmapped
        ##
        if edit in [0, 1] and absIdx in gtfDict[strand][chrom] \
                and len(gtfDict[strand][chrom][absIdx]) == 1:##restriction for uniquely assignable
            for txtID, relStart, relStop in gtfDict[strand][chrom][absIdx]:
                if transcript_id.startswith(txtID):##.startswith to deal with the _mRNA suffix
                    if relStart <= -25:
                        spansStart = True
                    elif relStart >= 25 and relStop <= -25:
                        pass
                    elif relStop >= 25:
                        spansStop = True
    ##
    return spansStart and spansStop


def main(args):
    # Parse input arguments
    if len(args) < 3 or len(args) > 4:
        print("Usage: python3 extractHighlyExpressedTranscripts.py "
              "inFile.gtf parquetDir outPrefix [N]")
        sys.exit(1)

    gtfFile    = args[0]
    parquetDir = args[1]
    outPrefix  = args[2]

    # N defaults to 10 (top ten highest-expressed transcripts).
    n = 10
    if len(args) == 4:
        try:
            n = int(args[3])
        except ValueError:
            print("Error: N must be an integer")
            sys.exit(1)

    # Library name is taken from the directory name.
    libraryName = os.path.basename(os.path.normpath(parquetDir))

    # Collect parquet files from the directory.
    parquetFiles = sorted(glob.glob(os.path.join(parquetDir, "*.parquet")))
    if not parquetFiles:
        print("Error: no .parquet files found in %s" % parquetDir)
        sys.exit(1)

    # Parse GTF file
    gtfDict = metaStartStop.parseGTF(gtfFile)
    ##gtfDict is of the format:
    ##{strand:{chr:{absIndx:(txtName,relStart,relStop)]}}}

    print('\nWorking on library %s (%d parquet files)...'
          % (libraryName, len(parquetFiles)))

    ##collect full-CDS-spanning reads in a list and build the DataFrame once at
    ##the end (avoids the O(n^2) concat-inside-the-loop pattern).
    passedRows = []
    for parquetFile in parquetFiles:
        ##read parquetFile into dataframe
        df = pd.read_parquet(parquetFile)
        ##
        #df=df[:1000]#useful for subsetting the data during troubleshooting.
        ##
        print('Parquet File %s has %d rows' % (parquetFile, len(df)))
        ##
        for index, row in df.iterrows():
            if readSpansCDS(row, gtfDict):
                ##this read spans the entire CDS. Keep it.
                row['library'] = libraryName
                passedRows.append(row)
    ##
    passedDF = pd.DataFrame(passedRows).reset_index(drop=True) \
        if passedRows else pd.DataFrame()

    if passedDF.empty:
        print("No reads spanned the full CDS; nothing to write.")
        return

    ##subset to the n most abundant transcripts (by full-CDS-spanning read count).
    topTranscripts = passedDF['transcript_id'].value_counts().index[:n]
    print('\nTop %d transcripts (by full-CDS-spanning reads):' % len(topTranscripts))
    counts = passedDF['transcript_id'].value_counts()
    for t in topTranscripts:
        print('  %s\t%d' % (t, counts[t]))
    passedDF = passedDF[passedDF['transcript_id'].isin(topTranscripts)]

    ##Now we will write out the passedDF dataframe to a new parquet file.
    outFile = f"{outPrefix}_highlyExpressedTranscripts.parquet"
    passedDF.to_parquet(outFile, index=False)
    ##
    print('\nWrote %d reads across %d transcripts to %s'
          % (len(passedDF), passedDF['transcript_id'].nunique(), outFile))
    print(passedDF)



if __name__ == '__main__':
    Tee()
    main(sys.argv[1:])