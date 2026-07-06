'''
July 2, 2026 LT

Going to try and make the shadowCDF plot with inspiration from JA's interEditDistance.py script. We're able to see an enrichment of ribosome signals in that script
so that should serve as a good starting point.
'''

import sys, common, collections, random, metaStartStop
import pandas as pd
from logJosh import Tee
from pyx import *
from pathlib import Path

def load_all_parquet_chunks(parquet_dir: str) -> pd.DataFrame:
    parquet_dir = Path(parquet_dir)
    chunks = sorted(parquet_dir.glob("*.parquet"))
    if not chunks:
        return pd.DataFrame()
    dfs = [pd.read_parquet(c) for c in chunks]
    df  = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df):,} reads from {len(chunks)} chunk(s).",
          file=sys.stderr)
    return df

def classify_region(relStarts, relStops):
    """
    Classify a shadow into UTR5/CDS/UTR3 using its rel positions.
    relStart is position relative to the start codon (negative = 5'UTR),
    relStop is position relative to the stop codon (positive = 3'UTR).
    """
    if max(relStarts) <= -30:
        return 'UTR5'
    elif min(relStarts) >= 30 and max(relStops) <= -30:
        return 'CDS'
    elif max(relStops) >= 30:
        return 'UTR3'
    return None


def find_shadows_in_read(positions, shadow_size):
    """
    positions: list of (relStart, relStop, edit) for ref=A sites on one read,
               in transcript order. edit: 0 = unedited (A), 1 = edited (G).

    A shadow is a maximal run of consecutive unedited ref=A positions whose
    span (max relStart - min relStart) is at least shadow_size nt.

    Returns a list of shadows, each a list of (relStart, relStop) tuples.

    Written by Claude
    """

    shadows = []
    run = []  # (relStart, relStop) for the current unedited run
    prev_pos = None

    def _close(run):
        if not run:
            return
        relStarts = [r[0] for r in run]
        if max(relStarts) - min(relStarts) >= shadow_size:
            shadows.append(run[:])

    for (relStart, relStop, edit) in positions:
        if edit == 0:
            if run and prev_pos is not None and relStart - prev_pos > shadow_size:
                _close(run)
                run = []
            run.append((relStart, relStop))
            prev_pos = relStart
        else:  # edited site breaks the run
            _close(run)
            run = []
            prev_pos = None
    _close(run)

    return shadows


def find_inter_shadow_distances(positions, shadow_size):
    """
    Given ref=A positions for one read, find shadows and return a list of
    (distance, region) for each consecutive pair of shadows.

    distance = first unedited A of shadow_{i+1} minus last unedited A of
    shadow_i (nt in transcript space). Region is classified from shadow_i.
    """
    shadows = find_shadows_in_read(positions, shadow_size)

    results = []
    for i in range(len(shadows) - 1):
        shadow_i = shadows[i]
        shadow_j = shadows[i + 1]

        end_i = max(r[0] for r in shadow_i)  # last A of shadow i
        start_j = min(r[0] for r in shadow_j)  # first A of shadow j
        distance = start_j - end_i

        region = classify_region([r[0] for r in shadow_i],
                                 [r[1] for r in shadow_i])
        if region:
            results.append((distance, region))

    return results


def makeCDF(values):
    """
    Given a list of distances, return a list of (distance, cumulative_fraction)
    points describing the empirical CDF.

    The CDF is computed over ALL values so the fraction is correct; if x_max
    is given, points are still returned across the full range (clip when
    plotting via the axis max).
    """
    if not values:
        return []
    sv = sorted(values)
    n = len(sv)
    points = [(sv[i], (i + 1) / n) for i in range(n)]
    return points


def mkPlot(libraries, outPrefix, x_max=500):
    """
    libraries: list of (label, interShadowDistances_dict).
    Plots the CDS-region inter-shadow distance CDF for all libraries.
    """
    g = graph.graphxy(width=8, height=8,
                      key=graph.key.key(pos='br'),
                      x=graph.axis.linear(min=0, max=x_max,
                                          title='Inter-shadow Distance (nt)'),
                      y=graph.axis.linear(min=0, max=1,
                                          title='Cumulative fraction'))

    for i, (label, dists) in enumerate(libraries):
        vals = dists.get('CDS', [])
        cdf  = makeCDF(vals)
        if not cdf:
            continue
        g.plot(
            graph.data.points(cdf, x=1, y=2,
                              title='%s' % label),
            [graph.style.line([common.colors(i)])]
        )

    g.writePDFfile(outPrefix)


def prepData(dataDict):
    """
    dataDict is of format {position:ct}
    Converts to a list of (relPos, freq) tuples ordered by position, plus a
    count version.
    """
    dataList1 = sorted([(position, ct) for position, ct in dataDict.items()],
                       key=lambda x: x[0])
    total = sum(dataDict.values())
    dataList2 = sorted([(position, ct / total)
                        for position, ct in dataDict.items() if ct > 0],
                       key=lambda x: x[0])
    return dataList1, dataList2


def collect_inter_shadow_distances(df, gtfDict, shadow_size):
    """
    Walk all reads in one library's DataFrame and return
    {UTR5/CDS/UTR3: [distances]}.
    """
    interShadowDistances = {'UTR5': [], 'CDS': [], 'UTR3': []}
    readCt = 0

    for index, row in df.iterrows():
        chrom = row['chrom']
        strand = row['gene_strand']
        transcript_id = row['transcript_id']

        if chrom in gtfDict[strand] and transcript_id and \
                'RDN' not in transcript_id:
            # readCt += 1
            editString = row['edit_string']

            positions = []  # (relStart, relStop, edit) for ref=A sites
            if row['global_edit_freq'] >= 0.3: # min edit freq per read
                readCt += 1
                for (refNt, readNt, edit, absIdx) in zip(
                        row['ref_sequence_aligned'],
                        row['read_sequence_aligned'],
                        editString,
                        row['absolute_indices']):

                    edit = int(edit)

                    if edit in [0, 1] and \
                            absIdx in gtfDict[strand][chrom] and \
                            len(gtfDict[strand][chrom][absIdx]) == 1:
                        txtID, relStart, relStop = \
                            gtfDict[strand][chrom][absIdx][0]
                        positions.append((relStart, relStop, edit))

                positions.sort(key=lambda x: x[0])

                for distance, region in find_inter_shadow_distances(
                        positions, shadow_size):
                    interShadowDistances[region].append(distance)

    print('  %d reads passed filters and were analyzed.' % readCt)
    return interShadowDistances

def main(args):
    ##
    gtfFile = args[0]
    outPrefix = args[1]
    shadow_size = int(args[2])
    librarySpecs = args[3:]  # each "dir" or "dir:label"
    ##
    if not librarySpecs:
        sys.exit("ERROR: provide at least one parquet directory.")
    ##
    print('Shadow size: %d nt' % shadow_size)
    ##
    gtfDict = metaStartStop.parseGTF(gtfFile)
    ##gtfDict is of the format {strand:{chr:{absIndx:(txtName,relStart,relStop)]}}}
    ##
    libraries = []  # (label, interShadowDistances_dict)
    for spec in librarySpecs:
        ##allow "dir:label"; otherwise label = directory name
        if ':' in spec:
            parquet_dir, label = spec.rsplit(':', 1)
        else:
            parquet_dir = spec
            label = Path(spec.rstrip('/')).name
        ##
        print('\nProcessing library "%s"  (%s)...' % (label, parquet_dir))
        df = load_all_parquet_chunks(parquet_dir)
        if df.empty:
            print('  WARNING: no reads loaded; skipping.')
            continue
        dists = collect_inter_shadow_distances(df, gtfDict, shadow_size)
        libraries.append((label, dists))
    ##
    if not libraries:
        sys.exit("ERROR: no libraries produced data.")
    ##
    mkPlot(libraries, outPrefix)


if __name__ == '__main__':
    Tee()
    main(sys.argv[1:])