'''
LT, August 2026

Companion to shadowingBamToParquetWithGTF2.py, focused specifically on LUTI
(Long Undecoded Transcript Isoform) discovery. Does not modify that script
or its output -- this writes its own, separate parquet.

Why this exists: findLUTICandidateReads.py's findSpannedTranscripts flags a
read as a LUTI candidate when its *aligned* footprint (absolute_indices)
passes through a neighboring transcript's CDS (+/-100nt) upstream of its
own. But a soft-clipped base never enters get_aligned_pairs() at all --
pysam skips CIGAR S/H ops entirely, not even a None placeholder the way an
insertion gets one -- so a true LUTI read whose 5' extension got
soft-clipped by minimap2 instead of chained through is invisible to that
check: nothing in the existing parquet even records that clipping
happened, let alone how long it was.

This script adds that signal in two stages (cheap prefilter, then
confirmatory rescue only where warranted):

  1. clip5_len / clip5_seq -- a strand-aware "biological 5' soft-clip
     length/sequence", computed for every read from
     read.query_alignment_start/end vs len(read.query_sequence), then
     flipped to sense orientation using gene_strand (not read.is_reverse)
     -- the same orientation source of truth shadowingBamToParquetWithGTF2.py
     already uses for absolute_indices/edit_string, since direct cDNA
     sequencing makes read.is_reverse unreliable on its own.

  2. luti_rescue_hit / luti_rescue_other_transcript_ids -- only attempted
     for reads whose clip5_len clears --min_clip_len (default 50nt;
     shorter fragments are unlikely to seed reliably in minimap2's
     minimizer index regardless). For those, clip5_seq is realigned
     against the whole genome with mappy (preset=map-ont) and the hit is
     checked against the same CDS+/-100nt position map
     (metaStartStop.parseGTF) findSpannedTranscripts uses, requiring the
     entire rescued fragment to land upstream of the read's own aligned
     body (read_start/read_end) -- i.e. the same signature
     findSpannedTranscripts looks for, recovered here for reads where
     minimap2 clipped it away instead of chaining through it.

First-pass filter, not an exhaustive one: very short or highly divergent
leaders may still fail to seed even with rescue; hard-clipped bases aren't
recoverable (sequence isn't stored in the BAM record); and only same-strand
neighbors are checked, matching findSpannedTranscripts' own
same-strand-only default.

Usage:
  python3 shadowingBamToGTFWithParquet_LUTI.py bam_file ref_fasta output_dir --gtf gtf_file
'''

import argparse
from collections import defaultdict
from pathlib import Path

import pysam
import pandas as pd
import mappy as mp

import metaStartStop
from shadowingBamToParquetWithGTF2 import (
    build_strand_index, get_transcript_info, build_barcode_lookup,
    reverse_complement_str, optimize_dataframe, write_chunk,
)


def get_clip5_info(read, is_reverse):
    '''Return (clip5_len, clip5_seq): length and sense-oriented (5'->3')
    sequence of the biological 5' soft-clip. Orientation is taken from
    gene_strand-derived is_reverse (not read.is_reverse) to match how
    absolute_indices/read_string are flipped elsewhere in this pipeline --
    see this module's docstring.'''
    seq            = read.query_sequence
    seq_len        = len(seq)
    start_clip_len = read.query_alignment_start
    end_clip_len   = seq_len - read.query_alignment_end

    if not is_reverse:
        clip_len = start_clip_len
        clip_seq = seq[:clip_len].upper() if clip_len else ''
    else:
        clip_len = end_clip_len
        clip_seq = reverse_complement_str(seq[seq_len - clip_len:].upper()) if clip_len else ''

    return clip_len, clip_seq


def rescue_upstream_hit(clip_seq, gene_strand, chrom, transcript_id,
                         read_start, read_end, aligner, gtf_dict, min_mapq):
    '''Realign a clipped 5' leader (already sense-oriented per
    get_clip5_info) against the genome and check whether it lands, in its
    entirety, upstream of the read's own aligned body (read_start/
    read_end) and inside a *different* transcript's CDS+/-100nt region per
    gtf_dict (metaStartStop.parseGTF format: {strand:{chr:{absIdx:
    [(txtID,relStart,relStop)]}}}) -- the same signature
    findLUTICandidateReads.findSpannedTranscripts requires from an
    aligned-through read.

    Expected hit.strand is +1 for a '+' gene_strand and -1 for a '-'
    gene_strand, since clip_seq was already reverse-complemented to sense
    for minus-strand genes; a hit in the other orientation is treated as a
    spurious/opposite-strand match and discarded, same caution this
    project already applies elsewhere to unconstrained minimap2 hits (see
    doradoAligner_AtoG.py's -G3000 fix).

    Returns (other_transcript_ids: set, hit_start, hit_end); empty set and
    None, None if no qualifying hit.'''
    expected_strand = 1 if gene_strand == '+' else -1

    best_hit = None
    for hit in aligner.map(clip_seq):
        if not hit.is_primary or hit.mapq < min_mapq:
            continue
        if hit.ctg != chrom or hit.strand != expected_strand:
            continue
        if best_hit is None or hit.mlen > best_hit.mlen:
            best_hit = hit

    if best_hit is None:
        return set(), None, None

    if gene_strand == '+':
        if best_hit.r_en > read_start:  ##not entirely upstream of the read's own body
            return set(), None, None
    else:
        if best_hit.r_st < read_end:
            return set(), None, None

    chrom_dict = gtf_dict.get(gene_strand, {}).get(chrom, {})
    other_ids  = set()
    for abs_idx in range(best_hit.r_st, best_hit.r_en):
        for txt_id, _, _ in chrom_dict.get(abs_idx, []):
            if not transcript_id.startswith(txt_id):  ##.startswith to deal with the _mRNA suffix
                other_ids.add(txt_id)

    if not other_ids:
        return set(), None, None
    return other_ids, best_hit.r_st, best_hit.r_en


def read_generator(bam_path, chrom, strand_index, gtf_dict, aligner,
                    min_clip_len=50, min_rescue_mapq=1, coding_only=False,
                    barcode_lookup=None):
    '''Yield one read dict at a time for chrom. barcode_lookup: optional
    read_id -> barcode_arrangement dict, see build_barcode_lookup.'''
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(chrom):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue

            barcode = read.get_tag('cI') if read.has_tag('cI') else None
            bar_seq = read.get_tag('cS') if read.has_tag('cS') else None

            # Always plus-strand genomic coordinates, captured before any
            # strand-flipping -- same convention as shadowingBamToParquetWithGTF2.py.
            read_start = read.reference_start
            read_end   = read.reference_end

            absolute_indices = [p[1] for p in read.get_aligned_pairs()]

            gene_strand, transcript_id, gene_name, gene_biotype = get_transcript_info(
                strand_index, chrom, read_start, read_end, read.is_reverse
            )

            # Optionally skip non-protein-coding reads
            if coding_only and gene_biotype != 'protein_coding':
                continue

            is_reverse = (gene_strand == '-')
            if is_reverse:
                absolute_indices = absolute_indices[::-1]

            clip5_len, clip5_seq = get_clip5_info(read, is_reverse)

            luti_rescue_attempted = False
            luti_rescue_hit        = False
            other_ids_str          = ''
            hit_start = hit_end     = None

            if (clip5_len >= min_clip_len and transcript_id and 'RDN' not in transcript_id
                    and gene_strand in gtf_dict and chrom in gtf_dict[gene_strand]):
                luti_rescue_attempted = True
                other_ids, hit_start, hit_end = rescue_upstream_hit(
                    clip5_seq, gene_strand, chrom, transcript_id,
                    read_start, read_end, aligner, gtf_dict, min_rescue_mapq
                )
                if other_ids:
                    luti_rescue_hit = True
                    other_ids_str   = ','.join(sorted(other_ids))

            yield {
                'chrom':                            chrom,
                'gene_strand':                       gene_strand,
                'is_reverse':                         is_reverse,
                'transcript_id':                     transcript_id,
                'gene_name':                         gene_name,
                'gene_biotype':                      gene_biotype,
                'read_id':                            read.query_name,
                'read_start':                         read_start,
                'read_end':                           read_end,
                'barcode':                            barcode,
                'bar_seq':                            bar_seq,
                'barcode_arrangement':                (barcode_lookup.get(read.query_name, 'unclassified')
                                                        if barcode_lookup is not None else None),
                'absolute_indices':                   absolute_indices,
                'clip5_len':                          clip5_len,
                'clip5_seq':                          clip5_seq,
                'luti_rescue_attempted':              luti_rescue_attempted,
                'luti_rescue_hit':                    luti_rescue_hit,
                'luti_rescue_other_transcript_ids':   other_ids_str,
                'luti_rescue_hit_start':               hit_start,
                'luti_rescue_hit_end':                 hit_end,
            }


def generate_parquet(bam_file, ref_fasta, output_dir, gtf_file,
                      min_clip_len=50, min_rescue_mapq=1, coding_only=False,
                      chunk_size=50000, barcode_lookup=None):
    '''Convert an aligned bam into chunked parquet files under output_dir,
    annotated with the LUTI-rescue columns described in this module's
    docstring. Returns the total number of reads written.

    barcode_lookup: optional read_id -> barcode_arrangement dict (see
    build_barcode_lookup). When given, output is split into one
    output_dir/<barcode>/ subdirectory per barcode (unmatched reads go to
    output_dir/unclassified/) instead of a single flat output_dir.'''
    bam_file   = Path(bam_file)
    ref_fasta  = Path(ref_fasta)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building strand index from {gtf_file}...")
    strand_index = build_strand_index(gtf_file)
    print(f"  Loaded annotations for {len(strand_index)} chromosomes")

    print(f"Building CDS position map from {gtf_file} (for upstream-overlap checks)...")
    gtf_dict = metaStartStop.parseGTF(gtf_file)

    print(f"Indexing {ref_fasta} for rescue realignment (mappy, preset=map-ont)...")
    aligner = mp.Aligner(str(ref_fasta), preset='map-ont')
    if not aligner:
        raise RuntimeError(f"Failed to build mappy index from {ref_fasta}")

    with pysam.AlignmentFile(str(bam_file), 'rb') as bam:
        chroms = list(bam.references)

    # One buffer + chunk counter per barcode group; key is None (single flat
    # output_dir) when barcode_lookup isn't given, else the barcode string.
    rows          = defaultdict(list)
    chunk_index   = defaultdict(int)
    total         = 0
    total_rescued = 0

    def flush(key):
        base_dir = output_dir if key is None else output_dir / key
        base_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = base_dir / f"{bam_file.stem}.parquet"
        df = pd.DataFrame(rows[key])
        df = optimize_dataframe(df)
        write_chunk(df, parquet_path, chunk_index[key])
        rows[key].clear()
        chunk_index[key] += 1

    for chrom in chroms:
        print(f"Processing chromosome {chrom}...")
        chrom_total = 0

        for record in read_generator(bam_file, chrom, strand_index, gtf_dict, aligner,
                                      min_clip_len=min_clip_len, min_rescue_mapq=min_rescue_mapq,
                                      coding_only=coding_only, barcode_lookup=barcode_lookup):
            key = record['barcode_arrangement'] if barcode_lookup is not None else None
            rows[key].append(record)
            total += 1
            chrom_total += 1
            if record['luti_rescue_hit']:
                total_rescued += 1

            if len(rows[key]) >= chunk_size:
                flush(key)

        print(f"  Finished {chrom}: {chrom_total} reads")

    # Write any remaining buffered rows for every barcode group
    for key in list(rows.keys()):
        if rows[key]:
            flush(key)

    n_groups = len(chunk_index)
    print(f"All done! {total} total reads written across {n_groups} barcode group(s), "
          f"{sum(chunk_index.values())} chunk(s) total. "
          f"{total_rescued} read(s) rescued as LUTI candidates via 5'-clip realignment.")
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Convert BAM to parquet chunks with LUTI-rescue columns: reads with a "
                    "long 5' soft clip get that clipped leader realigned to the genome to check "
                    "whether it lands in a neighboring transcript's CDS upstream of the read's "
                    "own aligned body -- catches LUTI candidates findLUTICandidateReads.py "
                    "misses because a soft-clipped leader never enters absolute_indices."
    )
    parser.add_argument("bam_file",   type=str, help="Input aligned BAM file")
    parser.add_argument("ref_fasta",  type=str,
                        help="Reference genome FASTA (same one used for the original alignment)")
    parser.add_argument("output_dir", type=str, help="Output directory for parquet chunks")
    parser.add_argument("--gtf", type=str, required=True,
                        help="GTF annotation file (for strand/transcript assignment and the CDS "
                             "position map)")
    parser.add_argument("--min_clip_len", type=int, default=50,
                        help="Minimum biological 5' soft-clip length (nt) before attempting "
                             "rescue realignment (default: 50 -- shorter fragments are unlikely "
                             "to seed reliably in minimap2's minimizer index anyway)")
    parser.add_argument("--min_rescue_mapq", type=int, default=1,
                        help="Minimum mapq for a rescue realignment hit to be trusted (default: 1)")
    parser.add_argument("--coding_only", action="store_true",
                        help="Only write reads assigned to protein-coding genes")
    parser.add_argument("--chunk_size", type=int, default=50000,
                        help="Rows per output parquet chunk (default: 50000)")
    parser.add_argument("--barcode_summary", type=str, default=None,
                        help="Path to a MinKNOW sequencing_summary.txt with a barcode_arrangement "
                             "column (native barcoding kit runs). When given, output is split into "
                             "output_dir/<barcode>/ subdirectories (unmatched reads go to "
                             "output_dir/unclassified/).")

    args = parser.parse_args()

    barcode_lookup = (build_barcode_lookup(args.barcode_summary)
                       if args.barcode_summary else None)

    generate_parquet(args.bam_file, args.ref_fasta, args.output_dir, args.gtf,
                      min_clip_len=args.min_clip_len, min_rescue_mapq=args.min_rescue_mapq,
                      coding_only=args.coding_only, chunk_size=args.chunk_size,
                      barcode_lookup=barcode_lookup)


if __name__ == '__main__':
    main()
