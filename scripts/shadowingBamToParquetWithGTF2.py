'''
October 20, 2025
LT

This script processes a BAM file to extract per-read info and saves data
efficiently using a generator and chunked parquet files.

Updated: added global_edit_freq and n_a_positions columns to support fast
         PCA filtering and CDF plotting.

         global_edit_freq  — edit freq over all A positions in the read
         n_a_positions     — number of non-indel A positions

Updated: parquet chunks are no longer split by chromosome. Instead, 'chrom'
         is included as a field in each read record, and all chromosomes are
         written into a single set of chunked parquet files.

Updated: strand is now determined from a GTF annotation file rather than
         read.is_reverse, which is unreliable for direct cDNA sequencing.
         For each read, the transcript with the most overlap is used to
         determine strand and assign transcript_id and gene_name fields.
         If no transcript overlaps, read.is_reverse is used as fallback
         and transcript_id/gene_name are set to None.

Updated: added read_start and read_end (genomic coordinates from
         read.reference_start and read.reference_end) to support fast
         vectorised overlap filtering in downstream analysis scripts.
         These are always in plus-strand genomic coordinates regardless
         of gene strand, captured before any strand-flipping.

Usage:
  python shadowingBamToParquet.py <bam_file> <reference_fasta> <output_dir> --gtf <gtf_file>
'''

import argparse
import pysam
from pathlib import Path
from Bio import SeqIO
from intervaltree import IntervalTree
import pandas as pd


def parse_gtf_attributes(attr_string):
    '''Parse GTF attribute string into a dict.'''
    attrs = {}
    for field in attr_string.strip().split(';'):
        field = field.strip()
        if not field:
            continue
        parts = field.split(' ', 1)
        if len(parts) == 2:
            attrs[parts[0]] = parts[1].strip('"')
    return attrs


def build_strand_index(gtf_path):
    '''Parse GTF and build a per-chromosome IntervalTree mapping
    genomic intervals to (strand, transcript_id, gene_name).
    Only 'transcript' features are used.'''
    index = {}  # chrom -> IntervalTree
    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            if fields[2] != 'transcript':
                continue
            chrom  = fields[0]
            start  = int(fields[3]) - 1  # GTF is 1-based, convert to 0-based
            end    = int(fields[4])
            strand = fields[6]
            attrs  = parse_gtf_attributes(fields[8])
            data   = {
                'strand':        strand,
                'transcript_id': attrs.get('transcript_id', None),
                'gene_name':     attrs.get('gene_name', None),
                'gene_biotype':  attrs.get('gene_biotype', None),
            }
            if chrom not in index:
                index[chrom] = IntervalTree()
            index[chrom].addi(start, end, data)
    return index


def get_transcript_info(strand_index, chrom, read_start, read_end, is_reverse):
    '''Look up the transcript with the most overlap with the read.
    Returns (strand, transcript_id, gene_name, gene_biotype).
    Falls back to is_reverse strand and None IDs if no transcript overlaps.'''
    tree = strand_index.get(chrom)
    if tree is None:
        return ('-' if is_reverse else '+'), None, None, None

    overlaps = tree.overlap(read_start, read_end)
    if not overlaps:
        return ('-' if is_reverse else '+'), None, None, None

    # Pick transcript with most overlap with the read
    best_data    = None
    best_overlap = 0
    for interval in overlaps:
        overlap = min(read_end, interval.end) - max(read_start, interval.begin)
        if overlap > best_overlap:
            best_overlap = overlap
            best_data    = interval.data

    return (best_data['strand'], best_data['transcript_id'],
            best_data['gene_name'], best_data['gene_biotype'])


def reverse_complement_str(seq):
    '''Return the reverse complement of a DNA string.'''
    comp = str.maketrans('ACGTacgt ', 'TGCAtgca ')
    return seq.translate(comp)[::-1]


def get_absolute_positions(read):
    '''Use pysam get_aligned_pairs() for ref positions — avoids custom CIGAR
    parsing errors. Returns list of ref positions (int or None for insertions).'''
    return [p[1] for p in read.get_aligned_pairs()]


def read_generator(bam_path, ref_sequence, chrom, strand_index, coding_only=False):
    '''Yield one read dict at a time. Includes chrom as a field.'''
    ref_seq = str(ref_sequence).upper()
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(chrom):
            if read.is_unmapped:
                continue
            barcode  = read.get_tag('cI') if read.has_tag('cI') else None
            bar_seq  = read.get_tag('cS') if read.has_tag('cS') else None
            read_seq = read.query_sequence.upper()

            # Capture genomic coordinates before any strand-flipping.
            # Always in plus-strand space; used for fast overlap filtering
            # downstream via vectorised read_start < win_end and
            # read_end > win_start comparisons.
            read_start = read.reference_start
            read_end   = read.reference_end

            aligned_pairs    = read.get_aligned_pairs()
            absolute_indices = get_absolute_positions(read)

            # Determine strand, transcript_id, gene_name, and gene_biotype from GTF
            gene_strand, transcript_id, gene_name, gene_biotype = get_transcript_info(
                strand_index, chrom,
                read_start, read_end,
                read.is_reverse
            )

            # Optionally skip non-protein-coding reads
            if coding_only and gene_biotype != 'protein_coding':
                continue

            is_reverse = (gene_strand == '-')

            edits       = []
            read_string = []
            ref_string  = []

            for read_pos, ref_pos in aligned_pairs:
                if ref_pos is not None and read_pos is not None:
                    if not is_reverse:
                        is_edit = ref_seq[ref_pos] == 'A' and read_seq[read_pos] == 'G'
                    else:
                        is_edit = ref_seq[ref_pos] == 'T' and read_seq[read_pos] == 'C'
                    edits.append(1 if is_edit else 0)
                    read_string.append(read_seq[read_pos])
                    ref_string.append(ref_seq[ref_pos])
                elif ref_pos is None:
                    edits.append(2)
                    read_string.append(read_seq[read_pos])
                    ref_string.append(' ')
                elif read_pos is None:
                    edits.append(2)
                    read_string.append(' ')
                    ref_string.append(ref_seq[ref_pos])

            edit_string = ''.join(str(i) for i in edits)
            read_string = ''.join(read_string)
            ref_string  = ''.join(ref_string)

            # If minus-strand, flip everything to sense orientation
            if is_reverse:
                read_string      = reverse_complement_str(read_string)
                ref_string       = reverse_complement_str(ref_string)
                edit_string      = edit_string[::-1]
                absolute_indices = absolute_indices[::-1]

            # global_edit_freq
            # Edit freq over all non-indel A positions in the read (always sense orientation).
            a_idx_all = [i for i, c in enumerate(ref_string)
                         if c == 'A' and i < len(edit_string)
                         and edit_string[i] != '2']
            n_a_all   = len(a_idx_all)
            global_edit_freq = (
                sum(1 for i in a_idx_all if edit_string[i] == '1') / n_a_all
                if n_a_all > 0 else 0.0
            )

            yield {
                'chrom':                     chrom,
                'gene_strand':               gene_strand,
                'is_reverse':                is_reverse,
                'transcript_id':             transcript_id,
                'gene_name':                 gene_name,
                'gene_biotype':              gene_biotype,
                'read_id':                   read.query_name,
                'read_start':                read_start,
                'read_end':                  read_end,
                'edit_string':               edit_string,
                'barcode':                   barcode,
                'bar_seq':                   bar_seq,
                'read_sequence':             read_seq,
                'read_sequence_aligned':     read_string,
                'ref_sequence_aligned':      ref_string,
                'aligned_pairs':             aligned_pairs,
                'absolute_indices':          absolute_indices,
                'global_edit_freq':          global_edit_freq,
                'n_a_positions':             n_a_all,
            }


def optimize_dataframe(df):
    '''Downcast numeric dtypes to save memory and disk space.'''
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df


def write_chunk(df, base_path, chunk_index):
    chunk_file = base_path.with_name(f"{base_path.stem}_chunk{chunk_index}.parquet")
    df.to_parquet(chunk_file, compression='zstd', index=False)
    print(f"Saved chunk {chunk_index} with {len(df)} rows to {chunk_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert BAM file to parquet chunks for PCA pipeline."
    )
    parser.add_argument("bam_file",      type=str, help="Input BAM file")
    parser.add_argument("ref_fasta",     type=str, help="Reference FASTA file")
    parser.add_argument("output_dir",    type=str, help="Output directory for parquet chunks")
    parser.add_argument("--gtf",         type=str, required=True,
                        help="GTF annotation file for strand determination")
    parser.add_argument("--coding_only", action="store_true",
                        help="Only write reads assigned to protein-coding genes")
    parser.add_argument("--chunk_size",  type=int, default=50000,
                        help="Rows per output parquet chunk (default: 50000)")

    args = parser.parse_args()

    bam_file   = Path(args.bam_file)
    ref_fasta  = Path(args.ref_fasta)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building strand index from {args.gtf}...")
    strand_index = build_strand_index(args.gtf)
    print(f"  Loaded annotations for {len(strand_index)} chromosomes")

    ref_dict = SeqIO.to_dict(SeqIO.parse(ref_fasta, "fasta"))

    # Single parquet base path for all chromosomes combined
    parquet_path = output_dir / f"{bam_file.stem}.parquet"

    rows        = []
    chunk_index = 0
    total       = 0

    for chrom, ref_seq in ref_dict.items():
        print(f"Processing chromosome {chrom}...")
        chrom_total = 0

        for record in read_generator(bam_file, ref_seq.seq, chrom, strand_index,
                                     coding_only=args.coding_only):
            rows.append(record)
            total += 1
            chrom_total += 1

            if len(rows) >= args.chunk_size:
                df = pd.DataFrame(rows)
                df = optimize_dataframe(df)
                write_chunk(df, parquet_path, chunk_index)
                rows.clear()
                chunk_index += 1

        print(f"  Finished {chrom}: {chrom_total} reads")

    # Write any remaining rows
    if rows:
        df = pd.DataFrame(rows)
        df = optimize_dataframe(df)
        write_chunk(df, parquet_path, chunk_index)

    print(f"All done! {total} total reads written across {chunk_index + 1} chunk(s)")


if __name__ == '__main__':
    main()