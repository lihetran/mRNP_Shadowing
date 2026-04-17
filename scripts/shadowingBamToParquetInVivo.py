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

Usage:
  python shadowingBamToParquet.py <bam_file> <reference_fasta> <output_dir>
'''

import argparse
import pysam
from pathlib import Path
from Bio import SeqIO
import pandas as pd


def get_absolute_positions(read):
    '''Use pysam get_aligned_pairs() for ref positions — avoids custom CIGAR
    parsing errors. Returns list of ref positions (int or None for insertions).'''
    return [p[1] for p in read.get_aligned_pairs()]


def read_generator(bam_path, ref_sequence, chrom):
    '''Yield one read dict at a time. Includes chrom as a field.'''
    ref_seq = ref_sequence.upper()
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(chrom):
            if read.is_unmapped:
                continue
            barcode  = read.get_tag('cI') if read.has_tag('cI') else None
            bar_seq  = read.get_tag('cS') if read.has_tag('cS') else None
            read_seq = read.query_sequence.upper()

            aligned_pairs    = read.get_aligned_pairs()
            absolute_indices = get_absolute_positions(read)

            edits       = []
            read_string = []
            ref_string  = []

            for read_pos, ref_pos in aligned_pairs:
                if ref_pos is not None and read_pos is not None:
                    if ref_seq[ref_pos] == 'A' and read_seq[read_pos] == 'G':
                        edits.append(1)
                    else:
                        edits.append(0)
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

            # global_edit_freq
            # Edit freq over all non-indel A positions in the read.
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
                'read_id':                   read.query_name,
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
    parser.add_argument("--chunk_size",  type=int, default=50000,
                        help="Rows per output parquet chunk (default: 50000)")

    args = parser.parse_args()

    bam_file   = Path(args.bam_file)
    ref_fasta  = Path(args.ref_fasta)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_dict = SeqIO.to_dict(SeqIO.parse(ref_fasta, "fasta"))

    # Single parquet base path for all chromosomes combined
    parquet_path = output_dir / f"{bam_file.stem}.parquet"

    rows        = []
    chunk_index = 0
    total       = 0

    for chrom, ref_seq in ref_dict.items():
        print(f"Processing chromosome {chrom}...")
        chrom_total = 0

        for record in read_generator(bam_file, ref_seq.seq, chrom):
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