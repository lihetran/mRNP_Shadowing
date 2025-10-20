'''
October 20, 2025
LT

This script processes a BAM file to extract per-read info and saves data efficiently using a generator,
chunked parquet files, and shelve for large fields to reduce memory usage.

Usage: python shadowingBamToPickle.py <bam_file> <reference_fasta> <output_dir>
'''

import pysam
from pathlib import Path
from Bio import SeqIO
import shelve
import pandas as pd
import sys

def get_absolute_positions(read):
    '''Calculate absolute positions of the read in reference based on CIGAR and alignment start'''
    start = read.reference_start
    aligned_positions = []
    ref_pos = start
    cigar_tuples = read.cigartuples  # list of (operation, length)

    # CIGAR operation codes from pysam docs:
    # 0:M, 1:I, 2:D, 3:N, 4:S, 5:H, 6:P, 7:=, 8:X
    for op, length in cigar_tuples:
        if op == 0 or op == 7 or op == 8:  # M, =, X -> aligned to reference
            aligned_positions.extend(range(ref_pos, ref_pos + length))
            ref_pos += length
        elif op == 1:  # insertion to reference
            aligned_positions.extend([None] * length)
        elif op == 2 or op == 3:  # deletion or skipped region in reference
            ref_pos += length
        elif op in [4,5,6]:  # clipping or padding - no reference advance
            pass
        else:
            pass
    return aligned_positions

def read_generator(bam_path, ref_sequence, chrom):
    '''Yield one read dict at a time'''
    ref_seq = ref_sequence.upper()
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(chrom):
            if read.is_unmapped:
                continue
            barcode = read.get_tag('cI') if read.has_tag('cI') else None
            bar_seq = read.get_tag('cS') if read.has_tag('cS') else None
            read_seq = read.query_sequence.upper()
            aligned_pairs = read.get_aligned_pairs()
            edits = []
            read_string = []
            ref_string = []
            absolute_indices = get_absolute_positions(read)

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
            ref_string = ''.join(ref_string)

            yield {
                'read_id': read.query_name,
                'edit_string': edit_string,
                'barcode': barcode,
                'bar_seq': bar_seq,
                'read_sequence': read_seq,
                'read_sequence_aligned': read_string,
                'ref_sequence_aligned': ref_string,
                'aligned_pairs': aligned_pairs,
                'absolute_indices': absolute_indices
            }

def optimize_dataframe(df):
    '''Downcast and convert dtypes to save memory'''
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    # convert object columns with low unique values to category if beneficial
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')
    return df

def write_chunk(df, base_path, chunk_index):
    chunk_file = base_path.with_name(f"{base_path.stem}_chunk{chunk_index}.parquet")
    df.to_parquet(chunk_file, compression='zstd', index=False)
    print(f"Saved chunk {chunk_index} with {len(df)} rows to {chunk_file}")

def main(args):
    bam_file = Path(args[0])
    ref_fasta = Path(args[1])
    output_dir = Path(args[2])

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference sequences
    ref_dict = SeqIO.to_dict(SeqIO.parse(ref_fasta, "fasta"))

    chunk_size = 50000  # tune this based on your RAM and speed requirements

    for chrom, ref_seq in ref_dict.items():
        print(f"Processing chromosome {chrom}...")
        parquet_path = output_dir / f"{bam_file.stem}_{chrom}.parquet"
        shelve_path = output_dir / f"{bam_file.stem}_{chrom}_largefields.shelve"

        rows = []
        chunk_index = 0
        total = 0

        with shelve.open(str(shelve_path)) as db:
            for record in read_generator(bam_file, ref_seq.seq, chrom):
                # Store large fields in shelve
                db[record['read_id']] = {
                    'aligned_pairs': record['aligned_pairs'],
                    'absolute_indices': record['absolute_indices']
                }
                # Remove bulky fields before putting into dataframe
                del record['aligned_pairs']
                del record['absolute_indices']

                rows.append(record)
                total += 1

                if len(rows) >= chunk_size:
                    df = pd.DataFrame(rows)
                    df = optimize_dataframe(df)
                    write_chunk(df, parquet_path, chunk_index)
                    rows.clear()
                    chunk_index += 1

            # Write remaining rows
            if rows:
                df = pd.DataFrame(rows)
                df = optimize_dataframe(df)
                write_chunk(df, parquet_path, chunk_index)

        print(f"Finished processing {total} reads for chromosome {chrom}")

    print("All done!")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python shadowingBamToPickle.py <bam_file> <reference_fasta> <output_dir>")
        sys.exit(1)
    main(sys.argv[1:])
