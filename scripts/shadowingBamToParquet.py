'''
October 20, 2025
LT

This script processes a BAM file to extract per-read info and saves data efficiently using a generator,
chunked parquet files, and shelve for large fields to reduce memory usage.

Updated: added global_edit_freq and n_a_positions columns to support fast
         incremental PCA filtering without per-row positDict parsing.

Usage: python shadowingBamToParquet.py <bam_file> <reference_fasta> <output_dir>
'''

import pysam
from pathlib import Path
from Bio import SeqIO
import pandas as pd
import sys


def get_absolute_positions(read):
    '''Use pysam's get_aligned_pairs() for ref positions — avoids custom CIGAR
    parsing errors. Returns list of ref positions (int or None for insertions).'''
    return [p[1] for p in read.get_aligned_pairs()]


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
            ref_string  = ''.join(ref_string)

            # ── Compute global edit frequency over A positions ───────────────
            # Done here so downstream PCA filtering can use a fast vectorised
            # column lookup instead of per-row positDict parsing.
            # edit_freq = (A→G edits) / (total A positions in aligned ref)
            a_indices  = [i for i, c in enumerate(ref_string) if c == 'A']
            a_edits    = sum(1 for i in a_indices
                             if i < len(edit_string) and edit_string[i] == '1')
            n_a        = len(a_indices)
            global_edit_freq = a_edits / n_a if n_a > 0 else 0.0

            yield {
                'read_id':                read.query_name,
                'edit_string':            edit_string,
                'barcode':                barcode,
                'bar_seq':                bar_seq,
                'read_sequence':          read_seq,
                'read_sequence_aligned':  read_string,
                'ref_sequence_aligned':   ref_string,
                'aligned_pairs':          aligned_pairs,
                'absolute_indices':       absolute_indices,
                'global_edit_freq':       global_edit_freq,   # ← new
                'n_a_positions':          n_a,                # ← new
            }


def optimize_dataframe(df):
    '''Downcast and convert dtypes to save memory'''
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df


def write_chunk(df, base_path, chunk_index):
    chunk_file = base_path.with_name(f"{base_path.stem}_chunk{chunk_index}.parquet")
    df.to_parquet(chunk_file, compression='zstd', index=False)
    print(f"Saved chunk {chunk_index} with {len(df)} rows to {chunk_file}")


def main(args):
    bam_file  = Path(args[0])
    ref_fasta = Path(args[1])
    output_dir = Path(args[2])

    output_dir.mkdir(parents=True, exist_ok=True)

    ref_dict = SeqIO.to_dict(SeqIO.parse(ref_fasta, "fasta"))

    chunk_size = 50000  # tune based on RAM and speed requirements

    for chrom, ref_seq in ref_dict.items():
        print(f"Processing chromosome {chrom}...")
        parquet_path = output_dir / f"{bam_file.stem}_{chrom}.parquet"

        rows = []
        chunk_index = 0
        total = 0

        for record in read_generator(bam_file, ref_seq.seq, chrom):
            rows.append(record)
            total += 1

            if len(rows) >= chunk_size:
                df = pd.DataFrame(rows)
                df = optimize_dataframe(df)
                write_chunk(df, parquet_path, chunk_index)
                rows.clear()
                chunk_index += 1

        if rows:
            df = pd.DataFrame(rows)
            df = optimize_dataframe(df)
            write_chunk(df, parquet_path, chunk_index)

        print(f"Finished processing {total} reads for chromosome {chrom}")

    print("All done!")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python shadowingBamToParquet.py <bam_file> <reference_fasta> <output_dir>")
        sys.exit(1)
    main(sys.argv[1:])