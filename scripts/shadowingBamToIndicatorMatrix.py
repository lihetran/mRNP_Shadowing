'''
January 22, 2026 LT
shadowingBamToPickle.py only stores information for A positions. This works great for MCA analyses but MCA via the prince library is not scalable for very large datasets.
I want to pivot to PCA-based analyses and smooth over all positions to convert to continuous values. Therefore, I need to store information for all positions, not just A positions.

This script is a modified version of shadowingBamToPickle.py that stores information for all positions in the read, not just A positions.
This will allow for more flexible downstream analyses, including PCA-based methods. I'm also going to implement chunked parquet files and shelve for large fields to reduce memory usage.

input: bam file, this file will usually be a merged bam file that is the result of mapBased_barcodeSplitting_Hamming.py
    reference fasta file
    output_dir - directory to save the pickle file
output: parquet files with the same name as the bam file, but with a .parquet extension

January 26, 2026 LT
Need to add ability to filter reads by edit frequency
'''

import pysam
from pathlib import Path
from Bio import SeqIO
import shelve
import pandas as pd
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.signal import convolve


def get_absolute_positions(read):
    '''need to calculate absolute position of the read in the reference sequence, will do this by getting the start of
    the alignment (4th field in the sam file) and the CIGAR string (5th field in the sam file) to get the absolute positions.'''
    # get the start position of the read
    start = read.query_alignment_start
    # get the CIGAR string
    cigar_string = read.cigarstring
    # get the aligned positions
    aligned_positions = []
    ref_pos = start
    for i in range(len(cigar_string)):
        if cigar_string[i].isdigit():
            continue
        else:
            length = int(cigar_string[:i])
            if cigar_string[i] == 'M':  # match or mismatch
                aligned_positions.extend(range(ref_pos, ref_pos + length))
                ref_pos += length
            elif cigar_string[i] == 'I':  # insertion
                aligned_positions.extend([None] * length)  # None for insertion
            elif cigar_string[i] == 'D':  # deletion
                ref_pos += length  # skip these positions in the reference
            cigar_string = cigar_string[i + 1:]
            break
    # continue processing the remaining CIGAR string
    while cigar_string:
        for i in range(len(cigar_string)):
            if cigar_string[i].isdigit():
                continue
            else:
                length = int(cigar_string[:i])
                if cigar_string[i] == 'M':  # match or mismatch
                    aligned_positions.extend(range(ref_pos, ref_pos + length))
                    ref_pos += length
                elif cigar_string[i] == 'I':  # insertion
                    aligned_positions.extend([None] * length)  # None for insertion
                elif cigar_string[i] == 'D':  # deletion
                    ref_pos += length  # skip these positions in the reference
                cigar_string = cigar_string[i + 1:]
                break
    # print(f"Read {read.query_name} aligned positions: {aligned_positions}")

    return aligned_positions


def read_generator(bam_path, ref_sequence, chrom, window_start, window_end, min_edit_freq=0.0):
    '''Yield one read dict at a time'''
    ref_seq = ref_sequence.upper()
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(chrom, window_start, window_end):
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
            total_A = 0
            total_edits = 0
            # First pass to calculate edit frequency
            for read_pos, ref_pos in aligned_pairs:
                if ref_pos is not None and read_pos is not None:
                    if ref_seq[ref_pos] == 'A' and read_seq[read_pos] == 'G':
                        edits.append(1)
                        total_edits += 1
                        total_A += 1
                    elif ref_seq[ref_pos] == 'A':
                        total_A += 1
                    else:
                        edits.append(0)
                    read_string.append(read_seq[read_pos])
                    ref_string.append(ref_seq[ref_pos])
                elif ref_pos is None: # insertions are 0 for smoothing purposes
                    edits.append(0)
                    read_string.append(read_seq[read_pos])
                    ref_string.append(' ')
                elif read_pos is None: # deletions are 0 for smoothing purposes
                    edits.append(0)
                    read_string.append(' ')
                    ref_string.append(ref_seq[ref_pos])

            edit_string = ''.join(str(i) for i in edits)
            read_string = ''.join(read_string)
            ref_string = ''.join(ref_string)

            # Calculate edit frequency for A positions
            edit_freq = (total_edits / total_A) if total_A > 0 else 0.0
            if edit_freq > float(min_edit_freq):
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

    return df


def write_chunk(df, base_path, chunk_index):
    chunk_file = base_path.with_name(f"{base_path.stem}_chunk{chunk_index}.parquet")
    df.to_parquet(chunk_file, compression='zstd', index=False)
    print(f"Saved chunk {chunk_index} with {len(df)} rows to {chunk_file}")


def smooth_indicator_matrix_from_reads(
    edit_strings,
    absolute_indices_list,
    window_start,
    window_end,
    window
):

    L = window_end - window_start
    kernel = np.ones(window, dtype=np.float32) / window

    X = np.zeros((len(edit_strings), L), dtype=np.float32)

    for r, (edits, positions) in enumerate(
        zip(edit_strings, absolute_indices_list)
    ):
        edit_vals = np.frombuffer(edits.encode(), dtype=np.uint8) - ord('0')

        for val, pos in zip(edit_vals, positions):
            if pos is None:
                continue
            if window_start <= pos < window_end:
                X[r, pos - window_start] = val

        X[r] = convolve(X[r], kernel, mode="same")

    return X


def main(args):
    bam_file = Path(args[0])
    ref_fasta = Path(args[1])
    min_edit_freq = args[2]
    output_dir = Path(args[3])

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference sequences
    ref_dict = SeqIO.to_dict(SeqIO.parse(ref_fasta, "fasta"))

    chunk_size = 50000

    # ---- smoothing parameters ----
    # window_start = 220
    # window_end   = 320
    window_start = 100
    window_end = 600
    smooth_window = 10

    for chrom, ref_seq in ref_dict.items():
        print(f"Processing chromosome {chrom}...")

        parquet_path = output_dir / f"{bam_file.stem}_{chrom}.parquet"

        edit_strings = []
        absolute_indices = []

        chunk_index = 0
        total = 0

        for record in read_generator(bam_file, ref_seq.seq, chrom, window_start, window_end, min_edit_freq):

            edit_strings.append(record["edit_string"])
            absolute_indices.append(record["absolute_indices"])
            total += 1

            if len(edit_strings) >= chunk_size:

                X = smooth_indicator_matrix_from_reads(
                    edit_strings,
                    absolute_indices,
                    window_start=window_start,
                    window_end=window_end,
                    window=smooth_window
                )

                df = pd.DataFrame(
                    X,
                    columns=[
                        f"pos_{i}"
                        for i in range(window_start, window_end)
                    ]
                )

                write_chunk(df, parquet_path, chunk_index)

                edit_strings.clear()
                absolute_indices.clear()
                chunk_index += 1
        # print('len matrix:', len(X))
        # ---- write remaining reads ----
        if edit_strings:

            X = smooth_indicator_matrix_from_reads(
                edit_strings,
                absolute_indices,
                window_start=window_start,
                window_end=window_end,
                window=smooth_window
            )

            df = pd.DataFrame(
                X,
                columns=[
                    f"pos_{i}"
                    for i in range(window_start, window_end)
                ]
            )

            write_chunk(df, parquet_path, chunk_index)

        print(f"Finished processing {total} reads for chromosome {chrom}")

    print("All done!")



if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python shadowingBamToIndicatorMatrix.py <bam_file> <reference_fasta> <output_dir>")
        sys.exit(1)
    main(sys.argv[1:])
