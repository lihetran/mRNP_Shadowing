'''
Inspect edit sites for a specific read ID from a parquet file.
Reports all positions where edit_string == '1', along with the
reference and read bases at that position.

Usage:
  python inspect_read_edits.py <parquet_file> <read_id>
'''

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Report all edited sites for a given read ID."
    )
    parser.add_argument("parquet_file", type=str, help="Path to parquet file")
    parser.add_argument("read_id",      type=str, help="Read ID to inspect")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet_file)

    row = df[df['read_id'] == args.read_id]
    if row.empty:
        print(f"Read ID '{args.read_id}' not found in {args.parquet_file}")
        return

    row = row.iloc[0]

    edit_string  = row['edit_string']
    ref_string   = row['ref_sequence_aligned']
    read_string  = row['read_sequence_aligned']
    abs_indices  = row['absolute_indices']
    is_reverse = row['is_reverse']

    if is_reverse:
        strand = '-'
    else:
        strand = '+'


    print(f"Read ID:  {row['read_id']}")
    print(f"Chrom:    {row['chrom']}")
    print(f"strand:    {strand}")
    print(f"Total aligned positions: {len(edit_string)}")
    print()
    print(f"{'Index':<8} {'Abs Pos':<12} {'Ref Base':<10} {'Read Base':<10}")
    print("-" * 42)

    for i, (e, ref_base, read_base) in enumerate(zip(edit_string, ref_string, read_string)):
        if e == '1':
            abs_pos = abs_indices[i] 
            print(f"{i:<8} {str(abs_pos):<12} {ref_base:<10} {read_base:<10}")

    # Alignment visualization
    print()
    print("Alignment:")
    match_string = ''.join(
        '|' if r == q else '*'
        for r, q in zip(ref_string, read_string)
    )
    print(f"  REF:  {ref_string}")
    print(f"        {match_string}")
    print(f"  READ: {read_string}")


if __name__ == '__main__':
    main()