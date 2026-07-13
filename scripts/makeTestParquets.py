'''
July 13, 2026 LT

I need to make test datasets to run my bayesianShadowClassifier.py script on. This script will generate a set of test parquet files for a single gene.

inputs:
- gene name
- parquet directory
- outputPrefix

'''

import sys
import pandas as pd
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

def filter_by_gene_name(df: pd.DataFrame, gene_name: str) -> pd.DataFrame:
    '''
    Get all reads from a parquet file that matches the gene name.
    '''
    filtered_df = df[df['gene_name'] == gene_name]
    return filtered_df

def main(args):
    parquet_dir = Path(args[0])
    gene_name = str(args[1])
    output_prefix = str(args[2])

    df = load_all_parquet_chunks(parquet_dir)
    filtered_df = filter_by_gene_name(df, gene_name)
    # write to parquet
    filtered_df.to_parquet(f"{output_prefix}_{gene_name}.parquet", compression='zstd', index=False)

if __name__ == "__main__":
    main(sys.argv[1:])
