'''
May 18, 2026
LT

Compute per-read editing efficiency from parquet files produced by
shadowingBamToParquet.py, excluding reads assigned to RDN loci.
Outputs a CDF plot of per-read global_edit_freq.

Usage:
  python editingEfficiencyCDF.py <parquet_dir> <output_dir>
'''

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_parquets(parquet_dir):
    '''Load only necessary columns from all parquet chunks in a directory.'''
    files = sorted(Path(parquet_dir).glob('*.parquet'))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    print(f"Loading {len(files)} parquet file(s)...")
    cols = ['gene_name', 'global_edit_freq']
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)


def is_rdn(gene_name):
    '''Return True if gene_name corresponds to an RDN/rRNA locus.'''
    if gene_name is None:
        return False
    return 'RRNA' in str(gene_name).upper()


def plot_cdf(edit_freqs, output_path):
    '''Plot and save a CDF of per-read editing efficiencies.'''
    sorted_freqs = edit_freqs.sort_values().reset_index(drop=True)
    cdf = (sorted_freqs.index + 1) / len(sorted_freqs)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sorted_freqs, cdf, linewidth=1.5)
    ax.set_xlabel('Per-read editing efficiency (global_edit_freq)')
    ax.set_ylabel('Cumulative fraction of reads')
    ax.set_title(f'Editing efficiency CDF (n={len(sorted_freqs):,} reads, RDN excluded)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved CDF plot to {output_path}")


def main(args):
    if len(args) < 2:
        print("Usage: python editingEfficiencyCDF.py <parquet_dir> <output_dir>")
        sys.exit(1)

    parquet_dir = Path(args[0])
    output_dir  = Path(args[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_parquets(parquet_dir)
    print(f"  Loaded {len(df):,} total reads")

    # Filter out RDN reads
    rdn_mask    = df['gene_name'].apply(is_rdn)
    df_filtered = df[~rdn_mask]
    print(f"  Excluded {rdn_mask.sum():,} RDN reads")
    print(f"  Remaining reads for CDF: {len(df_filtered):,}")

    plot_cdf(df_filtered['global_edit_freq'], output_dir / 'editing_efficiency_cdf.png')


if __name__ == '__main__':
    main(sys.argv[1:])