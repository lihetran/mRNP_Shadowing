'''
July 3, 2026

This script will generate a substitution profiles from polysome shadowing libraries stored in parquet format.
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

def init_substitution_profile():
    """
    Initialize a substitution profile dictionary with all possible substitutions.
    The keys are tuples of (ref_base, alt_base), and the values are counts initialized to 0.
    """
    bases = ['A', 'C', 'G', 'T']
    profile = {(ref, alt): 0 for ref in bases for alt in bases if ref != alt}
    return profile

def update_substitution_profile(profile, ref_base, alt_base):
    """
    Update the substitution profile with a new observation of a substitution.
    """
    if (ref_base, alt_base) in profile:
        profile[(ref_base, alt_base)] += 1
    else:
        print(f"Warning: Invalid substitution {ref_base} -> {alt_base}", file=sys.stderr)

def generate_substitution_profile(df):
    """
    Generate a substitution profile from the DataFrame of reads.
    The DataFrame is expected to have columns 'ref_sequence_aligned' and 'read_sequence_aligned'.
    """
    profile = init_substitution_profile()
    for _, row in df.iterrows():
        for r,q in zip(row['ref_sequence_aligned'].upper(), row['read_sequence_aligned'].upper()):
            update_substitution_profile(profile, r, q)
    return profile

