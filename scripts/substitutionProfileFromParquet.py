'''
July 3, 2026

This script will generate a substitution profiles from polysome shadowing libraries stored in parquet format.
'''

import sys, common, collections, random, metaStartStop
import pandas as pd
import numpy as np
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

def generate_substitution_profile(df):
    """
    Generate a substitution profile from the DataFrame of reads.
    The DataFrame is expected to have columns 'ref_sequence_aligned' and
    'read_sequence_aligned' (equal-length, gapped alignment strings).

    Vectorized over the whole DataFrame at once (concatenate every read's
    aligned pair into one byte array, compare, then tally mismatches with
    np.unique) instead of nested per-character Python loops -- ~10x faster
    on real-sized libraries.
    """
    profile = init_substitution_profile()
    if df.empty:
        return profile

    ref_col  = df['ref_sequence_aligned'].str.upper()
    read_col = df['read_sequence_aligned'].str.upper()

    ref_len  = ref_col.str.len()
    read_len = read_col.str.len()
    if not (ref_len == read_len).all():
        # Defensive: truncate any mismatched-length pairs to their shared
        # length so concatenation below can't misalign subsequent rows.
        min_len  = np.minimum(ref_len.values, read_len.values)
        ref_col  = pd.Series([s[:n] for s, n in zip(ref_col, min_len)])
        read_col = pd.Series([s[:n] for s, n in zip(read_col, min_len)])

    ref_bytes  = np.frombuffer(''.join(ref_col).encode(),  dtype='S1')
    read_bytes = np.frombuffer(''.join(read_col).encode(), dtype='S1')

    mismatch = ref_bytes != read_bytes
    if not mismatch.any():
        return profile

    pairs = np.char.add(ref_bytes[mismatch], read_bytes[mismatch])
    uniq, counts = np.unique(pairs, return_counts=True)
    for pair_bytes, count in zip(uniq, counts):
        key = (pair_bytes[:1].decode(), pair_bytes[1:].decode())
        if key in profile:
            profile[key] = int(count)
    return profile

_TEX_SPECIAL = {
    '\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
    '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
}

def tex_escape(s):
    """Escape characters (e.g. '_') that PyX's TeX text engine treats as
    special, so library names/labels render literally instead of erroring."""
    return ''.join(_TEX_SPECIAL.get(ch, ch) for ch in str(s))


def _bar_chart_pyx(libs, values_by_lib, ylabel, title, output_path):
    """
    Shared grouped-barplot renderer.

    libs: list of (profile, color, label) tuples (profile unused here).
    values_by_lib: list (parallel to libs) of {sub_type: value} dicts.
    """
    bases     = ['A', 'C', 'G', 'T']
    sub_types = [(ref, alt) for ref in bases for alt in bases if ref != alt]

    y_max = max((max(v.values()) for v in values_by_lib), default=0.0)
    y_max = max(y_max * 1.15, 1e-9)

    n_libs, n_subs = len(libs), len(sub_types)
    panel_w, panel_h = 14, 6

    c = canvas.canvas()
    g = graph.graphxy(
        width=panel_w, height=panel_h, xpos=0, ypos=0,
        x=graph.axis.linear(min=0, max=n_subs, parter=None,
                             title="Substitution type"),
        y=graph.axis.linear(min=0, max=y_max, title=ylabel),
    )
    c.insert(g)

    group_pad = 0.12
    usable    = 1.0 - 2 * group_pad
    bar_w     = usable / n_libs

    for si, sub in enumerate(sub_types):
        for li, (values, (_, col, label)) in enumerate(zip(values_by_lib, libs)):
            val = values[sub]
            bx0 = si + group_pad + li * bar_w
            bx1 = bx0 + bar_w
            cx0, cy0 = g.pos(bx0, 0.0)
            cx1, cy1 = g.pos(bx1, val)
            if cy1 > cy0:
                c.fill(path.rect(cx0, cy0, cx1 - cx0, cy1 - cy0), [col])
                c.stroke(path.rect(cx0, cy0, cx1 - cx0, cy1 - cy0),
                         [style.linewidth.thin, color.gray(0.3)])
        cxm, cym = g.pos(si + 0.5, 0.0)
        c.text(cxm, cym - 0.35, f"{sub[0]}{{$\\to$}}{sub[1]}",
               [text.halign.center, text.size.scriptsize])

    c.text(g.xpos + g.width / 2., g.ypos + g.height + 0.5,
           tex_escape(title), [text.halign.center, text.size.Large])

    leg_x     = g.xpos + g.width + 0.4
    leg_y_top = g.ypos + g.height - 0.3
    leg_lw, leg_dy = 0.6, 0.55
    for li, (_, col, label) in enumerate(libs):
        ly = leg_y_top - li * leg_dy
        c.fill(path.rect(leg_x, ly - 0.12, leg_lw, 0.24), [col])
        c.stroke(path.rect(leg_x, ly - 0.12, leg_lw, 0.24),
                 [style.linewidth.thin, color.gray(0.3)])
        c.text(leg_x + leg_lw + 0.15, ly, tex_escape(label),
               [text.valign.middle, text.size.small])

    c.writePDFfile(output_path)
    print(f"  Saved -> {output_path}.pdf", file=sys.stderr)


def plot_substitution_profile(profiles, output_prefix):
    """
    Render two grouped barplots comparing substitution profiles across
    libraries: one of raw counts, one of fractions (each library's
    substitutions normalized to sum to 1).

    profiles: list of (profile_dict, color, label) tuples, where
    profile_dict maps (ref_base, alt_base) -> count.
    """
    bases     = ['A', 'C', 'G', 'T']
    sub_types = [(ref, alt) for ref in bases for alt in bases if ref != alt]

    libs = [(profile, col, label) for profile, col, label in profiles if profile]
    if not libs:
        print("  No substitution data to plot.", file=sys.stderr)
        return

    def fractions(profile):
        total = sum(profile.values())
        if total == 0:
            return {sub: 0.0 for sub in sub_types}
        return {sub: profile[sub] / total for sub in sub_types}

    counts_by_lib = [{sub: profile[sub] for sub in sub_types}
                     for profile, _, _ in libs]
    frac_by_lib   = [fractions(profile) for profile, _, _ in libs]

    _bar_chart_pyx(libs, counts_by_lib, "Substitution count",
                   "Substitution profile (counts)",
                   f"{output_prefix}_substitution_counts_pyx")
    _bar_chart_pyx(libs, frac_by_lib, "Fraction of substitutions",
                   "Substitution profile (frequency)",
                   f"{output_prefix}_substitution_frequency_pyx")


def parse_parquet_libs_file(path):
    """
    Parse a line-delimited file of format:
        fileNamei repi directoryi
    (same inFilesParquet.txt convention used by polysomeShadowQC.py /
    polysomeShadowHMMQC.py), and return a list of (label, directory)
    tuples with label = 'fileName-rep'.
    """
    libs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            fileName, rep, directory = parts[0], parts[1], parts[2]
            libs.append((f"{fileName}-{rep}", directory))
    return libs


def main(args):
    if len(args) != 2:
        print("Usage: python3 substitutionProfileFromParquet.py "
              "inFilesParquet.txt output_prefix", file=sys.stderr)
        print("  inFilesParquet.txt: line-delimited 'fileName rep directory'",
              file=sys.stderr)
        sys.exit(1)

    parquet_libs_file = args[0]
    output_prefix     = args[1]

    libs_to_load = parse_parquet_libs_file(parquet_libs_file)
    if not libs_to_load:
        print(f"No libraries found in {parquet_libs_file}; exiting.",
              file=sys.stderr)
        sys.exit(1)

    profiles = []
    rows     = []
    for idx, (label, parquet_dir) in enumerate(libs_to_load):
        print(f"\nLoading {label} ({parquet_dir})...", file=sys.stderr)
        df = load_all_parquet_chunks(parquet_dir)
        if df.empty:
            print(f"  WARNING: no reads loaded for {label}; skipping.",
                  file=sys.stderr)
            continue

        profile = generate_substitution_profile(df)
        col     = common.colors(idx)
        profiles.append((profile, col, label))

        total = sum(profile.values())
        print(f"  [{label}] {total:,} total substitutions.", file=sys.stderr)
        for (ref, alt), count in profile.items():
            rows.append({
                "library":  label,
                "ref_base": ref,
                "alt_base": alt,
                "count":    count,
                "fraction": (count / total) if total else 0.0,
            })

    if not rows:
        print("No substitution data generated; exiting.", file=sys.stderr)
        return

    summary_csv = f"{output_prefix}_substitution_profile.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"\n  Saved substitution profile -> {summary_csv}", file=sys.stderr)

    print("\nGenerating substitution profile plot...", file=sys.stderr)
    try:
        plot_substitution_profile(profiles, output_prefix)
    except Exception as e:
        print(f"  WARNING: pyx plotting failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    Tee()
    main(sys.argv[1:])

