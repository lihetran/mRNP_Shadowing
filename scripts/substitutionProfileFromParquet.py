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
        # If the substitution is not in the profile, we can choose to ignore it or raise an error.
        # For now, we'll ignore it.
        pass

def generate_substitution_profile(df):
    """
    Generate a substitution profile from the DataFrame of reads.
    The DataFrame is expected to have columns 'ref_sequence_aligned' and 'read_sequence_aligned'.
    """
    profile = init_substitution_profile()
    for _, row in df.iterrows():
        for r,q in zip(row['ref_sequence_aligned'].upper(), row['read_sequence_aligned'].upper()):
            if r != q:
                update_substitution_profile(profile, r, q)
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


def main(args):
    if len(args) < 2:
        print("Usage: python3 substitutionProfileFromParquet.py "
              "output_prefix parquetDir1 [parquetDir2 ...]", file=sys.stderr)
        sys.exit(1)

    output_prefix = args[0]
    parquet_libs  = args[1:]

    # CMYK colours — one per library
    colours = [
        color.cmyk(0, 0, 0, 1),      # black
        color.cmyk(1, 0.5, 0, 0),    # blue
        color.cmyk(0, 1, 1, 0),      # red
        color.cmyk(0.6, 0, 0.9, 0),  # green
    ]

    profiles = []
    rows     = []
    for idx, parquet_dir in enumerate(parquet_libs):
        label = Path(parquet_dir).name
        print(f"\nLoading {label} ({parquet_dir})...", file=sys.stderr)
        df = load_all_parquet_chunks(parquet_dir)
        if df.empty:
            print(f"  WARNING: no reads loaded for {label}; skipping.",
                  file=sys.stderr)
            continue

        profile = generate_substitution_profile(df)
        col     = colours[idx % len(colours)]
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

