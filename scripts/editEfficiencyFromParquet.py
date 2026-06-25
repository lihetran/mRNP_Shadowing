'''
June 2026 LT

Computes per-read A->G editing efficiency from one or two parquet directories
output by shadowingBamToParquetWithGTF2.py. Uses the precomputed
global_edit_freq column (edit freq over all non-indel A positions in sense
orientation).

Produces:
  - Summary CSV with per-read editing efficiency for each library
  - Pyx CDF plot (one or two libraries)

Usage:
  # Single library
  python3 editingEfficiencyFromParquet.py \
      --parquet1 wt_parquet/ --label1 "WT" \
      --output output_prefix

  # Two libraries
  python3 editingEfficiencyFromParquet.py \
      --parquet1 wt_parquet/ --label1 "WT" \
      --parquet2 3at_parquet/ --label2 "3-AT" \
      --output output_prefix \
      [--min_a_positions 10] \
      [--gene_biotype protein_coding]
'''

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load parquets
# ─────────────────────────────────────────────────────────────────────────────

def load_parquet_chunks(parquet_dir: str) -> pd.DataFrame:
    parquet_dir = Path(parquet_dir)
    chunks = sorted(parquet_dir.glob("*.parquet"))
    if not chunks:
        print(f"  WARNING: no parquet files found in {parquet_dir}",
              file=sys.stderr)
        return pd.DataFrame()
    dfs = [pd.read_parquet(c) for c in chunks]
    df  = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df):,} reads from {len(chunks)} chunk(s).",
          file=sys.stderr)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Compute summary stats
# ─────────────────────────────────────────────────────────────────────────────

def summarise(df: pd.DataFrame, label: str,
               min_a_positions: int,
               gene_biotype: str = None):
    """
    Print summary statistics and return (eff_array, stats_series).
    """
    if df.empty:
        print(f"  [{label}] No reads.", file=sys.stderr)
        return np.array([]), pd.Series(dtype=float)

    # Strand summary
    if "is_reverse" in df.columns:
        n_rev = int(df["is_reverse"].sum())
        n_tot = len(df)
        print(f"  [{label}] forward: {n_tot - n_rev:,} "
              f"({100*(n_tot-n_rev)/n_tot:.1f}%)  "
              f"reverse (minus-strand gene): {n_rev:,} "
              f"({100*n_rev/n_tot:.1f}%)",
              file=sys.stderr)

    # Optional biotype filter
    if gene_biotype and "gene_biotype" in df.columns:
        before = len(df)
        df = df[df["gene_biotype"] == gene_biotype]
        print(f"  [{label}] {len(df):,} / {before:,} reads after "
              f"biotype filter ({gene_biotype}).", file=sys.stderr)

    # Min A positions filter
    if "n_a_positions" in df.columns:
        before = len(df)
        df = df[df["n_a_positions"] >= min_a_positions]
        print(f"  [{label}] {len(df):,} / {before:,} reads with >= "
              f"{min_a_positions} A positions.", file=sys.stderr)

    eff = df["global_edit_freq"].dropna()

    stats = pd.Series({
        "label":        label,
        "n_reads":      len(eff),
        "mean":         float(eff.mean()),
        "median":       float(eff.median()),
        "std":          float(eff.std()),
        "pct5":         float(eff.quantile(0.05)),
        "pct25":        float(eff.quantile(0.25)),
        "pct75":        float(eff.quantile(0.75)),
        "pct95":        float(eff.quantile(0.95)),
        "frac_zero":    float((eff == 0).mean()),
    })

    print(f"  [{label}] n={len(eff):,}  "
          f"median={stats['median']:.4f}  "
          f"mean={stats['mean']:.4f}  "
          f"zero={stats['frac_zero']*100:.1f}%",
          file=sys.stderr)

    return eff.values, stats


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pyx CDF plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_cdf_pyx(libraries: list, output_prefix: str):
    """
    CDF of per-read A->G editing efficiency.
    libraries: list of (eff_array, color, label, stats_series)
    """
    from pyx import canvas, graph, color, style, path, text as pyx_text

    panel_w = 8
    panel_h = 6
    leg_lw  = 0.8
    leg_dy  = 0.6

    c = canvas.canvas()

    g = graph.graphxy(
        width=panel_w, height=panel_h,
        xpos=0, ypos=0,
        x=graph.axis.linear(min=0, max=1,
                            title="edit freq per read"),
        y=graph.axis.linear(min=0, max=1,
                            title="Cumulative fraction"),
    )

    for eff, col, label, stats in libraries:
        if len(eff) == 0:
            continue
        sf  = np.sort(eff)
        cdf = np.arange(1, len(sf) + 1) / len(sf)
        g.plot(
            graph.data.points(list(zip(sf.tolist(), cdf.tolist())), x=1, y=2),
            [graph.style.line([col, style.linewidth.normal,
                               style.linestyle.solid])]
        )

    c.insert(g)

    c.text(g.xpos + g.width / 2., g.ypos + g.height + 0.5,
           "editing efficiency per read",
           [pyx_text.halign.center, pyx_text.size.Large])

    # Legend
    leg_x     = g.xpos + g.width + 0.5
    leg_y_top = g.ypos + g.height - 0.3

    for j, (eff, col, label, stats) in enumerate(libraries):
        ly = leg_y_top - j * leg_dy
        c.stroke(path.line(leg_x, ly, leg_x + leg_lw, ly),
                 [col, style.linewidth.normal, style.linestyle.solid])
        n      = int(stats.get("n_reads", 0))
        median = float(stats.get("median", float("nan")))
        c.text(leg_x + leg_lw + 0.15, ly,
               f"{label}  (n={n:,}, med={median:.4f})",
               [pyx_text.valign.middle, pyx_text.size.small])

    plot_path = f"{output_prefix}_editing_efficiency_cdf_pyx"
    c.writePDFfile(plot_path)
    print(f"  Saved -> {plot_path}.pdf", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Per-read editing efficiency CDF from parquet files."
    )
    p.add_argument("--parquet1",        required=True,
                   help="Parquet directory for library 1")
    p.add_argument("--label1",          default="Library1")
    p.add_argument("--parquet2",        default=None,
                   help="Parquet directory for library 2 (optional)")
    p.add_argument("--label2",          default="Library2")
    p.add_argument("--output",          default="editing_efficiency")
    p.add_argument("--min_a_positions", type=int, default=10,
                   help="Min ref=A positions per read (default: 10)")
    p.add_argument("--gene_biotype",    default=None,
                   help="Restrict to reads assigned to this biotype "
                        "(e.g. protein_coding)")
    return p.parse_args()


def main():
    args = parse_args()
    out  = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print("=== Per-read Editing Efficiency from Parquet ===", file=sys.stderr)

    # CMYK colours — one per library
    from pyx import color as pyx_color
    colours = [
        pyx_color.cmyk(0, 0, 0, 1),        # black
        pyx_color.cmyk(1, 0.5, 0, 0),      # blue
    ]

    libraries_data = []
    all_stats = []

    for idx, (parquet_dir, label, col) in enumerate([
        (args.parquet1, args.label1, colours[0]),
        (args.parquet2, args.label2, colours[1]),
    ]):
        if parquet_dir is None:
            continue
        print(f"\nLoading {label}...", file=sys.stderr)
        df  = load_parquet_chunks(parquet_dir)
        eff, stats = summarise(df, label, args.min_a_positions,
                                args.gene_biotype)
        libraries_data.append((eff, col, label, stats))
        all_stats.append(stats)

        # Save per-read values
        if len(eff) > 0:
            safe = label.replace(" ", "_").replace("/", "_")
            pd.Series(eff, name="global_edit_freq").to_csv(
                f"{out}_{safe}_per_read.csv.gz",
                index=False, compression="gzip"
            )

    # Save summary CSV
    summary_df = pd.DataFrame(all_stats)
    summary_csv = f"{out}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n  Saved summary -> {summary_csv}", file=sys.stderr)

    # Plot
    print(f"\nGenerating CDF plot...", file=sys.stderr)
    try:
        plot_cdf_pyx(libraries_data, out)
    except Exception as e:
        print(f"  WARNING: pyx plotting failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()