'''
June 2026 LT

Computes per-read A->G editing efficiency from one to three parquet directories
output by shadowingBamToParquetWithGTF2.py. Uses the precomputed
global_edit_freq column (edit freq over all non-indel A positions in sense
orientation).

Produces:
  - Summary CSV with per-read editing efficiency for each library
  - Pyx CDF plot (one to three libraries)
  - Per-context 3-mer editing rate CSV
  - Grouped barplot of A->G editing rate per editable 3-mer context (XAY)

Usage:
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


# Editable 3-mer contexts: middle base is always A (the editable base).
# 4 left bases x 4 right bases = 16 contexts.
BASES    = ["A", "C", "G", "T"]
CONTEXTS = [f"{x}A{y}" for x in BASES for y in BASES]


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
    numAs = df["n_a_positions"].dropna()

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

    return eff.values, stats, numAs.values


# ─────────────────────────────────────────────────────────────────────────────
# 2b. Per-context 3-mer editing computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_context_editing(df: pd.DataFrame, label: str,
                            gene_biotype: str = None) -> dict:
    """
    For each editable 3-mer context (XAY, middle base A), compute the A->G
    editing rate across all reads.

    Uses ref_sequence_aligned + edit_string, both sense-oriented by the
    parquet generator, so array-index adjacency (i-1, i, i+1) gives the
    trinucleotide in transcript order.

    edit_string encoding: '1' = A->G edit, '0' = unedited, '2' = indel/skip.
    Positions whose flanks are alignment gaps (not ACGT) are skipped so the
    context is always a real reference trinucleotide.

    Returns {context: {"n_edited", "n_total", "edit_rate"}}.
    """
    if df.empty:
        return {}
    if "ref_sequence_aligned" not in df.columns:
        print(f"  [{label}] WARNING: no ref_sequence_aligned column; "
              f"context analysis skipped.", file=sys.stderr)
        return {}

    if gene_biotype and "gene_biotype" in df.columns:
        df = df[df["gene_biotype"] == gene_biotype]

    counts = {ctx: [0, 0] for ctx in CONTEXTS}   # [n_edited, n_total]

    for read in df.itertuples():
        ref_str  = read.ref_sequence_aligned
        edit_str = read.edit_string
        if not ref_str or not edit_str:
            continue
        n = min(len(ref_str), len(edit_str))

        for i in range(1, n - 1):
            if ref_str[i] != "A":
                continue
            ev = edit_str[i]
            if ev == "2":
                continue
            left  = ref_str[i - 1]
            right = ref_str[i + 1]
            if left not in BASES or right not in BASES:
                continue
            ctx = left + "A" + right
            counts[ctx][1] += 1
            if ev == "1":
                counts[ctx][0] += 1

    result = {}
    for ctx, (n_ed, n_tot) in counts.items():
        result[ctx] = {
            "n_edited":  n_ed,
            "n_total":   n_tot,
            "edit_rate": (n_ed / n_tot) if n_tot > 0 else 0.0,
        }

    nonzero = [c for c in CONTEXTS if result[c]["n_total"] > 0]
    print(f"  [{label}] context editing computed for "
          f"{len(nonzero)}/{len(CONTEXTS)} contexts.", file=sys.stderr)
    return result


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

    for eff, col, label, stats, numAs in libraries:
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

    for j, (eff, col, label, stats, numAs) in enumerate(libraries):
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

def plot_numA_cdf_pyx(libraries: list, output_prefix: str):
    """
    CDF of number of editable sites per read.
    libraries: list of (eff_array, color, label, stats_series, numAs)
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
        x=graph.axis.linear(min=0,
                            title="Number of editable sites per read"),
        y=graph.axis.linear(min=0, max=1,
                            title="Cumulative fraction"),
    )

    for eff, col, label, stats, numAs in libraries:
        if len(numAs) == 0:
            continue
        sf  = np.sort(numAs)
        cdf = np.arange(1, len(sf) + 1) / len(sf)
        g.plot(
            graph.data.points(list(zip(sf.tolist(), cdf.tolist())), x=1, y=2),
            [graph.style.line([col, style.linewidth.normal,
                               style.linestyle.solid])]
        )

    c.insert(g)

    c.text(g.xpos + g.width / 2., g.ypos + g.height + 0.5,
           "number of editable sites per read",
           [pyx_text.halign.center, pyx_text.size.Large])

    # Legend
    leg_x     = g.xpos + g.width + 0.5
    leg_y_top = g.ypos + g.height - 0.3

    for j, (eff, col, label, stats, numAs) in enumerate(libraries):
        ly = leg_y_top - j * leg_dy
        c.stroke(path.line(leg_x, ly, leg_x + leg_lw, ly),
                 [col, style.linewidth.normal, style.linestyle.solid])
        n      = int(stats.get("n_reads", 0))
        # median = float(stats.get("median", float("nan")))
        c.text(leg_x + leg_lw + 0.15, ly,
               f"{label}  (n={n:,})",
               [pyx_text.valign.middle, pyx_text.size.small])

    plot_path = f"{output_prefix}_numA_cdf_pyx"
    c.writePDFfile(plot_path)
    print(f"  Saved -> {plot_path}.pdf", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# 3b. Grouped barplot of per-context editing rate
# ─────────────────────────────────────────────────────────────────────────────

def plot_context_barplot_pyx(context_results: list, output_prefix: str):
    """
    Grouped barplot of A->G editing rate per editable 3-mer context.

    context_results: list of (context_dict, color, label) tuples, where
    context_dict maps context -> {"edit_rate", ...}. One group of bars per
    context (16 groups), one bar per library within each group.
    """
    from pyx import canvas, graph, color, style, path, text as pyx_text

    # Keep only libraries that actually produced context data
    libs = [(res, col, label) for (res, col, label) in context_results
            if res]
    if not libs:
        print("  No context data to plot.", file=sys.stderr)
        return

    n_libs  = len(libs)
    n_ctx   = len(CONTEXTS)
    panel_w = 18
    panel_h = 7

    # y-axis max from the data
    y_max = 0.0
    for res, _, _ in libs:
        for ctx in CONTEXTS:
            if ctx in res:
                y_max = max(y_max, res[ctx]["edit_rate"])
    y_max = max(y_max * 1.15, 0.01)

    c = canvas.canvas()
    g = graph.graphxy(
        width=panel_w, height=panel_h, xpos=0, ypos=0,
        x=graph.axis.linear(min=0, max=n_ctx, parter=None,
                            title="Editable 3-mer context)"),
        y=graph.axis.linear(min=0, max=y_max,
                            title="A{$\\to$}G edit rate"),
    )
    c.insert(g)

    # Bar geometry: each context occupies a unit-wide slot [ci, ci+1].
    group_pad = 0.12
    usable    = 1.0 - 2 * group_pad
    bar_w     = usable / n_libs

    for ci, ctx in enumerate(CONTEXTS):
        for li, (res, col, label) in enumerate(libs):
            rate = res.get(ctx, {}).get("edit_rate", 0.0)
            bx0  = ci + group_pad + li * bar_w
            bx1  = bx0 + bar_w
            # Convert data coords -> canvas coords via g.pos()
            cx0, cy0 = g.pos(bx0, 0.0)
            cx1, cy1 = g.pos(bx1, rate)
            if cy1 > cy0:
                c.fill(path.rect(cx0, cy0, cx1 - cx0, cy1 - cy0),
                       [col])
                c.stroke(path.rect(cx0, cy0, cx1 - cx0, cy1 - cy0),
                         [style.linewidth.thin, color.gray(0.3)])

        # Context label centred under the group
        cxm, cym = g.pos(ci + 0.5, 0.0)
        c.text(cxm, cym - 0.35, ctx,
               [pyx_text.halign.center, pyx_text.size.scriptsize])

    # Title
    c.text(g.xpos + g.width / 2., g.ypos + g.height + 0.5,
           "A{$\\to$}G editing rate by 3-mer context",
           [pyx_text.halign.center, pyx_text.size.Large])

    # Legend
    leg_x     = g.xpos + g.width + 0.4
    leg_y_top = g.ypos + g.height - 0.3
    leg_lw    = 0.6
    leg_dy    = 0.55
    for li, (res, col, label) in enumerate(libs):
        ly = leg_y_top - li * leg_dy
        c.fill(path.rect(leg_x, ly - 0.12, leg_lw, 0.24), [col])
        c.stroke(path.rect(leg_x, ly - 0.12, leg_lw, 0.24),
                 [style.linewidth.thin, color.gray(0.3)])
        c.text(leg_x + leg_lw + 0.15, ly, label,
               [pyx_text.valign.middle, pyx_text.size.small])

    plot_path = f"{output_prefix}_context_editing_barplot_pyx"
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
    p.add_argument("--parquet3", default=None,
                   help="Parquet directory for library 3 (optional)")
    p.add_argument("--label3", default=None)
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
        pyx_color.cmyk(0, 1, 1, 0)         # red
    ]

    libraries_data = []
    all_stats      = []
    context_data   = []
    context_rows   = []

    for idx, (parquet_dir, label, col) in enumerate([
        (args.parquet1, args.label1, colours[0]),
        (args.parquet2, args.label2, colours[1]),
        (args.parquet3, args.label3, colours[2]),
    ]):
        if parquet_dir is None:
            continue
        lbl = label if label is not None else f"Library{idx+1}"
        print(f"\nLoading {lbl}...", file=sys.stderr)
        df  = load_parquet_chunks(parquet_dir)
        eff, stats, numAs = summarise(df, lbl, args.min_a_positions,
                                args.gene_biotype)
        libraries_data.append((eff, col, lbl, stats, numAs))
        all_stats.append(stats)

        # Per-context editing
        ctx_res = compute_context_editing(df, lbl, args.gene_biotype)
        if ctx_res:
            context_data.append((ctx_res, col, lbl))
            for ctx in CONTEXTS:
                d = ctx_res[ctx]
                context_rows.append({
                    "label":     lbl,
                    "context":   ctx,
                    "n_edited":  d["n_edited"],
                    "n_total":   d["n_total"],
                    "edit_rate": d["edit_rate"],
                })

        # Save per-read values
        if len(eff) > 0:
            safe = lbl.replace(" ", "_").replace("/", "_")
            pd.Series(eff, name="global_edit_freq").to_csv(
                f"{out}_{safe}_per_read.csv.gz",
                index=False, compression="gzip"
            )

    # Save summary CSV
    summary_df = pd.DataFrame(all_stats)
    summary_csv = f"{out}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n  Saved summary -> {summary_csv}", file=sys.stderr)

    # Save per-context CSV
    if context_rows:
        ctx_csv = f"{out}_context_editing.csv"
        pd.DataFrame(context_rows).to_csv(ctx_csv, index=False)
        print(f"  Saved context editing -> {ctx_csv}", file=sys.stderr)

    # CDF plot
    print(f"\nGenerating CDF plot...", file=sys.stderr)
    try:
        plot_cdf_pyx(libraries_data, out)
        plot_numA_cdf_pyx(libraries_data, out)
    except Exception as e:
        print(f"  WARNING: pyx CDF plotting failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)


    # Context barplot
    print(f"\nGenerating context barplot...", file=sys.stderr)
    try:
        plot_context_barplot_pyx(context_data, out)
    except Exception as e:
        print(f"  WARNING: pyx context barplot failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()