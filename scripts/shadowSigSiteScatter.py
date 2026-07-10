'''
July 10. 2026

This script will gather the number of significant sites called as an output of BinomialBREAM2.py and create a scatter plot comparing the number of significant sites in two different libraries.
The idea is to see if there is a correlation between the number of significant sites in ribosome-less (phenol) and ribosome-containing libraries. I'll do 3 panels, one for each of the 3 regions (5'UTR, CDS, 3'UTR).
The x-axis will be the number of significant sites in the ribosome-less library and the y-axis will be the number of significant sites in the ribosome-containing library.

For now, I'll do this with the n_sites_cds, n_sites_utr5, and n_sites_utr3 fields in the parquet output of BinomialBREAM2.py. In the future, we may want to compute this from the shadow calls themselves, but for now this will be a good starting point.

inputs:
    --parquet1, ribosome-less (phenol) shadow calls
    --parquet2, ribosome-containing library shadow calls
'''

import argparse
import sys

import numpy as np
import pandas as pd
from logJosh import Tee

REGIONS = [("utr5", "5'UTR"), ("cds", "CDS"), ("utr3", "3'UTR")]


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-gene significant-site rate scatter, two libraries.")
    p.add_argument("--summary1", required=True,
                   help="Ribosome-less (phenol) summary CSV")
    p.add_argument("--summary2", required=True,
                   help="Ribosome-containing summary CSV")
    p.add_argument("--label1", default="ribosome-less")
    p.add_argument("--label2", default="ribosome-containing")
    p.add_argument("--output", default="sig_site_compare")
    p.add_argument("--min_tested", type=int, default=10,
                   help="Min tested sites in a region (either library) for a "
                        "gene to be plotted in that panel (default: 10)")
    return p.parse_args()


def load_summary(path):
    df = pd.read_csv(path)
    needed = {"gene"}
    for key, _ in REGIONS:
        needed |= {f"frac_sig_{key}", f"n_sites_{key}"}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"ERROR: {path} missing columns: {sorted(missing)}")
    return df


def plot_pyx(merged, args):
    from pyx import canvas, graph, color, style, path, text as pyx_text

    col_pt = color.cmyk(1, 0.5, 0, 0)  # blue points
    col_diag = color.cmyk(0, 0, 0, 0.5)  # gray diagonal
    panel_w = 5.0
    panel_h = 5.0
    gap = 1.6

    c = canvas.canvas()

    for pi, (key, region_label) in enumerate(REGIONS):
        fx = f"frac_sig_{key}_1"
        fy = f"frac_sig_{key}_2"
        nx = f"n_sites_{key}_1"
        ny = f"n_sites_{key}_2"

        # Keep genes with enough tested sites in BOTH libraries
        sub = merged[(merged[nx] >= args.min_tested) &
                     (merged[ny] >= args.min_tested)].copy()

        xpos = pi * (panel_w + gap)

        g = graph.graphxy(
            width=panel_w, height=panel_h, xpos=xpos, ypos=0,
            x=graph.axis.linear(min=0, max=1, title=args.label1),
            y=graph.axis.linear(min=0, max=1, title=args.label2),
        )

        # y = x diagonal
        g.plot(graph.data.points([(0, 0), (1, 1)], x=1, y=2),
               [graph.style.line([col_diag, style.linewidth.thin,
                                  style.linestyle.dashed])])

        if not sub.empty:
            # size by evidence: min tested sites across libraries
            ev = np.minimum(sub[nx].values, sub[ny].values).astype(float)
            # map evidence to a radius 0.04..0.14 (log-scaled)
            lo, hi = ev.min(), ev.max()
            if hi > lo:
                r = 0.04 + 0.10 * (np.log10(ev) - np.log10(lo)) / \
                    (np.log10(hi) - np.log10(lo))
            else:
                r = np.full_like(ev, 0.07)

            pts = list(zip(sub[fx].tolist(), sub[fy].tolist(), r.tolist()))
            g.plot(graph.data.points(pts, x=1, y=2, size=3),
                   [graph.style.symbol(
                       graph.style.symbol.circle, size=0.1,
                       symbolattrs=[col_pt, style.linewidth.thin])])

        c.insert(g)
        c.text(xpos + panel_w / 2., panel_h + 0.35,
               "%s  (n=%d)" % (region_label, len(sub)),
               [pyx_text.halign.center, pyx_text.size.normalsize])

    c.text(1.5 * (panel_w + gap) - gap / 2., panel_h + 1.1,
           "Per-gene significant-site rate: %s vs %s" %
           (args.label1, args.label2),
           [pyx_text.halign.center, pyx_text.size.large])

    out_path = f"{args.output}_sig_site_scatter_pyx"
    c.writePDFfile(out_path)
    print(f"  Saved -> {out_path}.pdf", file=sys.stderr)


def main():
    args = parse_args()

    print("Loading summaries...", file=sys.stderr)
    df1 = load_summary(args.summary1)
    df2 = load_summary(args.summary2)
    print(f"  {args.label1}: {len(df1):,} genes", file=sys.stderr)
    print(f"  {args.label2}: {len(df2):,} genes", file=sys.stderr)

    # Join on gene; suffix _1 (lib1) / _2 (lib2)
    merged = df1.merge(df2, on="gene", suffixes=("_1", "_2"))
    print(f"  {len(merged):,} genes present in both.", file=sys.stderr)
    if merged.empty:
        sys.exit("ERROR: no genes shared between the two summaries.")

    # Per-region correlation on the plotted subset, printed for reference
    print("\nSpearman correlation of frac_sig (genes passing min_tested "
          f"= {args.min_tested} in both):", file=sys.stderr)
    for key, region_label in REGIONS:
        nx, ny = f"n_sites_{key}_1", f"n_sites_{key}_2"
        fx, fy = f"frac_sig_{key}_1", f"frac_sig_{key}_2"
        sub = merged[(merged[nx] >= args.min_tested) &
                     (merged[ny] >= args.min_tested)]
        if len(sub) >= 3:
            rho = sub[[fx, fy]].corr(method="spearman").iloc[0, 1]
            dy = float((sub[fy] - sub[fx]).mean())
            print(f"  {region_label:6s}  n={len(sub):4d}  rho={rho:+.3f}  "
                  f"mean(y-x)={dy:+.4f}", file=sys.stderr)
        else:
            print(f"  {region_label:6s}  n={len(sub):4d}  (too few to correlate)",
                  file=sys.stderr)

    # Save the merged table for downstream inspection
    merged_path = f"{args.output}_merged.csv"
    merged.to_csv(merged_path, index=False)
    print(f"\n  Merged table -> {merged_path}", file=sys.stderr)

    print("\nPlotting...", file=sys.stderr)
    try:
        plot_pyx(merged, args)
    except Exception as e:
        print(f"  WARNING: pyx plot failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    Tee()
    main()