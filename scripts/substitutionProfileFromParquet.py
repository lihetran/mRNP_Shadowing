'''
June 1, 2026 LT

This script will take two directories of parquet files corresponding to two libraries containing nanopore sequencing data for Polysome Shadowing and compute the error profiles for each substitution type.

inputs:
- parquet directory 1
- parquet directory 2
- ref fasta
- label 1
- label 2
- output_prefix

outputs:
- plot of error profiles for each library
'''

import argparse
import sys
import re
import collections
import math
from pathlib import Path

import pysam
import pandas as pd
import numpy as np
from pyx import canvas, color, style, path, text as pyx_text

################## read in parquets ##########################
def load_parquet_chunks(parquet_dir: str, gene: dict) -> pd.DataFrame:
    parquet_dir = Path(parquet_dir)
    chunks = sorted(parquet_dir.glob("*.parquet"))
    if not chunks:
        return pd.DataFrame()
    chrom  = gene["chrom"]
    strand = gene["strand"]
    dfs = []
    for chunk_path in chunks:
        df = pd.read_parquet(chunk_path)
        mask = (df["chrom"] == chrom) & (df["gene_strand"] == strand)
        df = df[mask]
        if not df.empty:
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_all_parquet_chunks(parquet_dir: str) -> pd.DataFrame:
    """Load all chunks from a parquet directory into one DataFrame."""
    parquet_dir = Path(parquet_dir)
    chunks = sorted(parquet_dir.glob("*.parquet"))
    if not chunks:
        return pd.DataFrame()
    dfs = [pd.read_parquet(c) for c in chunks]
    return pd.concat(dfs, ignore_index=True)

def count_mismatches_per_read(
    ref_fasta: pysam.FastaFile,
    dataframe: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """
    Count substitutions, insertions, and deletions for every read in
    *dataframe*, collapsing to the reference strand.

    Uses the pre-computed ``read_sequence_aligned`` / ``ref_sequence_aligned``
    column pair (gapped strings of equal length) instead of re-parsing CIGAR,
    and ``is_reverse`` for strand collapsing.

    Strand collapsing
    -----------------
    When ``is_reverse`` is True the observed (ref_base, read_base) pair is
    complement-flipped before the substitution key is formed so that, e.g.,
    a G>A mismatch on the minus strand is recorded as C>T.

    Parameters
    ----------
    ref_fasta : pysam.FastaFile
        Open handle to the reference FASTA.  Not used for base lookup here
        (ref bases come from ``ref_sequence_aligned``) but kept in the
        signature for API consistency.
    dataframe : pd.DataFrame
        One row per aligned read.  Required columns:

        =======================  ============================================
        read_id                  unique read identifier (str)
        is_reverse               True if read maps to minus strand (bool)
        read_sequence_aligned    gapped read sequence, e.g. "ACG--T" (str)
        ref_sequence_aligned     gapped ref  sequence, same length  (str)
        =======================  ============================================

    Returns
    -------
    aggregate : dict[str, int]
        Summed counts across **all** reads for the 14 event types
        (12 substitutions + ins + del).
    per_read : pd.DataFrame
        One row per read; columns are ``read_id`` plus one column per event
        type with that read's counts.

    Notes
    -----
    * Gap character is ``'-'``.  A gap in the ref string = insertion; a gap
      in the read string = deletion.  Double-gap positions are skipped.
    * Only canonical bases (A/C/G/T) at non-gap positions are scored;
      positions containing 'N' or other ambiguity codes are skipped.
    """
    CANONICAL = frozenset("ACGT")
    aggregate = init_substitution_dict()
    rows: list[dict] = []

    for _, read in dataframe.iterrows():
        counts     = init_substitution_dict()
        is_reverse = bool(read["is_reverse"])
        read_aln   = read["read_sequence_aligned"].upper()
        ref_aln    = read["ref_sequence_aligned"].upper()

        if len(read_aln) != len(ref_aln):
            # Malformed row – skip rather than crash
            continue

        for r_base, q_base in zip(ref_aln, read_aln):

            # ── insertions / deletions ──────────────────────────────────
            if r_base == "-" and q_base == "-":
                continue                        # double-gap: skip
            if r_base == "-":                   # gap in ref  → insertion
                counts["ins"]    += 1
                aggregate["ins"] += 1
                continue
            if q_base == "-":                   # gap in read → deletion
                counts["del"]    += 1
                aggregate["del"] += 1
                continue

            # ── substitutions ───────────────────────────────────────────
            if r_base not in CANONICAL or q_base not in CANONICAL:
                continue                        # skip ambiguous bases
            if r_base == q_base:
                continue                        # match – nothing to count

            if is_reverse:                      # strand-collapse
                r_base = complement_base(r_base)
                q_base = complement_base(q_base)

            key = f"{r_base}>{q_base}"
            counts[key]    += 1
            aggregate[key] += 1

        row = {"read_id": read["read_id"]}
        row.update(counts)
        rows.append(row)

    per_read = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["read_id"] + list(init_substitution_dict().keys())
    )
    return aggregate, per_read

def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]

def complement_base(b: str) -> str:
    return b.translate(str.maketrans("ACGTacgt", "TGCAtgca"))

def plot_error_profiles(
    aggregate1: dict,
    aggregate2: dict,
    label1: str,
    label2: str,
    output_prefix: str,
) -> None:
    """
    Plot error profiles for two libraries as a two-panel stacked figure using PyX.

    Top panel    : fraction of errors (count / total error events)
    Bottom panel : raw counts

    Both panels share the same x-axis ordering: alphabetical over the 12
    substitution types followed by 'del' and 'ins'.

    Parameters
    ----------
    aggregate1 : dict[str, int]
        Output of count_mismatches_per_read for library 1 (aggregate dict).
    aggregate2 : dict[str, int]
        Output of count_mismatches_per_read for library 2 (aggregate dict).
    label1 : str
        Legend label for library 1.
    label2 : str
        Legend label for library 2.
    output_prefix : str
        Path prefix for output; a '.pdf' suffix is appended automatically.
    """
    from pyx import canvas, graph, color, style, text, unit, document
    from pyx.graph import axis, key as graphkey

    text.set(text.LatexRunner)
    unit.set(xscale=1.2)

    # ── key ordering ────────────────────────────────────────────────────
    sub_keys = sorted(
        [k for k in aggregate1 if ">" in k]
    )                                           # alphabetical: A>C … T>G
    indel_keys = ["del", "ins"]
    keys = sub_keys + indel_keys                # 14 categories total
    n    = len(keys)
    pos  = {k: i for i, k in enumerate(keys)}  # str → x position

    # ── derived series ───────────────────────────────────────────────────
    def to_fractions(agg: dict) -> list[float]:
        total = sum(agg[k] for k in keys)
        if total == 0:
            return [0.0] * n
        return [agg[k] / total for k in keys]

    frac1   = to_fractions(aggregate1)
    frac2   = to_fractions(aggregate2)
    counts1 = [aggregate1[k] for k in keys]
    counts2 = [aggregate2[k] for k in keys]

    # ── color scheme ────────────────────────────────────────────────────
    # Substitution groups share a hue family; indels are grey.
    group_colors = {
        "A": color.rgb(0.22, 0.53, 0.82),   # blue
        "C": color.rgb(0.89, 0.29, 0.20),   # red
        "G": color.rgb(0.18, 0.68, 0.38),   # green
        "T": color.rgb(0.95, 0.60, 0.07),   # amber
        "indel": color.rgb(0.55, 0.55, 0.55),
    }

    def bar_color(key: str) -> color.color:
        if key in ("ins", "del"):
            return group_colors["indel"]
        return group_colors[key[0]]

    # ── bar geometry ────────────────────────────────────────────────────
    bar_w   = 0.30   # width of each individual bar  (graph units = category slots)
    gap     = 0.05   # gap between the two bars of a pair
    offsets = (-bar_w / 2 - gap / 2, bar_w / 2 + gap / 2)

    def make_bar_data(values: list[float], x_offset: float):
        """Return list of (x_left, x_right, y_bottom, y_top) tuples."""
        return [
            (i + x_offset - bar_w / 2,
             i + x_offset + bar_w / 2,
             0,
             v)
            for i, v in enumerate(values)
        ]

    # ── x-axis tick labels ───────────────────────────────────────────────
    def make_xaxis(show_labels: bool) -> axis.linear:
        ticks = [
            axis.tick(i, label=r"\texttt{" + k.replace(">", r"$\rightarrow$") + r"}" if show_labels else "")
            for i, k in enumerate(keys)
        ]
        return axis.linear(
            min=-0.8, max=n - 0.2,
            parter=axis.parter.preiterator(ticks),
            painter=axis.painter.regular(
                labeldirection=axis.painter.rotatetext(90) if show_labels else None,
                labelattrs=[text.size.small] if show_labels else [],
            ),
        )

    # ── build grouped-bar graph painter ─────────────────────────────────
    def draw_bars(g, values1, values2):
        for k, v1, v2 in zip(keys, values1, values2):
            i   = pos[k]
            c1  = bar_color(k)
            c2  = color.transparency(0.35, bar_color(k))   # lighter tint for lib2
            # library 1 – left bar
            x0, x1 = i + offsets[0] - bar_w / 2, i + offsets[0] + bar_w / 2
            g.fill(graph.style.barpos(
                graph.data.list(
                    [(x0, 0, x1, v1)],
                    xmin=0, ymin=1, xmax=2, ymax=3
                )
            ))

        # Use low-level path drawing instead for full control
        import pyx.path as ppath
        import pyx.deco as deco

        for k, v1, v2 in zip(keys, values1, values2):
            i  = pos[k]
            c1 = bar_color(k)
            c2 = color.transparency(0.4) + bar_color(k)

            for v, off, col, lbl in [
                (v1, offsets[0], c1, label1),
                (v2, offsets[1], c2, label2),
            ]:
                xl = i + off - bar_w / 2
                xr = i + off + bar_w / 2
                # convert to graph coords then draw filled rect
                p = ppath.rect(xl, 0, bar_w, v)
                g.plot(
                    graph.data.list(
                        [(xl + bar_w / 2, v)], x=1, y=2
                    ),
                    [graph.style.symbol(
                        graph.style.symbol.square,
                        size=0, symbolattrs=[col]
                    )]
                )

    # ── canvas & panels ──────────────────────────────────────────────────
    c    = canvas.canvas()
    W, H = 14, 4.5   # cm per panel

    def make_panel(yvals1, yvals2, ylabel: str, ymax: float,
                   show_xlabels: bool, y_offset: float) -> graph.graphxy:

        g = graph.graphxy(
            width=W,
            height=H,
            x=make_xaxis(show_xlabels),
            y=axis.linear(min=0, max=ymax,
                          title=ylabel,
                          painter=axis.painter.regular(
                              labelattrs=[text.size.small]
                          )),
            key=graphkey.key(
                pos="tr",
                dist=0.1,
            ) if y_offset == 0 else None,
        )

        # Draw bars as filled rectangles via low-level canvas paths
        for k, v1, v2 in zip(keys, yvals1, yvals2):
            xi   = pos[k]
            col1 = bar_color(k)
            col2 = bar_color(k)   # same hue, different alpha via deco

            for v, off, col, alpha, lbl in [
                (v1, offsets[0], col1, 1.0, label1),
                (v2, offsets[1], col2, 0.5, label2),
            ]:
                if v == 0:
                    continue
                # data → canvas coordinates
                x_data = xi + off
                x0_c, _ = g.pos(x_data - bar_w / 2, 0)
                x1_c, _ = g.pos(x_data + bar_w / 2, 0)
                _,  y0_c = g.pos(x_data, 0)
                _,  y1_c = g.pos(x_data, v)

                import pyx.path as ppath
                rect = ppath.rect(x0_c, y0_c, x1_c - x0_c, y1_c - y0_c)
                fill_color = color.transparency(1 - alpha) + col
                c.fill(rect, [fill_color])
                c.stroke(rect, [color.gray(0.3), style.linewidth.thin])

        # invisible scatter plot just to get the axes / title rendered
        g.plot(
            graph.data.list([(0, 0)], x=1, y=2),
            [graph.style.symbol(graph.style.symbol.circle, size=0)]
        )

        c.insert(g, [canvas.trafo.translate(0, y_offset)])
        return g

    # ── compute y-axis ceilings ──────────────────────────────────────────
    all_fracs  = frac1  + frac2
    all_counts = counts1 + counts2
    ymax_frac  = max(all_fracs)  * 1.15 or 0.1
    ymax_count = max(all_counts) * 1.15 or 10

    # top panel = fractions, bottom panel = counts
    panel_gap = 1.5   # cm between panels
    make_panel(frac1,   frac2,   r"Fraction of errors",  ymax_frac,
               show_xlabels=False, y_offset=H + panel_gap)
    make_panel(counts1, counts2, r"Raw error counts",    ymax_count,
               show_xlabels=True,  y_offset=0)

    # ── manual legend ────────────────────────────────────────────────────
    lx = W - 3.5
    ly = 2 * H + panel_gap + 0.3
    for i, (lbl, alpha) in enumerate([(label1, 1.0), (label2, 0.5)]):
        swatch_col = color.transparency(1 - alpha) + color.rgb(0.4, 0.4, 0.4)
        import pyx.path as ppath
        sw = ppath.rect(lx, ly - i * 0.5, 0.35, 0.25)
        c.fill(sw, [swatch_col])
        c.stroke(sw, [color.gray(0.3), style.linewidth.thin])
        c.text(lx + 0.45, ly - i * 0.5 + 0.05,
               r"\small " + lbl,
               [text.halign.left, text.valign.bottom])

    # ── group color legend (ref base) ───────────────────────────────────
    group_labels = [("A>x", "A"), ("C>x", "C"), ("G>x", "G"),
                    ("T>x", "T"), ("indel", "indel")]
    gx = 0.2
    gy = 2 * H + panel_gap + 0.3
    for j, (glbl, gkey) in enumerate(group_labels):
        import pyx.path as ppath
        sw = ppath.rect(gx + j * 2.2, gy, 0.25, 0.25)
        c.fill(sw, [group_colors[gkey]])
        c.text(gx + j * 2.2 + 0.35, gy + 0.05,
               r"\small " + glbl,
               [text.halign.left, text.valign.bottom])

    # ── write output ─────────────────────────────────────────────────────
    out_path = f"{output_prefix}.pdf"
    c.writePDFfile(out_path)
    print(f"Wrote {out_path}")