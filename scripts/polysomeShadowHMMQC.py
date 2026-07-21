"""
shadowSizeQC.py — Liam Tran, July 2026

Assess protected-region ("shadow") sizes from HMM shadow-call parquets produced
by bayesianShadowClassifier.py (posterior P_B schema, native list columns).

Two complementary views:
  VIEW 1  footprint SIZE at a fixed P_B cutoff (runs merged across small gaps,
          true genomic-nt size) -> "how big are footprints?"
  VIEW 2  stringency SWEEP across P_B cutoffs (runs unmerged, padded length)
          -> "how does calling stringency reshape what I detect?"

Input parquet columns used per read:
  read_id, gene_strand, absolute_indices, shadow_gpos, shadow_P_B
  (shadow_P_B is UNFILTERED: one posterior per Ref=A site, gaps visible.)

Run:
  python3 shadowSizeQC.py inFilesParquet.txt outPrefix
where inFilesParquet.txt is line-delimited:  fileName  rep  parquetFile
"""

import sys, math, collections
import numpy as np
import pandas as pd

try:
    from pyx import canvas, graph, color, style, text as pyx_text
except ImportError:
    canvas = None   # plotting optional; extraction still works without pyx


# ─────────────────────────────────────────────────────────────────────────
# Config (edit to your biology)
# ─────────────────────────────────────────────────────────────────────────
PROB_CUTOFFS   = [0.5, 0.6, 0.7, 0.8, 0.9]   # sweep for View 2
FIXED_CUTOFF   = 0.7                          # single cutoff for View 1
N_PAD          = 50                           # padding window (aligned-length def)
MAX_GAP_NT     = 20                           # bridge gaps <= this when merging (View 1)
SIZE_MEASURE   = "genomic_nt"                 # 'genomic_nt' | 'n_sites' | 'aligned_len'
SIZE_BIN       = 5                            # bp per bin, View 1
SIZE_RANGE     = (0, 120)                     # View 1 size axis
SWEEP_BIN      = 10                           # bp per bin, View 2
SWEEP_RANGE    = (30, 100)                    # View 2 length axis
PALETTE        = None                         # list of pyx colors, set after import

def _libcolor(i):
    return PALETTE[i % len(PALETTE)]


# ─────────────────────────────────────────────────────────────────────────
# Core extraction
# ─────────────────────────────────────────────────────────────────────────
def _read_arrays(row):
    """Native parquet gives numpy arrays; coerce to clean python scalars."""
    pb   = [float(x) for x in row.shadow_P_B]
    gpos = [int(g)   for g in row.shadow_gpos]
    ai   = row.absolute_indices
    return pb, gpos, ai


def extract_shadow_runs(df, probCutOff, N=N_PAD):
    """
    Unmerged protected runs (P_B > cutoff), one break per sub-cutoff site.
    Each run: read_id, aligned_len (padded), genomic_nt (unpadded), n_sites.
    Strand-safe; drops NaN positions.
    """
    halfN = N // 2
    runs = []
    for row in df.itertuples(index=False):
        pb, gpos, ai = _read_arrays(row)
        n_sh = len(pb)
        if n_sh == 0:
            continue
        absIndices = list(ai)
        numPos = len(absIndices)
        g2r = {int(g): idx for idx, g in enumerate(absIndices)
               if g is not None and g == g}          # drop NaN, int keys

        i = 0
        while i < n_sh:
            if pb[i] > probCutOff:
                j = i
                while j < n_sh and pb[j] > probCutOff:
                    j += 1
                fr, lr = g2r.get(gpos[i]), g2r.get(gpos[j-1])
                if fr is not None and lr is not None:
                    lo, hi = sorted((fr, lr))          # minus-strand safe
                    startI = max(0, lo - halfN)
                    endI   = min(numPos, hi + halfN + 1)
                    runs.append({
                        "read_id":     row.read_id,
                        "aligned_len": endI - startI,               # padded
                        "genomic_nt":  abs(gpos[j-1] - gpos[i]) + 1, # unpadded
                        "n_sites":     j - i,
                    })
                i = j
            else:
                i += 1
    return runs


def extract_merged_runs(df, probCutOff, N=N_PAD, max_gap_nt=MAX_GAP_NT):
    """
    Protected runs merged across genomic gaps <= max_gap_nt, so a footprint with
    a noisy interior site stays one footprint. Each run: read_id, genomic_nt,
    n_sites. Use for the footprint-SIZE view.
    """
    runs = []
    for row in df.itertuples(index=False):
        pb, gpos, ai = _read_arrays(row)
        if len(pb) == 0:
            continue
        prot = [k for k in range(len(pb)) if pb[k] > probCutOff]
        if not prot:
            continue
        groups = [[prot[0]]]
        for k in prot[1:]:
            if abs(gpos[k] - gpos[groups[-1][-1]]) <= max_gap_nt:
                groups[-1].append(k)
            else:
                groups.append([k])
        for grp in groups:
            g_first, g_last = gpos[grp[0]], gpos[grp[-1]]
            runs.append({
                "read_id":    row.read_id,
                "genomic_nt": abs(g_last - g_first) + 1,
                "n_sites":    len(grp),
            })
    return runs


# ─────────────────────────────────────────────────────────────────────────
# Histogramming
# ─────────────────────────────────────────────────────────────────────────
def _bin(sizes, lo, hi, bin_width):
    edges = list(range(lo, hi + bin_width, bin_width))
    counts = collections.defaultdict(int)
    for s in sizes:
        if lo <= s <= hi:
            counts[lo + ((s - lo) // bin_width) * bin_width] += 1
    return dict(counts), edges


def footprint_size_hist(df, probCutOff, N=N_PAD, max_gap_nt=MAX_GAP_NT,
                        measure=SIZE_MEASURE, bin_width=SIZE_BIN, size_range=SIZE_RANGE):
    runs  = extract_merged_runs(df, probCutOff, N, max_gap_nt)
    sizes = [r[measure] for r in runs]
    counts, edges = _bin(sizes, size_range[0], size_range[1], bin_width)
    return counts, edges, sizes


def stringency_sweep_hist(df, probCutOffs=PROB_CUTOFFS, N=N_PAD,
                          measure="aligned_len", bin_width=SWEEP_BIN,
                          size_range=SWEEP_RANGE):
    lo, hi = size_range
    out = {}
    for cut in probCutOffs:
        runs  = extract_shadow_runs(df, cut, N)
        sizes = [r[measure] for r in runs]
        counts, edges = _bin(sizes, lo, hi, bin_width)
        out[cut] = counts
        print("    cutoff %.2f: %d runs, %d in [%d,%d]"
              % (cut, len(runs), sum(counts.values()), lo, hi), file=sys.stderr)
    return out, edges


# ─────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────
def _nice(lim):
    raw = lim / 2.
    mag = 10. ** math.floor(math.log10(raw)) if raw > 0 else 1
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return graph.axis.parter.linear(tickdists=[m * mag])
    return None


def plot_footprint_sizes(lib_hists_by_cut, edges, pdf_path,
                         measure=SIZE_MEASURE, bin_width=SIZE_BIN, merged=True):
    """
    Multi-panel footprint SIZE figure, laid out like the stringency sweep:
    one panel per P_B cutoff (low at bottom, high at top), linked x-axes; within
    each panel one line per library (consistent color across panels), normalized
    to frequency. Key on the top panel.

    lib_hists_by_cut: {cutoff: {libraryID: (counts_dict, sizes_list)}}
    merged: only affects the title annotation (the histograms are already
            built merged or unmerged by the caller).
    """
    if canvas is None:
        print("pyx not available; skipping plot", file=sys.stderr); return
    cutoffs = sorted(lib_hists_by_cut)
    libIDs  = sorted({lib for h in lib_hists_by_cut.values() for lib in h})
    lefts   = edges[:-1]
    centers = [e + bin_width / 2. for e in lefts]
    unit    = {"genomic_nt": "nt", "n_sites": "Ref=A sites",
               "aligned_len": "aligned nt"}[measure]
    pw, ph, vgap = 10, 2.4, 0.5

    # shared y-max (frequency) across every cutoff/library/bin
    ymax = 0.0
    for cut in cutoffs:
        for libID, (counts, _sizes) in lib_hists_by_cut[cut].items():
            tot = sum(counts.values()) or 1
            for e in lefts:
                ymax = max(ymax, counts.get(e, 0) / tot)
    ymax = (ymax or 1) * 1.15

    tag = "merged" if merged else "unmerged"
    c = canvas.canvas(); bottom = None
    for ii, cut in enumerate(cutoffs):                 # low at bottom, high at top
        ypos = ii * (ph + vgap)
        is_top = (ii == len(cutoffs) - 1)
        if bottom is None:
            xax = graph.axis.linear(min=edges[0], max=edges[-1],
                                    title="Footprint size (%s, %s)" % (unit, tag))
        else:
            xax = graph.axis.linkedaxis(bottom.axes["x"], painter=None)
        gkw = dict(width=pw, height=ph, xpos=0, ypos=ypos, x=xax,
                   y=graph.axis.linear(min=0, max=ymax, parter=_nice(ymax),
                                       title=r"freq, P$_B>$%s" % cut))
        if is_top:
            gkw["key"] = graph.key.key(pos="tr", hinside=0)
        g = graph.graphxy(**gkw)
        for i, libID in enumerate(libIDs):
            counts, sizes = lib_hists_by_cut[cut].get(libID, ({}, []))
            tot = sum(counts.values()) or 1
            if is_top:
                med = np.median(sizes) if sizes else 0
                title = r"%s (med %.0f, n=%d)" % (libID.replace("_", r"\_"),
                                                  med, len(sizes))
            else:
                title = None
            g.plot(graph.data.points([(ctr, counts.get(e, 0) / tot)
                                      for ctr, e in zip(centers, lefts)],
                                     x=1, y=2, title=title),
                   [graph.style.line([_libcolor(i), style.linewidth.Thick])])
        c.insert(g)
        if bottom is None:
            bottom = g
    c.writePDFfile(str(pdf_path))


def plot_stringency_sweep(lib_hists, edges, pdf_path, bin_width=SWEEP_BIN,
                          normalize=True):
    """
    lib_hists: {libraryID: {cutoff: counts_dict}}. Vertical array of panels, one
    per cutoff (low at bottom, high at top), linked x-axes; within each panel one
    line per library (same color across panels). Key on the top panel, outside.
    """
    if canvas is None:
        print("pyx not available; skipping plot", file=sys.stderr); return
    libIDs  = sorted(lib_hists)
    cutoffs = sorted({c for h in lib_hists.values() for c in h})
    lefts   = edges[:-1]
    centers = [e + bin_width / 2. for e in lefts]
    pw, ph, vgap = 10, 2.4, 0.5

    # shared y-max across every library/cutoff/bin
    ymax = 0.0
    for libID in libIDs:
        for cut in cutoffs:
            counts = lib_hists[libID].get(cut, {})
            tot = sum(counts.values()) if normalize else 1
            for e in lefts:
                ymax = max(ymax, counts.get(e, 0) / (tot or 1))
    ymax = (ymax or 1) * 1.15
    ylab = "frequency" if normalize else "count"

    c = canvas.canvas(); bottom = None
    for ii, cut in enumerate(cutoffs):                 # low at bottom, high at top
        ypos = ii * (ph + vgap)
        is_top = (ii == len(cutoffs) - 1)
        if bottom is None:
            xax = graph.axis.linear(min=edges[0], max=edges[-1],
                                    title="Shadow length (aligned nt)")
        else:
            xax = graph.axis.linkedaxis(bottom.axes["x"], painter=None)
        gkw = dict(width=pw, height=ph, xpos=0, ypos=ypos, x=xax,
                   y=graph.axis.linear(min=0, max=ymax, parter=_nice(ymax),
                                       title=r"%s, P$_B>$%s" % (ylab, cut)))
        if is_top:
            gkw["key"] = graph.key.key(pos="tr", hinside=0)
        g = graph.graphxy(**gkw)
        for i, libID in enumerate(libIDs):
            counts = lib_hists[libID].get(cut, {})
            tot = sum(counts.values()) if normalize else 1
            title = libID.replace("_", r"\_") if is_top else None
            g.plot(graph.data.points([(ctr, counts.get(e, 0) / (tot or 1))
                                      for ctr, e in zip(centers, lefts)],
                                     x=1, y=2, title=title),
                   [graph.style.line([_libcolor(i), style.linewidth.Thick])])
        c.insert(g)
        if bottom is None:
            bottom = g
    c.writePDFfile(str(pdf_path))


# ─────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────
def _size_hist(df, cutoff, merged):
    """Footprint-size histogram for one library at one cutoff, merged or not."""
    if merged:
        runs = extract_merged_runs(df, cutoff, N_PAD, MAX_GAP_NT)
    else:
        runs = extract_shadow_runs(df, cutoff, N_PAD)
    sizes = [r[SIZE_MEASURE] for r in runs]
    counts, edges = _bin(sizes, SIZE_RANGE[0], SIZE_RANGE[1], SIZE_BIN)
    return counts, edges, sizes


def analyze_library(parquetFile, libraryID):
    """Return per-cutoff size hists (merged & unmerged) + the sweep, for one lib."""
    print("Loading %s (%s)..." % (parquetFile, libraryID), file=sys.stderr)
    df = pd.read_parquet(parquetFile)          # native columns, no JSON decode

    merged_by_cut   = {}    # cutoff -> (counts, sizes)
    unmerged_by_cut = {}
    edges = None
    for cut in PROB_CUTOFFS:
        mc, edges, ms = _size_hist(df, cut, merged=True)
        uc, _, us = _size_hist(df, cut, merged=False)
        merged_by_cut[cut]   = (mc, ms)
        unmerged_by_cut[cut] = (uc, us)
    # console summary at the fixed cutoff
    if FIXED_CUTOFF in merged_by_cut:
        ms = merged_by_cut[FIXED_CUTOFF][1]
        if ms:
            print("    P_B>%s merged: n=%d median=%.0f IQR=[%.0f,%.0f]"
                  % (FIXED_CUTOFF, len(ms), np.median(ms),
                     np.percentile(ms, 25), np.percentile(ms, 75)), file=sys.stderr)

    sweep_hist, sweep_edges = stringency_sweep_hist(df)
    return merged_by_cut, unmerged_by_cut, edges, sweep_hist, sweep_edges


def main(args):
    global PALETTE
    if canvas is not None:
        PALETTE = [color.cmyk(1, 0.5, 0, 0), color.cmyk(0, 1, 1, 0),
                   color.cmyk(0.4, 1, 0, 0), color.cmyk(1, 0, 1, 0.1),
                   color.cmyk(0, 0.5, 1, 0), color.cmyk(0.7, 0, 0, 0),
                   color.cmyk(0, 0, 0, 0.7), color.cmyk(0.3, 0, 1, 0.2)]

    parquetList, outPrefix = args[0], args[1]

    # {cutoff: {libID: (counts, sizes)}} for merged and unmerged
    merged_by_cut   = collections.defaultdict(dict)
    unmerged_by_cut = collections.defaultdict(dict)
    sweep_by_lib    = {}
    fp_edges = sweep_edges = None

    with open(parquetList) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            fileName, rep, parquetFile = parts[0], parts[1], parts[2]
            libraryID = "%s-%s" % (fileName, rep)
            mbc, ubc, fpe, swh, swe = analyze_library(parquetFile, libraryID)
            for cut in mbc:
                merged_by_cut[cut][libraryID]   = mbc[cut]
                unmerged_by_cut[cut][libraryID] = ubc[cut]
            sweep_by_lib[libraryID] = swh
            fp_edges, sweep_edges = fpe, swe

    if not sweep_by_lib:
        print("no libraries processed", file=sys.stderr); return

    print("Plotting combined figures across %d libraries..." % len(sweep_by_lib),
          file=sys.stderr)
    # View 1a: footprint size, MERGED, multi-panel over cutoffs
    plot_footprint_sizes(dict(merged_by_cut), fp_edges,
                         "%s.footprint_sizes.merged" % outPrefix, merged=True)
    # View 1b: footprint size, UNMERGED, same layout
    plot_footprint_sizes(dict(unmerged_by_cut), fp_edges,
                         "%s.footprint_sizes.unmerged" % outPrefix, merged=False)
    # # View 2: stringency sweep (aligned length)
    # plot_stringency_sweep(sweep_by_lib, sweep_edges,
    #                       "%s.stringency_sweep" % outPrefix)


if __name__ == "__main__":
    main(sys.argv[1:])