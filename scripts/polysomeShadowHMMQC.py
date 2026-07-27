"""
shadowSizeQC.py — Liam Tran, July 2026

Assess protected-region ("shadow") sizes from HMM shadow-call parquets produced
by bayesianShadowClassifier.py (posterior P_B schema, native list columns).

Two complementary views:
  VIEW 1  footprint SIZE at a fixed P_B cutoff, unmerged (one break per
          sub-cutoff site) -> "how big are footprints?"
  VIEW 2  stringency SWEEP across P_B cutoffs (unmerged, padded length)
          -> "how does calling stringency reshape what I detect?"
  VIEW 3  per-gene call RATE, and footprint size restricted to runs
          overlapping a His codon (via findHisCodonPositions.py's cache)

Input parquet columns used per read:
  read_id, shadow_gene, absolute_indices, shadow_gpos, shadow_P_B
  (shadow_P_B is UNFILTERED: one posterior per Ref=A site, gaps visible.)

Run:
  python3 shadowSizeQC.py inFilesParquet.txt outPrefix hisCodonPositions.pickle
where inFilesParquet.txt is line-delimited:  fileName  rep  parquetFile
and hisCodonPositions.pickle is findHisCodonPositions.py's output.
"""

import sys, math, collections, pickle
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
FIXED_CUTOFF   = 0.7                          # single cutoff for View 1 & His-codon rule
N_PAD          = 50                           # padding window (aligned-length def)
SIZE_MEASURE   = "genomic_nt"                 # 'genomic_nt' | 'n_sites' | 'aligned_len'
SIZE_BIN       = 5                            # bp per bin, View 1
SIZE_RANGE     = (0, 120)                     # View 1 size axis
SWEEP_BIN      = 10                           # bp per bin, View 2
SWEEP_RANGE    = (30, 100)                    # View 2 length axis
MIN_RUNS_PER_GENE  = 5                        # drop a gene from the size-shape
                                               # aggregate below this many runs
MIN_SITES_PER_GENE = 20                       # drop a gene from the call-rate
                                               # summary below this many scored sites
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
                    g_lo, g_hi = sorted((gpos[i], gpos[j-1]))
                    runs.append({
                        "read_id":     row.read_id,
                        "gene":        row.shadow_gene,
                        "aligned_len": endI - startI,               # padded
                        "genomic_nt":  abs(gpos[j-1] - gpos[i]) + 1, # unpadded
                        "n_sites":     j - i,
                        "gpos_lo":     g_lo,             # unpadded genomic span,
                        "gpos_hi":     g_hi,             # for His-codon overlap checks
                    })
                i = j
            else:
                i += 1
    return runs


# ─────────────────────────────────────────────────────────────────────────
# His codon overlap (findHisCodonPositions.py pickle as input)
# ─────────────────────────────────────────────────────────────────────────
def load_his_codon_gpos(pickle_path):
    """
    Load findHisCodonPositions.py's {gene: [tx_positions, gpos_positions]}
    cache and keep just the genomic side, {gene: [gpos, ...]} -- genomic
    space is the only coordinate system shadow runs here are compared
    against (shadow_gpos, not shadow_tx_pos).
    """
    with open(pickle_path, "rb") as f:
        his_positions = pickle.load(f)
    return {gene: gpos for gene, (_tx_pos, gpos) in his_positions.items()}


def _run_overlaps_his(run, his_gpos_by_gene):
    """True if run's unpadded genomic span [gpos_lo, gpos_hi] contains >=1
    His codon position from this run's own gene."""
    his_list = his_gpos_by_gene.get(run["gene"])
    if not his_list:
        return False
    return any(run["gpos_lo"] <= hp <= run["gpos_hi"] for hp in his_list)


def extract_his_shadow_runs(df, his_gpos_by_gene, cutoff=FIXED_CUTOFF, N=N_PAD):
    """
    Shadow runs (same unmerged extraction as extract_shadow_runs) kept only
    if P_B > cutoff AND the run's genomic span overlaps >=1 His codon
    position in its own gene. This is the rule as currently defined: a
    footprint "at a His codon" just needs to cover the codon somewhere in
    its unpadded span, not have the codon be one of its own scored sites.
    """
    runs = extract_shadow_runs(df, cutoff, N)
    return [r for r in runs if _run_overlaps_his(r, his_gpos_by_gene)]


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


def _gene_weighted_freq(runs, measure, lo, hi, bin_width,
                        min_runs_per_gene=MIN_RUNS_PER_GENE):
    """
    Per-gene-then-aggregate frequency: bin each gene's own runs and
    normalize THAT gene's histogram to sum to 1, then average those
    per-gene frequency curves bin-wise. Every gene gets one equal vote
    regardless of read depth -- pooling raw counts across the whole library
    first (the old behavior) lets a single highly-expressed gene set the
    shape (e.g. one real library: one gene was 40% of all reads).

    Genes with fewer than min_runs_per_gene runs are dropped entirely --
    too few observations for that gene's own histogram to mean anything.

    Returns (freq_dict, edges, n_genes_used, n_genes_dropped).
    """
    by_gene = collections.defaultdict(list)
    for r in runs:
        by_gene[r["gene"]].append(r[measure])

    edges = list(range(lo, hi + bin_width, bin_width))
    bin_lefts = edges[:-1]
    freq_sum = {e: 0.0 for e in bin_lefts}

    n_used = 0
    for gene, sizes in by_gene.items():
        if len(sizes) < min_runs_per_gene:
            continue
        counts, _ = _bin(sizes, lo, hi, bin_width)
        tot = sum(counts.values())
        if tot == 0:
            continue
        n_used += 1
        for e in bin_lefts:
            freq_sum[e] += counts.get(e, 0) / tot

    n_dropped = len(by_gene) - n_used
    if n_used == 0:
        return {e: 0.0 for e in bin_lefts}, edges, 0, n_dropped
    return {e: v / n_used for e, v in freq_sum.items()}, edges, n_used, n_dropped


def gene_call_rates(df, cutoff, min_sites_per_gene=MIN_SITES_PER_GENE):
    """
    Per-gene fraction of scored Ref=A sites called protected (P_B > cutoff),
    pooling all reads within each gene first. This is the MAGNITUDE
    counterpart to _gene_weighted_freq's shape aggregation: it deliberately
    keeps depth-independent per-gene rates comparable side by side across
    libraries (e.g. ribosome-containing vs. a ribosome-less control), rather
    than washing that difference out the way equal-weighting the size
    histograms does on purpose.

    Genes with fewer than min_sites_per_gene total scored sites are dropped
    -- too few sites to trust a rate from.

    Returns {gene: rate}.
    """
    n_prot = collections.defaultdict(int)
    n_tot  = collections.defaultdict(int)
    for row in df.itertuples(index=False):
        pb, _gpos, _ai = _read_arrays(row)
        gene = row.shadow_gene
        n_tot[gene]  += len(pb)
        n_prot[gene] += sum(1 for p in pb if p > cutoff)
    return {g: n_prot[g] / n_tot[g] for g in n_tot
            if n_tot[g] >= min_sites_per_gene}


def stringency_sweep_hist(df, probCutOffs=PROB_CUTOFFS, N=N_PAD,
                          measure="aligned_len", bin_width=SWEEP_BIN,
                          size_range=SWEEP_RANGE):
    lo, hi = size_range
    out = {}
    for cut in probCutOffs:
        runs = extract_shadow_runs(df, cut, N)
        freq, edges, n_used, n_dropped = _gene_weighted_freq(
            runs, measure, lo, hi, bin_width)
        out[cut] = freq
        print("    cutoff %.2f: %d runs, %d genes used, %d dropped (<%d runs)"
              % (cut, len(runs), n_used, n_dropped, MIN_RUNS_PER_GENE), file=sys.stderr)
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
                         measure=SIZE_MEASURE, bin_width=SIZE_BIN,
                         ylabel="freq"):
    """
    Multi-panel footprint SIZE figure, laid out like the stringency sweep:
    one panel per P_B cutoff (low at bottom, high at top), linked x-axes; within
    each panel one line per library (consistent color across panels). Key on
    the top panel.

    lib_hists_by_cut: {cutoff: {libraryID: (values_dict, sizes_list)}}. By
    default (ylabel="freq") values_dict is expected to be the per-gene-then-
    aggregated frequency from _gene_weighted_freq / _size_hist -- every gene
    voted once regardless of read depth, no re-normalization happens here.
    Pass ylabel="count" with a raw _bin(sizes, ...) counts dict instead for
    the simple pooled-count supplement to that.
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

    # shared y-max across every cutoff/library/bin
    ymax = 0.0
    for cut in cutoffs:
        for libID, (vals, _sizes) in lib_hists_by_cut[cut].items():
            for e in lefts:
                ymax = max(ymax, vals.get(e, 0))
    ymax = (ymax or 1) * 1.15

    c = canvas.canvas(); bottom = None
    for ii, cut in enumerate(cutoffs):                 # low at bottom, high at top
        ypos = ii * (ph + vgap)
        is_top = (ii == len(cutoffs) - 1)
        if bottom is None:
            xax = graph.axis.linear(min=edges[0], max=edges[-1],
                                    title="Footprint size (%s, unmerged)" % unit)
        else:
            xax = graph.axis.linkedaxis(bottom.axes["x"], painter=None)
        gkw = dict(width=pw, height=ph, xpos=0, ypos=ypos, x=xax,
                   y=graph.axis.linear(min=0, max=ymax, parter=_nice(ymax),
                                       title=r"%s, P$_B>$%s" % (ylabel, cut)))
        if is_top:
            gkw["key"] = graph.key.key(pos="tr", hinside=0)
        g = graph.graphxy(**gkw)
        for i, libID in enumerate(libIDs):
            vals, sizes = lib_hists_by_cut[cut].get(libID, ({}, []))
            if is_top:
                med = np.median(sizes) if sizes else 0
                title = r"%s (med %.0f, n=%d)" % (libID.replace("_", r"\_"),
                                                  med, len(sizes))
            else:
                title = None
            g.plot(graph.data.points([(ctr, vals.get(e, 0))
                                      for ctr, e in zip(centers, lefts)],
                                     x=1, y=2, title=title),
                   [graph.style.line([_libcolor(i), style.linewidth.Thick])])
        c.insert(g)
        if bottom is None:
            bottom = g
    c.writePDFfile(str(pdf_path))


def plot_stringency_sweep(lib_hists, edges, pdf_path, bin_width=SWEEP_BIN):
    """
    lib_hists: {libraryID: {cutoff: freq_dict}} where freq_dict is already a
    per-gene-then-aggregated frequency (see _gene_weighted_freq /
    stringency_sweep_hist) -- every gene voted once regardless of read depth.
    Vertical array of panels, one per cutoff (low at bottom, high at top),
    linked x-axes; within each panel one line per library (same color across
    panels). Key on the top panel, outside.
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
            freq = lib_hists[libID].get(cut, {})
            for e in lefts:
                ymax = max(ymax, freq.get(e, 0))
    ymax = (ymax or 1) * 1.15
    ylab = "frequency"

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
            freq = lib_hists[libID].get(cut, {})
            title = libID.replace("_", r"\_") if is_top else None
            g.plot(graph.data.points([(ctr, freq.get(e, 0))
                                      for ctr, e in zip(centers, lefts)],
                                     x=1, y=2, title=title),
                   [graph.style.line([_libcolor(i), style.linewidth.Thick])])
        c.insert(g)
        if bottom is None:
            bottom = g
    c.writePDFfile(str(pdf_path))


def plot_gene_call_rates(rates_by_lib, pdf_path, cutoff=FIXED_CUTOFF):
    """
    Per-gene protected-site call rate (gene_call_rates: n_protected_sites /
    n_scored_sites, pooled within each gene), one box+strip per library.
    This is the MAGNITUDE counterpart to the size histograms above, which
    equal-weight every gene on purpose and so wash out exactly the kind of
    library-to-library difference (e.g. ribosome-containing vs. a
    ribosome-less control) this plot is meant to show. Matplotlib rather
    than PyX -- a quick diagnostic, not a per-gene report figure.

    rates_by_lib: {libraryID: {gene: rate}}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    libIDs = sorted(rates_by_lib)
    data = [list(rates_by_lib[lib].values()) for lib in libIDs]

    fig, ax = plt.subplots(figsize=(1.2 * max(len(libIDs), 3) + 1, 5))
    ax.boxplot(data, labels=libIDs, showfliers=False)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        jitter = rng.normal(0, 0.05, size=len(vals))
        ax.scatter([i + j for j in jitter], vals, s=10, alpha=0.5, color="tab:blue")
        ax.text(i, ax.get_ylim()[0], f"n={len(vals)}", ha="center", va="top",
                fontsize=8)
    ax.set_ylabel(f"per-gene call rate (P$_B$ > {cutoff})")
    ax.set_title("Protected-site rate by library\n(one point per gene, "
                  f"genes with <{MIN_SITES_PER_GENE} scored sites dropped)",
                  fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(pdf_path, dpi=150)
    plt.close(fig)
    print(f"Wrote gene call-rate plot to {pdf_path}", file=sys.stderr)


def plot_his_codon_shadow_sizes(lib_hist, edges, pdf_path,
                                measure=SIZE_MEASURE, bin_width=SIZE_BIN,
                                cutoff=FIXED_CUTOFF):
    """
    Single-panel footprint-size frequency for shadow runs overlapping a His
    codon (P_B > cutoff AND run spans >=1 His codon position -- see
    extract_his_shadow_runs), one line per library. Only one cutoff is in
    play here (the rule is fixed, not swept), so unlike plot_footprint_sizes
    this doesn't need a per-cutoff panel stack.

    lib_hist: {libraryID: (freq_dict, sizes_list)} -- freq_dict already
    per-gene-then-aggregated (see _gene_weighted_freq), same equal-weighting
    rationale as the main size view.
    """
    if canvas is None:
        print("pyx not available; skipping plot", file=sys.stderr); return
    libIDs  = sorted(lib_hist)
    lefts   = edges[:-1]
    centers = [e + bin_width / 2. for e in lefts]
    unit    = {"genomic_nt": "nt", "n_sites": "Ref=A sites",
               "aligned_len": "aligned nt"}[measure]

    ymax = 0.0
    for freq, _sizes in lib_hist.values():
        for e in lefts:
            ymax = max(ymax, freq.get(e, 0))
    ymax = (ymax or 1) * 1.15

    g = graph.graphxy(
        width=10, height=6, xpos=0, ypos=0,
        x=graph.axis.linear(min=edges[0], max=edges[-1],
                            title="Footprint size at His codon (%s)" % unit),
        y=graph.axis.linear(min=0, max=ymax, parter=_nice(ymax),
                            title=r"freq, P$_B>$%s" % cutoff),
        key=graph.key.key(pos="tr", hinside=0))
    for i, libID in enumerate(libIDs):
        freq, sizes = lib_hist.get(libID, ({}, []))
        med = np.median(sizes) if sizes else 0
        title = r"%s (med %.0f, n=%d)" % (libID.replace("_", r"\_"),
                                          med, len(sizes))
        g.plot(graph.data.points([(ctr, freq.get(e, 0))
                                  for ctr, e in zip(centers, lefts)],
                                 x=1, y=2, title=title),
               [graph.style.line([_libcolor(i), style.linewidth.Thick])])
    c = canvas.canvas()
    c.insert(g)
    c.writePDFfile(str(pdf_path))
    print(f"Wrote His-codon shadow size plot to {pdf_path}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────
def _size_hist(df, cutoff):
    """
    Footprint-size histogram for one library at one cutoff (unmerged runs).
    The returned "counts" dict is actually a per-gene-then-aggregated
    frequency (see _gene_weighted_freq) -- every gene contributes one
    equally-weighted vote to the shape, regardless of its read depth.
    """
    runs = extract_shadow_runs(df, cutoff, N_PAD)
    sizes = [r[SIZE_MEASURE] for r in runs]
    freq, edges, n_used, n_dropped = _gene_weighted_freq(
        runs, SIZE_MEASURE, SIZE_RANGE[0], SIZE_RANGE[1], SIZE_BIN)
    print("    cutoff %.2f: %d runs, %d genes used, %d dropped (<%d runs)"
          % (cutoff, len(runs), n_used, n_dropped, MIN_RUNS_PER_GENE), file=sys.stderr)
    return freq, edges, sizes


def analyze_library(parquetFile, libraryID, his_gpos_by_gene):
    """Return per-cutoff size hists + the sweep + per-gene call rates + the
    His-codon-overlapping size hist, for one lib."""
    print("Loading %s (%s)..." % (parquetFile, libraryID), file=sys.stderr)
    df = pd.read_parquet(parquetFile)          # native columns, no JSON decode

    unmerged_by_cut = {}
    unmerged_counts_by_cut = {}  # cutoff -> (raw pooled counts, sizes) -- simple supplement
    edges = None
    for cut in PROB_CUTOFFS:
        uc, edges, us = _size_hist(df, cut)
        unmerged_by_cut[cut] = (uc, us)
        raw_counts, _ = _bin(us, SIZE_RANGE[0], SIZE_RANGE[1], SIZE_BIN)
        unmerged_counts_by_cut[cut] = (raw_counts, us)
    # console summary at the fixed cutoff
    if FIXED_CUTOFF in unmerged_by_cut:
        us = unmerged_by_cut[FIXED_CUTOFF][1]
        if us:
            print("    P_B>%s: n=%d median=%.0f IQR=[%.0f,%.0f]"
                  % (FIXED_CUTOFF, len(us), np.median(us),
                     np.percentile(us, 25), np.percentile(us, 75)), file=sys.stderr)

    sweep_hist, sweep_edges = stringency_sweep_hist(df)
    gene_rates = gene_call_rates(df, FIXED_CUTOFF)
    print("    P_B>%s call rate: %d genes with >=%d scored sites"
          % (FIXED_CUTOFF, len(gene_rates), MIN_SITES_PER_GENE), file=sys.stderr)

    his_runs = extract_his_shadow_runs(df, his_gpos_by_gene, cutoff=FIXED_CUTOFF, N=N_PAD)
    his_sizes = [r[SIZE_MEASURE] for r in his_runs]
    his_freq, his_edges, his_n_used, his_n_dropped = _gene_weighted_freq(
        his_runs, SIZE_MEASURE, SIZE_RANGE[0], SIZE_RANGE[1], SIZE_BIN)
    print("    P_B>%s His-codon-overlapping: %d runs, %d genes used, %d dropped (<%d runs)"
          % (FIXED_CUTOFF, len(his_runs), his_n_used, his_n_dropped,
             MIN_RUNS_PER_GENE), file=sys.stderr)

    return (unmerged_by_cut, unmerged_counts_by_cut, edges,
            sweep_hist, sweep_edges, gene_rates,
            (his_freq, his_sizes), his_edges)


def main(args):
    global PALETTE
    if canvas is not None:
        PALETTE = [color.cmyk(1, 0.5, 0, 0), color.cmyk(0, 1, 1, 0),
                   color.cmyk(0.4, 1, 0, 0), color.cmyk(1, 0, 1, 0.1),
                   color.cmyk(0, 0.5, 1, 0), color.cmyk(0.7, 0, 0, 0),
                   color.cmyk(0, 0, 0, 0.7), color.cmyk(0.3, 0, 1, 0.2)]

    parquetList, outPrefix, hisPicklePath = args[0], args[1], args[2]
    his_gpos_by_gene = load_his_codon_gpos(hisPicklePath)
    print("Loaded His codon positions for %d genes." % len(his_gpos_by_gene),
          file=sys.stderr)

    # {cutoff: {libID: (counts, sizes)}}
    unmerged_by_cut        = collections.defaultdict(dict)
    unmerged_counts_by_cut = collections.defaultdict(dict)
    sweep_by_lib    = {}
    rates_by_lib    = {}
    his_hist_by_lib = {}
    fp_edges = sweep_edges = his_edges = None

    with open(parquetList) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            fileName, rep, parquetFile = parts[0], parts[1], parts[2]
            libraryID = "%s-%s" % (fileName, rep)
            ubc, ucc, fpe, swh, swe, gr, his_h, hise = analyze_library(
                parquetFile, libraryID, his_gpos_by_gene)
            for cut in ubc:
                unmerged_by_cut[cut][libraryID]        = ubc[cut]
                unmerged_counts_by_cut[cut][libraryID] = ucc[cut]
            sweep_by_lib[libraryID]    = swh
            rates_by_lib[libraryID]    = gr
            his_hist_by_lib[libraryID] = his_h
            fp_edges, sweep_edges, his_edges = fpe, swe, hise

    if not sweep_by_lib:
        print("no libraries processed", file=sys.stderr); return

    print("Plotting combined figures across %d libraries..." % len(sweep_by_lib),
          file=sys.stderr)
    # View 1: footprint size, multi-panel over cutoffs (per-gene-then-aggregated freq)
    plot_footprint_sizes(dict(unmerged_by_cut), fp_edges,
                         "%s.footprint_sizes" % outPrefix)
    # View 1-supplement: same layout, raw pooled counts (no per-gene weighting)
    # -- simple sanity check against the per-gene-then-aggregated view above.
    plot_footprint_sizes(dict(unmerged_counts_by_cut), fp_edges,
                         "%s.footprint_sizes_counts" % outPrefix, ylabel="count")
    # # View 2: stringency sweep (aligned length)
    # plot_stringency_sweep(sweep_by_lib, sweep_edges,
    #                       "%s.stringency_sweep" % outPrefix)
    # View 3: per-gene call rate, magnitude comparison across libraries
    plot_gene_call_rates(rates_by_lib, "%s.gene_call_rates.png" % outPrefix)
    # View 4: footprint size restricted to runs overlapping a His codon
    plot_his_codon_shadow_sizes(his_hist_by_lib, his_edges,
                                "%s.his_codon_shadow_sizes" % outPrefix)


if __name__ == "__main__":
    main(sys.argv[1:])