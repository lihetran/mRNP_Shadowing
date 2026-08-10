"""
shadowCallSizeQC.py -- Liam Tran, August 2026

Every figure below is produced ONCE PER CONDITION (e.g. +3AT and -3AT
each get their own set of files, "outPrefix.<condition>.*.pdf" --
main() groups libraries by their own inFileParquet.txt condition column
before plotting), rather than one figure lumping every condition
together -- within a condition, that condition's reps are still
overlaid on the same figure so reps can be compared directly.

plot_calls_per_read ("outPrefix.<condition>.calls_per_read.pdf") -- one
line per library (within that condition), ALL overlaid on the same
read-length axis so reps can be compared directly (same PALETTE/legend
convention as polysomeShadowHMMQC.py's multi-library figures). x-axis: READ length
(nt, read_end-read_start), binned by READ_LEN_BIN_NT. Top panel: mean #
of qualifying shadow calls per read (P_B>=shadow_cutoff AND
genomic_nt>=min_run_nt via extract_shadow_runs -- same convention as
riboseqGeneCoverage.py/shadowMetagene.py, not every scored site), for
reads in that length bin -- EVERY read contributes, including ones with
zero qualifying calls (dropping those would inflate the mean). Bottom
panel: n_reads in that bin per library, the same signal+support pairing
used throughout this codebase's other metagene-style plots so a
thin-sample tail is visible rather than baked silently into an
unreliable mean.

plot_calls_per_read_cdf ("outPrefix.<condition>.calls_per_read_cdf.pdf")
-- the same per-read qualifying-call counts as above, but as a plain CDF
(one line per library, x-axis: # qualifying calls/read, y-axis:
cumulative fraction of reads) instead of binned-by-read-length means --
same sort-and-cumulative-fraction convention as shadowCDF.py's plot_CDF.
Reads cleaner than plot_calls_per_read for a quick "which rep carries
more shadow calls" comparison, with no read-length confound or per-bin
noise.

plot_call_ribo_metagene ("outPrefix.<condition>.call_ribo_metagene.pdf") -- for
every qualifying shadow call, ribo-seq read depth in a +/-window_nt
(default 100nt) window centered on that call's own genomic midpoint,
oriented 5'->3' by the call's gene's strand. Each call's local depth is
normalized by its GENE's own total ribo-seq depth (not the call's own
local-window total -- see build_call_ribo_metagene for why that
distinction matters) before being pooled -- same per-unit
fraction-then-average recipe as shadowMetagene.py's
build_his_density_metagene (there: per-GENE fraction of that gene's own
total depth, anchored on the nearest His codon; here: per-CALL local
depth as a fraction of that SAME call's gene's total depth, anchored on
the call's own midpoint), so every call is weighted equally (1)
regardless of how much ribo-seq coverage it happens to sit in, same
"pool per unit, then average units equally" rationale used throughout
this codebase. One line per library within that condition
(build_call_ribo_metagene) -- a condition with no ribo-seq BAM match
(e.g. "phenol" if riboBamList.txt only has +3AT/-3AT) gets no
call_ribo_metagene figure at all, but still gets its calls_per_read/
calls_per_read_cdf figures.

(plot_call_size_qc -- the earlier call-length-vs-ribo-seq design -- has
been REPLACED by plot_call_ribo_metagene above per your redirect; its
functions have been removed.)

Run:
  python3 shadowCallSizeQC.py inFileParquet.txt riboBamList.txt gtfFile outPrefix [shadow_cutoff] [min_run_nt] [read_len_bin_nt] [window_nt]
where inFileParquet.txt is a whitespace-delimited list, one library per
line: "condition  rep  shadow_calls.parquet" (the same convention as
riboSeqBam.txt / the training queue's own inFileParquet.txt --
libraryID = "condition-rep"; a bare-path line with no condition/rep
columns is also accepted, keyed by its own basename, but then has no
condition to match ribo-seq BAMs with and is skipped for
plot_call_ribo_metagene). Relative parquet paths are resolved against
inFileParquet.txt's own directory, so it can be run from anywhere.
riboBamList.txt is the usual "condition rep bamPath" list (e.g.
riboSeqBam.txt) -- each shadow library's OWN condition (from its own
inFileParquet.txt row) is used to pick its matching ribo-seq BAMs
(reps pooled), so a shadow condition with no match in riboBamList.txt
(e.g. "phenol" if riboBamList.txt only has +3AT/-3AT) is skipped for
plot_call_ribo_metagene but still included in the calls_per_read
figures (which need no ribo-seq data at all). gtfFile is the yeast GTF
(gene chrom/strand/span). outPrefix is the shared prefix for every
condition's output plots (see top of module for the per-condition
filename pattern), and shadow_cutoff/min_run_nt (optional) override the
P_B cutoff (default
0.5) and run-length floor (default 30nt) for what counts as a
qualifying shadow call -- applies to ALL analyses. read_len_bin_nt
(optional, default 25) is the calls_per_read read-length bin width, and
window_nt (optional, default 100) is plot_call_ribo_metagene's
+/-window around each call's own midpoint.
"""
import sys, os, collections
import pandas as pd

from polysomeShadowHMMQC import extract_shadow_runs
from riboseqGeneCoverage import (
    load_ribo_bam_list, ribo_coverage_track, SHADOW_CUTOFF, MIN_RUN_NT, TARGET_LENGTHS,
)
from runHMMPerGene import parse_gtf, compute_flank_caps

READ_LEN_BIN_NT  = 25    # calls-per-read analyses' read-length bin width
CALL_WINDOW_NT   = 100   # plot_call_ribo_metagene's +/-window around each call's own midpoint

PALETTE = [(1, 0.5, 0, 0), (0, 1, 1, 0), (0.4, 1, 0, 0), (1, 0, 1, 0.1),
          (0, 0.5, 1, 0), (0.7, 0, 0, 0), (0, 0, 0, 0.7), (0.3, 0, 1, 0.2)]


def _libcolor(i):
    from pyx import color
    return color.cmyk(*PALETTE[i % len(PALETTE)])


def load_shadow_parquet_list(path):
    """
    {libraryID: (parquetPath, condition)}, same "condition  rep  file"
    convention as inFileParquet.txt/riboSeqBam.txt (libraryID =
    "condition-rep"; a bare-path line with no condition/rep columns is
    keyed by its own basename with condition=None, mirroring
    load_ribo_bam_list's fallback for the older flat-list convention --
    condition=None means main() can't match it to a riboBamList.txt
    condition later). Relative parquetPaths are resolved against this
    list file's own directory (that's how the real inFileParquet.txt
    files on disk store them -- bare filenames, meant to be run from
    their own directory) so this script can be invoked from anywhere.
    """
    base_dir = os.path.dirname(os.path.abspath(path))
    libs = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) >= 3:
                cond, rep, parquet_path = parts[0], parts[1], parts[2]
                lib_id = f"{cond}-{rep}"
            else:
                parquet_path, cond = parts[0], None
                lib_id = os.path.basename(parquet_path)
            if not os.path.isabs(parquet_path):
                parquet_path = os.path.join(base_dir, parquet_path)
            libs[lib_id] = (parquet_path, cond)
    return libs


def load_shadow_calls_with_read_span(shadow_parquet_path):
    """
    Everything extract_shadow_runs needs (read_id, shadow_gene,
    shadow_gpos, shadow_P_B, shadow_region, absolute_indices) PLUS
    read_start/read_end -- riboseqGeneCoverage.py's own load_shadow_calls_df
    doesn't include those (its callers never needed a read's own genomic
    span), but plot_calls_per_read's whole x-axis IS that span.
    """
    return pd.read_parquet(shadow_parquet_path,
                           columns=["read_id", "shadow_gene", "shadow_gpos", "shadow_P_B",
                                   "shadow_region", "absolute_indices",
                                   "read_start", "read_end"])


def gene_padded_ribo_depth(bam_paths, gene, flank_5p, flank_3p):
    """
    {gpos: depth} for one gene, ribo_coverage_track's own fetch bounds
    (gene_start/gene_end) padded by flank_5p (start-codon side)/flank_3p
    (stop-codon side) nt -- mirrors shadowMetagene.py main()'s inline
    padded_gene pattern, EXCEPT flank_5p/flank_3p must already be capped
    per-gene by compute_flank_caps (runHMMPerGene.py), not a flat
    window_nt: this GTF packs genes close enough that ~7.4% have a real
    neighbor closer than 100nt on at least one side (checked directly),
    so an uncapped pad regularly fetches straight into a same-strand
    neighboring gene's own CDS -- ribo_coverage_track's sense filter only
    checks orientation, not which gene a read "belongs to", so that
    neighbor's coverage would get silently counted as this gene's own
    signal right at the far edge of the window (this is exactly what was
    producing the curling-upward-at-the-edges artifact). Fetched ONCE per
    gene (build_call_ribo_metagene's caller), not once per call.
    """
    # flank_5p/flank_3p are mRNA-sense-oriented (like compute_flank_caps'
    # own output) -- map back onto genomic left/right (gene_start/
    # gene_end) by strand, same convention as runHMMPerGene.py's
    # _padded_cds_segments (a minus-strand gene's 5' side is its
    # genomically-HIGHER/right side).
    left_pad, right_pad = (flank_5p, flank_3p) if gene["strand"] == "+" else (flank_3p, flank_5p)
    padded_gene = dict(gene)
    padded_gene["gene_start"] = max(0, gene["gene_start"] - left_pad)
    padded_gene["gene_end"]   = gene["gene_end"] + right_pad
    return ribo_coverage_track(bam_paths, padded_gene, target_lengths=TARGET_LENGTHS)


def call_relpos_depth(run, gene, ribo_gpos_depth, window_nt=CALL_WINDOW_NT):
    """
    {relpos: depth} for one shadow call -- ribo_gpos_depth (one gene's
    {gpos: depth}, from gene_padded_ribo_depth) resliced around this
    call's own genomic midpoint (integer-rounded down), relpos signed
    5'->3' by the gene's strand (a minus-strand gene's downstream/3'
    direction is genomically DEcreasing, so relpos there is
    center-gpos rather than gpos-center). Only positions with nonzero
    depth are included (a sparse dict, like ribo_gpos_depth itself) --
    zero-depth positions are dropped rather than stored as 0, same
    "only where there's actually signal" convention as
    shadowMetagene.py's build_his_density_metagene (see
    build_call_ribo_metagene for why that matters for the n_calls
    support panel).
    """
    lo, hi = run["gpos_lo"], run["gpos_hi"]
    center = (lo + hi) // 2
    sign = -1 if gene["strand"] == "-" else 1
    out = {}
    for offset in range(-window_nt, window_nt + 1):
        n = ribo_gpos_depth.get(center + sign * offset)
        if n:
            out[offset] = n
    return out


def _n_calls_per_read(shadow_df, shadow_cutoff=SHADOW_CUTOFF, min_run_nt=MIN_RUN_NT):
    """
    {read_id: n_qualifying_calls} -- qualifying = extract_shadow_runs'
    P_B>=shadow_cutoff AND genomic_nt>=min_run_nt, same convention as
    build_length_stats/riboseqGeneCoverage.py/shadowMetagene.py. Reads
    with zero qualifying calls are simply absent (callers use .get(id, 0)
    so they still count, at 0 -- excluding them outright would bias any
    per-read mean/CDF upward).
    """
    n_calls_by_read = collections.Counter()
    for r in extract_shadow_runs(shadow_df, shadow_cutoff):
        if r["genomic_nt"] >= min_run_nt:
            n_calls_by_read[r["read_id"]] += 1
    return n_calls_by_read


def build_calls_per_read_stats(shadow_df, shadow_cutoff=SHADOW_CUTOFF, min_run_nt=MIN_RUN_NT,
                               bin_width=READ_LEN_BIN_NT):
    """
    {bin_start: [n_reads, sum_calls]} -- EVERY read in shadow_df
    contributes (including reads with zero qualifying calls -- excluding
    them would inflate the mean upward), binned by its own genomic
    alignment span (read_end-read_start) in bin_width-nt bins. sum_calls
    is that read's count of qualifying shadow-call runs (see
    _n_calls_per_read).
    """
    n_calls_by_read = _n_calls_per_read(shadow_df, shadow_cutoff, min_run_nt)

    reads = shadow_df.drop_duplicates("read_id")[["read_id", "read_start", "read_end"]]
    stats = collections.defaultdict(lambda: [0, 0])
    for read_id, read_start, read_end in reads.itertuples(index=False):
        length = read_end - read_start
        b = (length // bin_width) * bin_width
        entry = stats[b]
        entry[0] += 1
        entry[1] += n_calls_by_read.get(read_id, 0)
    return stats


def build_calls_per_read_counts(shadow_df, shadow_cutoff=SHADOW_CUTOFF, min_run_nt=MIN_RUN_NT):
    """
    [n_qualifying_calls, ...] -- one entry per read in shadow_df
    (including 0 for reads with no qualifying call), for the CDF in
    plot_calls_per_read_cdf. Same qualifying-call definition as
    build_calls_per_read_stats, just not binned by read length -- the CDF
    is over the raw per-read count directly.
    """
    n_calls_by_read = _n_calls_per_read(shadow_df, shadow_cutoff, min_run_nt)
    read_ids = shadow_df["read_id"].drop_duplicates()
    return [n_calls_by_read.get(rid, 0) for rid in read_ids]


def plot_calls_per_read(stats_by_lib, pdf_path, bin_width=READ_LEN_BIN_NT, min_run_nt=MIN_RUN_NT,
                        shadow_cutoff=SHADOW_CUTOFF):
    """
    Two stacked panels sharing one binned read-length (nt) x-axis, same
    layout as shadowMetagene.py's plot_his_metagene: bottom = n_reads
    (support), top = mean # qualifying shadow calls per read at that
    read length (signal) -- ONE LINE PER LIBRARY in both panels, same
    PALETTE/legend convention as polysomeShadowHMMQC.py's multi-library
    figures, so libraries can be compared directly rather than eyeballed
    across separate files. stats_by_lib: {libraryID: build_calls_per_read_stats
    output}.
    """
    from pyx import canvas, graph, color, style, text as pyx_text

    libIDs = sorted(stats_by_lib)
    all_bins = sorted(set(b for stats in stats_by_lib.values() for b in stats))
    lo, hi = all_bins[0], all_bins[-1] + bin_width

    series = {}
    max_n_reads, max_mean = 1, 1.0
    for libID in libIDs:
        stats = stats_by_lib[libID]
        bins = sorted(stats.keys())
        xs = [b + bin_width / 2.0 for b in bins]
        n_reads = [stats[b][0] for b in bins]
        means = [(stats[b][1] / stats[b][0]) if stats[b][0] else None for b in bins]
        series[libID] = (xs, n_reads, means)
        max_n_reads = max(max_n_reads, max(n_reads, default=1))
        max_mean = max([max_mean] + [m for m in means if m is not None])

    c = canvas.canvas()
    panel_w, sig_h, dep_h, gap = 12, 3.5, 2.5, 0.8

    g_dep = graph.graphxy(
        width=panel_w, height=dep_h, xpos=0, ypos=0,
        x=graph.axis.linear(min=lo, max=hi, title="read length (nt)"),
        y=graph.axis.linear(min=0, max=max_n_reads * 1.05, title="n reads"))
    c.insert(g_dep)
    for i, libID in enumerate(libIDs):
        xs, n_reads, _means = series[libID]
        g_dep.plot(graph.data.points(list(zip(xs, n_reads)), x=1, y=2, title=None),
                  [graph.style.line([_libcolor(i), style.linewidth.Thick])])

    sig_ypos = dep_h + gap
    g_sig = graph.graphxy(
        width=panel_w, height=sig_h, xpos=0, ypos=sig_ypos,
        x=graph.axis.linkedaxis(g_dep.axes["x"], painter=graph.axis.painter.linked()),
        y=graph.axis.linear(min=0, max=max_mean * 1.1, title="mean shadow calls / read"),
        key=graph.key.key(pos="tr", hinside=0))
    c.insert(g_sig)
    for i, libID in enumerate(libIDs):
        xs, _n_reads, means = series[libID]
        pts = [(x, m) for x, m in zip(xs, means) if m is not None]
        if not pts:
            continue
        title = libID.replace("_", r"\_")
        g_sig.plot(graph.data.points(pts, x=1, y=2, title=title),
                  [graph.style.line([_libcolor(i), style.linewidth.Thick])])

    top_ypos = sig_ypos + sig_h
    n_reads_total = sum(sum(n_reads) for _xs, n_reads, _m in series.values())
    c.text(panel_w / 2., top_ypos + 0.5,
          "Read length vs. shadow-call count",
          [pyx_text.halign.center, pyx_text.size.large])
    c.text(panel_w / 2., top_ypos + 0.15,
          f"P$_B>${shadow_cutoff}, len$>${min_run_nt}nt -- {len(libIDs)} library(ies), "
          f"{n_reads_total} read(s) total -- {bin_width}nt read-length bins",
          [pyx_text.halign.center, pyx_text.size.scriptsize])

    c.writePDFfile(str(pdf_path))
    print(f"Wrote {pdf_path}", file=sys.stderr)


def plot_calls_per_read_cdf(counts_by_lib, pdf_path, min_run_nt=MIN_RUN_NT,
                            shadow_cutoff=SHADOW_CUTOFF):
    """
    CDF of # qualifying shadow calls per read, one line per library --
    same sort-and-cumulative-fraction convention as shadowCDF.py's
    plot_CDF, same PALETTE/legend convention as plot_calls_per_read
    above. Reads the noisy read-length confound out of the picture
    entirely: this is the plain distribution of build_calls_per_read_counts'
    per-read counts, so "does one library's reads carry more shadow calls
    than another's" reads directly off which curve sits further right/
    lower, no binning-noise or read-length-axis needed.
    """
    from pyx import canvas, graph, color, style, text as pyx_text

    libIDs = sorted(counts_by_lib)
    x_max = max((max(counts, default=0) for counts in counts_by_lib.values()), default=1)
    x_max = max(x_max, 1)

    c = canvas.canvas()
    panel_w, panel_h = 10, 6
    g = graph.graphxy(
        width=panel_w, height=panel_h, xpos=0, ypos=0,
        x=graph.axis.linear(min=0, max=x_max, title="qualifying shadow calls / read"),
        y=graph.axis.linear(min=0, max=1, title="cumulative fraction of reads"),
        key=graph.key.key(pos="br", hinside=0))

    n_reads_total = 0
    for i, libID in enumerate(libIDs):
        counts = sorted(counts_by_lib[libID])
        n = len(counts)
        n_reads_total += n
        if n == 0:
            continue
        cdf = [(k + 1) / n for k in range(n)]
        pts = list(zip(counts, cdf))
        if pts[-1][0] < x_max:
            pts.append((x_max, pts[-1][1]))
        title = r"%s (n=%d)" % (libID.replace("_", r"\_"), n)
        g.plot(graph.data.points(pts, x=1, y=2, title=title),
              [graph.style.line([_libcolor(i), style.linewidth.Thick])])

    c.insert(g)
    c.text(panel_w / 2., panel_h + 0.5,
          "Shadow calls per read (CDF)",
          [pyx_text.halign.center, pyx_text.size.large])
    c.text(panel_w / 2., panel_h + 0.15,
          f"P$_B>${shadow_cutoff}, len$>${min_run_nt}nt -- {len(libIDs)} library(ies), "
          f"{n_reads_total} read(s) total",
          [pyx_text.halign.center, pyx_text.size.scriptsize])

    c.writePDFfile(str(pdf_path))
    print(f"Wrote {pdf_path}", file=sys.stderr)


def build_call_ribo_metagene(shadow_df, genes, bam_paths, shadow_cutoff=SHADOW_CUTOFF,
                             min_run_nt=MIN_RUN_NT, window_nt=CALL_WINDOW_NT):
    """
    {relpos: [sum_frac, n_calls]} -- per-call-normalized ribo-seq depth
    around every qualifying shadow call's own genomic midpoint
    (extract_shadow_runs, P_B>=shadow_cutoff, genomic_nt>=min_run_nt).
    Same per-unit fraction-then-average recipe as shadowMetagene.py's
    build_his_density_metagene: there, each GENE's local depth (near a
    His codon) is divided by that gene's OWN TOTAL depth (over its whole
    scored region, not just near the His codon) before pooling; here,
    each CALL's local depth (call_relpos_depth) is divided by that call's
    GENE's own total depth (gene_padded_ribo_depth's full per-gene track,
    not just this one call's own +/-window_nt slice of it) before
    pooling -- same denominator, shared by every call in that gene, same
    as every His codon in a gene shares that gene's one total. This
    matters: normalizing by the call's OWN local-window total instead
    (tried first) makes every call's own window sum to 1 by construction,
    which (a) can only ever show the window's internal shape, never
    whether coverage near a call is enriched/depleted relative to its
    gene's baseline, and (b) inflates the fractional weight of any call
    with only a handful of nearby reads (small total -> huge per-read
    fraction) -- exactly the mechanism behind the upward curl you saw at
    the window edges, since a shadow call's own depleted center pushes a
    sparse call's few reads out toward the flanks, where the deflated
    per-call denominator then overweights them. Genes with zero total
    depth don't contribute at all (nothing to normalize by), same as a
    zero-depth gene in the precedent. ribo_coverage_track's own BAM fetch
    happens ONCE per gene (gene_padded_ribo_depth), not once per call, so
    a gene with many calls doesn't pay for redundant I/O. That fetch
    window is capped per-gene at the distance to its nearest neighbor
    (compute_flank_caps, runHMMPerGene.py) rather than a flat window_nt,
    so a gene close to a same-strand neighbor doesn't have that
    neighbor's own CDS coverage silently counted as its own signal at
    the far edge of the window.
    """
    flank_caps = compute_flank_caps(genes, window_nt)

    runs_by_gene = collections.defaultdict(list)
    for r in extract_shadow_runs(shadow_df, shadow_cutoff):
        if r["genomic_nt"] >= min_run_nt:
            runs_by_gene[r["gene"]].append(r)

    acc = collections.defaultdict(lambda: [0.0, 0])
    for gname, runs in runs_by_gene.items():
        gene = genes.get(gname)
        if gene is None:
            continue
        flank_5p, flank_3p = flank_caps.get(gname, (window_nt, window_nt))
        ribo_gpos_depth = gene_padded_ribo_depth(bam_paths, gene, flank_5p, flank_3p)
        total = sum(ribo_gpos_depth.values())
        if not total:
            continue
        for r in runs:
            relpos_depth = call_relpos_depth(r, gene, ribo_gpos_depth, window_nt)
            for relpos, n in relpos_depth.items():
                entry = acc[relpos]
                entry[0] += n / total
                entry[1] += 1
    return acc


def plot_call_ribo_metagene(acc_by_lib, pdf_path, window_nt=CALL_WINDOW_NT, min_run_nt=MIN_RUN_NT,
                            shadow_cutoff=SHADOW_CUTOFF):
    """
    Two stacked panels sharing one relative-position (nt, signed 5'->3'
    distance from each call's own midpoint) x-axis: bottom = n_calls
    (support -- varies by position, since a call only contributes where
    it actually had nonzero ribo-seq depth, see build_call_ribo_metagene/
    call_relpos_depth), top = mean per-call-normalized ribo-seq depth
    fraction (signal). ONE LINE PER LIBRARY in both panels, same
    PALETTE/legend convention as plot_calls_per_read. acc_by_lib:
    {libraryID: build_call_ribo_metagene output}.
    """
    from pyx import canvas, graph, color, style, text as pyx_text

    libIDs = sorted(acc_by_lib)
    xs = list(range(-window_nt, window_nt + 1))

    series = {}
    max_n_calls, max_mean = 1, 1e-9
    for libID in libIDs:
        acc = acc_by_lib[libID]
        n_calls = [acc.get(x, (0.0, 0))[1] for x in xs]
        means = [(acc[x][0] / acc[x][1]) if acc.get(x, (0.0, 0))[1] else None for x in xs]
        series[libID] = (n_calls, means)
        max_n_calls = max(max_n_calls, max(n_calls, default=1))
        max_mean = max([max_mean] + [m for m in means if m is not None])

    c = canvas.canvas()
    panel_w, sig_h, dep_h, gap = 12, 3.5, 2.5, 0.8

    g_dep = graph.graphxy(
        width=panel_w, height=dep_h, xpos=0, ypos=0,
        x=graph.axis.linear(min=xs[0], max=xs[-1],
                            title="distance from shadow-call center (nt, 5'$\\to$3')"),
        y=graph.axis.linear(min=0, max=max_n_calls * 1.05, title="n calls"))
    c.insert(g_dep)
    for i, libID in enumerate(libIDs):
        n_calls, _means = series[libID]
        g_dep.plot(graph.data.points(list(zip(xs, n_calls)), x=1, y=2, title=None),
                  [graph.style.line([_libcolor(i), style.linewidth.Thick])])

    sig_ypos = dep_h + gap
    g_sig = graph.graphxy(
        width=panel_w, height=sig_h, xpos=0, ypos=sig_ypos,
        x=graph.axis.linkedaxis(g_dep.axes["x"], painter=graph.axis.painter.linked()),
        y=graph.axis.linear(min=0, max=max_mean * 1.1, title="mean ribo-seq depth fraction / call"),
        key=graph.key.key(pos="tr", hinside=0))
    c.insert(g_sig)
    for i, libID in enumerate(libIDs):
        _n_calls, means = series[libID]
        pts = [(x, m) for x, m in zip(xs, means) if m is not None]
        if not pts:
            continue
        title = libID.replace("_", r"\_")
        g_sig.plot(graph.data.points(pts, x=1, y=2, title=title),
                  [graph.style.line([_libcolor(i), style.linewidth.Thick])])

    top_ypos = sig_ypos + sig_h
    c.text(panel_w / 2., top_ypos + 0.5,
          "Ribo-seq coverage around shadow calls",
          [pyx_text.halign.center, pyx_text.size.large])
    c.text(panel_w / 2., top_ypos + 0.15,
          f"P$_B>${shadow_cutoff}, len$>${min_run_nt}nt -- $\\pm${window_nt}nt window, "
          f"{len(libIDs)} library(ies) -- each call's depth normalized to its gene's total before averaging",
          [pyx_text.halign.center, pyx_text.size.scriptsize])

    c.writePDFfile(str(pdf_path))
    print(f"Wrote {pdf_path}", file=sys.stderr)


def main(args):
    inFileParquetPath, riboBamListPath, gtfPath, outPrefix = args[:4]
    shadow_cutoff   = float(args[4]) if len(args) > 4 else SHADOW_CUTOFF
    min_run_nt      = int(args[5])   if len(args) > 5 else MIN_RUN_NT
    read_len_bin_nt = int(args[6])   if len(args) > 6 else READ_LEN_BIN_NT
    window_nt       = int(args[7])   if len(args) > 7 else CALL_WINDOW_NT

    out_dir = os.path.dirname(outPrefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    libs = load_shadow_parquet_list(inFileParquetPath)
    print(f"{len(libs)} librar(ies) in {inFileParquetPath}: {', '.join(sorted(libs))}",
          file=sys.stderr)

    genes = parse_gtf(gtfPath)

    read_stats_by_lib = {}
    counts_by_lib = {}
    ribo_acc_by_lib = {}
    cond_by_lib = {}
    for libID, (parquetPath, cond) in libs.items():
        # bare-path lines (cond=None) have no condition to group by --
        # each is its own singleton group, keyed by its own libID.
        cond_by_lib[libID] = cond if cond is not None else libID

        print(f"Loading {libID}: {parquetPath} ...", file=sys.stderr)
        shadow_df = load_shadow_calls_with_read_span(parquetPath)
        print(f"  {len(shadow_df)} reads, {shadow_df['shadow_gene'].nunique()} genes.",
              file=sys.stderr)
        read_stats_by_lib[libID] = build_calls_per_read_stats(
            shadow_df, shadow_cutoff, min_run_nt, read_len_bin_nt)
        counts_by_lib[libID] = build_calls_per_read_counts(shadow_df, shadow_cutoff, min_run_nt)

        if cond is None:
            print(f"  {libID}: no condition column in {inFileParquetPath}, can't match "
                  f"ribo-seq BAMs -- skipping call_ribo_metagene for this library", file=sys.stderr)
            continue
        bam_paths = load_ribo_bam_list(riboBamListPath, cond)
        if not bam_paths:
            print(f"  {libID}: no ribo-seq BAM(s) match condition '{cond}' in "
                  f"{riboBamListPath} -- skipping call_ribo_metagene for this library",
                  file=sys.stderr)
            continue
        print(f"  {libID}: {len(bam_paths)} ribo-seq BAM(s) (condition={cond}); building "
              f"call-centered ribo metagene ...", file=sys.stderr)
        ribo_acc_by_lib[libID] = build_call_ribo_metagene(
            shadow_df, genes, bam_paths, shadow_cutoff, min_run_nt, window_nt)

    # One figure PER CONDITION (e.g. +3AT and -3AT split apart, each with
    # its own reps overlaid) rather than one figure with every condition
    # lumped together -- libs_by_cond groups libIDs by cond_by_lib.
    libs_by_cond = collections.defaultdict(list)
    for libID, cond in cond_by_lib.items():
        libs_by_cond[cond].append(libID)

    for cond in sorted(libs_by_cond):
        libIDs = libs_by_cond[cond]
        cond_read_stats = {lid: read_stats_by_lib[lid] for lid in libIDs}
        cond_counts     = {lid: counts_by_lib[lid] for lid in libIDs}
        cond_ribo_acc   = {lid: ribo_acc_by_lib[lid] for lid in libIDs if lid in ribo_acc_by_lib}

        plot_calls_per_read(cond_read_stats, f"{outPrefix}.{cond}.calls_per_read.pdf",
                           read_len_bin_nt, min_run_nt, shadow_cutoff)
        plot_calls_per_read_cdf(cond_counts, f"{outPrefix}.{cond}.calls_per_read_cdf.pdf",
                               min_run_nt, shadow_cutoff)

        if cond_ribo_acc:
            plot_call_ribo_metagene(cond_ribo_acc, f"{outPrefix}.{cond}.call_ribo_metagene.pdf",
                                   window_nt, min_run_nt, shadow_cutoff)
        else:
            print(f"  {cond}: no library had matching ribo-seq BAMs; skipping "
                  f"call_ribo_metagene plot for this condition.", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
