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
inFileParquet.txt row) is used to pick its matching ribo-seq BAMs
(reps pooled), so a shadow condition with no match in riboBamList.txt
(e.g. "phenol" if riboBamList.txt only has +3AT/-3AT) is skipped for
plot_call_ribo_metagene but still included in the calls_per_read
figures (which need no ribo-seq data at all). gtfFile is the yeast GTF
(gene chrom/strand/span). outPrefix is the shared prefix for every
condition's output plots (see top of module for the per-condition
filename pattern). colorMapPath is REQUIRED -- a manuscript color-map TSV
(name, rep, path, hex_color, no leading '#' -- the same convention/file
used by --color_map in the other scripts in this codebase) matched
against libraryID ("condition-rep") so a given library is the same color
here as in every other script's figures; libraries with no match fall
back to the built-in PALETTE cycle (with a warning), but the file itself
must be given every run. shadow_cutoff/min_run_nt (optional) override the
P_B cutoff (default 0.7) and run-length floor (default 25nt) for what
counts as a qualifying shadow call -- applies to ALL analyses.
read_len_bin_nt (optional, default 25) is the calls_per_read read-length
bin width, and window_nt (optional, default 100) is
plot_call_ribo_metagene's +/-window around each call's own midpoint.
split_strand (optional, default off -- pass "1"/"true"/"yes" to enable) is
a DIAGNOSTIC flag, not a normal-use one: in addition to the usual
call_ribo_metagene.pdf, also produces call_ribo_metagene.plus_strand.pdf/
.minus_strand.pdf built from ONLY plus-/minus-strand genes respectively
(see build_call_ribo_metagene's strand_filter) -- lets a plus-strand-only
comparison against another script's own metagene isolate whether an
observed 5'/3' pattern mismatch is a strand-sign bug (plus-strand genes
get no sign flip applied at all, so it can't be that if the mismatch
survives here) or something else (e.g. a real HSMM run-boundary
asymmetry). Triples the ribo-seq BAM I/O for the run, so leave it off
for normal use.
hisPicklePath (optional, default none) is findHisCodonPositions.py's
output pickle -- when given, ALSO produces
call_ribo_metagene.his_anchored_qc.pdf per condition: the same
per-call-normalized ribo-seq metagene, but anchored on each qualifying
call's nearest CONTAINED His codon instead of the call's own midpoint
(see build_call_his_ribo_metagene). This is a QC check, not a
replacement -- the primary call_ribo_metagene.pdf keeps the call-midpoint
anchor either way, since that's the more informative figure for the main
result (a call's own midpoint is a real measured boundary, not a fixed
sequence feature); the His-anchored version exists to check whether that
choice of anchor is smearing out a real feature (e.g. a queued/collided
ribosome sitting a roughly fixed distance from the true stall site,
which pools poorly against a call-midpoint anchor that itself jitters
relative to that true site -- confirmed on real data: qualifying calls
extend ~3nt further upstream of a contained His codon than downstream on
average, Wilcoxon p~3e-49 across ~5300 calls).
"""
import sys, os, collections
import pandas as pd

from polysomeShadowHMMQC import extract_shadow_runs, load_his_codon_gpos
from riboseqGeneCoverage import (
    load_ribo_bam_list, ribo_coverage_track, ribo_coverage_and_count_track,
    TARGET_LENGTHS,
)
from runHMMPerGene import parse_gtf, compute_flank_caps

# Hardcoded for this script specifically -- deliberately NOT
# riboseqGeneCoverage.py's own SHADOW_CUTOFF=0.5/MIN_RUN_NT=30 defaults
# (other scripts still get those; this module just no longer imports
# them), since every analysis in this file should use the same qualifying-
# call definition regardless of what CLI args a caller does or doesn't
# pass.
SHADOW_CUTOFF    = 0.7   # P_B cutoff for a qualifying shadow call
MIN_RUN_NT       = 25    # run length floor for a qualifying shadow call (len>25nt)
READ_LEN_BIN_NT  = 25    # calls-per-read analyses' read-length bin width
CALL_WINDOW_NT   = 100   # plot_call_ribo_metagene's +/-window around each call's own midpoint

PALETTE   = [(1, 0.5, 0, 0), (0, 1, 1, 0), (0.4, 1, 0, 0), (1, 0, 1, 0.1),
            (0, 0.5, 1, 0), (0.7, 0, 0, 0), (0, 0, 0, 0.7), (0.3, 0, 1, 0.2)]
COLOR_MAP = {}   # {libraryID: "#RRGGBB"}, set in main() if a colorMapPath was given --
                 # takes priority over PALETTE's index-based fallback so a library
                 # keeps the same color here as in every other script's figures.


def load_color_map(path: str) -> dict:
    """
    Parse a manuscript color-map TSV with columns:
        sample_name, rep, path, hex_color (no leading '#')
    Returns a dict keyed by "name_rep", "name-rep" (this script's own
    libraryID convention -- "condition-rep"), and bare "name" (first
    match wins for the bare key) mapping to "#RRGGBB".
    """
    color_map = {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 4:
                continue
            name, rep, _path, hexcol = fields[0], fields[1], fields[2], fields[3]
            hexcol = "#" + hexcol.strip().lstrip("#")
            if rep:
                color_map.setdefault(f"{name}_{rep}", hexcol)
                color_map.setdefault(f"{name}-{rep}", hexcol)
            color_map.setdefault(name, hexcol)
    return color_map


def hex_to_pyx_color(hexcol: str):
    from pyx import color
    hexcol = hexcol.lstrip("#")
    r = int(hexcol[0:2], 16) / 255.0
    g = int(hexcol[2:4], 16) / 255.0
    b = int(hexcol[4:6], 16) / 255.0
    return color.rgb(r, g, b)


def _libcolor(libID, i):
    """COLOR_MAP's manuscript hex color for libID if present, else the
    i'th color from the built-in PALETTE cycle."""
    from pyx import color
    hexcol = COLOR_MAP.get(libID)
    if hexcol:
        return hex_to_pyx_color(hexcol)
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


def _padded_gene_bounds(gene, flank_5p, flank_3p):
    """
    A copy of gene with gene_start/gene_end padded by flank_5p (start-codon
    side)/flank_3p (stop-codon side) nt -- flank_5p/flank_3p are
    mRNA-sense-oriented (like compute_flank_caps' own output) and get
    mapped back onto genomic left/right here by strand, same convention as
    runHMMPerGene.py's _padded_cds_segments (a minus-strand gene's 5' side
    is its genomically-HIGHER/right side). Shared by
    gene_padded_ribo_depth and gene_padded_ribo_depth_and_count so the two
    can't silently drift out of sync on how padding is applied.
    """
    left_pad, right_pad = (flank_5p, flank_3p) if gene["strand"] == "+" else (flank_3p, flank_5p)
    padded_gene = dict(gene)
    padded_gene["gene_start"] = max(0, gene["gene_start"] - left_pad)
    padded_gene["gene_end"]   = gene["gene_end"] + right_pad
    return padded_gene


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
    padded_gene = _padded_gene_bounds(gene, flank_5p, flank_3p)
    return ribo_coverage_track(bam_paths, padded_gene, target_lengths=TARGET_LENGTHS)


def gene_padded_ribo_depth_and_count(bam_paths, gene, flank_5p, flank_3p):
    """
    (depth, n_reads) for one gene, same padded fetch window as
    gene_padded_ribo_depth, but via ribo_coverage_and_count_track so the
    plain read COUNT comes back from the same BAM pass instead of a
    second, redundant fetch -- n_reads is what
    build_call_ribo_metagene/build_call_his_ribo_metagene now normalize
    each call's local depth by (see their own docstrings for why this
    replaced sum(depth.values()): that sum is inflated by footprint
    length, since a single 28nt read contributes 28 to it -- one per
    position it covers -- rather than counting as one read).
    """
    padded_gene = _padded_gene_bounds(gene, flank_5p, flank_3p)
    return ribo_coverage_and_count_track(bam_paths, padded_gene, target_lengths=TARGET_LENGTHS)


def call_relpos_depth(run, gene, ribo_gpos_depth, window_nt=CALL_WINDOW_NT, center=None):
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

    center: genomic anchor override -- defaults to the call's own
    genomic midpoint (as above). Pass an explicit gpos (e.g. a contained
    His codon) to reslice around a DIFFERENT anchor instead -- see
    build_call_his_ribo_metagene, a His-codon-anchored QC view that
    checks whether the main call_ribo_metagene figure's own choice of
    anchor (the call's own midpoint, kept as the primary figure on
    purpose -- see its docstring) is masking a real feature by
    miscentering it.
    """
    lo, hi = run["gpos_lo"], run["gpos_hi"]
    if center is None:
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
                  [graph.style.line([_libcolor(libID, i), style.linewidth.Thick])])

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
                  [graph.style.line([_libcolor(libID, i), style.linewidth.Thick])])

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
              [graph.style.line([_libcolor(libID, i), style.linewidth.Thick])])

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
                             min_run_nt=MIN_RUN_NT, window_nt=CALL_WINDOW_NT,
                             strand_filter=None):
    """
    {relpos: [sum_frac, n_calls]} -- per-call-normalized ribo-seq depth
    around every qualifying shadow call's own genomic midpoint
    (extract_shadow_runs, P_B>=shadow_cutoff, genomic_nt>=min_run_nt).
    Same per-unit fraction-then-average recipe as shadowMetagene.py's
    build_his_density_metagene: there, each GENE's local depth (near a
    His codon) is divided by that gene's OWN TOTAL depth before pooling;
    here, each CALL's local depth (call_relpos_depth) is divided by that
    call's GENE's own total ribo-seq READ COUNT (gene_padded_ribo_depth_and_count's
    n_reads, over the gene's whole padded window, not just this one call's
    own +/-window_nt slice of it) before pooling -- same denominator,
    shared by every call in that gene. Deliberately a plain READ COUNT,
    not sum(depth.values()) (tried first): that sum double-, triple-,
    etc.-counts a single read once per position it covers, so a 28nt read
    contributes 28 to it rather than 1 -- an inflated, footprint-length-
    dependent quantity that isn't really "how many reads did this gene
    get." A plain read count is the simpler, more standard normalization
    and doesn't carry that footprint-length dependence.

    Separately: normalizing by the call's OWN local-window total instead
    (tried even before that) makes every call's own window sum to 1 by
    construction, which (a) can only ever show the window's internal
    shape, never whether coverage near a call is enriched/depleted
    relative to its gene's baseline, and (b) inflates the fractional
    weight of any call with only a handful of nearby reads (small total ->
    huge per-read fraction) -- exactly the mechanism behind the upward
    curl you saw at the window edges, since a shadow call's own depleted
    center pushes a sparse call's few reads out toward the flanks, where
    the deflated per-call denominator then overweights them. Genes with
    zero total reads don't contribute at all (nothing to normalize by).
    ribo_coverage_and_count_track's own BAM fetch happens ONCE per gene
    (gene_padded_ribo_depth_and_count), not once per call and not twice
    per gene either (depth and read count come from the SAME pass), so a
    gene with many calls doesn't pay for redundant I/O. That fetch window
    is capped per-gene at the distance to its nearest neighbor
    (compute_flank_caps, runHMMPerGene.py) rather than a flat window_nt,
    so a gene close to a same-strand neighbor doesn't have that
    neighbor's own CDS coverage silently counted as its own signal at
    the far edge of the window.

    strand_filter: None (default) includes every gene, same as before this
    param existed. "+"/"-" restricts to genes on just that strand -- a
    DIAGNOSTIC knob, not a normal-use one: since call_relpos_depth's own
    sign flip only ever does anything for minus-strand genes (plus-strand
    genes get sign=+1, i.e. relpos IS the raw genomic difference, no flip
    applied at all), comparing a plus-strand-only metagene against another
    script's plus-strand-only metagene isolates whether an apparent 5'/3'
    mismatch between the two is a strand-sign bug (should vanish once
    minus-strand genes, the only ones any sign flip touches, are excluded)
    or something else entirely (e.g. a real HSMM run-boundary asymmetry,
    which would persist even with strand flipping taken out of the
    picture -- see the shadowCallSizeQC.py-vs-metaHistidineFromParquet.py
    discussion this was added for).

    Returns (acc, acc_gene_weighted) -- acc is the raw per-CALL pooling
    described above (one vote per call); acc_gene_weighted gives every
    GENE exactly one vote instead, regardless of how many calls it has:
    each gene's own calls are averaged into that gene's own {relpos: mean}
    profile first, and THOSE per-gene means are what gets pooled/averaged
    across genes. Same "equal weight per gene" principle as
    polysomeShadowHMMQC.py's _gene_weighted_freq. This matters because the
    per-call read-count normalization above only fixes MAGNITUDE (a
    highly-expressed gene's raw counts don't dominate) -- it does nothing
    about vote count: a gene with many densely-packed, heavily-OVERLAPPING
    calls (confirmed on real data -- e.g. one gene's His codons only 3-39nt
    apart, well under window_nt) resamples the SAME underlying ribo-seq
    reads once per nearby call, so it still gets many times the effective
    influence of a gene with few/no overlapping calls in the raw acc,
    even though each individual call's own fraction is correctly scaled.
    """
    flank_caps = compute_flank_caps(genes, window_nt)

    runs_by_gene = collections.defaultdict(list)
    for r in extract_shadow_runs(shadow_df, shadow_cutoff):
        if r["genomic_nt"] >= min_run_nt:
            runs_by_gene[r["gene"]].append(r)

    # Diagnostic counters -- every one of these being a silent `continue`
    # (no warning at all) is exactly what made an empty result
    # indistinguishable from a real-but-flat one before this existed: a
    # wrong/mismatched gtfFile (genes.get(gname) is None for every gene) or
    # a gene with zero ribo-seq depth in bam_paths both used to vanish with
    # no trace, leaving a plot that silently fell back to placeholder axis
    # ranges instead of erroring or warning.
    n_genes_no_gtf_match = 0
    n_genes_zero_ribo_depth = 0
    n_genes_wrong_strand = 0
    n_genes_contributing = 0

    acc = collections.defaultdict(lambda: [0.0, 0])
    acc_gene_weighted = collections.defaultdict(lambda: [0.0, 0])
    for gname, runs in runs_by_gene.items():
        gene = genes.get(gname)
        if gene is None:
            n_genes_no_gtf_match += 1
            continue
        if strand_filter is not None and gene["strand"] != strand_filter:
            n_genes_wrong_strand += 1
            continue
        flank_5p, flank_3p = flank_caps.get(gname, (window_nt, window_nt))
        ribo_gpos_depth, n_reads_total = gene_padded_ribo_depth_and_count(
            bam_paths, gene, flank_5p, flank_3p)
        if not n_reads_total:
            n_genes_zero_ribo_depth += 1
            continue
        n_genes_contributing += 1
        gene_acc = collections.defaultdict(lambda: [0.0, 0])
        for r in runs:
            relpos_depth = call_relpos_depth(r, gene, ribo_gpos_depth, window_nt)
            for relpos, n in relpos_depth.items():
                entry = acc[relpos]
                entry[0] += n / n_reads_total
                entry[1] += 1
                gentry = gene_acc[relpos]
                gentry[0] += n / n_reads_total
                gentry[1] += 1
        # Fold this gene's OWN mean profile into the gene-weighted
        # accumulator -- ONE contribution per gene per relpos, regardless
        # of how many of its own calls reached that relpos.
        for relpos, (gsum, gn) in gene_acc.items():
            gwentry = acc_gene_weighted[relpos]
            gwentry[0] += gsum / gn
            gwentry[1] += 1

    print(f"    build_call_ribo_metagene: {n_genes_contributing}/{len(runs_by_gene)} "
          f"gene(s) with qualifying shadow calls actually contributed "
          f"({n_genes_no_gtf_match} not found in the GTF passed in, "
          f"{n_genes_zero_ribo_depth} had zero ribo-seq depth"
          + (f", {n_genes_wrong_strand} excluded by strand_filter={strand_filter!r}"
             if strand_filter is not None else "") + ")", file=sys.stderr)
    if runs_by_gene and n_genes_contributing == 0:
        print("    WARNING: build_call_ribo_metagene got ZERO contributing genes -- "
              "the resulting metagene will be empty. If n_genes_no_gtf_match is "
              "high, double check the gtfFile argument matches the GTF the shadow "
              "calls' gene names actually came from (e.g. a wrong path, or a "
              "different annotation using systematic names instead of common "
              "gene names); if n_genes_zero_ribo_depth is high instead, check "
              "riboBamList.txt's condition column and bam paths.", file=sys.stderr)
    return acc, acc_gene_weighted


def build_call_his_ribo_metagene(shadow_df, genes, bam_paths, his_gpos_by_gene,
                                 shadow_cutoff=SHADOW_CUTOFF, min_run_nt=MIN_RUN_NT,
                                 window_nt=CALL_WINDOW_NT, strand_filter=None):
    """
    QC counterpart to build_call_ribo_metagene: same per-call, per-gene-
    normalized ribo-seq depth pooling, but anchored on the NEAREST His
    codon CONTAINED within each qualifying call (gpos_lo <= his <=
    gpos_hi), instead of the call's own geometric midpoint. Only calls
    that actually contain >=1 His codon are included here -- a call with
    none has nothing to anchor on and is silently excluded from THIS QC
    view (counted, see n_calls_no_his_codon below), NOT from the primary
    call_ribo_metagene figure, which is unaffected by this function and
    keeps every qualifying call regardless.

    Why this exists (a QC check, not a replacement for the main figure --
    the main figure's call-midpoint anchor is the more informative choice
    for the primary result, see its own docstring): real data shows
    shadow calls extend further upstream (5') of a contained His codon
    than downstream (3') on average (~3nt, Wilcoxon p~3e-49 across ~5300
    calls) -- i.e. the call's own midpoint is measurably NOT the same
    reference point as the presumed true stall site. Pooling by the
    call's own midpoint therefore smears any feature that's fixed
    relative to the TRUE stall site (e.g. a queued/collided ribosome
    sitting a roughly constant distance upstream of it) across a range of
    apparent relative positions instead of letting it show up as a sharp,
    resolvable bump. Re-centering here on the His codon itself removes
    that jitter, as a check for whether a real fixed-offset feature was
    being masked by it.

    Multiple His codons contained in one call (rare) are resolved by
    nearest to the call's OWN midpoint, just to have one consistent rule.

    Same read-COUNT normalization as build_call_ribo_metagene (not
    sum(depth.values()) -- see that function's docstring for why): each
    call's local depth is divided by its gene's plain ribo-seq read count
    (gene_padded_ribo_depth_and_count), not a footprint-length-inflated
    depth sum.

    Returns (acc, acc_gene_weighted) with the same shapes/meaning as
    build_call_ribo_metagene's own return (see that function's docstring
    for what "gene-weighted" fixes here that per-call read-count
    normalization alone doesn't) -- both directly plottable with
    plot_call_ribo_metagene.
    """
    flank_caps = compute_flank_caps(genes, window_nt)

    runs_by_gene = collections.defaultdict(list)
    for r in extract_shadow_runs(shadow_df, shadow_cutoff):
        if r["genomic_nt"] >= min_run_nt:
            runs_by_gene[r["gene"]].append(r)

    n_genes_no_gtf_match = 0
    n_genes_zero_ribo_depth = 0
    n_genes_wrong_strand = 0
    n_genes_no_his_codons = 0
    n_genes_contributing = 0
    n_calls_no_his_codon = 0

    acc = collections.defaultdict(lambda: [0.0, 0])
    acc_gene_weighted = collections.defaultdict(lambda: [0.0, 0])
    for gname, runs in runs_by_gene.items():
        gene = genes.get(gname)
        if gene is None:
            n_genes_no_gtf_match += 1
            continue
        if strand_filter is not None and gene["strand"] != strand_filter:
            n_genes_wrong_strand += 1
            continue
        his_set = his_gpos_by_gene.get(gname)
        if not his_set:
            n_genes_no_his_codons += 1
            continue
        flank_5p, flank_3p = flank_caps.get(gname, (window_nt, window_nt))
        ribo_gpos_depth, n_reads_total = gene_padded_ribo_depth_and_count(
            bam_paths, gene, flank_5p, flank_3p)
        if not n_reads_total:
            n_genes_zero_ribo_depth += 1
            continue
        n_genes_contributing += 1
        gene_acc = collections.defaultdict(lambda: [0.0, 0])
        for r in runs:
            lo, hi = r["gpos_lo"], r["gpos_hi"]
            contained = [h for h in his_set if lo <= h <= hi]
            if not contained:
                n_calls_no_his_codon += 1
                continue
            midpoint = (lo + hi) // 2
            anchor = min(contained, key=lambda h: abs(h - midpoint))
            relpos_depth = call_relpos_depth(r, gene, ribo_gpos_depth, window_nt, center=anchor)
            for relpos, n in relpos_depth.items():
                entry = acc[relpos]
                entry[0] += n / n_reads_total
                entry[1] += 1
                gentry = gene_acc[relpos]
                gentry[0] += n / n_reads_total
                gentry[1] += 1
        for relpos, (gsum, gn) in gene_acc.items():
            gwentry = acc_gene_weighted[relpos]
            gwentry[0] += gsum / gn
            gwentry[1] += 1

    print(f"    build_call_his_ribo_metagene: {n_genes_contributing}/{len(runs_by_gene)} "
          f"gene(s) contributed, {n_calls_no_his_codon} call(s) skipped for containing "
          f"no His codon ({n_genes_no_gtf_match} genes not found in the GTF passed in, "
          f"{n_genes_no_his_codons} genes with no His codons at all, "
          f"{n_genes_zero_ribo_depth} genes had zero ribo-seq depth"
          + (f", {n_genes_wrong_strand} genes excluded by strand_filter={strand_filter!r}"
             if strand_filter is not None else "") + ")", file=sys.stderr)
    return acc, acc_gene_weighted


def plot_call_ribo_metagene(acc_by_lib, pdf_path, window_nt=CALL_WINDOW_NT, min_run_nt=MIN_RUN_NT,
                            shadow_cutoff=SHADOW_CUTOFF, strand_label=None,
                            title="Ribo-seq coverage around shadow calls",
                            x_title="distance from shadow-call center (nt, 5'$\\to$3')"):
    """
    Two stacked panels sharing one relative-position (nt, signed 5'->3'
    distance from each call's own midpoint) x-axis: bottom = n_calls
    (support -- varies by position, since a call only contributes where
    it actually had nonzero ribo-seq depth, see build_call_ribo_metagene/
    call_relpos_depth), top = mean per-call-normalized ribo-seq depth
    fraction (signal). ONE LINE PER LIBRARY in both panels, same
    PALETTE/legend convention as plot_calls_per_read. acc_by_lib:
    {libraryID: build_call_ribo_metagene output}.

    strand_label: purely cosmetic -- if given (e.g. "+" or "-"), noted in
    the subtitle so a strand_filter-restricted diagnostic run (see
    build_call_ribo_metagene) is clearly distinguishable from the normal
    both-strands figure when the two PDFs sit side by side.

    title/x_title: override the default chart title / x-axis label --
    e.g. build_call_his_ribo_metagene's QC output isn't centered on "the
    shadow-call center" at all (it's centered on a contained His codon
    instead), so its caller passes different text here rather than
    mislabeling the axis.
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
        x=graph.axis.linear(min=xs[0], max=xs[-1], title=x_title),
        y=graph.axis.linear(min=0, max=max_n_calls * 1.05, title="n calls"))
    c.insert(g_dep)
    for i, libID in enumerate(libIDs):
        n_calls, _means = series[libID]
        g_dep.plot(graph.data.points(list(zip(xs, n_calls)), x=1, y=2, title=None),
                  [graph.style.line([_libcolor(libID, i), style.linewidth.Thick])])

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
        line_title = libID.replace("_", r"\_")
        g_sig.plot(graph.data.points(pts, x=1, y=2, title=line_title),
                  [graph.style.line([_libcolor(libID, i), style.linewidth.Thick])])

    top_ypos = sig_ypos + sig_h
    full_title = title
    if strand_label is not None:
        full_title += f" ({strand_label}-strand genes only)"
    c.text(panel_w / 2., top_ypos + 0.5, full_title,
          [pyx_text.halign.center, pyx_text.size.large])
    c.text(panel_w / 2., top_ypos + 0.15,
          f"P$_B>${shadow_cutoff}, len$>${min_run_nt}nt -- $\\pm${window_nt}nt window, "
          f"{len(libIDs)} library(ies) -- each call's depth normalized to its gene's total before averaging",
          [pyx_text.halign.center, pyx_text.size.scriptsize])

    c.writePDFfile(str(pdf_path))
    print(f"Wrote {pdf_path}", file=sys.stderr)


def main(args):
    global COLOR_MAP
    if len(args) < 5:
        print("Usage: python3 shadowCallSizeQC.py inFileParquet.txt riboBamList.txt "
              "gtfFile outPrefix colorMapPath [shadow_cutoff] [min_run_nt] "
              "[read_len_bin_nt] [window_nt]", file=sys.stderr)
        sys.exit(1)

    inFileParquetPath, riboBamListPath, gtfPath, outPrefix, colorMapPath = args[:5]
    shadow_cutoff   = float(args[5]) if len(args) > 5 else SHADOW_CUTOFF
    min_run_nt      = int(args[6])   if len(args) > 6 else MIN_RUN_NT
    read_len_bin_nt = int(args[7])   if len(args) > 7 else READ_LEN_BIN_NT
    window_nt       = int(args[8])   if len(args) > 8 else CALL_WINDOW_NT
    # DIAGNOSTIC ONLY (see build_call_ribo_metagene's strand_filter) -- not
    # meant to stick around as a normal-use option, just here to let a
    # plus-strand-only vs. minus-strand-only comparison isolate whether an
    # observed 5'/3' pattern mismatch against another script's metagene is
    # a strand-sign bug (plus-strand genes get no sign flip at all, so it
    # can't be that if the mismatch survives with only + genes included) or
    # something else (e.g. a real HSMM run-boundary asymmetry).
    split_strand    = len(args) > 9 and args[9].lower() in ("1", "true", "yes")
    # QC ONLY (see build_call_his_ribo_metagene) -- when given, produces an
    # ADDITIONAL call_ribo_metagene.his_anchored_qc.pdf per condition,
    # anchored on each qualifying call's nearest CONTAINED His codon
    # instead of the call's own midpoint. Does not change or replace the
    # primary call_ribo_metagene.pdf, which intentionally keeps the
    # call-midpoint anchor -- see that function's own docstring for why
    # it's the more informative figure.
    hisPicklePath   = args[10] if len(args) > 10 else None

    out_dir = os.path.dirname(outPrefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    libs = load_shadow_parquet_list(inFileParquetPath)
    print(f"{len(libs)} librar(ies) in {inFileParquetPath}: {', '.join(sorted(libs))}",
          file=sys.stderr)

    COLOR_MAP = load_color_map(colorMapPath)
    print(f"Loaded manuscript colors for {len(COLOR_MAP)} library key(s) from "
          f"{colorMapPath}.", file=sys.stderr)
    unmatched = [lib for lib in libs if lib not in COLOR_MAP]
    if unmatched:
        print(f"  WARNING: no color found in {colorMapPath} for "
              f"librar{'y' if len(unmatched) == 1 else 'ies'} {unmatched}; "
              f"falling back to the default palette.", file=sys.stderr)

    genes = parse_gtf(gtfPath)
    print(f"Parsed {len(genes)} gene(s) from {gtfPath}.", file=sys.stderr)
    if not genes:
        print(f"  WARNING: 0 genes parsed from {gtfPath} -- check this is really a "
              f"GTF file with CDS/exon features (every downstream gene lookup will "
              f"come back empty).", file=sys.stderr)

    his_gpos_by_gene = None
    if hisPicklePath:
        his_gpos_by_gene = load_his_codon_gpos(hisPicklePath)
        print(f"Loaded His codon positions for {len(his_gpos_by_gene)} gene(s) from "
              f"{hisPicklePath} -- will also produce a His-codon-anchored QC metagene "
              f"per condition.", file=sys.stderr)

    read_stats_by_lib = {}
    counts_by_lib = {}
    ribo_acc_by_lib = {}
    ribo_acc_gw_by_lib = {}
    ribo_acc_plus_by_lib = {}
    ribo_acc_minus_by_lib = {}
    ribo_his_acc_by_lib = {}
    ribo_his_acc_gw_by_lib = {}
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
        ribo_acc_by_lib[libID], ribo_acc_gw_by_lib[libID] = build_call_ribo_metagene(
            shadow_df, genes, bam_paths, shadow_cutoff, min_run_nt, window_nt)
        if split_strand:
            # Recomputes ribo-seq I/O for the same genes a second/third time
            # (once per strand) -- acceptable for a diagnostic-only run, not
            # worth optimizing for what's meant to be a temporary check.
            # Gene-weighting isn't relevant to this strand-sign diagnostic,
            # so only the raw per-call accumulator is kept here.
            ribo_acc_plus_by_lib[libID], _ = build_call_ribo_metagene(
                shadow_df, genes, bam_paths, shadow_cutoff, min_run_nt, window_nt,
                strand_filter="+")
            ribo_acc_minus_by_lib[libID], _ = build_call_ribo_metagene(
                shadow_df, genes, bam_paths, shadow_cutoff, min_run_nt, window_nt,
                strand_filter="-")
        if his_gpos_by_gene is not None:
            ribo_his_acc_by_lib[libID], ribo_his_acc_gw_by_lib[libID] = build_call_his_ribo_metagene(
                shadow_df, genes, bam_paths, his_gpos_by_gene, shadow_cutoff,
                min_run_nt, window_nt)

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
            cond_ribo_gw = {lid: ribo_acc_gw_by_lib[lid] for lid in libIDs
                            if lid in ribo_acc_gw_by_lib}
            if cond_ribo_gw:
                plot_call_ribo_metagene(
                    cond_ribo_gw, f"{outPrefix}.{cond}.call_ribo_metagene.gene_weighted.pdf",
                    window_nt, min_run_nt, shadow_cutoff,
                    title="Ribo-seq coverage around shadow calls (gene-weighted QC)")
            if split_strand:
                cond_ribo_plus  = {lid: ribo_acc_plus_by_lib[lid] for lid in libIDs
                                   if lid in ribo_acc_plus_by_lib}
                cond_ribo_minus = {lid: ribo_acc_minus_by_lib[lid] for lid in libIDs
                                   if lid in ribo_acc_minus_by_lib}
                plot_call_ribo_metagene(
                    cond_ribo_plus, f"{outPrefix}.{cond}.call_ribo_metagene.plus_strand.pdf",
                    window_nt, min_run_nt, shadow_cutoff, strand_label="+")
                plot_call_ribo_metagene(
                    cond_ribo_minus, f"{outPrefix}.{cond}.call_ribo_metagene.minus_strand.pdf",
                    window_nt, min_run_nt, shadow_cutoff, strand_label="-")
            if his_gpos_by_gene is not None:
                cond_ribo_his = {lid: ribo_his_acc_by_lib[lid] for lid in libIDs
                                 if lid in ribo_his_acc_by_lib}
                if cond_ribo_his:
                    plot_call_ribo_metagene(
                        cond_ribo_his,
                        f"{outPrefix}.{cond}.call_ribo_metagene.his_anchored_qc.pdf",
                        window_nt, min_run_nt, shadow_cutoff,
                        title="Ribo-seq coverage around shadow calls (His-codon-anchored QC)",
                        x_title="distance from nearest contained His codon (nt, 5'$\\to$3')")
                cond_ribo_his_gw = {lid: ribo_his_acc_gw_by_lib[lid] for lid in libIDs
                                    if lid in ribo_his_acc_gw_by_lib}
                if cond_ribo_his_gw:
                    plot_call_ribo_metagene(
                        cond_ribo_his_gw,
                        f"{outPrefix}.{cond}.call_ribo_metagene.his_anchored_qc.gene_weighted.pdf",
                        window_nt, min_run_nt, shadow_cutoff,
                        title="Ribo-seq coverage around shadow calls "
                              "(His-codon-anchored, gene-weighted QC)",
                        x_title="distance from nearest contained His codon (nt, 5'$\\to$3')")
        else:
            print(f"  {cond}: no library had matching ribo-seq BAMs; skipping "
                  f"call_ribo_metagene plot for this condition.", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
