"""
shadowMetagene.py -- Liam Tran, August 2026

"Meta" as in metagene: pool coverage across EVERY gene in one
shadow_calls.parquet, and look at it as a function of position relative
to a shared anchor point, instead of one figure per gene
(riboseqGeneCoverage.py's plot_gene_track/plot_gene_shadow_pileup). Two
anchors, two figures:

  1. Nearest His codon (plot_his_metagene, "outPrefix.his_metagene.pdf") --
     does either signal systematically concentrate near His codons,
     pooled across the whole dataset, not just visible gene-by-gene?
  2. Start codon AND stop codon (plot_startstop_metagene,
     "outPrefix.startstop_metagene.pdf", two panels side by side) --
     standard ribo-seq-style metagene, is there a positional bias
     independent of any given gene's His-codon layout?

Both anchors work in TRANSCRIPT-relative nucleotide distance, not raw
genomic distance -- introns would make raw genomic distance wrong for
spliced genes (in this yeast dataset, ~5% of genes have >1 CDS exon, and
that 5% is almost entirely the RPL/RPS ribosomal-protein paralog family,
so getting this wrong would selectively distort exactly the
highest-depth, most-often-looked-at genes).

Each figure overlays TWO coverage-style signals, deliberately mirroring
riboseqGeneCoverage.py's own per-gene coverage tracks rather than a
per-site probability average:
  - shadow-call depth: read-only reuse of shadow_coverage_track (P_B >=
    shadow_cutoff, run >= min_run_nt qualifying runs, same convention as
    riboseqGeneCoverage.py's shadow-call depth panel) -- "how many reads
    have a qualifying protected run covering this position."
  - ribo-seq depth: read-only reuse of ribo_coverage_track (CIGAR-aware
    read coverage via get_blocks(), footprint-length-restricted, same
    convention as riboseqGeneCoverage.py's ribo-seq depth panel) -- no
    P-site-offset calling here (see riboseqPsiteCalibration.py/
    riboseqShadowCorrelation.py for that, deliberately not used in this
    script), just raw read coverage depth, one count per read per
    position it covers.

Both come back as {gpos: depth} per gene (genomic coordinates); this
script converts each to {tx_pos: depth} via gene_tx_to_gpos_map, a
sequence-free reimplementation of runHMMPerGene.py's _full_tx_map's
interval walk (that function needs a reference FASTA only to report each
position's base identity, which this script never uses, so requiring one
here would be a pure-overhead dependency for no benefit). Each gene then
contributes, at each relative position, its OWN depth there as a
FRACTION of its own total depth (summed over the whole gene, not just the
window) -- not a raw pooled count -- averaged across genes, so a handful
of very highly-expressed/highly-called genes can't dominate the curve
(otherwise the plot would mostly just re-show which genes have the most
overall signal, not any positional pattern). n_genes contributing is
reported in each figure's subtitle rather than as its own depth panel, to
keep this at two panels per anchor instead of four.

Distance to the nearest His codon needs findHisCodonPositions.py's
pickle's tx-relative half: it stores {gene: [tx_positions,
gpos_positions]} per gene, but polysomeShadowHMMQC.py's own
load_his_codon_gpos throws the tx_positions half away and keeps only
gpos (fine for riboseqGeneCoverage.py, which plots against genomic
position) -- this script re-reads the same pickle itself
(load_his_tx_positions) to keep the tx side instead. Distance from the
start codon is tx_pos directly (0 = first CDS base); distance from the
stop codon is tx_pos - cds_length(gene) (0 = the first nt past the last
CDS base) -- both imported/derived read-only from runHMMPerGene.py's
parse_gtf/cds_length, no reimplementation.

Two MORE figures ("outPrefix.his_metagene_bypos.pdf",
"outPrefix.startstop_metagene_bypos.pdf") give the same two anchors a
"by position" alternative to the shadow-call signal above: raw per-site
P_B (gene_bypos_scores), no P_B>=shadow_cutoff/run>=min_run_nt
thresholding at all -- every scored ("A") site contributes, not just
ones already part of a qualifying run. Unlike the run-depth metagenes,
this is NOT per-gene-fraction-normalized -- P_B is already a bounded
[0,1] probability, directly comparable across genes as-is, so pooling
raw sites across every gene/read (build_his_bypos_metagene/
build_startstop_bypos_metagene) is the more direct measurement, not an
approximation that needs correcting for gene-to-gene expression/call-
volume differences the way a raw depth COUNT would. No ribo-seq overlay
on these -- ribo-seq depth has no run/no-run distinction to begin with,
so there's nothing for a "by position" version to contrast against.

Multiple replicates of the same treatment: shadowLibsFile (see Run:, below)
can list more than one shadow_calls.parquet, e.g. '-3AT rep1 .../rep1.parquet'
and '-3AT rep2 .../rep2.parquet'. Libraries are grouped by sample name (the
first column -- everything before the LAST '-' in the 'fileName-rep'
libraryID this convention builds, so a sample name that itself starts with
'-'/'+' still splits correctly, see split_sample_rep) and every replicate of
one sample lands on the SAME set of figures, one shadow-call curve per
replicate overlaid on shared axes (colored via color_map, see resolve_color)
-- one full set of 5 output figures PER SAMPLE, named
"outPrefix.sampleName.*.pdf". ribo-seq depth is NOT split by replicate: it's
pooled once per sample from riboBamList.txt using the SAME sample name as
the condition column (matching riboseqGeneCoverage.py's own condition
convention), then drawn once as a single shared reference curve alongside
however many shadow-call replicate curves that sample has.

Run:
  python3 shadowMetagene.py shadowLibsFile riboBamList.txt gtfFile hisPicklePath outPrefix [window_nt] [shadow_cutoff] [min_run_nt] [color_map.txt]
where shadowLibsFile is a line-delimited inFileParquet.txt-style file of
'fileName rep shadowCallsParquetPath' rows (same convention as
calculateProtectionAcrossParquets.py/polysomeShadowHMMQC.py -- see
parse_shadow_libs_file; fileName is the sample/treatment name, e.g. "-3AT",
and must match riboBamList.txt's own condition column for that sample's
ribo-seq BAMs to be found), riboBamList.txt selects the ribo-seq BAMs per
sample (same convention as riboseqGeneCoverage.py -- e.g.
/data16/liam/working/260804_riboSeq_vs_PS/riboSeqBam.txt), gtfFile is the
yeast GTF (for cds_length per gene), hisPicklePath is
findHisCodonPositions.py's output pickle, outPrefix names the output plots,
window_nt (optional, default 100) is how many nt on either side of each
anchor to include, shadow_cutoff/min_run_nt (optional) override the P_B
cutoff (default 0.5) and run-length floor (default 30nt) for what counts as
a shadow call -- same defaults/meaning as riboseqGeneCoverage.py -- and
color_map.txt (optional) is a manuscript color TSV 'name rep path hex_color'
(no leading '#'); labels are looked up as 'name-rep'/'name_rep' or bare
'name', and a library missing from color_map falls back to a small default
palette (see resolve_color).
"""
import sys, os, bisect, pickle, collections
import pandas as pd

from runHMMPerGene import parse_gtf, cds_length
from riboseqGeneCoverage import (
    load_ribo_bam_list, ribo_coverage_track, shadow_coverage_track,
    load_shadow_calls_df, SHADOW_CUTOFF, MIN_RUN_NT, TARGET_LENGTHS,
)

WINDOW_NT = 100

##Fallback colors (cycled) for a replicate missing from color_map, or when no
##color_map is given at all -- this module has no dependency on the shared
##`common` module (unlike Joshua Arribere's scripts), so this stays self-
##contained, matching compareProtectionToRiboRNAseq.py's own convention.
DEFAULT_REP_PALETTE_CMYK = [
    (0, 0, 0, 1),      # black
    (1, 0.5, 0, 0),    # blue
    (0, 1, 1, 0),      # red
    (1, 0, 1, 0),      # green
    (0, 0.6, 1, 0),    # orange
    (1, 1, 0, 0),      # purple
]


def load_color_map(path):
    """
    Parse a manuscript color-map TSV with columns:
        sample_name, rep, path, hex_color (no leading '#')
    Returns a dict keyed by "name_rep", "name-rep" (this pipeline's
    libraryID convention, see parse_shadow_libs_file), and bare "name"
    (first match wins for the bare key) mapping to "#RRGGBB".
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


def resolve_color(color_map, label, idx):
    """
    Look up label's manuscript color (color_map, from load_color_map) and
    return it as a pyx color object; fall back to
    DEFAULT_REP_PALETTE_CMYK[idx] (cycled) if unmatched or no color_map
    given. pyx is imported locally here (not at module level), matching
    this file's own convention of only importing pyx inside the plotting
    functions that actually need it.
    """
    from pyx import color
    hexcol = color_map.get(label) if color_map else None
    if hexcol:
        hexcol = hexcol.lstrip("#")
        r = int(hexcol[0:2], 16) / 255.0
        g = int(hexcol[2:4], 16) / 255.0
        b = int(hexcol[4:6], 16) / 255.0
        return color.rgb(r, g, b)
    return color.cmyk(*DEFAULT_REP_PALETTE_CMYK[idx % len(DEFAULT_REP_PALETTE_CMYK)])


##Candidate grey levels (light<->dark) for _ribo_grey_palette -- deliberately
##NOT evenly spaced top-to-bottom, so that filtering a few out (because they
##clash with a manuscript grey already in color_map) still leaves a spread
##covering the light/mid/dark range rather than one leftover cluster.
_RIBO_GREY_CANDIDATES = [0.2, 0.6, 0.4, 0.75, 0.3, 0.85, 0.5]


def _ribo_grey_palette(color_map, n, tol=0.12):
    """
    Returns n distinct pyx.color.grey shades for ribo-seq curves in the
    cross-sample overlay figure (plot_startstop_metagene_cross_sample),
    where ribo-seq is drawn as one grey curve per SAMPLE alongside every
    replicate's own manuscript-colored shadow-call curve -- grey (rather
    than that sample's own color) keeps "this is ribo-seq, not another
    replicate" visually obvious at a glance.

    Filters _RIBO_GREY_CANDIDATES down to shades that aren't within `tol`
    grey-level of any color_map hex value that is ITSELF greyish (R, G, B
    all within 0.08 of each other) -- i.e. any manuscript color already
    assigned to a sample/replicate that happens to be a shade of grey --
    so a reader can't mistake that sample's own shadow-call curve for a
    ribo-seq curve. Falls back to the full candidate list (cycled) if
    every candidate happens to clash, or if color_map is empty.
    """
    from pyx import color

    def hex_to_rgb(hexcol):
        hexcol = hexcol.lstrip("#")
        return tuple(int(hexcol[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    used_greys = []
    for hexcol in (color_map or {}).values():
        r, g, b = hex_to_rgb(hexcol)
        if max(r, g, b) - min(r, g, b) <= 0.08:
            used_greys.append((r + g + b) / 3.)

    safe = [lvl for lvl in _RIBO_GREY_CANDIDATES if all(abs(lvl - u) > tol for u in used_greys)]
    palette = safe or _RIBO_GREY_CANDIDATES
    return [color.grey(palette[i % len(palette)]) for i in range(n)]


def parse_shadow_libs_file(path):
    """
    Parse a line-delimited inFileParquet.txt file of format:
        fileName    rep    shadowCallsParquetPath
    (same convention as calculateProtectionAcrossParquets.py's
    parse_parquet_libs_file / polysomeShadowHMMQC.py). fileName is the
    sample/treatment name (e.g. "-3AT"), matched against riboBamList.txt's
    own condition column in main(). shadowCallsParquetPath is passed
    straight through to load_shadow_calls_df, which reads it via
    pd.read_parquet -- that already accepts either a single parquet file
    or a directory of chunks, so no globbing is needed here (unlike
    scripts that iterate parquet files manually row by row).

    Returns a list of (libraryID, shadowCallsParquetPath) tuples, with
    libraryID = 'fileName-rep'.
    """
    libs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            fileName, rep, parquetPath = parts[0], parts[1], parts[2]
            libs.append((f"{fileName}-{rep}", parquetPath))
    return libs


def split_sample_rep(label):
    """
    Splits a 'fileName-rep' libraryID (see parse_shadow_libs_file) back
    into (sampleName, repToken) -- always at the LAST '-', so a sample
    name that itself starts with '-' or '+' (e.g. '-3AT') still splits
    correctly, since only the final separator is treated as the rep
    boundary. A label with no '-' at all returns (label, '').
    """
    return tuple(label.rsplit("-", 1)) if "-" in label else (label, "")


def load_his_tx_positions(pickle_path):
    """
    findHisCodonPositions.py's pickle, KEEPING the transcript-relative half
    polysomeShadowHMMQC.load_his_codon_gpos discards -- {gene: sorted
    [tx_pos, ...]}, one entry per His (CAT/CAC) codon in that gene's CDS,
    splice-aware/CDS-anchored (same coordinate frame as gene_tx_to_gpos_map).
    """
    with open(pickle_path, "rb") as f:
        raw = pickle.load(f)
    return {gene: sorted(int(t) for t in tx_pos) for gene, (tx_pos, _gpos) in raw.items()}


def nearest_his_distance(tx_pos, his_tx_sorted):
    """
    Signed transcript-nt distance from tx_pos to the NEAREST entry in
    his_tx_sorted (already sorted ascending, non-empty). Positive =
    downstream (3' side) of that His codon, negative = upstream (5' side)
    -- meaningful because tx_pos is already strand-corrected transcript
    order (see module docstring), unlike a raw genomic subtraction would be
    on a minus-strand gene.
    """
    i = bisect.bisect_left(his_tx_sorted, tx_pos)
    candidates = []
    if i < len(his_tx_sorted):
        candidates.append(his_tx_sorted[i])
    if i > 0:
        candidates.append(his_tx_sorted[i - 1])
    nearest = min(candidates, key=lambda h: abs(tx_pos - h))
    return tx_pos - nearest


def gene_tx_to_gpos_map(gene, flank_nt=WINDOW_NT):
    """
    {tx_pos: gpos} for one gene: 0 = first CDS base (the start codon's own
    first nt), negative = before it, >= cds_length(gene) = after the last
    CDS base.

    Does NOT rely on GTF-annotated UTR features (gene["utr5"]/["utr3"],
    the exon-minus-CDS derivation runHMMPerGene.py's own _full_tx_map/
    _gpos_to_tx_map use) -- this GTF barely has any (4/5106 genes have any
    annotated 5'UTR at all, and every annotated "3'UTR" is exactly the
    stop codon's own 3nt), so building on those leaves a hard wall of no
    data just past the CDS on either side. Instead, mirroring
    metaStartStop.py's own parseGTF (Joshua Arribere's script, which has
    the same GTF-annotation gap and solves it the same way): pad the CDS's
    own first/last exon by flank_nt genomic nt on the appropriate side
    (before the start codon, after the stop codon) and walk THAT padded
    interval list -- flank_nt defaults to WINDOW_NT so the padding always
    covers whatever window a caller asks for, annotation or not.

    This can't detect a neighboring gene's own CDS/exon sitting inside
    that padding -- metaStartStop.py handles that with a separate
    genome-wide "is this position claimed by more than one transcript"
    pass over its own dict (calculateProtectionAcrossParquets.py's
    len(...)==1 check), which this per-gene function has no visibility
    into. Rare in yeast's compact genome and acceptable for a pooled
    metagene average, but a real simplification worth knowing about if
    this ever needs single-position precision.
    """
    cds = list(gene["cds"])
    if len(cds) == 1:
        s, e = cds[0]
        cds[0] = (max(0, s - flank_nt), e + flank_nt)
    else:
        first_s, first_e = cds[0]
        last_s, last_e = cds[-1]
        if gene["strand"] == "+":
            cds[0]  = (max(0, first_s - flank_nt), first_e)
            cds[-1] = (last_s, last_e + flank_nt)
        else:
            cds[0]  = (first_s, first_e + flank_nt)
            cds[-1] = (max(0, last_s - flank_nt), last_e)

    tx_to_gpos = {}
    tx = -flank_nt
    for (cs, ce) in cds:
        rng = range(cs, ce) if gene["strand"] == "+" else range(ce - 1, cs - 1, -1)
        for gpos in rng:
            tx_to_gpos[tx] = gpos
            tx += 1
    return tx_to_gpos


def gene_depth_to_tx(depth_by_gpos, gene, flank_nt=WINDOW_NT):
    """
    One gene's {gpos: depth} (from ribo_coverage_track or
    shadow_coverage_track, both genomic-keyed) converted to {tx_pos:
    depth} via gene_tx_to_gpos_map's reverse. A gpos with no
    corresponding transcript position for this gene (outside the padded
    CDS window) is dropped.
    """
    gpos_to_tx = {gpos: tx for tx, gpos in gene_tx_to_gpos_map(gene, flank_nt).items()}
    counts = collections.Counter()
    for gpos, n in depth_by_gpos.items():
        tx = gpos_to_tx.get(gpos)
        if tx is not None:
            counts[tx] += n
    return counts


def gene_bypos_scores(gene_df, gene, flank_nt=WINDOW_NT):
    """
    [(tx_pos, P_B), ...] for EVERY scored site of EVERY read of one gene
    -- the "by position" counterpart to shadow_coverage_track's
    run-thresholded depth (see build_his_bypos_metagene). No P_B>=cutoff
    or run-length filtering at all here: this is the raw per-site signal
    write_shadow_calls_to_df produced before any downstream consumer
    thresholds it, same "un-thresholded at the source" design
    write_shadow_calls_to_df's own docstring describes. Converts each
    site's shadow_gpos via gene_tx_to_gpos_map's reverse, same
    flank-padded coordinate frame as the run-depth signal, so both are
    directly comparable position-for-position.
    """
    gpos_to_tx = {gpos: tx for tx, gpos in gene_tx_to_gpos_map(gene, flank_nt).items()}
    out = []
    for row in gene_df.itertuples(index=False):
        for gpos, pb in zip(row.shadow_gpos, row.shadow_P_B):
            tx = gpos_to_tx.get(int(gpos))
            if tx is not None:
                out.append((tx, float(pb)))
    return out


def build_his_density_metagene(tx_depth_by_gene, his_tx_by_gene, window_nt=WINDOW_NT):
    """
    {relpos: [sum_frac, n_genes]} -- pools each gene's own depth-fraction
    (that gene's local depth divided by its own total depth) at signed
    transcript-nt distance to the NEAREST His codon in that gene (see
    nearest_his_distance), restricted to |relpos| <= window_nt. Shared by
    both the shadow-call and ribo-seq density metagenes -- same per-gene
    fraction-averaging rationale in both cases (see module docstring).
    Genes with no His codons, or with zero total depth, don't contribute.
    """
    acc = collections.defaultdict(lambda: [0.0, 0])
    for gname, tx_counts in tx_depth_by_gene.items():
        his_tx = his_tx_by_gene.get(gname)
        if not his_tx or not tx_counts:
            continue
        total = sum(tx_counts.values())
        if not total:
            continue
        for tx, n in tx_counts.items():
            rel = nearest_his_distance(tx, his_tx)
            if -window_nt <= rel <= window_nt:
                entry = acc[rel]
                entry[0] += n / total
                entry[1] += 1
    return acc


def build_startstop_density_metagene(tx_depth_by_gene, gene_cds_len, window_nt=WINDOW_NT):
    """Two {relpos: [sum_frac, n_genes]} dicts -- start-relative (relpos =
    tx_pos) and stop-relative (relpos = tx_pos - cds_length(gene)). See
    build_his_density_metagene for the per-gene fraction-averaging
    rationale; genes missing from gene_cds_len (not in the GTF), or with
    zero total depth, don't contribute to either."""
    start_acc = collections.defaultdict(lambda: [0.0, 0])
    stop_acc  = collections.defaultdict(lambda: [0.0, 0])
    for gname, tx_counts in tx_depth_by_gene.items():
        cds_len = gene_cds_len.get(gname)
        if cds_len is None or not tx_counts:
            continue
        total = sum(tx_counts.values())
        if not total:
            continue
        for tx, n in tx_counts.items():
            frac = n / total
            if -window_nt <= tx <= window_nt:
                entry = start_acc[tx]
                entry[0] += frac
                entry[1] += 1
            stop_rel = tx - cds_len
            if -window_nt <= stop_rel <= window_nt:
                entry = stop_acc[stop_rel]
                entry[0] += frac
                entry[1] += 1
    return start_acc, stop_acc


def build_his_bypos_metagene(sites_by_gene, his_tx_by_gene, window_nt=WINDOW_NT):
    """
    {relpos: [sum_P_B, n_sites]} -- the "by position" counterpart to
    build_his_density_metagene: pools EVERY scored site's raw P_B
    directly (gene_bypos_scores, no run thresholding) at signed
    transcript-nt distance to the nearest His codon. NOT per-gene-
    fraction-normalized like the run-depth metagenes -- P_B is already a
    bounded [0,1] probability, directly comparable across genes as-is,
    unlike a raw depth count (which scales with a gene's own expression/
    call volume and would need that normalization to avoid a few
    highly-called genes dominating the average).
    """
    acc = collections.defaultdict(lambda: [0.0, 0])
    for gname, sites in sites_by_gene.items():
        his_tx = his_tx_by_gene.get(gname)
        if not his_tx or not sites:
            continue
        for tx, pb in sites:
            rel = nearest_his_distance(tx, his_tx)
            if -window_nt <= rel <= window_nt:
                entry = acc[rel]
                entry[0] += pb
                entry[1] += 1
    return acc


def build_startstop_bypos_metagene(sites_by_gene, gene_cds_len, window_nt=WINDOW_NT):
    """By-position counterpart to build_startstop_density_metagene -- see
    build_his_bypos_metagene for why this pools raw P_B directly instead
    of per-gene-normalized fractions."""
    start_acc = collections.defaultdict(lambda: [0.0, 0])
    stop_acc  = collections.defaultdict(lambda: [0.0, 0])
    for gname, sites in sites_by_gene.items():
        cds_len = gene_cds_len.get(gname)
        if cds_len is None or not sites:
            continue
        for tx, pb in sites:
            if -window_nt <= tx <= window_nt:
                entry = start_acc[tx]
                entry[0] += pb
                entry[1] += 1
            stop_rel = tx - cds_len
            if -window_nt <= stop_rel <= window_nt:
                entry = stop_acc[stop_rel]
                entry[0] += pb
                entry[1] += 1
    return start_acc, stop_acc


def _means(acc, window_nt):
    """{acc} -> (xs, means) for x in [-window_nt, window_nt], means is
    None at any x with zero contributing genes (so the plot leaves a true
    gap there rather than drawing a misleading interpolated 0)."""
    xs = list(range(-window_nt, window_nt + 1))
    means = []
    for x in xs:
        s, n = acc.get(x, (0.0, 0))
        means.append(s / n if n else None)
    return xs, means


def _series_max(*series_lists):
    """
    series_lists: one or more lists of (label,col,xs,means) tuples (see
    _plot_anchor_panels). Returns 1.1x the max non-None mean across ALL
    of them combined, or 1e-9 if none contribute -- used to give
    side-by-side panels (e.g. start vs stop) a SHARED y-axis scale
    instead of each auto-ranging independently off just its own data
    (see plot_startstop_metagene's "consistent axes" note).
    """
    vals = [m for series in series_lists for _l, _c, xs, means in series
            for _x, m in zip(xs, means) if m is not None]
    return max(vals) * 1.1 if vals else 1e-9


def _plot_anchor_panels(c, graph, style, color, pyx_text, path, xpos, ypos,
                        panel_w, shadow_h, ribo_h, gap,
                        shadow_series, ribo_series, x_title, anchor_label,
                        shadow_max=None, ribo_max=None):
    """
    One anchor's worth of panels (shared by plot_his_metagene,
    plot_startstop_metagene, and plot_startstop_metagene_cross_sample),
    bottom to top: mean per-gene shadow-call depth fraction, then mean
    per-gene ribo-seq depth fraction -- each an overlaid curve PER ENTRY
    in shadow_series/ribo_series respectively. Both panels share one
    x-axis and a dashed vertical reference line at x=0 (the anchor
    itself). Returns the top-of-stack y-position (for the
    anchor_label/figure-title text above it).

    shadow_series/ribo_series are each a list of (label,col,xs,means)
    tuples -- every entry shares the same window_nt range (see main()),
    so a single x-axis built from the first entry is valid for all of
    them. Two different usages share this same shape:
    - plot_his_metagene/plot_startstop_metagene: shadow_series has one
      entry per REPLICATE of one sample; ribo_series has exactly ONE
      entry (a single curve pooled across every BAM for that sample's
      condition, not split by replicate -- see main()).
    - plot_startstop_metagene_cross_sample: shadow_series has one entry
      per REPLICATE spanning EVERY sample (not just one), each in its own
      manuscript color; ribo_series has one entry PER SAMPLE (ribo-seq
      depth is pooled per sample/condition, not per replicate -- see
      main()), each a distinct shade of grey (_ribo_grey_palette) instead
      of that sample's own color, so ribo-seq stays visually distinct
      from every replicate's shadow-call curve.

    shadow_max/ribo_max, if given, override this call's own data-driven
    axis max -- the two-anchor callers compute these ONCE across both
    the start and stop anchors (see _series_max) so their y-axes share
    the same scale/tick labels instead of each auto-ranging independently
    ("consistent axes" across the two side-by-side figures). None
    (plot_his_metagene's single-anchor case, with no other anchor to
    share a scale with) falls back to this anchor's own range.
    """
    all_shadow_pts = [(x, m) for _label, _col, xs, means in shadow_series
                       for x, m in zip(xs, means) if m is not None]
    if shadow_max is None:
        shadow_max = max((m for _x, m in all_shadow_pts), default=1e-9) * 1.1 \
            if all_shadow_pts else 1e-9
    x_lo, x_hi = (shadow_series[0][2][0], shadow_series[0][2][-1]) \
        if shadow_series else (ribo_series[0][2][0], ribo_series[0][2][-1])
    g_shadow = graph.graphxy(
        width=panel_w, height=shadow_h, xpos=xpos, ypos=ypos,
        x=graph.axis.linear(min=x_lo, max=x_hi, title=x_title),
        y=graph.axis.linear(min=0, max=shadow_max, title="shadow-call frac."))
    c.insert(g_shadow)
    for _label, col, xs, means in shadow_series:
        pts = [(x, m) for x, m in zip(xs, means) if m is not None]
        if pts:
            g_shadow.plot(graph.data.points(pts, x=1, y=2),
                          [graph.style.line([col, style.linewidth.Thick])])
    g_shadow.plot(graph.data.function("x(y)=0", min=0, max=shadow_max),
                  [graph.style.line([color.grey(0.6), style.linewidth.thin, style.linestyle.dashed])])

    ribo_ypos = ypos + shadow_h + gap
    all_ribo_pts = [(x, m) for _label, _col, xs, means in ribo_series
                     for x, m in zip(xs, means) if m is not None]
    if ribo_max is None:
        ribo_max = max((m for _x, m in all_ribo_pts), default=1e-9) * 1.1 if all_ribo_pts else 1e-9
    g_ribo = graph.graphxy(
        width=panel_w, height=ribo_h, xpos=xpos, ypos=ribo_ypos,
        x=graph.axis.linkedaxis(g_shadow.axes["x"], painter=graph.axis.painter.linked()),
        y=graph.axis.linear(min=0, max=ribo_max, title="ribo-seq frac."))
    c.insert(g_ribo)
    for _label, col, xs, means in ribo_series:
        pts = [(x, m) for x, m in zip(xs, means) if m is not None]
        if pts:
            g_ribo.plot(graph.data.points(pts, x=1, y=2),
                       [graph.style.line([col, style.linewidth.Thick])])
    g_ribo.plot(graph.data.function("x(y)=0", min=0, max=ribo_max),
               [graph.style.line([color.grey(0.6), style.linewidth.thin, style.linestyle.dashed])])

    top_ypos = ribo_ypos + ribo_h
    c.text(xpos + panel_w / 2., top_ypos + 0.3,
          anchor_label, [pyx_text.halign.center, pyx_text.size.normalsize])
    return top_ypos


def _draw_rep_legend(c, path, style, pyx_text, legend_entries, legend_x, legend_y_top, line_dy=0.4):
    """
    Draws a small color-keyed legend -- one line per (label,col) in
    legend_entries -- starting at (legend_x,legend_y_top) and stacking
    downward by line_dy per entry. Used by the multi-replicate metagene
    figures so each overlaid curve's replicate identity is visible
    (color alone isn't self-explanatory without one). legend_x/
    legend_y_top are plain canvas coordinates (not a graphxy's own local
    frame), so no xpos/ypos offset is needed here.
    """
    for i, (label, col) in enumerate(legend_entries):
        ly = legend_y_top - i * line_dy
        c.stroke(path.line(legend_x, ly, legend_x + 0.6, ly),
                 [col, style.linewidth.Thick])
        c.text(legend_x + 0.75, ly, label, [pyx_text.valign.middle, pyx_text.size.small])


def plot_his_metagene(shadow_acc_by_rep, ribo_acc, pdf_path, window_nt=WINDOW_NT,
                      rep_colors=None, n_genes_shadow_by_rep=None, n_genes_ribo=0):
    """
    One figure: mean per-gene shadow-call depth fraction -- one overlaid
    curve per replicate (see main()'s per-sample grouping) -- and mean
    per-gene ribo-seq depth fraction (single curve, pooled once across
    this sample's condition; not split by replicate) vs. signed
    transcript-nt distance to the nearest His codon. See
    build_his_density_metagene.

    shadow_acc_by_rep is {repLabel:acc}; rep_colors is {repLabel:pyx
    color} (see resolve_color). Replicates are drawn (and legended, if
    there's more than one) in sorted(shadow_acc_by_rep) order.
    """
    from pyx import canvas, graph, color, style, text as pyx_text, path

    rep_colors = rep_colors or {}
    n_genes_shadow_by_rep = n_genes_shadow_by_rep or {}
    default_col = color.cmyk(1, 0.3, 0, 0.1)
    rep_labels = sorted(shadow_acc_by_rep.keys())
    shadow_series = []
    for label in rep_labels:
        xs, means = _means(shadow_acc_by_rep[label], window_nt)
        shadow_series.append((label, rep_colors.get(label, default_col), xs, means))
    ribo_xs, ribo_means = _means(ribo_acc, window_nt)
    col_ribo = color.cmyk(0.7, 0, 1, 0.2)
    ribo_series = [("ribo-seq", col_ribo, ribo_xs, ribo_means)]

    c = canvas.canvas()
    panel_w, shadow_h, ribo_h, gap = 12, 3.5, 3.5, 0.8
    top_ypos = _plot_anchor_panels(
        c, graph, style, color, pyx_text, path, 0, 0, panel_w, shadow_h, ribo_h, gap,
        shadow_series, ribo_series,
        "nt from nearest His codon (transcript-relative)",
        "Metagene: shadow-call + ribo-seq depth around His codons")

    if len(rep_labels) > 1:
        _draw_rep_legend(c, path, style, pyx_text,
                         [(label, rep_colors.get(label, default_col)) for label in rep_labels],
                         panel_w + 0.3, shadow_h + ribo_h + gap - 0.3)

    genes_str = ", ".join(f"{label}: {n_genes_shadow_by_rep.get(label, 0)}" for label in rep_labels)
    c.text(panel_w / 2., top_ypos + 0.85,
          f"gene(s) with shadow-call depth within $\\pm${window_nt}nt of a His codon -- "
          f"{genes_str}; {n_genes_ribo} gene(s) with ribo-seq depth",
          [pyx_text.halign.center, pyx_text.size.scriptsize])

    c.writePDFfile(str(pdf_path))
    print(f"Wrote {pdf_path}", file=sys.stderr)


def plot_startstop_metagene(start_shadow_acc_by_rep, stop_shadow_acc_by_rep,
                            start_ribo_acc, stop_ribo_acc, pdf_path, window_nt=WINDOW_NT,
                            rep_colors=None, n_genes_shadow_by_rep=None, n_genes_ribo=0):
    """
    One figure, two side-by-side panel pairs: mean per-gene shadow-call
    (one overlaid curve per replicate) and ribo-seq (single, pooled)
    depth fractions vs. transcript-nt distance from the start codon
    (left) and from the stop codon (right). See
    build_startstop_density_metagene.
    """
    from pyx import canvas, graph, color, style, text as pyx_text, path

    rep_colors = rep_colors or {}
    n_genes_shadow_by_rep = n_genes_shadow_by_rep or {}
    default_col = color.cmyk(1, 0.3, 0, 0.1)
    rep_labels = sorted(start_shadow_acc_by_rep.keys())

    def build_series(acc_by_rep):
        series = []
        for label in rep_labels:
            xs, means = _means(acc_by_rep[label], window_nt)
            series.append((label, rep_colors.get(label, default_col), xs, means))
        return series

    start_shadow_series = build_series(start_shadow_acc_by_rep)
    stop_shadow_series  = build_series(stop_shadow_acc_by_rep)
    start_ribo_xs, start_ribo_means = _means(start_ribo_acc, window_nt)
    stop_ribo_xs, stop_ribo_means   = _means(stop_ribo_acc, window_nt)
    col_ribo = color.cmyk(0.7, 0, 1, 0.2)
    start_ribo_series = [("ribo-seq", col_ribo, start_ribo_xs, start_ribo_means)]
    stop_ribo_series  = [("ribo-seq", col_ribo, stop_ribo_xs, stop_ribo_means)]

    # Shared y-axis scale across the start AND stop panels (see _series_max)
    # -- otherwise each side auto-ranges off only its own data and the two
    # panels' tick labels don't line up, making them hard to compare.
    shadow_max = _series_max(start_shadow_series, stop_shadow_series)
    ribo_max = _series_max(start_ribo_series, stop_ribo_series)

    c = canvas.canvas()
    # shadow_h==ribo_h==panel_w so EACH panel (not just the stacked pair) is its
    # own square; panel_gap must clear the stop panel's y-axis title/tick-label
    # width or it overlaps the start panel's right edge -- verified empirically
    # via graph.graphxy.bbox(), panel_gap=1.5 overlaps (by ~0.04cm) at this
    # panel_w, panel_gap>=2 does not (same class of bug as
    # compareProtectionToRiboRNAseq.py's mkTEProtectionPanels).
    panel_w, shadow_h, ribo_h, gap, panel_gap = 8, 8, 8, 1.0, 2.0

    top1 = _plot_anchor_panels(
        c, graph, style, color, pyx_text, path, 0, 0, panel_w, shadow_h, ribo_h, gap,
        start_shadow_series, start_ribo_series,
        "nt from start codon", "Start codon", shadow_max=shadow_max, ribo_max=ribo_max)
    _top2 = _plot_anchor_panels(
        c, graph, style, color, pyx_text, path, panel_w + panel_gap, 0, panel_w, shadow_h, ribo_h, gap,
        stop_shadow_series, stop_ribo_series,
        "nt from stop codon", "Stop codon", shadow_max=shadow_max, ribo_max=ribo_max)

    if len(rep_labels) > 1:
        _draw_rep_legend(c, path, style, pyx_text,
                         [(label, rep_colors.get(label, default_col)) for label in rep_labels],
                         2 * panel_w + panel_gap + 0.3, shadow_h + ribo_h + gap - 0.3)

    # anchor_label (drawn inside _plot_anchor_panels) sits at top1+0.3 and, at
    # normalsize, extends up to roughly top1+0.65 -- title/detail need to clear
    # that, not just leave a flat 0.2/0.4 gap (which visibly overlapped it; see
    # the module's overlap-fix history for _plot_anchor_panels/panel_gap).
    genes_str = ", ".join(f"{label}: {n_genes_shadow_by_rep.get(label, 0)}" for label in rep_labels)
    c.text((2 * panel_w + panel_gap) / 2., top1 + 1.3,
          "Metagene: shadow-call + ribo-seq depth around start/stop codons",
          [pyx_text.halign.center, pyx_text.size.normalsize])
    c.text((2 * panel_w + panel_gap) / 2., top1 + 0.9,
          f"gene(s) with shadow-call depth within $\\pm${window_nt}nt of start/stop -- "
          f"{genes_str}; {n_genes_ribo} gene(s) with ribo-seq depth",
          [pyx_text.halign.center, pyx_text.size.scriptsize])

    c.writePDFfile(str(pdf_path))
    print(f"Wrote {pdf_path}", file=sys.stderr)


def plot_startstop_metagene_cross_sample(start_shadow_acc_by_rep, stop_shadow_acc_by_rep,
                                         start_ribo_acc_by_sample, stop_ribo_acc_by_sample,
                                         pdf_path, window_nt=WINDOW_NT, rep_colors=None,
                                         ribo_colors_by_sample=None,
                                         n_genes_shadow_by_rep=None, n_genes_ribo_by_sample=None):
    """
    Cross-sample counterpart to plot_startstop_metagene: same two
    side-by-side Start codon | Stop codon panel pairs, but overlays EVERY
    replicate of EVERY sample (not just one sample's own reps) on the
    shadow-call panels, color-coded per replicate via rep_colors -- same
    manuscript-color convention as the per-sample figures, just spanning
    the whole run instead of one sample -- alongside one ribo-seq curve
    PER SAMPLE (ribo-seq depth is pooled per sample/condition, not per
    replicate -- see main()), drawn in a distinct shade of grey
    (ribo_colors_by_sample, see _ribo_grey_palette) instead of that
    sample's own color, so ribo-seq stays visually distinct from every
    replicate's shadow-call curve at a glance.
    """
    from pyx import canvas, graph, color, style, text as pyx_text, path

    rep_colors = rep_colors or {}
    ribo_colors_by_sample = ribo_colors_by_sample or {}
    n_genes_shadow_by_rep = n_genes_shadow_by_rep or {}
    n_genes_ribo_by_sample = n_genes_ribo_by_sample or {}
    default_col = color.cmyk(1, 0.3, 0, 0.1)
    default_grey = color.grey(0.5)
    rep_labels = sorted(start_shadow_acc_by_rep.keys())
    sample_names = sorted(start_ribo_acc_by_sample.keys())

    def build_shadow_series(acc_by_rep):
        series = []
        for label in rep_labels:
            xs, means = _means(acc_by_rep[label], window_nt)
            series.append((label, rep_colors.get(label, default_col), xs, means))
        return series

    def build_ribo_series(acc_by_sample):
        series = []
        for s in sample_names:
            xs, means = _means(acc_by_sample[s], window_nt)
            series.append((s, ribo_colors_by_sample.get(s, default_grey), xs, means))
        return series

    start_shadow_series = build_shadow_series(start_shadow_acc_by_rep)
    stop_shadow_series  = build_shadow_series(stop_shadow_acc_by_rep)
    start_ribo_series   = build_ribo_series(start_ribo_acc_by_sample)
    stop_ribo_series    = build_ribo_series(stop_ribo_acc_by_sample)

    # Shared y-axis scale across the start AND stop panels (see _series_max)
    # -- otherwise each side auto-ranges off only its own data and the two
    # panels' tick labels don't line up, making them hard to compare.
    shadow_max = _series_max(start_shadow_series, stop_shadow_series)
    ribo_max = _series_max(start_ribo_series, stop_ribo_series)

    c = canvas.canvas()
    # Same square-panel/panel_gap geometry as plot_startstop_metagene -- see
    # that function's comment for why panel_gap must be >=2 at this panel_w.
    panel_w, shadow_h, ribo_h, gap, panel_gap = 8, 8, 8, 1.0, 2.0

    top1 = _plot_anchor_panels(
        c, graph, style, color, pyx_text, path, 0, 0, panel_w, shadow_h, ribo_h, gap,
        start_shadow_series, start_ribo_series,
        "nt from start codon", "Start codon", shadow_max=shadow_max, ribo_max=ribo_max)
    _top2 = _plot_anchor_panels(
        c, graph, style, color, pyx_text, path, panel_w + panel_gap, 0, panel_w, shadow_h, ribo_h, gap,
        stop_shadow_series, stop_ribo_series,
        "nt from stop codon", "Stop codon", shadow_max=shadow_max, ribo_max=ribo_max)

    legend_entries = [(label, rep_colors.get(label, default_col)) for label in rep_labels] + \
        [(f"ribo-seq: {s}", ribo_colors_by_sample.get(s, default_grey)) for s in sample_names]
    if legend_entries:
        _draw_rep_legend(c, path, style, pyx_text, legend_entries,
                         2 * panel_w + panel_gap + 0.3, shadow_h + ribo_h + gap - 0.3)

    # Same title/detail offsets as plot_startstop_metagene (top1+1.3/+0.9) --
    # already verified overlap-free for this exact panel geometry/text sizes,
    # including the $\pm$ glyph-height quirk (see that function's history).
    shadow_str = "; ".join(f"{label}: {n_genes_shadow_by_rep.get(label, 0)}" for label in rep_labels)
    ribo_str = "; ".join(f"{s}: {n_genes_ribo_by_sample.get(s, 0)}" for s in sample_names)
    c.text((2 * panel_w + panel_gap) / 2., top1 + 1.3,
          "Metagene: shadow-call + ribo-seq depth around start/stop codons, samples overlaid",
          [pyx_text.halign.center, pyx_text.size.normalsize])
    c.text((2 * panel_w + panel_gap) / 2., top1 + 0.9,
          f"shadow-call gene(s) w/ depth within $\\pm${window_nt}nt -- {shadow_str}; "
          f"ribo-seq gene(s) -- {ribo_str}",
          [pyx_text.halign.center, pyx_text.size.scriptsize])

    c.writePDFfile(str(pdf_path))
    print(f"Wrote {pdf_path}", file=sys.stderr)


def _bypos_series_maxes(*series_lists):
    """
    series_lists: one or more lists of (label,col,xs,means,depths) tuples
    (see _plot_bypos_anchor). Returns (sig_max,depth_max) -- 1.1x the max
    non-None mean-P_B / the max site count, each across ALL of them
    combined -- used to give side-by-side panels (e.g. start vs stop) a
    SHARED y-axis scale instead of each auto-ranging independently off
    just its own data (see plot_startstop_bypos_metagene's "consistent
    axes" note).
    """
    means_vals = [m for series in series_lists for _l, _c, xs, means, _d in series
                  for _x, m in zip(xs, means) if m is not None]
    depth_vals = [d for series in series_lists for _l, _c, _xs, _m, depths in series
                  for d in depths]
    sig_max = max(means_vals) * 1.1 if means_vals else 1.0
    depth_max = max(depth_vals) if depth_vals else 1
    return sig_max, depth_max


def _plot_bypos_anchor(c, graph, style, color, pyx_text, path, xpos, ypos,
                       panel_w, sig_h, dep_h, gap, series, x_title, anchor_label,
                       sig_max=None, depth_max=None):
    """
    One anchor's raw by-position panels: n_sites depth (bottom), mean
    P_B (top) -- the "by position" counterpart to _plot_anchor_panels'
    per-gene-fraction run-depth signal. One overlaid curve per replicate
    in BOTH panels (series), so replicates are comparable in support
    (depth) as well as signal (P_B), instead of one library's raw P_B
    against a single fixed-gray depth curve. A dotted P_B=0.5 reference
    (this codebase's own shadow-call P_B cutoff convention) marks where
    the run-depth figures' threshold sits, so the two are visually
    comparable despite plotting different quantities.

    series is a list of (label,col,xs,means,depths) tuples, one per
    replicate, all sharing the same window_nt range.

    sig_max/depth_max, if given, override this call's own data-driven
    axis max -- plot_startstop_bypos_metagene computes these ONCE across
    both the start and stop anchors (see _bypos_series_maxes) so their
    y-axes share the same scale/tick labels ("consistent axes" across the
    two side-by-side figures). None (plot_his_bypos_metagene's
    single-anchor case) falls back to this anchor's own range.
    """
    x_lo, x_hi = series[0][2][0], series[0][2][-1]
    if depth_max is None:
        depth_max = max((d for _l, _c, _xs, _m, depths in series for d in depths), default=1)
    g_dep = graph.graphxy(
        width=panel_w, height=dep_h, xpos=xpos, ypos=ypos,
        x=graph.axis.linear(min=x_lo, max=x_hi, title=x_title),
        y=graph.axis.linear(min=0, max=max(1, depth_max), title="n sites"))
    c.insert(g_dep)
    for _label, col, xs, _means, depths in series:
        g_dep.plot(graph.data.points(list(zip(xs, depths)), x=1, y=2),
                  [graph.style.line([col, style.linewidth.Thick])])
    g_dep.plot(graph.data.function("x(y)=0", min=0, max=max(1, depth_max)),
              [graph.style.line([color.grey(0.6), style.linewidth.thin, style.linestyle.dashed])])

    sig_ypos = ypos + dep_h + gap
    if sig_max is None:
        all_pts = [(x, m) for _l, _c, xs, means, _d in series for x, m in zip(xs, means) if m is not None]
        sig_max = max((m for _x, m in all_pts), default=1.0) * 1.1 if all_pts else 1.0
    g_sig = graph.graphxy(
        width=panel_w, height=sig_h, xpos=xpos, ypos=sig_ypos,
        x=graph.axis.linkedaxis(g_dep.axes["x"], painter=graph.axis.painter.linked()),
        y=graph.axis.linear(min=0, max=max(sig_max, 0.55), title="mean P$_B$"))
    c.insert(g_sig)
    for _label, col, xs, means, _depths in series:
        pts = [(x, m) for x, m in zip(xs, means) if m is not None]
        if pts:
            g_sig.plot(graph.data.points(pts, x=1, y=2),
                      [graph.style.line([col, style.linewidth.Thick])])
    g_sig.plot(graph.data.function("x(y)=0", min=0, max=max(sig_max, 0.55)),
              [graph.style.line([color.grey(0.6), style.linewidth.thin, style.linestyle.dashed])])
    g_sig.plot(graph.data.function("y(x)=0.5", min=x_lo, max=x_hi),
              [graph.style.line([color.grey(0.6), style.linewidth.thin, style.linestyle.dotted])])

    top_ypos = sig_ypos + sig_h
    c.text(xpos + panel_w / 2., top_ypos + 0.3,
          anchor_label, [pyx_text.halign.center, pyx_text.size.normalsize])
    return top_ypos


def plot_his_bypos_metagene(acc_by_rep, pdf_path, window_nt=WINDOW_NT,
                            rep_colors=None, n_genes_by_rep=None, n_sites_by_rep=None):
    """
    One figure: mean raw per-site P_B (no run thresholding, one overlaid
    curve per replicate) vs. signed transcript-nt distance to the
    nearest His codon, plus each replicate's own n_sites depth-of-
    support curve. The "by position" counterpart to plot_his_metagene --
    see build_his_bypos_metagene.
    """
    from pyx import canvas, graph, color, style, text as pyx_text, path

    rep_colors = rep_colors or {}
    n_genes_by_rep = n_genes_by_rep or {}
    n_sites_by_rep = n_sites_by_rep or {}
    default_col = color.cmyk(1, 0.3, 0, 0.1)
    rep_labels = sorted(acc_by_rep.keys())
    series = []
    for label in rep_labels:
        xs, means = _means(acc_by_rep[label], window_nt)
        depths = [acc_by_rep[label].get(x, (0.0, 0))[1] for x in xs]
        series.append((label, rep_colors.get(label, default_col), xs, means, depths))

    c = canvas.canvas()
    panel_w, sig_h, dep_h, gap = 12, 3.5, 2.0, 0.8
    top_ypos = _plot_bypos_anchor(
        c, graph, style, color, pyx_text, path, 0, 0, panel_w, sig_h, dep_h, gap,
        series, "nt from nearest His codon (transcript-relative)",
        "By-position metagene: raw shadow-call P$_B$ around His codons")

    if len(rep_labels) > 1:
        _draw_rep_legend(c, path, style, pyx_text,
                         [(label, rep_colors.get(label, default_col)) for label in rep_labels],
                         panel_w + 0.3, sig_h + dep_h + gap - 0.3)

    detail_str = "; ".join(f"{label}: {n_genes_by_rep.get(label, 0)} gene(s), "
                            f"{n_sites_by_rep.get(label, 0)} site(s)" for label in rep_labels)
    c.text(panel_w / 2., top_ypos + 0.85,
          f"within $\\pm${window_nt}nt of a His codon (every scored site, no P$_B$ cutoff or "
          f"run-length filtering) -- {detail_str}",
          [pyx_text.halign.center, pyx_text.size.scriptsize])

    c.writePDFfile(str(pdf_path))
    print(f"Wrote {pdf_path}", file=sys.stderr)


def plot_startstop_bypos_metagene(start_acc_by_rep, stop_acc_by_rep, pdf_path, window_nt=WINDOW_NT,
                                  rep_colors=None, n_genes_by_rep=None,
                                  n_sites_start_by_rep=None, n_sites_stop_by_rep=None):
    """
    One figure, two side-by-side panel pairs: mean raw per-site P_B (no
    run thresholding, one overlaid curve per replicate) vs. transcript-nt
    distance from the start codon (left) and from the stop codon
    (right). The "by position" counterpart to plot_startstop_metagene --
    see build_startstop_bypos_metagene.
    """
    from pyx import canvas, graph, color, style, text as pyx_text, path

    rep_colors = rep_colors or {}
    n_genes_by_rep = n_genes_by_rep or {}
    n_sites_start_by_rep = n_sites_start_by_rep or {}
    n_sites_stop_by_rep = n_sites_stop_by_rep or {}
    default_col = color.cmyk(1, 0.3, 0, 0.1)
    rep_labels = sorted(start_acc_by_rep.keys())

    def build_series(acc_by_rep):
        series = []
        for label in rep_labels:
            xs, means = _means(acc_by_rep[label], window_nt)
            depths = [acc_by_rep[label].get(x, (0.0, 0))[1] for x in xs]
            series.append((label, rep_colors.get(label, default_col), xs, means, depths))
        return series

    start_series = build_series(start_acc_by_rep)
    stop_series  = build_series(stop_acc_by_rep)

    # Shared y-axis scale across the start AND stop panels (see
    # _bypos_series_maxes) -- otherwise each side auto-ranges off only its own
    # data and the two panels' tick labels don't line up.
    sig_max, depth_max = _bypos_series_maxes(start_series, stop_series)

    c = canvas.canvas()
    # sig_h==dep_h==panel_w so EACH panel (not just the stacked pair) is its own
    # square; panel_gap must clear the stop panel's y-axis title/tick-label width
    # or it overlaps the start panel's right edge -- verified empirically via
    # graph.graphxy.bbox(), panel_gap=1.5 overlaps (by ~0.02cm) at this panel_w,
    # panel_gap>=2 does not (same class of bug as
    # compareProtectionToRiboRNAseq.py's mkTEProtectionPanels).
    panel_w, sig_h, dep_h, gap, panel_gap = 8, 8, 8, 1.0, 2.0

    top1 = _plot_bypos_anchor(
        c, graph, style, color, pyx_text, path, 0, 0, panel_w, sig_h, dep_h, gap,
        start_series, "nt from start codon", "Start codon", sig_max=sig_max, depth_max=depth_max)
    _top2 = _plot_bypos_anchor(
        c, graph, style, color, pyx_text, path, panel_w + panel_gap, 0, panel_w, sig_h, dep_h, gap,
        stop_series, "nt from stop codon", "Stop codon", sig_max=sig_max, depth_max=depth_max)

    if len(rep_labels) > 1:
        _draw_rep_legend(c, path, style, pyx_text,
                         [(label, rep_colors.get(label, default_col)) for label in rep_labels],
                         2 * panel_w + panel_gap + 0.3, sig_h + dep_h + gap - 0.3)

    # anchor_label (drawn inside _plot_bypos_anchor) sits at top1+0.3 and, at
    # normalsize, extends up to roughly top1+0.65 -- title/detail need to clear
    # that, not just leave a flat 0.2/0.4 gap (which visibly overlapped it).
    detail_str = "; ".join(
        f"{label}: {n_genes_by_rep.get(label, 0)} gene(s) -- "
        f"{n_sites_start_by_rep.get(label, 0)} near start, {n_sites_stop_by_rep.get(label, 0)} near stop"
        for label in rep_labels)
    c.text((2 * panel_w + panel_gap) / 2., top1 + 1.35,
          "By-position metagene: raw shadow-call P$_B$ around start/stop codons",
          [pyx_text.halign.center, pyx_text.size.normalsize])
    c.text((2 * panel_w + panel_gap) / 2., top1 + 0.85,
          f"(every scored site, no P$_B$ cutoff or run-length filtering) -- {detail_str}",
          [pyx_text.halign.center, pyx_text.size.scriptsize])

    c.writePDFfile(str(pdf_path))
    print(f"Wrote {pdf_path}", file=sys.stderr)


def main(args):
    shadowLibsPath, riboBamListPath, gtfPath, hisPicklePath, outPrefix = args[:5]
    window_nt     = int(args[5])   if len(args) > 5 else WINDOW_NT
    shadow_cutoff = float(args[6]) if len(args) > 6 else SHADOW_CUTOFF
    min_run_nt    = int(args[7])   if len(args) > 7 else MIN_RUN_NT
    colorMapPath  = args[8] if len(args) > 8 else None

    color_map = load_color_map(colorMapPath) if colorMapPath else {}

    out_dir = os.path.dirname(outPrefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    libs = parse_shadow_libs_file(shadowLibsPath)
    if not libs:
        sys.exit(f"No libraries found in {shadowLibsPath}; exiting.")
    if colorMapPath:
        unmatched = [label for label, _ in libs if label not in color_map]
        if unmatched:
            print(f"  WARNING: no color found in {colorMapPath} for librar"
                  f"{'y' if len(unmatched) == 1 else 'ies'} {unmatched}; "
                  f"falling back to the default palette.", file=sys.stderr)

    # Group library labels (e.g. '-3AT-rep1','-3AT-rep2') by sample name --
    # replicates of the same treatment get overlaid on the same figures (see
    # module docstring); ribo-seq condition selection below also uses this
    # same sample name (riboBamList.txt's own condition column, e.g. '-3AT').
    samples_to_libs = collections.defaultdict(list)  # sampleName -> [(repToken,label,parquetPath),...]
    for label, parquet_path in libs:
        sample_name, rep_token = split_sample_rep(label)
        samples_to_libs[sample_name].append((rep_token, label, parquet_path))

    genes = parse_gtf(gtfPath)
    his_tx_by_gene = load_his_tx_positions(hisPicklePath)
    print(f"  {sum(1 for tx in his_tx_by_gene.values() if tx)} gene(s) have >=1 His codon.",
          file=sys.stderr)

    # A GLOBAL rep_colors, spanning every replicate label across every sample
    # (not just one sample's own reps, unlike the per-sample rep_colors built
    # inside the loop below) -- needed for the cross-sample overlay figures,
    # so two different samples' same-index fallback-palette replicate (e.g.
    # both samples' "rep1" if color_map doesn't cover them) don't collide on
    # the same color; idx is assigned once, in one global sorted pass, same
    # resolve_color/DEFAULT_REP_PALETTE_CMYK convention as everywhere else.
    all_rep_labels = sorted(label for label, _p in libs)
    global_rep_colors = {label: resolve_color(color_map, label, idx)
                         for idx, label in enumerate(all_rep_labels)}

    # Collected across the per-sample loop below (just accumulating the SAME
    # per-rep/per-sample accumulators each per-sample figure already builds,
    # nothing recomputed) to build the cross-sample start/stop overlay
    # figures (plot_startstop_metagene_cross_sample) once every sample has
    # been processed -- see the module docstring / _plot_anchor_panels for
    # why ribo-seq depth there is one curve PER SAMPLE (unlike the per-sample
    # figures, where it's a single curve shared by every one of that
    # sample's reps) while shadow-call depth is one curve per REPLICATE.
    all_start_shadow_acc_by_rep = {}
    all_stop_shadow_acc_by_rep  = {}
    all_n_genes_shadow_by_rep   = {}
    all_start_ribo_acc_by_sample = {}
    all_stop_ribo_acc_by_sample  = {}
    all_n_genes_ribo_by_sample   = {}
    all_start_bypos_acc_by_rep = {}
    all_stop_bypos_acc_by_rep  = {}
    all_n_genes_bypos_by_rep   = {}
    all_n_sites_start_bypos_by_rep = {}
    all_n_sites_stop_bypos_by_rep  = {}

    for sample_name in sorted(samples_to_libs.keys()):
        reps = sorted(samples_to_libs[sample_name])
        rep_labels = [label for _rep, label, _p in reps]
        print(f"\n=== Sample {sample_name}: {len(reps)} replicate(s) {rep_labels} ===",
              file=sys.stderr)
        rep_colors = {label: resolve_color(color_map, label, idx)
                      for idx, (_rep, label, _p) in enumerate(reps)}

        # Load each replicate's shadow calls, and take the union of genes they
        # touch -- ribo-seq depth (pooled once per sample, shared across its
        # reps) and gene_cds_len are both computed over that shared gene set.
        shadow_df_by_rep = {}
        gene_names_union = set()
        for _rep, label, parquet_path in reps:
            print(f"  Loading {label}: {parquet_path} ...", file=sys.stderr)
            sdf = load_shadow_calls_df(parquet_path)
            print(f"    {len(sdf)} reads, {sdf['shadow_gene'].nunique()} genes.", file=sys.stderr)
            shadow_df_by_rep[label] = sdf
            gene_names_union.update(sdf["shadow_gene"].unique())

        missing_gtf = sorted(g for g in gene_names_union if g not in genes)
        if missing_gtf:
            print(f"  {len(missing_gtf)} gene(s) not found in GTF, skipped: "
                  f"{missing_gtf[:10]}{'...' if len(missing_gtf) > 10 else ''}", file=sys.stderr)
        gene_names = sorted(g for g in gene_names_union if g in genes)
        gene_cds_len = {g: cds_length(genes[g]) for g in gene_names}

        bam_paths = load_ribo_bam_list(riboBamListPath, sample_name)
        if not bam_paths:
            print(f"  WARNING: no ribo-seq BAMs found for condition '{sample_name}' "
                  f"in {riboBamListPath}.", file=sys.stderr)
        print(f"  Building ribo-seq depth tracks for {len(gene_names)} gene(s) "
              f"({len(bam_paths)} BAM(s), condition={sample_name}) ...", file=sys.stderr)
        ribo_tx_depth = {}
        for gname in gene_names:
            gene = genes[gname]
            # ribo_coverage_track fetches reads within gene["gene_start"]/["gene_end"]
            # only -- those bounds are themselves ~3nt past the CDS in this GTF (same
            # annotation gap as gene_tx_to_gpos_map works around), so without widening
            # them here too, ribo_coverage_track would never even look at the flank
            # region gene_tx_to_gpos_map now has room for.
            padded_gene = dict(gene)
            padded_gene["gene_start"] = max(0, gene["gene_start"] - window_nt)
            padded_gene["gene_end"]   = gene["gene_end"] + window_nt
            ribo_gpos_depth = ribo_coverage_track(bam_paths, padded_gene, target_lengths=TARGET_LENGTHS)
            ribo_tx_depth[gname] = gene_depth_to_tx(ribo_gpos_depth, gene, flank_nt=window_nt)

        print(f"  Building shadow-call depth tracks per replicate ...", file=sys.stderr)
        shadow_tx_depth_by_rep = {}
        sites_by_gene_by_rep = {}
        for _rep, label, _p in reps:
            sdf = shadow_df_by_rep[label]
            shadow_tx_depth = {}
            sites_by_gene = {}
            for gname in gene_names:
                gene = genes[gname]
                gdf = sdf[sdf["shadow_gene"] == gname]
                # shadow_coverage_track's own default clips to gene["exons"] only --
                # widen it the same way ribo_coverage_track's fetch window is widened
                # above, else the real flank-region shadow calls the HMM fix now
                # scores would get silently clipped away right back to a wall here.
                shadow_gpos_depth = shadow_coverage_track(gdf, gname, gene, shadow_cutoff, min_run_nt,
                                                          flank_5p=window_nt, flank_3p=window_nt)
                shadow_tx_depth[gname] = gene_depth_to_tx(shadow_gpos_depth, gene, flank_nt=window_nt)
                # raw per-site P_B, no run thresholding -- the "by position" figures'
                # own data source, independent of shadow_cutoff/min_run_nt entirely
                sites_by_gene[gname] = gene_bypos_scores(gdf, gene, flank_nt=window_nt)
            shadow_tx_depth_by_rep[label] = shadow_tx_depth
            sites_by_gene_by_rep[label] = sites_by_gene

        print(f"  Building His-codon metagene (window=+/-{window_nt}nt) ...", file=sys.stderr)
        his_ribo_acc = build_his_density_metagene(ribo_tx_depth, his_tx_by_gene, window_nt)
        n_genes_his_ribo = sum(1 for g, d in ribo_tx_depth.items() if d and his_tx_by_gene.get(g))
        his_shadow_acc_by_rep = {}
        n_genes_his_shadow_by_rep = {}
        for label, tx_depth in shadow_tx_depth_by_rep.items():
            his_shadow_acc_by_rep[label] = build_his_density_metagene(tx_depth, his_tx_by_gene, window_nt)
            n_genes_his_shadow_by_rep[label] = sum(1 for g, d in tx_depth.items()
                                                   if d and his_tx_by_gene.get(g))
        plot_his_metagene(his_shadow_acc_by_rep, his_ribo_acc,
                          f"{outPrefix}.{sample_name}.his_metagene.pdf", window_nt,
                          rep_colors=rep_colors, n_genes_shadow_by_rep=n_genes_his_shadow_by_rep,
                          n_genes_ribo=n_genes_his_ribo)

        print(f"  Building start/stop-codon metagene (window=+/-{window_nt}nt) ...", file=sys.stderr)
        start_ribo_acc, stop_ribo_acc = build_startstop_density_metagene(
            ribo_tx_depth, gene_cds_len, window_nt)
        n_genes_ribo = sum(1 for d in ribo_tx_depth.values() if d)
        start_shadow_acc_by_rep = {}
        stop_shadow_acc_by_rep = {}
        n_genes_shadow_by_rep = {}
        for label, tx_depth in shadow_tx_depth_by_rep.items():
            start_acc, stop_acc = build_startstop_density_metagene(tx_depth, gene_cds_len, window_nt)
            start_shadow_acc_by_rep[label] = start_acc
            stop_shadow_acc_by_rep[label]  = stop_acc
            n_genes_shadow_by_rep[label] = sum(1 for d in tx_depth.values() if d)
        plot_startstop_metagene(start_shadow_acc_by_rep, stop_shadow_acc_by_rep,
                                start_ribo_acc, stop_ribo_acc,
                                f"{outPrefix}.{sample_name}.startstop_metagene.pdf", window_nt,
                                rep_colors=rep_colors, n_genes_shadow_by_rep=n_genes_shadow_by_rep,
                                n_genes_ribo=n_genes_ribo)
        all_start_shadow_acc_by_rep.update(start_shadow_acc_by_rep)
        all_stop_shadow_acc_by_rep.update(stop_shadow_acc_by_rep)
        all_n_genes_shadow_by_rep.update(n_genes_shadow_by_rep)
        all_start_ribo_acc_by_sample[sample_name] = start_ribo_acc
        all_stop_ribo_acc_by_sample[sample_name]  = stop_ribo_acc
        all_n_genes_ribo_by_sample[sample_name]   = n_genes_ribo

        print(f"  Building by-position (raw P_B, no run thresholding) metagenes "
              f"(window=+/-{window_nt}nt) ...", file=sys.stderr)
        his_bypos_acc_by_rep = {}
        n_genes_his_bypos_by_rep = {}
        n_sites_his_bypos_by_rep = {}
        for label, sites_by_gene in sites_by_gene_by_rep.items():
            acc = build_his_bypos_metagene(sites_by_gene, his_tx_by_gene, window_nt)
            his_bypos_acc_by_rep[label] = acc
            n_sites_his_bypos_by_rep[label] = sum(n for _s, n in acc.values())
            n_genes_his_bypos_by_rep[label] = sum(1 for g in gene_names
                                                  if his_tx_by_gene.get(g) and sites_by_gene.get(g))
        plot_his_bypos_metagene(his_bypos_acc_by_rep,
                                f"{outPrefix}.{sample_name}.his_metagene_bypos.pdf", window_nt,
                                rep_colors=rep_colors, n_genes_by_rep=n_genes_his_bypos_by_rep,
                                n_sites_by_rep=n_sites_his_bypos_by_rep)

        start_bypos_acc_by_rep = {}
        stop_bypos_acc_by_rep = {}
        n_genes_bypos_by_rep = {}
        n_sites_start_bypos_by_rep = {}
        n_sites_stop_bypos_by_rep = {}
        for label, sites_by_gene in sites_by_gene_by_rep.items():
            start_acc, stop_acc = build_startstop_bypos_metagene(sites_by_gene, gene_cds_len, window_nt)
            start_bypos_acc_by_rep[label] = start_acc
            stop_bypos_acc_by_rep[label]  = stop_acc
            n_sites_start_bypos_by_rep[label] = sum(n for _s, n in start_acc.values())
            n_sites_stop_bypos_by_rep[label]  = sum(n for _s, n in stop_acc.values())
            n_genes_bypos_by_rep[label] = sum(1 for g in gene_names if sites_by_gene.get(g))
        plot_startstop_bypos_metagene(start_bypos_acc_by_rep, stop_bypos_acc_by_rep,
                                      f"{outPrefix}.{sample_name}.startstop_metagene_bypos.pdf", window_nt,
                                      rep_colors=rep_colors, n_genes_by_rep=n_genes_bypos_by_rep,
                                      n_sites_start_by_rep=n_sites_start_bypos_by_rep,
                                      n_sites_stop_by_rep=n_sites_stop_bypos_by_rep)
        all_start_bypos_acc_by_rep.update(start_bypos_acc_by_rep)
        all_stop_bypos_acc_by_rep.update(stop_bypos_acc_by_rep)
        all_n_genes_bypos_by_rep.update(n_genes_bypos_by_rep)
        all_n_sites_start_bypos_by_rep.update(n_sites_start_bypos_by_rep)
        all_n_sites_stop_bypos_by_rep.update(n_sites_stop_bypos_by_rep)

    # Cross-sample overlay figures: every replicate across every sample
    # overlaid on the same Start | Stop panels, color-coded per replicate
    # (global_rep_colors) -- plus, for the shadow-call+ribo-seq figure, one
    # grey-shaded ribo-seq curve per sample (ribo-seq depth is pooled per
    # sample/condition, not per replicate). Nothing here is recomputed --
    # just the SAME per-rep/per-sample accumulators each per-sample figure
    # above already built, collected across the whole run (see
    # plot_startstop_metagene_cross_sample / module docstring).
    sample_names = sorted(samples_to_libs.keys())
    if len(sample_names) > 1:
        print(f"\n=== Cross-sample overlay: {len(sample_names)} sample(s) {sample_names} ===",
              file=sys.stderr)
        ribo_grey_by_sample = dict(zip(sample_names, _ribo_grey_palette(color_map, len(sample_names))))

        plot_startstop_metagene_cross_sample(
            all_start_shadow_acc_by_rep, all_stop_shadow_acc_by_rep,
            all_start_ribo_acc_by_sample, all_stop_ribo_acc_by_sample,
            f"{outPrefix}.allSamples.startstop_metagene.pdf", window_nt,
            rep_colors=global_rep_colors, ribo_colors_by_sample=ribo_grey_by_sample,
            n_genes_shadow_by_rep=all_n_genes_shadow_by_rep,
            n_genes_ribo_by_sample=all_n_genes_ribo_by_sample)

        # plot_startstop_bypos_metagene is already generic "by replicate" and
        # has no ribo-seq panel to begin with -- every replicate across every
        # sample slots in directly, no dedicated cross-sample function needed.
        plot_startstop_bypos_metagene(
            all_start_bypos_acc_by_rep, all_stop_bypos_acc_by_rep,
            f"{outPrefix}.allSamples.startstop_metagene_bypos.pdf", window_nt,
            rep_colors=global_rep_colors, n_genes_by_rep=all_n_genes_bypos_by_rep,
            n_sites_start_by_rep=all_n_sites_start_bypos_by_rep,
            n_sites_stop_by_rep=all_n_sites_stop_bypos_by_rep)


if __name__ == "__main__":
    main(sys.argv[1:])
