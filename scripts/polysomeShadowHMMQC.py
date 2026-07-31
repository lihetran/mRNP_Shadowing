"""
shadowSizeQC.py — Liam Tran, July 2026

Assess protected-region ("shadow") sizes from HMM shadow-call parquets produced
by bayesianShadowClassifier.py (posterior P_B schema, native list columns).

Two complementary views:
  VIEW 1  footprint SIZE at a fixed P_B cutoff, unmerged (one break per
          sub-cutoff site) -> "how big are footprints?"
  VIEW 2  stringency SWEEP across P_B cutoffs (unmerged, padded length)
          -> "how does calling stringency reshape what I detect?"
  VIEW 3  per-gene call RATE, overall and restricted to His codon sites
          (via findHisCodonPositions.py's cache)
  VIEW 7  shadow DENSITY (shadows / nt / read) by gene region -- UTR5 vs
          CDS vs UTR3 -- so regions/genes of different sizes AND different
          sequencing depth are directly comparable, and so is one region
          across libraries.

Input parquet columns used per read:
  read_id, shadow_gene, absolute_indices, shadow_gpos, shadow_P_B,
  shadow_region, n_sites_utr5, n_sites_cds, n_sites_utr3
  (shadow_P_B/shadow_gpos/shadow_region/n_sites_* cover EVERY scored Ref=A
  site for the read, not just ones above some cutoff -- write_shadow_calls_to_df
  stopped pre-filtering to P_B>=threshold at the source, so every view in
  this module applies whatever cutoff it needs itself, e.g. via PROB_CUTOFFS
  or FIXED_CUTOFF.)

Run:
  python3 shadowSizeQC.py inFilesParquet.txt outPrefix hisCodonPositions.pickle gtfFile
where inFilesParquet.txt is line-delimited:  fileName  rep  parquetFile,
hisCodonPositions.pickle is findHisCodonPositions.py's output, and gtfFile
is the GTF used to derive each gene's UTR5/CDS/UTR3 lengths for View 7.
"""

import sys, math, collections, pickle
import numpy as np
import pandas as pd

from runHMMPerGene import parse_gtf, cds_length
# NOTE: parse_gtf/cds_length are imported rather than duplicated a third
# time (trainHMMPerGene.py already imports _forward_backward_hsmm/
# _duration_pmf_default from runHMMPerGene.py the same way) -- these are
# the exact same GTF-parsing logic verbatim in both other scripts, with no
# per-script variation, unlike e.g. _gpos_to_tx_map which each script keeps
# its own copy of because it differs slightly per use case.

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
MIN_RUNS_PER_GENE  = 0                        # drop a gene from the size-shape
                                               # aggregate below this many runs
MIN_SITES_PER_GENE = 20                       # drop a gene from the call-rate
                                               # summary below this many scored sites
MIN_HIS_OBS_PER_GENE = 1                      # drop a gene from the His-codon
                                               # P_B distribution below this many His sites
MIN_HIS_READS_PER_GENE = 5                    # drop a gene from the His-shadow-rate (View 6)
                                               # below this many reads with >=1 His-codon site
MIN_READS_FOR_RUN_RATE = 20                   # View 9: drop a gene from the runs-per-read
                                               # rate below this many reads (see caveat in
                                               # _gene_run_counts_by_reads about what "reads"
                                               # means here)
MIN_RUN_NT     = 25                           # View 7: a "shadow" is a protected run of at
                                               # least this many genomic nt (footprint-sized),
                                               # not a bare scored site -- an isolated 1-3nt
                                               # blip above P_B>=FIXED_CUTOFF isn't the same
                                               # evidence as an actual footprint-length stretch
HIS_CENTER_FRAC = 0.5                          # View 8: a His codon counts as "centering" a
                                               # run if it falls within this fraction of the
                                               # run's own span, centered on the run's midpoint
                                               # (0.5 = middle half of the run) -- scales with
                                               # run length rather than a fixed nt tolerance, so
                                               # a longer run gets proportionally more slack
HIS_PB_BIN     = 0.05                          # bin width for the His-codon P_B distribution
HIS_PB_RANGE   = (0.0, 1.0)                   # full P_B range -- shadow_P_B is no longer
                                               # pre-filtered at the source (see
                                               # write_shadow_calls_to_df), so values below
                                               # 0.5 are real and would be clipped by a
                                               # narrower range here
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
    Unmerged protected runs (P_B >= cutoff), one break per sub-cutoff site.
    Each run: read_id, gene, aligned_len (padded), genomic_nt (unpadded),
    n_sites, region, gpos_lo, gpos_hi. Strand-safe; drops NaN positions.

    region: majority shadow_region among the run's sites -- a run CAN
    straddle a UTR/CDS boundary at its edges (rare, since scored sites are
    dense relative to region boundaries, but possible), so this is a
    majority vote among the run's own sites, not guaranteed unanimous.

    gpos_lo/gpos_hi: the run's own genomic span (unpadded, min/max of its
    member sites' gpos, minus-strand safe) -- e.g. for testing whether
    something (a His codon, ...) sits near the run's own midpoint, not the
    padded aligned_len window (which pads outward for a different purpose).
    """
    halfN = N // 2
    runs = []
    for row in df.itertuples(index=False):
        pb, gpos, ai = _read_arrays(row)
        regions = list(row.shadow_region)
        n_sh = len(pb)
        if n_sh == 0:
            continue
        absIndices = list(ai)
        numPos = len(absIndices)
        g2r = {int(g): idx for idx, g in enumerate(absIndices)
               if g is not None and g == g}          # drop NaN, int keys

        i = 0
        while i < n_sh:
            if pb[i] >= probCutOff:
                j = i
                while j < n_sh and pb[j] >= probCutOff:
                    j += 1
                fr, lr = g2r.get(gpos[i]), g2r.get(gpos[j-1])
                if fr is not None and lr is not None:
                    lo, hi = sorted((fr, lr))          # minus-strand safe
                    startI = max(0, lo - halfN)
                    endI   = min(numPos, hi + halfN + 1)
                    run_regions = regions[i:j]
                    region = (collections.Counter(run_regions).most_common(1)[0][0]
                              if run_regions else None)
                    run_gpos = gpos[i:j]
                    runs.append({
                        "read_id":     row.read_id,
                        "gene":        row.shadow_gene,
                        "aligned_len": endI - startI,               # padded
                        "genomic_nt":  abs(gpos[j-1] - gpos[i]) + 1, # unpadded
                        "n_sites":     j - i,
                        "region":      region,
                        "gpos_lo":     min(run_gpos),
                        "gpos_hi":     max(run_gpos),
                    })
                i = j
            else:
                i += 1
    return runs


# ─────────────────────────────────────────────────────────────────────────
# His codon call rate (findHisCodonPositions.py pickle as input)
# ─────────────────────────────────────────────────────────────────────────
def load_his_codon_gpos(pickle_path):
    """
    Load findHisCodonPositions.py's {gene: [tx_positions, gpos_positions]}
    cache and keep just the genomic side, {gene: set(gpos)} -- genomic space
    is the only coordinate system scored sites here are compared against
    (shadow_gpos, not shadow_tx_pos). A set, not a list, since this is now
    an exact-position membership test (is this scored site a His codon?),
    not an interval overlap.
    """
    with open(pickle_path, "rb") as f:
        his_positions = pickle.load(f)
    return {gene: set(gpos) for gene, (_tx_pos, gpos) in his_positions.items()}


def his_codon_pb_observations(df, his_gpos_by_gene):
    """
    Every (read, His-codon site) observation's raw P_B value, tagged by
    gene -- no cutoff applied. Feed into _gene_weighted_freq_float for a
    per-gene-then-aggregated P_B DISTRIBUTION at His codons.

    This replaces an earlier version (his_codon_call_rates) that collapsed
    this to a single P_B>0.7 rate. Confirmed on real data that a fixed
    cutoff can sit awkwardly relative to one specific site's own achievable
    range: one gene's His-codon site never exceeded P_B=0.64 in ANY
    library, reading as "0% protected" even though that's a meaningfully
    elevated value for that site. The full distribution shows that shape
    instead of hiding it behind one line.

    Returns a list of {"gene": gene, "pb": p} dicts.
    """
    obs = []
    for row in df.itertuples(index=False):
        gene = row.shadow_gene
        his_set = his_gpos_by_gene.get(gene)
        if not his_set:
            continue
        pb, gpos, _ai = _read_arrays(row)
        for p, g in zip(pb, gpos):
            if g in his_set:
                obs.append({"gene": gene, "pb": p})
    return obs


def his_codon_pb_per_gene(df, his_gpos_by_gene, min_obs_per_gene=MIN_HIS_OBS_PER_GENE):
    """
    Per-gene MEAN P_B at His-codon sites (pooling all reads/sites for that
    gene). This is the paired counterpart to his_codon_pb_observations: that
    function pools every gene's raw P_B values into one marginal
    distribution, which can hide a real, consistent per-gene shift between
    libraries when baseline P_B varies a lot gene to gene (confirmed on
    real HSMM output: 9/11 genes shared across -3AT/+3AT/phenol show
    +3AT > -3AT at their own His-codon sites, individually reproducible,
    despite gene baselines ranging ~0.70-0.92 -- wide enough to swamp that
    shift in the pooled marginal view). Compare this per-gene, per-library
    to see the paired effect directly (see plot_gene_call_rates's
    connect_matched option).

    Returns {gene: mean_pb}.
    """
    obs = his_codon_pb_observations(df, his_gpos_by_gene)
    by_gene = collections.defaultdict(list)
    for o in obs:
        by_gene[o["gene"]].append(o["pb"])
    return {g: float(np.mean(vals)) for g, vals in by_gene.items()
            if len(vals) >= min_obs_per_gene}


def _gene_site_counts_his_split(df, his_gpos_by_gene, cutoff=FIXED_CUTOFF):
    """
    Site-level shadow COUNTS (P_B > cutoff), split per gene into His-codon
    sites vs every OTHER scored site -- a "shadow" here is a single scored
    site with P_B > cutoff, same definition gene_call_rates uses.

    Also counts, per gene, the number of READS that have >=1 His-codon
    scored site at all ("his_read_tot") -- a per-read opportunity count for
    normalizing the His-shadow rate. n_his_shadows / n_other_shadows (the
    original View 6 ratio) divides by a count that itself collapses toward
    zero in a library with little protection overall (e.g. a ribosome-less
    control): with very few other-codon shadows, even 1-2 His shadows blow
    the ratio up, which reads as "His-codon enrichment" but is really just
    "this library barely has any shadows at all." Dividing by reads that
    had the opportunity to show a His-codon shadow instead gives a rate
    that isn't distorted by how protected the rest of the gene happens to
    be in that particular library.

    Returns (his_counts, other_counts, his_read_tot):
      his_counts, other_counts: {gene: (n_prot, n_tot)} as before -- n_tot
        kept for pairing genes on sample size before dividing.
      his_read_tot: {gene: n_reads} -- reads with >=1 His-codon scored site.
    """
    his_prot = collections.defaultdict(int); his_tot = collections.defaultdict(int)
    oth_prot = collections.defaultdict(int); oth_tot = collections.defaultdict(int)
    his_read_tot = collections.defaultdict(int)
    for row in df.itertuples(index=False):
        gene = row.shadow_gene
        his_set = his_gpos_by_gene.get(gene)
        pb, gpos, _ai = _read_arrays(row)
        read_has_his_site = False
        for p, g in zip(pb, gpos):
            if his_set is not None and g in his_set:
                his_tot[gene] += 1
                read_has_his_site = True
                if p > cutoff:
                    his_prot[gene] += 1
            else:
                oth_tot[gene] += 1
                if p > cutoff:
                    oth_prot[gene] += 1
        if read_has_his_site:
            his_read_tot[gene] += 1
    his_counts = {g: (his_prot[g], his_tot[g]) for g in his_tot}
    oth_counts = {g: (oth_prot[g], oth_tot[g]) for g in oth_tot}
    return his_counts, oth_counts, dict(his_read_tot)


def region_lengths(gene):
    """
    (utr5_len, cds_len, utr3_len) in nt, from a parse_gtf gene dict.
    cds_length() already exists in runHMMPerGene.py; UTR lengths are the
    same sum-of-interval-spans, just no dedicated helper existed yet.
    """
    utr5_len = sum(e - s for s, e in gene.get("utr5", []))
    cds_len  = cds_length(gene)
    utr3_len = sum(e - s for s, e in gene.get("utr3", []))
    return utr5_len, cds_len, utr3_len


def _gene_region_run_counts(runs, min_genomic_nt=MIN_RUN_NT):
    """
    Per-gene count of qualifying protected RUNS, split by region -- a
    "shadow" for View 7 means a footprint-sized run (already thresholded at
    whatever probCutOff `runs` was extracted with, e.g. FIXED_CUTOFF, plus
    genomic_nt >= min_genomic_nt here), not a bare scored site. An isolated
    1-3nt blip above the P_B cutoff with no length support isn't the same
    evidence as an actual ~30nt protected stretch, which is why this counts
    RUNS rather than reusing the site-level n_sites_utr5/cds/utr3 counts
    (those don't carry any length information at all -- superseded by this).

    runs: output of extract_shadow_runs (each dict has "gene", "region",
    "genomic_nt"). Returns {gene: {"UTR5": n, "CDS": n, "UTR3": n}} --
    only region keys that actually had >=1 qualifying run are present.
    """
    counts = collections.defaultdict(lambda: collections.defaultdict(int))
    for r in runs:
        if r["genomic_nt"] < min_genomic_nt or r["region"] is None:
            continue
        counts[r["gene"]][r["region"]] += 1
    return {g: dict(regs) for g, regs in counts.items()}


_REGION_TO_NSITES_COL = {"UTR5": "n_sites_utr5", "CDS": "n_sites_cds", "UTR3": "n_sites_utr3"}


def _gene_region_read_coverage(df):
    """
    Per-gene, per-region READ coverage: {gene: {"UTR5": n, "CDS": n,
    "UTR3": n}} -- n = number of reads with >=1 scored site in that region
    (n_sites_<region> > 0). This is the depth term View 7's shadow-density
    normalization needs alongside region length: density = n_qualifying_runs
    / (region_length_nt * n_reads_covering_region), the same length+depth
    double normalization RNA-seq RPKM uses. Region length alone doesn't
    correct for a region simply being sampled by fewer reads/molecules --
    that would look like "less shadowed" purely from lower depth, not a
    real biological difference.

    Only meaningful now that write_shadow_calls_to_df stopped pre-filtering
    to P_B>=threshold: n_sites_<region> reflects EVERY scored site in that
    region for the read, not just ones that happened to call as protected
    (see module docstring) -- so ">0" genuinely means "this read had
    coverage here," not "this read happened to show a shadow here."

    df: the shadow_calls dataframe (one row per read). Returns only region
    keys with >=1 covering read present, same convention as
    _gene_region_run_counts.
    """
    coverage = collections.defaultdict(lambda: collections.defaultdict(int))
    for row in df.itertuples(index=False):
        gene = row.shadow_gene
        for region, col in _REGION_TO_NSITES_COL.items():
            if getattr(row, col) > 0:
                coverage[gene][region] += 1
    return {g: dict(regs) for g, regs in coverage.items()}


def _gene_run_counts_by_reads(df, runs, min_genomic_nt=MIN_RUN_NT):
    """
    Raw ingredients for a per-gene "shadows per read" rate: {gene:
    (n_qualifying_runs, n_reads)}. n_qualifying_runs = protected RUNS with
    genomic_nt >= min_genomic_nt (same footprint-size floor as View 7's
    _gene_region_run_counts, just not split by region), pooled across all
    reads in the gene. n_reads = every read observed for that gene in df --
    now the TRUE per-gene read count, since write_shadow_calls_to_df no
    longer drops reads with zero above-threshold sites (previously this
    undercounted, the same blind spot _gene_site_counts' n_tot used to
    have -- both are fixed by the same upstream change).

    df: the same shadow_calls dataframe runs was extracted from (for read
    counts). runs: output of extract_shadow_runs (each dict has "gene",
    "genomic_nt").
    """
    n_reads = collections.defaultdict(int)
    for row in df.itertuples(index=False):
        n_reads[row.shadow_gene] += 1

    n_runs = collections.defaultdict(int)
    for r in runs:
        if r["genomic_nt"] >= min_genomic_nt:
            n_runs[r["gene"]] += 1

    return {g: (n_runs.get(g, 0), n_reads[g]) for g in n_reads}


def _gene_his_centered_run_counts(runs, his_gpos_by_gene,
                                  min_genomic_nt=MIN_RUN_NT,
                                  center_frac=HIS_CENTER_FRAC):
    """
    Per-gene count of qualifying protected RUNS (genomic_nt >=
    min_genomic_nt, same footprint-size floor as View 7) that are CENTERED
    on a His codon -- not just overlapping one anywhere in the run, but
    with a His-codon position falling within the middle `center_frac`
    fraction of the run's own span. This is a stricter, more specific test
    than View 6 (n_his_shadows / n_reads_with_a_His_site): a run merely
    overlapping a His codon near one edge could be coincidental, whereas a
    run genuinely centered on it is what you'd expect if the His codon
    itself is what triggered the stall/protection, rather than the His
    codon just happening to sit inside a protected region caused by
    something else nearby.

    runs: output of extract_shadow_runs (each dict has "gene", "genomic_nt",
    "gpos_lo", "gpos_hi"). Returns {gene: n_centered_runs} -- only genes
    with >=1 centered run are present (mirrors _gene_region_run_counts).
    """
    counts = collections.defaultdict(int)
    for r in runs:
        if r["genomic_nt"] < min_genomic_nt:
            continue
        gene = r["gene"]
        his_set = his_gpos_by_gene.get(gene)
        if not his_set:
            continue
        lo, hi = r["gpos_lo"], r["gpos_hi"]
        midpoint = (lo + hi) / 2.0
        tolerance = center_frac * (hi - lo) / 2.0
        if any(lo <= h <= hi and abs(h - midpoint) <= tolerance for h in his_set):
            counts[gene] += 1
    return dict(counts)


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


def _bin_float(values, lo, hi, bin_width):
    """
    Like _bin, but for float-valued data (P_B) -- keyed by integer bin
    INDEX rather than the float left-edge. range()/repeated-addition on
    floats can produce edge values that don't hash-equal themselves later
    (0.5 + 0.05*3 isn't guaranteed to == 0.65), which would silently break
    dict lookups downstream; integer indices sidestep that entirely.
    """
    n_bins = int(round((hi - lo) / bin_width))
    counts = collections.defaultdict(int)
    for v in values:
        if lo <= v <= hi:
            idx = min(int((v - lo) / bin_width), n_bins - 1)
            counts[idx] += 1
    edges = [lo + i * bin_width for i in range(n_bins + 1)]
    return dict(counts), edges


def _gene_weighted_freq_float(obs, measure, lo, hi, bin_width,
                              min_obs_per_gene=MIN_HIS_OBS_PER_GENE):
    """
    Float-valued counterpart to _gene_weighted_freq (same per-gene-then-
    aggregate rationale -- every gene one equal vote regardless of depth),
    for continuous measures like raw P_B rather than integer-nt sizes.

    obs: list of {"gene": ..., measure: float_value}.
    Returns (freq_by_bin_index, edges, n_genes_used, n_genes_dropped).
    """
    by_gene = collections.defaultdict(list)
    for o in obs:
        by_gene[o["gene"]].append(o[measure])

    n_bins = int(round((hi - lo) / bin_width))
    freq_sum = {i: 0.0 for i in range(n_bins)}

    n_used = 0
    for gene, vals in by_gene.items():
        if len(vals) < min_obs_per_gene:
            continue
        counts, edges = _bin_float(vals, lo, hi, bin_width)
        tot = sum(counts.values())
        if tot == 0:
            continue
        n_used += 1
        for i in range(n_bins):
            freq_sum[i] += counts.get(i, 0) / tot

    n_dropped = len(by_gene) - n_used
    edges = [lo + i * bin_width for i in range(n_bins + 1)]
    if n_used == 0:
        return {i: 0.0 for i in range(n_bins)}, edges, 0, n_dropped
    return {i: v / n_used for i, v in freq_sum.items()}, edges, n_used, n_dropped


def _count_by_gene(items):
    """{gene: count} -- generic counter over any list of dicts with a
    "gene" key (shadow runs, His-codon observations, ...). Used to decide
    which genes qualify for a paired cross-library comparison (see
    _common_genes)."""
    c = collections.defaultdict(int)
    for it in items:
        c[it["gene"]] += 1
    return dict(c)


def _common_genes(counts_by_lib, min_n):
    """
    counts_by_lib: {libID: {gene: n}}, n = however many qualifying
    observations that gene has in that library (runs, scored sites, or
    His-codon observations, depending on caller).

    Returns the set of genes with >= min_n in EVERY library. This is the
    ONE pairing rule every per-gene comparison in this module applies: a
    gene only counts toward a cross-library plot if it clears that plot's
    minimum-observation bar in ALL libraries being compared, not just one.
    Comparing each library's own full passing-gene set directly is
    misleading -- confirmed on real data: an unrestricted per-library
    comparison made a 42-gene library look higher overall than an 11-gene
    library, but the 11 genes actually shared between them mostly moved
    the OTHER way once compared like-for-like.
    """
    if not counts_by_lib:
        return set()
    qualifying = [{g for g, n in counts.items() if n >= min_n}
                  for counts in counts_by_lib.values()]
    return set.intersection(*qualifying)


def _gene_site_counts(df, cutoff):
    """
    Raw ingredients for a per-gene protected-site rate: {gene: (n_prot,
    n_tot)}, pooling all reads within each gene. Kept separate from the
    rate itself (gene_call_rates) so a caller can intersect qualifying
    genes across libraries BEFORE dividing -- computing the rate first and
    filtering after would still be fine numerically, but every other
    per-gene metric in this module follows raw-then-filter-then-aggregate,
    so this stays consistent with that.
    """
    n_prot = collections.defaultdict(int)
    n_tot  = collections.defaultdict(int)
    for row in df.itertuples(index=False):
        pb, _gpos, _ai = _read_arrays(row)
        gene = row.shadow_gene
        n_tot[gene]  += len(pb)
        n_prot[gene] += sum(1 for p in pb if p > cutoff)
    return {g: (n_prot[g], n_tot[g]) for g in n_tot}


def gene_call_rates(df, cutoff, min_sites_per_gene=MIN_SITES_PER_GENE):
    """
    Per-gene fraction of scored Ref=A sites called protected (P_B > cutoff),
    pooling all reads within each gene first. This is the MAGNITUDE
    counterpart to _gene_weighted_freq's shape aggregation: it deliberately
    keeps depth-independent per-gene rates comparable side by side across
    libraries (e.g. ribosome-containing vs. a ribosome-less control), rather
    than washing that difference out the way equal-weighting the size
    histograms does on purpose.

    Standalone convenience wrapper around _gene_site_counts for a SINGLE
    library -- main()'s driver calls _gene_site_counts directly so it can
    pair genes across libraries before computing any rate at all.

    Genes with fewer than min_sites_per_gene total scored sites are dropped
    -- too few sites to trust a rate from.

    Returns {gene: rate}.
    """
    counts = _gene_site_counts(df, cutoff)
    return {g: prot / tot for g, (prot, tot) in counts.items()
            if tot >= min_sites_per_gene}


def gene_run_rates(df, runs, min_genomic_nt=MIN_RUN_NT,
                   min_reads_per_gene=MIN_READS_FOR_RUN_RATE):
    """
    Per-gene rate of qualifying protected RUNS (genomic_nt >= min_genomic_nt)
    per read, pooling all reads within each gene first. RUN-level
    counterpart to gene_call_rates' site-level rate: a bare protected-SITE
    rate doesn't distinguish one long, footprint-sized run from many short,
    likely-noise blips (see MIN_RUN_NT's rationale), which is exactly the
    distinction a false-positive-rate sanity check (e.g. on a ribosome-less
    control) needs.

    Genes with fewer than min_reads_per_gene reads are dropped -- too few
    reads to trust a rate from.

    Standalone convenience wrapper around _gene_run_counts_by_reads for a
    SINGLE library -- pair genes across libraries via _common_genes before
    computing rates if comparing multiple libraries.

    Returns {gene: rate}.
    """
    counts = _gene_run_counts_by_reads(df, runs, min_genomic_nt)
    return {g: n_runs / n_reads for g, (n_runs, n_reads) in counts.items()
            if n_reads >= min_reads_per_gene}


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


def plot_gene_call_rates(rates_by_lib, pdf_path, cutoff=FIXED_CUTOFF,
                         title="Protected-site rate by library",
                         min_n=MIN_SITES_PER_GENE, min_n_label="scored sites",
                         ylabel=None, connect_matched=False):
    """
    Per-gene metric, one box+strip per library. This is the MAGNITUDE
    counterpart to the size histograms above, which equal-weight every gene
    on purpose and so wash out exactly the kind of library-to-library
    difference (e.g. ribosome-containing vs. a ribosome-less control) this
    plot is meant to show. Matplotlib rather than PyX -- a quick
    diagnostic, not a per-gene report figure.

    rates_by_lib: {libraryID: {gene: value}} -- from gene_call_rates (every
    scored A site) or his_codon_pb_per_gene (mean P_B at His-codon sites);
    title/min_n/min_n_label/ylabel just describe the plot itself.

    connect_matched: restrict EVERY library's box+strip to only the genes
    common to all libraries shown, and draw a thin line through each gene's
    point across libraries (a paired/spaghetti overlay). Comparing full
    per-library gene sets box-to-box is misleading here -- a library with
    many more passing genes than another can shift its whole box off a
    baseline-level difference in WHICH genes it includes, not a real
    per-gene effect (confirmed on real data: unrestricted boxes made a
    library with 42 genes look higher overall than one with only 11, but
    the 11 genes actually shared between them mostly moved the other way).
    Restricting to the shared set makes both the box and the lines honest
    paired comparisons of the same genes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    libIDs = sorted(rates_by_lib)

    if connect_matched:
        common_genes = sorted(set.intersection(*(set(rates_by_lib[lib]) for lib in libIDs))) \
                       if libIDs else []
        data = [[rates_by_lib[lib][g] for g in common_genes] for lib in libIDs]
    else:
        data = [list(rates_by_lib[lib].values()) for lib in libIDs]

    fig, ax = plt.subplots(figsize=(1.2 * max(len(libIDs), 3) + 1, 5))
    ax.boxplot(data, labels=libIDs, showfliers=False, zorder=3)
    rng = np.random.default_rng(0)

    if connect_matched:
        for gi in range(len(common_genes)):
            ys = [data[li][gi] for li in range(len(libIDs))]
            ax.plot(range(1, len(libIDs) + 1), ys, color="grey",
                    linewidth=0.6, alpha=0.5, zorder=1)

    for i, vals in enumerate(data, start=1):
        jitter = rng.normal(0, 0.05, size=len(vals))
        ax.scatter([i + j for j in jitter], vals, s=10, alpha=0.5,
                   color="tab:blue", zorder=2)
        ax.text(i, ax.get_ylim()[0], f"n={len(vals)}", ha="center", va="top",
                fontsize=8)
    ax.set_ylabel(ylabel or f"per-gene call rate (P$_B$ > {cutoff})")
    n_note = (f"{len(common_genes)} genes common to all libraries"
              if connect_matched else
              f"genes with <{min_n} {min_n_label} dropped")
    ax.set_title(f"{title}\n(one point per gene, {n_note})",
                  fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(pdf_path, dpi=150)
    plt.close(fig)
    print(f"Wrote gene call-rate plot to {pdf_path}", file=sys.stderr)


def plot_region_density_grouped_bars(region_density_mean, region_density_n, pdf_path,
                                     libIDs, region_names=("UTR5", "CDS", "UTR3"),
                                     ylabel="mean shadows / nt / read"):
    """
    Grouped bar chart: one group per region (UTR5/CDS/UTR3), one bar per
    library within each group -- all libraries and all regions visible on
    one plot, rather than three separate per-region figures.

    NOT gene-paired across libraries (see View 7 in main()): each bar is
    that library's own mean per-gene density among its own qualifying
    genes, so a different number of genes can back each bar (annotated as
    n= above it) -- libraries aren't held to the same shared gene set the
    way plot_gene_call_rates' connect_matched option does.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_libs    = len(libIDs)
    n_regions = len(region_names)
    x     = np.arange(n_regions)
    width = 0.8 / n_libs

    fig, ax = plt.subplots(figsize=(2.2 * n_regions + 1, 5))
    for i, lib in enumerate(libIDs):
        heights = [region_density_mean[region][lib] for region in region_names]
        ns      = [region_density_n[region][lib] for region in region_names]
        offset  = (i - (n_libs - 1) / 2) * width
        ax.bar(x + offset, heights, width, label=lib)
        for xi, h, n in zip(x + offset, heights, ns):
            ax.text(xi, h, f"n={n}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(region_names)
    ax.set_ylabel(ylabel)
    ax.set_title("Shadow density by region and library\n"
                 "(mean per-gene density, not gene-paired across libraries)",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(pdf_path), dpi=150)
    plt.close(fig)
    print(f"Wrote region shadow-density barplot to {pdf_path}", file=sys.stderr)


def plot_his_codon_pb_distribution(lib_hist, edges, pdf_path, bin_width=HIS_PB_BIN):
    """
    Per-gene-then-aggregated P_B DISTRIBUTION at His-codon sites, one line
    per library -- see his_codon_pb_observations for why this replaced a
    single P_B>cutoff rate. A vertical dashed line at 0.7 is drawn for
    visual reference only, not as a hard cutoff baked into the data.

    lib_hist: {libraryID: (freq_by_bin_index, n_obs)}.
    """
    if canvas is None:
        print("pyx not available; skipping plot", file=sys.stderr); return
    libIDs  = sorted(lib_hist)
    lefts   = edges[:-1]
    centers = [e + bin_width / 2. for e in lefts]

    ymax = 0.0
    for freq, _n in lib_hist.values():
        for i in range(len(lefts)):
            ymax = max(ymax, freq.get(i, 0))
    ymax = (ymax or 1) * 1.15

    g = graph.graphxy(
        width=10, height=6, xpos=0, ypos=0,
        x=graph.axis.linear(min=edges[0], max=edges[-1], title=r"P$_B$ at His codon"),
        y=graph.axis.linear(min=0, max=ymax, parter=_nice(ymax),
                            title="freq (per-gene-then-aggregated)"),
        key=graph.key.key(pos="tr", hinside=0))
    g.plot(graph.data.points([(0.7, 0), (0.7, ymax)], x=1, y=2, title=None),
           [graph.style.line([color.cmyk(0, 0, 0, 0.5), style.linewidth.thin,
                              style.linestyle.dashed])])
    for i, libID in enumerate(libIDs):
        freq, n = lib_hist.get(libID, ({}, 0))
        title = r"%s (n=%d)" % (libID.replace("_", r"\_"), n)
        g.plot(graph.data.points([(ctr, freq.get(k, 0))
                                  for k, ctr in enumerate(centers)],
                                 x=1, y=2, title=title),
               [graph.style.line([_libcolor(i), style.linewidth.Thick])])
    c = canvas.canvas()
    c.insert(g)
    c.writePDFfile(str(pdf_path))
    print(f"Wrote His-codon P_B distribution plot to {pdf_path}", file=sys.stderr)




# ─────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────
def extract_library_raw(parquetFile, libraryID, his_gpos_by_gene):
    """
    Load one library and extract RAW per-gene ingredients only -- no
    cross-library aggregation happens here. Every per-gene comparison in
    this module is now PAIRED (see _common_genes): a gene only counts
    toward any plot if it clears that plot's own minimum-observation bar in
    EVERY library being compared, not just this one. That decision needs
    visibility across every library at once, so it happens in main() after
    every library's raw data has been collected by this function.

    Returns {"runs_by_cut": {cutoff: [run dicts]}, "site_counts":
    {gene: (n_prot, n_tot)} at FIXED_CUTOFF, "his_obs": [{"gene","pb"}],
    "his_shadow_counts"/"other_shadow_counts": {gene: (n_prot, n_tot)} --
    scored sites (at FIXED_CUTOFF) split into His-codon sites vs every
    other site, n_prot = count of those sites with P_B>cutoff,
    "his_read_totals": {gene: n_reads} -- reads with >=1 His-codon site,
    the per-read denominator for the His-shadow rate (see
    _gene_site_counts_his_split), "region_run_counts": {gene: {"UTR5": n,
    "CDS": n, "UTR3": n}} -- count of qualifying (>=MIN_RUN_NT nt) protected
    RUNS by region at FIXED_CUTOFF (see _gene_region_run_counts), the
    numerator for View 7's density, "region_read_coverage": {gene: {"UTR5":
    n, "CDS": n, "UTR3": n}} -- count of reads with >=1 scored site in that
    region (see _gene_region_read_coverage), the depth term for View 7's
    density denominator alongside region length, "genes_observed": set of
    every gene with >=1 scored site in this library, the population View 7
    divides over so a gene with no qualifying run still counts as a true 0,
    "his_centered_run_counts": {gene: n} -- count of qualifying runs
    CENTERED on a His codon (see _gene_his_centered_run_counts), the
    numerator for View 8's stricter His-codon rate}.
    """
    print("Loading %s (%s)..." % (parquetFile, libraryID), file=sys.stderr)
    df = pd.read_parquet(parquetFile)          # native columns, no JSON decode

    runs_by_cut = {cut: extract_shadow_runs(df, cut, N_PAD) for cut in PROB_CUTOFFS}

    if FIXED_CUTOFF in runs_by_cut:
        sizes = [r[SIZE_MEASURE] for r in runs_by_cut[FIXED_CUTOFF]]
        if sizes:
            print("    P_B>=%s: n=%d median=%.0f IQR=[%.0f,%.0f] (this library alone, "
                  "before pairing across libraries)"
                  % (FIXED_CUTOFF, len(sizes), np.median(sizes),
                     np.percentile(sizes, 25), np.percentile(sizes, 75)), file=sys.stderr)

    site_counts = _gene_site_counts(df, FIXED_CUTOFF)
    his_obs = his_codon_pb_observations(df, his_gpos_by_gene)
    his_shadow_counts, other_shadow_counts, his_read_totals = _gene_site_counts_his_split(
        df, his_gpos_by_gene, FIXED_CUTOFF)
    region_run_counts = _gene_region_run_counts(runs_by_cut.get(FIXED_CUTOFF, []),
                                                MIN_RUN_NT)
    region_read_coverage = _gene_region_read_coverage(df)
    his_centered_run_counts = _gene_his_centered_run_counts(
        runs_by_cut.get(FIXED_CUTOFF, []), his_gpos_by_gene, MIN_RUN_NT, HIS_CENTER_FRAC)
    # every gene with ANY scored site in this library -- the population
    # View 7 iterates over, so a gene with data but no run reaching
    # MIN_RUN_NT still contributes a true 0 rather than being silently
    # dropped (region_run_counts only lists genes with >=1 qualifying run).
    genes_observed = set(df["shadow_gene"].unique())

    return {"runs_by_cut": runs_by_cut, "site_counts": site_counts,
            "his_obs": his_obs, "his_shadow_counts": his_shadow_counts,
            "other_shadow_counts": other_shadow_counts,
            "region_read_coverage": region_read_coverage,
            "his_read_totals": his_read_totals,
            "region_run_counts": region_run_counts,
            "genes_observed": genes_observed,
            "his_centered_run_counts": his_centered_run_counts}


def main(args):
    global PALETTE
    if canvas is not None:
        PALETTE = [color.cmyk(1, 0.5, 0, 0), color.cmyk(0, 1, 1, 0),
                   color.cmyk(0.4, 1, 0, 0), color.cmyk(1, 0, 1, 0.1),
                   color.cmyk(0, 0.5, 1, 0), color.cmyk(0.7, 0, 0, 0),
                   color.cmyk(0, 0, 0, 0.7), color.cmyk(0.3, 0, 1, 0.2)]

    parquetList, outPrefix, hisPicklePath, gtfPath = args[0], args[1], args[2], args[3]
    his_gpos_by_gene = load_his_codon_gpos(hisPicklePath)
    print("Loaded His codon positions for %d genes." % len(his_gpos_by_gene),
          file=sys.stderr)

    genes = parse_gtf(gtfPath)
    region_len_by_gene = {g: region_lengths(gene) for g, gene in genes.items()}
    print("Loaded region lengths for %d genes from GTF." % len(region_len_by_gene),
          file=sys.stderr)

    raw_by_lib = {}
    with open(parquetList) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            fileName, rep, parquetFile = parts[0], parts[1], parts[2]
            libraryID = "%s-%s" % (fileName, rep)
            raw_by_lib[libraryID] = extract_library_raw(parquetFile, libraryID, his_gpos_by_gene)

    if not raw_by_lib:
        print("no libraries processed", file=sys.stderr); return

    libIDs = sorted(raw_by_lib)
    print("Pairing across %d libraries and building combined figures..." % len(libIDs),
          file=sys.stderr)

    # ---- View 1 / 1-supplement: footprint size, paired per cutoff ----
    # A gene only contributes to a cutoff's panel if it has >=MIN_RUNS_PER_GENE
    # runs in EVERY library shown (not just the one it's plotted from).
    unmerged_by_cut        = collections.defaultdict(dict)
    unmerged_counts_by_cut = collections.defaultdict(dict)
    fp_edges = None
    for cut in PROB_CUTOFFS:
        counts_by_lib = {lib: _count_by_gene(raw_by_lib[lib]["runs_by_cut"][cut])
                         for lib in libIDs}
        common = _common_genes(counts_by_lib, MIN_RUNS_PER_GENE)
        print("    cutoff %.2f: %d genes common to all %d libraries (>=%d runs each)"
              % (cut, len(common), len(libIDs), MIN_RUNS_PER_GENE), file=sys.stderr)
        for lib in libIDs:
            filtered = [r for r in raw_by_lib[lib]["runs_by_cut"][cut]
                       if r["gene"] in common]
            sizes = [r[SIZE_MEASURE] for r in filtered]
            freq, fp_edges, _, _ = _gene_weighted_freq(
                filtered, SIZE_MEASURE, SIZE_RANGE[0], SIZE_RANGE[1], SIZE_BIN,
                min_runs_per_gene=1)   # already restricted to `common`, above the real bar
            raw_counts, _ = _bin(sizes, SIZE_RANGE[0], SIZE_RANGE[1], SIZE_BIN)
            unmerged_by_cut[cut][lib]        = (freq, sizes)
            unmerged_counts_by_cut[cut][lib] = (raw_counts, sizes)

    # ---- View 3: per-gene call rate, paired ----
    totals_by_lib = {lib: {g: tot for g, (_prot, tot) in raw_by_lib[lib]["site_counts"].items()}
                     for lib in libIDs}
    common_sites = _common_genes(totals_by_lib, MIN_SITES_PER_GENE)
    print("    P_B>%s call rate: %d genes common to all %d libraries (>=%d scored sites each)"
          % (FIXED_CUTOFF, len(common_sites), len(libIDs), MIN_SITES_PER_GENE), file=sys.stderr)
    rates_by_lib = {
        lib: {g: raw_by_lib[lib]["site_counts"][g][0] / raw_by_lib[lib]["site_counts"][g][1]
              for g in common_sites}
        for lib in libIDs
    }

    # ---- View 4 / 5: His-codon P_B distribution + per-gene mean, paired ----
    his_counts_by_lib = {lib: _count_by_gene(raw_by_lib[lib]["his_obs"]) for lib in libIDs}
    common_his = _common_genes(his_counts_by_lib, MIN_HIS_OBS_PER_GENE)
    print("    His-codon sites: %d genes common to all %d libraries (>=%d His-codon sites each)"
          % (len(common_his), len(libIDs), MIN_HIS_OBS_PER_GENE), file=sys.stderr)
    his_hist_by_lib = {}
    his_gene_by_lib = {}
    his_edges = None
    for lib in libIDs:
        filtered = [o for o in raw_by_lib[lib]["his_obs"] if o["gene"] in common_his]
        his_freq, his_edges, _, _ = _gene_weighted_freq_float(
            filtered, "pb", HIS_PB_RANGE[0], HIS_PB_RANGE[1], HIS_PB_BIN,
            min_obs_per_gene=1)        # already restricted to `common_his`, above the real bar
        his_hist_by_lib[lib] = (his_freq, len(filtered))
        by_gene = collections.defaultdict(list)
        for o in filtered:
            by_gene[o["gene"]].append(o["pb"])
        his_gene_by_lib[lib] = {g: float(np.mean(vals)) for g, vals in by_gene.items()}

    # ---- View 6: n_his_shadows / n_reads_with_a_His_site, paired ----
    # Previously normalized by n_other_shadows (count of non-His-codon
    # shadows), which is itself a shadow count -- in a library with little
    # protection overall (e.g. a ribosome-less/phenol control), that count
    # collapses toward zero, which inflates the ratio from having a tiny
    # denominator rather than from any real His-codon-specific enrichment
    # (exactly what made phenol look disproportionately "enriched"). Reads
    # with >=1 His-codon site is a per-read opportunity count instead --
    # it doesn't move just because the rest of the gene happens to be more
    # or less protected in a given library, so it isolates the His-codon
    # effect specifically rather than confounding it with overall
    # protection level.
    #
    # A gene qualifies only if it clears MIN_HIS_OBS_PER_GENE His-codon
    # sites AND MIN_HIS_READS_PER_GENE qualifying reads in EVERY library.
    his_totals_by_lib = {lib: {g: tot for g, (_prot, tot) in raw_by_lib[lib]["his_shadow_counts"].items()}
                        for lib in libIDs}
    his_read_totals_by_lib = {lib: raw_by_lib[lib]["his_read_totals"] for lib in libIDs}
    common_his_vs_other = (_common_genes(his_totals_by_lib, MIN_HIS_OBS_PER_GENE) &
                          _common_genes(his_read_totals_by_lib, MIN_HIS_READS_PER_GENE))
    print("    His-shadow-count / reads-with-His-site: %d genes common to all %d libraries "
          "(>=%d His-codon sites, >=%d qualifying reads each)"
          % (len(common_his_vs_other), len(libIDs), MIN_HIS_OBS_PER_GENE, MIN_HIS_READS_PER_GENE),
          file=sys.stderr)
    his_vs_other_ratio_by_lib = {}
    for lib in libIDs:
        his_c = raw_by_lib[lib]["his_shadow_counts"]
        read_tot = raw_by_lib[lib]["his_read_totals"]
        n_zero = sum(1 for g in common_his_vs_other if read_tot.get(g, 0) == 0)
        if n_zero:
            print(f"    {lib}: dropped {n_zero} gene(s) with zero qualifying reads "
                  f"(undefined rate)", file=sys.stderr)
        his_vs_other_ratio_by_lib[lib] = {
            g: his_c[g][0] / read_tot[g]
            for g in common_his_vs_other if read_tot.get(g, 0) > 0
        }

    # ---- View 8: His-CENTERED-run rate, same pairing/denominator as View 6 ----
    # Stricter than View 6: instead of counting any His-codon SITE with
    # P_B>=cutoff, this counts qualifying (>=MIN_RUN_NT nt) protected RUNS
    # that are actually CENTERED on a His codon (His-codon position within
    # the middle HIS_CENTER_FRAC of the run's own span -- see
    # _gene_his_centered_run_counts). A run merely overlapping a His codon
    # near one edge could be incidental; one centered on it is what you'd
    # expect if the His codon itself triggered the stall, not just
    # something else nearby. Reuses View 6's exact gene set
    # (common_his_vs_other) and denominator (reads with >=1 His-codon site)
    # so the two rates are directly comparable side by side -- this view is
    # additional, not a replacement for View 6.
    his_centered_ratio_by_lib = {}
    for lib in libIDs:
        centered_c = raw_by_lib[lib]["his_centered_run_counts"]
        read_tot   = raw_by_lib[lib]["his_read_totals"]
        his_centered_ratio_by_lib[lib] = {
            g: centered_c.get(g, 0) / read_tot[g]
            for g in common_his_vs_other if read_tot.get(g, 0) > 0
        }

    # ---- View 7: shadow density (footprint-sized runs / nt / read) by region ----
    # A "shadow" here means a protected RUN of >=MIN_RUN_NT genomic nt at
    # P_B>=FIXED_CUTOFF (see extract_shadow_runs/_gene_region_run_counts),
    # not a bare scored site -- a footprint-sized event, not an isolated
    # blip. Normalized by BOTH region LENGTH (nt, from the GTF) AND READ
    # COVERAGE in that region (see _gene_region_read_coverage) -- the same
    # length+depth double normalization RNA-seq RPKM uses. Length alone
    # isn't enough: a region sampled by fewer reads/molecules would look
    # artificially "less shadowed" purely from lower depth, independent of
    # region size or true biology, so regions/genes of different sizes AND
    # different sequencing depth are directly comparable, and so is one
    # region across libraries.
    #
    # NOT gene-paired across libraries (unlike Views 3/5/6): each library's
    # bar is the mean density among every gene it has ANY shadow data for
    # (genes_observed), same "every gene gets one vote" philosophy as
    # _gene_weighted_freq, just without requiring the SAME genes to clear a
    # bar in every library at once. No minimum-run-count floor either -- a
    # gene with 0 qualifying runs in a region (but >=1 read covering it) is
    # a real, informative data point (density 0), not noise to be dropped;
    # excluding it would bias the mean upward by only counting genes that
    # already showed a hit. A region with ZERO covering reads, though, is
    # missing data rather than a true zero, so that IS excluded below.
    REGION_NAMES = ["UTR5", "CDS", "UTR3"]
    region_density_mean = {region: {} for region in REGION_NAMES}
    region_density_n    = {region: {} for region in REGION_NAMES}
    for ri, region in enumerate(REGION_NAMES):
        for lib in libIDs:
            run_counts     = raw_by_lib[lib]["region_run_counts"]
            read_coverage  = raw_by_lib[lib]["region_read_coverage"]
            genes_observed = raw_by_lib[lib]["genes_observed"]
            vals = [run_counts.get(g, {}).get(region, 0)
                    / (region_len_by_gene[g][ri] * read_coverage.get(g, {}).get(region, 0))
                    for g in genes_observed
                    if g in region_len_by_gene and region_len_by_gene[g][ri] > 0
                    and read_coverage.get(g, {}).get(region, 0) > 0]
            region_density_mean[region][lib] = float(np.mean(vals)) if vals else 0.0
            region_density_n[region][lib] = len(vals)
            print("    %s shadow density (%s): %d genes, mean=%.6g"
                  % (region, lib, len(vals), region_density_mean[region][lib]), file=sys.stderr)

    print("Plotting combined figures across %d libraries..." % len(libIDs), file=sys.stderr)
    # View 1: footprint size, multi-panel over cutoffs (per-gene-then-aggregated freq)
    plot_footprint_sizes(dict(unmerged_by_cut), fp_edges,
                         "%s.footprint_sizes" % outPrefix)
    # View 1-supplement: same layout, raw pooled counts (no per-gene weighting)
    # -- simple sanity check against the per-gene-then-aggregated view above.
    plot_footprint_sizes(dict(unmerged_counts_by_cut), fp_edges,
                         "%s.footprint_sizes_counts" % outPrefix, ylabel="count")
    # View 3: per-gene call rate, magnitude comparison across libraries
    plot_gene_call_rates(rates_by_lib, "%s.gene_call_rates.png" % outPrefix,
                         connect_matched=True)
    # View 4: P_B distribution at His-codon sites, not collapsed to one cutoff
    plot_his_codon_pb_distribution(his_hist_by_lib, his_edges,
                                   "%s.his_codon_pb_distribution" % outPrefix)
    # View 5: mean His-codon P_B per gene, paired across libraries -- the
    # pooled distribution above can hide a real per-gene shift when
    # baseline P_B varies a lot gene to gene; connect_matched makes each
    # gene's own trajectory across libraries visible.
    plot_gene_call_rates(his_gene_by_lib, "%s.his_codon_mean_pb_paired.png" % outPrefix,
                         title="Mean His-codon P$_B$ by library (paired per gene)",
                         min_n=MIN_HIS_OBS_PER_GENE, min_n_label="His-codon sites",
                         ylabel="mean P$_B$ at His-codon sites", connect_matched=True)
    # View 6: n_his_shadows / n_reads_with_a_His_site, per-gene rate, paired
    # across libraries -- normalized by reads that had the opportunity to
    # show a His-codon shadow, not by n(other shadows), so a library with
    # little protection overall (e.g. phenol/ribosome-less) doesn't get an
    # inflated ratio just from having a tiny other-shadow denominator.
    plot_gene_call_rates(his_vs_other_ratio_by_lib, "%s.his_codon_enrichment.png" % outPrefix,
                         title="His-shadow-count / reads-with-His-site by library",
                         min_n=MIN_HIS_READS_PER_GENE, min_n_label="qualifying reads",
                         ylabel="n(His shadows) / n(reads with His site)", connect_matched=True)
    # View 8: same as View 6, but the numerator only counts qualifying runs
    # CENTERED on a His codon rather than any His-codon site -- a stricter,
    # additional test of whether the His codon itself is driving the
    # protection, not a replacement for View 6.
    plot_gene_call_rates(his_centered_ratio_by_lib, "%s.his_codon_centered_enrichment.png" % outPrefix,
                         title="His-CENTERED-run-count / reads-with-His-site by library",
                         min_n=MIN_HIS_READS_PER_GENE, min_n_label="qualifying reads",
                         ylabel="n(His-centered runs) / n(reads with His site)", connect_matched=True)
    # View 7: shadow density (shadows/nt/read) by region -- one grouped barplot,
    # all regions and libraries together, not gene-paired across libraries.
    plot_region_density_grouped_bars(region_density_mean, region_density_n,
                                     "%s.shadow_density_by_region.png" % outPrefix,
                                     libIDs, region_names=REGION_NAMES,
                                     ylabel="mean shadows / nt / read")


if __name__ == "__main__":
    main(sys.argv[1:])