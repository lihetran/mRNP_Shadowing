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
  VIEW 3-TE  per-gene log2FC in footprint/shadow-run rate (relative to the
          phenol library) against translation EFFICIENCY from an external
          RPF/RNA table, one scatter panel per library -- no statistics
          yet, just the raw points -- only produced if a TE file is
          passed (optional 5th CLI arg) AND a phenol library is present.
  VIEW 10  nt GAP between consecutive shadow RUNS on the same read (P_B>=
          GAP_CUTOFF, default 0.7) -> "how far apart are shadow calls?" --
          one step-CDF line per library, raw-pooled across every read with
          >=2 shadow calls (not gene-weighted, unlike View 1/2's shape
          aggregates).

Input parquet columns used per read:
  read_id, shadow_gene, absolute_indices, shadow_gpos, shadow_P_B,
  shadow_region, n_sites_utr5, n_sites_cds, n_sites_utr3
  (shadow_P_B/shadow_gpos/shadow_region/n_sites_* cover EVERY scored Ref=A
  site for the read, not just ones above some cutoff -- write_shadow_calls_to_df
  stopped pre-filtering to P_B>=threshold at the source, so every view in
  this module applies whatever cutoff it needs itself, e.g. via PROB_CUTOFFS
  or FIXED_CUTOFF.)

Run:
  python3 shadowSizeQC.py inFilesParquet.txt outPrefix hisCodonPositions.pickle gtfFile [tePath] [flank_nt] [colorMapPath]
where inFilesParquet.txt is line-delimited:  fileName  rep  parquetFile,
hisCodonPositions.pickle is findHisCodonPositions.py's output, gtfFile
is the GTF used to derive each gene's CDS length for View 7 (UTR5/UTR3
lengths come from flank_nt instead -- see region_lengths), the optional
tePath is a per-gene translation-efficiency table (a .parquet with
gene_name/TE_score columns, e.g. Weinberg/Bartel RPF-vs-RNA data -- see
load_translation_efficiency) enabling View 3-TE, the optional flank_nt
(default 150) MUST match whatever --flank_nt runHMMPerGene.py/
trainHMMPerGene.py actually scored these parquets with, since UTR5/UTR3
"length" for View 7's density normalization is now that scored flank
width (capped per gene at the nearest-neighbor gap via
compute_flank_caps), not a GTF-annotated UTR span -- this GTF barely
annotates any (every "UTR3" is exactly the stop codon's own 3nt), so
using that instead would wildly distort View 7's UTR density -- and the
optional colorMapPath is a manuscript color-map TSV (name, rep, path,
hex_color, no leading '#' -- the same convention/file used across the
other scripts' --color_map options) matched against libraryID
("fileName-rep") to keep a given library the same color across every
plot in this module AND consistent with every other script's figures;
libraries with no match fall back to the built-in PALETTE cycle.
"""

import sys, math, collections, pickle
import numpy as np
import pandas as pd

from runHMMPerGene import parse_gtf, cds_length, compute_flank_caps
# NOTE: parse_gtf/cds_length are imported rather than duplicated a third
# time (trainHMMPerGene.py already imports _forward_backward_hsmm/
# _duration_pmf_default from runHMMPerGene.py the same way) -- these are
# the exact same GTF-parsing logic verbatim in both other scripts, with no
# per-script variation, unlike e.g. _gpos_to_tx_map which each script keeps
# its own copy of because it differs slightly per use case.

try:
    from pyx import canvas, graph, color, style, path, deco, trafo, text as pyx_text
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
MIN_READS_FOR_RUN_RATE = 10                   # View 9: drop a gene from the runs-per-read
                                               # rate below this many reads (see caveat in
                                               # _gene_run_counts_by_reads about what "reads"
                                               # means here)
FLANK_NT       = 150                          # View 7: UTR5/UTR3 region length -- must match
                                               # whatever --flank_nt runHMMPerGene.py/
                                               # trainHMMPerGene.py actually scored this
                                               # parquet with (see region_lengths)
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
GAP_CUTOFF     = 0.7                          # View 10: P_B cutoff shadow runs are called at
                                               # before measuring the nt gap between consecutive
                                               # runs on the same read -- same value as
                                               # FIXED_CUTOFF, kept as its own name since a
                                               # caller may want to sweep this independently of
                                               # View 1/His-codon's cutoff later
GAP_RANGE      = (0, 500)                     # View 10: nt-gap x-axis range for the CDF plot --
                                               # wider than SIZE_RANGE/SWEEP_RANGE since this is
                                               # UNPROTECTED space BETWEEN two shadow calls, not
                                               # a single footprint's own size
MIN_GAPS_PER_GENE = 0                         # View 10-paired: drop a gene from the gene-matched
                                               # gap CDF below this many gap observations in ANY
                                               # library -- 0 (same default as MIN_RUNS_PER_GENE)
                                               # since this CDF is raw-pooled, not per-gene
                                               # weighted, so a gene only needs to be PRESENT in
                                               # every library to be a fair inclusion, not have
                                               # enough observations to trust its own shape
TE_RANGE       = (0.75, 2.75)                 # View 3-TE: x-axis (translation efficiency) crop
                                               # range for the log2FC-vs-TE scatter -- picked to
                                               # match where the real Weinberg/Bartel TE_score
                                               # distribution actually sits, not derived from the
                                               # data itself like x_max used to be, so a gene
                                               # outside it is a real outlier worth a stderr
                                               # warning rather than silently rescaling the axis
PALETTE        = None                         # list of pyx colors, set after import
COLOR_MAP      = {}                           # {libraryID: "#RRGGBB"}, set in main() if
                                               # a colorMapPath was given -- takes priority
                                               # over PALETTE's index-based fallback so a
                                               # given library keeps the same color across
                                               # every plot in this module and every other
                                               # script sharing the same manuscript color file


def load_color_map(path: str) -> dict:
    """
    Parse a manuscript color-map TSV with columns:
        sample_name, rep, path, hex_color (no leading '#')
    Returns a dict keyed by "name_rep", "name-rep" (this module's own
    libraryID convention -- see main()'s "%s-%s" % (fileName, rep)), and
    bare "name" (first match wins for the bare key) mapping to "#RRGGBB".
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
    hexcol = hexcol.lstrip("#")
    r = int(hexcol[0:2], 16) / 255.0
    g = int(hexcol[2:4], 16) / 255.0
    b = int(hexcol[4:6], 16) / 255.0
    return color.rgb(r, g, b)


def _libcolor(libID, i):
    """Manuscript color for libID if COLOR_MAP has one, else the i'th
    PALETTE color (index-based fallback, same as before colorMapPath
    existed)."""
    hexcol = COLOR_MAP.get(libID)
    if hexcol:
        return hex_to_pyx_color(hexcol)
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


def shadow_run_gaps(runs):
    """
    View 10: per-read nucleotide GAP between consecutive shadow runs (i.e.
    the unprotected stretch separating one shadow call from the next along
    the same read), pooled across every read with >=2 runs.

    runs: output of extract_shadow_runs at a single cutoff (e.g.
    GAP_CUTOFF) -- runs are grouped by read_id, sorted by gpos_lo (genomic-
    ascending, strand-safe the same way extract_shadow_runs' own gpos_lo/
    gpos_hi already are), and the gap between each adjacent PAIR is
    gpos_lo(next) - gpos_hi(prev) - 1: the count of nt strictly between the
    two runs' own spans, in the same genomic-nt space extract_shadow_runs
    uses for "genomic_nt" (not the padded aligned_len window, which pads
    outward for a different purpose and would double-count real footprint
    nt as if it were unprotected gap). A read with N runs contributes N-1
    gaps.

    Returns a list of {"gene", "read_id", "gap"} dicts.
    """
    by_read = collections.defaultdict(list)
    for r in runs:
        by_read[r["read_id"]].append(r)
    gaps = []
    for read_id, read_runs in by_read.items():
        if len(read_runs) < 2:
            continue
        read_runs.sort(key=lambda r: r["gpos_lo"])
        for a, b in zip(read_runs, read_runs[1:]):
            gaps.append({"gene": a["gene"], "read_id": read_id,
                         "gap": b["gpos_lo"] - a["gpos_hi"] - 1})
    return gaps


def read_gap_eligibility(runs, n_reads_total):
    """
    View 10-companion: per-library breakdown of how many reads CAN vs
    CANNOT contribute a gap to shadow_run_gaps' CDF, at whatever cutoff
    `runs` was extracted with (GAP_CUTOFF) -- a read needs >=2 shadow runs
    to have even one gap to measure, so a read with 0 or 1 runs is
    structurally excluded, not filtered out by any gene-pairing or range
    choice downstream.

    extract_shadow_runs simply omits a read from `runs` entirely once it
    has zero qualifying runs (there's nothing to append), so the "zero
    runs" count can't be read off `runs` directly -- it has to be backed
    out as n_reads_total minus however many DISTINCT read_ids appear in
    `runs` at all. n_reads_total must come from the same library's full
    read count (see extract_library_raw's "n_reads_total", not `len(runs)`
    or anything else derived only from the already-filtered runs list).

    Returns {"n_reads_total", "n_zero_runs", "n_one_run",
    "n_two_plus_runs"}: n_zero_runs + n_one_run is the read count EXCLUDED
    from the gap CDF ("reads where we can't make a gap at all");
    n_two_plus_runs is the read count that actually contributes >=1 gap.
    """
    run_counts_by_read = collections.Counter(r["read_id"] for r in runs)
    n_one_run  = sum(1 for c in run_counts_by_read.values() if c == 1)
    n_two_plus = sum(1 for c in run_counts_by_read.values() if c >= 2)
    n_zero_runs = n_reads_total - len(run_counts_by_read)
    return {"n_reads_total": n_reads_total, "n_zero_runs": n_zero_runs,
            "n_one_run": n_one_run, "n_two_plus_runs": n_two_plus}


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


def load_translation_efficiency(te_path):
    """
    {gene_name: TE_score} from a Weinberg/Bartel-style per-gene translation-
    efficiency table (RPF_RPKM / RNA_RPKM), keyed by the same SGD gene_name
    parse_gtf uses (e.g. "RPL17A"), not systematic_name ("YKL180W") -- so it
    joins directly against shadow_gene / rates_by_lib's gene keys with no
    extra ID-mapping step.

    Accepts either a parquet with gene_name/TE_score columns (e.g.
    260508_translationEfficiency_histogram.parquet, which also carries
    decile_rank/quartile_rank -- not used here, but there if a caller wants
    quartile-level grouping instead of the continuous score) or a flat
    newline-delimited gene-list .txt (e.g. top_decile.txt) -- the latter
    carries no actual score, so every listed gene maps to TE_score=1.0 and
    genes not listed are simply absent from the returned dict, i.e. this
    degrades to a binary membership set rather than a real correlate.
    """
    te_path = str(te_path)
    if te_path.endswith(".parquet"):
        te_df = pd.read_parquet(te_path, columns=["gene_name", "TE_score"])
        te_df = te_df.dropna(subset=["gene_name", "TE_score"])
        return dict(zip(te_df["gene_name"], te_df["TE_score"].astype(float)))

    with open(te_path) as f:
        genes = {line.strip() for line in f if line.strip()}
    print(f"load_translation_efficiency: {te_path} has no TE_score column "
          f"(not a .parquet) -- treating its {len(genes)} genes as a binary "
          f"membership set (TE_score=1.0), not a real correlate", file=sys.stderr)
    return {g: 1.0 for g in genes}


def his_codon_pb_observations(df, his_gpos_by_gene):
    """
    Every (read, His-codon site) observation's raw P_B value, tagged by
    gene -- no cutoff applied. Averaged per gene (see main()'s View 5
    construction of his_gene_by_lib) into a per-gene mean His-codon P_B.

    This replaces an earlier version (his_codon_call_rates) that collapsed
    this to a single P_B>0.7 rate. Confirmed on real data that a fixed
    cutoff can sit awkwardly relative to one specific site's own achievable
    range: one gene's His-codon site never exceeded P_B=0.64 in ANY
    library, reading as "0% protected" even though that's a meaningfully
    elevated value for that site.

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


def region_lengths(gene, flank_5p, flank_3p):
    """
    (utr5_len, cds_len, utr3_len) in nt, from a parse_gtf gene dict.

    utr5_len/utr3_len are NOT the GTF-annotated gene["utr5"]/["utr3"]
    interval spans -- this GTF barely annotates any (every "UTR3" is
    exactly the stop codon's own 3nt; almost no gene has any UTR5 at
    all), so that would either wildly inflate View 7's UTR3 density
    (dividing real shadow calls, now correctly scored ~150nt into the
    flank, by a 3nt denominator) or silently exclude UTR5 entirely (the
    region_len_by_gene[g][ri] > 0 guard at the call site). flank_5p/
    flank_3p are the gene's own compute_flank_caps entry instead -- the
    ACTUAL flank width runHMMPerGene.py/trainHMMPerGene.py scored for
    this gene (capped at the gap to its nearest neighbor), so the
    denominator here matches what was truly scored, not an annotation
    artifact. cds_length() already exists in runHMMPerGene.py; that part
    was never wrong.
    """
    cds_len = cds_length(gene)
    return flank_5p, cds_len, flank_3p


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
                   [graph.style.line([_libcolor(libID, i), style.linewidth.Thick])])
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
                   [graph.style.line([_libcolor(libID, i), style.linewidth.Thick])])
        c.insert(g)
        if bottom is None:
            bottom = g
    c.writePDFfile(str(pdf_path))


def plot_gene_call_rates(rates_by_lib, pdf_path, cutoff=FIXED_CUTOFF,
                         title="Protected-site rate by library",
                         min_n=MIN_SITES_PER_GENE, min_n_label="scored sites",
                         ylabel=None, connect_matched=False):
    """
    Per-gene metric, one box (unfilled, Tukey whiskers -- matching
    matplotlib's own default boxplot convention) + jittered strip per
    library, PyX -- matching this module's other library-colored figures.
    This is the MAGNITUDE counterpart to the size histograms above, which
    equal-weight every gene on purpose and so wash out exactly the kind of
    library-to-library difference (e.g. ribosome-containing vs. a
    ribosome-less control) this plot is meant to show.

    rates_by_lib: {libraryID: {gene: value}} -- from gene_call_rates (every
    scored A site) or his_codon_pb_per_gene (mean P_B at His-codon sites);
    title/min_n/min_n_label/ylabel just describe the plot itself.

    Strip points (and the box outline) are colored by _libcolor(libID, i)
    -- COLOR_MAP's manuscript hex color if libID has one, else the
    PALETTE index fallback, same mechanism as every other library-colored
    plot in this module.

    connect_matched: restrict EVERY library's box+strip to only the genes
    common to all libraries shown, and draw a thin grey line through each
    gene's point across libraries (a paired/spaghetti overlay). Comparing
    full per-library gene sets box-to-box is misleading here -- a library
    with many more passing genes than another can shift its whole box off
    a baseline-level difference in WHICH genes it includes, not a real
    per-gene effect (confirmed on real data: unrestricted boxes made a
    library with 42 genes look higher overall than one with only 11, but
    the 11 genes actually shared between them mostly moved the other way).
    Restricting to the shared set makes both the box and the lines honest
    paired comparisons of the same genes.
    """
    if canvas is None:
        print("pyx not available; skipping plot", file=sys.stderr); return

    libIDs = sorted(rates_by_lib)
    n_libs = len(libIDs)

    if connect_matched:
        common_genes = sorted(set.intersection(*(set(rates_by_lib[lib]) for lib in libIDs))) \
                       if libIDs else []
        data = {lib: [rates_by_lib[lib][g] for g in common_genes] for lib in libIDs}
    else:
        common_genes = None
        data = {lib: list(rates_by_lib[lib].values()) for lib in libIDs}

    all_vals = [v for lib in libIDs for v in data[lib]]
    if not all_vals:
        print(f"plot_gene_call_rates: no data to plot, skipping ({pdf_path})",
              file=sys.stderr)
        return
    y_lo = min(0.0, min(all_vals))
    y_hi = max(all_vals) * 1.1 if max(all_vals) > 0 else 1.0

    panel_w, panel_h = max(1.4 * n_libs, 4), 6
    g = graph.graphxy(
        width=panel_w, height=panel_h, xpos=0, ypos=0,
        x=graph.axis.linear(min=0.3, max=n_libs + 0.7, parter=None),
        y=graph.axis.linear(min=y_lo, max=y_hi,
                            title=ylabel or f"per-gene call rate (P$_B>${cutoff})"),
    )
    c = canvas.canvas()
    c.insert(g)

    rng = np.random.default_rng(0)

    # spaghetti lines (unjittered x, one per common gene) -- drawn first
    if connect_matched and common_genes:
        for gi in range(len(common_genes)):
            pts = [(i + 1, data[lib][gi]) for i, lib in enumerate(libIDs)]
            g.plot(graph.data.points(pts, x=1, y=2),
                   [graph.style.line([color.gray(0.6), style.linewidth.thin])])

    box_hw = 0.18   # box/whisker-cap half-width, in x-axis units
    for i, lib in enumerate(libIDs):
        xi  = i + 1
        vals = np.asarray(data[lib])
        col = _libcolor(lib, i)

        if len(vals):
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            iqr = q3 - q1
            lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            in_range = vals[(vals >= lo_fence) & (vals <= hi_fence)]
            whisk_lo = in_range.min() if len(in_range) else q1
            whisk_hi = in_range.max() if len(in_range) else q3

            for y0v, y1v in [(whisk_lo, q1), (q3, whisk_hi)]:
                x0, y0 = g.pos(xi, y0v)
                x1, y1 = g.pos(xi, y1v)
                c.stroke(path.line(x0, y0, x1, y1), [style.linewidth.thin, color.gray(0.3)])
            for wv in (whisk_lo, whisk_hi):
                xa, ya = g.pos(xi - box_hw / 2, wv)
                xb, yb = g.pos(xi + box_hw / 2, wv)
                c.stroke(path.line(xa, ya, xb, yb), [style.linewidth.thin, color.gray(0.3)])

            xa, ya = g.pos(xi - box_hw, q1)
            xb, yb = g.pos(xi + box_hw, q3)
            c.stroke(path.rect(xa, ya, xb - xa, yb - ya), [style.linewidth.thin, color.gray(0.2)])
            xa2, ya2 = g.pos(xi - box_hw, med)
            xb2, yb2 = g.pos(xi + box_hw, med)
            c.stroke(path.line(xa2, ya2, xb2, yb2), [style.linewidth.thick, color.gray(0.2)])

        jitter = rng.normal(0, 0.06, size=len(vals))
        pts = [(xi + j, v) for j, v in zip(jitter, vals)]
        if pts:
            g.plot(graph.data.points(pts, x=1, y=2),
                   [graph.style.symbol(graph.style.symbol.circle, size=0.07,
                                       symbolattrs=[deco.filled([col]), deco.stroked([col])])])

        xn, yn = g.pos(xi, y_lo)
        c.text(xn, yn - 0.2, "%s (n=%d)" % (lib.replace("_", r"\_"), len(vals)),
              [pyx_text.halign.right, pyx_text.size.small, trafo.rotate(30)])

    n_note = (f"{len(common_genes)} genes common to all libraries"
              if connect_matched else
              f"genes with <{min_n} {min_n_label} dropped")
    c.text(g.xpos + g.width / 2., g.ypos + g.height + 0.5, title,
          [pyx_text.halign.center, pyx_text.size.normalsize])
    c.text(g.xpos + g.width / 2., g.ypos + g.height + 0.2,
          f"(one point per gene, {n_note})",
          [pyx_text.halign.center, pyx_text.size.scriptsize])

    c.writePDFfile(str(pdf_path))
    print(f"Wrote gene call-rate plot to {pdf_path}.pdf", file=sys.stderr)


def compute_log2fc_vs_reference(rates_by_lib, reference_lib, pseudo=1e-3):
    """
    {libraryID: {gene: log2fc}} -- log2((rate+pseudo) / (ref_rate+pseudo))
    per gene, for every library EXCEPT reference_lib itself (its own log2FC
    against itself would be trivially 0 for every gene). pseudo avoids
    log2(0) the same way metaHistidineFromParquet.py's own
    transcript_normalised_agg/compute_log2fc_agg add a pseudocount before
    ratio-ing two possibly-zero rates.

    Only genes present in BOTH a library's own rates_by_lib entry AND
    reference_lib's are included for that library -- not assumed
    pre-matched, since a caller's rates_by_lib may restrict genes
    per-library (e.g. a per-library min-reads floor).
    """
    ref_rates = rates_by_lib.get(reference_lib, {})
    log2fc_by_lib = {}
    for lib, gene_rates in rates_by_lib.items():
        if lib == reference_lib:
            continue
        genes = sorted(set(gene_rates) & set(ref_rates))
        log2fc_by_lib[lib] = {
            g: math.log2((gene_rates[g] + pseudo) / (ref_rates[g] + pseudo))
            for g in genes
        }
    return log2fc_by_lib


def plot_shadow_log2fc_vs_te(log2fc_by_lib, te_by_gene, pdf_path,
                             ylabel="log2FC of Shadow Calls against Phenol",
                             xlabel="Translation Efficiency", x_range=TE_RANGE):
    """
    One pooled PyX scatter -- every library's genes on the SAME panel
    (not one panel per library), colored by _libcolor(libID, i) -- COLOR_MAP's
    manuscript hex color if libID has one, else the PALETTE index fallback,
    the same mechanism/colors as every other library-colored plot in this
    module (size/sweep/His-codon/region-density figures).

    log2fc_by_lib: {libraryID: {gene: log2fc}}, from
    compute_log2fc_vs_reference -- already excludes the reference (phenol)
    library itself. te_by_gene: {gene: TE_score}.

    Deliberately no statistics yet (no regression line, no correlation
    annotation, no gene-name labels) -- just the raw per-gene points, to
    look at before deciding what (if anything) is worth annotating.

    Square panel, y-axis cropped to [0, max] rather than the symmetric
    [-max, max] range a diverging log2FC axis would normally get -- on
    real data every gene here has come out log2FC>=0 (P_B>=cutoff shadow
    calls only going up relative to phenol, never down), so the bottom
    half of a symmetric axis was empty space. Any point that DOES land
    below 0 still gets a stderr warning rather than silently vanishing
    off the cropped axis (see n_below_zero below).

    x-axis is likewise cropped to a fixed x_range (TE_RANGE) rather than
    scaled to whatever this call's own TE_score values happen to span --
    real TE_scores cluster there, so a fixed window makes different calls
    of this function (different cutoffs, different libraries) directly
    comparable panel-to-panel. Genes outside x_range get a stderr warning
    (see n_outside_range below), same "don't silently crop" treatment as
    n_below_zero.
    """
    if canvas is None:
        print("pyx not available; skipping plot", file=sys.stderr); return

    libIDs = sorted(log2fc_by_lib)
    genes_by_lib = {lib: sorted(set(log2fc_by_lib[lib]) & set(te_by_gene))
                    for lib in libIDs}
    for lib in libIDs:
        n_have_rate = len(log2fc_by_lib[lib])
        n_matched   = len(genes_by_lib[lib])
        if n_have_rate:
            print(f"  [{lib}] {n_matched}/{n_have_rate} genes have a TE_score "
                  f"match ({n_have_rate - n_matched} dropped, no TE data)",
                  file=sys.stderr)

    all_te = [te_by_gene[g] for lib in libIDs for g in genes_by_lib[lib]]
    all_fc = [log2fc_by_lib[lib][g] for lib in libIDs for g in genes_by_lib[lib]]
    if not all_te:
        print("plot_shadow_log2fc_vs_te: no genes with both a log2FC and a "
              "TE_score; skipping.", file=sys.stderr)
        return

    x_lo, x_hi = x_range
    y_max = max(max(all_fc), 0.5) * 1.15

    n_below_zero = sum(1 for v in all_fc if v < 0)
    if n_below_zero:
        print(f"  WARNING: {n_below_zero}/{len(all_fc)} points have log2FC<0 "
              f"and will be cropped off this plot's [0, {y_max:.2g}] y-axis",
              file=sys.stderr)

    n_outside_range = sum(1 for v in all_te if v < x_lo or v > x_hi)
    if n_outside_range:
        print(f"  WARNING: {n_outside_range}/{len(all_te)} points have a "
              f"TE_score outside [{x_lo}, {x_hi}] and will be cropped off "
              f"this plot's x-axis", file=sys.stderr)

    g = graph.graphxy(
        width=7, height=7, xpos=0, ypos=0,
        x=graph.axis.linear(min=x_lo, max=x_hi, title=xlabel),
        y=graph.axis.linear(min=0, max=y_max, title=ylabel),
        key=graph.key.key(pos="tr", hinside=0))

    for i, lib in enumerate(libIDs):
        genes = genes_by_lib[lib]
        if not genes:
            continue
        pts = [(te_by_gene[g], log2fc_by_lib[lib][g]) for g in genes]
        col = _libcolor(lib, i)
        title = r"%s (n=%d)" % (lib.replace("_", r"\_"), len(genes))
        g.plot(graph.data.points(pts, x=1, y=2, title=title),
               [graph.style.symbol(graph.style.symbol.circle, size=0.09,
                                   symbolattrs=[deco.filled([col]), deco.stroked([col])])])

    c = canvas.canvas()
    c.insert(g)
    c.writePDFfile(str(pdf_path))
    print(f"Wrote shadow log2FC vs. TE scatter plot to {pdf_path}.pdf", file=sys.stderr)


def plot_region_density_grouped_bars(region_density_mean, region_density_n, pdf_path,
                                     libIDs, region_names=("UTR5", "CDS", "UTR3"),
                                     ylabel="mean shadows / nt / read"):
    """
    Grouped bar chart (PyX, matching this module's other library-colored
    figures): one group per region (UTR5/CDS/UTR3), one bar per library
    within each group -- all libraries and all regions visible on one
    plot, rather than three separate per-region figures.

    NOT gene-paired across libraries (see View 7 in main()): each bar is
    that library's own mean per-gene density among its own qualifying
    genes, so a different number of genes can back each bar (annotated as
    a rotated n= label above it) -- libraries aren't held to the same
    shared gene set the way plot_gene_call_rates' connect_matched option
    does.

    Bar color comes from _libcolor(libID, i) -- COLOR_MAP's manuscript hex
    color if libID has one, else the PALETTE index fallback -- the same
    mechanism as every other library-colored plot in this module, so a
    given library is the same color here as in the size/sweep/His-codon
    figures.
    """
    if canvas is None:
        print("pyx not available; skipping plot", file=sys.stderr); return

    n_libs    = len(libIDs)
    n_regions = len(region_names)

    y_max = 0.0
    for region in region_names:
        for lib in libIDs:
            y_max = max(y_max, region_density_mean[region][lib])
    y_max = (y_max or 1e-9) * 1.2

    group_pad = 0.12
    usable    = 1.0 - 2 * group_pad
    bar_w     = usable / n_libs
    panel_w, panel_h = max(2.5 * n_regions, 6), 6

    c = canvas.canvas()
    g = graph.graphxy(
        width=panel_w, height=panel_h, xpos=0, ypos=0,
        x=graph.axis.linear(min=0, max=n_regions, parter=None, title="Gene region"),
        y=graph.axis.linear(min=0, max=y_max, title=ylabel),
    )
    c.insert(g)

    for ri, region in enumerate(region_names):
        for i, lib in enumerate(libIDs):
            val = region_density_mean[region][lib]
            n   = region_density_n[region][lib]
            bx0 = ri + group_pad + i * bar_w
            bx1 = bx0 + bar_w
            cx0, cy0 = g.pos(bx0, 0.0)
            cx1, cy1 = g.pos(bx1, val)
            col = _libcolor(lib, i)
            if cy1 > cy0:
                c.fill(path.rect(cx0, cy0, cx1 - cx0, cy1 - cy0), [col])
                c.stroke(path.rect(cx0, cy0, cx1 - cx0, cy1 - cy0),
                        [style.linewidth.thin, color.gray(0.3)])
            c.text(cx0 + (cx1 - cx0) / 2., cy1 + 0.05, "n=%d" % n,
                  [pyx_text.halign.left, pyx_text.valign.middle,
                   pyx_text.size.tiny, trafo.rotate(90)])
        cxm, cym = g.pos(ri + 0.5, 0.0)
        c.text(cxm, cym - 0.4, region,
              [pyx_text.halign.center, pyx_text.size.small])

    c.text(g.xpos + g.width / 2., g.ypos + g.height + 0.6,
          "Shadow density by region and library",
          [pyx_text.halign.center, pyx_text.size.normalsize])
    c.text(g.xpos + g.width / 2., g.ypos + g.height + 0.25,
          "(mean per-gene density, not gene-paired across libraries)",
          [pyx_text.halign.center, pyx_text.size.scriptsize])

    leg_x, leg_y_top = g.xpos + g.width + 0.4, g.ypos + g.height - 0.3
    leg_lw, leg_dy    = 0.6, 0.55
    for i, lib in enumerate(libIDs):
        ly  = leg_y_top - i * leg_dy
        col = _libcolor(lib, i)
        c.fill(path.rect(leg_x, ly - 0.12, leg_lw, 0.24), [col])
        c.stroke(path.rect(leg_x, ly - 0.12, leg_lw, 0.24),
                [style.linewidth.thin, color.gray(0.3)])
        c.text(leg_x + leg_lw + 0.15, ly, lib.replace("_", r"\_"),
              [pyx_text.valign.middle, pyx_text.size.small])

    c.writePDFfile(str(pdf_path))
    print(f"Wrote region shadow-density barplot to {pdf_path}.pdf", file=sys.stderr)


def _ecdf_step_points(values):
    """
    Stepped (x, y) points tracing an empirical CDF: a plain PyX line drawn
    through these renders as a right-continuous step function (flat, then a
    vertical jump at each observed value), not the diagonal-interpolation a
    naive (sorted_value, rank/n) line would give.
    """
    vals = sorted(values)
    n = len(vals)
    pts = [(vals[0], 0.0)]
    for i, v in enumerate(vals):
        y = (i + 1) / n
        pts.append((v, pts[-1][1]))   # flat to the new x at the old y
        pts.append((v, y))            # then jump to the new y
    return pts


def plot_shadow_gap_cdf(gaps_by_lib, pdf_path, x_range=GAP_RANGE, cutoff=GAP_CUTOFF):
    """
    View 10: one panel, one step-CDF line per library (same _libcolor as
    every other library-colored figure in this module) of the nt GAP
    between consecutive shadow runs on the same read (see shadow_run_gaps)
    -- i.e. how much unprotected sequence typically separates one shadow
    call from the next, not a single shadow's own size (View 1/2's subject).

    Deliberately raw-pooled per library, NOT gene-weighted like
    _gene_weighted_freq's size histograms -- an empirical CDF is
    conventionally the pooled distribution of the actual observations, and
    this is the direct nt-gap counterpart to View 1's raw-count supplement
    panel (plot_footprint_sizes' ylabel="count" call), not a shape
    aggregate meant to give every gene an equal vote.

    gaps_by_lib: {libraryID: [gap dicts]} from shadow_run_gaps, each with a
    "gap" key. Only reads with >=2 shadow runs contribute (a read with a
    single shadow, or none, has no gap to measure) -- the per-library n in
    the key is that gap count, not the read or gene count.

    The x-axis is clipped to x_range for readability; the ECDF itself is
    still computed over the FULL gap list first (see _ecdf_step_points), so
    a library's curve reaching y<1 at x_range's right edge means real mass
    beyond the visible window, not a truncated calculation -- the fraction
    still off-screen is reported to stderr per library rather than silently
    dropped.
    """
    if canvas is None:
        print("pyx not available; skipping plot", file=sys.stderr); return

    libIDs = sorted(gaps_by_lib)
    x_lo, x_hi = x_range

    g = graph.graphxy(
        width=7, height=7, xpos=0, ypos=0,
        x=graph.axis.linear(min=x_lo, max=x_hi,
                            title=r"Gap between consecutive shadow calls (nt, P$_B\geq$%s)" % cutoff),
        y=graph.axis.linear(min=0, max=1, title="cumulative fraction of gaps"),
        key=graph.key.key(pos="tr", hinside=0),
    )
    c = canvas.canvas()
    c.insert(g)

    for i, libID in enumerate(libIDs):
        gaps = [d["gap"] for d in gaps_by_lib[libID]]
        if not gaps:
            print(f"  [{libID}] no reads with >=2 shadow calls; skipping its CDF line",
                  file=sys.stderr)
            continue
        frac_beyond = sum(1 for v in gaps if v > x_hi) / len(gaps)
        print("    %s: n=%d gaps, median=%.0f nt, %.1f%% beyond the %d-nt axis"
              % (libID, len(gaps), np.median(gaps), 100 * frac_beyond, x_hi), file=sys.stderr)
        pts = _ecdf_step_points(gaps)
        col = _libcolor(libID, i)
        title = r"%s (n=%d, med %.0f nt)" % (libID.replace("_", r"\_"), len(gaps),
                                             np.median(gaps))
        g.plot(graph.data.points(pts, x=1, y=2, title=title),
               [graph.style.line([col, style.linewidth.Thick])])

    c.writePDFfile(str(pdf_path))
    print(f"Wrote shadow-gap CDF plot to {pdf_path}.pdf", file=sys.stderr)


def plot_read_gap_exclusion_bars(eligibility_by_lib, pdf_path,
                                 title="Reads excluded from the shadow-gap CDF"):
    """
    View 10-companion: one bar per library (PyX, _libcolor-colored like
    every other library plot in this module) of the fraction of reads that
    CANNOT contribute a gap to shadow_run_gaps' CDF -- reads with 0 or 1
    shadow runs at GAP_CUTOFF, which structurally have no "next" run to
    measure a distance to (see read_gap_eligibility). This is a data-
    quality companion to plot_shadow_gap_cdf, not an alternative view of the
    same measurement: it answers "how much of each library's reads never
    even get a chance to show up in that CDF," which the CDF itself can't
    reveal since excluded reads simply aren't there to begin with.

    eligibility_by_lib: {libraryID: {"n_reads_total", "n_zero_runs",
    "n_one_run", "n_two_plus_runs"}} from read_gap_eligibility. Each bar's
    height is (n_zero_runs + n_one_run) / n_reads_total -- the zero-run vs
    one-run split itself isn't drawn on the bar (kept it uncluttered); it's
    still in main()'s own stderr breakdown for whoever wants the detail.
    """
    if canvas is None:
        print("pyx not available; skipping plot", file=sys.stderr); return

    libIDs = sorted(eligibility_by_lib)
    n_libs = len(libIDs)

    fractions = {}
    for lib in libIDs:
        e = eligibility_by_lib[lib]
        n_excluded = e["n_zero_runs"] + e["n_one_run"]
        fractions[lib] = n_excluded / e["n_reads_total"] if e["n_reads_total"] else 0.0

    g = graph.graphxy(
        width=7, height=7, xpos=0, ypos=0,
        x=graph.axis.linear(min=0.3, max=n_libs + 0.7, parter=None),
        y=graph.axis.linear(min=0, max=max(fractions.values(), default=1.0) * 1.25 or 1.0,
                            title=r"fraction of reads with $<$2 shadow calls"),
    )
    c = canvas.canvas()
    c.insert(g)

    bar_hw = 0.3
    for i, lib in enumerate(libIDs):
        xi  = i + 1
        e   = eligibility_by_lib[lib]
        col = _libcolor(lib, i)
        frac = fractions[lib]

        xa, ya = g.pos(xi - bar_hw, 0.0)
        xb, yb = g.pos(xi + bar_hw, frac)
        c.fill(path.rect(xa, ya, xb - xa, yb - ya), [col])
        c.stroke(path.rect(xa, ya, xb - xa, yb - ya), [style.linewidth.thin, color.gray(0.3)])

        xn0, yn0 = g.pos(xi, 0.0)
        c.text(xn0, yn0 - 0.2, "%s (n=%d)" % (lib.replace("_", r"\_"), e["n_reads_total"]),
              [pyx_text.halign.right, pyx_text.size.small, trafo.rotate(30)])

    c.text(g.xpos + g.width / 2., g.ypos + g.height + 0.5, title,
          [pyx_text.halign.center, pyx_text.size.normalsize])
    c.text(g.xpos + g.width / 2., g.ypos + g.height + 0.2,
          "(0 or 1 shadow calls -- no \"next\" call to measure a gap to)",
          [pyx_text.halign.center, pyx_text.size.scriptsize])

    c.writePDFfile(str(pdf_path))
    print(f"Wrote shadow-gap read-exclusion barplot to {pdf_path}.pdf", file=sys.stderr)


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
    numerator for View 8's stricter His-codon rate, "run_rate_counts":
    {gene: (n_qualifying_runs, n_reads)} at FIXED_CUTOFF (see
    _gene_run_counts_by_reads) -- the footprint/shadow-run-level counterpart
    to "site_counts": a qualifying run is an actual footprint-sized
    (>=MIN_RUN_NT nt) protected stretch, not a bare scored site, so this is
    what "shadow rate" means when the site-level rate isn't specific
    enough (e.g. for View 3-TE's log2FC-vs-TE scatter), "n_reads_total":
    total read count for this library (len(df), one row per read) -- the
    TRUE denominator for View 10-companion's exclusion fractions, since
    extract_shadow_runs simply omits a read from its output entirely if
    that read has zero qualifying runs, so nothing else returned here
    counts reads that never appear in any runs_by_cut entry at all.}.
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
    run_rate_counts = _gene_run_counts_by_reads(df, runs_by_cut.get(FIXED_CUTOFF, []),
                                                MIN_RUN_NT)

    return {"runs_by_cut": runs_by_cut, "site_counts": site_counts,
            "his_obs": his_obs, "his_shadow_counts": his_shadow_counts,
            "other_shadow_counts": other_shadow_counts,
            "region_read_coverage": region_read_coverage,
            "his_read_totals": his_read_totals,
            "region_run_counts": region_run_counts,
            "genes_observed": genes_observed,
            "his_centered_run_counts": his_centered_run_counts,
            "run_rate_counts": run_rate_counts,
            "n_reads_total": len(df)}


def main(args):
    global PALETTE, COLOR_MAP
    if canvas is not None:
        PALETTE = [color.cmyk(1, 0.5, 0, 0), color.cmyk(0, 1, 1, 0),
                   color.cmyk(0.4, 1, 0, 0), color.cmyk(1, 0, 1, 0.1),
                   color.cmyk(0, 0.5, 1, 0), color.cmyk(0.7, 0, 0, 0),
                   color.cmyk(0, 0, 0, 0.7), color.cmyk(0.3, 0, 1, 0.2)]

    parquetList, outPrefix, hisPicklePath, gtfPath = args[0], args[1], args[2], args[3]
    tePath = args[4] if len(args) > 4 else None
    flank_nt = int(args[5]) if len(args) > 5 else FLANK_NT
    colorMapPath = args[6] if len(args) > 6 else None
    if colorMapPath:
        COLOR_MAP = load_color_map(colorMapPath)
        print("Loaded manuscript colors for %d library key(s) from %s."
              % (len(COLOR_MAP), colorMapPath), file=sys.stderr)
    his_gpos_by_gene = load_his_codon_gpos(hisPicklePath)
    print("Loaded His codon positions for %d genes." % len(his_gpos_by_gene),
          file=sys.stderr)

    te_by_gene = load_translation_efficiency(tePath) if tePath else None
    if te_by_gene is not None:
        print("Loaded translation efficiency for %d genes." % len(te_by_gene),
              file=sys.stderr)

    genes = parse_gtf(gtfPath)
    flank_caps = compute_flank_caps(genes, flank_nt)
    region_len_by_gene = {g: region_lengths(gene, *flank_caps[g]) for g, gene in genes.items()}
    print("Loaded region lengths for %d genes from GTF (UTR5/UTR3 = flank_nt=%d, "
          "capped per gene at nearest-neighbor distance)." % (len(region_len_by_gene), flank_nt),
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
    if colorMapPath:
        unmatched = [lib for lib in libIDs if lib not in COLOR_MAP]
        if unmatched:
            print("  WARNING: no color found in %s for librar%s %s; falling back "
                  "to the default palette for %s." %
                  (colorMapPath, "y" if len(unmatched) == 1 else "ies",
                   unmatched, "it" if len(unmatched) == 1 else "them"), file=sys.stderr)

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

    # ---- View 3-TE: per-gene footprint/shadow-RUN rate, paired ----
    # Deliberately NOT the same rate as rates_by_lib above: rates_by_lib is a
    # per-NUCLEOTIDE rate (fraction of individual scored A sites with
    # P_B>cutoff, no size requirement -- a lone 1nt blip counts the same as
    # a site in the middle of a real footprint). For a TE correlation the
    # more meaningful "shadow" is an actual footprint-sized (>=MIN_RUN_NT nt)
    # protected RUN, same definition View 7's shadow density uses.
    run_read_totals_by_lib = {
        lib: {g: n_reads for g, (_n_runs, n_reads) in raw_by_lib[lib]["run_rate_counts"].items()}
        for lib in libIDs
    }
    common_run_rate_genes = _common_genes(run_read_totals_by_lib, MIN_READS_FOR_RUN_RATE)
    print("    footprint-run rate: %d genes common to all %d libraries (>=%d reads each)"
          % (len(common_run_rate_genes), len(libIDs), MIN_READS_FOR_RUN_RATE), file=sys.stderr)
    run_rates_by_lib = {
        lib: {g: raw_by_lib[lib]["run_rate_counts"][g][0] / raw_by_lib[lib]["run_rate_counts"][g][1]
              for g in common_run_rate_genes}
        for lib in libIDs
    }

    # ---- View 5: mean His-codon P_B per gene, paired ----
    his_counts_by_lib = {lib: _count_by_gene(raw_by_lib[lib]["his_obs"]) for lib in libIDs}
    common_his = _common_genes(his_counts_by_lib, MIN_HIS_OBS_PER_GENE)
    print("    His-codon sites: %d genes common to all %d libraries (>=%d His-codon sites each)"
          % (len(common_his), len(libIDs), MIN_HIS_OBS_PER_GENE), file=sys.stderr)
    his_gene_by_lib = {}
    for lib in libIDs:
        filtered = [o for o in raw_by_lib[lib]["his_obs"] if o["gene"] in common_his]
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

    # ---- View 10: nt gap between consecutive shadow calls on the same read ----
    # Reuses the GAP_CUTOFF (== FIXED_CUTOFF) runs already extracted per
    # library above (runs_by_cut), so no re-extraction is needed here -- just
    # the per-read consecutive-run gap (see shadow_run_gaps). Not gene-paired
    # (unlike Views 3/5/6/8): every read with >=2 shadow calls in a library
    # contributes, library-pooled, same "raw pooled" philosophy as View
    # 1-supplement's counts panel.
    gaps_by_lib = {lib: shadow_run_gaps(raw_by_lib[lib]["runs_by_cut"][GAP_CUTOFF])
                   for lib in libIDs}

    # ---- View 10-paired: same nt-gap CDF, restricted to genes common to all
    # libraries -- the raw-pooled CDF above isn't gene-matched, so a library
    # whose reads happen to cover a different set of genes could shift its
    # curve purely from WHICH genes it includes, not a real difference in
    # spacing (the same confound _common_genes exists to rule out for every
    # other cross-library comparison in this module, e.g. View 1/3/5/6/8).
    gap_counts_by_lib = {lib: _count_by_gene(gaps_by_lib[lib]) for lib in libIDs}
    common_gap_genes = _common_genes(gap_counts_by_lib, MIN_GAPS_PER_GENE)
    print("    shadow-gap CDF: %d genes common to all %d libraries (>=%d gap "
          "observations each)" % (len(common_gap_genes), len(libIDs), MIN_GAPS_PER_GENE),
          file=sys.stderr)
    gaps_by_lib_paired = {
        lib: [d for d in gaps_by_lib[lib] if d["gene"] in common_gap_genes]
        for lib in libIDs
    }

    # ---- View 10-companion: reads excluded from the gap CDF entirely ----
    # A read needs >=2 shadow runs at GAP_CUTOFF to contribute even one gap;
    # this reports, per library, what fraction of ALL reads (n_reads_total,
    # not just the ones that made it into runs_by_cut) never clear that bar
    # -- see read_gap_eligibility for why "zero runs" can't be counted
    # directly off runs_by_cut.
    eligibility_by_lib = {
        lib: read_gap_eligibility(raw_by_lib[lib]["runs_by_cut"][GAP_CUTOFF],
                                  raw_by_lib[lib]["n_reads_total"])
        for lib in libIDs
    }
    for lib in libIDs:
        e = eligibility_by_lib[lib]
        n_excluded = e["n_zero_runs"] + e["n_one_run"]
        print("    %s: %d/%d reads (%.1f%%) excluded from the gap CDF "
              "(%d zero-run, %d one-run, %d contribute >=1 gap)"
              % (lib, n_excluded, e["n_reads_total"],
                 100 * n_excluded / e["n_reads_total"] if e["n_reads_total"] else 0.0,
                 e["n_zero_runs"], e["n_one_run"], e["n_two_plus_runs"]), file=sys.stderr)

    print("Plotting combined figures across %d libraries..." % len(libIDs), file=sys.stderr)
    # View 1: footprint size, multi-panel over cutoffs (per-gene-then-aggregated freq)
    plot_footprint_sizes(dict(unmerged_by_cut), fp_edges,
                         "%s.footprint_sizes" % outPrefix)
    # View 1-supplement: same layout, raw pooled counts (no per-gene weighting)
    # -- simple sanity check against the per-gene-then-aggregated view above.
    plot_footprint_sizes(dict(unmerged_counts_by_cut), fp_edges,
                         "%s.footprint_sizes_counts" % outPrefix, ylabel="count")
    # View 3: per-gene call rate, magnitude comparison across libraries
    plot_gene_call_rates(rates_by_lib, "%s.gene_call_rates" % outPrefix,
                         connect_matched=True)
    # View 3-TE: per-gene footprint/shadow-RUN rate (run_rates_by_lib, NOT
    # the nucleotide-level rates_by_lib -- see its construction above),
    # log2FC relative to the phenol library, against translation efficiency
    # (Weinberg/Bartel RPF/RNA data), every library pooled onto one scatter,
    # colored per-library -- only produced if a TE file was passed on the
    # CLI (args[4]) AND a phenol library is present to serve as the reference.
    if te_by_gene is not None:
        phenol_libs = [lib for lib in libIDs if lib.lower().startswith("phenol")]
        if not phenol_libs:
            print("  WARNING: no phenol library found (libraryID starting with "
                  "'phenol'); skipping View 3-TE log2FC-vs-TE scatter.", file=sys.stderr)
        else:
            if len(phenol_libs) > 1:
                print(f"  WARNING: {len(phenol_libs)} phenol libraries found "
                      f"{phenol_libs}; using {phenol_libs[0]} as the reference.",
                      file=sys.stderr)
            reference_lib = phenol_libs[0]
            run_rate_log2fc_by_lib = compute_log2fc_vs_reference(run_rates_by_lib, reference_lib)
            plot_shadow_log2fc_vs_te(run_rate_log2fc_by_lib, te_by_gene,
                                     "%s.gene_run_rates.log2fc_vs_te" % outPrefix)
    # View 5: mean His-codon P_B per gene, paired across libraries.
    plot_gene_call_rates(his_gene_by_lib, "%s.his_codon_mean_pb_paired" % outPrefix,
                         title="Mean His-codon P$_B$ by library (paired per gene)",
                         min_n=MIN_HIS_OBS_PER_GENE, min_n_label="His-codon sites",
                         ylabel="mean P$_B$ at His-codon sites", connect_matched=True)
    # View 6: n_his_shadows / n_reads_with_a_His_site, per-gene rate, paired
    # across libraries -- normalized by reads that had the opportunity to
    # show a His-codon shadow, not by n(other shadows), so a library with
    # little protection overall (e.g. phenol/ribosome-less) doesn't get an
    # inflated ratio just from having a tiny other-shadow denominator.
    plot_gene_call_rates(his_vs_other_ratio_by_lib, "%s.his_codon_enrichment" % outPrefix,
                         title="His-shadow-count / reads-with-His-site by library",
                         min_n=MIN_HIS_READS_PER_GENE, min_n_label="qualifying reads",
                         ylabel="n(His shadows) / n(reads with His site)", connect_matched=True)
    # View 8: same as View 6, but the numerator only counts qualifying runs
    # CENTERED on a His codon rather than any His-codon site -- a stricter,
    # additional test of whether the His codon itself is driving the
    # protection, not a replacement for View 6.
    plot_gene_call_rates(his_centered_ratio_by_lib, "%s.his_codon_centered_enrichment" % outPrefix,
                         title="His-CENTERED-run-count / reads-with-His-site by library",
                         min_n=MIN_HIS_READS_PER_GENE, min_n_label="qualifying reads",
                         ylabel="n(His-centered runs) / n(reads with His site)", connect_matched=True)
    # View 7: shadow density (shadows/nt/read) by region -- one grouped barplot,
    # all regions and libraries together, not gene-paired across libraries.
    plot_region_density_grouped_bars(region_density_mean, region_density_n,
                                     "%s.shadow_density_by_region" % outPrefix,
                                     libIDs, region_names=REGION_NAMES,
                                     ylabel="mean shadows / nt / read")
    # View 10: nt gap between consecutive shadow calls, one step-CDF line per
    # library, pooled across all reads with >=2 shadow calls (P_B>=GAP_CUTOFF).
    plot_shadow_gap_cdf(gaps_by_lib, "%s.shadow_gap_cdf" % outPrefix)
    # View 10-paired: same nt-gap CDF, restricted to genes common to all
    # libraries -- an apples-to-apples comparison of the SAME genes' spacing
    # across libraries, rather than whichever genes each library happens to
    # have shadow calls from.
    plot_shadow_gap_cdf(gaps_by_lib_paired, "%s.shadow_gap_cdf_paired" % outPrefix)
    # View 10-companion: fraction of reads excluded from the gap CDF entirely
    # (0 or 1 shadow calls, no "next" call to measure a distance to).
    plot_read_gap_exclusion_bars(eligibility_by_lib, "%s.shadow_gap_read_exclusion" % outPrefix)


if __name__ == "__main__":
    main(sys.argv[1:])