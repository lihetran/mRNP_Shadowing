"""
riboseqGeneCoverage.py -- Liam Tran, August 2026

Simplest possible starting point, before any shadow-call matching: how many
ribo-seq reads land on each gene that appears in a shadow_calls.parquet,
and (now) how many shadow calls does that same gene have? Two bars per
gene, same gene order, sorted by ribo-seq depth -- no P-site offsets, no
run/window matching logic tying the two together, just two independent
per-gene frequencies plotted side by side so they're easy to eyeball
against each other. Ribo-seq reads are restricted to footprint-sized
lengths (21/22/28/29nt) -- still no offset/matching logic, just the same
basic "is this actually a ribosome footprint" length floor used everywhere
else in this codebase's ribo-seq handling.

shadow call here = a contiguous run (extract_shadow_runs, imported
read-only from polysomeShadowHMMQC.py, NOT reimplemented) of P_B >= 0.5
sites spanning >= 25 genomic nt -- a looser P_B cutoff than that module's
own FIXED_CUTOFF=0.7 default, per what was asked for here specifically.

Also produces an IGV-style per-position coverage TRACK (ribo-seq depth +
shadow-call depth, stacked, sharing a genomic-position x-axis, plus a
gene-model strip AND His-codon positions, from findHisCodonPositions.py's
pickle cache via polysomeShadowHMMQC.load_his_codon_gpos) for every gene
in shadowCallsParquet by default (or an explicit subset, if named) --
written one PDF per gene into a directory named after shadowCallsParquet
itself (its filename minus ".parquet"), alongside outPrefix's own
directory. This is a genuinely different view from the two bar charts
above: those give ONE number per gene (how much, in total), this shows
WHERE along the gene that signal actually sits, position by position, so
you can see whether ribo-seq density and shadow calls line up on the same
stretches of the gene (and on His codons specifically) or not.

Also produces, in the SAME per-gene figure as the coverage tracks
(plot_gene_track), a per-gene INDIVIDUAL-READS section stacked directly
above the gene model: the first NUM_READS=10 reads in shadowCallsParquet
for that gene, one panel per read, the RAW per-site confidence_score
trace (log10(1-P_A) -- unbounded, NOT a bare P_B probability) across
genomic position, colored per-position -- no run-averaging, each site
keeps its own individual score, matching how runHMMPerGene.py's
plot_signed_log_pyx plots this same transform -- see read_site_scores.

The shadow-call bar chart's second panel is a RATE (qualifying runs per
Nanopore read), not the raw run count -- a gene sampled by more Nanopore
reads racks up more raw runs purely from having more chances to see one,
independent of whether it's genuinely more protected per read. See
shadow_run_rate_per_gene.

Also produces a SECOND per-gene figure, plot_gene_shadow_pileup
("gene.pileup.pdf"), complementary to plot_gene_track's individual-reads
section (which only shows the first NUM_READS reads' full per-site
traces): an IGV "squished"-track-style pileup of EVERY qualifying shadow
call for the gene at once, one row per read, rows packed as tightly as
possible by non-overlap (pack_rows) rather than one row per read
regardless of overlap or a single smeared-together row. Each run is
colored by the average confidence_score over its own center 10nt (not a
whole-run average, which the ramp-up/ramp-down edges would dilute -- see
gene_shadow_pileup_data), on the same fixed quartile color scale as
plot_gene_track's reads. A read with multiple qualifying runs gets a thin
grey line connecting them across its own row. The figure's second panel
is the same shadow-call depth curve as plot_gene_track's, but banded by
the average confidence_score of the calls covering each position instead
of one flat color.

Run:
  python3 riboseqGeneCoverage.py shadowCallsParquet riboBamList.txt condition gtfFile hisPicklePath outPrefix [geneNames] [shadow_cutoff] [min_run_nt] [flank_nt]
where shadowCallsParquet is ONE library's shadow_calls.parquet (its
shadow_gene column defines which genes to look at), riboBamList.txt is
e.g. /data16/liam/working/260804_riboSeq_vs_PS/riboSeqBam.txt --
whitespace-delimited condition/rep/path rows (one BAM per row; multiple
reps of one condition are pooled), condition selects which rows to use
(e.g. "-3AT" or "+3AT", matched against the file's first column) for the
Nanopore library shadowCallsParquet came from, gtfFile is the yeast GTF
(for each gene's chrom/strand/span), hisPicklePath is
findHisCodonPositions.py's output (e.g.
210524_sacCer_HisCodonPositions.pickle), outPrefix names the output
plots, geneNames (optional) is a comma-separated list of genes to draw a
track for (default: every gene), shadow_cutoff/min_run_nt (optional)
override the P_B cutoff (default 0.5) and run-length floor (default
25nt) for what counts as a shadow call, and flank_nt (optional, default
150) MUST match whatever --flank_nt runHMMPerGene.py actually scored
shadowCallsParquet with -- the per-gene track/pileup figures widen past
the CDS by this much (capped per gene at the nearest-neighbor gap via
compute_flank_caps) so real shadow calls out there (this GTF barely
annotates any UTR, so runHMMPerGene.py pads the CDS instead -- see that
script's own docstring) aren't cut off the plotted range.
"""
import sys, os, math, collections
import pandas as pd
import pysam

from runHMMPerGene import parse_gtf, tex_escape, compute_flank_caps
from polysomeShadowHMMQC import extract_shadow_runs, load_his_codon_gpos

SHADOW_CUTOFF  = 0.5
MIN_RUN_NT     = 30
TARGET_LENGTHS = (21, 22, 28, 29)   # same footprint-length convention as riboseqShadowCorrelation.py
RIBO_SHORT_LENGTHS = (21, 22)       # plot_gene_track/plot_gene_shadow_pileup's ribo panel: split
RIBO_LONG_LENGTHS  = (28, 29)       # TARGET_LENGTHS into its two length classes, plotted separately
FLANK_NT       = 150                # per-gene plots: widen past the CDS by this many nt (capped per
                                     # gene at the nearest-neighbor gap via compute_flank_caps) --
                                     # MUST match whatever --flank_nt runHMMPerGene.py actually
                                     # scored shadowCallsParquet with, same convention as
                                     # polysomeShadowHMMQC.py's own flank_nt
NUM_READS      = 10                 # plot_gene_track: how many individual reads to show per gene
MAX_PILEUP_ROWS = 60                # plot_gene_shadow_pileup: row cap -- some genes are short and
                                     # densely covered enough (e.g. CCW12: 727 reads with calls) that
                                     # pack_rows's non-overlap packing barely compacts at all (rows
                                     # approach the peak depth), so without a cap the figure grows to
                                     # hundreds of rows tall. Reads beyond the cap are dropped (first-
                                     # come by pack_rows's own row assignment) and the omission is
                                     # reported in the subtitle and on stderr, same convention as
                                     # NUM_READS's truncation elsewhere in this file.
CONF_EPS       = 1e-6               # confidence-score floor, same role as plot_signed_log_pyx's eps
# CONF_RANGE is a FIXED scale (same color/axis meaning across every gene AND
# every library), not a per-gene dynamic one -- the whole point of a fixed
# scale is to let genuinely comparative claims ("library X has more
# confident calls than library Y") hold up, which a per-gene-rescaled color
# can't support. Anchored at P_A=0.1 ("very confident"): a site whose P_A
# has dropped below 0.1 is treated as fully saturated, and everything from
# there back to P_A=1 (background) is split into N_CONF_BINS quartiles for
# discrete coloring rather than a continuous gradient.
CONF_RANGE     = 1.0                # -log10(0.1): fully-saturated color at or past this score
N_CONF_BINS    = 4                  # discrete confidence quartiles from background to "very confident"


def load_ribo_bam_list(path, condition=None):
    """
    BAM paths from a whitespace-delimited list file. Each line is either
    "condition  rep  path" (the riboSeqBam.txt convention -- e.g. all 4
    Wu_2019 BAMs in one file, "-3AT"/"+3AT" x rep1/rep2) or a bare path
    with no condition/rep columns (the older flat-list convention this
    replaces). condition, if given, keeps only rows whose first column
    matches it exactly -- bare-path rows have no condition to match and
    are kept unconditionally. Every matching row's path is returned
    (reps of the same condition are pooled, same as before).
    """
    paths = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) >= 3:
                cond, _rep, bam_path = parts[0], parts[1], parts[2]
                if condition is not None and cond != condition:
                    continue
                paths.append(bam_path)
            else:
                paths.append(parts[0])
    return paths


def genes_in_shadow_calls(shadow_parquet_path):
    """Sorted list of every gene_name appearing in this shadow_calls.parquet."""
    df = pd.read_parquet(shadow_parquet_path, columns=["shadow_gene"])
    return sorted(df["shadow_gene"].unique())


def load_shadow_calls_df(shadow_parquet_path):
    """Full columns extract_shadow_runs needs (read_id, shadow_gene,
    shadow_gpos, shadow_P_B, shadow_region, absolute_indices), plus
    shadow_P_A for the per-read confidence-score plots (plot_gene_track's
    individual-reads section) -- extract_shadow_runs itself ignores the
    extra column, so one shared loader still works for both callers."""
    return pd.read_parquet(shadow_parquet_path,
                           columns=["read_id", "shadow_gene", "shadow_gpos",
                                   "shadow_P_B", "shadow_P_A",
                                   "shadow_region", "absolute_indices"])


def count_shadow_runs_per_gene(df, cutoff=SHADOW_CUTOFF, min_run_nt=MIN_RUN_NT):
    """
    {gene_name: n_qualifying_shadow_runs} -- a shadow call is a contiguous
    run (extract_shadow_runs) of P_B>=cutoff sites spanning >=min_run_nt
    genomic nt. Counts RUNS, not reads or sites -- one read can contribute
    more than one run if it has multiple separate qualifying stretches.
    """
    runs = extract_shadow_runs(df, cutoff)
    counts = collections.defaultdict(int)
    n_qualifying = 0
    for r in runs:
        if r["genomic_nt"] >= min_run_nt:
            counts[r["gene"]] += 1
            n_qualifying += 1
    print(f"  {n_qualifying}/{len(runs)} runs clear P_B>={cutoff}, >={min_run_nt}nt.",
          file=sys.stderr)
    return dict(counts)


def count_reads_per_gene(df):
    """
    {gene_name: n_reads} -- number of Nanopore reads (rows) scored for
    each gene in shadow_calls.parquet. The depth denominator
    count_shadow_runs_per_gene's raw counts need normalizing by: a gene
    sampled by more Nanopore reads racks up more raw qualifying runs
    purely from having more chances to see one, not necessarily from
    being more protected per read (see shadow_run_rate_per_gene).
    """
    return df["shadow_gene"].value_counts().to_dict()


def shadow_run_rate_per_gene(shadow_counts, read_counts):
    """
    {gene_name: n_qualifying_runs / n_reads} -- shadow-call runs per
    Nanopore read, the depth-normalized counterpart to
    count_shadow_runs_per_gene's raw count. Only genes with >=1 read
    contribute (undefined otherwise); a gene with reads but zero
    qualifying runs still gets a true rate of 0.0, not an absence.
    """
    return {g: shadow_counts.get(g, 0) / n for g, n in read_counts.items() if n > 0}


def count_ribo_reads_per_gene(bam_paths, genes, gene_names,
                              require_sense=True, require_unique=True,
                              target_lengths=TARGET_LENGTHS):
    """
    {gene_name: n_reads} -- total ribo-seq reads overlapping each gene's
    genomic span, pooled across every BAM in bam_paths. Restricted to
    footprint-sized reads (length in target_lengths, default {21,22,28,29}
    -- same convention as riboseqShadowCorrelation.py's PSITE_OFFSETS),
    NH==1 (unless require_unique=False), and sense-orientation to the gene
    (unless require_sense=False) -- the same basic quality filters used
    everywhere else in this codebase's ribo-seq handling. Pass
    target_lengths=None to skip the length filter entirely (raw depth,
    any read length).
    """
    counts = collections.defaultdict(int)
    for bam_path in bam_paths:
        print(f"  scanning {bam_path} ...", file=sys.stderr)
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for gname in gene_names:
                gene = genes[gname]
                for read in bam.fetch(gene["chrom"], gene["gene_start"], gene["gene_end"]):
                    if read.is_unmapped:
                        continue
                    if target_lengths is not None and read.query_length not in target_lengths:
                        continue
                    if require_unique and read.has_tag("NH") and read.get_tag("NH") != 1:
                        continue
                    read_is_sense = (read.is_reverse == (gene["strand"] == "-"))
                    if require_sense and not read_is_sense:
                        continue
                    counts[gname] += 1
    return dict(counts)


def ribo_coverage_track(bam_paths, gene, target_lengths=TARGET_LENGTHS,
                        require_sense=True, require_unique=True):
    """
    {gpos: depth} -- per-POSITION ribo-seq read depth across one gene's
    genomic span, pooled across every BAM (IGV-pileup style). Uses
    read.get_blocks() -- the CIGAR-aware list of actually-aligned
    sub-intervals, split at N/D (intron/deletion) operations -- rather
    than the raw [reference_start, reference_end) span, so a read whose
    alignment skips over an intron (spliced, or straddling a splice
    junction) doesn't get counted as if it covered that intron too. Same
    quality filters as count_ribo_reads_per_gene -- this is that
    function's per-position counterpart, for one gene at a time rather
    than one total per gene.
    """
    depth = collections.Counter()
    for bam_path in bam_paths:
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for read in bam.fetch(gene["chrom"], gene["gene_start"], gene["gene_end"]):
                if read.is_unmapped:
                    continue
                if target_lengths is not None and read.query_length not in target_lengths:
                    continue
                if require_unique and read.has_tag("NH") and read.get_tag("NH") != 1:
                    continue
                read_is_sense = (read.is_reverse == (gene["strand"] == "-"))
                if require_sense and not read_is_sense:
                    continue
                for block_start, block_end in read.get_blocks():
                    for gpos in range(block_start, block_end):
                        depth[gpos] += 1
    return depth


def _padded_cds_segments(gene, flank_5p, flank_3p):
    """
    A copy of gene["cds"] with its first/last (in transcript order)
    interval's outer boundary padded by flank_5p (before the start
    codon)/flank_3p (after the stop codon) nt -- single-exon genes pad
    both ends of the sole interval. Same technique (and the same
    duplicated-per-file convention) as runHMMPerGene.py's
    _padded_cds_segments/compute_flank_caps: shadow_calls.parquet's
    shadow_gpos can now include real positions out here (the HMM scoring
    pipeline was fixed to pad the CDS the same way instead of relying on
    GTF-annotated UTR features, which this GTF barely has), so
    shadow_coverage_track's own exon clipping needs to allow this region
    too, not just gene["exons"].
    """
    cds = list(gene["cds"])
    if len(cds) == 1:
        s, e = cds[0]
        lo_pad, hi_pad = (flank_5p, flank_3p) if gene["strand"] == "+" else (flank_3p, flank_5p)
        cds[0] = (max(0, s - lo_pad), e + hi_pad)
    else:
        first_s, first_e = cds[0]
        last_s, last_e = cds[-1]
        if gene["strand"] == "+":
            cds[0]  = (max(0, first_s - flank_5p), first_e)
            cds[-1] = (last_s, last_e + flank_3p)
        else:
            cds[0]  = (first_s, first_e + flank_5p)
            cds[-1] = (max(0, last_s - flank_3p), last_e)
    return cds


def shadow_coverage_track(shadow_df, gene_name, gene, cutoff=SHADOW_CUTOFF, min_run_nt=MIN_RUN_NT,
                          flank_5p=0, flank_3p=0):
    """
    {gpos: depth} -- per-position count of qualifying shadow-call runs
    (extract_shadow_runs, P_B>=cutoff, >=min_run_nt) covering that gpos,
    for one gene -- the shadow-call analog of ribo_coverage_track. Depth
    at a position = how many separate reads' protected runs span it, not
    just whether >=1 does.

    A run's gpos_lo/gpos_hi come from its member sites' min/max gpos, but
    those sites are TRANSCRIPT-adjacent, not necessarily genomically
    adjacent -- a run whose flanking sites sit on either side of an
    intron has a numeric gpos_lo..gpos_hi span that jumps clean across
    it. Naively filling every position in that range would wrongly mark
    the intron itself as "covered" by the run, when no scored site (and
    no real mRNA sequence) is actually there. Positions are clipped to
    the gene's own annotated exon intervals (gene["exons"]), UNIONED with
    a flank_5p/flank_3p-padded CDS region if given (see
    _padded_cds_segments) -- default flank_5p=flank_3p=0 reproduces the
    original exons-only behavior exactly, so existing callers
    (plot_gene_track/plot_gene_shadow_pileup) are unaffected; callers
    that DO pass a flank (e.g. shadowMetagene.py, working with a
    shadow_calls.parquet scored by a flank-padded runHMMPerGene.py) get
    real per-position depth out there instead of a silently-clipped wall.

    shadow_df should already be restricted to this gene (a caller with a
    multi-gene df pays for run-extraction on every OTHER gene too
    otherwise) -- count_shadow_runs_per_gene doesn't need this since it
    only cares about run-level counts, not per-position depth, but this
    does.
    """
    exons = gene.get("exons", [])
    padded_cds = _padded_cds_segments(gene, flank_5p, flank_3p) if (flank_5p or flank_3p) else []
    runs = extract_shadow_runs(shadow_df, cutoff)
    depth = collections.Counter()
    for r in runs:
        if r["gene"] != gene_name or r["genomic_nt"] < min_run_nt:
            continue
        for gpos in range(r["gpos_lo"], r["gpos_hi"] + 1):
            if (any(s <= gpos < e for s, e in exons) or
                    any(s <= gpos < e for s, e in padded_cds)):
                depth[gpos] += 1
    return depth


def confidence_score(p_a, eps=CONF_EPS):
    """
    -log10(P_A): 0 at P_A=1 (fully confident the site is UNPROTECTED --
    the best this score can do, since P_A tops out at 1), increasingly
    positive as that confidence in "unprotected" drops -- through a mild
    value at genuine 50/50 uncertainty (P_A=0.5 -> 0.301) up to a
    ceiling of -log10(CONF_EPS) (=6 by default) for a site confidently
    PROTECTED (P_A near/at the eps floor). Log-scaled rather than linear
    so P_A=0.0001 reads as clearly more confidently protected than
    P_A=0.1, not indistinguishable the way a linear 0-1 map would leave
    them. Uses only the P_A column (not P_B), per what was asked for
    here specifically. Always >= 0 -- there is no negative side.
    """
    return -math.log10(max(p_a, eps))


# RGB stops sampled from seaborn's "flare" sequential colormap at the
# N_CONF_BINS quartile midpoints (light peach -> deep magenta/purple),
# via sns.color_palette("flare", as_cmap=True). Hardcoded here so this
# script doesn't need seaborn installed just to plot.
CONF_COLORS = [
    (0.929, 0.689, 0.504),
    (0.872, 0.363, 0.360),
    (0.604, 0.210, 0.439),
    (0.294, 0.137, 0.384),
]


def confidence_bin(score, conf_range=CONF_RANGE, n_bins=N_CONF_BINS):
    """
    Quantize a confidence_score into one of n_bins equal-width quartile
    bins spanning [0, conf_range] (0 = background bin, n_bins-1 = "very
    confident" bin). Scores at or past conf_range land in the last bin.
    Discrete rather than continuous so the color scale reads as a small
    fixed set of confidence tiers (matching across every gene/library)
    instead of a smooth gradient where two nearby shades are hard to
    tell apart.
    """
    s = min(max(score, 0.0), conf_range)
    bin_width = conf_range / n_bins
    return min(n_bins - 1, int(s / bin_width))


def confidence_color(score, conf_range=CONF_RANGE):
    """
    "flare"-palette quartile coloring for a confidence_score (always >=
    0): light peach for the background-most quartile (confidently
    unprotected), shifting through two intermediate shades to deep
    purple for the "very confident" quartile (score at or past
    conf_range, i.e. P_A at or below the anchor), so a real footprint
    call visually pops out against the light background instead of
    blending in.
    """
    from pyx import color
    return color.rgb(*CONF_COLORS[confidence_bin(score, conf_range)])


def draw_confidence_colorbar(c, x0, y_bottom, y_top, score_hi):
    """
    Vertical color bar, to the side of the plot -- replaces a horizontal
    row of swatches above it, which got hard to read once the row grew to
    4 items plus adjacent legends. Shows the same 4 confidence quartile
    colors confidence_color uses, stacked bottom (background, P_A~1) to
    top (very confident, P_A around 0.1 or less), each with its P_A range
    labeled alongside it. Shared by plot_gene_track and
    plot_gene_shadow_pileup so both figures' confidence-colored bands/
    dots mean the same thing at a glance.
    """
    from pyx import style, text as pyx_text, path

    bar_w = 0.35
    quartile_labels = [
        "background (P_A 0.56-1)",
        "weak call (P_A 0.32-0.56)",
        "moderate call (P_A 0.18-0.32)",
        "very confident call (P_A around 0.1 or less)",
    ]
    n = len(quartile_labels)
    seg_h = (y_top - y_bottom) / n
    for i, lab in enumerate(quartile_labels):
        sc = (i + 0.5) * (score_hi / n)
        y = y_bottom + i * seg_h
        c.fill(path.rect(x0, y, bar_w, seg_h), [confidence_color(sc, score_hi)])
        c.text(x0 + bar_w + 0.15, y + seg_h / 2.,
              tex_escape(lab), [pyx_text.valign.middle, pyx_text.size.tiny])
    c.stroke(path.rect(x0, y_bottom, bar_w, y_top - y_bottom), [style.linewidth.thin])


def read_site_scores(row):
    """
    Per-scored-site RAW confidence_score for one read (a row from
    shadow_calls.parquet), sorted by GENOMIC position (genomic rather
    than transcript position so this can share an x-axis with
    plot_gene_track's coverage panels, which are already genomic-
    position, IGV-style): [(gpos, score), ...] for the whole read.

    confidence_score is log10(1-P_A) -- NOT a bare P_B probability,
    unbounded below 0, so it routinely has |value| > 1.

    No run-averaging here -- every site keeps its own individual score,
    matching runHMMPerGene.py's plot_signed_log_pyx, which plots the raw
    per-site transform directly rather than flattening any stretch of it.
    """
    gp = [int(g) for g in row.shadow_gpos]
    pa = [float(p) for p in row.shadow_P_A]
    order = sorted(range(len(gp)), key=lambda k: gp[k])
    gp_s = [gp[k] for k in order]
    pa_s = [pa[k] for k in order]
    return list(zip(gp_s, (confidence_score(a) for a in pa_s)))


def draw_gene_model_strip(c, gene, lo, hi, his_in_range, panel_w, model_h):
    """
    The gene-model strip shared by plot_gene_track and
    plot_gene_shadow_pileup: UTR5/CDS/UTR3 as filled rects, His-codon
    positions as thin vertical lines, "5'"/"3'" text at the gene's own
    start/end (not a single arrow glyph, which would sit on top of the
    "- strand"/"+ strand" title text and contradict it visually for
    minus-strand genes). Returns (g_model, col_his) -- g_model so callers
    link their own panels' x-axis to it (graph.axis.linkedaxis), col_his
    so callers can draw the same His-codon color in their own panels.

    lo/hi are the PANEL's axis bounds -- callers now widen these past the
    gene's own gene_start/gene_end to fit the flank_5p/flank_3p region
    runHMMPerGene.py's flank fix scores (see plot_gene_track/
    plot_gene_shadow_pileup), so the "5'"/"3'" labels below are placed at
    gene["gene_start"]/["gene_end"] specifically (the TRUE gene boundary),
    not lo/hi -- otherwise they'd drift out to the flank edge instead of
    marking the real start/stop codon.
    """
    from pyx import graph, color, style, text as pyx_text, path

    col_his = color.cmyk(0, 1, 1, 0)
    col_cds = color.cmyk(0, 0, 0, 0.75)
    col_utr = color.cmyk(0, 0, 0, 0.25)

    g_model = graph.graphxy(
        width=panel_w, height=model_h, xpos=0, ypos=0,
        x=graph.axis.linear(min=lo, max=hi, title=f"genomic position ({tex_escape(gene['chrom'])})"),
        y=graph.axis.linear(min=0, max=1, parter=None))
    c.insert(g_model)
    for seg, col in ((gene.get("utr5", []), col_utr), (gene.get("utr3", []), col_utr),
                    (gene.get("cds", []), col_cds)):
        for s, e in seg:
            x0, y0 = g_model.pos(max(s, lo), 0.25)
            x1, y1 = g_model.pos(min(e, hi), 0.75)
            c.fill(path.rect(x0, y0, x1 - x0, y1 - y0), [col])
    for hg in his_in_range:
        g_model.plot(graph.data.function(f"x(y)={hg}", min=0, max=1),
                    [graph.style.line([col_his, style.linewidth.thin, style.linestyle.solid])])
    gene_lo, gene_hi = gene["gene_start"], gene["gene_end"]
    if gene["strand"] == "+":
        five_x, five_ha, three_x, three_ha = gene_lo, pyx_text.halign.left, gene_hi, pyx_text.halign.right
    else:
        five_x, five_ha, three_x, three_ha = gene_hi, pyx_text.halign.right, gene_lo, pyx_text.halign.left
    fx, fy = g_model.pos(five_x, 0.5)
    tx, ty = g_model.pos(three_x, 0.5)
    c.text(fx, fy, "5'", [five_ha, pyx_text.valign.middle, pyx_text.size.scriptsize])
    c.text(tx, ty, "3'", [three_ha, pyx_text.valign.middle, pyx_text.size.scriptsize])
    return g_model, col_his


def gene_shadow_pileup_data(gene_df, gene_name, gene, cutoff=SHADOW_CUTOFF,
                            min_run_nt=MIN_RUN_NT, center_nt=10, flank_5p=0, flank_3p=0):
    """
    Everything plot_gene_shadow_pileup needs, computed in one pass over
    gene_df so read_site_scores (an array copy + sort per read) doesn't
    get paid for twice:
      - runs_by_read: {read_id: [run_dict, ...]} -- qualifying runs
        (extract_shadow_runs, P_B>=cutoff, >=min_run_nt) for this gene,
        each augmented with a "score" key: the average confidence_score
        (read_site_scores) over the center_nt genomic positions at the
        run's own midpoint, NOT a whole-run average -- a run's ramp-up/
        ramp-down edges are lower-confidence transition zones that would
        dilute that, the same reasoning that kept read_site_scores itself
        from averaging across a run. A run whose own center_nt window
        happens to have no scored site is left out of runs_by_read (no
        center score to give it), but -- see pos_scores below -- that
        exclusion is scoped to THIS dict only.
      - pos_scores: {gpos: [score, ...]} -- for coloring the depth panel
        by confidence at each position rather than a single flat color.
        NOT a per-site average -- a scored ("A") site is only ~1/4 of any
        given nt, but shadow_coverage_track's depth fills EVERY nt across
        a run's span (a protected footprint covers contiguous sequence,
        not just its A's), so a per-site-only average leaves ~3/4 of
        real depth positions with no entry at all -- defaulting those to
        score 0 ("background") would be actively wrong, not just sparse:
        a C/G/T in the middle of a well-established run isn't
        "confidently unprotected," it just has no site of its own to
        score. Instead, each qualifying run broadcasts ONE representative
        score across its ENTIRE span (every nt it covers, not just its
        A's) -- the same center_nt-window average used for that run's
        own runs_by_read entry above, so the depth panel's color and the
        pileup panel's per-run segment color agree; falling back to that
        run's own full-length average (its own A-sites only, still) if
        the center_nt window specifically has none, so a run that missed
        runs_by_read for that reason still contributes its own real
        confidence to the depth panel rather than nothing. Positions are
        clipped to the gene's annotated exons, unioned with a
        flank_5p/flank_3p-padded CDS region if given (see
        _padded_cds_segments) -- same convention as shadow_coverage_track,
        so this dict's positions cover the same range as that function's
        depth values instead of silently defaulting past the true exons.
    Reads with zero qualifying runs contribute to neither dict.
    """
    exons = gene.get("exons", [])
    padded_cds = _padded_cds_segments(gene, flank_5p, flank_3p) if (flank_5p or flank_3p) else []
    runs = [r for r in extract_shadow_runs(gene_df, cutoff)
           if r["gene"] == gene_name and r["genomic_nt"] >= min_run_nt]
    runs_by_read_id = collections.defaultdict(list)
    for r in runs:
        runs_by_read_id[r["read_id"]].append(r)

    half = center_nt // 2
    runs_by_read = {}
    pos_scores = collections.defaultdict(list)
    for row in gene_df.itertuples(index=False):
        read_runs = runs_by_read_id.get(row.read_id)
        if not read_runs:
            continue
        sites = dict(read_site_scores(row))
        scored_runs = []
        for r in read_runs:
            center = (r["gpos_lo"] + r["gpos_hi"]) // 2
            window = [sites[g] for g in range(center - half, center + half)
                     if g in sites]
            if window:
                run_score = sum(window) / len(window)
                scored_runs.append({**r, "score": run_score})
            else:
                all_sites = [sites[g] for g in range(r["gpos_lo"], r["gpos_hi"] + 1)
                            if g in sites]
                run_score = sum(all_sites) / len(all_sites) if all_sites else None
            if run_score is not None:
                for gpos in range(r["gpos_lo"], r["gpos_hi"] + 1):
                    if (any(s <= gpos < e for s, e in exons) or
                            any(s <= gpos < e for s, e in padded_cds)):
                        pos_scores[gpos].append(run_score)
        if scored_runs:
            runs_by_read[row.read_id] = scored_runs
    return runs_by_read, pos_scores


def pack_rows(spans):
    """
    Greedy interval packing (the classic minimum-rows-for-non-overlap
    algorithm): sort spans by start, assign each to the first row whose
    last-placed span already ends before this one starts, else open a
    new row. Returns (row_of, n_rows) -- row_of is a parallel list of row
    indices (0 = bottom row). This is what makes plot_gene_shadow_pileup
    a "compact multi-row" layout rather than one row per read regardless
    of overlap.
    """
    order = sorted(range(len(spans)), key=lambda i: spans[i][0])
    row_ends = []
    row_of = [None] * len(spans)
    for i in order:
        lo, hi = spans[i]
        for row, end in enumerate(row_ends):
            if end < lo:
                row_ends[row] = hi
                row_of[i] = row
                break
        else:
            row_ends.append(hi)
            row_of[i] = len(row_ends) - 1
    return row_of, len(row_ends)


def plot_gene_shadow_pileup(gene_name, gene, ribo_depth_short, ribo_depth_long, shadow_depth,
                            gene_df, pdf_path, his_gpos=None, shadow_cutoff=SHADOW_CUTOFF,
                            min_run_nt=MIN_RUN_NT, flank_5p=0, flank_3p=0):
    """
    A second per-gene figure, complementary to plot_gene_track's
    individual-reads panels (which only show the first NUM_READS reads,
    one full per-site trace each): this shows EVERY qualifying shadow
    call for the gene at once, IGV "squished"-track style -- one row per
    read, rows packed as tightly as possible by non-overlap (pack_rows),
    not one row per read regardless of overlap and not a single
    "collapsed" row either (which would smear overlapping calls into an
    unreadable pile for anything but the lowest-depth genes).

    Each read's own qualifying runs are drawn as thick colored segments
    on its row, colored by confidence_color of that run's OWN score --
    the average confidence_score over the run's center 10nt (see
    gene_shadow_pileup_data), on the same fixed CONF_RANGE/quartile scale
    plot_gene_track's read panels use, so colors mean the same thing in
    both figures and across libraries. A read with >1 qualifying run gets
    a thin grey line spanning its own row's full extent, connecting its
    calls, so which calls came off the same molecule is still visible
    even though many unrelated reads share a row.

    Second and third panels: the same shadow-call depth and ribo-seq
    depth curves as plot_gene_track's own panels -- shadow-call depth
    colored in bands by the AVERAGE confidence_score of the qualifying-
    run sites covering each position (gene_shadow_pileup_data's
    pos_scores; positions with no qualifying call default to score 0,
    i.e. the same "background" bin the fixed color scale already uses
    for P_A~1) instead of a single flat color, so this panel shows not
    just how much shadow-call depth a position has but how confident
    those calls are; ribo-seq depth split into its two length classes
    (matching plot_gene_track's own ribo panel) -- light grey for the
    RIBO_SHORT_LENGTHS (21/22nt) footprints, dark grey for
    RIBO_LONG_LENGTHS (28/29nt) -- so the two figures can be read side
    by side.

    flank_5p/flank_3p (typically a gene's own compute_flank_caps entry,
    same convention as runHMMPerGene.py) widen the panel's own axis range
    past gene["gene_start"]/["gene_end"] -- shadow_calls.parquet's
    shadow_gpos can now include real positions out there (the HMM
    scoring pipeline pads the CDS the same way instead of relying on
    GTF-annotated UTR features, which this GTF barely has), and without
    this the flank-region calls/coverage ribo_depth_*/shadow_depth
    already carry (see main()'s padded ribo_coverage_track fetch and
    shadow_coverage_track's own flank_5p/flank_3p) would just fall
    outside this panel's plotted range and never be drawn.
    """
    from pyx import canvas, graph, color, style, text as pyx_text, path

    col_ribo_short = color.grey(0.75)
    col_ribo_long  = color.grey(0.3)

    # widen past the true gene span by the flank on the appropriate
    # genomic side (strand-aware, same mapping as get_gene_df/
    # _padded_cds_segments) -- draw_gene_model_strip still uses
    # gene["gene_start"]/["gene_end"] directly for the 5'/3' labels, so
    # widening lo/hi here doesn't move those.
    if gene["strand"] == "+":
        lo, hi = gene["gene_start"] - flank_5p, gene["gene_end"] + flank_3p
    else:
        lo, hi = gene["gene_start"] - flank_3p, gene["gene_end"] + flank_5p
    xs = list(range(lo, hi))
    ribo_short_ys = [ribo_depth_short.get(x, 0) for x in xs]
    ribo_long_ys  = [ribo_depth_long.get(x, 0) for x in xs]
    shadow_ys = [shadow_depth.get(x, 0) for x in xs]
    his_in_range = sorted(g for g in (his_gpos or ()) if lo <= g < hi)

    runs_by_read, pos_scores = gene_shadow_pileup_data(
        gene_df, gene_name, gene, shadow_cutoff, min_run_nt,
        flank_5p=flank_5p, flank_3p=flank_3p)

    read_ids = list(runs_by_read)
    spans = [(min(r["gpos_lo"] for r in runs_by_read[rid]),
             max(r["gpos_hi"] for r in runs_by_read[rid]))
            for rid in read_ids]
    row_of, n_rows = pack_rows(spans)

    n_dropped_reads = 0
    if n_rows > MAX_PILEUP_ROWS:
        n_rows_needed = n_rows
        keep = [i for i in range(len(read_ids)) if row_of[i] < MAX_PILEUP_ROWS]
        n_dropped_reads = len(read_ids) - len(keep)
        read_ids = [read_ids[i] for i in keep]
        spans    = [spans[i] for i in keep]
        row_of   = [row_of[i] for i in keep]
        n_rows   = MAX_PILEUP_ROWS
        print(f"plot_gene_shadow_pileup: {gene_name} needed {n_rows_needed} rows, "
              f"capped at {MAX_PILEUP_ROWS} -- {n_dropped_reads} read(s) omitted", file=sys.stderr)

    score_hi = CONF_RANGE
    panel_w, ribo_h, shadow_h, model_h, gap = 14, 3.5, 3.5, 0.6, 1.1
    row_h = 0.12
    pileup_h = max(row_h, n_rows * row_h)
    ribo_max = max(ribo_short_ys + ribo_long_ys) * 1.1 if any(ribo_short_ys) or any(ribo_long_ys) else 1.0
    shadow_max = max(shadow_ys) * 1.1 if any(shadow_ys) else 1.0

    c = canvas.canvas()
    g_model, col_his = draw_gene_model_strip(c, gene, lo, hi, his_in_range, panel_w, model_h)

    pileup_ypos = model_h + gap
    g_pileup = graph.graphxy(
        width=panel_w, height=pileup_h, xpos=0, ypos=pileup_ypos,
        x=graph.axis.linkedaxis(g_model.axes["x"], painter=graph.axis.painter.linked()),
        y=graph.axis.linear(min=0, max=max(1, n_rows), title="reads", parter=None))
    c.insert(g_pileup)
    for hg in his_in_range:
        g_pileup.plot(graph.data.function(f"x(y)={hg}", min=0, max=max(1, n_rows)),
                     [graph.style.line([col_his, style.linewidth.thin, style.linestyle.dashed])])
    for idx, rid in enumerate(read_ids):
        row = row_of[idx]
        runs = runs_by_read[rid]
        span_lo, span_hi = spans[idx]
        if len(runs) > 1:
            x0, y0 = g_pileup.pos(span_lo, row + 0.5)
            x1, y1 = g_pileup.pos(span_hi, row + 0.5)
            c.stroke(path.line(x0, y0, x1, y1), [color.grey(0.6), style.linewidth.thin])
        for r in runs:
            x0, y0 = g_pileup.pos(r["gpos_lo"], row + 0.15)
            x1, y1 = g_pileup.pos(r["gpos_hi"], row + 0.85)
            c.fill(path.rect(x0, y0, max(x1 - x0, 0.01), y1 - y0),
                  [confidence_color(r["score"], score_hi)])

    shadow_ypos = pileup_ypos + pileup_h + gap
    g_shadow = graph.graphxy(
        width=panel_w, height=shadow_h, xpos=0, ypos=shadow_ypos,
        x=graph.axis.linkedaxis(g_model.axes["x"], painter=graph.axis.painter.linked()),
        y=graph.axis.linear(min=0, max=shadow_max, title="shadow-call depth"))
    c.insert(g_shadow)
    # banded by confidence_bin rather than one plot() call per position --
    # a few hundred bin-transition segments instead of thousands of
    # single-nt draws, same idea as the legend's discrete quartile swatches
    seg_start = 0
    prev_bin = None
    for i, x in enumerate(xs):
        avg_score = (sum(pos_scores[x]) / len(pos_scores[x])) if pos_scores.get(x) else 0.0
        b = confidence_bin(avg_score, score_hi)
        if prev_bin is None:
            prev_bin = b
        elif b != prev_bin:
            seg_xs, seg_ys = xs[seg_start:i + 1], shadow_ys[seg_start:i + 1]
            rep_score = (prev_bin + 0.5) * (score_hi / N_CONF_BINS)
            g_shadow.plot(graph.data.points(list(zip(seg_xs, seg_ys)), x=1, y=2),
                         [graph.style.line([confidence_color(rep_score, score_hi), style.linewidth.Thick])])
            seg_start = i
            prev_bin = b
    if prev_bin is not None:
        seg_xs, seg_ys = xs[seg_start:], shadow_ys[seg_start:]
        rep_score = (prev_bin + 0.5) * (score_hi / N_CONF_BINS)
        g_shadow.plot(graph.data.points(list(zip(seg_xs, seg_ys)), x=1, y=2),
                     [graph.style.line([confidence_color(rep_score, score_hi), style.linewidth.Thick])])
    for hg in his_in_range:
        g_shadow.plot(graph.data.function(f"x(y)={hg}", min=0, max=shadow_max),
                     [graph.style.line([col_his, style.linewidth.thin, style.linestyle.dashed])])

    ribo_ypos = shadow_ypos + shadow_h + gap
    g_ribo = graph.graphxy(
        width=panel_w, height=ribo_h, xpos=0, ypos=ribo_ypos,
        x=graph.axis.linkedaxis(g_model.axes["x"], painter=graph.axis.painter.linked()),
        y=graph.axis.linear(min=0, max=ribo_max, title="ribo-seq depth"))
    c.insert(g_ribo)
    g_ribo.plot(graph.data.points(list(zip(xs, ribo_long_ys)), x=1, y=2),
               [graph.style.line([col_ribo_long, style.linewidth.Thick])])
    g_ribo.plot(graph.data.points(list(zip(xs, ribo_short_ys)), x=1, y=2),
               [graph.style.line([col_ribo_short, style.linewidth.Thick])])
    for hg in his_in_range:
        g_ribo.plot(graph.data.function(f"x(y)={hg}", min=0, max=ribo_max),
                   [graph.style.line([col_his, style.linewidth.thin, style.linestyle.dashed])])

    top_ypos = ribo_ypos + ribo_h + gap
    c.text(g_ribo.xpos + g_ribo.width / 2., top_ypos + 0.5,
          tex_escape(gene_name),
          [pyx_text.halign.center, pyx_text.size.large])
    omitted_note = f", {n_dropped_reads} more read(s) omitted (row cap)" if n_dropped_reads else ""
    c.text(g_ribo.xpos + g_ribo.width / 2., top_ypos + 0.1,
          f"{tex_escape(gene['chrom'])}:{lo:,}-{hi:,}, {gene['strand']} strand -- "
          f"{len(his_in_range)} His codon(s) -- shadow-call P$_B>${shadow_cutoff}, "
          f"len$>${min_run_nt}nt, {len(read_ids)} reads with qualifying calls, "
          f"{n_rows} pileup rows{omitted_note}",
          [pyx_text.halign.center, pyx_text.size.scriptsize])
    draw_confidence_colorbar(c, panel_w + 2.5, pileup_ypos, top_ypos, score_hi)
    ribo_leg_y = top_ypos - 0.3
    c.stroke(path.line(0.2, ribo_leg_y, 0.85, ribo_leg_y), [col_ribo_short, style.linewidth.Thick])
    c.text(1.05, ribo_leg_y, f"ribo-seq {RIBO_SHORT_LENGTHS[0]}/{RIBO_SHORT_LENGTHS[1]}nt",
          [pyx_text.valign.middle, pyx_text.size.tiny])
    c.stroke(path.line(3.5, ribo_leg_y, 4.15, ribo_leg_y), [col_ribo_long, style.linewidth.Thick])
    c.text(4.35, ribo_leg_y, f"ribo-seq {RIBO_LONG_LENGTHS[0]}/{RIBO_LONG_LENGTHS[1]}nt",
          [pyx_text.valign.middle, pyx_text.size.tiny])

    c.writePDFfile(str(pdf_path))
    print(f"Wrote {pdf_path}", file=sys.stderr)


def plot_gene_track(gene_name, gene, ribo_depth_short, ribo_depth_long, shadow_depth, gene_df,
                    pdf_path, his_gpos=None, shadow_cutoff=SHADOW_CUTOFF, min_run_nt=MIN_RUN_NT,
                    num_reads=NUM_READS, flank_5p=0, flank_3p=0):
    """
    Combined per-gene figure, PyX/PDF (see runHMMPerGene.py's
    plot_pb_by_tx_pyx for the house style this follows: canvas + stacked
    graphxy panels at manually-set ypos, graph.axis.linkedaxis to share
    one x-axis without repainting it on every panel, graph.data.function's
    "x(y)=CONST" trick for vertical reference lines, cmyk colors). All
    panels share ONE genomic-position x-axis (IGV-style, left-to-right
    regardless of gene strand), bottom to top:
      - gene-model strip (UTR5/CDS/UTR3 as filled rects via graphxy.pos()
        + canvas.fill(path.rect(...)), "5'"/"3'" text at the gene's own
        start/end -- not a single arrow glyph, which would sit on top of
        the "- strand"/"+ strand" title text and contradict it visually
        for minus-strand genes)
      - the first num_reads individual reads from gene_df, right above
        the gene model -- the RAW per-site confidence_score trace across
        genomic position (real numeric axis), colored per-position with
        no run-averaging (read_site_scores), matching how
        runHMMPerGene.py's plot_signed_log_pyx plots this same transform
      - shadow-call depth, ribo-seq depth (plain line traces, not filled
        -- matching this codebase's existing "coverage as a line"
        convention, e.g. plot_pb_by_tx_pyx's col_cov), with the title/
        confidence legend above them at the very top. Ribo-seq depth is
        split into its two length classes rather than pooled: light
        grey for RIBO_SHORT_LENGTHS (21/22nt), dark grey for
        RIBO_LONG_LENGTHS (28/29nt) -- these correspond to different
        ribosome conformational states, so pooling them into one trace
        would wash out a real difference between the two.

    his_gpos: optional set of genomic positions of His codons in this
    gene (from findHisCodonPositions.py's pickle, via
    polysomeShadowHMMQC.load_his_codon_gpos) -- drawn as thin vertical
    lines through the depth panels and the gene model, same color as
    runHMMPerGene.py's own His-codon marker (col_his = cmyk(0,1,1,0)).

    flank_5p/flank_3p (typically a gene's own compute_flank_caps entry,
    same convention as runHMMPerGene.py) widen the panel's own axis range
    past gene["gene_start"]/["gene_end"] -- see plot_gene_shadow_pileup's
    docstring for why (shadow_calls.parquet can now have real positions
    out there, and without this they'd fall outside the plotted range).
    """
    from pyx import canvas, graph, color, style, text as pyx_text, path

    col_ribo_short = color.grey(0.75)
    col_ribo_long  = color.grey(0.3)
    col_shadow = color.cmyk(0, 0.5, 1, 0)

    if gene["strand"] == "+":
        lo, hi = gene["gene_start"] - flank_5p, gene["gene_end"] + flank_3p
    else:
        lo, hi = gene["gene_start"] - flank_3p, gene["gene_end"] + flank_5p
    xs = list(range(lo, hi))
    ribo_short_ys = [ribo_depth_short.get(x, 0) for x in xs]
    ribo_long_ys  = [ribo_depth_long.get(x, 0) for x in xs]
    shadow_ys = [shadow_depth.get(x, 0) for x in xs]
    his_in_range = sorted(g for g in (his_gpos or ()) if lo <= g < hi)

    panel_w, ribo_h, shadow_h, model_h, gap = 14, 3.5, 3.5, 0.6, 1.1
    read_h, read_gap = 1.1, 0.35
    ribo_max = max(ribo_short_ys + ribo_long_ys) * 1.1 if any(ribo_short_ys) or any(ribo_long_ys) else 1.0
    shadow_max = max(shadow_ys) * 1.1 if any(shadow_ys) else 1.0

    c = canvas.canvas()

    # gene-model strip built first -- its x-axis is what every panel
    # above links to (painter.linked() suppresses their duplicate ticks/
    # title but keeps the panel's own border, unlike a bare painter=None
    # which drops the border line too -- see graph.axis.painter.linked)
    g_model, col_his = draw_gene_model_strip(c, gene, lo, hi, his_in_range, panel_w, model_h)

    # individual reads, stacked directly above the gene model -- first
    # read in gene_df at the TOP of this block (closest to the depth
    # panels above), matching plot_pb_by_tx_pyx's own read ordering
    reads_base_ypos = model_h + gap
    rows = list(gene_df.itertuples(index=False))[:num_reads]
    n_reads = len(rows)
    read_data = [read_site_scores(row) for row in rows]

    # score_hi is the FIXED color scale (CONF_RANGE) -- comparing "library
    # X has more confident calls than library Y" only means anything if
    # the same score always maps to the same COLOR everywhere, not one
    # rescaled per gene. The axis itself, though, is dynamic: it stays at
    # score_hi for the common case (no call below the P_A=0.1 anchor),
    # but stretches to fit any site that actually dips lower, so those
    # outlier calls are still visible instead of getting flattened at
    # the top of the panel. A dashed line at score_hi marks where that
    # P_A=0.1 "very confident" anchor sits once the axis stretches past it.
    score_hi = CONF_RANGE
    all_scores = [s for sites in read_data for _, s in sites]
    axis_hi = max(score_hi, max(all_scores) * 1.05) if all_scores else score_hi

    for jj, row in enumerate(rows):
        ypos = reads_base_ypos + (n_reads - 1 - jj) * (read_h + read_gap)
        sites = read_data[jj]
        g_read = graph.graphxy(
            width=panel_w, height=read_h, xpos=0, ypos=ypos,
            x=graph.axis.linkedaxis(g_model.axes["x"], painter=graph.axis.painter.linked()),
            y=graph.axis.linear(min=0, max=axis_hi, title="score"))
        c.insert(g_read)
        if axis_hi > score_hi:
            g_read.plot(graph.data.function(f"y(x)={score_hi}", min=lo, max=hi),
                       [graph.style.line([color.grey(0.6), style.linewidth.thin, style.linestyle.dotted])])
        # per-position trace: each consecutive site pair is its own line
        # segment, colored by the LEFT site's own raw score (no run-
        # averaging -- see read_site_scores), matching the raw per-site
        # transform runHMMPerGene.py's plot_signed_log_pyx plots
        for k in range(len(sites) - 1):
            gp1, s1 = sites[k]
            gp2, s2 = sites[k + 1]
            g_read.plot(graph.data.points([(gp1, s1), (gp2, s2)], x=1, y=2),
                       [graph.style.line([confidence_color(s1, score_hi), style.linewidth.Thick])])
        for gp, s in sites:
            g_read.plot(graph.data.points([(gp, s)], x=1, y=2),
                       [graph.style.symbol(graph.style.symbol.circle, size=0.06,
                                           symbolattrs=[confidence_color(s, score_hi)])])
        for hg in his_in_range:
            g_read.plot(graph.data.function(f"x(y)={hg}", min=0, max=axis_hi),
                       [graph.style.line([col_his, style.linewidth.thin, style.linestyle.dashed])])
        label = str(row.read_id)[:16]
        c.text(panel_w + 0.15, ypos + read_h / 2., tex_escape(label),
              [pyx_text.valign.middle, pyx_text.size.tiny])
    if n_reads == 0:
        print(f"plot_gene_track: no reads for {gene_name}, skipping read panels", file=sys.stderr)
    reads_top_ypos = reads_base_ypos + n_reads * (read_h + read_gap)

    shadow_ypos = reads_top_ypos + gap
    g_shadow = graph.graphxy(
        width=panel_w, height=shadow_h, xpos=0, ypos=shadow_ypos,
        x=graph.axis.linkedaxis(g_model.axes["x"], painter=graph.axis.painter.linked()),
        y=graph.axis.linear(min=0, max=shadow_max, title="shadow-call depth"))
    c.insert(g_shadow)
    g_shadow.plot(graph.data.points(list(zip(xs, shadow_ys)), x=1, y=2),
                 [graph.style.line([col_shadow, style.linewidth.Thick])])
    for hg in his_in_range:
        g_shadow.plot(graph.data.function(f"x(y)={hg}", min=0, max=shadow_max),
                     [graph.style.line([col_his, style.linewidth.thin, style.linestyle.dashed])])

    ribo_ypos = shadow_ypos + shadow_h + gap
    g_ribo = graph.graphxy(
        width=panel_w, height=ribo_h, xpos=0, ypos=ribo_ypos,
        x=graph.axis.linkedaxis(g_model.axes["x"], painter=graph.axis.painter.linked()),
        y=graph.axis.linear(min=0, max=ribo_max, title="ribo-seq depth"))
    c.insert(g_ribo)
    g_ribo.plot(graph.data.points(list(zip(xs, ribo_long_ys)), x=1, y=2),
               [graph.style.line([col_ribo_long, style.linewidth.Thick])])
    g_ribo.plot(graph.data.points(list(zip(xs, ribo_short_ys)), x=1, y=2),
               [graph.style.line([col_ribo_short, style.linewidth.Thick])])
    for hg in his_in_range:
        g_ribo.plot(graph.data.function(f"x(y)={hg}", min=0, max=ribo_max),
                   [graph.style.line([col_his, style.linewidth.thin, style.linestyle.dashed])])

    top_ypos = ribo_ypos + ribo_h + gap

    c.text(g_ribo.xpos + g_ribo.width / 2., top_ypos + 0.75,
          tex_escape(gene_name),
          [pyx_text.halign.center, pyx_text.size.large])
    # ">" has no glyph in plain (OT1) text-mode fonts and silently renders as
    # an inverted "?" (¿) -- wrapped in $...$ (math mode) here to avoid that,
    # same reason polysomeShadowHMMQC.py's "P$_B>$%s" titles do it. Only the
    # data-derived chrom name goes through tex_escape; the rest of this
    # template is hand-written TeX we control, so escaping it would just
    # mangle the intentional $ signs.
    c.text(g_ribo.xpos + g_ribo.width / 2., top_ypos + 0.35,
          f"{tex_escape(gene['chrom'])}:{lo:,}-{hi:,}, {gene['strand']} strand -- "
          f"{len(his_in_range)} His codon(s) -- shadow-call P$_B>${shadow_cutoff}, "
          f"len$>${min_run_nt}nt, {n_reads} reads shown",
          [pyx_text.halign.center, pyx_text.size.scriptsize])

    # confidence-score legend for the footprint-call run blocks -- a
    # vertical color bar to the side (see draw_confidence_colorbar),
    # spanning from the reads section up to the title, rather than a
    # horizontal swatch row competing for space above the plot
    if n_reads:
        draw_confidence_colorbar(c, panel_w + 2.5, reads_base_ypos, top_ypos, score_hi)
        # the grey dotted reference line drawn in the read panels (only
        # when the axis actually stretches past score_hi)
        thresh_y = top_ypos - 0.05
        c.stroke(path.line(0.2, thresh_y, 0.85, thresh_y),
                [color.grey(0.6), style.linewidth.thin, style.linestyle.dotted])
        c.text(1.05, thresh_y, tex_escape("P_A=0.1 threshold (shown when the axis stretches past it)"),
              [pyx_text.valign.middle, pyx_text.size.tiny])

    ribo_leg_y = top_ypos - 0.35
    c.stroke(path.line(0.2, ribo_leg_y, 0.85, ribo_leg_y), [col_ribo_short, style.linewidth.Thick])
    c.text(1.05, ribo_leg_y, f"ribo-seq {RIBO_SHORT_LENGTHS[0]}/{RIBO_SHORT_LENGTHS[1]}nt",
          [pyx_text.valign.middle, pyx_text.size.tiny])
    c.stroke(path.line(3.5, ribo_leg_y, 4.15, ribo_leg_y), [col_ribo_long, style.linewidth.Thick])
    c.text(4.35, ribo_leg_y, f"ribo-seq {RIBO_LONG_LENGTHS[0]}/{RIBO_LONG_LENGTHS[1]}nt",
          [pyx_text.valign.middle, pyx_text.size.tiny])

    c.writePDFfile(str(pdf_path))
    print(f"Wrote {pdf_path}", file=sys.stderr)


def plot_gene_coverage(ribo_counts, shadow_rates, pdf_path, title,
                       shadow_cutoff=SHADOW_CUTOFF, min_run_nt=MIN_RUN_NT):
    """
    Two stacked panels, PyX/PDF, same gene order (sorted by ribo-seq read
    count, descending) on both -- top: ribo-seq reads/gene (log scale,
    spans orders of magnitude); bottom: shadow-call runs PER NANOPORE
    READ (linear) -- a RATE (see shadow_run_rate_per_gene), not the raw
    run count, since a gene sampled by more Nanopore reads racks up more
    raw qualifying runs purely from having more chances to see one, not
    necessarily from being more protected per read.

    Each gene is a single vertical line segment from the axis floor to
    its value ("lollipop"/stem, via a 2-point graph.data.points+
    style.line call per gene) rather than a connected profile line --
    genes are discrete, unordered categories (sorted for readability, but
    not on a continuous axis), so a connecting line between adjacent
    unrelated genes would visually imply a trend that isn't there.
    Gene-name labels are placed via graphxy.pos() + rotated canvas.text,
    since PyX's built-in categorical axis (graph.axis.bar) needs a nested
    (name, value) tuple data format not worth adopting for a plot this
    simple.
    """
    from pyx import canvas, graph, color, style, text as pyx_text, trafo

    col_ribo   = color.cmyk(1, 0.5, 0, 0)
    col_shadow = color.cmyk(0, 0.5, 1, 0)

    genes = sorted(ribo_counts, key=lambda g: ribo_counts[g], reverse=True)
    n = len(genes)
    ribo_vals   = [ribo_counts[g] for g in genes]
    shadow_vals = [shadow_rates.get(g, 0.0) for g in genes]

    panel_w = max(10, 0.16 * n)
    ribo_h, shadow_h, gap = 4, 4, 1.2
    ribo_floor = 1
    ribo_max   = max(ribo_vals) * 2 if ribo_vals else 10
    shadow_max = max(shadow_vals) * 1.15 if any(shadow_vals) else 1.0

    c = canvas.canvas()

    g_shadow = graph.graphxy(
        width=panel_w, height=shadow_h, xpos=0, ypos=0,
        x=graph.axis.linear(min=-0.5, max=n - 0.5, parter=None),
        y=graph.axis.linear(min=0, max=shadow_max, title="shadow-call runs / Nanopore read"))
    c.insert(g_shadow)
    for i, v in enumerate(shadow_vals):
        g_shadow.plot(graph.data.points([(i, 0), (i, v)], x=1, y=2),
                     [graph.style.line([col_shadow, style.linewidth.THick])])

    ribo_ypos = shadow_h + gap
    g_ribo = graph.graphxy(
        width=panel_w, height=ribo_h, xpos=0, ypos=ribo_ypos,
        x=graph.axis.linkedaxis(g_shadow.axes["x"], painter=graph.axis.painter.linked()),
        y=graph.axis.log(min=ribo_floor, max=ribo_max, title="ribo-seq reads / gene (log scale)"))
    c.insert(g_ribo)
    for i, v in enumerate(ribo_vals):
        g_ribo.plot(graph.data.points([(i, ribo_floor), (i, max(v, ribo_floor))], x=1, y=2),
                   [graph.style.line([col_ribo, style.linewidth.THick])])

    for i, gname in enumerate(genes):
        x_pt, y_pt = g_shadow.pos(i, 0)
        c.text(x_pt, y_pt - 0.15, tex_escape(gname),
              [pyx_text.halign.right, pyx_text.valign.middle, pyx_text.size.tiny, trafo.rotate(45)])

    # ">" has no glyph in plain (OT1) text-mode fonts and silently renders as
    # an inverted "?" (¿) -- wrapped in $...$ (math mode) below to avoid that.
    # Only `title` is data/caller-supplied and goes through tex_escape; the
    # rest is hand-written TeX we control, so escaping it would just mangle
    # the intentional $ signs.
    c.text(g_ribo.xpos + g_ribo.width / 2., ribo_ypos + ribo_h + 0.4,
          f"{tex_escape(title)} ({n} genes from shadow\\_calls.parquet) -- "
          f"ribo-seq {'/'.join(map(str, TARGET_LENGTHS))}nt, "
          f"shadow-call P$_B>${shadow_cutoff}, len$>${min_run_nt}nt",
          [pyx_text.halign.center, pyx_text.size.normalsize])

    c.writePDFfile(str(pdf_path))
    print(f"Wrote {pdf_path}", file=sys.stderr)


def main(args):
    shadowPath, riboBamListPath, condition, gtfPath, hisPicklePath, outPrefix = (
        args[0], args[1], args[2], args[3], args[4], args[5])
    gene_names_arg = args[6] if len(args) > 6 and args[6] else None
    shadow_cutoff = float(args[7]) if len(args) > 7 else SHADOW_CUTOFF
    min_run_nt    = int(args[8])   if len(args) > 8 else MIN_RUN_NT
    flank_nt      = int(args[9])   if len(args) > 9 else FLANK_NT

    out_dir = os.path.dirname(outPrefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    his_gpos_by_gene = load_his_codon_gpos(hisPicklePath)
    print(f"Loaded His codon positions for {len(his_gpos_by_gene)} genes.", file=sys.stderr)

    gene_list = genes_in_shadow_calls(shadowPath)
    print(f"{len(gene_list)} genes in {shadowPath}.", file=sys.stderr)

    genes = parse_gtf(gtfPath)
    flank_caps = compute_flank_caps(genes, flank_nt)
    missing = [g for g in gene_list if g not in genes]
    if missing:
        print(f"  {len(missing)} genes not found in GTF, skipped: "
              f"{missing[:10]}{'...' if len(missing) > 10 else ''}", file=sys.stderr)
    gene_list = [g for g in gene_list if g in genes]

    bam_paths = load_ribo_bam_list(riboBamListPath, condition)
    print(f"Counting ribo-seq reads across {len(bam_paths)} BAM(s) "
          f"(condition={condition}) for {len(gene_list)} genes...", file=sys.stderr)
    counts = count_ribo_reads_per_gene(bam_paths, genes, gene_list)

    print("\nTop 10 by read count:", file=sys.stderr)
    for g in sorted(counts, key=lambda g: -counts[g])[:10]:
        print(f"  {g}: {counts[g]:,} reads", file=sys.stderr)
    n_zero = sum(1 for g in gene_list if counts.get(g, 0) == 0)
    print(f"{n_zero}/{len(gene_list)} genes have zero overlapping ribo-seq reads.",
          file=sys.stderr)

    print(f"\nExtracting shadow calls (P_B>={shadow_cutoff}, >={min_run_nt}nt)...",
          file=sys.stderr)
    shadow_df = load_shadow_calls_df(shadowPath)
    shadow_counts = count_shadow_runs_per_gene(shadow_df, shadow_cutoff, min_run_nt)
    read_counts   = count_reads_per_gene(shadow_df)
    shadow_rates  = shadow_run_rate_per_gene(shadow_counts, read_counts)
    print("Top 10 by shadow-call count (raw):", file=sys.stderr)
    for g in sorted(shadow_counts, key=lambda g: -shadow_counts[g])[:10]:
        print(f"  {g}: {shadow_counts[g]:,} runs / {read_counts.get(g, 0):,} reads", file=sys.stderr)
    print("Top 10 by shadow-call RATE (runs / Nanopore read):", file=sys.stderr)
    for g in sorted(shadow_rates, key=lambda g: -shadow_rates[g])[:10]:
        print(f"  {g}: {shadow_rates[g]:.3f} runs/read ({shadow_counts.get(g, 0):,} runs / "
              f"{read_counts[g]:,} reads)", file=sys.stderr)

    # plot_gene_coverage(counts, shadow_rates, f"{outPrefix}.ribo_and_shadow_per_gene.pdf",
    #                    title="Ribo-seq reads vs. shadow-call rate per gene",
    #                    shadow_cutoff=shadow_cutoff, min_run_nt=min_run_nt)

    if gene_names_arg:
        track_genes = [g.strip() for g in gene_names_arg.split(",") if g.strip()]
        missing_track = [g for g in track_genes if g not in genes]
        if missing_track:
            print(f"  requested track gene(s) not in GTF, skipped: {missing_track}",
                  file=sys.stderr)
        track_genes = [g for g in track_genes if g in genes]
    else:
        track_genes = list(gene_list)
        print(f"\nNo geneNames given -- plotting all {len(track_genes)} genes.", file=sys.stderr)

    parquet_stem = os.path.splitext(os.path.basename(shadowPath))[0]
    track_dir = os.path.join(out_dir, parquet_stem) if out_dir else parquet_stem
    os.makedirs(track_dir, exist_ok=True)

    print(f"\nBuilding per-position coverage track(s) for {len(track_genes)} gene(s) "
          f"into {track_dir}/ ...", file=sys.stderr)
    for gname in track_genes:
        gene = genes[gname]
        flank_5p, flank_3p = flank_caps[gname]
        # ribo_coverage_track fetches reads within gene["gene_start"]/["gene_end"]
        # only -- widen that fetch window the same way shadowMetagene.py does,
        # else it'd never even look at the flank region the plots below now show.
        padded_gene = dict(gene)
        if gene["strand"] == "+":
            padded_gene["gene_start"] = max(0, gene["gene_start"] - flank_5p)
            padded_gene["gene_end"]   = gene["gene_end"] + flank_3p
        else:
            padded_gene["gene_start"] = max(0, gene["gene_start"] - flank_3p)
            padded_gene["gene_end"]   = gene["gene_end"] + flank_5p
        ribo_depth_short = ribo_coverage_track(bam_paths, padded_gene, target_lengths=RIBO_SHORT_LENGTHS)
        ribo_depth_long  = ribo_coverage_track(bam_paths, padded_gene, target_lengths=RIBO_LONG_LENGTHS)
        gene_df      = shadow_df[shadow_df["shadow_gene"] == gname]
        shadow_depth = shadow_coverage_track(gene_df, gname, gene, shadow_cutoff, min_run_nt,
                                             flank_5p=flank_5p, flank_3p=flank_3p)
        plot_gene_track(gname, gene, ribo_depth_short, ribo_depth_long, shadow_depth, gene_df,
                        os.path.join(track_dir, f"{gname}.pdf"),
                        his_gpos=his_gpos_by_gene.get(gname),
                        shadow_cutoff=shadow_cutoff, min_run_nt=min_run_nt,
                        flank_5p=flank_5p, flank_3p=flank_3p)
        plot_gene_shadow_pileup(gname, gene, ribo_depth_short, ribo_depth_long, shadow_depth, gene_df,
                                os.path.join(track_dir, f"{gname}.pileup.pdf"),
                                his_gpos=his_gpos_by_gene.get(gname),
                                shadow_cutoff=shadow_cutoff, min_run_nt=min_run_nt,
                                flank_5p=flank_5p, flank_3p=flank_3p)


if __name__ == "__main__":
    main(sys.argv[1:])
