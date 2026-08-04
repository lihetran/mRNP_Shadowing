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

Also produces a per-gene INDIVIDUAL-READS view (plot_gene_reads), styled
after runHMMPerGene.py's plot_pb_by_tx_pyx: the first NUM_READS=10 reads
in shadowCallsParquet for that gene, one stacked panel per read, P_B
across transcript position -- but colored by confidence_score (the same
signed log-transform runHMMPerGene.py's plot_signed_log_pyx uses for its
yA/yB traces, here collapsed into one signed scalar and mapped to a
blue(unprotected)-grey(uncertain)-orange(protected) gradient) rather than
one flat color per read. Sites inside a qualifying footprint call share
that call's MEAN confidence score rather than each fluctuating on its own
noisy per-site value -- see read_site_colors.

The shadow-call bar chart's second panel is a RATE (qualifying runs per
Nanopore read), not the raw run count -- a gene sampled by more Nanopore
reads racks up more raw runs purely from having more chances to see one,
independent of whether it's genuinely more protected per read. See
shadow_run_rate_per_gene.

Run:
  python3 riboseqGeneCoverage.py shadowCallsParquet riboBamList.txt condition gtfFile hisPicklePath outPrefix [geneNames] [shadow_cutoff] [min_run_nt]
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
track for (default: every gene), and shadow_cutoff/min_run_nt (optional)
override the P_B cutoff (default 0.5) and run-length floor (default
25nt) for what counts as a shadow call.
"""
import sys, os, math, collections
import pandas as pd
import pysam

from runHMMPerGene import parse_gtf, tex_escape
from polysomeShadowHMMQC import extract_shadow_runs, load_his_codon_gpos

SHADOW_CUTOFF  = 0.5
MIN_RUN_NT     = 30
TARGET_LENGTHS = (21, 22, 28, 29)   # same footprint-length convention as riboseqShadowCorrelation.py
NUM_READS      = 10                 # plot_gene_reads: how many individual reads to show per gene
CONF_EPS       = 1e-6               # confidence-score floor, same role as plot_signed_log_pyx's eps
CONF_RANGE     = 4.0                # |score| past this is already fully-saturated color


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
    shadow_tx_pos/shadow_P_A for the per-read confidence-colored plots
    (plot_gene_reads) -- extract_shadow_runs itself ignores the extra
    columns, so one shared loader still works for both callers."""
    return pd.read_parquet(shadow_parquet_path,
                           columns=["read_id", "shadow_gene", "shadow_gpos",
                                   "shadow_P_B", "shadow_P_A", "shadow_tx_pos",
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


def shadow_coverage_track(shadow_df, gene_name, gene, cutoff=SHADOW_CUTOFF, min_run_nt=MIN_RUN_NT):
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
    the gene's own annotated exon intervals (gene["exons"]) so only
    genuine (spliced-in) sequence gets a depth value -- the intron shows
    as a true gap, same as ribo_coverage_track's CIGAR-aware fix.

    shadow_df should already be restricted to this gene (a caller with a
    multi-gene df pays for run-extraction on every OTHER gene too
    otherwise) -- count_shadow_runs_per_gene doesn't need this since it
    only cares about run-level counts, not per-position depth, but this
    does.
    """
    exons = gene.get("exons", [])
    runs = extract_shadow_runs(shadow_df, cutoff)
    depth = collections.Counter()
    for r in runs:
        if r["gene"] != gene_name or r["genomic_nt"] < min_run_nt:
            continue
        for gpos in range(r["gpos_lo"], r["gpos_hi"] + 1):
            if any(s <= gpos < e for s, e in exons):
                depth[gpos] += 1
    return depth


def confidence_score(p_a, eps=CONF_EPS):
    """
    log10(1 - P_A): 0 at P_A=0 (fully confident the site is PROTECTED --
    1-P_A tops out at 1 there), increasingly negative as that confidence
    drops -- through a mild negative value at genuine 50/50 uncertainty
    (P_A=0.5 -> log10(0.5)=-0.301) down to strongly negative for a site
    confidently UNPROTECTED (P_A near 1, so 1-P_A near 0). Log-scaled
    rather than linear so P_A=0.999 reads as clearly more confidently
    unprotected than P_A=0.9, not indistinguishable the way a linear 0-1
    map would leave them. Uses only the P_A column (not P_B), per what
    was asked for here specifically.
    """
    return math.log10(max(1.0 - p_a, eps))


def confidence_color(score, conf_range=CONF_RANGE):
    """
    Purple-to-yellow gradient for a confidence_score (always <= 0):
    purple at score=0 (confidently protected -- a strong footprint call),
    shifting to yellow as score drops toward -conf_range (confidently
    unprotected -- background), so a real footprint call visually pops
    out against the yellow background instead of blending in. Anything
    at or past -conf_range is fully saturated yellow and stays that
    color -- score doesn't keep changing appearance past that point, so
    a few extreme-confidence outliers don't wash out the scale for
    everything else.
    """
    from pyx import color
    frac = min(1.0, max(0.0, -score / conf_range))
    purple, yellow = (0.35, 0.10, 0.45), (0.95, 0.90, 0.15)
    return color.rgb(*(p + (y - p) * frac for p, y in zip(purple, yellow)))


def read_pb_trace_and_runs(row, cutoff=SHADOW_CUTOFF, min_run_nt=MIN_RUN_NT):
    """
    For one read (a row from shadow_calls.parquet): its plain (gpos, P_B)
    trace sorted by GENOMIC position (genomic rather than transcript
    position so this can share an x-axis with plot_gene_track's coverage
    panels, which are already genomic-position, IGV-style), plus the
    list of qualifying footprint-CALL runs within it.

    A footprint call is shown as a RUN (a solid block spanning
    [gpos_lo, gpos_hi]), not per-position coloring -- using the exact
    same run definition as shadow_coverage_track/count_shadow_runs_per_gene
    (extract_shadow_runs, P_B>=cutoff, genomic_nt>=min_run_nt), so a call
    drawn here is identical to what the coverage plot's shadow-call depth
    panel already counts. Each run's color comes from confidence_score
    averaged across its own member sites (not any single site's noisier
    value).

    Returns (trace, runs): trace = [(gpos, P_B), ...] for the whole read;
    runs = [(gpos_lo, gpos_hi, mean_score), ...] for qualifying runs only.
    """
    gp = [int(g) for g in row.shadow_gpos]
    pa = [float(p) for p in row.shadow_P_A]
    pb = [float(p) for p in row.shadow_P_B]
    order = sorted(range(len(gp)), key=lambda k: gp[k])
    gp_s = [gp[k] for k in order]
    pa_s = [pa[k] for k in order]; pb_s = [pb[k] for k in order]
    own_score = [confidence_score(a) for a in pa_s]

    single_read_df = pd.DataFrame({
        "read_id": [row.read_id], "shadow_gene": [row.shadow_gene],
        "shadow_gpos": [gp_s], "shadow_P_B": [pb_s],
        "shadow_region": [list(row.shadow_region)],
        "absolute_indices": [row.absolute_indices],
    })
    runs = []
    for r in extract_shadow_runs(single_read_df, cutoff):
        if r["genomic_nt"] < min_run_nt:
            continue
        lo, hi = min(r["gpos_lo"], r["gpos_hi"]), max(r["gpos_lo"], r["gpos_hi"])
        member_idx = [i for i, g in enumerate(gp_s) if lo <= g <= hi]
        if not member_idx:
            continue
        mean_score = sum(own_score[i] for i in member_idx) / len(member_idx)
        runs.append((lo, hi, mean_score))

    return list(zip(gp_s, pb_s)), runs


def plot_gene_track(gene_name, gene, ribo_depth, shadow_depth, gene_df, pdf_path,
                    his_gpos=None, shadow_cutoff=SHADOW_CUTOFF, min_run_nt=MIN_RUN_NT,
                    num_reads=NUM_READS):
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
        the gene model -- P_B across genomic position (plain grey line,
        real numeric axis) with qualifying footprint CALLS drawn as solid
        confidence-colored RUN blocks (read_pb_trace_and_runs), not
        per-position coloring, so a call reads as one clear block the
        same way shadow-call depth below shows runs, not noisy points
      - shadow-call depth, ribo-seq depth (plain line traces, not filled
        -- matching this codebase's existing "coverage as a line"
        convention, e.g. plot_pb_by_tx_pyx's col_cov), with the title/
        confidence legend above them at the very top

    his_gpos: optional set of genomic positions of His codons in this
    gene (from findHisCodonPositions.py's pickle, via
    polysomeShadowHMMQC.load_his_codon_gpos) -- drawn as thin vertical
    lines through the depth panels and the gene model, same color as
    runHMMPerGene.py's own His-codon marker (col_his = cmyk(0,1,1,0)).
    """
    from pyx import canvas, graph, color, style, text as pyx_text, path

    col_ribo   = color.cmyk(1, 0.5, 0, 0)
    col_shadow = color.cmyk(0, 0.5, 1, 0)
    col_his    = color.cmyk(0, 1, 1, 0)
    col_cds    = color.cmyk(0, 0, 0, 0.75)
    col_utr    = color.cmyk(0, 0, 0, 0.25)
    col_pb     = color.grey(0.25)

    lo, hi = gene["gene_start"], gene["gene_end"]
    xs = list(range(lo, hi))
    ribo_ys   = [ribo_depth.get(x, 0) for x in xs]
    shadow_ys = [shadow_depth.get(x, 0) for x in xs]
    his_in_range = sorted(g for g in (his_gpos or ()) if lo <= g < hi)

    panel_w, ribo_h, shadow_h, model_h, gap = 14, 3.5, 3.5, 0.6, 1.1
    read_h, read_gap = 1.1, 0.35
    ribo_max   = max(ribo_ys) * 1.1 if any(ribo_ys) else 1.0
    shadow_max = max(shadow_ys) * 1.1 if any(shadow_ys) else 1.0

    c = canvas.canvas()

    # gene-model strip built first -- its x-axis is what every panel
    # above links to (painter.linked() suppresses their duplicate ticks/
    # title but keeps the panel's own border, unlike a bare painter=None
    # which drops the border line too -- see graph.axis.painter.linked)
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
    if gene["strand"] == "+":
        five_x, five_ha, three_x, three_ha = lo, pyx_text.halign.left, hi, pyx_text.halign.right
    else:
        five_x, five_ha, three_x, three_ha = hi, pyx_text.halign.right, lo, pyx_text.halign.left
    fx, fy = g_model.pos(five_x, 0.5)
    tx, ty = g_model.pos(three_x, 0.5)
    c.text(fx, fy, "5'", [five_ha, pyx_text.valign.middle, pyx_text.size.scriptsize])
    c.text(tx, ty, "3'", [three_ha, pyx_text.valign.middle, pyx_text.size.scriptsize])

    # individual reads, stacked directly above the gene model -- first
    # read in gene_df at the TOP of this block (closest to the depth
    # panels above), matching plot_pb_by_tx_pyx's own read ordering
    reads_base_ypos = model_h + gap
    rows = list(gene_df.itertuples(index=False))[:num_reads]
    n_reads = len(rows)
    for jj, row in enumerate(rows):
        ypos = reads_base_ypos + (n_reads - 1 - jj) * (read_h + read_gap)
        trace, runs = read_pb_trace_and_runs(row, shadow_cutoff, min_run_nt)
        g_read = graph.graphxy(
            width=panel_w, height=read_h, xpos=0, ypos=ypos,
            x=graph.axis.linkedaxis(g_model.axes["x"], painter=graph.axis.painter.linked()),
            y=graph.axis.linear(min=0, max=1, title="P$_B$"))
        c.insert(g_read)
        # footprint-call runs drawn first, as solid blocks, so the plain
        # P_B trace renders on top of them, still visible for shape
        for run_lo, run_hi, mean_score in runs:
            x0, y0 = g_read.pos(max(run_lo, lo), 0.0)
            x1, y1 = g_read.pos(min(run_hi, hi), 1.0)
            c.fill(path.rect(x0, y0, x1 - x0, y1 - y0), [confidence_color(mean_score)])
        g_read.plot(graph.data.points(trace, x=1, y=2),
                   [graph.style.line([col_pb, style.linewidth.Thick])])
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
    g_ribo.plot(graph.data.points(list(zip(xs, ribo_ys)), x=1, y=2),
               [graph.style.line([col_ribo, style.linewidth.Thick])])
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
          f"{len(his_in_range)} His codon(s) -- ribo-seq {'/'.join(map(str, TARGET_LENGTHS))}nt, "
          f"shadow-call P$_B>${shadow_cutoff}, len$>${min_run_nt}nt, "
          f"{n_reads} reads shown",
          [pyx_text.halign.center, pyx_text.size.scriptsize])

    # confidence-score legend for the footprint-call run blocks
    if n_reads:
        leg_y = top_ypos - 0.05
        for i, (lab, sc) in enumerate([("high-confidence call", 0.0),
                                       ("moderate", -CONF_RANGE / 2),
                                       ("low-confidence call (near cutoff)", -CONF_RANGE)]):
            x0 = 0.3 + i * 4.6
            c.fill(path.rect(x0, leg_y - 0.08, 0.16, 0.16), [confidence_color(sc)])
            c.text(x0 + 0.25, leg_y, tex_escape(lab), [pyx_text.valign.middle, pyx_text.size.tiny])

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

    out_dir = os.path.dirname(outPrefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    his_gpos_by_gene = load_his_codon_gpos(hisPicklePath)
    print(f"Loaded His codon positions for {len(his_gpos_by_gene)} genes.", file=sys.stderr)

    gene_list = genes_in_shadow_calls(shadowPath)
    print(f"{len(gene_list)} genes in {shadowPath}.", file=sys.stderr)

    genes = parse_gtf(gtfPath)
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

    plot_gene_coverage(counts, shadow_rates, f"{outPrefix}.ribo_and_shadow_per_gene.pdf",
                       title="Ribo-seq reads vs. shadow-call rate per gene",
                       shadow_cutoff=shadow_cutoff, min_run_nt=min_run_nt)

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
        ribo_depth   = ribo_coverage_track(bam_paths, gene)
        gene_df      = shadow_df[shadow_df["shadow_gene"] == gname]
        shadow_depth = shadow_coverage_track(gene_df, gname, gene, shadow_cutoff, min_run_nt)
        plot_gene_track(gname, gene, ribo_depth, shadow_depth, gene_df,
                        os.path.join(track_dir, f"{gname}.pdf"),
                        his_gpos=his_gpos_by_gene.get(gname),
                        shadow_cutoff=shadow_cutoff, min_run_nt=min_run_nt)


if __name__ == "__main__":
    main(sys.argv[1:])
