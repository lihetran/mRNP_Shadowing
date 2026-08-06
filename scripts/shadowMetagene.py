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

Run:
  python3 shadowMetagene.py shadowCallsParquet riboBamList.txt condition gtfFile hisPicklePath outPrefix [window_nt] [shadow_cutoff] [min_run_nt]
where shadowCallsParquet is ONE library's shadow_calls.parquet,
riboBamList.txt/condition select the ribo-seq BAMs (same convention as
riboseqGeneCoverage.py -- e.g.
/data16/liam/working/260804_riboSeq_vs_PS/riboSeqBam.txt and "-3AT"),
gtfFile is the yeast GTF (for cds_length per gene), hisPicklePath is
findHisCodonPositions.py's output pickle, outPrefix names the output
plots, window_nt (optional, default 100) is how many nt on either side of
each anchor to include, and shadow_cutoff/min_run_nt (optional) override
the P_B cutoff (default 0.5) and run-length floor (default 30nt) for what
counts as a shadow call -- same defaults/meaning as
riboseqGeneCoverage.py.
"""
import sys, os, bisect, pickle, collections
import pandas as pd

from runHMMPerGene import parse_gtf, cds_length
from riboseqGeneCoverage import (
    load_ribo_bam_list, ribo_coverage_track, shadow_coverage_track,
    load_shadow_calls_df, SHADOW_CUTOFF, MIN_RUN_NT, TARGET_LENGTHS,
)

WINDOW_NT = 100


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


def _plot_anchor_panels(c, graph, style, color, pyx_text, path, xpos, ypos,
                        panel_w, shadow_h, ribo_h, gap,
                        shadow_xs, shadow_means, ribo_xs, ribo_means,
                        shadow_col, ribo_col, x_title, anchor_label):
    """
    One anchor's worth of panels (shared by plot_his_metagene and
    plot_startstop_metagene), bottom to top: mean per-gene shadow-call
    depth fraction, mean per-gene ribo-seq depth fraction -- sharing one
    x-axis and a dashed vertical reference line at x=0 (the anchor
    itself) through both. Returns the top-of-stack y-position (for the
    anchor_label/figure-title text above it).
    """
    shadow_pts = [(x, m) for x, m in zip(shadow_xs, shadow_means) if m is not None]
    shadow_max = max((m for _x, m in shadow_pts), default=1e-9) * 1.1 if shadow_pts else 1e-9
    g_shadow = graph.graphxy(
        width=panel_w, height=shadow_h, xpos=xpos, ypos=ypos,
        x=graph.axis.linear(min=shadow_xs[0], max=shadow_xs[-1], title=x_title),
        y=graph.axis.linear(min=0, max=shadow_max, title="shadow-call frac."))
    c.insert(g_shadow)
    if shadow_pts:
        g_shadow.plot(graph.data.points(shadow_pts, x=1, y=2),
                      [graph.style.line([shadow_col, style.linewidth.Thick])])
    g_shadow.plot(graph.data.function("x(y)=0", min=0, max=shadow_max),
                  [graph.style.line([color.grey(0.6), style.linewidth.thin, style.linestyle.dashed])])

    ribo_ypos = ypos + shadow_h + gap
    ribo_pts = [(x, m) for x, m in zip(ribo_xs, ribo_means) if m is not None]
    ribo_max = max((m for _x, m in ribo_pts), default=1e-9) * 1.1 if ribo_pts else 1e-9
    g_ribo = graph.graphxy(
        width=panel_w, height=ribo_h, xpos=xpos, ypos=ribo_ypos,
        x=graph.axis.linkedaxis(g_shadow.axes["x"], painter=graph.axis.painter.linked()),
        y=graph.axis.linear(min=0, max=ribo_max, title="ribo-seq frac."))
    c.insert(g_ribo)
    if ribo_pts:
        g_ribo.plot(graph.data.points(ribo_pts, x=1, y=2),
                   [graph.style.line([ribo_col, style.linewidth.Thick])])
    g_ribo.plot(graph.data.function("x(y)=0", min=0, max=ribo_max),
               [graph.style.line([color.grey(0.6), style.linewidth.thin, style.linestyle.dashed])])

    top_ypos = ribo_ypos + ribo_h
    c.text(xpos + panel_w / 2., top_ypos + 0.3,
          anchor_label, [pyx_text.halign.center, pyx_text.size.normalsize])
    return top_ypos


def plot_his_metagene(shadow_acc, ribo_acc, pdf_path, window_nt=WINDOW_NT,
                      n_genes_shadow=0, n_genes_ribo=0):
    """
    One figure: mean per-gene shadow-call depth fraction (blue) and mean
    per-gene ribo-seq depth fraction (green) vs. signed transcript-nt
    distance to the nearest His codon. See build_his_density_metagene.
    """
    from pyx import canvas, graph, color, style, text as pyx_text, path

    shadow_xs, shadow_means = _means(shadow_acc, window_nt)
    ribo_xs, ribo_means = _means(ribo_acc, window_nt)
    col_shadow = color.cmyk(1, 0.3, 0, 0.1)
    col_ribo   = color.cmyk(0.7, 0, 1, 0.2)

    c = canvas.canvas()
    panel_w, shadow_h, ribo_h, gap = 12, 3.5, 3.5, 0.8
    top_ypos = _plot_anchor_panels(
        c, graph, style, color, pyx_text, path, 0, 0, panel_w, shadow_h, ribo_h, gap,
        shadow_xs, shadow_means, ribo_xs, ribo_means, col_shadow, col_ribo,
        "nt from nearest His codon (transcript-relative)",
        "Metagene: shadow-call + ribo-seq depth around His codons")

    c.text(panel_w / 2., top_ypos + 0.7,
          f"{n_genes_shadow} gene(s) with shadow-call depth, {n_genes_ribo} gene(s) with "
          f"ribo-seq depth, within $\\pm${window_nt}nt of a His codon",
          [pyx_text.halign.center, pyx_text.size.scriptsize])

    c.writePDFfile(str(pdf_path))
    print(f"Wrote {pdf_path}", file=sys.stderr)


def plot_startstop_metagene(start_shadow_acc, stop_shadow_acc, start_ribo_acc, stop_ribo_acc,
                            pdf_path, window_nt=WINDOW_NT,
                            n_genes_shadow=0, n_genes_ribo=0):
    """
    One figure, two side-by-side panel pairs: mean per-gene shadow-call
    and ribo-seq depth fractions vs. transcript-nt distance from the
    start codon (left) and from the stop codon (right). See
    build_startstop_density_metagene.
    """
    from pyx import canvas, graph, color, style, text as pyx_text, path

    start_shadow_xs, start_shadow_means = _means(start_shadow_acc, window_nt)
    stop_shadow_xs, stop_shadow_means   = _means(stop_shadow_acc, window_nt)
    start_ribo_xs, start_ribo_means     = _means(start_ribo_acc, window_nt)
    stop_ribo_xs, stop_ribo_means       = _means(stop_ribo_acc, window_nt)
    col_shadow = color.cmyk(1, 0.3, 0, 0.1)
    col_ribo   = color.cmyk(0.7, 0, 1, 0.2)

    c = canvas.canvas()
    panel_w, shadow_h, ribo_h, gap, panel_gap = 8, 3.5, 3.5, 0.8, 1.5

    top1 = _plot_anchor_panels(
        c, graph, style, color, pyx_text, path, 0, 0, panel_w, shadow_h, ribo_h, gap,
        start_shadow_xs, start_shadow_means, start_ribo_xs, start_ribo_means,
        col_shadow, col_ribo, "nt from start codon", "Start codon")
    _top2 = _plot_anchor_panels(
        c, graph, style, color, pyx_text, path, panel_w + panel_gap, 0, panel_w, shadow_h, ribo_h, gap,
        stop_shadow_xs, stop_shadow_means, stop_ribo_xs, stop_ribo_means,
        col_shadow, col_ribo, "nt from stop codon", "Stop codon")

    c.text((2 * panel_w + panel_gap) / 2., top1 + 0.9,
          "Metagene: shadow-call + ribo-seq depth around start/stop codons",
          [pyx_text.halign.center, pyx_text.size.normalsize])
    c.text((2 * panel_w + panel_gap) / 2., top1 + 0.5,
          f"{n_genes_shadow} gene(s) with shadow-call depth, {n_genes_ribo} gene(s) with "
          f"ribo-seq depth, within $\\pm${window_nt}nt of start/stop",
          [pyx_text.halign.center, pyx_text.size.scriptsize])

    c.writePDFfile(str(pdf_path))
    print(f"Wrote {pdf_path}", file=sys.stderr)


def _plot_bypos_anchor(c, graph, style, color, pyx_text, path, xpos, ypos,
                       panel_w, sig_h, dep_h, gap, xs, means, depths, depth_max,
                       x_title, anchor_label):
    """
    One anchor's raw by-position panels: n_sites depth (bottom), mean
    P_B (top) -- the "by position" counterpart to _plot_anchor_panels'
    per-gene-fraction run-depth signal. A dotted P_B=0.5 reference (this
    codebase's own shadow-call P_B cutoff convention) marks where the
    run-depth figures' threshold sits, so the two are visually
    comparable despite plotting different quantities.
    """
    col_sig = color.cmyk(1, 0.3, 0, 0.1)
    g_dep = graph.graphxy(
        width=panel_w, height=dep_h, xpos=xpos, ypos=ypos,
        x=graph.axis.linear(min=xs[0], max=xs[-1], title=x_title),
        y=graph.axis.linear(min=0, max=max(1, depth_max), title="n sites"))
    c.insert(g_dep)
    g_dep.plot(graph.data.points(list(zip(xs, depths)), x=1, y=2),
              [graph.style.line([color.grey(0.4), style.linewidth.Thick])])
    g_dep.plot(graph.data.function("x(y)=0", min=0, max=max(1, depth_max)),
              [graph.style.line([color.grey(0.6), style.linewidth.thin, style.linestyle.dashed])])

    sig_ypos = ypos + dep_h + gap
    pts = [(x, m) for x, m in zip(xs, means) if m is not None]
    sig_max = max((m for _x, m in pts), default=1.0) * 1.1 if pts else 1.0
    g_sig = graph.graphxy(
        width=panel_w, height=sig_h, xpos=xpos, ypos=sig_ypos,
        x=graph.axis.linkedaxis(g_dep.axes["x"], painter=graph.axis.painter.linked()),
        y=graph.axis.linear(min=0, max=max(sig_max, 0.55), title="mean P$_B$"))
    c.insert(g_sig)
    if pts:
        g_sig.plot(graph.data.points(pts, x=1, y=2),
                  [graph.style.line([col_sig, style.linewidth.Thick])])
    g_sig.plot(graph.data.function("x(y)=0", min=0, max=max(sig_max, 0.55)),
              [graph.style.line([color.grey(0.6), style.linewidth.thin, style.linestyle.dashed])])
    g_sig.plot(graph.data.function("y(x)=0.5", min=xs[0], max=xs[-1]),
              [graph.style.line([color.grey(0.6), style.linewidth.thin, style.linestyle.dotted])])

    top_ypos = sig_ypos + sig_h
    c.text(xpos + panel_w / 2., top_ypos + 0.3,
          anchor_label, [pyx_text.halign.center, pyx_text.size.normalsize])
    return top_ypos


def plot_his_bypos_metagene(acc, pdf_path, window_nt=WINDOW_NT, n_genes=0, n_sites=0):
    """
    One figure: mean raw per-site P_B (no run thresholding) vs. signed
    transcript-nt distance to the nearest His codon, plus its n_sites
    depth-of-support panel. The "by position" counterpart to
    plot_his_metagene -- see build_his_bypos_metagene.
    """
    from pyx import canvas, graph, color, style, text as pyx_text, path

    xs, means = _means(acc, window_nt)
    depths = [acc.get(x, (0.0, 0))[1] for x in xs]

    c = canvas.canvas()
    panel_w, sig_h, dep_h, gap = 12, 3.5, 2.0, 0.8
    top_ypos = _plot_bypos_anchor(
        c, graph, style, color, pyx_text, path, 0, 0, panel_w, sig_h, dep_h, gap,
        xs, means, depths, max(depths, default=1),
        "nt from nearest His codon (transcript-relative)",
        "By-position metagene: raw shadow-call P$_B$ around His codons")

    c.text(panel_w / 2., top_ypos + 0.7,
          f"{n_genes} gene(s), {n_sites} scored site(s) within $\\pm${window_nt}nt of a His "
          f"codon (every scored site, no P$_B$ cutoff or run-length filtering)",
          [pyx_text.halign.center, pyx_text.size.scriptsize])

    c.writePDFfile(str(pdf_path))
    print(f"Wrote {pdf_path}", file=sys.stderr)


def plot_startstop_bypos_metagene(start_acc, stop_acc, pdf_path, window_nt=WINDOW_NT,
                                  n_genes=0, n_sites_start=0, n_sites_stop=0):
    """
    One figure, two side-by-side panel pairs: mean raw per-site P_B (no
    run thresholding) vs. transcript-nt distance from the start codon
    (left) and from the stop codon (right). The "by position" counterpart
    to plot_startstop_metagene -- see build_startstop_bypos_metagene.
    """
    from pyx import canvas, graph, color, style, text as pyx_text, path

    start_xs, start_means = _means(start_acc, window_nt)
    stop_xs, stop_means   = _means(stop_acc, window_nt)
    start_depths = [start_acc.get(x, (0.0, 0))[1] for x in start_xs]
    stop_depths  = [stop_acc.get(x, (0.0, 0))[1] for x in stop_xs]

    c = canvas.canvas()
    panel_w, sig_h, dep_h, gap, panel_gap = 8, 3.5, 2.0, 0.8, 1.5

    top1 = _plot_bypos_anchor(
        c, graph, style, color, pyx_text, path, 0, 0, panel_w, sig_h, dep_h, gap,
        start_xs, start_means, start_depths, max(start_depths, default=1),
        "nt from start codon", "Start codon")
    _top2 = _plot_bypos_anchor(
        c, graph, style, color, pyx_text, path, panel_w + panel_gap, 0, panel_w, sig_h, dep_h, gap,
        stop_xs, stop_means, stop_depths, max(stop_depths, default=1),
        "nt from stop codon", "Stop codon")

    c.text((2 * panel_w + panel_gap) / 2., top1 + 0.9,
          "By-position metagene: raw shadow-call P$_B$ around start/stop codons",
          [pyx_text.halign.center, pyx_text.size.normalsize])
    c.text((2 * panel_w + panel_gap) / 2., top1 + 0.5,
          f"{n_genes} gene(s) -- {n_sites_start} site(s) near start, {n_sites_stop} near stop "
          f"(every scored site, no P$_B$ cutoff or run-length filtering)",
          [pyx_text.halign.center, pyx_text.size.scriptsize])

    c.writePDFfile(str(pdf_path))
    print(f"Wrote {pdf_path}", file=sys.stderr)


def main(args):
    shadowPath, riboBamListPath, condition, gtfPath, hisPicklePath, outPrefix = args[:6]
    window_nt     = int(args[6])   if len(args) > 6 else WINDOW_NT
    shadow_cutoff = float(args[7]) if len(args) > 7 else SHADOW_CUTOFF
    min_run_nt    = int(args[8])   if len(args) > 8 else MIN_RUN_NT

    out_dir = os.path.dirname(outPrefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {shadowPath} ...", file=sys.stderr)
    shadow_df = load_shadow_calls_df(shadowPath)
    print(f"  {len(shadow_df)} reads, {shadow_df['shadow_gene'].nunique()} genes.", file=sys.stderr)

    genes = parse_gtf(gtfPath)
    gene_names = sorted(shadow_df["shadow_gene"].unique())
    missing_gtf = [g for g in gene_names if g not in genes]
    if missing_gtf:
        print(f"  {len(missing_gtf)} gene(s) not found in GTF, skipped: "
              f"{missing_gtf[:10]}{'...' if len(missing_gtf) > 10 else ''}", file=sys.stderr)
    gene_names = [g for g in gene_names if g in genes]
    gene_cds_len = {g: cds_length(genes[g]) for g in gene_names}

    his_tx_by_gene = load_his_tx_positions(hisPicklePath)
    print(f"  {sum(1 for g in gene_names if his_tx_by_gene.get(g))}/{len(gene_names)} genes "
          f"have >=1 His codon.", file=sys.stderr)

    bam_paths = load_ribo_bam_list(riboBamListPath, condition)
    print(f"Building per-gene depth tracks for {len(gene_names)} gene(s) "
          f"({len(bam_paths)} ribo-seq BAM(s), condition={condition}) ...", file=sys.stderr)
    shadow_tx_depth = {}
    ribo_tx_depth = {}
    sites_by_gene = {}
    for gname in gene_names:
        gene = genes[gname]
        gdf = shadow_df[shadow_df["shadow_gene"] == gname]
        # shadow_coverage_track's own default clips to gene["exons"] only --
        # widen it the same way ribo_coverage_track's fetch window is widened
        # below, else the real flank-region shadow calls the HMM fix now
        # scores would get silently clipped away right back to a wall here.
        shadow_gpos_depth = shadow_coverage_track(gdf, gname, gene, shadow_cutoff, min_run_nt,
                                                  flank_5p=window_nt, flank_3p=window_nt)
        shadow_tx_depth[gname] = gene_depth_to_tx(shadow_gpos_depth, gene, flank_nt=window_nt)
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
        # raw per-site P_B, no run thresholding -- the "by position" figures'
        # own data source, independent of shadow_cutoff/min_run_nt entirely
        sites_by_gene[gname] = gene_bypos_scores(gdf, gene, flank_nt=window_nt)

    print(f"Building His-codon metagene (window=+/-{window_nt}nt) ...", file=sys.stderr)
    his_shadow_acc = build_his_density_metagene(shadow_tx_depth, his_tx_by_gene, window_nt)
    his_ribo_acc   = build_his_density_metagene(ribo_tx_depth, his_tx_by_gene, window_nt)
    n_genes_his_shadow = sum(1 for g, d in shadow_tx_depth.items() if d and his_tx_by_gene.get(g))
    n_genes_his_ribo   = sum(1 for g, d in ribo_tx_depth.items() if d and his_tx_by_gene.get(g))
    plot_his_metagene(his_shadow_acc, his_ribo_acc, f"{outPrefix}.his_metagene.pdf", window_nt,
                      n_genes_shadow=n_genes_his_shadow, n_genes_ribo=n_genes_his_ribo)

    print(f"Building start/stop-codon metagene (window=+/-{window_nt}nt) ...", file=sys.stderr)
    start_shadow_acc, stop_shadow_acc = build_startstop_density_metagene(
        shadow_tx_depth, gene_cds_len, window_nt)
    start_ribo_acc, stop_ribo_acc = build_startstop_density_metagene(
        ribo_tx_depth, gene_cds_len, window_nt)
    n_genes_shadow = sum(1 for d in shadow_tx_depth.values() if d)
    n_genes_ribo   = sum(1 for d in ribo_tx_depth.values() if d)
    plot_startstop_metagene(start_shadow_acc, stop_shadow_acc, start_ribo_acc, stop_ribo_acc,
                            f"{outPrefix}.startstop_metagene.pdf", window_nt,
                            n_genes_shadow=n_genes_shadow, n_genes_ribo=n_genes_ribo)

    print(f"Building by-position (raw P_B, no run thresholding) metagenes "
          f"(window=+/-{window_nt}nt) ...", file=sys.stderr)
    his_bypos_acc = build_his_bypos_metagene(sites_by_gene, his_tx_by_gene, window_nt)
    n_sites_his_bypos = sum(n for _s, n in his_bypos_acc.values())
    n_genes_his_bypos = sum(1 for g in gene_names if his_tx_by_gene.get(g) and sites_by_gene.get(g))
    plot_his_bypos_metagene(his_bypos_acc, f"{outPrefix}.his_metagene_bypos.pdf", window_nt,
                            n_genes=n_genes_his_bypos, n_sites=n_sites_his_bypos)

    start_bypos_acc, stop_bypos_acc = build_startstop_bypos_metagene(
        sites_by_gene, gene_cds_len, window_nt)
    n_sites_start_bypos = sum(n for _s, n in start_bypos_acc.values())
    n_sites_stop_bypos  = sum(n for _s, n in stop_bypos_acc.values())
    n_genes_bypos = sum(1 for g in gene_names if sites_by_gene.get(g))
    plot_startstop_bypos_metagene(start_bypos_acc, stop_bypos_acc,
                                  f"{outPrefix}.startstop_metagene_bypos.pdf", window_nt,
                                  n_genes=n_genes_bypos,
                                  n_sites_start=n_sites_start_bypos, n_sites_stop=n_sites_stop_bypos)


if __name__ == "__main__":
    main(sys.argv[1:])
