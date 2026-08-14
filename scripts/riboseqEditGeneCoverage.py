"""
riboseqEditGeneCoverage.py -- Liam Tran, August 2026

Per-gene IGV-style coverage track, sibling to riboseqGeneCoverage.py's own
plot_gene_track: same ribo-seq depth + shadow-call depth panels, same
gene-model strip and His-codon markers (all reused read-only from that
module, not reimplemented), PLUS a third track this module adds -- RAW
per-position A->G edit frequency, i.e. shadow_edit pooled directly across
reads at each shadow_gpos, with NO HMM/window/Markov model involved at
all. This is the empirical signal write_shadow_calls_to_df carries
alongside whatever model scored the parquet (its own docstring: "NOT
thresholded on P_B -- every scored site for every read... gets a row
here"); shadow_P_B/shadow_P_A are that model's POSTERIOR on top of this
same raw edit call. Plotting the raw frequency alongside the model's own
shadow-call depth is a sanity check on what the model is doing with the
same underlying observations -- does the model's protected-run track
actually track a real DIP in raw editing, or is it calling protection
somewhere the raw signal doesn't obviously support?

Un-modeled also means un-smoothed: raw_edit_freq_track pools every read's
own 0/1 edit call at each position with no run-averaging or Bayesian
shrinkage, so a position with few reads can show a noisy 0% or 100%
frequency. A dedicated support panel (n reads scored at that position,
directly below the frequency panel) is plotted alongside it for exactly
this reason -- same "never show a magnitude without its own support"
principle used throughout this codebase (e.g. shadowMetagene.py's
by-position views, shadowCallSizeQC.py's n_calls panel) -- so a
thinly-supported frequency reads as such rather than looking as
trustworthy as a well-covered one.

Four panels per gene, bottom to top (gene model strip first, via
draw_gene_model_strip, same as plot_gene_track): n reads scored (edit
frequency's own support), raw edit frequency (0-1 axis), shadow-call
depth, ribo-seq depth (short/long split, same RIBO_SHORT_LENGTHS/
RIBO_LONG_LENGTHS convention as riboseqGeneCoverage.py). All four share
one genomic-position x-axis (graph.axis.linkedaxis to draw_gene_model_strip's
own axis), same PyX house style as plot_gene_track (canvas + stacked
graphxy panels at manually-set ypos). No individual-reads section and no
squished-pileup figure here -- riboseqGeneCoverage.py already has both;
this module is scoped to the one new track it adds plus the two existing
depth tracks needed to compare against it, not a full reimplementation of
that module's own feature set.

Run:
  python3 riboseqEditGeneCoverage.py shadowCallsParquet riboBamList.txt condition gtfFile hisPicklePath outPrefix [geneNames] [shadow_cutoff] [min_run_nt] [flank_nt] [colorMapPath] [rep]
Same arguments, same meaning, as riboseqGeneCoverage.py's own CLI (see
that module's docstring) -- shadowCallsParquet is ONE library's
shadow_calls.parquet, riboBamList.txt/condition select the matching
ribo-seq BAMs, gtfFile is the yeast GTF, hisPicklePath is
findHisCodonPositions.py's output, outPrefix names the output (one PDF
per gene, written into a directory named after shadowCallsParquet itself,
same convention as riboseqGeneCoverage.py), geneNames (optional,
comma-separated) restricts which genes get a track (default: every gene
in shadowCallsParquet), shadow_cutoff/min_run_nt (optional) override the
shadow-call P_B cutoff/run-length floor (defaults match
riboseqGeneCoverage.py's own: 0.5/25nt) -- these only affect the
shadow-call depth panel, NOT the raw edit frequency panel, which has no
cutoff at all by design (see module docstring) -- and flank_nt (optional,
default 150) MUST match whatever --flank_nt runHMMPerGene.py actually
scored shadowCallsParquet with, same requirement as
riboseqGeneCoverage.py's own flank_nt.

colorMapPath (optional) is a manuscript color-map TSV (name, rep, path,
hex_color, no leading '#' -- the same convention/file used across this
codebase's other --color_map options, e.g. shadowCallSizeQC.py) matched
against libraryID ("condition-rep", condition from this script's own
`condition` argument and rep from the optional `rep` argument below) --
when given, ONLY the raw-edit-frequency curve picks up that library's
color (ribo-seq and shadow-call depth keep their existing fixed
grey/orange colors regardless, since those aren't library-specific
signals the way the raw edit call is -- see plot_gene_edit_track's own
docstring). rep (optional) is this library's own rep label (e.g. "rep1")
-- omit it to match on bare condition instead (whichever rep's color was
read first in colorMapPath for that condition, per load_color_map's own
bare-name fallback); a library with no match in colorMapPath falls back
to DEFAULT_EDIT_COLOR (with a warning) rather than erroring.
"""
import sys, os, collections
import pandas as pd

from runHMMPerGene import parse_gtf, tex_escape, compute_flank_caps
from polysomeShadowHMMQC import load_his_codon_gpos
from riboseqGeneCoverage import (
    load_ribo_bam_list, ribo_coverage_track, shadow_coverage_track,
    genes_in_shadow_calls, load_shadow_calls_df, draw_gene_model_strip,
    SHADOW_CUTOFF, MIN_RUN_NT, TARGET_LENGTHS, RIBO_SHORT_LENGTHS,
    RIBO_LONG_LENGTHS, FLANK_NT,
)

MIN_EDIT_OBS = 1   # raw_edit_freq_track: drop a position below this many pooled
                   # observations -- default 1 (no filtering) since the support
                   # count is ALSO returned/plotted for the caller to judge,
                   # rather than being silently hidden here
DEFAULT_EDIT_COLOR = (0.7, 0, 1, 0.2)   # cmyk fallback for the edit-freq curve when
                                       # no colorMapPath is given, or this library
                                       # has no entry in it -- same green used before
                                       # colorMapPath support existed


def load_color_map(path: str) -> dict:
    """
    Parse a manuscript color-map TSV with columns:
        sample_name, rep, path, hex_color (no leading '#')
    Returns a dict keyed by "name_rep", "name-rep" (this codebase's
    libraryID convention, "condition-rep"), and bare "name" (first match
    wins for the bare key) mapping to "#RRGGBB". Same duplicated-per-script
    convention as every other script's own load_color_map in this codebase
    (e.g. shadowCallSizeQC.py, polysomeShadowHMMQC.py) -- not imported from
    them, kept as its own copy here too.
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


def load_shadow_calls_with_edits(shadow_parquet_path):
    """
    Just the columns raw_edit_freq_track needs: read_id, shadow_gene,
    shadow_gpos, shadow_edit -- shadow_edit is write_shadow_calls_to_df's
    own per-read, per-site RAW 0/1 edit call (rec["edits"][k], this read's
    actual A or G at that position), NOT the HMM's shadow_P_B/shadow_P_A
    posterior on top of it (see load_shadow_calls_df, riboseqGeneCoverage.py's
    own loader, which pulls those instead for its different callers).
    """
    return pd.read_parquet(shadow_parquet_path,
                           columns=["read_id", "shadow_gene", "shadow_gpos", "shadow_edit"])


def raw_edit_freq_track(gene_df, min_obs=MIN_EDIT_OBS):
    """
    {gpos: (edit_fraction, n_obs)} -- raw empirical A->G edit frequency at
    every scored (Ref=A) genomic position for one gene, pooling
    shadow_edit across every read in gene_df via shadow_gpos. Deliberately
    the UNMODELED signal (see module docstring): no P_B cutoff, no run
    merging, no posterior -- just observed_edits / observed_sites at each
    position, the same raw ingredient the HMM/window/Markov model's own
    posterior is built FROM, not a derivative of the model's own output.

    gpos entries can be None/NaN (a scored site with no gpos_to_tx_map
    entry for it, e.g. outside the gene's own padded window -- see
    write_shadow_calls_to_df) -- dropped here the same way every other
    per-site consumer in this codebase drops them.

    min_obs: positions below this many pooled reads are excluded from the
    returned dict entirely (default 1 -- no filtering; n_obs is still
    returned for every position kept, so plot_gene_edit_track's own
    support panel is what actually flags a thin position, not a silent
    drop here).

    Returns {gpos: (edit_fraction, n_obs)}.
    """
    edit_sum = collections.defaultdict(int)
    n_obs = collections.defaultdict(int)
    for row in gene_df.itertuples(index=False):
        for gpos, edit in zip(row.shadow_gpos, row.shadow_edit):
            if gpos is None or (isinstance(gpos, float) and gpos != gpos):
                continue
            gpos = int(gpos)
            edit_sum[gpos] += int(edit)
            n_obs[gpos] += 1
    return {g: (edit_sum[g] / n_obs[g], n_obs[g])
            for g in n_obs if n_obs[g] >= min_obs}


def plot_gene_edit_track(gene_name, gene, ribo_depth_short, ribo_depth_long, shadow_depth,
                         edit_freq_track, pdf_path, his_gpos=None,
                         shadow_cutoff=SHADOW_CUTOFF, min_run_nt=MIN_RUN_NT,
                         flank_5p=0, flank_3p=0, edit_color=None):
    """
    Combined per-gene figure, PyX/PDF -- same house style as
    riboseqGeneCoverage.py's plot_gene_track (canvas + stacked graphxy
    panels at manually-set ypos, graph.axis.linkedaxis to share one
    genomic-position x-axis, draw_gene_model_strip's own strand-aware
    axis reversal handling the 5'->3' left-to-right convention for both
    strands automatically). Bottom to top:
      - gene-model strip (draw_gene_model_strip -- UTR5/CDS/UTR3 rects,
        His-codon lines, 5'/3' labels)
      - n reads scored (support for the edit-frequency panel directly
        above it -- see module docstring for why this needs its own
        panel rather than trusting the frequency alone)
      - raw edit frequency (0-1 axis; NOT the model's shadow_P_B/P_A --
        see raw_edit_freq_track)
      - shadow-call depth (the model's own qualifying-run track, same
        shadow_coverage_track this figure's data comes from as
        riboseqGeneCoverage.py's plot_gene_track)
      - ribo-seq depth, split into RIBO_SHORT_LENGTHS/RIBO_LONG_LENGTHS
        (same two-length-class convention as plot_gene_track, same
        rationale: pooling would wash out a real difference between the
        two ribosome conformational states)

    his_gpos/flank_5p/flank_3p: same meaning as plot_gene_track's own
    (see that function's docstring) -- His-codon markers and panel-width
    padding past the gene's own annotated span.

    edit_color: overrides the raw-edit-frequency curve's own color (a pyx
    color object, e.g. from hex_to_pyx_color) -- this is the ONE curve in
    this figure that's specifically "this library's own data" (ribo-seq
    is condition-pooled across BAMs, shadow-call depth stays the fixed
    orange used everywhere else in this codebase for that signal), so
    it's the one that picks up the manuscript color map (see main()) when
    a colorMapPath is given; defaults to DEFAULT_EDIT_COLOR otherwise.
    Ribo-seq/shadow-call colors are NOT parameterized here on purpose --
    only the edit-frequency curve was asked to vary by library.
    """
    from pyx import canvas, graph, color, style, text as pyx_text, path

    col_ribo_short = color.grey(0.75)
    col_ribo_long  = color.grey(0.3)
    col_shadow = color.cmyk(0, 0.5, 1, 0)
    col_edit   = edit_color if edit_color is not None else color.cmyk(*DEFAULT_EDIT_COLOR)
    col_support = color.grey(0.5)

    if gene["strand"] == "+":
        lo, hi = gene["gene_start"] - flank_5p, gene["gene_end"] + flank_3p
    else:
        lo, hi = gene["gene_start"] - flank_3p, gene["gene_end"] + flank_5p
    xs = list(range(lo, hi))
    ribo_short_ys = [ribo_depth_short.get(x, 0) for x in xs]
    ribo_long_ys  = [ribo_depth_long.get(x, 0) for x in xs]
    shadow_ys = [shadow_depth.get(x, 0) for x in xs]
    edit_xs   = [x for x in xs if x in edit_freq_track]
    edit_ys   = [edit_freq_track[x][0] for x in edit_xs]
    support_ys = [edit_freq_track[x][1] for x in edit_xs]
    his_in_range = sorted(g for g in (his_gpos or ()) if lo <= g < hi)

    panel_w, ribo_h, shadow_h, edit_h, support_h, model_h, gap = 14, 3.5, 3.5, 2.5, 1.5, 0.6, 1.1
    ribo_max = max(ribo_short_ys + ribo_long_ys) * 1.1 if any(ribo_short_ys) or any(ribo_long_ys) else 1.0
    shadow_max = max(shadow_ys) * 1.1 if any(shadow_ys) else 1.0
    support_max = max(support_ys) * 1.1 if support_ys else 1.0

    c = canvas.canvas()

    g_model, col_his = draw_gene_model_strip(c, gene, lo, hi, his_in_range, panel_w, model_h)

    support_ypos = model_h + gap
    g_support = graph.graphxy(
        width=panel_w, height=support_h, xpos=0, ypos=support_ypos,
        x=graph.axis.linkedaxis(g_model.axes["x"], painter=graph.axis.painter.linked()),
        y=graph.axis.linear(min=0, max=support_max, title="n reads scored"))
    c.insert(g_support)
    if edit_xs:
        g_support.plot(graph.data.points(list(zip(edit_xs, support_ys)), x=1, y=2),
                       [graph.style.line([col_support, style.linewidth.Thick])])
    for hg in his_in_range:
        g_support.plot(graph.data.function(f"x(y)={hg}", min=0, max=support_max),
                       [graph.style.line([col_his, style.linewidth.thin, style.linestyle.dashed])])

    edit_ypos = support_ypos + support_h + gap
    g_edit = graph.graphxy(
        width=panel_w, height=edit_h, xpos=0, ypos=edit_ypos,
        x=graph.axis.linkedaxis(g_model.axes["x"], painter=graph.axis.painter.linked()),
        y=graph.axis.linear(min=0, max=1, title="raw edit freq."))
    c.insert(g_edit)
    if edit_xs:
        g_edit.plot(graph.data.points(list(zip(edit_xs, edit_ys)), x=1, y=2),
                    [graph.style.line([col_edit, style.linewidth.Thick])])
    for hg in his_in_range:
        g_edit.plot(graph.data.function(f"x(y)={hg}", min=0, max=1),
                    [graph.style.line([col_his, style.linewidth.thin, style.linestyle.dashed])])

    shadow_ypos = edit_ypos + edit_h + gap
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
    c.text(g_ribo.xpos + g_ribo.width / 2., top_ypos + 0.35,
          f"{tex_escape(gene['chrom'])}:{lo:,}-{hi:,}, {gene['strand']} strand -- "
          f"{len(his_in_range)} His codon(s) -- shadow-call P$_B>${shadow_cutoff}, "
          f"len$>${min_run_nt}nt -- raw edit freq. is un-thresholded/un-modeled",
          [pyx_text.halign.center, pyx_text.size.scriptsize])

    leg_y = top_ypos - 0.35
    c.stroke(path.line(0.2, leg_y, 0.85, leg_y), [col_ribo_short, style.linewidth.Thick])
    c.text(1.05, leg_y, f"ribo-seq {RIBO_SHORT_LENGTHS[0]}/{RIBO_SHORT_LENGTHS[1]}nt",
          [pyx_text.valign.middle, pyx_text.size.tiny])
    c.stroke(path.line(3.5, leg_y, 4.15, leg_y), [col_ribo_long, style.linewidth.Thick])
    c.text(4.35, leg_y, f"ribo-seq {RIBO_LONG_LENGTHS[0]}/{RIBO_LONG_LENGTHS[1]}nt",
          [pyx_text.valign.middle, pyx_text.size.tiny])

    c.writePDFfile(str(pdf_path))
    print(f"Wrote {pdf_path}", file=sys.stderr)


def main(args):
    shadowPath, riboBamListPath, condition, gtfPath, hisPicklePath, outPrefix = (
        args[0], args[1], args[2], args[3], args[4], args[5])
    colorMapPath  = args[6] if len(args) > 6 and args[6] else None
    rep           = args[7] if len(args) > 7 and args[7] else None
    gene_names_arg = args[8] if len(args) > 8 and args[8] else None
    shadow_cutoff = float(args[9]) if len(args) > 9 else SHADOW_CUTOFF
    min_run_nt    = int(args[10])   if len(args) > 10 else MIN_RUN_NT
    flank_nt      = int(args[11])   if len(args) > 11 else FLANK_NT


    out_dir = os.path.dirname(outPrefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    edit_color = None
    if colorMapPath:
        color_map = load_color_map(colorMapPath)
        libraryID = f"{condition}-{rep}" if rep else condition
        hexcol = color_map.get(libraryID) or color_map.get(condition)
        if hexcol:
            edit_color = hex_to_pyx_color(hexcol)
            print(f"Loaded manuscript color {hexcol} for {libraryID!r} from "
                  f"{colorMapPath} -- coloring the edit-frequency curve with it.",
                  file=sys.stderr)
        else:
            print(f"  WARNING: no color found in {colorMapPath} for {libraryID!r}; "
                  f"edit-frequency curve falls back to its default color.", file=sys.stderr)

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
    print(f"{len(bam_paths)} ribo-seq BAM(s) (condition={condition}).", file=sys.stderr)

    shadow_df = load_shadow_calls_df(shadowPath)
    edit_df   = load_shadow_calls_with_edits(shadowPath)

    if gene_names_arg:
        track_genes = [g.strip() for g in gene_names_arg.split(",") if g.strip()]
        missing_track = [g for g in track_genes if g not in genes]
        if missing_track:
            print(f"  requested track gene(s) not in GTF, skipped: {missing_track}",
                  file=sys.stderr)
        track_genes = [g for g in track_genes if g in genes]
    else:
        track_genes = list(gene_list)
        print(f"No geneNames given -- plotting all {len(track_genes)} genes.", file=sys.stderr)

    parquet_stem = os.path.splitext(os.path.basename(shadowPath))[0]
    track_dir = os.path.join(out_dir, parquet_stem) if out_dir else parquet_stem
    os.makedirs(track_dir, exist_ok=True)

    print(f"Building per-gene edit/ribo/shadow tracks for {len(track_genes)} gene(s) "
          f"into {track_dir}/ ...", file=sys.stderr)
    for gname in track_genes:
        gene = genes[gname]
        flank_5p, flank_3p = flank_caps[gname]
        padded_gene = dict(gene)
        if gene["strand"] == "+":
            padded_gene["gene_start"] = max(0, gene["gene_start"] - flank_5p)
            padded_gene["gene_end"]   = gene["gene_end"] + flank_3p
        else:
            padded_gene["gene_start"] = max(0, gene["gene_start"] - flank_3p)
            padded_gene["gene_end"]   = gene["gene_end"] + flank_5p
        ribo_depth_short = ribo_coverage_track(bam_paths, padded_gene, target_lengths=RIBO_SHORT_LENGTHS)
        ribo_depth_long  = ribo_coverage_track(bam_paths, padded_gene, target_lengths=RIBO_LONG_LENGTHS)
        gene_shadow_df = shadow_df[shadow_df["shadow_gene"] == gname]
        shadow_depth = shadow_coverage_track(gene_shadow_df, gname, gene, shadow_cutoff, min_run_nt,
                                             flank_5p=flank_5p, flank_3p=flank_3p)
        gene_edit_df = edit_df[edit_df["shadow_gene"] == gname]
        edit_freq_track = raw_edit_freq_track(gene_edit_df)
        plot_gene_edit_track(gname, gene, ribo_depth_short, ribo_depth_long, shadow_depth,
                             edit_freq_track, os.path.join(track_dir, f"{gname}.edit_track.pdf"),
                             his_gpos=his_gpos_by_gene.get(gname),
                             shadow_cutoff=shadow_cutoff, min_run_nt=min_run_nt,
                             flank_5p=flank_5p, flank_3p=flank_3p, edit_color=edit_color)


if __name__ == "__main__":
    main(sys.argv[1:])
