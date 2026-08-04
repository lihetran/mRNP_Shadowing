"""
riboseqShadowCorrelation.py -- Liam Tran, August 2026

How often does a ribo-seq read's P-site coincide with a shadow call from the
Nanopore HMM pipeline? P-site offsets used here were derived empirically in
riboseqPsiteCalibration.py (stop-codon-anchored 5'-end pileup, confirmed
sense-orientation, pooled across all 4 Wu_2019 BAMs): {21:14, 22:15, 28:16,
29:16} nt from a read's 5' end -- see that script's docstring for why (no
existing P-site-calling code anywhere in this codebase or
/data16/arriberelab-git, and why the standard start-codon/5'UTR pileup
method doesn't work on this GTF, which has almost no annotated 5'UTRs).

Definitions:
  shadow call = a WINDOW-based definition specific to this script (NOT the
      same as polysomeShadowHMMQC.py's extract_shadow_runs/MIN_RUN_NT,
      which every OTHER view in this codebase still uses unchanged): a
      scored site qualifies as a "center" if EVERY scored site within
      +/-CENTER_NT/2 nt of it (default 10nt total, so +/-5nt) -- including
      itself -- clears P_B >= cutoff (default FIXED_CUTOFF=0.7). Any
      qualifying center's surrounding TOTAL_CALL_NT-wide window (default
      45nt, i.e. +/-22.5nt) becomes a shadow call in full, regardless of
      what P_B looks like at the call's own edges -- unlike
      extract_shadow_runs, which requires EVERY site across the WHOLE call
      to individually clear cutoff. Rationale: a real ribosome footprint is
      a fixed ~40-50nt physical extent; requiring uniformly strong
      protection across that whole span is stricter than the biology
      needs if edge positions edit more freely than the well-protected
      center (e.g. from partial nuclease/TadA accessibility at the
      footprint's boundary) -- so only the CENTER has to prove it's
      protected, and the surrounding footprint-sized window is granted
      once it does. Adjacent/overlapping qualifying windows from different
      center sites in the same read are merged (interval union) into one
      call. See windowed_shadow_call_spans_by_read. Every (read, site)
      observation is still kept as its own row -- not deduplicated by
      genomic position first -- matching how the rest of this codebase
      (_gene_site_counts et al.) pools scored sites.
  P-site (mode="psite", the default) = a ribo-seq read's inferred ribosome
      P-site genomic position: the read's 5'-end genomic position, shifted
      PSITE_OFFSETS[length] nt in the gene's sense direction (+ for '+'
      strand genes, - for '-'). Only sense-orientation reads (alignment
      strand == gene strand), uniquely mapped (NH==1), length in
      PSITE_OFFSETS are used -- same filters validated in the calibration
      script. match = a shadow-call observation's shadow_gpos falls within
      +/-MATCH_TOL nt (default 0, i.e. exact) of >=1 ribo-seq P-site.
  read coverage (mode="coverage", the "less smart" alternative) = drops
      the P-site offset math entirely: a position counts as matched if ANY
      qualifying ribo-seq read's FULL alignment span [reference_start,
      reference_end) overlaps it at all -- i.e. "did a real footprint-sized
      read land anywhere within this shadow call's bounds", no calibration,
      no exact-position requirement. Same basic quality filters as psite
      mode (sense-orientation, NH==1, length in PSITE_OFFSETS) -- those
      aren't the "smart" part, only the offset-position math is dropped.
      See load_ribo_read_coverage.

Reports (same for both modes, so they're a true apples-to-apples
comparison of "smart" P-site matching vs. "dumb" read-overlap matching):
  - pooled 2x2 contingency: P(ribo-seq match at this site | shadow-called)
    vs. P(... | scored but NOT shadow-called), the background rate every
    scored site would show from baseline ribosome density -- so the
    shadow-called rate can be read as enrichment over that baseline, not a
    bare, context-free percentage. Tested with Fisher's exact test.
  - per-gene paired rates (same two quantities, one point per gene),
    box+strip, analogous to polysomeShadowHMMQC.py's plot_gene_call_rates.

Run:
  python3 riboseqShadowCorrelation.py shadowCallsParquet riboBamList.txt condition gtfFile outPrefix [cutoff] [center_nt] [total_nt] [mode]
where shadowCallsParquet is ONE library's shadow_calls.parquet (e.g.
minus3AT_146_shadow_calls.parquet), riboBamList.txt is e.g.
/data16/liam/working/260804_riboSeq_vs_PS/riboSeqBam.txt --
whitespace-delimited condition/rep/path rows (one BAM per row; multiple
reps of one condition are pooled), condition selects the MATCHING ribo-seq
condition for shadowCallsParquet's Nanopore library (e.g. "-3AT" for
minus3AT_146_shadow_calls.parquet, matched against the file's first
column), gtfFile is the yeast GTF, outPrefix names outputs, cutoff
(optional) overrides FIXED_CUTOFF, center_nt/total_nt (optional) override
CENTER_NT/TOTAL_CALL_NT (the center-window and full-call widths), and mode
(optional, "psite" or "coverage") selects which ribo-seq matching
definition to use (default "psite").
"""
import sys, collections
import numpy as np
import pandas as pd
import pysam
from scipy.stats import fisher_exact

from runHMMPerGene import parse_gtf

FIXED_CUTOFF   = 0.7
PSITE_OFFSETS  = {21: 14, 22: 15, 28: 16, 29: 16}   # nt from 5' end, confirmed empirically
MATCH_TOL      = 0        # +/- nt tolerance when matching a shadow site to a ribo-seq P-site
MIN_OBS_PER_GENE = 20      # per-gene view: need at least this many scored-site observations
CENTER_NT      = 10        # width of the "does the center clear cutoff" window
TOTAL_CALL_NT  = 45        # width of the full shadow call once a center qualifies (40-50nt target)


def load_ribo_bam_list(path, condition=None):
    """
    BAM paths from a whitespace-delimited list file. Each line is either
    "condition  rep  path" (the riboSeqBam.txt convention -- e.g. all 4
    Wu_2019 BAMs in one file, "-3AT"/"+3AT" x rep1/rep2) or a bare path
    with no condition/rep columns (the older flat-list convention this
    replaces). condition, if given, keeps only rows whose first column
    matches it exactly -- bare-path rows have no condition to match and
    are kept unconditionally. Every matching row's path is returned (reps
    of the same condition are pooled, same as before).
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


def load_ribo_psite_positions(bam_paths, genes, offsets=PSITE_OFFSETS):
    """
    {gene_name: collections.Counter({gpos: n_psite_reads})} -- ribo-seq
    P-site read depth per genomic position, pooled across every BAM in
    bam_paths. Sense-orientation (alignment strand == gene strand),
    uniquely-mapped (NH==1) reads only, restricted to lengths in `offsets`.

    P-site genomic position = read's mRNA-sense 5' end +/- offsets[length]
    nt in the gene's own reading direction -- purely a strand-aware
    genomic-coordinate shift, no transcript-coordinate map needed (unlike
    riboseqPsiteCalibration.py, which needed tx-space to anchor on the stop
    codon; here we only need genomic coordinates to match against
    shadow_gpos directly).
    """
    psites = collections.defaultdict(collections.Counter)

    for bam_path in bam_paths:
        print(f"  scanning {bam_path} ...", file=sys.stderr)
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for gname, gene in genes.items():
                direction = 1 if gene["strand"] == "+" else -1
                for read in bam.fetch(gene["chrom"], gene["gene_start"], gene["gene_end"]):
                    if read.is_unmapped:
                        continue
                    off = offsets.get(read.query_length)
                    if off is None:
                        continue
                    if read.has_tag("NH") and read.get_tag("NH") != 1:
                        continue

                    read_is_sense = (read.is_reverse == (gene["strand"] == "-"))
                    if not read_is_sense:
                        continue

                    five_prime_gpos = (read.reference_start if gene["strand"] == "+"
                                      else read.reference_end - 1)
                    p_site_gpos = five_prime_gpos + direction * off
                    psites[gname][p_site_gpos] += 1

    return psites


def load_ribo_read_coverage(bam_paths, genes, target_lengths=tuple(PSITE_OFFSETS)):
    """
    "Dumb" counterpart to load_ribo_psite_positions: {gene_name:
    Counter({gpos: n_reads_covering})} built from each qualifying read's
    FULL aligned span [reference_start, reference_end), not a single
    computed P-site position -- no offset calibration, no per-length
    lookup, no strand-aware shift math. A position counts as covered if
    ANY qualifying read's alignment overlaps it at all.

    Same basic quality filters as load_ribo_psite_positions (sense-
    orientation, NH==1, length in target_lengths) -- those aren't the
    "smart" part being tested here (a footprint-length/uniqueness/
    orientation floor is a basic correctness requirement, not a modeling
    choice), only the P-site-offset position math is dropped, in favor of
    "did a real footprint-sized read land anywhere in this span at all".
    """
    coverage = collections.defaultdict(collections.Counter)

    for bam_path in bam_paths:
        print(f"  scanning {bam_path} ...", file=sys.stderr)
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for gname, gene in genes.items():
                for read in bam.fetch(gene["chrom"], gene["gene_start"], gene["gene_end"]):
                    if read.is_unmapped:
                        continue
                    if read.query_length not in target_lengths:
                        continue
                    if read.has_tag("NH") and read.get_tag("NH") != 1:
                        continue

                    read_is_sense = (read.is_reverse == (gene["strand"] == "-"))
                    if not read_is_sense:
                        continue

                    for gpos in range(read.reference_start, read.reference_end):
                        coverage[gname][gpos] += 1

    return coverage


def windowed_shadow_call_spans_by_read(df, cutoff=FIXED_CUTOFF,
                                       center_nt=CENTER_NT, total_nt=TOTAL_CALL_NT):
    """
    {read_id: [(lo, hi), ...]} -- this script's own "shadow call" definition
    (see module docstring), deliberately DIFFERENT from
    polysomeShadowHMMQC.py's extract_shadow_runs/MIN_RUN_NT, which every
    other view in this codebase still uses unchanged.

    For each scored site i (sorted by gpos within its read), site i is a
    qualifying CENTER iff every scored site within +/-center_nt/2 of it
    (itself included) has P_B >= cutoff. A qualifying center's call window
    is [gpos_i - total_nt/2, gpos_i + total_nt/2] -- granted in full,
    regardless of P_B at the window's own edges. Overlapping/adjacent call
    windows (common: several consecutive sites along one strong stretch all
    qualify as their own center) are merged via interval union, so one real
    protected stretch doesn't fragment into duplicate overlapping calls.

    O(n) per read via a two-pointer sweep over gpos-sorted sites (both
    pointers only advance, never retreat, since sites are sorted).
    """
    half_center = center_nt / 2.0
    half_total  = total_nt / 2.0
    spans_by_read = collections.defaultdict(list)
    n_reads_with_call = 0

    for row in df.itertuples(index=False):
        pb   = [float(x) for x in row.shadow_P_B]
        gpos = [int(g)   for g in row.shadow_gpos]
        n = len(pb)
        if n == 0:
            continue
        order  = sorted(range(n), key=lambda k: gpos[k])
        pb_s   = [pb[k]   for k in order]
        gpos_s = [gpos[k] for k in order]

        windows = []
        lo_ptr = 0
        hi_ptr = -1
        for i in range(n):
            while gpos_s[lo_ptr] < gpos_s[i] - half_center:
                lo_ptr += 1
            while hi_ptr + 1 < n and gpos_s[hi_ptr + 1] <= gpos_s[i] + half_center:
                hi_ptr += 1
            if all(pb_s[k] >= cutoff for k in range(lo_ptr, hi_ptr + 1)):
                windows.append((gpos_s[i] - half_total, gpos_s[i] + half_total))

        if not windows:
            continue
        windows.sort()
        merged = [windows[0]]
        for lo, hi in windows[1:]:
            if lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        spans_by_read[row.read_id].extend(merged)
        n_reads_with_call += 1

    print(f"  {n_reads_with_call}/{len(df)} reads have >=1 qualifying "
          f"{center_nt:.0f}nt-center / {total_nt:.0f}nt-call shadow window.", file=sys.stderr)
    return dict(spans_by_read)


def load_shadow_site_observations(shadow_parquet_path, cutoff=FIXED_CUTOFF,
                                  center_nt=CENTER_NT, total_nt=TOTAL_CALL_NT):
    """
    One row per (read, scored-site) observation: gene, gpos, P_B,
    is_shadow_called -- exploded from shadow_calls.parquet's per-read list
    columns (shadow_gene broadcast to every site in that read's
    shadow_gpos/shadow_P_B lists). Not deduplicated by (gene, gpos) first --
    see module docstring for why.

    is_shadow_called: True iff this site's gpos falls within one of ITS OWN
    READ's qualifying call windows (see windowed_shadow_call_spans_by_read)
    -- this script's own center-triggered, fixed-width definition of
    "shadow", not polysomeShadowHMMQC.py's contiguous-run one. Checked
    per-read, not pooled across a gene's other reads, since a call window
    is only meaningful for the read it came from.

    Returns a DataFrame with columns [gene, gpos, P_B, is_shadow_called].
    """
    df = pd.read_parquet(shadow_parquet_path,
                         columns=["read_id", "shadow_gene", "shadow_gpos", "shadow_P_B"])
    run_spans = windowed_shadow_call_spans_by_read(df, cutoff, center_nt, total_nt)

    read_ids_col, genes_col, gpos_col, pb_col = [], [], [], []
    for read_id, gene, gpos_list, pb_list in zip(
            df["read_id"], df["shadow_gene"], df["shadow_gpos"], df["shadow_P_B"]):
        for gpos, pb in zip(gpos_list, pb_list):
            read_ids_col.append(read_id)
            genes_col.append(gene)
            gpos_col.append(int(gpos))
            pb_col.append(float(pb))
    obs = pd.DataFrame({"read_id": read_ids_col, "gene": genes_col,
                        "gpos": gpos_col, "P_B": pb_col})

    def _in_a_qualifying_run(read_id, gpos):
        return any(lo <= gpos <= hi for lo, hi in run_spans.get(read_id, ()))

    obs["is_shadow_called"] = [_in_a_qualifying_run(rid, gp)
                              for rid, gp in zip(obs["read_id"], obs["gpos"])]
    return obs


def psite_depth_at(gene, gpos, psites_by_gene, tol=MATCH_TOL):
    """Ribo-seq P-site read depth at gpos (summed over +/-tol nt)."""
    counter = psites_by_gene.get(gene)
    if not counter:
        return 0
    if tol == 0:
        return counter.get(gpos, 0)
    return sum(counter.get(gpos + d, 0) for d in range(-tol, tol + 1))


def has_psite_match(gene, gpos, psites_by_gene, tol=MATCH_TOL):
    return psite_depth_at(gene, gpos, psites_by_gene, tol) > 0


def annotate_matches(obs, psites_by_gene, tol=MATCH_TOL):
    """Adds has_psite (bool) and psite_depth (int) columns to obs (from
    load_shadow_site_observations). psite_depth is the more sensitive of
    the two -- ribo-seq coverage depth is highly skewed by gene expression
    (confirmed on real data: 0 to 451 reads at a single position, 25th
    percentile is 0 even among positions with SOME depth), so a well-
    covered gene's sites nearly all clear "has_psite" regardless of
    whether they're specifically shadow-called -- has_psite alone can
    saturate; psite_depth still has room to show a shadow-vs-background
    difference in genes where presence/absence stops discriminating."""
    obs = obs.copy()
    depths = [psite_depth_at(g, p, psites_by_gene, tol)
             for g, p in zip(obs["gene"], obs["gpos"])]
    obs["psite_depth"] = depths
    obs["has_psite"] = [d > 0 for d in depths]
    return obs


def pooled_contingency(obs):
    """
    2x2 (is_shadow_called x has_psite) contingency, pooled across every
    observation regardless of gene -- the headline "how often" answer, with
    the non-shadow-called rate as the background/baseline it's judged
    against. Returns (table, rate_shadow, rate_background, odds_ratio, pval).
    table = [[n_shadow_and_psite, n_shadow_no_psite],
             [n_bg_and_psite,     n_bg_no_psite]]
    """
    shadow = obs[obs["is_shadow_called"]]
    bg     = obs[~obs["is_shadow_called"]]
    table = [[int((shadow["has_psite"]).sum()), int((~shadow["has_psite"]).sum())],
            [int((bg["has_psite"]).sum()),     int((~bg["has_psite"]).sum())]]
    rate_shadow = table[0][0] / len(shadow) if len(shadow) else float("nan")
    rate_bg     = table[1][0] / len(bg) if len(bg) else float("nan")
    odds_ratio, pval = fisher_exact(table)
    return table, rate_shadow, rate_bg, odds_ratio, pval


def per_gene_rates(obs, min_obs_per_gene=MIN_OBS_PER_GENE):
    """
    {gene: (rate_shadow, n_shadow, rate_bg, n_bg)} -- same two quantities
    as pooled_contingency, split per gene. A gene needs >=min_obs_per_gene
    observations in EACH of the shadow-called / background groups to
    contribute a rate for that group (too few sites to trust a rate from
    otherwise) -- same "raw-then-filter" floor convention used throughout
    polysomeShadowHMMQC.py's per-gene metrics.
    """
    out = {}
    for gene, g_obs in obs.groupby("gene"):
        shadow = g_obs[g_obs["is_shadow_called"]]
        bg     = g_obs[~g_obs["is_shadow_called"]]
        if len(shadow) < min_obs_per_gene or len(bg) < min_obs_per_gene:
            continue
        rate_shadow = shadow["has_psite"].mean()
        rate_bg     = bg["has_psite"].mean()
        out[gene] = (rate_shadow, len(shadow), rate_bg, len(bg))
    return out


def per_gene_depth(obs, min_obs_per_gene=MIN_OBS_PER_GENE):
    """
    {gene: (mean_depth_shadow, n_shadow, mean_depth_bg, n_bg)} -- depth-
    weighted counterpart to per_gene_rates. Mean ribo-seq P-site depth is
    more sensitive than has_psite when a gene's coverage is deep enough
    that most sites already clear "any P-site" regardless of shadow
    status -- there's still room for shadow-called sites to show HIGHER
    depth specifically, even where the binary rate has saturated.
    """
    out = {}
    for gene, g_obs in obs.groupby("gene"):
        shadow = g_obs[g_obs["is_shadow_called"]]
        bg     = g_obs[~g_obs["is_shadow_called"]]
        if len(shadow) < min_obs_per_gene or len(bg) < min_obs_per_gene:
            continue
        out[gene] = (shadow["psite_depth"].mean(), len(shadow),
                    bg["psite_depth"].mean(), len(bg))
    return out


def paired_signed_rank_test(gene_metric):
    """
    Wilcoxon signed-rank test on (shadow_value, bg_value) pairs from
    per_gene_rates / per_gene_depth's output -- is the shadow-called value
    systematically higher (or lower) than the SAME gene's background value,
    across genes? Paired, so it controls for gene-level baseline (overall
    expression/coverage) the way the raw pooled contingency table cannot.
    Returns (n_genes, median_diff, statistic, pval), or (n, nan, nan, nan)
    if too few genes / all differences are exactly zero.
    """
    from scipy.stats import wilcoxon
    genes = sorted(gene_metric)
    shadow_vals = np.array([gene_metric[g][0] for g in genes])
    bg_vals     = np.array([gene_metric[g][2] for g in genes])
    diffs = shadow_vals - bg_vals
    median_diff = float(np.median(diffs))
    if len(genes) < 2 or np.all(diffs == 0):
        return len(genes), median_diff, float("nan"), float("nan")
    stat, pval = wilcoxon(shadow_vals, bg_vals)
    return len(genes), median_diff, stat, pval


def plot_gene_metric(gene_metric, pdf_path, ylabel, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    genes = sorted(gene_metric)
    shadow_vals = [gene_metric[g][0] for g in genes]
    bg_vals     = [gene_metric[g][2] for g in genes]
    n_genes, median_diff, _stat, pval = paired_signed_rank_test(gene_metric)

    fig, ax = plt.subplots(figsize=(5, 5.5))
    ax.boxplot([shadow_vals, bg_vals], labels=["shadow-called\nsites", "background\n(non-shadow) sites"],
              showfliers=False, zorder=3)
    rng = np.random.default_rng(0)
    for i, vals in enumerate([shadow_vals, bg_vals], start=1):
        jitter = rng.normal(0, 0.05, size=len(vals))
        ax.scatter([i + j for j in jitter], vals, s=10, alpha=0.5, color="tab:blue", zorder=2)
    for gi in range(len(genes)):
        ax.plot([1, 2], [shadow_vals[gi], bg_vals[gi]], color="grey", linewidth=0.5, alpha=0.4, zorder=1)

    ax.set_ylabel(ylabel)
    pval_txt = f"p={pval:.2g}" if pval == pval else "p=n/a"
    ax.set_title(f"{title}\n(n={n_genes} genes, paired; Wilcoxon {pval_txt}, "
                f"median diff={median_diff:+.3g})", fontsize=9)
    fig.tight_layout()
    fig.savefig(pdf_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {pdf_path}", file=sys.stderr)


def main(args):
    shadowPath, riboBamListPath, condition, gtfPath, outPrefix = (
        args[0], args[1], args[2], args[3], args[4])
    cutoff    = float(args[5]) if len(args) > 5 else FIXED_CUTOFF
    center_nt = float(args[6]) if len(args) > 6 else CENTER_NT
    total_nt  = float(args[7]) if len(args) > 7 else TOTAL_CALL_NT
    mode = args[8] if len(args) > 8 else "psite"
    if mode not in ("psite", "coverage"):
        raise ValueError(f"mode must be 'psite' or 'coverage', got {mode!r}")

    genes = parse_gtf(gtfPath)
    print(f"{len(genes):,} genes parsed from GTF.", file=sys.stderr)

    ribo_bam_paths = load_ribo_bam_list(riboBamListPath, condition)
    print(f"Loading ribo-seq {'P-sites' if mode == 'psite' else 'read coverage'} "
          f"from {len(ribo_bam_paths)} BAM(s) (condition={condition}, mode={mode})...",
          file=sys.stderr)
    psites_by_gene = (load_ribo_psite_positions(ribo_bam_paths, genes) if mode == "psite"
                      else load_ribo_read_coverage(ribo_bam_paths, genes))
    n_psite_genes = sum(1 for c in psites_by_gene.values() if c)
    print(f"  positions found for {n_psite_genes} genes.", file=sys.stderr)

    print(f"Loading shadow calls from {shadowPath} "
          f"(cutoff={cutoff}, center_nt={center_nt}, total_nt={total_nt})...", file=sys.stderr)
    obs = load_shadow_site_observations(shadowPath, cutoff, center_nt, total_nt)
    print(f"  {len(obs):,} (read, site) observations across "
          f"{obs['gene'].nunique()} genes "
          f"({obs['is_shadow_called'].sum():,} shadow-called).", file=sys.stderr)

    obs = annotate_matches(obs, psites_by_gene)
    unit = "P-site" if mode == "psite" else "read"

    table, rate_shadow, rate_bg, odds_ratio, pval = pooled_contingency(obs)
    print(f"\nPooled contingency (is_shadow_called x has_{unit}):", file=sys.stderr)
    print(f"  shadow-called: {table[0][0]}/{table[0][0]+table[0][1]} = {rate_shadow:.4f} have a ribo-seq {unit} match",
          file=sys.stderr)
    print(f"  background:    {table[1][0]}/{table[1][0]+table[1][1]} = {rate_bg:.4f} have a ribo-seq {unit} match",
          file=sys.stderr)
    print(f"  odds ratio = {odds_ratio:.3f}, Fisher's exact p = {pval:.3g}", file=sys.stderr)

    gene_rates = per_gene_rates(obs)
    print(f"\n{len(gene_rates)} genes qualify for the per-gene view "
          f"(>={MIN_OBS_PER_GENE} obs in both groups).", file=sys.stderr)
    if gene_rates:
        n_g, med_diff, _stat, wpval = paired_signed_rank_test(gene_rates)
        print(f"  paired Wilcoxon on match RATE: median diff={med_diff:+.4f}, p={wpval:.3g}",
              file=sys.stderr)
        plot_gene_metric(gene_rates, f"{outPrefix}.{mode}_match_rates.png",
                         ylabel=f"fraction of scored sites with a ribo-seq {unit} match",
                         title=f"Ribo-seq {unit} match RATE ({mode} mode): shadow-called vs. background")

    gene_depth = per_gene_depth(obs)
    if gene_depth:
        n_g, med_diff, _stat, wpval = paired_signed_rank_test(gene_depth)
        print(f"  paired Wilcoxon on mean {unit} DEPTH: median diff={med_diff:+.4f}, p={wpval:.3g}",
              file=sys.stderr)
        plot_gene_metric(gene_depth, f"{outPrefix}.{mode}_depth.png",
                         ylabel=f"mean ribo-seq {unit} depth at scored sites",
                         title=f"Ribo-seq {unit} DEPTH ({mode} mode): shadow-called vs. background")


if __name__ == "__main__":
    main(sys.argv[1:])
