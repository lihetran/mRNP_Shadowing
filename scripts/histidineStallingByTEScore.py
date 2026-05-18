'''
260508 LT
Stratifying genes by TE score. See findHighlyTranslatedRNAs.py for how I did this.
I have 4 lists of genes that correspond to quartiles of TE scores. I want to make meta histidine
editing plots for each quartile. Higher TE scores should have a stronger, maybe larger ribosome footprint.

inputs:
    - parquet of genes with quartile rank
    - bam file
    - genome fasta
    - gtf annotations
    - parquet with gene TE scores/quartile rank
    [--window 50] \
    [--min_coverage 10] \
    [--min_edit_fraction 0.01] \
    [--output comparison] \
    [--bam label] \
'''

import argparse
import sys
import re
import collections
from pathlib import Path

import pysam
import pandas as pd
import numpy as np

# ── Codon tables ──────────────────────────────────────────────────────────────
# His codons (CAU/CAC on the mRNA; on the reference these appear as CAT/CAC for
# + strand and their reverse complements for − strand).
HIS_CODONS = {"CAT", "CAC"}  # genomic (+ strand) representation

# Control codons: similar GC content / context, but not edited by ADAR
CONTROL_CODONS = {
    "Asn": {"AAT", "AAC"},   # Asn — N, pyrimidine-ending like His
    "Asp": {"GAT", "GAC"},   # Asp — D, same pattern shifted
    "Tyr": {"TAT", "TAC"},   # Tyr — Y, another A-middle codon
}


def parse_gtf_cds(gtf_path: str) -> dict:
    cds_by_chrom = collections.defaultdict(list)
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "CDS":
                continue
            chrom = fields[0]
            start = int(fields[3]) - 1
            end = int(fields[4])
            strand = fields[6]
            m = re.search(r'transcript_id "([^"]+)"', fields[8])
            tid = m.group(1) if m else "."
            m2 = re.search(r'gene_name "([^"]+)"', fields[8])
            gname = m2.group(1) if m2 else "."
            cds_by_chrom[chrom].append((start, end, strand, tid, gname))
    for chrom in cds_by_chrom:
        cds_by_chrom[chrom].sort()
    return dict(cds_by_chrom)


def find_codon_positions(ref_fasta: pysam.FastaFile,
                         cds_by_chrom: dict,
                         window: int,
                         target_codons: set,
                         codon_label: str = "codon") -> list:
    """
    General version of find_his_positions — finds any set of target codons.
    edit_pos is always the middle base (position 1) of the codon, which is
    the A in His (CAT/CAC) and equivalent position in control codons.
    """
    sites = []
    for chrom, intervals in cds_by_chrom.items():
        try:
            chrom_len = ref_fasta.get_reference_length(chrom)
        except KeyError:
            continue
        for (cds_start, cds_end, strand, tid, gname) in intervals:
            cds_seq = ref_fasta.fetch(chrom, cds_start, cds_end).upper()
            cds_len = cds_end - cds_start
            for i in range(0, cds_len - 2, 3):
                codon = cds_seq[i:i + 3]
                if strand == "-":
                    codon = reverse_complement(codon)
                if codon in target_codons:
                    codon_ref_start = cds_start + i
                    edit_pos = cds_start + i + 1  # middle base, both strands
                    win_start = max(0, edit_pos - window)
                    win_end = min(chrom_len, edit_pos + window + 1)
                    sites.append({
                        "chrom": chrom,
                        "edit_pos": edit_pos,
                        "codon_start": codon_ref_start,
                        "strand": strand,
                        "codon": codon,
                        "transcript": tid,
                        "gene_name": gname,
                        "win_start": win_start,
                        "win_end": win_end,
                        "codon_label": codon_label,
                    })

    tid_counter: dict = collections.defaultdict(int)
    for site in sites:
        tid_counter[site["transcript"]] += 1
        site["his_rank"] = tid_counter[site["transcript"]]

    seen_positions = set()
    deduped = []
    for site in sites:
        key = (site["chrom"], site["edit_pos"])
        if key not in seen_positions:
            seen_positions.add(key)
            deduped.append(site)

    print(f"  [{codon_label}] {len(deduped):,} unique sites "
          f"(deduplicated from {len(sites):,}).", file=sys.stderr)
    return deduped


def find_his_positions(ref_fasta: pysam.FastaFile,
                       cds_by_chrom: dict,
                       window: int) -> list:
    return find_codon_positions(
        ref_fasta, cds_by_chrom, window,
        target_codons=HIS_CODONS,
        codon_label="His",
    )


def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]


def complement_base(b: str) -> str:
    return b.translate(str.maketrans("ACGTacgt", "TGCAtgca"))


def count_mismatches_at_site(bam: pysam.AlignmentFile,
                             ref_fasta: pysam.FastaFile,
                             site: dict,
                             min_mapq: int = 20,
                             min_baseq: int = 10) -> dict:
    chrom = site["chrom"]
    edit_pos = site["edit_pos"]
    win_start = site["win_start"]
    win_end = site["win_end"]
    strand = site["strand"]
    pos_data = {}

    for pcolumn in bam.pileup(
            chrom, win_start, win_end,
            truncate=True,
            min_mapping_quality=min_mapq,
            min_base_quality=min_baseq,
            stepper="samtools",
            ignore_overlaps=False,
    ):
        ref_pos = pcolumn.reference_pos
        if ref_pos < win_start or ref_pos >= win_end:
            continue
        ref_base = ref_fasta.fetch(chrom, ref_pos, ref_pos + 1).upper()
        rel_pos = ref_pos - edit_pos
        if strand == "-":
            rel_pos = -rel_pos

        counts = collections.Counter()
        for pread in pcolumn.pileups:
            if pread.is_del or pread.is_refskip:
                continue
            qbase_raw = pread.alignment.query_sequence[pread.query_position].upper()
            # Convert to transcript coordinates using XOR of read strand and gene strand:
            # needs_complement when read orientation doesn't match gene orientation
            needs_complement = pread.alignment.is_reverse != (strand == "-")
            qbase = complement_base(qbase_raw) if needs_complement else qbase_raw
            counts[qbase] += 1

        total = sum(counts.values())
        pos_data[rel_pos] = {
            "ref_pos": ref_pos,
            "ref_base": ref_base if strand == "+" else complement_base(ref_base),
            "A": counts.get("A", 0),
            "G": counts.get("G", 0),
            "C": counts.get("C", 0),
            "T": counts.get("T", 0),
            "cov": total,
        }
    return pos_data


def aggregate_sites(sites: list,
                    bam: pysam.AlignmentFile,
                    ref_fasta: pysam.FastaFile,
                    min_coverage: int,
                    min_mapq: int,
                    min_baseq: int,
                    label: str) -> pd.DataFrame:
    records = []
    for i, site in enumerate(sites):
        if (i + 1) % 100 == 0:
            print(f"  [{label}] Processing site {i + 1}/{len(sites)}…", file=sys.stderr)

        pos_data = count_mismatches_at_site(
            bam, ref_fasta, site,
            min_mapq=min_mapq, min_baseq=min_baseq
        )
        for rel_pos, counts in pos_data.items():
            if counts["cov"] < min_coverage:
                continue
            ag_denom = counts["A"] + counts["G"]
            ag_edit = counts["G"] / ag_denom \
                if counts["ref_base"] == "A" and ag_denom > 0 else np.nan
            records.append({
                "site_id": f"{site['chrom']}:{site['edit_pos']}",
                "transcript": site["transcript"],
                "gene_name": site["gene_name"],
                "his_rank": site["his_rank"],
                "chrom": site["chrom"],
                "edit_pos": site["edit_pos"],
                "strand": site["strand"],
                "codon": site["codon"],
                "rel_pos": rel_pos,
                "ref_base": counts["ref_base"],
                "in_his_codon": rel_pos in (-1, 0, 1),
                "A": counts["A"],
                "G": counts["G"],
                "C": counts["C"],
                "T": counts["T"],
                "coverage": counts["cov"],
                "ag_edit_frac": ag_edit,
                "is_his_A": rel_pos == 0 and counts["ref_base"] == "A",
            })
    return pd.DataFrame(records)


def transcript_normalised_agg(df: pd.DataFrame,
                              group_cols: list = None,
                              pseudo: float = 1e-3) -> pd.DataFrame:
    """
    Two-stage aggregation restricted to ref=A positions only, using
    ag_edit_frac = G/(A+G) + pseudocount as the editing metric:
      1. Average (ag_edit_frac + pseudo) across all His sites within a
         transcript at each rel_pos
      2. Average those transcript means across transcripts at each rel_pos

    A pseudocount is added so that sites with zero editing still contribute
    rather than being dropped, keeping the meta-plot anchored to all sites.

    Non-A ref positions and codon flanks are excluded (see in_his_codon filter).
    group_cols: additional columns to group by before rel_pos (e.g. ["his_rank"])
    Returns DataFrame with rel_pos (+ group_cols), mean_edit_frac, sem_edit_frac, n_transcripts.
    """
    if group_cols is None:
        group_cols = []

    ref_a = df[
        df["ag_edit_frac"].notna() &
        (df["ref_base"] == "A") &
        (~df["in_his_codon"] | df["is_his_A"])
        ].copy()
    ref_a["ag_edit_frac_ps"] = ref_a["ag_edit_frac"] + pseudo

    # Stage 1: per-transcript mean at each rel_pos
    tx_mean = (
        ref_a.groupby(group_cols + ["transcript", "rel_pos"])["ag_edit_frac_ps"]
        .mean()
        .reset_index()
        .rename(columns={"ag_edit_frac_ps": "tx_mean_edit_frac"})
    )

    # Stage 2: grand mean across transcripts
    agg = (
        tx_mean.groupby(group_cols + ["rel_pos"])
        .agg(
            mean_edit_frac=("tx_mean_edit_frac", "mean"),
            sem_edit_frac=("tx_mean_edit_frac", lambda x: x.sem()),
            n_transcripts=("transcript", "nunique"),
        )
        .reset_index()
    )
    return agg


def collect_his_site_reads(sites: list,
                           bam: pysam.AlignmentFile,
                           min_mapq: int = 20) -> set:
    """
    Collect the set of query names of all reads that span at least one
    His site window. These are the reads that contributed to the meta plots.
    """
    read_names = set()
    for site in sites:
        for read in bam.fetch(site["chrom"], site["win_start"], site["win_end"]):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality < min_mapq:
                continue
            read_names.add(read.query_name)
    return read_names


def _pyx_meta_graph(c, xpos, ypos, datasets, window,
                    y_title="Edit Frac",
                    x_title="Relative Position",
                    share_xaxis=None,
                    panel_w=5, panel_h=3):
    """
    Insert one meta-analysis panel into canvas c.
    datasets: list of (rel_agg DataFrame, pyx color, linestyle, label) tuples.
              All datasets are plotted before the graph is inserted so
              overlaying multiple lines on the same axes works correctly.
    Returns the graph object for x-axis linking.
    """
    from pyx import graph, color, style

    # Compute y_max across all datasets
    y_max = 0.02
    for item in datasets:
        rel_agg = item[0]
        if rel_agg.empty:
            continue
        frac = rel_agg["mean_edit_frac"].values
        sem = rel_agg["sem_edit_frac"].values
        candidate = float(np.nanmax(frac + sem)) * 1.15 if len(frac) > 0 else 0
        y_max = max(y_max, candidate)

    x_axis = graph.axis.linear(min=-window, max=window,
                               title=x_title) \
        if share_xaxis is None \
        else graph.axis.linkedaxis(share_xaxis.axes["x"])

    g = graph.graphxy(
        width=panel_w, height=panel_h,
        xpos=xpos, ypos=ypos,
        x=x_axis,
        y=graph.axis.linear(min=0, max=y_max, title=y_title),
    )

    # Codon span and His A line — drawn first so data sits on top
    g.plot(graph.data.function("x(y)=-1", min=0, max=y_max),
           [graph.style.line([color.gray(0.8), style.linewidth.thin])])
    g.plot(graph.data.function("x(y)=1", min=0, max=y_max),
           [graph.style.line([color.gray(0.8), style.linewidth.thin])])
    g.plot(graph.data.function("x(y)=0", min=0, max=y_max),
           [graph.style.line([color.cmyk(0, 1, 1, 0),
                              style.linewidth.thick,
                              style.linestyle.dashed])])

    # Plot each dataset
    for item in datasets:
        rel_agg, col, ls = item[0], item[1], item[2]
        if rel_agg.empty:
            continue
        pos = rel_agg["rel_pos"].values
        frac = rel_agg["mean_edit_frac"].values
        sem = rel_agg["sem_edit_frac"].values

        # SEM dotted bounds
        for pts in [list(zip(pos.tolist(), (frac - sem).tolist())),
                    list(zip(pos.tolist(), (frac + sem).tolist()))]:
            g.plot(graph.data.points(pts, x=1, y=2),
                   [graph.style.line([col, style.linewidth.thin,
                                      style.linestyle.dotted])])
        # Mean line
        g.plot(graph.data.points(list(zip(pos.tolist(), frac.tolist())), x=1, y=2),
               [graph.style.line([col, style.linewidth.normal, ls])])

    c.insert(g)
    return g


def plot_codon_specificity_pyx(his_agg_bam1: pd.DataFrame,
                               his_agg_bam2: pd.DataFrame,
                               control_aggs: dict,
                               label1: str,
                               label2: str,
                               output_prefix: str,
                               window: int):
    """
    Pyx version of the codon specificity figure.
    Layout: 2 rows (BAM1 top, BAM2 bottom) × (n_codons + 1) columns.
    Last column in each row overlays all codons.
    Each codon gets a distinct CMYK colour.
    """
    from pyx import canvas, color, style, text as pyx_text

    codon_names = ["His"] + list(control_aggs.keys())

    # Distinct CMYK colours for each codon
    codon_colors_cmyk = [
        color.cmyk(0, 1, 1, 0),  # red   — His
        color.cmyk(1, 0.5, 0, 0),  # blue  — control 1
        color.cmyk(0.1, 0.05, 0.9, 0),  # green — control 2
        color.cmyk(0.97, 0, 0.75, 0),  # teal  — control 3
        color.cmyk(0, 0.6, 0.3, 0),  # pink  — control 4
    ]
    codon_color_map = {name: codon_colors_cmyk[i]
                       for i, name in enumerate(codon_names)}

    panel_w = 3
    panel_h = 2.5
    gap = 2.5  # wide enough for y-axis labels
    row_gap = 2.0

    c = canvas.canvas()

    for row_idx, (bam_label, his_agg) in enumerate([
        (label1, his_agg_bam1),
        (label2, his_agg_bam2),
    ]):
        ypos = (1 - row_idx) * (panel_h + row_gap)
        # Only bottom row (row_idx==1) gets x-axis label
        x_title = "Relative Position" if row_idx == 1 else ""

        for col_idx, codon_name in enumerate(codon_names):
            xpos = col_idx * (panel_w + gap)
            agg = his_agg if codon_name == "His" \
                else control_aggs[codon_name].get(bam_label, pd.DataFrame())
            col = codon_color_map[codon_name]
            y_title = f"{bam_label} edit frac" if col_idx == 0 else ""

            g = _pyx_meta_graph(
                c, xpos=xpos, ypos=ypos,
                datasets=[(agg, col, style.linestyle.solid)],
                y_title=y_title,
                x_title=x_title,
                window=window, panel_w=panel_w, panel_h=panel_h,
            )
            c.text(g.xpos + g.width / 2., g.ypos + g.height + 0.3,
                   codon_name,
                   [pyx_text.halign.center, pyx_text.size.small])

        overlay_datasets = []
        for codon_name in codon_names:
            agg = his_agg if codon_name == "His" \
                else control_aggs[codon_name].get(bam_label, pd.DataFrame())
            ls = style.linestyle.solid if codon_name == "His" \
                else style.linestyle.dashed
            overlay_datasets.append((agg, codon_color_map[codon_name], ls))

        xpos_overlay = len(codon_names) * (panel_w + gap)
        g_ov = _pyx_meta_graph(
            c, xpos=xpos_overlay, ypos=ypos,
            datasets=overlay_datasets,
            y_title="",
            x_title=x_title,
            window=window, panel_w=panel_w, panel_h=panel_h,
        )
        c.text(g_ov.xpos + g_ov.width / 2., g_ov.ypos + g_ov.height + 0.3,
               "Overlay",
               [pyx_text.halign.center, pyx_text.size.small])

    plot_path = f"{output_prefix}_codon_specificity_pyx"
    c.writePDFfile(plot_path)
    print(f"  Saved pyx codon specificity plots → {plot_path}.pdf",
          file=sys.stderr)


def plot_codon_specificity_overlay_pyx(his_agg_bam1: pd.DataFrame,
                                       his_agg_bam2: pd.DataFrame,
                                       control_aggs: dict,
                                       label1: str,
                                       label2: str,
                                       output_prefix: str,
                                       window: int):
    """
    Pyx overlay-only figure: one panel per BAM (stacked), all codons overlaid,
    with a manually drawn legend showing a short line segment + codon name
    for each codon.
    """
    from pyx import canvas, color, style, path, text as pyx_text

    codon_names = ["His"] + list(control_aggs.keys())

    codon_colors_cmyk = [
        color.cmyk(0, 1, 1, 0),
        color.cmyk(1, 0.5, 0, 0),
        color.cmyk(0.1, 0.05, 0.9, 0),
        color.cmyk(0.97, 0, 0.75, 0),
        color.cmyk(0, 0.6, 0.3, 0),
    ]
    codon_color_map = {name: codon_colors_cmyk[i]
                       for i, name in enumerate(codon_names)}

    panel_w = 7
    panel_h = 3.5
    gap = 1.5  # vertical gap between the two panels
    leg_lw = 0.8  # legend line length in cm
    leg_dy = 0.55  # vertical spacing between legend entries

    c = canvas.canvas()

    for row_idx, (bam_label, his_agg) in enumerate([
        (label1, his_agg_bam1),
        (label2, his_agg_bam2),
    ]):
        ypos = (1 - row_idx) * (panel_h + gap)
        x_title = "Relative Position" if row_idx == 1 else ""

        overlay_datasets = []
        for codon_name in codon_names:
            agg = his_agg if codon_name == "His" \
                else control_aggs[codon_name].get(bam_label, pd.DataFrame())
            ls = style.linestyle.solid if codon_name == "His" \
                else style.linestyle.dashed
            overlay_datasets.append((agg, codon_color_map[codon_name], ls))

        g = _pyx_meta_graph(
            c, xpos=0, ypos=ypos,
            datasets=overlay_datasets,
            y_title="Edit Frac",
            x_title=x_title,
            window=window, panel_w=panel_w, panel_h=panel_h,
        )

        c.text(g.xpos + g.width / 2., g.ypos + g.height + 0.4, bam_label,
               [pyx_text.halign.center, pyx_text.size.normalsize])

        # Manual legend — only draw on the top panel to avoid duplication
        if row_idx == 0:
            leg_y_start = g.ypos + g.height - 0.2
            leg_x_start = g.xpos + g.width + 0.4
            for j, codon_name in enumerate(codon_names):
                col = codon_color_map[codon_name]
                ls = style.linestyle.solid if codon_name == "His" \
                    else style.linestyle.dashed
                ly = leg_y_start - j * leg_dy

                c.stroke(
                    path.line(leg_x_start, ly, leg_x_start + leg_lw, ly),
                    [col, style.linewidth.normal, ls]
                )
                c.text(leg_x_start + leg_lw + 0.15, ly, codon_name,
                       [pyx_text.valign.middle, pyx_text.size.small])

    plot_path = f"{output_prefix}_codon_overlay_pyx"
    c.writePDFfile(plot_path)
    print(f"  Saved pyx codon overlay plots → {plot_path}.pdf", file=sys.stderr)


# ── TE quartile meta-plot ─────────────────────────────────────────────────────

def plot_te_quartiles_pyx(bam_agg: dict,
                          output_prefix: str,
                          window: int,
                          quartile_col: str = "TE_quartile"):
    """
    For each BAM, produce one panel per TE quartile (Q1–Q4) showing the His
    A→G editing meta-plot.  All four quartile panels are overlaid in a final
    summary panel.  Layout: 2 rows (one per BAM) × 5 panels (Q1–Q4 + overlay).

    Colour ramp: light → dark blue for Q1 → Q4 (higher TE = more intense).
    """
    from pyx import canvas, color, style, path, text as pyx_text

    quartile_labels = [1, 2, 3, 4]
    # Light→dark CMYK blue ramp
    quartile_colors = [
        color.cmyk(0.6, 0.3, 0.0, 0.0),   # Q1 — light blue
        color.cmyk(0.8, 0.4, 0.0, 0.0),   # Q2
        color.cmyk(1.0, 0.5, 0.0, 0.1),   # Q3
        color.cmyk(1.0, 0.6, 0.0, 0.3),   # Q4 — dark blue
    ]
    q_color_map = dict(zip(quartile_labels, quartile_colors))

    panel_w = 3.5
    panel_h = 2.8
    h_gap = 2.2   # horizontal gap (room for y-axis label)
    row_gap = 2.0

    labels = list(bam_agg.keys())
    c = canvas.canvas()

    for row_idx, bam_label in enumerate(labels):
        df = bam_agg[bam_label]
        ypos = (len(labels) - 1 - row_idx) * (panel_h + row_gap)
        x_title = "Relative Position" if row_idx == len(labels) - 1 else ""

        overlay_datasets = []

        for col_idx, q in enumerate(quartile_labels):
            xpos = col_idx * (panel_w + h_gap)
            q_df = df[df[quartile_col] == q] if quartile_col in df.columns else pd.DataFrame()

            if q_df.empty:
                q_agg = pd.DataFrame(columns=["rel_pos", "mean_edit_frac",
                                               "sem_edit_frac", "n_transcripts"])
            else:
                q_agg = transcript_normalised_agg(q_df)

            col = q_color_map[q]
            y_title = f"{bam_label}\nedit frac" if col_idx == 0 else ""

            g = _pyx_meta_graph(
                c, xpos=xpos, ypos=ypos,
                datasets=[(q_agg, col, style.linestyle.solid)],
                y_title=y_title,
                x_title=x_title,
                window=window, panel_w=panel_w, panel_h=panel_h,
            )
            c.text(g.xpos + g.width / 2., g.ypos + g.height + 0.3,
                   f"Q{q} (n={q_agg['n_transcripts'].max() if not q_agg.empty else 0})",
                   [pyx_text.halign.center, pyx_text.size.small])

            overlay_datasets.append((q_agg, col, style.linestyle.solid))

        # Overlay panel — all quartiles on the same axes
        xpos_ov = len(quartile_labels) * (panel_w + h_gap)
        g_ov = _pyx_meta_graph(
            c, xpos=xpos_ov, ypos=ypos,
            datasets=overlay_datasets,
            y_title="",
            x_title=x_title,
            window=window, panel_w=panel_w, panel_h=panel_h,
        )
        c.text(g_ov.xpos + g_ov.width / 2., g_ov.ypos + g_ov.height + 0.3,
               "All quartiles",
               [pyx_text.halign.center, pyx_text.size.small])

        # Legend (top row only, to the right of overlay panel)
        if row_idx == 0:
            leg_x = g_ov.xpos + g_ov.width + 0.4
            leg_y = g_ov.ypos + g_ov.height - 0.2
            leg_lw = 0.8
            leg_dy = 0.55
            for j, q in enumerate(quartile_labels):
                col = q_color_map[q]
                ly = leg_y - j * leg_dy
                from pyx import path as pyx_path
                c.stroke(
                    pyx_path.line(leg_x, ly, leg_x + leg_lw, ly),
                    [col, style.linewidth.normal, style.linestyle.solid]
                )
                c.text(leg_x + leg_lw + 0.15, ly, f"Q{q}",
                       [pyx_text.valign.middle, pyx_text.size.small])

    plot_path = f"{output_prefix}_TE_quartiles_pyx"
    c.writePDFfile(plot_path)
    print(f"  Saved TE quartile plots → {plot_path}.pdf", file=sys.stderr)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare His A→G editing between two nanopore BAM files."
    )
    p.add_argument("--bam1", required=True, help="BAM file for condition 1")
    p.add_argument("--bam2", required=True, help="BAM file for condition 2")
    p.add_argument("--label1", default="BAM1", help="Label for condition 1 (default: BAM1)")
    p.add_argument("--label2", default="BAM2", help="Label for condition 2 (default: BAM2)")
    p.add_argument("--ref", required=True, help="Reference FASTA (indexed)")
    p.add_argument("--gtf", required=True, help="GTF annotation file")
    p.add_argument("--window", type=int, default=50,
                   help="Nucleotides either side of His A (default: 50)")
    p.add_argument("--min_coverage", type=int, default=10,
                   help="Minimum read depth per position (default: 10)")
    p.add_argument("--min_edit_fraction", type=float, default=0.01,
                   help="Min A→G fraction to call a site edited (default: 0.01)")
    p.add_argument("--min_mapq", type=int, default=20)
    p.add_argument("--min_baseq", type=int, default=10)
    p.add_argument("--output", default="his_edit_comparison",
                   help="Output file prefix (default: his_edit_comparison)")
    p.add_argument("--gene_parquet", default=None,
                   help="Parquet with 'gene_name' and 'quartile_rank' columns. "
                        "If provided, genes are stratified into TE quartiles for meta plots.")
    p.add_argument("--control_codons", nargs="*", default=None,
                   help="Control codon names to run specificity analysis "
                        f"(default: {list(CONTROL_CODONS.keys())}). "
                        "Must be keys of the CONTROL_CODONS dict in the script, "
                        "or pass none to skip.")
    p.add_argument("--chroms", nargs="*", default=None,
                   help="Restrict to specific chromosomes/contigs")
    return p.parse_args()


def main():
    args = parse_args()
    out = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    # ── Shared reference data (only parsed once) ──────────────────────────────
    print("Opening reference and parsing GTF…", file=sys.stderr)
    ref_fasta = pysam.FastaFile(args.ref)
    cds_by_chrom = parse_gtf_cds(args.gtf)

    # Filter chromosomes if requested
    if args.chroms:
        cds_by_chrom = {k: v for k, v in cds_by_chrom.items() if k in args.chroms}
        print(f"  Restricted GTF to {len(cds_by_chrom)} chroms: {args.chroms}",
              file=sys.stderr)

    # ── TE quartile map ───────────────────────────────────────────────────────
    if args.gene_parquet:
        TE_df = pd.read_parquet(args.gene_parquet)
        gene_to_quartile = dict(zip(TE_df["gene_name"], TE_df["quartile_rank"]))
        print(f"  Loaded TE quartile map for {len(gene_to_quartile):,} genes.",
              file=sys.stderr)
    else:
        gene_to_quartile = None

    # ── Find His sites once (same coordinates for both BAMs) ─────────────────
    print("Finding His codon positions…", file=sys.stderr)
    his_sites = find_his_positions(ref_fasta, cds_by_chrom, args.window)

    # ── Process each BAM ─────────────────────────────────────────────────────
    bam_agg: dict[str, pd.DataFrame] = {}

    for bam_path, label in [(args.bam1, args.label1), (args.bam2, args.label2)]:
        print(f"\nProcessing BAM: {bam_path} ({label})", file=sys.stderr)
        bam = pysam.AlignmentFile(bam_path, "rb")

        agg_df = aggregate_sites(
            his_sites, bam, ref_fasta,
            min_coverage=args.min_coverage,
            min_mapq=args.min_mapq,
            min_baseq=args.min_baseq,
            label=label,
        )

        # Attach TE quartile info if available
        if gene_to_quartile is not None:
            agg_df["TE_quartile"] = agg_df["gene_name"].map(gene_to_quartile)

        bam_agg[label] = agg_df

        # Save per-BAM raw aggregate
        parquet_path = f"{out}_{label}_raw_agg.parquet"
        agg_df.to_parquet(parquet_path, index=False)
        print(f"  Saved raw aggregate → {parquet_path}", file=sys.stderr)

        bam.close()

    # ── His meta-plot aggregates (overall, both BAMs) ─────────────────────────
    print("\nComputing His meta-plot aggregates…", file=sys.stderr)
    his_agg: dict[str, pd.DataFrame] = {
        label: transcript_normalised_agg(df)
        for label, df in bam_agg.items()
    }

    # ── Control codon specificity analysis ────────────────────────────────────
    control_names = args.control_codons \
        if args.control_codons is not None \
        else list(CONTROL_CODONS.keys())

    # control_aggs[codon_name][bam_label] = aggregated DataFrame
    control_aggs: dict[str, dict[str, pd.DataFrame]] = {}

    if control_names:
        print("\nRunning control codon specificity analysis…", file=sys.stderr)
        for codon_name in control_names:
            if codon_name not in CONTROL_CODONS:
                print(f"  WARNING: '{codon_name}' not in CONTROL_CODONS — skipping.",
                      file=sys.stderr)
                continue
            codon_set = CONTROL_CODONS[codon_name]
            print(f"  Codon: {codon_name} {codon_set}", file=sys.stderr)
            ctrl_sites = find_codon_positions(
                ref_fasta, cds_by_chrom, args.window,
                target_codons=codon_set,
                codon_label=codon_name,
            )
            control_aggs[codon_name] = {}
            for bam_path, label in [(args.bam1, args.label1), (args.bam2, args.label2)]:
                bam = pysam.AlignmentFile(bam_path, "rb")
                ctrl_df = aggregate_sites(
                    ctrl_sites, bam, ref_fasta,
                    min_coverage=args.min_coverage,
                    min_mapq=args.min_mapq,
                    min_baseq=args.min_baseq,
                    label=f"{label}_{codon_name}",
                )
                control_aggs[codon_name][label] = transcript_normalised_agg(ctrl_df)
                bam.close()

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots…", file=sys.stderr)

    # 1. Codon specificity (grid + overlay) — only if controls were run
    if control_aggs:
        plot_codon_specificity_pyx(
            his_agg_bam1=his_agg[args.label1],
            his_agg_bam2=his_agg[args.label2],
            control_aggs=control_aggs,
            label1=args.label1,
            label2=args.label2,
            output_prefix=out,
            window=args.window,
        )
        plot_codon_specificity_overlay_pyx(
            his_agg_bam1=his_agg[args.label1],
            his_agg_bam2=his_agg[args.label2],
            control_aggs=control_aggs,
            label1=args.label1,
            label2=args.label2,
            output_prefix=out,
            window=args.window,
        )

    # 2. TE quartile meta-plots — only if gene_parquet was provided
    if gene_to_quartile is not None:
        plot_te_quartiles_pyx(
            bam_agg=bam_agg,
            output_prefix=out,
            window=args.window,
        )
    else:
        print("  No gene_parquet provided — skipping TE quartile plots.", file=sys.stderr)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()