#!/usr/bin/env python3
"""
Meta ORF A->G Editing Plot
===========================
Computes transcript-normalised mean A->G editing fraction at each position
relative to the start codon and stop codon across all protein-coding genes
with sufficient coverage. Produces a broken-axis figure with two panels:
  Left:  window around start codon (ref=A positions only)
  Right: window around stop codon  (ref=A positions only)
Connected by a fixed break representing the variable gene body.

One figure per BAM, saved as separate PDFs using pyx.

Usage:
    python3 metaORF.py \
        --bam1 condition1.bam --label1 "WT" \
        --bam2 condition2.bam --label2 "3-AT" \
        --ref   reference.fa \
        --gtf   annotation.gtf \
        --output output_prefix \
        [--window 50] \
        [--min_coverage 20] \
        [--min_mapq 20] \
        [--min_baseq 10] \
        [--gene_list genes.txt]

Requirements:
    pip install pysam pandas numpy
    pyx (for plotting)
"""

import argparse
import sys
import re
import collections
from pathlib import Path

import pysam
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. GTF parsing — collect per-transcript CDS start and stop positions
# ─────────────────────────────────────────────────────────────────────────────

def parse_gtf_transcripts(gtf_path: str) -> list:
    """
    Parse CDS features and compute, per transcript:
        - start codon genomic position (first base of CDS in transcript order)
        - stop codon genomic position  (last base of CDS in transcript order)
        - strand, chrom, gene_name, transcript_id

    For plus-strand genes:
        cds_start = min CDS start across all CDS intervals
        cds_stop  = max CDS end   across all CDS intervals (first base past CDS)

    For minus-strand genes:
        cds_start = max CDS end   (transcript 5'-end is genomic right)
        cds_stop  = min CDS start (transcript 3'-end is genomic left)

    Returns a list of dicts, one per transcript.
    """
    # Collect CDS intervals per transcript
    tx_cds  = collections.defaultdict(list)
    tx_meta = {}

    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "CDS":
                continue
            chrom  = fields[0]
            start  = int(fields[3]) - 1   # 0-based
            end    = int(fields[4])
            strand = fields[6]
            m_tid  = re.search(r'transcript_id "([^"]+)"', fields[8])
            m_gn   = re.search(r'gene_name "([^"]+)"', fields[8])
            tid    = m_tid.group(1) if m_tid else "."
            gname  = m_gn.group(1)  if m_gn  else "."

            tx_cds[tid].append((start, end))
            if tid not in tx_meta:
                tx_meta[tid] = {
                    "chrom": chrom, "strand": strand,
                    "transcript": tid, "gene_name": gname,
                }

    transcripts = []
    for tid, intervals in tx_cds.items():
        meta   = tx_meta[tid]
        strand = meta["strand"]
        chrom  = meta["chrom"]
        chrom_len_approx = max(e for _, e in intervals) + 1

        all_starts = [s for s, e in intervals]
        all_ends   = [e for s, e in intervals]

        if strand == "+":
            # Start codon: first base of first CDS interval
            start_pos = min(all_starts)
            # Stop codon:  first base past last CDS = last CDS end
            stop_pos  = max(all_ends) - 1   # last included base
        else:
            # Start codon (5' end on minus strand): rightmost genomic end - 1
            start_pos = max(all_ends) - 1
            # Stop codon:  leftmost genomic start
            stop_pos  = min(all_starts)

        transcripts.append({
            "transcript": tid,
            "gene_name":  meta["gene_name"],
            "chrom":      chrom,
            "strand":     strand,
            "start_pos":  start_pos,   # genomic coord of first CDS base
            "stop_pos":   stop_pos,    # genomic coord of last CDS base
            "cds_len":    sum(e - s for s, e in intervals),
        })

    print(f"  Parsed {len(transcripts):,} transcripts with CDS.", file=sys.stderr)
    return transcripts


# ─────────────────────────────────────────────────────────────────────────────
# 2. Helpers
# ─────────────────────────────────────────────────────────────────────────────

def complement_base(b: str) -> str:
    return b.translate(str.maketrans("ACGTacgt", "TGCAtgca"))


def mean_coverage_around(bam: pysam.AlignmentFile,
                          chrom: str, pos: int, window: int,
                          min_mapq: int) -> float:
    """Mean pileup depth in [pos-window, pos+window]."""
    total, bases = 0, 0
    win_start = max(0, pos - window)
    win_end   = pos + window + 1
    for col in bam.pileup(chrom, win_start, win_end,
                           truncate=True,
                           min_mapping_quality=min_mapq,
                           stepper="samtools"):
        total += col.nsegments
        bases += 1
    return total / bases if bases > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pileup around one anchor position (start or stop codon)
# ─────────────────────────────────────────────────────────────────────────────

def pileup_around_anchor(bam: pysam.AlignmentFile,
                          ref_fasta: pysam.FastaFile,
                          transcript: dict,
                          anchor_gpos: int,
                          window: int,
                          min_mapq: int,
                          min_baseq: int,
                          min_coverage: int) -> dict:
    """
    Pileup ±window around anchor_gpos for one transcript.
    Returns {rel_pos: ag_edit_frac} for ref=A positions that pass coverage.
    rel_pos is in transcript coordinates (negative = upstream of anchor).
    """
    chrom  = transcript["chrom"]
    strand = transcript["strand"]

    win_start = max(0, anchor_gpos - window)
    win_end   = anchor_gpos + window + 1

    result = {}

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

        ref_base_genomic = ref_fasta.fetch(chrom, ref_pos, ref_pos + 1).upper()
        ref_base_tx = complement_base(ref_base_genomic) \
                      if strand == "-" else ref_base_genomic

        if ref_base_tx != "A":
            continue

        counts = collections.Counter()
        for pread in pcolumn.pileups:
            if pread.is_del or pread.is_refskip:
                continue
            qbase_raw = pread.alignment.query_sequence[
                pread.query_position].upper()
            needs_complement = (pread.alignment.is_reverse != (strand == "-"))
            qbase = complement_base(qbase_raw) if needs_complement else qbase_raw
            counts[qbase] += 1

        ag_total = counts["A"] + counts["G"]
        if ag_total < min_coverage:
            continue

        rel = ref_pos - anchor_gpos
        if strand == "-":
            rel = -rel

        result[rel] = counts["G"] / ag_total

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Transcript-normalised aggregation (same two-stage approach as main script)
# ─────────────────────────────────────────────────────────────────────────────

def transcript_normalised_agg(records: list) -> pd.DataFrame:
    """
    records: list of {"transcript": str, "rel_pos": int, "ag_edit_frac": float}

    Stage 1: mean ag_edit_frac per (transcript, rel_pos)
    Stage 2: mean + SEM of those transcript means across transcripts at each rel_pos

    Returns DataFrame(rel_pos, mean_edit_frac, sem_edit_frac, n_transcripts).
    """
    if not records:
        return pd.DataFrame(columns=["rel_pos", "mean_edit_frac",
                                     "sem_edit_frac", "n_transcripts"])

    df = pd.DataFrame(records)

    # Stage 1
    tx_mean = (
        df.groupby(["transcript", "rel_pos"])["ag_edit_frac"]
          .mean()
          .reset_index()
          .rename(columns={"ag_edit_frac": "tx_mean"})
    )

    # Stage 2
    agg = (
        tx_mean.groupby("rel_pos")
               .agg(
                   mean_edit_frac=("tx_mean", "mean"),
                   sem_edit_frac =("tx_mean", lambda x: x.sem()),
                   n_transcripts =("tx_mean", "count"),
               )
               .reset_index()
    )
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 5. Collect pileup data for all transcripts around start and stop codons
# ─────────────────────────────────────────────────────────────────────────────

def collect_orf_data(bam_path: str,
                      ref_fasta: pysam.FastaFile,
                      transcripts: list,
                      window: int,
                      min_coverage: int,
                      min_mapq: int,
                      min_baseq: int,
                      label: str) -> tuple:
    """
    For each transcript, pileup ±window around the start and stop codons.
    Returns (start_agg, stop_agg) DataFrames of transcript-normalised
    editing fractions.
    """
    start_records = []
    stop_records  = []

    bam = pysam.AlignmentFile(bam_path, "rb")

    for i, tx in enumerate(transcripts):
        if (i + 1) % 200 == 0:
            print(f"  [{label}] {i+1}/{len(transcripts)}…", file=sys.stderr)

        chrom = tx["chrom"]
        try:
            ref_fasta.fetch(chrom, 0, 1)
        except (KeyError, ValueError):
            continue

        # Start codon window
        start_data = pileup_around_anchor(
            bam, ref_fasta, tx, tx["start_pos"],
            window, min_mapq, min_baseq, min_coverage,
        )
        for rel, frac in start_data.items():
            start_records.append({
                "transcript":   tx["transcript"],
                "rel_pos":      rel,
                "ag_edit_frac": frac,
            })

        # Stop codon window
        stop_data = pileup_around_anchor(
            bam, ref_fasta, tx, tx["stop_pos"],
            window, min_mapq, min_baseq, min_coverage,
        )
        for rel, frac in stop_data.items():
            stop_records.append({
                "transcript":   tx["transcript"],
                "rel_pos":      rel,
                "ag_edit_frac": frac,
            })

    bam.close()

    start_agg = transcript_normalised_agg(start_records)
    stop_agg  = transcript_normalised_agg(stop_records)

    n_start_tx = start_agg["n_transcripts"].max() if not start_agg.empty else 0
    n_stop_tx  = stop_agg["n_transcripts"].max()  if not stop_agg.empty else 0
    print(f"  [{label}] Start codon: {int(n_start_tx):,} transcripts contributed.",
          file=sys.stderr)
    print(f"  [{label}] Stop codon:  {int(n_stop_tx):,} transcripts contributed.",
          file=sys.stderr)

    return start_agg, stop_agg


# ─────────────────────────────────────────────────────────────────────────────
# 6. Plot: broken-axis figure using pyx
# ─────────────────────────────────────────────────────────────────────────────

def plot_meta_orf_pyx(start_agg: pd.DataFrame,
                       stop_agg: pd.DataFrame,
                       label: str,
                       window: int,
                       output_path: str):
    """
    Two-panel broken-axis pyx figure:
      Left panel:  mean A->G edit frac ±window around start codon
      Right panel: mean A->G edit frac ±window around stop codon
    Connected by a fixed break gap with diagonal slash marks.
    SEM shown as dotted bounds above/below the mean line.
    """
    from pyx import canvas, graph, color, style, path, text as pyx_text

    col_line = color.cmyk(0, 0, 0, 1)      # black
    col_sem  = color.cmyk(0, 0, 0, 0.4)    # grey for dotted SEM bounds
    col_anch = color.cmyk(0, 1, 1, 0)      # red vertical line at anchor

    panel_w  = 7.0    # cm per panel
    panel_h  = 4.0    # cm
    break_w  = 1.5    # cm for the gap between panels
    slash_hw = 0.25   # half-width of diagonal slash marks in cm

    # Compute shared y range across both panels
    all_fracs = []
    for agg in [start_agg, stop_agg]:
        if not agg.empty:
            all_fracs.extend(
                (agg["mean_edit_frac"] + agg["sem_edit_frac"]).tolist()
            )
    y_max = max(all_fracs) * 1.15 if all_fracs else 0.1
    y_max = max(y_max, 0.01)

    c = canvas.canvas()

    def _make_panel(xpos, agg, x_title, anchor_label):
        if agg.empty:
            return None

        g = graph.graphxy(
            width=panel_w, height=panel_h,
            xpos=xpos, ypos=0,
            x=graph.axis.linear(min=-window, max=window, title=x_title),
            y=graph.axis.linear(min=0, max=y_max, title="Edit Frac"),
        )

        pos  = agg["rel_pos"].values
        frac = agg["mean_edit_frac"].values
        sem  = agg["sem_edit_frac"].values

        # Anchor vertical line
        g.plot(graph.data.function(f"x(y)=0", min=0, max=y_max),
               [graph.style.line([col_anch, style.linewidth.normal,
                                  style.linestyle.dashed])])

        # SEM dotted bounds
        for pts in [list(zip(pos.tolist(), (frac - sem).tolist())),
                    list(zip(pos.tolist(), (frac + sem).tolist()))]:
            g.plot(graph.data.points(pts, x=1, y=2),
                   [graph.style.line([col_sem, style.linewidth.thin,
                                      style.linestyle.dotted])])

        # Mean line
        g.plot(graph.data.points(list(zip(pos.tolist(), frac.tolist())),
                                  x=1, y=2),
               [graph.style.line([col_line, style.linewidth.normal,
                                  style.linestyle.solid])])

        c.insert(g)
        return g

    g_start = _make_panel(xpos=0,
                           agg=start_agg,
                           x_title="Position relative to start codon (nt)",
                           anchor_label="Start")
    g_stop  = _make_panel(xpos=panel_w + break_w,
                           agg=stop_agg,
                           x_title="Position relative to stop codon (nt)",
                           anchor_label="Stop")

    # ── Broken axis visual: diagonal slashes at the break edges ───────────────
    # Left side of break (right edge of left panel)
    break_x_left  = panel_w
    break_x_right = panel_w + break_w

    for bx in [break_x_left, break_x_right]:
        for by in [panel_h * 0.35, panel_h * 0.65]:
            c.stroke(path.line(bx - slash_hw, by - slash_hw * 1.5,
                               bx + slash_hw, by + slash_hw * 1.5),
                     [color.cmyk(0, 0, 0, 1), style.linewidth.normal])

    # Horizontal connecting lines at top and bottom of break
    for by in [0, panel_h]:
        c.stroke(path.line(break_x_left, by, break_x_right, by),
                 [color.cmyk(0, 0, 0, 1), style.linewidth.thin])

    # ── Title ─────────────────────────────────────────────────────────────────
    total_w = panel_w + break_w + panel_w
    ref_g   = g_start if g_start is not None else g_stop
    if ref_g is not None:
        c.text(total_w / 2., panel_h + 0.5,
               f"{label} - Meta ORF A->G Editing",
               [pyx_text.halign.center, pyx_text.size.normalsize])

    # ── n transcripts annotation ───────────────────────────────────────────────
    for agg, xpos, anchor in [
        (start_agg, panel_w / 2.,              "start"),
        (stop_agg,  panel_w + break_w + panel_w / 2., "stop"),
    ]:
        if not agg.empty:
            n = int(agg["n_transcripts"].max())
            c.text(xpos, -0.6, f"n={n:,} tx",
                   [pyx_text.halign.center, pyx_text.size.small])

    c.writePDFfile(output_path)
    print(f"  Saved -> {output_path}.pdf", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# 7. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Meta ORF A->G editing plot around start and stop codons."
    )
    p.add_argument("--bam1",   required=True)
    p.add_argument("--bam2",   required=True)
    p.add_argument("--label1", default="BAM1")
    p.add_argument("--label2", default="BAM2")
    p.add_argument("--ref",    required=True)
    p.add_argument("--gtf",    required=True)
    p.add_argument("--output", default="metaORF")
    p.add_argument("--window", type=int, default=50,
                   help="nt window around start and stop codons (default: 50)")
    p.add_argument("--min_coverage", type=int, default=20,
                   help="Min reads at a position to include it (default: 20)")
    p.add_argument("--min_mapq",  type=int, default=20)
    p.add_argument("--min_baseq", type=int, default=10)
    p.add_argument("--gene_list", default=None,
                   help="Optional file of gene names to restrict analysis")
    return p.parse_args()


def main():
    args = parse_args()
    out  = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print("=== Meta ORF A->G Editing Analysis ===", file=sys.stderr)

    # ── Parse GTF ─────────────────────────────────────────────────────────────
    print("\nParsing GTF...", file=sys.stderr)
    transcripts = parse_gtf_transcripts(args.gtf)

    if args.gene_list:
        with open(args.gene_list) as fh:
            allowed = {l.strip() for l in fh if l.strip()}
        transcripts = [t for t in transcripts if t["gene_name"] in allowed]
        print(f"  {len(transcripts):,} transcripts after gene list filter.",
              file=sys.stderr)

    if not transcripts:
        print("ERROR: No transcripts found.", file=sys.stderr)
        sys.exit(1)

    ref_fasta = pysam.FastaFile(args.ref)

    # ── Process each BAM ──────────────────────────────────────────────────────
    for bam_path, label in [(args.bam1, args.label1), (args.bam2, args.label2)]:
        print(f"\nProcessing {label}...", file=sys.stderr)

        start_agg, stop_agg = collect_orf_data(
            bam_path, ref_fasta, transcripts,
            window=args.window,
            min_coverage=args.min_coverage,
            min_mapq=args.min_mapq,
            min_baseq=args.min_baseq,
            label=label,
        )

        # Save aggregations
        safe_label = re.sub(r"[^\w\-]", "_", label)
        start_agg.to_csv(f"{out}_{safe_label}_start_agg.csv", index=False)
        stop_agg.to_csv( f"{out}_{safe_label}_stop_agg.csv",  index=False)

        # Plot
        try:
            plot_meta_orf_pyx(
                start_agg=start_agg,
                stop_agg=stop_agg,
                label=label,
                window=args.window,
                output_path=f"{out}_{safe_label}",
            )
        except Exception as e:
            print(f"  WARNING: pyx plotting failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

    ref_fasta.close()
    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()