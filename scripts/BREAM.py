#!/usr/bin/env python3
"""
April 10, 2026 LT

#### BREAM probably doesn't work with data with less than 70-80% editing ######

Joint Probability Shadowing Analysis
=====================================
Adapts the oligoShadowingJointProbability.py algorithm (BREAM) to nanopore BAM data.

For each high-coverage gene:
  1. Build a reference editing frequency from BAM1 (P(edit) and P(no-edit)
     at every ref=A position across the gene body, in transcript coordinates).
  2. For each read in BAM2, slide a window across the transcript and compute
     the joint probability of the observed edit/no-edit pattern at all ref=A
     sites in the window, using the reference frequencies.
  3. Plot:
     - Meta read plot: median ± deciles of -log10(joint prob) across all reads
       at each window position, with His codon positions marked.
     - Individual read plots: per-read -log10(joint prob) traces for a
       sample of reads, with edit/no-edit tick marks below each trace.

Usage:
    python3 jointProbShadowing.py \
        --bam1 reference.bam --label1 "WT" \
        --bam2 query.bam     --label2 "3-AT" \
        --ref reference.fa \
        --gtf annotation.gtf \
        --output output_prefix \
        [--window 30] \
        [--min_coverage 50] \
        [--min_sites 5] \
        [--num_reads 10] \
        [--min_mapq 20] \
        [--min_baseq 10] \
        [--gene_list genes.txt]

Requirements:
    pip install pysam pandas numpy scipy
    pyx (for plotting)
"""

import argparse
import sys
import re
import math
import collections
from pathlib import Path

import pysam
import numpy as np
import pandas as pd
import scipy.stats


# ── Histidine codons ─────────────────────────────────────────────────────────
HIS_CODONS = {"CAT", "CAC"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. GTF parsing  (same as individualGeneHistidineAnalysis.py)
# ─────────────────────────────────────────────────────────────────────────────

def parse_gtf(gtf_path: str) -> dict:
    genes = {}
    gene_extents = {}
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            feature = fields[2]
            chrom   = fields[0]
            start   = int(fields[3]) - 1
            end     = int(fields[4])
            strand  = fields[6]
            m_gn    = re.search(r'gene_name "([^"]+)"', fields[8])
            m_tid   = re.search(r'transcript_id "([^"]+)"', fields[8])
            gname   = m_gn.group(1)  if m_gn  else None
            tid     = m_tid.group(1) if m_tid else "."
            if gname is None:
                continue
            if feature == "gene":
                gene_extents[gname] = (start, end)
            if feature == "CDS":
                if gname not in genes:
                    genes[gname] = {
                        "chrom": chrom, "strand": strand,
                        "transcript": tid, "gene_name": gname, "cds": [],
                    }
                genes[gname]["cds"].append((start, end))
    for gname, g in genes.items():
        if gname in gene_extents:
            g["gene_start"], g["gene_end"] = gene_extents[gname]
        else:
            g["gene_start"] = min(s for s, e in g["cds"])
            g["gene_end"]   = max(e for s, e in g["cds"])
        g["cds"].sort(key=lambda x: x[0], reverse=(g["strand"] == "-"))
    return genes


def complement_base(b: str) -> str:
    return b.translate(str.maketrans("ACGTacgt", "TGCAtgca"))


def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Coverage filter
# ─────────────────────────────────────────────────────────────────────────────

def mean_gene_coverage(bam: pysam.AlignmentFile, gene: dict,
                        min_mapq: int) -> float:
    total, bases = 0, 0
    for (s, e) in gene["cds"]:
        for col in bam.pileup(gene["chrom"], s, e, truncate=True,
                              min_mapping_quality=min_mapq,
                              stepper="samtools"):
            total += col.nsegments
            bases += 1
    return total / bases if bases > 0 else 0.0


def filter_high_coverage_genes(genes: dict, bam1_path: str, bam2_path: str,
                                 min_coverage: float, min_mapq: int) -> list:
    passing = []
    bam1 = pysam.AlignmentFile(bam1_path, "rb")
    bam2 = pysam.AlignmentFile(bam2_path, "rb")
    for i, (gname, gene) in enumerate(genes.items()):
        if (i + 1) % 200 == 0:
            print(f"  Coverage check {i+1}/{len(genes)}…", file=sys.stderr)
        if (mean_gene_coverage(bam1, gene, min_mapq) >= min_coverage and
                mean_gene_coverage(bam2, gene, min_mapq) >= min_coverage):
            passing.append(gname)
    bam1.close()
    bam2.close()
    print(f"  {len(passing):,}/{len(genes):,} genes pass coverage filter.",
          file=sys.stderr)
    return passing


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build reference editing frequencies from BAM1
#    Analogous to getReferenceFreq in the original script.
# ─────────────────────────────────────────────────────────────────────────────

def build_reference_freq(bam: pysam.AlignmentFile,
                          ref_fasta: pysam.FastaFile,
                          gene: dict,
                          min_mapq: int,
                          min_baseq: int) -> dict:
    """
    For every ref=A position in the gene body (transcript coordinates),
    collect observed bases across all reads and compute:
        {tx_pos: (freq_no_edit, freq_edit)}
    where freq_edit = fraction of A+G reads that show G (i.e. ag_edit_frac),
    and freq_no_edit = 1 - freq_edit.

    tx_pos is 0-based transcript coordinate.
    """
    chrom      = gene["chrom"]
    strand     = gene["strand"]
    gene_start = gene["gene_start"]
    gene_end   = gene["gene_end"]

    chrom_seq = ref_fasta.fetch(chrom).upper()

    # Walk gene body in transcript order
    if strand == "+":
        gpos_range = range(gene_start, gene_end)
    else:
        gpos_range = range(gene_end - 1, gene_start - 1, -1)

    ref_freq = {}   # tx_pos → (freq0, freq1)
    tx_pos = 0

    for gpos in gpos_range:
        ref_base_genomic = chrom_seq[gpos]
        ref_base_tx = complement_base(ref_base_genomic) \
                      if strand == "-" else ref_base_genomic

        if ref_base_tx == "A":
            counts = collections.Counter()
            for col in bam.pileup(chrom, gpos, gpos + 1, truncate=True,
                                   min_mapping_quality=min_mapq,
                                   min_base_quality=min_baseq,
                                   stepper="samtools"):
                if col.reference_pos != gpos:
                    continue
                for pread in col.pileups:
                    if pread.is_del or pread.is_refskip:
                        continue
                    qbase_raw = pread.alignment.query_sequence[
                        pread.query_position].upper()
                    needs_complement = pread.alignment.is_reverse != (strand == "-")
                    qbase = complement_base(qbase_raw) if needs_complement \
                            else qbase_raw
                    if qbase in ("A", "G"):
                        counts[qbase] += 1

            ag_total = counts["A"] + counts["G"]
            if ag_total > 0:
                freq1 = counts["G"] / ag_total   # P(edit)
                freq0 = counts["A"] / ag_total   # P(no edit)
                # Clamp to avoid log(0) — same spirit as the original
                freq1 = max(freq1, 1e-6)
                freq0 = max(freq0, 1e-6)
                ref_freq[tx_pos] = (freq0, freq1)

        tx_pos += 1

    return ref_freq


# ─────────────────────────────────────────────────────────────────────────────
# 4. Collect per-read edit observations from BAM2
# ─────────────────────────────────────────────────────────────────────────────

def collect_read_edits(bam: pysam.AlignmentFile,
                        ref_fasta: pysam.FastaFile,
                        gene: dict,
                        min_mapq: int,
                        min_baseq: int) -> dict:
    """
    For every read in BAM2 covering the gene body, record the edit status
    (0=no edit, 1=edit) at each ref=A position the read covers.

    Returns:
        {read_name: {tx_pos: edit_status (0 or 1)}}
    """
    chrom      = gene["chrom"]
    strand     = gene["strand"]
    gene_start = gene["gene_start"]
    gene_end   = gene["gene_end"]

    chrom_seq = ref_fasta.fetch(chrom).upper()

    # Map genomic position → tx_pos for this gene
    gpos_to_tx = {}
    tx_pos = 0
    if strand == "+":
        for gpos in range(gene_start, gene_end):
            ref_base_genomic = chrom_seq[gpos]
            if ref_base_genomic == "A":
                gpos_to_tx[gpos] = tx_pos
            tx_pos += 1
    else:
        for gpos in range(gene_end - 1, gene_start - 1, -1):
            ref_base_genomic = chrom_seq[gpos]
            if complement_base(ref_base_genomic) == "A":
                gpos_to_tx[gpos] = tx_pos
            tx_pos += 1

    read_edits = collections.defaultdict(dict)

    for read in bam.fetch(chrom, gene_start, gene_end):
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if read.mapping_quality < min_mapq:
            continue
        if read.query_sequence is None:
            continue

        is_rev = read.is_reverse
        needs_complement = is_rev != (strand == "-")

        for qpos, rpos in read.get_aligned_pairs(matches_only=True):
            if rpos not in gpos_to_tx:
                continue
            if read.query_qualities is not None:
                if read.query_qualities[qpos] < min_baseq:
                    continue
            qbase_raw = read.query_sequence[qpos].upper()
            qbase = complement_base(qbase_raw) if needs_complement else qbase_raw
            if qbase in ("A", "G"):
                tx = gpos_to_tx[rpos]
                read_edits[read.query_name][tx] = 1 if qbase == "G" else 0

    return dict(read_edits)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Compute joint probabilities per read
#    Analogous to computeJointProbabilitiesPerRead in the original script.
# ─────────────────────────────────────────────────────────────────────────────

def compute_joint_prob_per_read(read_edits: dict,
                                  ref_freq: dict,
                                  nt_window: int,
                                  min_sites: int,
                                  gene_len: int) -> dict:
    """
    For each read, slide a window of nt_window across transcript positions
    0..gene_len. At each window centre, collect all tx_pos where:
      - the read has an observation (edit or no-edit), AND
      - ref_freq has a frequency estimate
    If there are >= min_sites such positions, compute the joint probability
    in log space (sum of log probs) to avoid floating-point underflow,
    then convert back to a probability for storage.

    Returns:
        {read_name: [(window_centre, joint_prob), ...]}
    """
    joint_probs = {}

    for read_name, pos_dict in read_edits.items():
        trace = []
        for start in range(0, gene_len, 1):
            end = start + nt_window
            log_prob_sum = 0.0
            n_sites = 0
            for tx in range(start, end):
                if tx in pos_dict and tx in ref_freq:
                    edit_val = pos_dict[tx]
                    prob = ref_freq[tx][edit_val]  # already clamped >= 1e-6
                    log_prob_sum += math.log10(prob)
                    n_sites += 1
            if n_sites >= min_sites:
                centre = start + nt_window / 2.
                # Store as -log10(joint_prob) directly to avoid underflow
                trace.append((centre, -log_prob_sum))
        if trace:
            joint_probs[read_name] = trace

    return joint_probs


# ─────────────────────────────────────────────────────────────────────────────
# 6. Meta read aggregation
#    Analogous to getMetaReadPerBarcode in the original script.
# ─────────────────────────────────────────────────────────────────────────────

def get_meta_read(joint_probs: dict) -> list:
    """
    Aggregate -log10(joint prob) scores across all reads at each window centre.
    Returns [(centre, 5th pctile, median, 95th pctile), ...] sorted by centre.
    Values are already in -log10 space (higher = more unusual).
    """
    by_pos = collections.defaultdict(list)
    for read_name, trace in joint_probs.items():
        for centre, neg_log_p in trace:
            by_pos[centre].append(neg_log_p)

    meta = []
    for centre in sorted(by_pos.keys()):
        vals = by_pos[centre]
        meta.append((
            centre,
            float(np.quantile(vals, 0.05)),   # low end
            float(np.quantile(vals, 0.50)),   # median
            float(np.quantile(vals, 0.95)),   # high end
        ))

    return meta

# ─────────────────────────────────────────────────────────────────────────────
# 7. His codon tx positions
# ─────────────────────────────────────────────────────────────────────────────

def find_his_codon_tx_positions(ref_fasta: pysam.FastaFile,
                                 gene: dict) -> list:
    chrom  = gene["chrom"]
    strand = gene["strand"]
    tx_seq = ""
    for (cs, ce) in gene["cds"]:
        seg = ref_fasta.fetch(chrom, cs, ce).upper()
        if strand == "-":
            seg = reverse_complement(seg)
        tx_seq += seg
    his_pos = []
    for i in range(0, len(tx_seq) - 2, 3):
        if tx_seq[i:i+3] in HIS_CODONS:
            his_pos.append(i + 1)
    return his_pos

# ─────────────────────────────────────────────────────────────────────────────
# 8. Plotting (pyx)
#    Analogous to mkMetaPlot and mkIndividualReadPlots.
# ─────────────────────────────────────────────────────────────────────────────

def plot_gene_pyx(gene_name: str,
                   meta: list,
                   joint_probs: dict,
                   his_positions: list,
                   read_edits: dict,
                   label1: str,
                   label2: str,
                   gene_len: int,
                   num_reads: int,
                   pdf_path: str):
    """
    Two-section PDF per gene:
      Section A: meta read plot (-log10 joint prob, median ± deciles)
      Section B: individual read traces stacked vertically, with edit
                 tick marks below each trace.
    His codon positions marked with vertical lines on all panels.
    """
    from pyx import canvas, graph, color, style, path, text as pyx_text, deco

    col_ref  = color.cmyk(0, 0, 0, 1)       # black  — label1 (ref)
    col_qry  = color.cmyk(1, 0.5, 0, 0)     # blue   — label2 (query)
    col_his  = color.cmyk(0, 1, 1, 0)       # red    — His codon
    col_edit = color.cmyk(0, 0, 0, 1)       # black  — edit tick
    col_no   = color.cmyk(0, 0.8, 1, 0)     # orange — no-edit tick

    x_min, x_max = 0, gene_len
    panel_w = 12
    tick_h  = 0.25
    read_h  = 1.2
    gap     = 0.6

    # ── y range for meta plot ─────────────────────────────────────────────────
    all_y = [entry[3] for entry in meta]
    y_max_meta = max(all_y) * 1.1 if all_y else 15
    y_max_meta = max(y_max_meta, 1.0)

    c = canvas.canvas()

    # ── Meta plot ─────────────────────────────────────────────────────────────
    meta_ypos = num_reads * (read_h + tick_h + gap) + gap * 2

    g_meta = graph.graphxy(
        width=panel_w, height=3,
        xpos=0, ypos=meta_ypos,
        x=graph.axis.linear(min=x_min, max=x_max,
                            title="Position Along Transcript (nt)"),
        y=graph.axis.linear(min=0, max=y_max_meta,
                            title="-log10 Joint Prob"),
    )

    # His codon lines
    for hp in his_positions:
        g_meta.plot(
            graph.data.function(f"x(y)={hp}", min=0, max=y_max_meta),
            [graph.style.line([col_his, style.linewidth.thin,
                               style.linestyle.solid])]
        )

    # 5th/95th decile dotted bounds
    for col_idx in [1, 3]:
        pts = [(entry[0], entry[col_idx]) for entry in meta]
        if pts:
            g_meta.plot(graph.data.points(pts, x=1, y=2),
                        [graph.style.line([col_qry, style.linewidth.thin,
                                           style.linestyle.dotted])])

    # Median solid line
    med_pts = [(entry[0], entry[2]) for entry in meta]
    if med_pts:
        g_meta.plot(graph.data.points(med_pts, x=1, y=2),
                    [graph.style.line([col_qry, style.linewidth.normal,
                                       style.linestyle.solid])])

    c.insert(g_meta)
    c.text(g_meta.xpos + g_meta.width / 2.,
           g_meta.ypos + g_meta.height + 0.4,
           f"{gene_name} - {label2} joint prob (ref: {label1})",
           [pyx_text.halign.center, pyx_text.size.normalsize])

    # ── Individual read plots ─────────────────────────────────────────────────
    read_names = list(joint_probs.keys())[:num_reads]

    for jj, read_name in enumerate(read_names):
        trace = joint_probs[read_name]
        ypos  = (num_reads - 1 - jj) * (read_h + tick_h + gap)

        # Compute y range for this read
        read_y_vals = [v for _, v in trace]
        y_min_read  = max(min(read_y_vals), 0) if read_y_vals else 0
        y_max_read  = max(read_y_vals) * 1.1   if read_y_vals else 15
        y_max_read  = max(y_max_read, 1.0)

        g_read = graph.graphxy(
            width=panel_w, height=read_h,
            xpos=0, ypos=ypos + tick_h,
            x=graph.axis.linkedaxis(g_meta.axes["x"]),
            y=graph.axis.linear(min=y_min_read, max=y_max_read, title=""),
        )

        # His codon lines
        for hp in his_positions:
            g_read.plot(
                graph.data.function(f"x(y)={hp}", min=y_min_read, max=y_max_read),
                [graph.style.line([col_his, style.linewidth.thin,
                                   style.linestyle.solid])]
            )

        # Joint prob trace — values already in -log10 space
        if trace:
            g_read.plot(graph.data.points(list(trace), x=1, y=2),
                        [graph.style.line([col_qry, style.linewidth.thin,
                                           style.linestyle.solid])])

        c.insert(g_read)

        # Edit/no-edit tick marks below the trace
        g_ticks = graph.graphxy(
            width=panel_w, height=tick_h,
            xpos=0, ypos=ypos,
            x=graph.axis.linkedaxis(g_meta.axes["x"]),
            y=graph.axis.linear(min=0, max=1),
        )

        pos_dict = read_edits.get(read_name, {})
        edit_pts   = [(tx, 0.5) for tx, v in pos_dict.items() if v == 1]
        noedit_pts = [(tx, 0.5) for tx, v in pos_dict.items() if v == 0]

        for pts, col in [(edit_pts, col_edit), (noedit_pts, col_no)]:
            for tx, y in pts:
                g_ticks.plot(
                    graph.data.function(f"x(y)={tx}", min=0, max=1),
                    [graph.style.line([col, style.linewidth.thin,
                                       style.linestyle.solid])]
                )

        c.insert(g_ticks)

        # Short read label to the right
        c.text(panel_w + 0.15,
               ypos + tick_h + read_h / 2.,
               read_name[:20],
               [pyx_text.valign.middle, pyx_text.size.tiny])

    c.writePDFfile(pdf_path)


# ─────────────────────────────────────────────────────────────────────────────
# 9. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Joint probability shadowing analysis for nanopore BAMs."
    )
    p.add_argument("--bam1",   required=True, help="Reference BAM (e.g. WT)")
    p.add_argument("--bam2",   required=True, help="Query BAM (e.g. 3-AT)")
    p.add_argument("--label1", default="BAM1")
    p.add_argument("--label2", default="BAM2")
    p.add_argument("--ref",    required=True)
    p.add_argument("--gtf",    required=True)
    p.add_argument("--output", default="jointProb")
    p.add_argument("--window", type=int, default=30,
                   help="Sliding window size in nt (default: 30)")
    p.add_argument("--min_coverage", type=float, default=50.0)
    p.add_argument("--min_sites", type=int, default=5,
                   help="Min ref=A sites per window to compute joint prob (default: 5)")
    p.add_argument("--num_reads", type=int, default=10,
                   help="Number of individual reads to plot per gene (default: 10)")
    p.add_argument("--min_mapq",  type=int, default=20)
    p.add_argument("--min_baseq", type=int, default=10)
    p.add_argument("--gene_list", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out  = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print("=== Joint Probability Shadowing Analysis ===", file=sys.stderr)

    # ── Parse GTF ─────────────────────────────────────────────────────────────
    print("\nParsing GTF…", file=sys.stderr)
    genes = parse_gtf(args.gtf)
    print(f"  {len(genes):,} genes.", file=sys.stderr)

    if args.gene_list:
        with open(args.gene_list) as fh:
            allowed = {l.strip() for l in fh if l.strip()}
        genes = {k: v for k, v in genes.items() if k in allowed}
        print(f"  {len(genes):,} after gene list filter.", file=sys.stderr)

    # ── Coverage filter ───────────────────────────────────────────────────────
    print(f"\nFiltering genes (mean CDS coverage >= {args.min_coverage}x)…",
          file=sys.stderr)
    passing = filter_high_coverage_genes(
        genes, args.bam1, args.bam2,
        min_coverage=args.min_coverage, min_mapq=args.min_mapq,
    )
    if not passing:
        print("ERROR: No genes passed coverage filter.", file=sys.stderr)
        sys.exit(1)

    # ── Open resources ────────────────────────────────────────────────────────
    ref_fasta = pysam.FastaFile(args.ref)
    bam1      = pysam.AlignmentFile(args.bam1, "rb")
    bam2      = pysam.AlignmentFile(args.bam2, "rb")

    pdf_dir = Path(f"{out}_gene_pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    print(f"\nProcessing {len(passing):,} genes…", file=sys.stderr)

    for i, gname in enumerate(passing):
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(passing)}: {gname}…", file=sys.stderr)

        gene     = genes[gname]
        gene_len = gene["gene_end"] - gene["gene_start"]

        # Reference frequencies from BAM1
        ref_freq = build_reference_freq(
            bam1, ref_fasta, gene, args.min_mapq, args.min_baseq
        )
        if not ref_freq:
            continue

        # Per-read edit observations from BAM2
        read_edits = collect_read_edits(
            bam2, ref_fasta, gene, args.min_mapq, args.min_baseq
        )
        if not read_edits:
            continue

        # Joint probabilities
        joint_probs = compute_joint_prob_per_read(
            read_edits, ref_freq,
            nt_window=args.window,
            min_sites=args.min_sites,
            gene_len=gene_len,
        )
        if not joint_probs:
            continue

        # Meta read aggregation
        meta = get_meta_read(joint_probs)

        # His codon positions
        his_positions = find_his_codon_tx_positions(ref_fasta, gene)

        # Summary
        all_scores = [v for trace in joint_probs.values() for _, v in trace]
        summary_rows.append({
            "gene":          gname,
            "n_reads":       len(joint_probs),
            "n_his_codons":  len(his_positions),
            "gene_len":      gene_len,
            "n_ref_a_sites": len(ref_freq),
            "median_neg_log10p": float(np.median(all_scores)),
        })

        # Plot
        safe_name = re.sub(r"[^\w\-]", "_", gname)
        plot_gene_pyx(
            gene_name=gname,
            meta=meta,
            joint_probs=joint_probs,
            his_positions=his_positions,
            read_edits=read_edits,
            label1=args.label1,
            label2=args.label2,
            gene_len=gene_len,
            num_reads=args.num_reads,
            pdf_path=str(pdf_dir / safe_name),
        )

    bam1.close()
    bam2.close()
    ref_fasta.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows).sort_values(
        "median_log10p", ascending=False
    )
    summary_csv = f"{out}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n  Saved summary → {summary_csv}", file=sys.stderr)
    print(f"  Gene PDFs    → {pdf_dir}/", file=sys.stderr)
    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()