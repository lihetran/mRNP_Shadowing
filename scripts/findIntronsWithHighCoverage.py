#!/usr/bin/env python3
"""
find_intron_genes.py
--------------------
Identifies genes that contain introns from a GTF annotation file,
and optionally validates them using splice-junction-spanning reads
from a BAM file.

A gene has introns (per GTF) when consecutive exons in a transcript
model have a gap between them.

A read spans a junction when its CIGAR string contains an N operation
(skip / intron), which is how splice-aware aligners (STAR, HISAT2, etc.)
encode split-read alignments across introns.

When --bam is supplied the script:
  1. Computes mean exonic coverage per gene (pileup over merged exons).
  2. Counts junction-spanning reads per gene (reads with N in CIGAR).
  3. Validates each annotated intron against observed splice junctions
     in the BAM, reporting how many GTF introns are confirmed by >= 1
     supporting read.
  4. Ranks output by mean exonic coverage (descending).

Requirements:
    pysam  (only needed when --bam is supplied)
    Install: pip install pysam

Usage:
    python find_intron_genes.py annotation.gtf
    python find_intron_genes.py annotation.gtf -o results.tsv
    python find_intron_genes.py annotation.gtf.gz             # gzip ok

    # With BAM:
    python find_intron_genes.py annotation.gtf --bam sample.bam -o ranked.tsv
    python find_intron_genes.py annotation.gtf --bam sample.bam \\
        --with-introns-only --min-coverage 10 -o filtered.tsv
"""

import argparse
import gzip
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# GTF helpers
# ---------------------------------------------------------------------------

def open_gtf(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def parse_attribute(attr_string, key):
    for field in attr_string.split(";"):
        field = field.strip()
        if field.startswith(key + " "):
            return field[len(key):].strip().strip('"')
    return None


def parse_gtf(path):
    """
    Returns:
        gene_info  : gene_id -> {chrom, strand, start, end, gene_name}
        transcripts: gene_id -> {transcript_id -> [(start, end), ...]}
    Coordinates are 0-based half-open internally.
    """
    gene_info = {}
    transcripts = defaultdict(lambda: defaultdict(list))

    with open_gtf(path) as fh:
        for lineno, line in enumerate(fh, 1):
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _, feature, start, end, _, strand, _, attrs = parts
            feature = feature.lower()
            gene_id = parse_attribute(attrs, "gene_id")
            if not gene_id:
                continue
            try:
                start, end = int(start) - 1, int(end)   # 0-based half-open
            except ValueError:
                sys.stderr.write(f"Warning: bad coords line {lineno}, skipping.\n")
                continue

            if feature == "gene":
                gene_info[gene_id] = {
                    "chrom": chrom, "strand": strand,
                    "start": start, "end": end,
                    "gene_name": parse_attribute(attrs, "gene_name") or gene_id,
                }
            elif feature == "exon":
                tx_id = parse_attribute(attrs, "transcript_id")
                if not tx_id:
                    continue
                if gene_id not in gene_info:
                    gene_info[gene_id] = {
                        "chrom": chrom, "strand": strand,
                        "start": start, "end": end,
                        "gene_name": parse_attribute(attrs, "gene_name") or gene_id,
                    }
                else:
                    gene_info[gene_id]["start"] = min(gene_info[gene_id]["start"], start)
                    gene_info[gene_id]["end"]   = max(gene_info[gene_id]["end"],   end)
                transcripts[gene_id][tx_id].append((start, end))

    return gene_info, transcripts


# ---------------------------------------------------------------------------
# Intron / interval helpers
# ---------------------------------------------------------------------------

def find_introns_for_transcript(exons):
    """Return intron intervals as gaps between sorted exons."""
    sorted_exons = sorted(exons)
    introns = []
    for i in range(1, len(sorted_exons)):
        prev_end   = sorted_exons[i - 1][1]
        curr_start = sorted_exons[i][0]
        if curr_start > prev_end:
            introns.append((prev_end, curr_start))
    return introns


def merge_intervals(intervals):
    if not intervals:
        return []
    merged = [list(sorted(intervals)[0])]
    for s, e in sorted(intervals)[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(iv) for iv in merged]


def collect_exon_union(tx_dict):
    all_exons = [iv for exons in tx_dict.values() for iv in exons]
    return merge_intervals(all_exons)


def collect_annotated_introns(tx_dict):
    """Return the set of unique annotated intron intervals for a gene."""
    introns = set()
    for exons in tx_dict.values():
        for iv in find_introns_for_transcript(exons):
            introns.add(iv)
    return introns


def cigar_junctions(read_start, cigar_tuples):
    """
    Walk a CIGAR tuple list and return a list of (intron_start, intron_end)
    splice junctions encoded by N (op=3) operations.
    Coordinates are 0-based half-open, on the reference.

    CIGAR ops that consume the reference: M(0) D(2) N(3) =(7) X(8)
    CIGAR ops that do NOT consume reference: I(1) S(4) H(5) P(6)
    """
    ref_pos = read_start
    junctions = []
    for op, length in cigar_tuples:
        if op in (0, 2, 7, 8):   # M, D, =, X — advance reference
            ref_pos += length
        elif op == 3:             # N — splice junction
            junctions.append((ref_pos, ref_pos + length))
            ref_pos += length
        # I, S, H, P do not advance reference
    return junctions


# ---------------------------------------------------------------------------
# BAM analysis — coverage + junction validation
# ---------------------------------------------------------------------------

def compute_bam_stats(bam_path, gene_info, transcripts):
    """
    For each gene compute:
      - mean_coverage          : mean depth over merged exonic bases
      - total_reads            : reads overlapping the gene body
      - exonic_bases           : total merged exonic base pairs
      - junction_spanning_reads: reads with >= 1 N in CIGAR inside gene
      - observed_junctions     : set of (start, end) junctions seen in BAM
      - gtf_validated_junctions: annotated introns with >= 1 BAM-support
      - n_unique_introns       : total annotated introns (from GTF)
      - junction_validation_rate: gtf_validated / n_unique_introns

    Returns dict: gene_id -> stats dict
    """
    try:
        import pysam
    except ImportError:
        sys.stderr.write(
            "Error: pysam is required for BAM analysis.\n"
            "Install: pip install pysam\n"
        )
        sys.exit(1)

    sys.stderr.write(f"Opening BAM: {bam_path}\n")
    bam = pysam.AlignmentFile(bam_path, "rb")
    try:
        bam.check_index()
    except (ValueError, AttributeError):
        sys.stderr.write(
            "Warning: BAM index not found — this may be slow or fail.\n"
            "Index with: samtools index sample.bam\n"
        )

    all_gene_ids = sorted(set(gene_info) | set(transcripts))
    n = len(all_gene_ids)
    stats = {}

    for i, gene_id in enumerate(all_gene_ids, 1):
        if i % 500 == 0 or i == n:
            sys.stderr.write(f"  BAM: {i}/{n} genes\r")

        info   = gene_info.get(gene_id)
        tx_dict = transcripts.get(gene_id, {})
        if not info or not tx_dict:
            continue

        chrom      = info["chrom"]
        gene_start = info["start"]
        gene_end   = info["end"]
        exon_union = collect_exon_union(tx_dict)
        annotated_introns = collect_annotated_introns(tx_dict)

        # ── 1. Exonic coverage via pileup ──────────────────────────────────
        total_depth  = 0
        exonic_bases = 0
        skip = False

        for ex_start, ex_end in exon_union:
            if skip:
                break
            exonic_bases += ex_end - ex_start
            try:
                for col in bam.pileup(
                    chrom, ex_start, ex_end,
                    truncate=True, stepper="nofilter", ignore_overlaps=False,
                ):
                    total_depth += col.nsegments
            except (ValueError, KeyError):
                skip = True

        mean_cov = total_depth / exonic_bases if exonic_bases > 0 else 0.0

        # ── 2. Junction-spanning reads + observed junctions ────────────────
        junction_read_count = 0
        observed_junctions  = set()   # (intron_start, intron_end) on reference

        if not skip:
            try:
                for read in bam.fetch(chrom, gene_start, gene_end):
                    if read.is_unmapped or read.cigartuples is None:
                        continue
                    junctions = cigar_junctions(read.reference_start, read.cigartuples)
                    if junctions:
                        junction_read_count += 1
                        for junc in junctions:
                            # Only keep junctions that fall within the gene
                            if junc[0] >= gene_start and junc[1] <= gene_end:
                                observed_junctions.add(junc)
            except (ValueError, KeyError):
                pass

        # ── 3. Validate annotated introns against observed junctions ───────
        validated = annotated_introns & observed_junctions
        n_annotated = len(annotated_introns)
        n_validated = len(validated)
        val_rate = n_validated / n_annotated if n_annotated > 0 else float("nan")

        # ── 4. Total reads overlapping gene body ───────────────────────────
        total_reads = 0
        if not skip:
            try:
                total_reads = bam.count(chrom, gene_start, gene_end)
            except (ValueError, KeyError):
                pass

        stats[gene_id] = {
            "mean_coverage":           mean_cov,
            "total_reads":             total_reads,
            "exonic_bases":            exonic_bases,
            "junction_spanning_reads": junction_read_count,
            "gtf_validated_junctions": n_validated,
            "n_unique_introns":        n_annotated,
            "junction_validation_rate": val_rate,
        }

    sys.stderr.write("\n")
    bam.close()
    return stats


# ---------------------------------------------------------------------------
# Gene classification
# ---------------------------------------------------------------------------

def classify_genes(gene_info, transcripts, bam_stats=None):
    results = []
    all_gene_ids = set(gene_info) | set(transcripts)

    for gene_id in sorted(all_gene_ids):
        info    = gene_info.get(gene_id, {"chrom":"?","strand":"?","start":0,"end":0,"gene_name":gene_id})
        tx_dict = transcripts.get(gene_id, {})
        if not tx_dict:
            continue

        gene_has_introns           = False
        n_transcripts_with_introns = 0
        all_introns                = set()

        for tx_id, exons in tx_dict.items():
            ivs = find_introns_for_transcript(exons)
            if ivs:
                gene_has_introns = True
                n_transcripts_with_introns += 1
                all_introns.update(ivs)

        result = {
            "gene_id":                   gene_id,
            "gene_name":                 info["gene_name"],
            "chrom":                     info["chrom"],
            "start":                     info["start"] + 1,   # 1-based output
            "end":                       info["end"],
            "strand":                    info["strand"],
            "has_introns":               gene_has_introns,
            "n_transcripts":             len(tx_dict),
            "n_transcripts_with_introns": n_transcripts_with_introns,
            "n_unique_introns":          len(all_introns),
        }

        if bam_stats is not None:
            s = bam_stats.get(gene_id, {
                "mean_coverage": float("nan"),
                "total_reads": 0,
                "exonic_bases": 0,
                "junction_spanning_reads": 0,
                "gtf_validated_junctions": 0,
                "junction_validation_rate": float("nan"),
            })
            # n_unique_introns from BAM stats may differ if BAM was not skipped;
            # prefer the GTF-derived count already in result.
            result["mean_coverage"]           = s["mean_coverage"]
            result["total_reads"]             = s["total_reads"]
            result["exonic_bases"]            = s["exonic_bases"]
            result["junction_spanning_reads"] = s["junction_spanning_reads"]
            result["gtf_validated_junctions"] = s["gtf_validated_junctions"]
            result["junction_validation_rate"] = s["junction_validation_rate"]

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def fmt_float(v, decimals=4):
    """Format float; return 'NA' for NaN."""
    return f"{v:.{decimals}f}" if v == v else "NA"


def write_output(results, out_fh, summary_only=False, has_bam=False):
    if not summary_only:
        base_cols = [
            "rank", "gene_id", "gene_name", "chrom", "start", "end", "strand",
            "has_introns", "n_transcripts",
            "n_transcripts_with_introns", "n_unique_introns",
        ]
        bam_cols = [
            "mean_coverage", "total_reads", "exonic_bases",
            "junction_spanning_reads",
            "gtf_validated_junctions", "junction_validation_rate",
        ] if has_bam else []

        out_fh.write("\t".join(base_cols + bam_cols) + "\n")

        for rank, r in enumerate(results, 1):
            row = [
                str(rank),
                r["gene_id"], r["gene_name"],
                r["chrom"], str(r["start"]), str(r["end"]), r["strand"],
                str(r["has_introns"]),
                str(r["n_transcripts"]),
                str(r["n_transcripts_with_introns"]),
                str(r["n_unique_introns"]),
            ]
            if has_bam:
                row += [
                    fmt_float(r.get("mean_coverage", float("nan"))),
                    str(r.get("total_reads", 0)),
                    str(r.get("exonic_bases", 0)),
                    str(r.get("junction_spanning_reads", 0)),
                    str(r.get("gtf_validated_junctions", 0)),
                    fmt_float(r.get("junction_validation_rate", float("nan"))),
                ]
            out_fh.write("\t".join(row) + "\n")

    # Summary
    total        = len(results)
    with_introns = sum(1 for r in results if r["has_introns"])
    pct = lambda x: f"{100*x/total:.1f}%" if total else "n/a"

    out_fh.write("\n# -- Summary ------------------------------------------\n")
    out_fh.write(f"# Total genes analysed       : {total}\n")
    out_fh.write(f"# Genes WITH introns (GTF)   : {with_introns}  ({pct(with_introns)})\n")
    out_fh.write(f"# Genes WITHOUT introns (GTF): {total-with_introns}  ({pct(total-with_introns)})\n")

    if has_bam and total:
        covs = [r["mean_coverage"] for r in results if r.get("mean_coverage") == r.get("mean_coverage")]
        if covs:
            out_fh.write(f"# Mean exonic coverage       : {sum(covs)/len(covs):.2f}x\n")
            out_fh.write(f"# Top coverage gene          : {results[0]['gene_name']}  ({covs[0]:.2f}x)\n")

        jsr_total = sum(r.get("junction_spanning_reads", 0) for r in results)
        out_fh.write(f"# Junction-spanning reads    : {jsr_total:,}\n")

        validated_genes = sum(1 for r in results if r.get("gtf_validated_junctions", 0) > 0)
        out_fh.write(f"# Genes with >= 1 validated junction: {validated_genes}\n")

        val_rates = [r["junction_validation_rate"] for r in results
                     if r.get("n_unique_introns", 0) > 0
                     and r.get("junction_validation_rate") == r.get("junction_validation_rate")]
        if val_rates:
            out_fh.write(f"# Mean junction validation rate : {sum(val_rates)/len(val_rates):.1%}\n")

        out_fh.write(f"# Ranking                    : mean exonic coverage (descending)\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Identify intron-containing genes from a GTF, optionally validating "
            "annotated introns against splice-junction-spanning reads in a BAM."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GTF only
  python find_intron_genes.py annotation.gtf -o results.tsv

  # GTF + BAM: coverage ranking + junction validation
  python find_intron_genes.py annotation.gtf --bam sample.bam -o ranked.tsv

  # Intron genes, min 10x coverage
  python find_intron_genes.py annotation.gtf --bam sample.bam \\
      --with-introns-only --min-coverage 10 -o filtered.tsv
""",
    )
    parser.add_argument("gtf", help="GTF annotation file (plain or .gz)")
    parser.add_argument("--bam", default=None, metavar="BAM",
                        help="Indexed BAM file. Enables coverage computation, "
                             "junction-spanning read counting, and GTF intron "
                             "validation. Requires pysam.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output TSV (default: stdout)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print only summary statistics")
    parser.add_argument("--with-introns-only", action="store_true",
                        help="Output only genes with >= 1 annotated intron")
    parser.add_argument("--min-coverage", type=float, default=None, metavar="FLOAT",
                        help="(Requires --bam) Drop genes below this mean exonic coverage")
    args = parser.parse_args()

    if args.min_coverage is not None and not args.bam:
        parser.error("--min-coverage requires --bam")

    sys.stderr.write(f"Parsing GTF: {args.gtf}\n")
    gene_info, transcripts = parse_gtf(args.gtf)
    sys.stderr.write(f"  Genes       : {len(gene_info)}\n")
    sys.stderr.write(f"  Transcripts : {sum(len(v) for v in transcripts.values())}\n")

    bam_stats = None
    if args.bam:
        bam_stats = compute_bam_stats(args.bam, gene_info, transcripts)

    sys.stderr.write("Classifying genes...\n")
    results = classify_genes(gene_info, transcripts, bam_stats)

    if args.with_introns_only:
        results = [r for r in results if r["has_introns"]]

    if args.min_coverage is not None:
        before = len(results)
        results = [r for r in results if r.get("mean_coverage", 0) >= args.min_coverage]
        sys.stderr.write(f"  min-coverage filter: {before} -> {len(results)} genes\n")

    if bam_stats is not None:
        results.sort(key=lambda r: (-r.get("mean_coverage", 0), r["gene_id"]))

    out_fh = open(args.output, "w") if args.output else sys.stdout
    try:
        write_output(results, out_fh,
                     summary_only=args.summary_only,
                     has_bam=(bam_stats is not None))
    finally:
        if args.output:
            out_fh.close()
            sys.stderr.write(f"Results written to: {args.output}\n")


if __name__ == "__main__":
    main()