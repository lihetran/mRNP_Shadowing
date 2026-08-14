"""
doradoAlignToParquetPipeline.py -- Liam Tran

Chains doradoAligner_AtoG.py's dual A->G/T->C pathway alignment (see that
module's docstring for why both pathways are needed to correctly capture
reads from genes on both genomic strands) directly into
shadowingBamToParquetWithGTF2.py's per-read parquet extraction, so a raw
unaligned dorado bam goes straight to parquet chunks in one call.

Usage:
  python3 doradoAlignToParquetPipeline.py \
      --reads_bam  unaligned.bam \
      --ref_fasta  genome.fa \
      --gtf        genome.gtf \
      --output_dir parquet_out/ \
      [--out_bam aligned.bam] [--coding_only] [--chunk_size 50000] \
      [--barcode_summary sequencing_summary.txt] [--shard_size 2000000]

  from doradoAlignToParquetPipeline import run_pipeline
  run_pipeline(reads_bam="...", ref_fasta="...", gtf="...", output_dir="...")
"""
import argparse
import subprocess
import pysam
from pathlib import Path

from doradoAligner_AtoG import align_reads
from shadowingBamToParquetWithGTF2 import generate_parquet, build_barcode_lookup


def shard_reads_bam(reads_bam, shard_size, shard_dir):
    """
    Stream reads_bam into a sequence of smaller unaligned bam shards of at
    most shard_size reads each, written under shard_dir. A generator, not a
    pre-materialized list: each shard path is yielded as soon as that shard
    is written, so a caller that processes-and-deletes each shard before
    asking for the next never needs more than ~1 extra shard's worth of
    disk space, regardless of how many shards there are in total.
    """
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(reads_bam).stem

    with pysam.AlignmentFile(reads_bam, "rb", check_sq=False) as in_bam:
        shard_index = 0
        out = None
        count = 0
        shard_path = None
        for read in in_bam:
            if out is None:
                shard_path = shard_dir / f"{stem}_shard{shard_index}.bam"
                out = pysam.AlignmentFile(str(shard_path), "wb", template=in_bam)
                count = 0
            out.write(read)
            count += 1
            if count >= shard_size:
                out.close()
                yield shard_path
                out = None
                shard_index += 1
        if out is not None:
            out.close()
            yield shard_path


def merge_shard_bams(shard_aligned_bams, merged_bam):
    """
    Merge a list of per-shard sorted+indexed aligned bams (as produced by
    doradoAligner_AtoG.align_reads, one per shard) into one sorted+indexed
    bam at merged_bam via `samtools merge`. Returns merged_bam.
    """
    cmd = 'samtools merge -f {} {}'.format(merged_bam, ' '.join(shard_aligned_bams))
    print(cmd)
    subprocess.call(cmd, shell=True)
    subprocess.call('samtools index {}'.format(merged_bam), shell=True)
    return merged_bam


def run_pipeline(reads_bam, ref_fasta, gtf, output_dir,
                  out_bam=None, coding_only=False, chunk_size=50000,
                  keep_intermediates=True, barcode_summary=None, shard_size=None,
                  splice_aware=False, use_junc_bed=True, max_intron_length=3000,
                  flag_intron_threshold=2500, threads=3):
    """
    Align reads_bam against ref_fasta (dual A->G/T->C pathway), then convert
    the resulting aligned bam straight into parquet chunks under output_dir.
    Returns (aligned_bam_path, total_reads_written).

    keep_intermediates: True by default -- keeps align_reads' own per-run
    intermediates (mutated bams, per-pathway tmp alignments, pre-sort merged
    bam) instead of deleting them. This is deliberately the default (not
    opt-in): a parquet's absolute_indices/aligned_pairs are only as
    trustworthy as the alignment that produced them, and a bam that's
    already been deleted by the time something downstream looks wrong (e.g.
    findLUTICandidateReads.py flagging a spurious transcript-spanning read)
    can't be pulled back up to check its actual CIGAR against what the
    parquet claims -- exactly what happened chasing down a
    fabricated-junction false positive this pipeline's --splice_aware path
    produced. Pass False to restore the old cleanup behavior for these (e.g.
    for very large runs where disk space matters more than being able to
    re-inspect a specific alignment later).

    In the sharded path specifically, aligned_bam itself is never kept
    per-shard regardless of this flag: every shard's aligned bam is always
    merged into one combined, sorted+indexed bam at out_bam (see
    merge_shard_bams) once all shards are done, and the now-redundant
    per-shard raw and aligned bams are always deleted right after -- one bam
    to check instead of hunting through N shard bams is the whole point of
    keeping an aligned bam around in the first place.

    barcode_summary: optional path to a MinKNOW sequencing_summary.txt with a
    barcode_arrangement column (native barcoding kit runs). When given,
    output is split into output_dir/<barcode>/ subdirectories (unmatched
    reads go to output_dir/unclassified/).

    shard_size: optional read count per shard. doradoAligner_AtoG's
    dual-pathway alignment holds every read's full sequence in memory twice
    (once per pathway) for the whole duration of the alignment step, which
    OOMs on tens-of-millions-of-read runs. When shard_size is given,
    reads_bam is streamed into shards of at most this many reads, each
    fully aligned and converted to parquet (accumulating into the same
    output_dir/<barcode>/ subdirectories) before the next shard is even
    created, bounding peak memory to one shard's worth instead of the
    whole input. When omitted, behaves exactly as before (single pass).

    splice_aware: reads_bam is spliced mRNA/cDNA that may span introns.
    Forwarded to doradoAligner_AtoG.align_reads -- see its docstring and
    minimap2_align_splice_aware's docstring for why this does NOT just mean
    "pass -x splice": plain splice-preset alignment was confirmed to
    fabricate huge numbers of false splice junctions on real data, so this
    uses a validated max-intron-length cap via dorado's own bundled
    minimap2 binary instead of `dorado aligner` directly. False (default)
    aligns as plain genomic DNA (map-ont) -- use this if reads_bam has no
    real introns to worry about.

    use_junc_bed: only relevant when splice_aware=True. Build a --junc-bed
    from gtf and pass it to every alignment call (see
    doradoAligner_AtoG.build_junction_bed) so minimap2 is biased toward
    real annotated splice junctions. Set False to fall back to bare
    splice-preset alignment (e.g. to compare against, or for a GTF whose
    annotation isn't trustworthy).

    max_intron_length: only relevant when splice_aware=True -- forwarded to
    align_reads (minimap2 -G). Default 3000 is validated for this yeast
    genome; pass something larger for organisms with bigger real introns.

    flag_intron_threshold: forwarded to align_reads -- tags (doesn't drop)
    reads whose longest N op still exceeds this length despite the -G cap,
    since that cap is a soft minimap2 chaining heuristic and a residual few
    percent of junctions slip past it on real data. See
    doradoAligner_AtoG.merge_dual_pathway_alignments's docstring.

    threads: forwarded to align_reads. Default (3) is conservative for a
    shared machine; raise it when there's known headroom (idle cores, free
    memory) -- safe to increase without multiplying memory usage, unlike
    running multiple libraries through this pipeline concurrently.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Built once regardless of sharding -- it doesn't depend on the shard,
    # and building it before alignment would otherwise sit in memory
    # needlessly throughout the (already memory-heavy) alignment step.
    barcode_lookup = None

    if not shard_size:
        if out_bam is None:
            out_bam = str(output_dir / f"{Path(reads_bam).stem}_aligned.bam")

        print(f"=== Step 1/2: aligning {reads_bam} ===")
        aligned_bam = align_reads(reads_bam, ref_fasta, out_bam,
                                   keep_intermediates=keep_intermediates,
                                   gtf=gtf if use_junc_bed else None,
                                   splice_aware=splice_aware, max_intron_length=max_intron_length,
                                   flag_intron_threshold=flag_intron_threshold, threads=threads)

        barcode_lookup = build_barcode_lookup(barcode_summary) if barcode_summary else None

        print(f"\n=== Step 2/2: generating parquet from {aligned_bam} ===")
        total = generate_parquet(aligned_bam, ref_fasta, output_dir, gtf,
                                  coding_only=coding_only, chunk_size=chunk_size,
                                  barcode_lookup=barcode_lookup)

        return aligned_bam, total

    # Sharded path: barcode lookup is shared across shards, built once up
    # front since it doesn't grow with shard count and every shard needs it.
    barcode_lookup = build_barcode_lookup(barcode_summary) if barcode_summary else None

    shard_dir = output_dir / f"{Path(reads_bam).stem}_shards"
    total = 0
    shard_paths = []
    shard_aligned_bams = []

    for i, shard_path in enumerate(shard_reads_bam(reads_bam, shard_size, shard_dir)):
        print(f"\n=== Shard {i}: {shard_path} ===")
        shard_out_bam = str(shard_dir / f"{shard_path.stem}_aligned.bam")

        print(f"--- Step 1/2: aligning {shard_path} ---")
        aligned_bam = align_reads(str(shard_path), ref_fasta, shard_out_bam,
                                   keep_intermediates=keep_intermediates,
                                   gtf=gtf if use_junc_bed else None,
                                   splice_aware=splice_aware, max_intron_length=max_intron_length,
                                   flag_intron_threshold=flag_intron_threshold, threads=threads)

        print(f"--- Step 2/2: generating parquet from {aligned_bam} ---")
        shard_total = generate_parquet(aligned_bam, ref_fasta, output_dir, gtf,
                                        coding_only=coding_only, chunk_size=chunk_size,
                                        barcode_lookup=barcode_lookup)
        total += shard_total
        shard_paths.append(shard_path)
        shard_aligned_bams.append(aligned_bam)
        print(f"  Shard {i} done: {shard_total} reads (running total: {total})")

    # Concatenate every shard's aligned bam into one combined, sorted+indexed
    # bam -- the same final path the non-sharded path above would have used
    # -- then clean up the now-redundant per-shard raw and aligned bams.
    # This always happens (independent of keep_intermediates, which only
    # controls align_reads' own internal per-shard intermediates above):
    # the merged bam is the one artifact worth keeping around afterward,
    # and having to hunt through N per-shard bams instead of one defeats
    # the point of keeping intermediates in the first place.
    if out_bam is None:
        out_bam = str(output_dir / f"{Path(reads_bam).stem}_aligned.bam")
    print(f"\n=== Merging {len(shard_aligned_bams)} shard bam(s) into {out_bam} ===")
    merged_bam = merge_shard_bams(shard_aligned_bams, out_bam)

    for p in shard_paths + shard_aligned_bams:
        for path in (Path(p), Path(str(p) + ".bai")):
            if path.exists():
                path.unlink()
    if shard_dir.exists() and not any(shard_dir.iterdir()):
        shard_dir.rmdir()

    return merged_bam, total


def main():
    parser = argparse.ArgumentParser(
        description="Align a raw dorado bam and convert straight to parquet."
    )
    parser.add_argument('--reads_bam',  required=True, help='Input unaligned bam file from Dorado basecalling')
    parser.add_argument('--ref_fasta',  required=True, help='Reference fasta file')
    parser.add_argument('--gtf',        required=True, help='GTF annotation file for strand determination')
    parser.add_argument('--output_dir', required=True, help='Output directory for parquet chunks')
    parser.add_argument('--out_bam',    default=None,
                        help='Path for the intermediate aligned bam '
                             '(default: <output_dir>/<reads_bam_stem>_aligned.bam)')
    parser.add_argument('--coding_only', action='store_true',
                        help='Only write reads assigned to protein-coding genes')
    parser.add_argument('--chunk_size', type=int, default=50000,
                        help='Rows per output parquet chunk (default: 50000)')
    parser.add_argument('--remove_intermediates', action='store_true',
                        help='Delete per-run intermediate alignment bams instead of keeping them '
                             '(the default is to keep them, so a downstream-looking-wrong '
                             'parquet can be traced back to the exact bam/CIGAR that produced '
                             'it -- pass this flag to opt back into the old cleanup behavior, '
                             'e.g. for very large runs where disk space matters more).')
    parser.add_argument('--barcode_summary', default=None,
                        help='Path to a MinKNOW sequencing_summary.txt with a '
                             'barcode_arrangement column (native barcoding kit runs). '
                             'When given, output is split into output_dir/<barcode>/ '
                             'subdirectories (unmatched reads go to output_dir/unclassified/).')
    parser.add_argument('--shard_size', type=int, default=None,
                        help='Read count per shard. doradoAligner_AtoG holds every read\'s '
                             'full sequence in memory twice for the whole alignment step, '
                             'which OOMs on tens-of-millions-of-read runs; sharding bounds '
                             'peak memory to one shard at a time (try 2000000 as a starting '
                             'point). Omit for the original single-pass behavior.')
    parser.add_argument('--splice_aware', action='store_true',
                        help='Align reads_bam as spliced mRNA/cDNA that may span introns, '
                             'instead of plain genomic DNA. See '
                             'doradoAligner_AtoG.minimap2_align_splice_aware\'s docstring for '
                             'why this uses a validated max-intron-length cap via dorado\'s '
                             'own bundled minimap2 binary rather than plain `dorado aligner '
                             '-x splice` (which was confirmed to fabricate huge numbers of '
                             'false splice junctions on real data).')
    parser.add_argument('--no_junc_bed', action='store_true',
                        help='Only relevant with --splice_aware. Skip building/using a '
                             '--junc-bed from --gtf. By default the GTF is converted to a '
                             'junction BED12 and passed to minimap2 to bias splice-site '
                             'placement toward real annotated junctions.')
    parser.add_argument('--max_intron_length', type=int, default=3000,
                        help='Only relevant with --splice_aware: cap (bp) on minimap2 -G. '
                             'Default 3000 is validated for this yeast genome; pass something '
                             'larger for organisms with bigger real introns.')
    parser.add_argument('--flag_intron_threshold', type=int, default=2500,
                        help='Tag (not drop) reads whose longest N op exceeds this length '
                             '(bp) with XJ:i:<length>, since --max_intron_length\'s -G cap is '
                             'a soft chaining heuristic in minimap2 and a residual few percent '
                             'of junctions still slip past it on real data. Default 2500 '
                             'matches this genome\'s real annotated max intron (2483bp). Pass '
                             '0 (or a negative value) to disable tagging.')
    parser.add_argument('--threads', type=int, default=3,
                        help='Threads for the alignment step (default 3, conservative for a '
                             'shared machine). Raise this when there\'s known headroom.')
    args = parser.parse_args()

    flag_intron_threshold = args.flag_intron_threshold if args.flag_intron_threshold > 0 else None
    aligned_bam, total = run_pipeline(
        args.reads_bam, args.ref_fasta, args.gtf, args.output_dir,
        out_bam=args.out_bam, coding_only=args.coding_only, chunk_size=args.chunk_size,
        keep_intermediates=not args.remove_intermediates, barcode_summary=args.barcode_summary,
        shard_size=args.shard_size, splice_aware=args.splice_aware,
        use_junc_bed=not args.no_junc_bed, max_intron_length=args.max_intron_length,
        flag_intron_threshold=flag_intron_threshold, threads=args.threads)

    print(f"\nPipeline complete: {total} reads written from {aligned_bam}")


if __name__ == '__main__':
    main()
