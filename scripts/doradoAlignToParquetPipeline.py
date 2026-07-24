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
      [--out_bam aligned.bam] [--coding_only] [--chunk_size 50000]

  from doradoAlignToParquetPipeline import run_pipeline
  run_pipeline(reads_bam="...", ref_fasta="...", gtf="...", output_dir="...")
"""
import argparse
from pathlib import Path

from doradoAligner_AtoG import align_reads
from shadowingBamToParquetWithGTF2 import generate_parquet


def run_pipeline(reads_bam, ref_fasta, gtf, output_dir,
                  out_bam=None, coding_only=False, chunk_size=50000):
    """
    Align reads_bam against ref_fasta (dual A->G/T->C pathway), then convert
    the resulting aligned bam straight into parquet chunks under output_dir.
    Returns (aligned_bam_path, total_reads_written).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if out_bam is None:
        out_bam = str(output_dir / f"{Path(reads_bam).stem}_aligned.bam")

    print(f"=== Step 1/2: aligning {reads_bam} ===")
    aligned_bam = align_reads(reads_bam, ref_fasta, out_bam)

    print(f"\n=== Step 2/2: generating parquet from {aligned_bam} ===")
    total = generate_parquet(aligned_bam, ref_fasta, output_dir, gtf,
                              coding_only=coding_only, chunk_size=chunk_size)

    return aligned_bam, total


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
    args = parser.parse_args()

    aligned_bam, total = run_pipeline(
        args.reads_bam, args.ref_fasta, args.gtf, args.output_dir,
        out_bam=args.out_bam, coding_only=args.coding_only, chunk_size=args.chunk_size)

    print(f"\nPipeline complete: {total} reads written from {aligned_bam}")


if __name__ == '__main__':
    main()
