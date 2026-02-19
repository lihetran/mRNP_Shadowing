'''
February 19, 2026 LT

Companion script for running incremental_pca_pipeline.py.

Pipeline:
  PASS 1 — parse_to_edit_matrix()
            Stream raw parquets → reconstruct per-read edit vectors →
            write intermediate "edit matrix" parquets
            (columns = genomic positions, rows = reads; also saves barcode list)

  PASS 2 — compute_column_stats()
            Stream edit-matrix parquets → compute per-column mean via
            Welford's online algorithm (exact, no approximation)

  PASS 3 — fit_incremental_pca()
            Stream edit-matrix parquets → smooth (window=5, seam-padded) →
            impute NaNs with precomputed column means → partial_fit()

  PASS 4 — transform_incremental_pca()
            Same streaming + smooth/impute → transform() → save *_pcs.npy

  plot_pca()
            Load *_pcs.npy files → scatter plots of consecutive PC pairs

Usage:
  from incremental_pca_pipeline import run_pipeline
  run_pipeline(
      raw_parquet_dir = "./parquets",
      edit_matrix_dir = "./edit_matrix",
      output_dir      = "./pca_output",
      num_reads       = 100,
      min_edit_freq   = 0.01,
      n_components    = 5,
      max_abs_idx     = 695,
  )

inputs:
    raw_parquet_dir: directory containing raw parquet files (output of shadowingBamToParquet.py)
    edit_matrix_dir: directory to save intermediate "edit matrix" parquets
    output_dir: directory to save final PCA outputs (e.g. *_pcs.npy)
    num_reads: number of reads to process (for testing; set to None for all)
    min_edit_freq: minimum edit frequency to retain a column (e.g. 0.01 for 1%)
    n_components: number of PCA components to compute
    max_abs_idx: maximum absolute genomic index to consider (to limit dimensionality)
'''

import argparse
from incremental_pca_pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="Run incremental PCA pipeline on BAM-derived parquet files."
    )
    parser.add_argument("--raw_parquet_dir", type=str, required=True,
                        help="Directory containing raw parquet files from shadowingBamToParquet.py")
    parser.add_argument("--edit_matrix_dir", type=str, required=True,
                        help="Directory for intermediate edit-matrix parquets (can delete after run)")
    parser.add_argument("--output_dir",      type=str, required=True,
                        help="Directory for final PCA outputs (model, PCs, plots)")
    parser.add_argument("--num_reads",       type=int, default=50,
                        help="Max reads per barcode to select (default: 50)")
    parser.add_argument("--min_edit_freq",   type=float, default=0.8,
                        help="Minimum A→G edit frequency to keep a read (default: 0.8)")
    parser.add_argument("--n_components",    type=int, default=5,
                        help="Number of PCA components (default: 5)")
    parser.add_argument("--max_abs_idx",     type=int, default=695,
                        help="Upper bound on genomic position index (default: 695)")
    parser.add_argument("--chunk_size",      type=int, default=50000,
                        help="Rows per edit-matrix parquet chunk (default: 50000)")
    parser.add_argument("--no_plot",         action="store_true",
                        help="Skip scatter plot generation")
    parser.add_argument("--window_start",    type=int, default=None,
                        help="Start of genomic position window, inclusive (default: None = use all positions)")
    parser.add_argument("--window_end",      type=int, default=None,
                        help="End of genomic position window, inclusive (default: None = use all positions)")

    args = parser.parse_args()

    ipca, pc_files, barcodes = run_pipeline(
        raw_parquet_dir = args.raw_parquet_dir,
        edit_matrix_dir = args.edit_matrix_dir,
        output_dir      = args.output_dir,
        num_reads       = args.num_reads,
        min_edit_freq   = args.min_edit_freq,
        n_components    = args.n_components,
        max_abs_idx     = args.max_abs_idx,
        chunk_size      = args.chunk_size,
        plot            = not args.no_plot,
        window_start    = args.window_start,
        window_end      = args.window_end,
    )

    print(f"\nDone. {len(pc_files)} PC file(s), {len(barcodes)} total reads.")
    print(f"Explained variance: {ipca.explained_variance_ratio_}")
    print(f"Total variance explained: {ipca.explained_variance_ratio_.sum()*100:.2f}%")


if __name__ == "__main__":
    main()
