"""
run_rare_read_pca.py
---------------------
Command-line companion to rare_read_pca_pipeline.py

Usage:
  # Run full pipeline
  python run_rare_read_pca.py \\
      --raw_parquet_dir ./parquets \\
      --edit_matrix_dir ./edit_matrix \\
      --output_dir      ./pca_output \\
      --min_edit_freq   0.8 \\
      --n_components    5

  # Plot edit frequency CDF only (no PCA)
  python run_rare_read_pca.py \\
      --raw_parquet_dir ./parquets \\
      --output_dir      ./pca_output \\
      --plot_cdf
"""

import argparse
from pathlib import Path
from rare_read_pca_pipeline import run_pipeline, plot_edit_freq_cdf


def main():
    parser = argparse.ArgumentParser(
        description="Run rare-read PCA pipeline on BAM-derived parquet files."
    )
    parser.add_argument("--raw_parquet_dir", type=str, required=True,
                        help="Directory containing raw parquet files from shadowingBamToParquet.py")
    parser.add_argument("--edit_matrix_dir", type=str, default=None,
                        help="Directory for intermediate edit-matrix parquets. "
                             "Required unless --plot_cdf is set.")
    parser.add_argument("--output_dir",      type=str, required=True,
                        help="Directory for final outputs (PCs, plots)")
    parser.add_argument("--min_edit_freq",   type=float, default=0.8,
                        help="Minimum A→G edit frequency to keep a read (default: 0.8)")
    parser.add_argument("--max_abs_idx",     type=int, default=695,
                        help="Upper bound on genomic position index (default: 695)")
    parser.add_argument("--n_components",    type=int, default=5,
                        help="Number of PCA components (default: 5)")
    parser.add_argument("--chunk_size",      type=int, default=50000,
                        help="Rows per edit-matrix parquet chunk (default: 50000)")
    parser.add_argument("--window_start",    type=int, default=None,
                        help="Start of genomic position window, inclusive "
                             "(default: None = use all positions)")
    parser.add_argument("--window_end",      type=int, default=None,
                        help="End of genomic position window, inclusive "
                             "(default: None = use all positions)")
    parser.add_argument("--barcodes",        nargs="+", default=None,
                        help="Barcodes to include (default: all). "
                             "E.g. --barcodes bc1 bc2 bc3")
    parser.add_argument("--no_plot",         action="store_true",
                        help="Skip PCA scatter plot generation")
    parser.add_argument("--plot_cdf",        action="store_true",
                        help="Plot edit frequency CDF only — skip PCA entirely. "
                             "--edit_matrix_dir is not required in this mode.")

    args = parser.parse_args()

    if not args.plot_cdf and args.edit_matrix_dir is None:
        parser.error("--edit_matrix_dir is required unless --plot_cdf is set")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_prefix = str(Path(args.output_dir) / "result")

    if args.plot_cdf:
        plot_edit_freq_cdf(
            raw_parquet_dir = args.raw_parquet_dir,
            out_prefix      = out_prefix,
            barcodes        = args.barcodes,
            min_edit_freq   = args.min_edit_freq,
        )
        return

    ipca, pc_files, all_bcs = run_pipeline(
        raw_parquet_dir = args.raw_parquet_dir,
        edit_matrix_dir = args.edit_matrix_dir,
        output_dir      = args.output_dir,
        min_edit_freq   = args.min_edit_freq,
        max_abs_idx     = args.max_abs_idx,
        n_components    = args.n_components,
        chunk_size      = args.chunk_size,
        plot            = not args.no_plot,
        window_start    = args.window_start,
        window_end      = args.window_end,
        barcodes        = args.barcodes,
    )

    print(f"\nDone. {len(pc_files)} PC file(s), {len(all_bcs)} total reads.")
    print(f"Explained variance: {ipca.explained_variance_ratio_}")
    print(f"Total variance explained: {ipca.explained_variance_ratio_.sum()*100:.2f}%")


if __name__ == "__main__":
    main()