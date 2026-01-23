'''
January 23, 2026 LT

This script will perform PCA on the indicator matrix generated from shadowingBamToIndicatorMatrix.py.
It reads the parquet files containing the indicator matrix, performs PCA, and saves the results. Functionality to do this in
chunks is included to handle large datasets.

input: parquet files with indicator matrix
       output_dir - directory to save PCA results
output: PCA results saved as numpy arrays and plots
'''

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import joblib

def incremental_pca(
    parquet_dir,
    output_dir,
    n_components=30,
    batch_size=50000,
    dtype=np.float32
):
    parquet_dir = Path(parquet_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(parquet_dir.glob("*.parquet"))

    if not files:
        raise ValueError("No parquet files found")

    print(f"Found {len(files)} parquet chunks")

    ipca = IncrementalPCA(
        n_components=n_components,
        batch_size=batch_size
    )

    # ------------------
    # FIT PCA
    # ------------------
    print("Fitting PCA incrementally...")

    for i, pq in enumerate(files, 1):
        X = pd.read_parquet(pq).astype(dtype).values
        ipca.partial_fit(X)

        print(f"  [{i}/{len(files)}] fitted {pq.name}")

    # ------------------
    # SAVE PCA MODEL
    # ------------------
    joblib.dump(ipca, output_dir / "incremental_pca_model.joblib")

    np.save(
        output_dir / "explained_variance_ratio.npy",
        ipca.explained_variance_ratio_
    )

    np.save(
        output_dir / "components.npy",
        ipca.components_
    )

    print("PCA fitting complete")

    # ------------------
    # TRANSFORM DATA
    # ------------------
    print("Projecting reads into PC space...")

    pc_files = []

    for pq in files:
        X = pd.read_parquet(pq).astype(dtype).values
        pcs = ipca.transform(X)

        out = output_dir / f"{pq.stem}_pcs.npy"
        np.save(out, pcs)

        pc_files.append(out.name)
        print(f"  wrote {out.name}")

    print("All done.")

    return ipca

def plot_scatter(pc_array, output_path, pc_x=1, pc_y=2, alpha=0.5, s=1):
    plt.figure(figsize=(8, 8))
    plt.scatter(
        pc_array[:, pc_x - 1],
        pc_array[:, pc_y - 1],
        alpha=alpha,
        s=s
    )
    plt.xlabel(f"PC{pc_x}")
    plt.ylabel(f"PC{pc_y}")
    plt.title(f"Scatter plot of PC{pc_x} vs PC{pc_y}")
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(output_path)
    plt.close()

def main(args):
    parquet_dir = args[0]
    output_dir = args[1]

    incremental_pca(
        parquet_dir,
        output_dir,
        n_components=30,
        batch_size=50000,
        dtype=np.float32
    )

if __name__ == '__main__':
    main(sys.argv[1:])