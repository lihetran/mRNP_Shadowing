"""
incremental_pca_pipeline.py
----------------------------
Full out-of-core PCA pipeline adapted from:
  - parsePickleFileForPCA()
  - formatForPCA2()
  - smoothAndImpute()
  - doPCAandPlot()

Input:  directory of raw parquet chunks produced by shadowingBamToParquet.py
        (columns: read_id, edit_string, barcode, ref_sequence_aligned,
                  aligned_pairs, absolute_indices, ...)

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
"""

from pathlib import Path
import ast

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA


# ═══════════════════════════════════════════════════════════════════════════════
# PASS 1 — parse raw parquets → edit-matrix parquets
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_read_row(row, max_abs_idx=695):
    """
    Replicate parsePickleFileForPCA logic for a single row of the raw parquet.

    IMPORTANT: only stores positions where ref base == 'A', matching the
    original code which used is_A to filter the edit vector and edit freq.

    Returns dict: {abs_position: (edit_int, is_A_int)} or None if unusable.
    """
    edit_string = row["edit_string"]
    ref_aligned = row["ref_sequence_aligned"]
    abs_indices = row["absolute_indices"]

    # Normalise absolute_indices to a plain Python list of scalars.
    if isinstance(abs_indices, str):
        abs_indices = ast.literal_eval(abs_indices)
    elif hasattr(abs_indices, "tolist"):   # numpy array / pandas Series
        abs_indices = abs_indices.tolist()

    positDict = {}
    for ii, abs_idx in enumerate(abs_indices):
        # Skip None and float NaN
        if abs_idx is None:
            continue
        try:
            if np.isnan(float(abs_idx)):
                continue
        except (TypeError, ValueError):
            continue
        abs_idx = int(abs_idx)
        if abs_idx > max_abs_idx:
            continue
        if ii >= len(edit_string) or ii >= len(ref_aligned):
            continue
        edit = edit_string[ii]
        if edit == "2":
            continue
        seq = ref_aligned[ii].upper()
        is_A = 1 if seq == "A" else 0
        # ── KEY FIX: only keep A positions in the edit vector ──────────────
        # Original parsePickleFileForPCA stored both A and non-A positions,
        # but formatForPCA2 built edit vectors using v[0] for ALL positions.
        # However, the meaningful editing signal (A→G) only exists at A sites.
        # Keeping non-A positions adds noise and dilutes explained variance.
        if is_A == 0:
            continue
        positDict[abs_idx] = (int(edit), is_A)

    return positDict if positDict else None


def _edit_freq(positDict):
    """Replicate the minEditFreq filter from formatForPCA2."""
    denom = sum(v[1] for v in positDict.values())
    if denom == 0:
        return 0.0
    return sum(v[0] for v in positDict.values()) / denom


def parse_to_edit_matrix(
    raw_parquet_dir,
    edit_matrix_dir,
    num_reads=100,
    min_edit_freq=0.01,
    max_abs_idx=695,
    chunk_size=50000,
    window_start=None,
    window_end=None,
):
    """
    PASS 1: Stream raw parquets, apply parsePickleFileForPCA + formatForPCA2
    logic, write edit-matrix parquets.

    Each output parquet has:
      - one column per genomic position (named by abs_idx, e.g. "123")
      - a 'barcode' column
      - a 'read_id' column
    Values are float32 edit scores (0/1/NaN).

    window_start / window_end : optional genomic position bounds to restrict
    the feature matrix to a sub-region. If None, all positions are used.

    Returns
    -------
    positions : sorted list of int genomic positions (defines column order)
    edit_matrix_dir : Path
    """
    raw_parquet_dir  = Path(raw_parquet_dir)
    edit_matrix_dir  = Path(edit_matrix_dir)
    edit_matrix_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_parquet_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No parquet files found in {raw_parquet_dir}")

    # Resolve window bounds — fall back to full range if not specified
    effective_min = window_start if window_start is not None else 0
    effective_max = window_end   if window_end   is not None else max_abs_idx
    if window_start is not None or window_end is not None:
        print(f"[parse] genomic window: {effective_min} – {effective_max}")

    print(f"[parse] {len(files)} raw parquet file(s)")

    # ── Step 1: collect per-barcode read dicts (streaming) ──────────────────
    # We need num_reads per barcode, selected by length (replicating formatForPCA2).
    # We accumulate a lightweight per-BC structure: {bc: [(readID, positDict), ...]}
    from collections import defaultdict
    bc_reads = defaultdict(list)  # bc → list of (readID, positDict)

    for i, fpath in enumerate(files, 1):
        pf = pq.ParquetFile(fpath)
        print(f"  [{i}/{len(files)}] parsing {fpath.name}")
        for rg in range(pf.metadata.num_row_groups):
            df = pf.read_row_group(rg).to_pandas()
            for _, row in df.iterrows():
                bc      = row.get("barcode", None)
                read_id = row.get("read_id", None)
                if bc is None:
                    continue
                positDict = _parse_read_row(row, max_abs_idx=max_abs_idx)
                if positDict is None:
                    continue
                if _edit_freq(positDict) < min_edit_freq:
                    continue
                bc_reads[bc].append((read_id, positDict))

    print(f"[parse] {len(bc_reads)} barcodes found")

    # ── Step 2: select top-N reads per barcode (original formatForPCA2 logic) ─
    # Uses value membership test on the top-N lengths, not a hard slice.
    # This means reads tied with the Nth longest are all included, matching
    # the original behaviour where more than num_reads can be selected per bc.
    selected = []  # list of (readID, bc, positDict)
    for bc, read_list in bc_reads.items():
        # Get sorted lengths descending (mirrors original temp list)
        lengths = sorted([len(r[1]) for r in read_list], reverse=True)
        # The length threshold: value of the num_reads-th longest read
        length_threshold = lengths[:num_reads]
        for read_id, positDict in read_list:
            if len(positDict) in length_threshold:
                selected.append((read_id, bc, positDict))

    print(f"[parse] {len(selected)} reads selected after filtering")

    # ── Step 3: determine global position set ────────────────────────────────
    raw_positions = set()
    for _, _, pd_ in selected:
        for pos in pd_:
            try:
                if pos is None or np.isnan(float(pos)):
                    print(f"[warn] dropping NaN/None key: {pos!r}")
                    continue
            except (TypeError, ValueError):
                print(f"[warn] dropping unconvertible key: {pos!r}")
                continue
            ipos = int(pos)
            # Apply window AFTER edit freq filtering — reads selected globally,
            # features restricted to window
            if ipos < effective_min or ipos > effective_max:
                continue
            raw_positions.add(ipos)
    positions = sorted(raw_positions)
    print(f"[parse] {len(positions)} unique genomic positions")

    # Paranoia check — duplicate column names will crash parquet write
    pos_str_check = ["read_id", "barcode"] + [str(p) for p in positions]
    if len(pos_str_check) != len(set(pos_str_check)):
        from collections import Counter
        dupes = [k for k, v in Counter(pos_str_check).items() if v > 1]
        raise RuntimeError(f"Duplicate column names after cleaning: {dupes}")

    # Save position index for later passes
    np.save(edit_matrix_dir / "positions.npy", np.array(positions))

    # ── Step 4: write edit-matrix parquets ───────────────────────────────────
    pos_str = [str(p) for p in positions]  # column names must be strings
    out_files = []
    chunk_idx = 0

    for start in range(0, len(selected), chunk_size):
        batch = selected[start : start + chunk_size]
        rows  = []
        for read_id, bc, positDict in batch:
            vec = [float(positDict[p][0]) if p in positDict else np.nan
                   for p in positions]
            rows.append([read_id, bc] + vec)

        cols = ["read_id", "barcode"] + pos_str
        df   = pd.DataFrame(rows, columns=cols)
        out  = edit_matrix_dir / f"edit_matrix_chunk{chunk_idx}.parquet"
        df.to_parquet(out, compression="zstd", index=False)
        out_files.append(out)
        print(f"  wrote {out.name}  ({len(df)} rows)")
        chunk_idx += 1

    print(f"[parse] done — {chunk_idx} edit-matrix parquet(s)\n")
    return positions, edit_matrix_dir


# ═══════════════════════════════════════════════════════════════════════════════
# PASS 2 — compute per-column means (Welford's online algorithm)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_column_stats(edit_matrix_dir, positions):
    """
    PASS 2: Stream edit-matrix parquets, compute per-column mean ignoring NaNs
    using Welford's online algorithm.

    Returns
    -------
    col_means : np.ndarray of shape (n_positions,)
    """
    edit_matrix_dir = Path(edit_matrix_dir)
    files = sorted(edit_matrix_dir.glob("edit_matrix_chunk*.parquet"))
    pos_str = [str(p) for p in positions]

    # Welford accumulators
    n_obs  = np.zeros(len(positions), dtype=np.float64)
    means  = np.zeros(len(positions), dtype=np.float64)

    print("[stats] computing per-column means (Welford) …")
    for fpath in files:
        df = pd.read_parquet(fpath, columns=pos_str)
        arr = df.to_numpy(dtype=np.float64)
        for row in arr:
            for j, val in enumerate(row):
                if not np.isnan(val):
                    n_obs[j]  += 1
                    delta      = val - means[j]
                    means[j]  += delta / n_obs[j]

    # Any column never observed → impute with 0
    means[n_obs == 0] = 0.0
    np.save(edit_matrix_dir / "col_means.npy", means)
    print(f"[stats] done — {len(positions)} column means computed\n")
    return means


# ═══════════════════════════════════════════════════════════════════════════════
# shared smooth+impute helper (chunk-aware, seam-padded)
# ═══════════════════════════════════════════════════════════════════════════════

SMOOTH_WINDOW = 5
SEAM_PAD      = SMOOTH_WINDOW // 2   # 2 rows carried between chunks


def _smooth_and_impute_chunk(arr: np.ndarray, col_means: np.ndarray,
                              prev_tail: np.ndarray | None):
    """
    Apply rolling smooth (window=5) + NaN imputation to a chunk.

    To avoid boundary artefacts at chunk seams we prepend the last SEAM_PAD
    rows of the previous chunk before smoothing, then strip them off again.

    Parameters
    ----------
    arr       : (n_rows, n_cols) float64 array, may contain NaNs
    col_means : (n_cols,) precomputed column means for imputation
    prev_tail : last SEAM_PAD rows of the previous chunk, or None

    Returns
    -------
    smoothed  : (n_rows, n_cols) float64, no NaNs
    new_tail  : last SEAM_PAD rows of arr (unsmoothed) for next call
    """
    new_tail = arr[-SEAM_PAD:].copy()

    if prev_tail is not None:
        padded = np.vstack([prev_tail, arr])
    else:
        padded = arr

    df = pd.DataFrame(padded)
    smoothed_full = df.rolling(window=SMOOTH_WINDOW, min_periods=1,
                               center=True).mean().to_numpy(dtype=np.float64)

    # Strip the padding rows back off
    if prev_tail is not None:
        smoothed = smoothed_full[SEAM_PAD:]
    else:
        smoothed = smoothed_full

    # Impute remaining NaNs with precomputed column means
    nan_mask = np.isnan(smoothed)
    smoothed[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    return smoothed, new_tail


# ═══════════════════════════════════════════════════════════════════════════════
# PASS 3 — fit IncrementalPCA
# ═══════════════════════════════════════════════════════════════════════════════

def fit_incremental_pca(edit_matrix_dir, positions, col_means,
                        n_components=5, dtype=np.float32):
    """
    PASS 3: Stream edit-matrix parquets → smooth/impute → partial_fit().
    Returns fitted IncrementalPCA object.
    """
    edit_matrix_dir = Path(edit_matrix_dir)
    files   = sorted(edit_matrix_dir.glob("edit_matrix_chunk*.parquet"))
    pos_str = [str(p) for p in positions]

    ipca     = IncrementalPCA(n_components=n_components)
    prev_tail = None
    total_rows = 0

    print("[fit] fitting IncrementalPCA …")
    for fpath in files:
        df  = pd.read_parquet(fpath, columns=pos_str)
        arr = df.to_numpy(dtype=np.float64)

        smoothed, prev_tail = _smooth_and_impute_chunk(arr, col_means, prev_tail)
        smoothed = smoothed.astype(dtype)

        if smoothed.shape[0] < n_components:
            print(f"  skipping {fpath.name} — too few rows ({smoothed.shape[0]})")
            continue

        ipca.partial_fit(smoothed)
        total_rows += smoothed.shape[0]
        print(f"  partial_fit on {fpath.name}  ({smoothed.shape[0]} rows, "
              f"{total_rows} total)")

    joblib.dump(ipca, edit_matrix_dir / "incremental_pca_model.joblib")
    np.save(edit_matrix_dir / "explained_variance_ratio.npy",
            ipca.explained_variance_ratio_)
    np.save(edit_matrix_dir / "components.npy", ipca.components_)

    total_var = ipca.explained_variance_ratio_.sum() * 100
    print(f"[fit] done — {total_var:.2f}% variance explained\n")
    return ipca


# ═══════════════════════════════════════════════════════════════════════════════
# PASS 4 — transform
# ═══════════════════════════════════════════════════════════════════════════════

def transform_incremental_pca(ipca, edit_matrix_dir, positions, col_means,
                               output_dir, dtype=np.float32):
    """
    PASS 4: Stream edit-matrix parquets → smooth/impute → transform() → save.

    Returns
    -------
    pc_files  : list of Path to *_pcs.npy
    all_bcs   : np.ndarray of barcode labels (same row order as pc_files stacked)
    """
    edit_matrix_dir = Path(edit_matrix_dir)
    output_dir      = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files   = sorted(edit_matrix_dir.glob("edit_matrix_chunk*.parquet"))
    pos_str = [str(p) for p in positions]

    pc_files  = []
    all_bcs   = []
    prev_tail = None

    print("[transform] projecting into PC space …")
    for fpath in files:
        df      = pd.read_parquet(fpath, columns=["barcode"] + pos_str)
        bcs     = df["barcode"].to_numpy()
        arr     = df[pos_str].to_numpy(dtype=np.float64)

        smoothed, prev_tail = _smooth_and_impute_chunk(arr, col_means, prev_tail)
        smoothed = smoothed.astype(dtype)

        if smoothed.shape[0] < ipca.n_components:
            print(f"  skipping {fpath.name} — too few rows")
            continue

        pcs = ipca.transform(smoothed)
        out = output_dir / f"{fpath.stem}_pcs.npy"
        np.save(out, pcs)

        pc_files.append(out)
        all_bcs.append(bcs)
        print(f"  {fpath.name} → {out.name}  shape={pcs.shape}")

    all_bcs = np.concatenate(all_bcs)
    np.save(output_dir / "barcodes.npy", all_bcs)

    print(f"[transform] done — {len(pc_files)} file(s), "
          f"{len(all_bcs)} total reads\n")
    return pc_files, all_bcs


# ═══════════════════════════════════════════════════════════════════════════════
# plotting  (your original doPCAandPlot logic, adapted)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pca(pc_files, out_prefix, labels=None, n_pairs=4):
    """
    Scatter plots of consecutive PC pairs (PC1v2, PC2v3, …).

    Parameters
    ----------
    pc_files   : list of paths to *_pcs.npy (output of transform_incremental_pca)
    out_prefix : file prefix; plots saved as <out_prefix>.PCA.<i>.<j>.png
    labels     : 1-D array of per-row label strings, or None (all one colour)
    n_pairs    : how many consecutive pairs to plot (default 4)
    """
    X_pca = np.vstack([np.load(f) for f in pc_files])
    n_components = X_pca.shape[1]
    n_pairs = min(n_pairs, n_components - 1)

    pc_cols     = {f"PC{i+1}": X_pca[:, i] for i in range(n_pairs + 1)}
    X_pca_df    = pd.DataFrame(pc_cols)
    X_pca_df["label"] = np.asarray(labels) if labels is not None else "data"

    unique_labels = X_pca_df["label"].unique()
    print(f"[plot]  {n_pairs} plot(s), {len(unique_labels)} label(s)")

    for ii in range(n_pairs):
        fig, ax = plt.subplots(figsize=(6, 6))
        for label in unique_labels:
            subset = X_pca_df[X_pca_df["label"] == label]
            ax.scatter(subset[f"PC{ii+1}"], subset[f"PC{ii+2}"],
                       label=label, s=10, alpha=0.7)

        ax.set_title("PCA of Binary Vectors")
        ax.set_xlabel(f"PCA Dimension {ii+1}")
        ax.set_ylabel(f"PCA Dimension {ii+2}")
        ax.legend(markerscale=2, bbox_to_anchor=(1.01, 1), loc="upper left",
                  borderaxespad=0, fontsize=8)

        out_path = f"{out_prefix}.PCA.{ii}.{ii+1}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# top-level convenience wrapper
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    raw_parquet_dir,
    edit_matrix_dir,
    output_dir,
    num_reads     = 100,
    min_edit_freq = 0.01,
    max_abs_idx   = 695,
    n_components  = 5,
    chunk_size    = 50000,
    plot          = True,
    window_start  = None,
    window_end    = None,
):
    """
    Run the full 4-pass incremental PCA pipeline.

    Parameters
    ----------
    raw_parquet_dir : directory of parquets from shadowingBamToParquet.py
    edit_matrix_dir : intermediate directory for edit-matrix parquets
    output_dir      : final outputs (pcs, model, plots)
    num_reads       : max reads per barcode (mirrors formatForPCA2)
    min_edit_freq   : minimum A→G edit fraction to keep a read
    max_abs_idx     : upper bound on genomic position (default 695)
    n_components    : number of PCs
    chunk_size      : rows per edit-matrix parquet chunk
    plot            : whether to generate scatter plots
    window_start    : optional start of genomic position window (inclusive)
    window_end      : optional end of genomic position window (inclusive)
    """
    # Pass 1
    positions, edit_matrix_dir = parse_to_edit_matrix(
        raw_parquet_dir, edit_matrix_dir,
        num_reads=num_reads, min_edit_freq=min_edit_freq,
        max_abs_idx=max_abs_idx, chunk_size=chunk_size,
        window_start=window_start, window_end=window_end,
    )

    # Pass 2
    col_means = compute_column_stats(edit_matrix_dir, positions)

    # Pass 3
    ipca = fit_incremental_pca(
        edit_matrix_dir, positions, col_means, n_components=n_components
    )

    # Pass 4
    pc_files, barcodes = transform_incremental_pca(
        ipca, edit_matrix_dir, positions, col_means, output_dir
    )

    # Plot (barcodes as labels)
    if plot:
        if window_start is not None or window_end is not None:
            win_str = f"window_{window_start or 0}_{window_end or max_abs_idx}"
        else:
            win_str = "all_positions"
        plot_pca(
            pc_files,
            out_prefix=str(Path(output_dir) / f"result_{win_str}"),
            labels=barcodes,
        )

    return ipca, pc_files, barcodes
    return ipca, pc_files, barcodes


# ═══════════════════════════════════════════════════════════════════════════════
# example
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_pipeline(
        raw_parquet_dir = "./parquets",
        edit_matrix_dir = "./edit_matrix",
        output_dir      = "./pca_output",
        num_reads       = 100,
        min_edit_freq   = 0.01,
        max_abs_idx     = 695,
        n_components    = 5,
    )