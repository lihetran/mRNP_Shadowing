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
    barcodes=None,
):
    """
    PASS 1: Three-sub-pass approach to build edit-matrix parquets
    without loading all reads into memory.

    Sub-pass 1a — vectorised filter using precomputed global_edit_freq column:
        Stream raw parquets, filter on global_edit_freq, write a lightweight
        summary parquet per barcode: (read_id, barcode, n_a_positions).
        Memory: O(chunk_size) at any time.

    Sub-pass 1b — top-N selection:
        Read summary parquets, select top-N read_ids per barcode by
        n_a_positions (mirrors original formatForPCA2 length-based selection).
        Memory: O(total_passing_reads) for summaries only — tiny since no
        positDicts are held.

    Sub-pass 1c — edit matrix construction:
        Stream raw parquets again, parse positDicts only for selected read_ids,
        write edit matrix rows on the fly. Memory: O(chunk_size * n_positions).

    window_start / window_end : optional genomic position bounds to restrict
    the feature matrix to a sub-region. Applied after global edit freq filter
    so read selection is always based on global editing behaviour.
    barcodes : optional list of barcode strings to include. If None, all
    barcodes present in the data are used.

    NOTE: requires 'global_edit_freq' and 'n_a_positions' columns in the raw
    parquets — produced by the updated shadowingBamToParquet.py.

    Returns
    -------
    positions : sorted list of int genomic positions (defines column order)
    edit_matrix_dir : Path
    """
    import heapq
    from collections import defaultdict

    raw_parquet_dir = Path(raw_parquet_dir)
    edit_matrix_dir = Path(edit_matrix_dir)
    edit_matrix_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_parquet_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No parquet files found in {raw_parquet_dir}")

    # Resolve window bounds
    effective_min = window_start if window_start is not None else 0
    effective_max = window_end   if window_end   is not None else max_abs_idx
    if window_start is not None or window_end is not None:
        print(f"[parse] genomic window: {effective_min} – {effective_max}")
    if barcodes is not None:
        print(f"[parse] barcode filter: {barcodes}")

    print(f"[parse] {len(files)} raw parquet file(s)")

    # ── Sub-pass 1a: barcode filter + edit freq via positDict parsing ──────────
    # global_edit_freq in the parquet is computed without the max_abs_idx cap,
    # so it systematically underestimates edit freq vs _edit_freq(positDict).
    # For correctness we compute edit freq from positDicts here, matching the
    # working version exactly. The vectorised barcode filter still provides a
    # meaningful speedup by skipping excluded barcodes before parsing.
    print("[parse 1a] filtering reads by barcode and edit frequency...")
    total_seen = 0
    total_passed = 0
    selected_bc  = {}   # read_id → barcode (passing reads only)

    using_window = window_start is not None or window_end is not None

    for i, fpath in enumerate(files, 1):
        pf = pq.ParquetFile(fpath)
        print(f"  [{i}/{len(files)}] {fpath.name}")
        for rg in range(pf.metadata.num_row_groups):
            df = pf.read_row_group(rg).to_pandas()
            total_seen += len(df)
            # Fast barcode pre-filter before expensive positDict parsing
            bc_mask = df["barcode"].notna()
            if barcodes is not None:
                bc_mask = bc_mask & df["barcode"].isin(barcodes)
            df_bc = df[bc_mask]
            for _, row in df_bc.iterrows():
                read_id = row["read_id"]
                bc      = row["barcode"]
                positDict = _parse_read_row(row, max_abs_idx=max_abs_idx)
                if positDict is None:
                    continue
                if _edit_freq(positDict) < min_edit_freq:
                    continue
                selected_bc[read_id] = bc
                total_passed += 1

    print(f"[parse 1a] {total_passed:,} / {total_seen:,} reads passed filter")

    selected_ids = set(selected_bc.keys())
    print(f"[parse 1b] {len(selected_ids):,} candidate reads across "
          f"{len(set(selected_bc.values()))} barcodes — selection deferred to 1c")

    # ── Sub-pass 1c: two streams for all cases ───────────────────────────────
    #
    # Stream 1 — parse positDicts for all candidates, record ranking length
    #            (len(positDict), capped by max_abs_idx), free positDict.
    #            When window active, also count A positions within window.
    #            Do top-N selection from these lengths.
    #            Memory: O(n_candidates) for length dict — just one int per read.
    #
    # Stream 2 — parse positDicts only for final selected reads, collect
    #            positions, write edit matrix.
    #            Memory: O(num_reads * barcodes * positions)

    # ── 1c stream 1: compute ranking lengths, select top-N ───────────────────
    print("[parse 1c-1] computing ranking lengths per read...")
    rank_count  = {}   # read_id → ranking length (len(positDict) or window A count)

    for fpath in files:
        pf = pq.ParquetFile(fpath)
        for rg in range(pf.metadata.num_row_groups):
            df = pf.read_row_group(rg).to_pandas()
            df_sel = df[df["read_id"].isin(selected_ids)]
            for _, row in df_sel.iterrows():
                read_id = row["read_id"]
                if read_id in rank_count:
                    continue
                positDict = _parse_read_row(row, max_abs_idx=max_abs_idx)
                if positDict is None:
                    rank_count[read_id] = 0
                    continue
                if using_window:
                    # Rank by A positions within window
                    rank_count[read_id] = sum(
                        1 for pos in positDict
                        if effective_min <= int(pos) <= effective_max
                    )
                else:
                    # Rank by total capped A positions — matches len(positDict)
                    # in the working version exactly
                    rank_count[read_id] = len(positDict)
                # positDict intentionally NOT stored — freed here

    # Top-N selection using value-membership, matching original formatForPCA2
    print("[parse 1c-1] selecting top-N reads per barcode...")
    final_ids     = set()
    bc_candidates = {}
    for read_id, bc in selected_bc.items():
        if read_id not in rank_count:
            continue
        bc_candidates.setdefault(bc, []).append((rank_count[read_id], read_id))

    for bc, candidates in bc_candidates.items():
        lengths          = sorted([c[0] for c in candidates], reverse=True)
        length_threshold = lengths[:num_reads]   # list, not set — matches original
        for count, read_id in candidates:
            if count in length_threshold:
                final_ids.add(read_id)

    del rank_count, bc_candidates
    selected_ids = final_ids
    print(f"[parse 1c-1] {len(selected_ids):,} reads selected across "
          f"{len(set(selected_bc.values()))} barcodes")

    # ── 1c stream 2: parse positDicts for selected reads only ────────────────
    print("[parse 1c-2] parsing positDicts for selected reads...")
    raw_positions   = set()
    selected_parsed = {}   # read_id → positDict

    for fpath in files:
        pf = pq.ParquetFile(fpath)
        for rg in range(pf.metadata.num_row_groups):
            df = pf.read_row_group(rg).to_pandas()
            df_sel = df[df["read_id"].isin(selected_ids)]
            for _, row in df_sel.iterrows():
                read_id = row["read_id"]
                if read_id in selected_parsed:
                    continue
                positDict = _parse_read_row(row, max_abs_idx=max_abs_idx)
                if positDict is None:
                    continue
                selected_parsed[read_id] = positDict
                for pos in positDict:
                    ipos = int(pos)
                    if effective_min <= ipos <= effective_max:
                        raw_positions.add(ipos)

    positions = sorted(raw_positions)
    print(f"[parse 1c] {len(positions)} unique genomic positions")

    # Paranoia check
    pos_str_check = ["read_id", "barcode"] + [str(p) for p in positions]
    if len(pos_str_check) != len(set(pos_str_check)):
        from collections import Counter
        dupes = [k for k, v in Counter(pos_str_check).items() if v > 1]
        raise RuntimeError(f"Duplicate column names after cleaning: {dupes}")

    np.save(edit_matrix_dir / "positions.npy", np.array(positions))

    # Write edit matrix parquets in chunks
    pos_str  = [str(p) for p in positions]
    entries  = list(selected_parsed.items())   # [(read_id, positDict), ...]
    chunk_idx = 0

    for start in range(0, len(entries), chunk_size):
        batch = entries[start : start + chunk_size]
        rows  = []
        for read_id, positDict in batch:
            bc  = selected_bc.get(read_id, None)
            vec = [float(positDict[p][0]) if p in positDict else np.nan
                   for p in positions]
            rows.append([read_id, bc] + vec)

        cols = ["read_id", "barcode"] + pos_str
        df   = pd.DataFrame(rows, columns=cols)
        out  = edit_matrix_dir / f"edit_matrix_chunk{chunk_idx}.parquet"
        df.to_parquet(out, compression="zstd", index=False)
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




def plot_edit_freq_cdf(raw_parquet_dir, out_prefix, barcodes=None):
    """
    Plot the CDF of global edit frequencies, one line per barcode.
    Streams raw parquets using the precomputed global_edit_freq column —
    no positDict parsing needed.

    Parameters
    ----------
    raw_parquet_dir : path to directory of raw parquets from shadowingBamToParquet.py
    out_prefix      : output file prefix; saved as <out_prefix>.edit_freq_cdf.png
    barcodes        : optional list of barcode strings to include (default: all)
    """
    raw_parquet_dir = Path(raw_parquet_dir)
    files = sorted(raw_parquet_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No parquet files found in {raw_parquet_dir}")

    print("[cdf] collecting edit frequencies...")
    bc_freqs = {}   # bc -> list of global_edit_freq values

    for fpath in files:
        pf = pq.ParquetFile(fpath)
        for rg in range(pf.metadata.num_row_groups):
            df = pf.read_row_group(
                rg, columns=["barcode", "global_edit_freq"]
            ).to_pandas()
            if barcodes is not None:
                df = df[df["barcode"].isin(barcodes)]
            df = df[df["barcode"].notna()]
            for bc, grp in df.groupby("barcode"):
                if bc not in bc_freqs:
                    bc_freqs[bc] = []
                bc_freqs[bc].extend(grp["global_edit_freq"].tolist())

    print(f"[cdf] {len(bc_freqs)} barcode(s) found")

    fig, ax = plt.subplots(figsize=(7, 5))

    for bc, freqs in sorted(bc_freqs.items()):
        freqs_sorted = np.sort(freqs)
        cdf = np.arange(1, len(freqs_sorted) + 1) / len(freqs_sorted)
        ax.plot(freqs_sorted, cdf, label=f"{bc} (n={len(freqs):,})", linewidth=1.5)
        print(f"  {bc}: {len(freqs):,} reads, "
              f"median={np.median(freqs):.3f}, "
              f"mean={np.mean(freqs):.3f}")

    ax.set_xlabel("Global Edit Frequency (A->G edits / A positions)")
    ax.set_ylabel("Cumulative Fraction of Reads")
    ax.set_title("CDF of Global Edit Frequencies by Barcode")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0, fontsize=8)
    ax.axvline(x=0.8, color="grey", linestyle="--", linewidth=0.8,
               label="default threshold (0.8)")

    out_path = f"{out_prefix}.edit_freq_cdf.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[cdf] saved -> {out_path}")

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
    barcodes      = None,
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
    barcodes        : optional list of barcode strings to include (default: all)
    """
    # Pass 1
    positions, edit_matrix_dir = parse_to_edit_matrix(
        raw_parquet_dir, edit_matrix_dir,
        num_reads=num_reads, min_edit_freq=min_edit_freq,
        max_abs_idx=max_abs_idx, chunk_size=chunk_size,
        window_start=window_start, window_end=window_end,
        barcodes=barcodes,
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