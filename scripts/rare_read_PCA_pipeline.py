"""
rare_read_pca_pipeline.py
--------------------------
Out-of-core PCA pipeline for identifying rare read populations in large
sequencing datasets. Designed for cases where the population of interest
comprises 1-5% of total reads, distinguished by editing patterns in a
specific genomic window.

All reads passing the global edit frequency filter enter the PCA —
there is no top-N subsampling. The rare population is identified
post-hoc from PCA cluster separation.

Input:  directory of raw parquet chunks produced by shadowingBamToParquet.py
        (columns: read_id, edit_string, barcode, ref_sequence_aligned,
                  absolute_indices, global_edit_freq, global_edit_freq_capped,
                  n_a_positions, ...)

Pipeline:
  PASS 1 — parse_to_edit_matrix()
    Sub-pass 1a: stream raw parquets, apply barcode filter + edit freq
                 filter via _parse_read_row + _edit_freq (exact, capped
                 by max_abs_idx). Build selected_bc dict.
    Sub-pass 1c scan 1: stream again, collect global position set only
                 (no positDicts stored).
    Sub-pass 1c scan 2: stream again, parse positDicts one at a time,
                 write edit matrix rows immediately — O(chunk_size) memory.

  PASS 2 — compute_column_stats()
    Stream edit-matrix parquets → per-column mean via Welford's algorithm.

  PASS 3 — fit_incremental_pca()
    Stream edit-matrix parquets → smooth (window=5, seam-padded) →
    impute NaNs → partial_fit().

  PASS 4 — transform_incremental_pca()
    Same streaming → transform() → save *_pcs.npy.

  plot_pca()
    Scatter plots of consecutive PC pairs.

  plot_edit_freq_cdf()
    CDF of global_edit_freq_capped per barcode — use to choose
    min_edit_freq threshold before running the full pipeline.

Usage:
  from rare_read_pca_pipeline import run_pipeline
  run_pipeline(
      raw_parquet_dir = "./parquets",
      edit_matrix_dir = "./edit_matrix",
      output_dir      = "./pca_output",
      min_edit_freq   = 0.8,
      n_components    = 5,
      max_abs_idx     = 695,
  )
"""

from pathlib import Path
import ast
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA


# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_read_row(row, max_abs_idx=695):
    """
    Parse one raw parquet row into a positDict.

    Only stores A positions (ref base == 'A') and skips indels (edit == '2'),
    matching the original parsePickleFileForPCA logic exactly.

    Returns dict: {abs_position: (edit_int, is_A_int)} or None if unusable.
    """
    edit_string = row["edit_string"]
    ref_aligned = row["ref_sequence_aligned"]
    abs_indices = row["absolute_indices"]

    if isinstance(abs_indices, str):
        abs_indices = ast.literal_eval(abs_indices)
    elif hasattr(abs_indices, "tolist"):
        abs_indices = abs_indices.tolist()

    positDict = {}
    for ii, abs_idx in enumerate(abs_indices):
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
        if seq != "A":
            continue
        positDict[abs_idx] = (int(edit), 1)

    return positDict if positDict else None


def _edit_freq(positDict):
    """Compute A→G edit frequency from a positDict."""
    denom = sum(v[1] for v in positDict.values())
    if denom == 0:
        return 0.0
    return sum(v[0] for v in positDict.values()) / denom


# ═══════════════════════════════════════════════════════════════════════════════
# PASS 1 — parse raw parquets → edit-matrix parquets
# ═══════════════════════════════════════════════════════════════════════════════

def parse_to_edit_matrix(
    raw_parquet_dir,
    edit_matrix_dir,
    min_edit_freq   = 0.8,
    max_abs_idx     = 695,
    chunk_size      = 50000,
    window_start    = None,
    window_end      = None,
    barcodes        = None,
):
    """
    PASS 1: Three-scan approach — memory stays O(chunk_size) throughout.

    Sub-pass 1a:
        Stream raw parquets. Apply vectorised barcode pre-filter, then
        parse positDicts and filter by _edit_freq (exact, capped by
        max_abs_idx). Build selected_bc: {read_id → barcode}.

    Sub-pass 1c scan 1:
        Stream raw parquets for selected reads only. Collect the global
        position set without storing positDicts.

    Sub-pass 1c scan 2:
        Stream raw parquets for selected reads only. Parse one positDict
        at a time, write edit matrix rows immediately, free each positDict.
        Memory at any point: O(chunk_size × n_positions).

    window_start / window_end:
        Restrict features to this genomic sub-region. Read selection is
        always based on global edit freq — window only affects which
        positions appear as columns.

    barcodes:
        Optional list of barcode strings to include. Default: all.

    Returns
    -------
    positions       : sorted list of int genomic positions (column order)
    edit_matrix_dir : Path
    """
    raw_parquet_dir = Path(raw_parquet_dir)
    edit_matrix_dir = Path(edit_matrix_dir)
    edit_matrix_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_parquet_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No parquet files found in {raw_parquet_dir}")

    effective_min = window_start if window_start is not None else 0
    effective_max = window_end   if window_end   is not None else max_abs_idx
    if window_start is not None or window_end is not None:
        print(f"[parse] genomic window: {effective_min} – {effective_max}")
    if barcodes is not None:
        print(f"[parse] barcode filter: {barcodes}")
    print(f"[parse] {len(files)} raw parquet file(s)")

    # ── Sub-pass 1a: barcode pre-filter + edit freq via positDict ────────────
    # Vectorised barcode mask avoids parsing rows from excluded barcodes.
    # Edit freq is computed from _parse_read_row + _edit_freq so the max_abs_idx
    # cap is applied consistently — global_edit_freq in the parquet is uncapped
    # and must NOT be used here.
    print("[parse 1a] filtering reads...")
    selected_bc = {}   # read_id → barcode
    total_seen  = 0
    total_passed = 0

    for i, fpath in enumerate(files, 1):
        pf = pq.ParquetFile(fpath)
        print(f"  [{i}/{len(files)}] {fpath.name}")
        for rg in range(pf.metadata.num_row_groups):
            df = pf.read_row_group(rg).to_pandas()
            total_seen += len(df)
            bc_mask = df["barcode"].notna()
            if barcodes is not None:
                bc_mask = bc_mask & df["barcode"].isin(barcodes)
            for _, row in df[bc_mask].iterrows():
                read_id   = row["read_id"]
                bc        = row["barcode"]
                positDict = _parse_read_row(row, max_abs_idx=max_abs_idx)
                if positDict is None:
                    continue
                if _edit_freq(positDict) < min_edit_freq:
                    continue
                selected_bc[read_id] = bc
                total_passed += 1

    selected_ids = set(selected_bc.keys())
    print(f"[parse 1a] {total_passed:,} / {total_seen:,} reads passed filter")
    print(f"           {Counter(selected_bc.values())}\n")

    # ── Sub-pass 1c scan 1: collect position set — no positDicts stored ───────
    print("[parse 1c-1] collecting position set...")
    raw_positions = set()

    for fpath in files:
        pf = pq.ParquetFile(fpath)
        for rg in range(pf.metadata.num_row_groups):
            df     = pf.read_row_group(rg).to_pandas()
            df_sel = df[df["read_id"].isin(selected_ids)]
            for _, row in df_sel.iterrows():
                positDict = _parse_read_row(row, max_abs_idx=max_abs_idx)
                if positDict is None:
                    continue
                for pos in positDict:
                    ipos = int(pos)
                    if effective_min <= ipos <= effective_max:
                        raw_positions.add(ipos)

    positions = sorted(raw_positions)
    print(f"[parse 1c-1] {len(positions)} unique genomic positions\n")

    # Paranoia check
    pos_str_check = ["read_id", "barcode"] + [str(p) for p in positions]
    if len(pos_str_check) != len(set(pos_str_check)):
        dupes = [k for k, v in Counter(pos_str_check).items() if v > 1]
        raise RuntimeError(f"Duplicate column names: {dupes}")

    np.save(edit_matrix_dir / "positions.npy", np.array(positions))

    # ── Sub-pass 1c scan 2: write edit matrix rows on the fly ────────────────
    # Parse one positDict at a time and write immediately — O(chunk_size) peak.
    print("[parse 1c-2] writing edit matrix...")
    pos_str   = [str(p) for p in positions]
    chunk_idx = 0
    rows_buf  = []
    seen_ids  = set()   # guard against reads spanning multiple row groups

    def _flush(buf, idx):
        cols = ["read_id", "barcode"] + pos_str
        df   = pd.DataFrame(buf, columns=cols)
        out  = edit_matrix_dir / f"edit_matrix_chunk{idx}.parquet"
        df.to_parquet(out, compression="zstd", index=False)
        print(f"  wrote {out.name}  ({len(df)} rows)")
        return idx + 1

    for fpath in files:
        pf = pq.ParquetFile(fpath)
        for rg in range(pf.metadata.num_row_groups):
            df     = pf.read_row_group(rg).to_pandas()
            df_sel = df[df["read_id"].isin(selected_ids)]
            for _, row in df_sel.iterrows():
                read_id = row["read_id"]
                if read_id in seen_ids:
                    continue
                seen_ids.add(read_id)
                positDict = _parse_read_row(row, max_abs_idx=max_abs_idx)
                if positDict is None:
                    continue
                bc  = selected_bc[read_id]
                vec = [float(positDict[p][0]) if p in positDict else np.nan
                       for p in positions]
                rows_buf.append([read_id, bc] + vec)
                if len(rows_buf) >= chunk_size:
                    chunk_idx = _flush(rows_buf, chunk_idx)
                    rows_buf  = []

    if rows_buf:
        chunk_idx = _flush(rows_buf, chunk_idx)

    print(f"[parse] done — {chunk_idx} edit-matrix parquet(s)\n")
    return positions, edit_matrix_dir


# ═══════════════════════════════════════════════════════════════════════════════
# PASS 2 — compute per-column means (Welford's online algorithm)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_column_stats(edit_matrix_dir, positions):
    """
    PASS 2: Stream edit-matrix parquets, compute per-column mean ignoring NaNs.

    Returns
    -------
    col_means : np.ndarray of shape (n_positions,)
    """
    edit_matrix_dir = Path(edit_matrix_dir)
    files   = sorted(edit_matrix_dir.glob("edit_matrix_chunk*.parquet"))
    pos_str = [str(p) for p in positions]

    n_obs  = np.zeros(len(positions), dtype=np.float64)
    means  = np.zeros(len(positions), dtype=np.float64)

    print("[stats] computing per-column means (Welford)...")
    for fpath in files:
        df  = pd.read_parquet(fpath, columns=pos_str)
        arr = df.to_numpy(dtype=np.float64)
        for row in arr:
            for j, val in enumerate(row):
                if not np.isnan(val):
                    n_obs[j] += 1
                    delta     = val - means[j]
                    means[j] += delta / n_obs[j]

    means[n_obs == 0] = 0.0
    np.save(edit_matrix_dir / "col_means.npy", means)
    print(f"[stats] done — {len(positions)} column means computed\n")
    return means


# ═══════════════════════════════════════════════════════════════════════════════
# shared smooth+impute helper (chunk-aware, seam-padded)
# ═══════════════════════════════════════════════════════════════════════════════

SMOOTH_WINDOW = 5
SEAM_PAD      = SMOOTH_WINDOW // 2


def _smooth_and_impute_chunk(arr, col_means, prev_tail):
    """
    Apply rolling smooth (window=5) + NaN imputation to one chunk.
    Seam-pads with the tail of the previous chunk to avoid boundary artefacts.
    """
    new_tail = arr[-SEAM_PAD:].copy()
    padded   = np.vstack([prev_tail, arr]) if prev_tail is not None else arr

    df           = pd.DataFrame(padded)
    smoothed_full = df.rolling(window=SMOOTH_WINDOW, min_periods=1,
                                center=True).mean().to_numpy(dtype=np.float64)
    smoothed = smoothed_full[SEAM_PAD:] if prev_tail is not None else smoothed_full

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

    ipca      = IncrementalPCA(n_components=n_components)
    prev_tail = None
    total_rows = 0

    print("[fit] fitting IncrementalPCA...")
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

    print(f"[fit] done — {ipca.explained_variance_ratio_.sum()*100:.2f}% "
          f"variance explained\n")
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
    pc_files : list of Path to *_pcs.npy
    all_bcs  : np.ndarray of barcode labels (same row order as stacked pc_files)
    """
    edit_matrix_dir = Path(edit_matrix_dir)
    output_dir      = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files   = sorted(edit_matrix_dir.glob("edit_matrix_chunk*.parquet"))
    pos_str = [str(p) for p in positions]

    pc_files  = []
    all_bcs   = []
    prev_tail = None

    print("[transform] projecting into PC space...")
    for fpath in files:
        df       = pd.read_parquet(fpath, columns=["barcode"] + pos_str)
        bcs      = df["barcode"].to_numpy()
        arr      = df[pos_str].to_numpy(dtype=np.float64)

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
    print(f"[transform] done — {len(pc_files)} file(s), {len(all_bcs)} total reads\n")
    return pc_files, all_bcs


# ═══════════════════════════════════════════════════════════════════════════════
# plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pca(pc_files, out_prefix, labels=None, n_pairs=4):
    """
    Scatter plots of consecutive PC pairs (PC1v2, PC2v3, ...).

    Parameters
    ----------
    pc_files   : list of paths to *_pcs.npy
    out_prefix : prefix for output files
    labels     : 1-D array of per-row label strings, or None
    n_pairs    : number of consecutive pairs to plot (default 4)
    """
    X_pca        = np.vstack([np.load(f) for f in pc_files])
    n_components = X_pca.shape[1]
    n_pairs      = min(n_pairs, n_components - 1)

    pc_cols          = {f"PC{i+1}": X_pca[:, i] for i in range(n_pairs + 1)}
    X_pca_df         = pd.DataFrame(pc_cols)
    X_pca_df["label"] = np.asarray(labels) if labels is not None else "data"
    unique_labels    = X_pca_df["label"].unique()

    print(f"[plot] {n_pairs} plot(s), {len(unique_labels)} label(s)")
    for ii in range(n_pairs):
        fig, ax = plt.subplots(figsize=(6, 6))
        for label in unique_labels:
            subset = X_pca_df[X_pca_df["label"] == label]
            ax.scatter(subset[f"PC{ii+1}"], subset[f"PC{ii+2}"],
                       label=label, s=10, alpha=0.7)
        ax.set_title("PCA of Edit Vectors")
        ax.set_xlabel(f"PC{ii+1}")
        ax.set_ylabel(f"PC{ii+2}")
        ax.legend(markerscale=2, bbox_to_anchor=(1.01, 1), loc="upper left",
                  borderaxespad=0, fontsize=8)
        out_path = f"{out_prefix}.PCA.{ii+1}.{ii+2}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  saved → {out_path}")


def plot_edit_freq_cdf(raw_parquet_dir, out_prefix, barcodes=None,
                       use_capped=True, min_edit_freq=None):
    """
    Plot the CDF of edit frequencies per barcode.

    Uses global_edit_freq_capped (max_abs_idx-capped) by default, which
    matches the threshold used by the pipeline for read selection.
    Falls back to global_edit_freq if the capped column is not present.

    Parameters
    ----------
    raw_parquet_dir : path to directory of raw parquets
    out_prefix      : output file prefix
    barcodes        : optional list of barcode strings to include
    use_capped      : if True, use global_edit_freq_capped (default)
    min_edit_freq   : if provided, draw a vertical threshold line
    """
    raw_parquet_dir = Path(raw_parquet_dir)
    files = sorted(raw_parquet_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No parquet files found in {raw_parquet_dir}")

    # Detect which column to use
    schema = pq.read_schema(files[0])
    if use_capped and "global_edit_freq_capped" in schema.names:
        freq_col = "global_edit_freq_capped"
    else:
        freq_col = "global_edit_freq"
        if use_capped:
            print("[cdf] global_edit_freq_capped not found — falling back to "
                  "global_edit_freq (uncapped). Re-run shadowingBamToParquet.py "
                  "with --max_abs_idx to generate the capped column.")
    print(f"[cdf] using column: {freq_col}")

    bc_freqs = {}
    for fpath in files:
        pf = pq.ParquetFile(fpath)
        for rg in range(pf.metadata.num_row_groups):
            df = pf.read_row_group(rg, columns=["barcode", freq_col]).to_pandas()
            if barcodes is not None:
                df = df[df["barcode"].isin(barcodes)]
            df = df[df["barcode"].notna()]
            for bc, grp in df.groupby("barcode"):
                bc_freqs.setdefault(bc, []).extend(grp[freq_col].tolist())

    print(f"[cdf] {len(bc_freqs)} barcode(s) found")

    fig, ax = plt.subplots(figsize=(7, 5))
    for bc, freqs in sorted(bc_freqs.items()):
        freqs_sorted = np.sort(freqs)
        cdf = np.arange(1, len(freqs_sorted) + 1) / len(freqs_sorted)
        ax.plot(freqs_sorted, cdf, label=f"{bc} (n={len(freqs):,})", linewidth=1.5)
        print(f"  {bc}: {len(freqs):,} reads  "
              f"median={np.median(freqs):.3f}  mean={np.mean(freqs):.3f}")

    if min_edit_freq is not None:
        ax.axvline(x=min_edit_freq, color="grey", linestyle="--", linewidth=0.8,
                   label=f"threshold ({min_edit_freq})")

    ax.set_xlabel(f"Edit Frequency ({freq_col})")
    ax.set_ylabel("Cumulative Fraction of Reads")
    ax.set_title("CDF of Edit Frequencies by Barcode")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0, fontsize=8)

    out_path = f"{out_prefix}.edit_freq_cdf.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[cdf] saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# top-level convenience wrapper
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    raw_parquet_dir,
    edit_matrix_dir,
    output_dir,
    min_edit_freq   = 0.8,
    max_abs_idx     = 695,
    n_components    = 5,
    chunk_size      = 50000,
    plot            = True,
    window_start    = None,
    window_end      = None,
    barcodes        = None,
):
    """
    Run the full rare-read PCA pipeline.

    Parameters
    ----------
    raw_parquet_dir : directory of parquets from shadowingBamToParquet.py
    edit_matrix_dir : intermediate directory for edit-matrix parquets
    output_dir      : final outputs (pcs, model, plots)
    min_edit_freq   : minimum A→G edit fraction to keep a read (default 0.8)
    max_abs_idx     : upper bound on genomic position index (default 695)
    n_components    : number of PCA components (default 5)
    chunk_size      : rows per edit-matrix parquet chunk (default 50000)
    plot            : whether to generate scatter plots (default True)
    window_start    : optional start of genomic position window (inclusive)
    window_end      : optional end of genomic position window (inclusive)
    barcodes        : optional list of barcode strings to include (default: all)
    """
    positions, edit_matrix_dir = parse_to_edit_matrix(
        raw_parquet_dir, edit_matrix_dir,
        min_edit_freq=min_edit_freq,
        max_abs_idx=max_abs_idx,
        chunk_size=chunk_size,
        window_start=window_start,
        window_end=window_end,
        barcodes=barcodes,
    )

    col_means = compute_column_stats(edit_matrix_dir, positions)

    ipca = fit_incremental_pca(
        edit_matrix_dir, positions, col_means, n_components=n_components
    )

    pc_files, all_bcs = transform_incremental_pca(
        ipca, edit_matrix_dir, positions, col_means, output_dir
    )

    if plot:
        win_str    = (f"window_{window_start or 0}_{window_end or max_abs_idx}"
                      if window_start is not None or window_end is not None
                      else "all_positions")
        out_prefix = str(Path(output_dir) / f"result_{win_str}")
        plot_pca(pc_files, out_prefix=out_prefix, labels=all_bcs)

    return ipca, pc_files, all_bcs