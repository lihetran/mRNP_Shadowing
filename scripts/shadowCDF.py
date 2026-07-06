'''
June 30, 2026 LT

This script takes 3 libraries (parquet directories) and computes inter-shadow
distance per read. A shadow is a run of transcript nucleotides, at least
shadow_size nt long, in which every ref=A position covered by the read is
unedited (no A->G edit).

Distances are measured between consecutive shadows: from the end of one
shadow to the start of the next, in transcript nucleotides.

Per Read:
    1. Build a ref=A edit map in transcript coordinates
    2. Find shadows (runs of unedited ref=A spanning >= shadow_size nt)
    3. Record the gap distance between each consecutive pair of shadows

inputs:
    library1/2/3: paths to three parquet directories
    ref:          reference FASTA
    gtf:          annotation GTF

outputs:
    CDF of inter-shadow distances for each library
'''
import pandas as pd
import numpy as np
import argparse
import sys
import re
import math
import collections
from pathlib import Path

import pysam


HIS_CODONS = {"CAT", "CAC"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def complement_base(b: str) -> str:
    return b.translate(str.maketrans("ACGTacgt", "TGCAtgca"))

def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]


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
            gname   = m_gn.group(1) if m_gn else None
            if gname is None:
                continue
            if feature == "gene":
                gene_extents[gname] = (start, end)
            if feature == "CDS":
                if gname not in genes:
                    genes[gname] = {
                        "chrom": chrom, "strand": strand,
                        "gene_name": gname, "cds": [],
                    }
                genes[gname]["cds"].append((start, end))

    for gname, g in genes.items():
        if gname in gene_extents:
            g["gene_start"], g["gene_end"] = gene_extents[gname]
        else:
            g["gene_start"] = min(s for s, e in g["cds"])
            g["gene_end"]   = max(e for s, e in g["cds"])
        g["cds"].sort(key=lambda x: x[0], reverse=(g["strand"] == "-"))
        g["cds_genomic_start"] = min(s for s, e in g["cds"])
        g["cds_genomic_end"]   = max(e for s, e in g["cds"])
    return genes


def cds_length(gene: dict) -> int:
    return sum(ce - cs for cs, ce in gene["cds"])


def _gpos_to_tx_map(gene: dict, ref_fasta: pysam.FastaFile) -> dict:
    """
    Map genomic position -> spliced CDS transcript position for all ref=A
    positions (transcript sense). tx_pos is contiguous over CDS segments.
    """
    chrom     = gene["chrom"]
    strand    = gene["strand"]
    chrom_seq = ref_fasta.fetch(chrom).upper()

    gpos_to_tx = {}
    tx_pos = 0
    for (cs, ce) in gene["cds"]:
        if strand == "+":
            for gpos in range(cs, ce):
                if chrom_seq[gpos] == "A":
                    gpos_to_tx[gpos] = tx_pos
                tx_pos += 1
        else:
            for gpos in range(ce - 1, cs - 1, -1):
                if chrom_seq[gpos] == "T":   # ref=T on minus = A in transcript
                    gpos_to_tx[gpos] = tx_pos
                tx_pos += 1
    return gpos_to_tx


# ─────────────────────────────────────────────────────────────────────────────
# Parquet loading
# ─────────────────────────────────────────────────────────────────────────────

def load_all_parquet_chunks(parquet_dir: str) -> pd.DataFrame:
    parquet_dir = Path(parquet_dir)
    chunks = sorted(parquet_dir.glob("*.parquet"))
    if not chunks:
        return pd.DataFrame()
    dfs = [pd.read_parquet(c) for c in chunks]
    df  = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df):,} reads from {len(chunks)} chunk(s).",
          file=sys.stderr)
    return df


def get_gene_df(df_all: pd.DataFrame, gene: dict) -> pd.DataFrame:
    mask = ((df_all["chrom"]       == gene["chrom"]) &
            (df_all["gene_strand"] == gene["strand"]))
    if "read_start" in df_all.columns and "read_end" in df_all.columns:
        mask &= ((df_all["read_start"] < gene["gene_end"]) &
                 (df_all["read_end"]   > gene["gene_start"]))
    return df_all[mask]


# ─────────────────────────────────────────────────────────────────────────────
# Per-read ref=A edit map and shadow finding
# ─────────────────────────────────────────────────────────────────────────────

def collect_read_edits(df: pd.DataFrame, gpos_to_tx: dict) -> dict:
    """
    {read_id: {tx_pos: 0_or_1}} using absolute_indices + edit_string,
    restricted to ref=A positions (via gpos_to_tx).
    1 = edited (G), 0 = unedited (A). edit_string '2' (indel) is skipped.
    """
    if df.empty:
        return {}

    min_gp = min(gpos_to_tx.keys(), default=None)
    max_gp = max(gpos_to_tx.keys(), default=None)
    if min_gp is None:
        return {}

    if "read_start" in df.columns and "read_end" in df.columns:
        sub = df[(df["read_start"] <= max_gp) & (df["read_end"] >= min_gp)]
    else:
        sub = df

    read_edits = collections.defaultdict(dict)

    for read in sub.itertuples():
        edit_str    = read.edit_string
        abs_indices = read.absolute_indices
        n_edit      = len(edit_str)

        for i, ref_pos in enumerate(abs_indices):
            if ref_pos is None:
                continue
            if isinstance(ref_pos, float) and ref_pos != ref_pos:
                continue
            ref_pos = int(ref_pos)
            if ref_pos < min_gp or ref_pos > max_gp:
                continue
            if ref_pos not in gpos_to_tx:
                continue
            if i >= n_edit:
                continue
            ev = edit_str[i]
            if ev == "2":
                continue
            read_edits[read.read_id][gpos_to_tx[ref_pos]] = int(ev)

    return dict(read_edits)


def find_distances_per_read(edit_at_tx: dict, shadow_size: int,
                            min_sites: int = 1,
                            max_site_gap: int = None):
    """
    Given {tx_pos: 0_or_1} for one read (ref=A positions only, in transcript
    nt coordinates), find shadows and return inter-shadow gap distances in nt.

    A shadow is a maximal run of consecutive unedited ref=A sites that:
      - spans at least shadow_size nt (last_pos - first_pos + 1)
      - contains at least min_sites ref=A positions

    A run is broken by:
      - an edited (G) ref=A site, OR
      - a gap larger than max_site_gap between consecutive unedited ref=A
        sites (prevents joining distant clusters across regions with no
        observed A's into one spurious shadow).

    max_site_gap defaults to shadow_size if not given — i.e. if two unedited
    A's are more than one shadow-width apart with nothing in between, they
    are not treated as the same continuous protected region.

    Distance = start of next shadow minus end of current shadow (in nt).
    """
    if not edit_at_tx:
        return []

    if max_site_gap is None:
        max_site_gap = shadow_size

    positions = sorted(edit_at_tx.keys())

    shadows = []
    run_positions = []

    def _close_run(run):
        if len(run) < min_sites:
            return
        span = run[-1] - run[0] + 1
        if span >= shadow_size:
            shadows.append((run[0], run[-1] + 1))   # end exclusive

    prev_pos = None
    for p in positions:
        if edit_at_tx[p] == 0:
            # Break the run if this unedited site is too far from the previous
            # unedited site (gap in observed A's, not continuous protection).
            if (run_positions and prev_pos is not None
                    and p - prev_pos > max_site_gap):
                _close_run(run_positions)
                run_positions = []
            run_positions.append(p)
            prev_pos = p
        else:  # edited site breaks the run
            _close_run(run_positions)
            run_positions = []
            prev_pos = None
    _close_run(run_positions)

    distances = []
    for i in range(len(shadows) - 1):
        distances.append(shadows[i + 1][0] - shadows[i][1])

    return distances


def find_shadow_dist_per_library(df_all: pd.DataFrame, genes: dict,
                                  ref_fasta: pysam.FastaFile,
                                  shadow_size: int,
                                  min_sites: int,
                                  max_site_gap: int = None) -> pd.DataFrame:
    """
    For every gene, build per-read ref=A edit maps in transcript coordinates,
    find shadows, and record inter-shadow distances.

    Returns a DataFrame with columns 'read_id', 'gene', 'shadow_distances'.
    """
    results = []

    gene_names = list(genes.keys())
    for gi, gname in enumerate(gene_names):
        if (gi + 1) % 200 == 0:
            print(f"    {gi+1}/{len(gene_names)} genes...", file=sys.stderr)

        gene    = genes[gname]
        gene_df = get_gene_df(df_all, gene)
        if gene_df.empty:
            continue

        gpos_to_tx = _gpos_to_tx_map(gene, ref_fasta)
        if not gpos_to_tx:
            continue

        read_edits = collect_read_edits(gene_df, gpos_to_tx)
        for read_id, edit_at_tx in read_edits.items():
            distances = find_distances_per_read(
                edit_at_tx, shadow_size, min_sites, max_site_gap)
            if distances:
                results.append({
                    "read_id":          read_id,
                    "gene":             gname,
                    "shadow_distances": distances,
                })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_CDF(results_df1: pd.DataFrame, results_df2: pd.DataFrame,
             results_df3: pd.DataFrame,
             label1="lib1", label2="lib2", label3="lib3",
             output_prefix="shadow_distance", x_max=1000):
    """
    Overlaid CDF of inter-shadow distances from three libraries.
    Each results_df has a 'shadow_distances' column holding a list of
    distances per read; these are flattened across all reads.
    """
    from pyx import canvas, graph, color, style, path, text as pyx_text

    col1    = color.cmyk(0, 0, 0, 1)        # black
    col2    = color.cmyk(1, 0.5, 0, 0)      # blue
    col3    = color.cmyk(0, 1, 1, 0)        # red
    panel_w = 8
    panel_h = 6

    def _flatten(df):
        if df is None or df.empty or "shadow_distances" not in df.columns:
            return np.array([])
        vals = [d for lst in df["shadow_distances"] for d in lst]
        return np.array(vals, dtype=float)

    series = [
        (_flatten(results_df1), col1, label1),
        (_flatten(results_df2), col2, label2),
        (_flatten(results_df3), col3, label3),
    ]

    c = canvas.canvas()
    g = graph.graphxy(
        width=panel_w, height=panel_h, xpos=0, ypos=0,
        x=graph.axis.linear(min=0, max=x_max,
                            title="Shadow distance (nt)"),
        y=graph.axis.linear(min=0, max=1,
                            title="Cumulative fraction"),
    )

    leg_x  = panel_w + 0.5
    leg_lw = 0.8
    leg_dy = 0.6
    leg_y0 = panel_h - 0.3

    for j, (vals, col, label) in enumerate(series):
        if len(vals) == 0:
            continue
        # CDF over ALL distances so y reaches 1.0 correctly, then draw only
        # the portion within [0, x_max].
        sv  = np.sort(vals)
        cdf = np.arange(1, len(sv) + 1) / len(sv)

        keep  = sv <= x_max
        sv_k  = sv[keep]
        cdf_k = cdf[keep]
        if len(sv_k) and sv_k[-1] < x_max:
            sv_k  = np.append(sv_k,  x_max)
            cdf_k = np.append(cdf_k, cdf_k[-1])

        g.plot(graph.data.points(list(zip(sv_k.tolist(), cdf_k.tolist())),
                                 x=1, y=2),
               [graph.style.line([col, style.linewidth.normal,
                                  style.linestyle.solid])])

        ly = leg_y0 - j * leg_dy
        c.stroke(path.line(leg_x, ly, leg_x + leg_lw, ly),
                 [col, style.linewidth.normal, style.linestyle.solid])
        frac_shown = float((vals <= x_max).mean())
        c.text(leg_x + leg_lw + 0.15, ly,
               f"{label}",
               [pyx_text.valign.middle, pyx_text.size.small])

    c.insert(g)
    c.text(panel_w / 2., panel_h + 0.5,
           "Inter-shadow distance CDF",
           [pyx_text.halign.center, pyx_text.size.normalsize])

    plot_path = f"{output_prefix}_shadow_distance_cdf_pyx"
    c.writePDFfile(plot_path)
    print(f"  Saved -> {plot_path}.pdf", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Editing-efficiency matching (control for global edit rate confound)
# ─────────────────────────────────────────────────────────────────────────────

def match_editing_efficiency(dfs: list, labels: list,
                             bin_width: float = 0.02,
                             seed: int = 42) -> list:
    """
    Downsample multiple libraries so their per-read global_edit_freq
    distributions are identical.

    For each global_edit_freq bin, finds the minimum read count across all
    libraries and randomly keeps exactly that many reads per bin in every
    library. This removes editing-rate as a confound: any residual
    difference in shadow distances afterward cannot be attributed to one
    library simply being more or less edited.

    Returns a list of downsampled DataFrames in the same order as input.
    Requires a 'global_edit_freq' column.
    """
    rng = np.random.default_rng(seed)

    # Verify column present
    for df, label in zip(dfs, labels):
        if "global_edit_freq" not in df.columns:
            print(f"  WARNING: '{label}' has no global_edit_freq column; "
                  f"matching skipped.", file=sys.stderr)
            return dfs

    # Assign each read to a bin
    max_freq = 1.0
    n_bins   = int(math.ceil(max_freq / bin_width))
    edges    = np.arange(0, (n_bins + 1) * bin_width, bin_width)

    # Per-library, per-bin read indices
    binned = []   # list of {bin_idx: array_of_row_positions}
    for df in dfs:
        freqs = df["global_edit_freq"].fillna(-1).values
        bin_idx = np.clip((freqs / bin_width).astype(int), 0, n_bins - 1)
        # reads with freq < 0 (NaN) get bin -1 → exclude
        bin_idx[freqs < 0] = -1
        by_bin = collections.defaultdict(list)
        for row_pos, b in enumerate(bin_idx):
            if b >= 0:
                by_bin[b].append(row_pos)
        binned.append(by_bin)

    # For each bin, the target count is the min across libraries
    all_bins = set()
    for by_bin in binned:
        all_bins.update(by_bin.keys())

    keep_indices = [[] for _ in dfs]
    total_before = [len(df) for df in dfs]

    for b in sorted(all_bins):
        counts = [len(by_bin.get(b, [])) for by_bin in binned]
        target = min(counts)
        if target == 0:
            continue   # at least one library has no reads here → drop bin
        for li, by_bin in enumerate(binned):
            rows = by_bin[b]
            if len(rows) > target:
                chosen = rng.choice(rows, size=target, replace=False)
            else:
                chosen = rows
            keep_indices[li].extend(chosen)

    matched = []
    for li, df in enumerate(dfs):
        idx = sorted(keep_indices[li])
        matched.append(df.iloc[idx].reset_index(drop=True))

    print("  Editing-efficiency matching:", file=sys.stderr)
    for label, before, after_df in zip(labels, total_before, matched):
        print(f"    [{label}] {before:,} -> {len(after_df):,} reads "
              f"after matching", file=sys.stderr)

    return matched




def parse_args():
    p = argparse.ArgumentParser(
        description="Inter Shadow Distance CDF."
    )
    p.add_argument("--parquet1", required=True,
                   help="Parquet directory for library 1")
    p.add_argument("--parquet2", required=True,
                   help="Parquet directory for library 2")
    p.add_argument("--parquet3", required=True,
                   help="Parquet directory for library 3")
    p.add_argument("--ref",      required=True, help="Reference FASTA")
    p.add_argument("--gtf",      required=True, help="Annotation GTF")
    p.add_argument("--label1",   default="Lib1")
    p.add_argument("--label2",   default="Lib2")
    p.add_argument("--label3",   default="Lib3")
    p.add_argument("--output",   default="shadow_distance_cdf")
    p.add_argument("--window",   type=int, default=50,
                   help="Shadow size: min run length in nt (default: 50)")
    p.add_argument("--min_sites", type=int, default=1,
                   help="Min ref=A sites a shadow must contain (default: 1)")
    p.add_argument("--max_site_gap", type=int, default=None,
                   help="Max nt gap between consecutive unedited ref=A sites "
                        "before the shadow run is broken. Defaults to "
                        "shadow size (--window).")
    p.add_argument("--x_max",    type=int, default=1000,
                   help="Max distance shown on the CDF x-axis (default: 1000)")
    p.add_argument("--match_editing", action="store_true",
                   help="Downsample all libraries to identical per-read "
                        "global_edit_freq distributions before computing "
                        "shadows. Controls for the editing-rate confound.")
    p.add_argument("--match_bin_width", type=float, default=0.02,
                   help="Bin width for editing-efficiency matching "
                        "(default: 0.02)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for matching subsample (default: 42)")
    return p.parse_args()


def main():
    args = parse_args()
    out  = args.output
    shadow_size = args.window

    print("=== Inter-shadow Distance CDF ===", file=sys.stderr)
    print(f"  Shadow size: {shadow_size} nt", file=sys.stderr)
    print(f"  Min ref=A sites per shadow: {args.min_sites}", file=sys.stderr)

    print("\nParsing GTF...", file=sys.stderr)
    genes = parse_gtf(args.gtf)
    print(f"  {len(genes):,} genes.", file=sys.stderr)

    ref_fasta = pysam.FastaFile(args.ref)

    print("\nLoading parquet chunks...", file=sys.stderr)
    print(f"  {args.label1}:", file=sys.stderr)
    df_all1 = load_all_parquet_chunks(args.parquet1)
    print(f"  {args.label2}:", file=sys.stderr)
    df_all2 = load_all_parquet_chunks(args.parquet2)
    print(f"  {args.label3}:", file=sys.stderr)
    df_all3 = load_all_parquet_chunks(args.parquet3)

    # Optional: match editing efficiency across libraries to remove the
    # global edit-rate confound before computing shadows.
    if args.match_editing:
        print("\nMatching editing efficiency across libraries...",
              file=sys.stderr)
        df_all1, df_all2, df_all3 = match_editing_efficiency(
            [df_all1, df_all2, df_all3],
            [args.label1, args.label2, args.label3],
            bin_width=args.match_bin_width,
            seed=args.seed,
        )

    print("\nFinding shadow distances per library...", file=sys.stderr)
    print(f"  {args.label1}...", file=sys.stderr)
    results_df1 = find_shadow_dist_per_library(
        df_all1, genes, ref_fasta, shadow_size, args.min_sites,
        args.max_site_gap)
    print(f"  {args.label2}...", file=sys.stderr)
    results_df2 = find_shadow_dist_per_library(
        df_all2, genes, ref_fasta, shadow_size, args.min_sites,
        args.max_site_gap)
    print(f"  {args.label3}...", file=sys.stderr)
    results_df3 = find_shadow_dist_per_library(
        df_all3, genes, ref_fasta, shadow_size, args.min_sites,
        args.max_site_gap)

    ref_fasta.close()

    # Report counts
    for rdf, label in [(results_df1, args.label1),
                       (results_df2, args.label2),
                       (results_df3, args.label3)]:
        n_reads = len(rdf)
        n_dist  = sum(len(d) for d in rdf["shadow_distances"]) \
                  if not rdf.empty else 0
        print(f"  [{label}] {n_reads:,} reads with >=2 shadows, "
              f"{n_dist:,} distances.", file=sys.stderr)

    print("\nPlotting...", file=sys.stderr)
    plot_CDF(results_df1, results_df2, results_df3,
             label1=args.label1, label2=args.label2, label3=args.label3,
             output_prefix=out, x_max=args.x_max)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()