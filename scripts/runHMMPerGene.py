import argparse
import sys
import re
import math
import collections
from pathlib import Path

import pysam
import numpy as np
import pandas as pd
import scipy.stats
from logJosh import Tee
import json
import pickle

HIS_CODONS = {"CAT", "CAC"}
def complement_base(b: str) -> str:
    return b.translate(str.maketrans("ACGTacgt", "TGCAtgca"))

def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]

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

def load_model_from_pickle(pickle_path: str) -> dict:
    with open(pickle_path, "rb") as f:
        return pickle.load(f)

def parse_gtf(gtf_path: str) -> dict:
    """
    Parse a GTF carrying only `exon` and `CDS` features (plus optional `gene`).

    UTRs are derived as exon - CDS in genomic space, then assigned to 5'/3' by
    strand. Each gene is pinned to a single transcript (the first encountered)
    so isoforms are never blended into one coordinate space.

    Returns {gene_name: {
        chrom, strand, transcript, gene_name,
        cds, exons, utr5, utr3,          # (start, end) lists, transcript order
        gene_start, gene_end,
        cds_genomic_start, cds_genomic_end,
    }}
    """
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

            m_gn  = re.search(r'gene_name "([^"]+)"', fields[8])
            m_tid = re.search(r'transcript_id "([^"]+)"', fields[8])
            gname = m_gn.group(1)  if m_gn  else None
            tid   = m_tid.group(1) if m_tid else "."
            if gname is None:
                continue

            if feature == "gene":
                gene_extents[gname] = (start, end)
                continue

            if feature not in ("CDS", "exon"):
                continue

            if gname not in genes:
                genes[gname] = {
                    "chrom": chrom, "strand": strand,
                    "transcript": tid, "gene_name": gname,
                    "cds": [], "exons": [],
                }
            elif genes[gname]["transcript"] != tid:
                continue                 # pin to first transcript only

            key = "cds" if feature == "CDS" else "exons"
            genes[gname][key].append((start, end))

    drop = []
    for gname, g in genes.items():
        if not g["cds"]:
            drop.append(gname)           # non-coding: no CDS to anchor on
            continue

        if gname in gene_extents:
            g["gene_start"], g["gene_end"] = gene_extents[gname]
        else:
            spans = g["exons"] or g["cds"]
            g["gene_start"] = min(s for s, e in spans)
            g["gene_end"]   = max(e for s, e in spans)

        g["cds_genomic_start"] = min(s for s, e in g["cds"])
        g["cds_genomic_end"]   = max(e for s, e in g["cds"])

        # UTRs = exon - CDS, split by genomic side, then assigned by strand
        utr = _subtract_intervals(g["exons"], g["cds"])
        left  = [iv for iv in utr if iv[1] <= g["cds_genomic_start"]]
        right = [iv for iv in utr if iv[0] >= g["cds_genomic_end"]]

        if g["strand"] == "+":
            g["utr5"], g["utr3"] = left, right
        else:
            g["utr5"], g["utr3"] = right, left

        # Sort every segment list into transcript order
        rev = (g["strand"] == "-")
        for key in ("cds", "exons", "utr5", "utr3"):
            g[key].sort(key=lambda x: x[0], reverse=rev)

    for gname in drop:
        del genes[gname]

    return genes

def cds_length(gene: dict) -> int:
    return sum(ce - cs for cs, ce in gene["cds"])

def find_his_codon_tx_positions(ref_fasta, gene):
    chrom, strand = gene["chrom"], gene["strand"]
    tx_seq = ""
    for (cs, ce) in gene["cds"]:
        seg = ref_fasta.fetch(chrom, cs, ce).upper()
        if strand == "-":
            seg = reverse_complement(seg)
        tx_seq += seg
    his_pos, seen = [], set()
    for i in range(0, len(tx_seq) - 2, 3):
        if tx_seq[i:i+3] in HIS_CODONS:
            p = i + 1
            if p not in seen:
                seen.add(p)
                his_pos.append(p)
    return his_pos

def _subtract_intervals(exons, cds):
    """
    exons, cds: lists of (start, end) genomic half-open intervals, unsorted.
    Returns the parts of exons not covered by any cds interval, sorted by start.
    """
    if not cds:
        return sorted(exons)

    cds = sorted(cds)
    out = []
    for (es, ee) in sorted(exons):
        cur = es
        for (cs, ce) in cds:
            if ce <= cur or cs >= ee:
                continue                  # no overlap with the remaining piece
            if cs > cur:
                out.append((cur, cs))     # piece before this CDS block
            cur = max(cur, ce)
            if cur >= ee:
                break
        if cur < ee:
            out.append((cur, ee))         # trailing piece
    return out

def get_gene_df(df_all: pd.DataFrame, gene: dict,
                cds_spanning: bool = False,
                min_edit_freq: float = 0.0) -> pd.DataFrame:
    """
    Built with Claude
    Fast vectorised pre-filter to reads overlapping this gene.

    If cds_spanning is True, only keep reads whose alignment spans the full
    CDS (read_start <= cds_genomic_start and read_end >= cds_genomic_end),
    so every read had the opportunity to be edited at every position.

    If min_edit_freq > 0, only keep reads whose global_edit_freq column
    (per-read A->G edit fraction) is >= min_edit_freq. Reads below this
    threshold are likely unedited / poorly edited molecules that carry no
    protection signal and only add noise.
    """
    mask = ((df_all["chrom"] == gene["chrom"]) & (df_all["gene_strand"] == gene["strand"]))
    if "read_start" in df_all.columns and "read_end" in df_all.columns:
        if cds_spanning:
            cds_start = gene.get("cds_genomic_start", gene["gene_start"])
            cds_end = gene.get("cds_genomic_end",   gene["gene_end"])
            mask &= ((df_all["read_start"] <= cds_start) & (df_all["read_end"]   >= cds_end))
        else:
            mask &= ((df_all["read_start"] < gene["gene_end"]) & (df_all["read_end"] > gene["gene_start"]))
    if min_edit_freq > 0.0 and "global_edit_freq" in df_all.columns:
        mask &= (df_all["global_edit_freq"] >= min_edit_freq)
    return df_all[mask]

def _full_tx_map(gene: dict, ref_fasta: pysam.FastaFile,
                 include_utrs: bool = True) -> dict:
    """
    Built with Claude
    Map tx_pos -> (gpos, ref_base_sense) for EVERY transcript position (all
    bases, not just A). ref_base_sense is the transcript-sense reference base
    (complemented for minus-strand genes), so 'A' marks editable sites.

    Coordinates are CDS-relative: the first CDS base is tx_pos 0, 5'UTR
    positions are negative, 3'UTR positions are >= cds_length(gene).
    """
    chrom_seq = ref_fasta.fetch(gene["chrom"]).upper()
    strand    = gene["strand"]

    def _walk(segments, tx_start):
        """Yield (tx_pos, gpos, sense_base) in transcript order."""
        tx = tx_start
        for (cs, ce) in segments:
            rng = range(cs, ce) if strand == "+" else range(ce - 1, cs - 1, -1)
            for gpos in rng:
                base = chrom_seq[gpos]
                if strand == "-":
                    base = complement_base(base)
                yield tx, gpos, base
                tx += 1

    full = {}
    for tx, gpos, base in _walk(gene["cds"], 0):
        full[tx] = (gpos, base)

    if include_utrs:
        cds_len = cds_length(gene)
        for tx, gpos, base in _walk(gene.get("utr3", []), cds_len):
            full[tx] = (gpos, base)

        # 5'UTR: walk from 0 in transcript order, then shift so it ends at -1
        u5 = list(_walk(gene.get("utr5", []), 0))
        n5 = len(u5)
        for tx, gpos, base in u5:
            full[tx - n5] = (gpos, base)

    return full

def collect_read_edits(df: pd.DataFrame, gpos_to_tx: dict,
                        gene: dict) -> dict:
    """
    {read_id: {tx_pos: 0_or_1}} from parquet using absolute_indices +
    edit_string. Restricts to ref=A positions via gpos_to_tx.
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

def _gpos_to_tx_map(gene, ref_fasta, include_utrs=True):
    chrom_seq = ref_fasta.fetch(gene["chrom"]).upper()
    strand    = gene["strand"]
    want      = "A" if strand == "+" else "T"

    def _walk(segments, tx_start):
        """Yield (gpos, tx_pos) in transcript order, starting at tx_start."""
        tx = tx_start
        for (cs, ce) in segments:
            rng = range(cs, ce) if strand == "+" else range(ce - 1, cs - 1, -1)
            for gpos in rng:
                yield gpos, tx
                tx += 1

    out = {}
    for gpos, tx in _walk(gene["cds"], 0):
        if chrom_seq[gpos] == want:
            out[gpos] = tx

    if include_utrs:
        cds_len = cds_length(gene)
        for gpos, tx in _walk(gene["utr3"], cds_len):
            if chrom_seq[gpos] == want:
                out[gpos] = tx
        # 5'UTR: walk in transcript order, then offset so it ends at -1
        u5 = list(_walk(gene["utr5"], 0))
        n5 = len(u5)
        for gpos, tx in u5:
            if chrom_seq[gpos] == want:
                out[gpos] = tx - n5

    return out

def classify_tx(tx_pos, cds_len):
    if tx_pos < 0: return "UTR5"
    if tx_pos < cds_len: return "CDS"
    return "UTR3"

def classify_positions_hmm2(model, read_id, chrom, edit_string,
                           absolute_indices, gpos_to_tx,
                           mean_block_nt=None, coord="tx", use_prior=True):
    """
    Two-state HMM over the scored (Ref=A) sites of one read.
      state A = unprotected (ribosome absent, TadA edits at pA[tx])
      state B = protected   (ribosome bound, TadA blocked, pB[tx] ~ 0)
    Emissions are the trained rates, unchanged. Transitions encode
    'protection comes in contiguous ~mean_block_nt runs at ~occupancy'.
    """
    n_edit = len(edit_string)

    # ---- pass 1: collect scored sites (read order) -----------------------
    sites = []                       # (tx, ref_pos, bit, pA_i, pB_i)
    for i, ref_pos in enumerate(absolute_indices):
        if ref_pos is None or (isinstance(ref_pos, float) and ref_pos != ref_pos):
            continue
        ref_pos = int(ref_pos)
        if ref_pos not in gpos_to_tx or i >= n_edit:
            continue
        ev = edit_string[i]
        if ev == "2":
            continue
        tx = gpos_to_tx[ref_pos]
        if tx not in model["pA"] or tx not in model["pB"]:     # need BOTH
            continue
        sites.append((tx, ref_pos, int(ev), model["pA"][tx], model["pB"][tx]))

    if not sites:
        yield {"read_id": read_id, "chrom": chrom, "absolute_indices": [], "tx": [],
               "labels": [], "P_A": [], "P_B": [], "logL_A": [], "logL_B": [],
               "surprise": [], "edits": [], "win_n": []}
        return

    key = (lambda s: s[0]) if coord == "tx" else (lambda s: s[1])

    # keep an explicit READ-ORDER INDEX alongside each site, so duplicate
    # coordinates can never collapse into one another
    indexed = list(enumerate(sites))                     # (read_idx, site)
    ordered = sorted(indexed, key=lambda p: key(p[1]))   # sorted by coordinate
    coords  = [key(s) for _ri, s in ordered]
    n = len(ordered)

    # ---- transitions from biology ---------------------------------------
    if mean_block_nt is None:
        mean_block_nt = model.get("rpf_len_nt", 30)
    occupancy = (1.0 - model["prior_A"]) if use_prior else 0.5
    a_BA = 1.0 / mean_block_nt                     # exit protected, per nt
    a_AB = a_BA * occupancy / (1.0 - occupancy)    # enter protected, per nt
    T  = np.array([[1 - a_AB, a_AB],
                   [a_BA,     1 - a_BA]])
    pi = np.array([1 - occupancy, occupancy])

    # gap correction: transition across d nt = T^d
    Tstep = [np.linalg.matrix_power(T, max(1, int(coords[k] - coords[k-1])))
             for k in range(1, n)]

    def emis(k):
        _tx, _rp, bit, pA_i, pB_i = ordered[k][1]
        return np.array([pA_i if bit else 1 - pA_i,
                         pB_i if bit else 1 - pB_i])

    # ---- forward-backward, scaled ---------------------------------------
    alpha = np.zeros((n, 2)); c = np.zeros(n)
    v = pi * emis(0); c[0] = v.sum(); alpha[0] = v / c[0]
    for k in range(1, n):
        v = (alpha[k-1] @ Tstep[k-1]) * emis(k)
        c[k] = v.sum(); alpha[k] = v / c[k]
    beta = np.zeros((n, 2)); beta[-1] = 1.0
    for k in range(n-2, -1, -1):
        beta[k] = (Tstep[k] @ (emis(k+1) * beta[k+1])) / c[k+1]
    post = alpha * beta
    post /= post.sum(1, keepdims=True)

    # ---- map back to READ ORDER by index, not by coordinate --------------
    pB_by_readidx  = {ri: float(post[j, 1]) for j, (ri, _s) in enumerate(ordered)}
    sur_by_readidx = {ri: -math.log10(max(float(c[j]), 1e-300))
                      for j, (ri, _s) in enumerate(ordered)}

    positions, txs, labels = [], [], []
    A, B, lA, lB, sur, eds, win_n = [], [], [], [], [], [], []
    for ri, s in enumerate(sites):                  # read order
        tx, ref_pos, bit, pA_i, pB_i = s
        pB_post = pB_by_readidx[ri]
        positions.append(ref_pos); txs.append(tx)
        A.append(1.0 - pB_post);  B.append(pB_post)
        lA.append(math.log10(pA_i if bit else 1 - pA_i))   # per-site emission
        lB.append(math.log10(pB_i if bit else 1 - pB_i))
        sur.append(sur_by_readidx[ri])                     # -log10 P(bit | past)
        labels.append("B" if pB_post >= 0.5 else "A")
        eds.append(bit); win_n.append(n)

    yield {"read_id": read_id, "chrom": chrom, "absolute_indices": positions,
           "tx": txs, "labels": labels, "P_A": A, "P_B": B,
           "logL_A": lA, "logL_B": lB, "surprise": sur,
           "edits": eds, "win_n": win_n}


def write_shadow_calls_to_df(gene, df_qry, records, read_edits,
                             ref_cov, gpos_to_tx, tx_to_gpos,
                             prob_threshold=0.5, min_win_n=1,
                             cds_start_tx=0, cds_end_tx=None):
    """
    Shadow calls in read-level format, mirroring the source parquet schema.
    Site-anchored: a site is "in shadow" when its posterior P_B >= prob_threshold.
    Serves all three models (window / Markov / HMM) via the shared P_B schema.
    """
    qry_by_id = {row.read_id: row for row in df_qry.itertuples()}

    def region_of(txp):
        if txp < cds_start_tx:
            return "UTR5"
        if cds_end_tx is not None and txp >= cds_end_tx:
            return "UTR3"
        return "CDS"

    out_rows = []
    for rec in records:
        rid = rec["read_id"]

        prot_idx = [k for k in range(len(rec["tx"]))
                    if rec["P_B"][k] >= prob_threshold
                    and rec["win_n"][k] >= min_win_n]
        if not prot_idx:
            continue

        src = qry_by_id.get(rid)
        if src is None:
            continue
        src = src._asdict()

        edit_string = src["edit_string"]
        abs_idx     = src["absolute_indices"]
        protected_tx = {rec["tx"][k] for k in prot_idx}
        shadow_chars = []
        for i, ref_pos in enumerate(abs_idx):
            if ref_pos is None or (isinstance(ref_pos, float) and ref_pos != ref_pos):
                shadow_chars.append("2"); continue
            ref_pos = int(ref_pos)
            if i >= len(edit_string) or edit_string[i] == "2":
                shadow_chars.append("2"); continue
            tx = gpos_to_tx.get(ref_pos)
            shadow_chars.append("1" if tx in protected_tx else "0")
        shadow_string = "".join(shadow_chars)

        tx_pos  = [rec["tx"][k]                   for k in prot_idx]
        gpos    = [tx_to_gpos.get(rec["tx"][k])   for k in prot_idx]
        regions = [region_of(t)                   for t in tx_pos]
        P_B     = [rec["P_B"][k]                  for k in prot_idx]
        P_A     = [rec["P_A"][k]                  for k in prot_idx]
        edit    = [rec["edits"][k]                for k in prot_idx]
        rcov    = [ref_cov.get(rec["tx"][k])      for k in prot_idx]

        row = dict(src)
        row.pop("Index", None)
        row.update({
            "shadow_gene":    gene,
            "shadow_string":  shadow_string,
            "shadow_tx_pos":  tx_pos,
            "shadow_gpos":    gpos,
            "shadow_region":  regions,
            "shadow_P_B":     P_B,
            "shadow_P_A":     P_A,
            "shadow_edit":    edit,
            "shadow_ref_cov": rcov,
            "n_shadow_sites": len(prot_idx),
            "n_sites_utr5":   regions.count("UTR5"),
            "n_sites_cds":    regions.count("CDS"),
            "n_sites_utr3":   regions.count("UTR3"),
            "max_P_B":        max(P_B),
        })
        out_rows.append(row)

    return pd.DataFrame(out_rows)

def plot_pb_by_tx_pyx(gene_name, df, his_positions, tx_lo, tx_hi, pdf_path,
                      label1="A", label2="B", ref_cov=None,
                      tx_col="tx", pb_col="P_B", edit_col="edits",
                      pct=(5, 95), num_reads=10):
    from pyx import canvas, graph, color, style, text as pyx_text
    from collections import defaultdict
    import numpy as np

    col_qry  = color.cmyk(1, 0.5, 0, 0)
    col_his  = color.cmyk(0, 1, 1, 0)
    col_sig  = color.cmyk(0, 0, 0, 0.4)
    col_edit = color.cmyk(0, 0, 0, 1)
    col_no   = color.cmyk(0, 0.8, 1, 0)
    col_cov  = color.cmyk(0.7, 0, 0.7, 0.1)   # green — background coverage

    pdf_path = str(pdf_path)

    x_min, x_max = tx_lo, tx_hi
    panel_w  = 12
    tick_h   = 0.25
    read_h   = 1.2
    gap      = 0.6
    cov_h    = 1.5

    # aggregate per-tx distribution of P_B across all reads (meta)
    pb_by_tx = defaultdict(list)
    for tx_list, pb_list in zip(df[tx_col], df[pb_col]):
        for t, pb in zip(tx_list, pb_list):
            pb_by_tx[int(t)].append(pb)
    items    = sorted(pb_by_tx.items())
    xs       = [t for t, _ in items]
    mean_pb  = [float(np.mean(v))               for _, v in items]
    lo_pb    = [float(np.percentile(v, pct[0])) for _, v in items]
    hi_pb    = [float(np.percentile(v, pct[1])) for _, v in items]
    cov_self = {t: len(v) for t, v in items}

    c         = canvas.canvas()
    meta_ypos = num_reads * (read_h + tick_h + gap) + gap * 2

    g_meta = graph.graphxy(
        width=panel_w, height=3, xpos=0, ypos=meta_ypos,
        x=graph.axis.linear(min=x_min, max=x_max,
                            title="Position Along Transcript (nt)"),
        y=graph.axis.linear(min=0, max=1, title="P(Shadow)"),
    )
    for hp in his_positions:
        g_meta.plot(
            graph.data.function(f"x(y)={hp}", min=0, max=1),
            [graph.style.line([col_his, style.linewidth.thin,
                               style.linestyle.solid])])
    g_meta.plot(
        graph.data.function("y(x)=0.5", min=x_min, max=x_max),
        [graph.style.line([col_sig, style.linewidth.thin,
                           style.linestyle.dashed])])
    for yb in (lo_pb, hi_pb):
        if xs:
            g_meta.plot(graph.data.points(list(zip(xs, yb)), x=1, y=2),
                        [graph.style.line([col_qry, style.linewidth.thin,
                                           style.linestyle.dotted])])
    if xs:
        g_meta.plot(graph.data.points(list(zip(xs, mean_pb)), x=1, y=2),
                    [graph.style.line([col_qry, style.linewidth.normal,
                                       style.linestyle.solid])])
    c.insert(g_meta)

    # ── Background coverage panel above the meta plot ─────────────────────────
    cov_ypos = meta_ypos + 3 + gap
    title_y  = cov_ypos + 3 + 0.4   # default if no coverage panel drawn

    cov_source = ref_cov if ref_cov else cov_self
    if cov_source:
        cov_items = sorted(cov_source.items())
        cov_x     = [tx for tx, _ in cov_items]
        cov_y     = [n  for _, n  in cov_items]
        cov_y_max = max(cov_y) * 1.1 if cov_y else 1.0

        g_cov = graph.graphxy(
            width=panel_w, height=cov_h,
            xpos=0, ypos=cov_ypos,
            x=graph.axis.linkedaxis(g_meta.axes["x"]),
            y=graph.axis.linear(min=0, max=cov_y_max,
                                title="Ref cov"),
        )
        for hp in his_positions:
            g_cov.plot(
                graph.data.function(f"x(y)={hp}", min=0, max=cov_y_max),
                [graph.style.line([col_his, style.linewidth.thin,
                                   style.linestyle.solid])])
        if len(cov_items) > 0:
            g_cov.plot(
                graph.data.points(list(zip(cov_x, cov_y)), x=1, y=2),
                [graph.style.line([col_cov, style.linewidth.normal,
                                   style.linestyle.solid])])
        c.insert(g_cov)
        title_y = g_cov.ypos + g_cov.height + 0.4

    c.text(g_meta.xpos + g_meta.width / 2., title_y,
           f"{gene_name} - {label2}",
           [pyx_text.halign.center, pyx_text.size.normalsize])

    n_show = min(num_reads, len(df))
    for jj in range(n_show):
        row = df.iloc[jj]
        tx_r = list(row[tx_col]); pb_r = list(row[pb_col])
        if not tx_r:
            continue
        order = np.argsort(tx_r)
        tx_s = [tx_r[k] for k in order]; pb_s = [pb_r[k] for k in order]

        ypos = (num_reads - 1 - jj) * (read_h + tick_h + gap)

        g_read = graph.graphxy(
            width=panel_w, height=read_h, xpos=0, ypos=ypos + tick_h,
            x=graph.axis.linkedaxis(g_meta.axes["x"]),
            y=graph.axis.linear(min=0, max=1, title=""),
        )
        for hp in his_positions:
            g_read.plot(
                graph.data.function(f"x(y)={hp}", min=0, max=1),
                [graph.style.line([col_his, style.linewidth.thin,
                                   style.linestyle.solid])])
        g_read.plot(
            graph.data.function("y(x)=0.5", min=x_min, max=x_max),
            [graph.style.line([col_sig, style.linewidth.thin,
                               style.linestyle.dashed])])
        g_read.plot(
            graph.data.points(list(zip(tx_s, pb_s)), x=1, y=2),
            [graph.style.line([col_qry, style.linewidth.thin,
                               style.linestyle.solid])])
        c.insert(g_read)

        g_ticks = graph.graphxy(
            width=panel_w, height=tick_h, xpos=0, ypos=ypos,
            x=graph.axis.linkedaxis(g_meta.axes["x"]),
            y=graph.axis.linear(min=0, max=1),
        )
        edit_r = list(row[edit_col]) if edit_col in df.columns else None
        for k in order:
            v = int(edit_r[k]) if edit_r is not None else (0 if pb_r[k] >= 0.5 else 1)
            col = col_edit if v == 1 else col_no
            g_ticks.plot(
                graph.data.function(f"x(y)={tx_r[k]}", min=0, max=1),
                [graph.style.line([col, style.linewidth.thin,
                                   style.linestyle.solid])])
        c.insert(g_ticks)

        label_txt = str(row["read_id"])[:20] if "read_id" in df.columns else f"read {jj}"
        c.text(panel_w + 0.15, ypos + tick_h + read_h / 2.,
               label_txt, [pyx_text.valign.middle, pyx_text.size.tiny])

    c.writePDFfile(pdf_path)


def plot_signed_log_pyx(gene_name, df, his_positions, tx_lo, tx_hi, pdf_path,
                        label1="A", label2="B", label3="sample",
                        ref_cov_A=None, ref_cov_B=None,
                        tx_col="tx", pa_col="P_A", pb_col="P_B", edit_col="edits",
                        pct=(5, 95), num_reads=10, eps=1e-6):
    from pyx import canvas, graph, color, style, text as pyx_text
    from collections import defaultdict
    import numpy as np, math

    def nice_parter(lim):
        """explicit tick spacing so PyX never has to SEARCH for a partition"""
        import math
        from pyx import graph
        if not (lim > 0) or math.isinf(lim) or math.isnan(lim):
            return None
        raw = lim / 2.0
        mag = 10.0 ** math.floor(math.log10(raw))
        for m in (1, 2, 2.5, 5, 10):
            if raw <= m * mag:
                return graph.axis.parter.linear(tickdists=[m * mag])
        return None

    def safe_lim(v, floor=0.2):
        """guard against nan / inf / zero collapsing the axis"""
        import math
        try:
            v = float(v)
        except Exception:
            return floor
        if math.isnan(v) or math.isinf(v) or v <= 0:
            return floor
        return max(v, floor)

    col_qry  = color.cmyk(1, 0.5, 0, 0)       # blue  — P_A trace (flipped, +)
    col_pb   = color.cmyk(0.4, 1, 0, 0)       # purple — P_B trace (unflipped, -)
    col_his  = color.cmyk(0, 1, 1, 0)
    col_sig  = color.cmyk(0, 0, 0, 0.4)
    col_edit = color.cmyk(0, 0, 0, 1)
    col_no   = color.cmyk(0, 0.8, 1, 0)
    col_covA = color.cmyk(0.7, 0, 0.7, 0.1)   # green — ref A coverage
    col_covB = color.cmyk(0, 0.5, 1, 0.1)     # orange — ref B coverage

    pdf_path = str(pdf_path)
    x_min, x_max = tx_lo, tx_hi
    panel_w, tick_h, read_h, gap, cov_h = 12, 0.25, 1.2, 0.6, 1.5

    # the transform: log-scale the less-likely probability, sign per the rule.
    # NO branching needed -- the likely class collapses to ~0 on its own.
    def yA(pa): return -math.log10(max(pa, eps))     # P_A less likely -> flip -> positive
    def yB(pb): return  math.log10(max(pb, eps))     # P_B less likely -> keep -> negative

    # aggregate per-tx distributions of both transformed traces across reads
    ya_by, yb_by = defaultdict(list), defaultdict(list)
    for tx_l, pa_l, pb_l in zip(df[tx_col], df[pa_col], df[pb_col]):
        for t, pa, pb in zip(tx_l, pa_l, pb_l):
            ya_by[int(t)].append(yA(pa)); yb_by[int(t)].append(yB(pb))

    items   = sorted(ya_by)
    xs      = list(items)
    ya_mean = [float(np.mean(ya_by[t]))               for t in items]
    ya_lo   = [float(np.percentile(ya_by[t], pct[0])) for t in items]
    ya_hi   = [float(np.percentile(ya_by[t], pct[1])) for t in items]
    yb_mean = [float(np.mean(yb_by[t]))               for t in items]
    yb_lo   = [float(np.percentile(yb_by[t], pct[0])) for t in items]
    yb_hi   = [float(np.percentile(yb_by[t], pct[1])) for t in items]
    cov_self = {t: len(ya_by[t]) for t in items}

    # INDEPENDENT limits: blue caps ~+3 (evidence asymmetry), purple floors at -6 (eps)
    hi_vals  = [abs(v) for v in ya_hi + ya_mean] or [1.0]
    lo_vals  = [abs(v) for v in yb_lo + yb_mean] or [1.0]
    y_lim_hi = safe_lim(max(hi_vals) * 1.1)
    y_lim_lo = safe_lim(max(lo_vals) * 1.1)

    def draw_his(g):
        for hp in his_positions:
            g.plot(graph.data.function(f"x(y)={hp}", min=-y_lim_lo, max=y_lim_hi),
                   [graph.style.line([col_his, style.linewidth.thin, style.linestyle.solid])])
    def draw_zero(g):
        g.plot(graph.data.function("y(x)=0", min=x_min, max=x_max),
               [graph.style.line([col_sig, style.linewidth.thin, style.linestyle.dashed])])

    c = canvas.canvas()
    meta_ypos = num_reads * (read_h + tick_h + gap) + gap * 2

    # ── meta panel: both traces, mean + percentile bounds ────────────────────
    g_meta = graph.graphxy(
        width=panel_w, height=3, xpos=0, ypos=meta_ypos,
        x=graph.axis.linear(min=x_min, max=x_max, title="Position Along Transcript (nt)"),
        y=graph.axis.linear(min=-y_lim_lo, max=y_lim_hi, title="log10 P(Protection)",
                            parter=nice_parter(y_lim_lo + y_lim_hi)),
    )
    draw_his(g_meta); draw_zero(g_meta)
    if xs:
        for yb in (ya_lo, ya_hi):
            g_meta.plot(graph.data.points(list(zip(xs, yb)), x=1, y=2),
                        [graph.style.line([col_qry, style.linewidth.thin, style.linestyle.dotted])])
        for yb in (yb_lo, yb_hi):
            g_meta.plot(graph.data.points(list(zip(xs, yb)), x=1, y=2),
                        [graph.style.line([col_pb, style.linewidth.thin, style.linestyle.dotted])])
        g_meta.plot(graph.data.points(list(zip(xs, ya_mean)), x=1, y=2),
                    [graph.style.line([col_qry, style.linewidth.normal, style.linestyle.solid])])
        g_meta.plot(graph.data.points(list(zip(xs, yb_mean)), x=1, y=2),
                    [graph.style.line([col_pb, style.linewidth.normal, style.linestyle.solid])])
    c.insert(g_meta)

    # directional labels, horizontal, inside the top-left / bottom-left corners
    lab_x = g_meta.xpos + 0.2
    c.text(lab_x, meta_ypos + 3 - 0.25, "log P(protected)",
           [pyx_text.halign.left, pyx_text.valign.top, pyx_text.size.small])
    c.text(lab_x, meta_ypos + 0.25, "log P(unprotected)",
           [pyx_text.halign.left, pyx_text.valign.bottom, pyx_text.size.small])

    # ── coverage panel: both reference tracks ────────────────────────────────
    cov_ypos = meta_ypos + 3 + gap
    title_y  = cov_ypos + 3 + 0.4

    cov_tracks = []
    if ref_cov_A:
        cov_tracks.append((ref_cov_A, col_covA, label1))
    if ref_cov_B:
        cov_tracks.append((ref_cov_B, col_covB, label2))
    if not cov_tracks and cov_self:
        cov_tracks.append((cov_self, col_covA, "query"))

    if cov_tracks:
        cov_y_max = safe_lim(max(max(d.values()) for d, _, _ in cov_tracks) * 1.1, floor=1.0)
        g_cov = graph.graphxy(width=panel_w, height=cov_h, xpos=0, ypos=cov_ypos,
                              x=graph.axis.linkedaxis(g_meta.axes["x"]),
                              y=graph.axis.linear(min=0, max=cov_y_max, title="Ref cov",
                                                  parter=nice_parter(cov_y_max)))

        for hp in his_positions:
            g_cov.plot(graph.data.function(f"x(y)={hp}", min=0, max=cov_y_max),
                       [graph.style.line([col_his, style.linewidth.thin, style.linestyle.solid])])
        for d, col, _lab in cov_tracks:
            g_cov.plot(graph.data.points(sorted(d.items()), x=1, y=2),
                       [graph.style.line([col, style.linewidth.normal, style.linestyle.solid])])
        c.insert(g_cov)

        # inline legend, top-left inside the coverage panel
        for kk, (_d, col, lab) in enumerate(cov_tracks):
            c.text(g_cov.xpos + 0.2, g_cov.ypos + cov_h - 0.18 - kk * 0.28, lab,
                   [pyx_text.halign.left, pyx_text.valign.top, pyx_text.size.tiny, col])
        title_y = g_cov.ypos + g_cov.height + 0.4

    c.text(g_meta.xpos + g_meta.width / 2., title_y, f"{gene_name} - {label3}",
           [pyx_text.halign.center, pyx_text.size.normalsize])

    # ── individual read panels: upper trace / edit ticks / lower trace ───────
    n_show = min(num_reads, len(df))
    for jj in range(n_show):
        row = df.iloc[jj]
        tx_r = list(row[tx_col]); pa_r = list(row[pa_col]); pb_r = list(row[pb_col])
        if not tx_r:
            continue
        order = np.argsort(tx_r)
        tx_s  = [tx_r[k] for k in order]
        yaS   = [yA(pa_r[k]) for k in order]
        ybS   = [yB(pb_r[k]) for k in order]

        ypos   = (num_reads - 1 - jj) * (read_h + tick_h + gap)
        half_h = read_h / 2.0

        # lower half: unprotected trace (negative), below the ticks
        g_lo = graph.graphxy(width=panel_w, height=half_h, xpos=0, ypos=ypos,
                             x=graph.axis.linkedaxis(g_meta.axes["x"]),
                             y=graph.axis.linear(min=-y_lim_lo, max=0, title="",
                                                 parter=nice_parter(y_lim_lo)))
        for hp in his_positions:
            g_lo.plot(graph.data.function(f"x(y)={hp}", min=-y_lim_lo, max=0),
                      [graph.style.line([col_his, style.linewidth.thin, style.linestyle.solid])])
        g_lo.plot(graph.data.points(list(zip(tx_s, ybS)), x=1, y=2),
                  [graph.style.line([col_pb, style.linewidth.thin, style.linestyle.solid])])
        c.insert(g_lo)

        # upper half: protected trace (positive), above the ticks
        g_hi = graph.graphxy(width=panel_w, height=half_h, xpos=0,
                             ypos=ypos + half_h + tick_h,
                             x=graph.axis.linkedaxis(g_meta.axes["x"]),
                             y=graph.axis.linear(min=0, max=y_lim_hi, title="",
                                                 parter=nice_parter(y_lim_hi)))
        for hp in his_positions:
            g_hi.plot(graph.data.function(f"x(y)={hp}", min=0, max=y_lim_hi),
                      [graph.style.line([col_his, style.linewidth.thin, style.linestyle.solid])])
        g_hi.plot(graph.data.points(list(zip(tx_s, yaS)), x=1, y=2),
                  [graph.style.line([col_qry, style.linewidth.thin, style.linestyle.solid])])
        c.insert(g_hi)

        # centre strip: edit ticks, no y-axis labels
        g_ticks = graph.graphxy(width=panel_w, height=tick_h, xpos=0, ypos=ypos + half_h,
            x=graph.axis.linkedaxis(g_meta.axes["x"]),
            y=graph.axis.linear(min=0, max=1, parter=None, painter=None))
        edit_r = list(row[edit_col]) if edit_col in df.columns else None
        for k in order:
            v = int(edit_r[k]) if edit_r is not None else (0 if pa_r[k] < 0.5 else 1)
            col = col_edit if v == 1 else col_no
            g_ticks.plot(graph.data.function(f"x(y)={tx_r[k]}", min=0, max=1),
                         [graph.style.line([col, style.linewidth.thin, style.linestyle.solid])])
        c.insert(g_ticks)

        label_txt = str(row["read_id"])[:20] if "read_id" in df.columns else f"read {jj}"
        c.text(panel_w + 0.15, ypos + half_h + tick_h / 2., label_txt,
               [pyx_text.valign.middle, pyx_text.size.tiny])

    # print(f"y_lim_hi={y_lim_hi!r}  y_lim_lo={y_lim_lo!r}  "
    #       f"cov_y_max={cov_y_max if cov_tracks else None!r}")
    # c.writePDFfile(pdf_path)
    c.writePDFfile(pdf_path)

def parse_args():
    p = argparse.ArgumentParser(
        description="Bernoulli Naive Bayes Shadow Classifier."
    )
    p.add_argument("--parquet", required=True)
    p.add_argument("--label", default="sample")
    p.add_argument("--model", required=True, default=None, help="Trained model stored as pickle file")
    p.add_argument("--ref", required=True)
    p.add_argument("--gtf", required=True)
    p.add_argument("--output", default="HMM")
    p.add_argument("--window", type=int,   default=30)
    p.add_argument("--min_query_reads", type=int, default=10,
                   help="Minimum number of query (spanning) reads a gene must "
                        "have to be analyzed (default: 10). Separate from "
                        "--min_coverage, which gates the reference background.")
    p.add_argument("--num_reads",    type=int,   default=10)
    p.add_argument("--top_n_plots",  type=int,   default=10,
                   help="Number of genes to plot, ranked by number of "
                        "CDS-spanning query reads (default: 10)")
    p.add_argument("--require_his_codon", action="store_true",
                   help="Only process genes that contain at least one His "
                        "codon (CAT/CAC).")
    p.add_argument("--min_edit_freq", type=float, default=0.0,
                   help="Only consider query reads whose per-read "
                        "global_edit_freq is >= this value. Filters out "
                        "unedited/poorly-edited molecules that carry no "
                        "protection signal. Applied to query reads only, "
                        "not the background estimate. Default: 0.0 (off)")

    p.add_argument("--cds_spanning", action="store_true",
                   help="Only include reads whose alignment spans the full "
                        "CDS of the gene.")
    return p.parse_args()

