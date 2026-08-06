"""
riboseqPsiteCalibration.py -- Liam Tran, August 2026

Empirically derive the ribosome P-site offset (nt from a ribo-seq read's 5'
end to the ribosome's P-site) for a given set of read lengths, from real
Wu_2019 ribo-seq BAM(s) + the same yeast GTF the mRNP-shadowing HMM pipeline
uses. No P-site-offset-calling code existed anywhere in this codebase or in
/data16/arriberelab-git (checked) -- this derives it directly rather than
assuming a value from the literature, since the exact offset is sensitive to
library prep / nuclease digestion conditions specific to this dataset.

Anchored on the STOP codon, not the start codon: ribosomes queue up and
pause during termination regardless of gene, producing a real, sharp
5'-end pileup a fixed distance upstream of the stop codon -- and unlike the
start codon, this needs no UTR annotation at all (just cds_length, always
exact). The start-codon/5'UTR-initiation-pileup approach that's standard in
the literature does NOT work on this particular GTF: checked directly --
only 4 of 5106 genes have ANY annotated 5'UTR (parse_gtf's "exon" features
here are essentially CDS-only), so there's no 5'UTR read density to look
for a pileup in at all.

Earlier version of this script tried a "frame purity by candidate offset"
sweep (shift each read's 5' position by a candidate offset, see which
offset gives the most lopsided mod-3 histogram) and it was mathematically
broken: shifting every element of a fixed list by a constant only ROTATES
which residue-class bucket is largest, it can never change the max
bucket's *size* -- so that metric is provably identical at every offset
and cannot discriminate between them. Fixed here by finding the offset two
different, valid ways instead: (1) the position of the sharpest 5'-end
pileup peak upstream of the stop codon (a real spatial signal, not
offset-invariant), and (2) an independent, unparameterized phase check --
the raw (untransformed) mod-3 histogram of every read's stop-relative
position across the whole window, which should show one dominant residue
class if real triplet periodicity is present. The peak-derived offset's
own residue class should agree with whichever bucket dominates (2); this
is reported so a human can sanity-check the two signals agree before
trusting either one.

Also tries BOTH orientation hypotheses (reads aligning sense vs antisense
to their gene) since that library-prep strandedness convention isn't known
up front -- real signal should only show up under the correct one.

Run:
  python3 riboseqPsiteCalibration.py bamList.txt gtfFile outPrefix
where bamList.txt is e.g. /data16/liam/working/260804_riboSeq_vs_PS/riboSeqBam.txt
-- whitespace-delimited condition/rep/path rows. Reps sharing the same
condition label are pooled together, but each condition is calibrated
SEPARATELY (own plots, own offset call) -- P-site offset could genuinely
differ by condition (e.g. +3AT stress altering nuclease digestion
characteristics), so pooling across conditions would hide that rather
than reveal it. See load_ribo_bam_groups. gtfFile is the yeast GTF, and
outPrefix names the output plots (one set per condition, suffixed with
the condition label).
"""
import sys, collections
import numpy as np
import pysam


def load_ribo_bam_groups(path):
    """
    {condition_label: [bam_paths]} -- groups every row of a whitespace-
    delimited list file by its first column (the riboSeqBam.txt
    convention: "condition  rep  path"), so replicates sharing the same
    label get pooled together but different conditions stay separate.
    Bare-path rows (no condition/rep columns) all fall into one shared
    "unlabeled" group.
    """
    groups = collections.defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) >= 3:
                cond, _rep, bam_path = parts[0], parts[1], parts[2]
                groups[cond].append(bam_path)
            else:
                groups["unlabeled"].append(parts[0])
    return dict(groups)

from runHMMPerGene import parse_gtf, cds_length, _full_tx_map

TARGET_LENGTHS = (21, 22, 28, 29)
WINDOW         = (-60, 15)   # position window relative to the STOP codon (0 = first nt of stop)
MIN_CDS_LEN    = 150         # skip very short CDSs -- not enough room for a clean window


def _gpos_to_tx_full(gene, ref_fasta):
    """{gpos: tx} for EVERY transcript base (not just A/T) -- inverse of
    _full_tx_map, reused as-is from runHMMPerGene.py so tx=0 means the same
    thing (first nt of the start codon) everywhere in this codebase.

    _full_tx_map's UTR-annotation-based extension (gene["utr5"]/["utr3"])
    was replaced with a flank_5p/flank_3p CDS-padding scheme (see
    runHMMPerGene.py's compute_flank_caps) to work around this GTF having
    essentially no annotated UTRs -- flank_5p=0, flank_3p=0 here
    reproduces this script's own prior include_utrs=True behavior (this
    GTF's annotated UTRs are themselves negligible, so the difference is
    tiny), left as a minimal call-site fix rather than adopting the deeper
    fix, even though this script's own WINDOW needs real data up to +15nt
    past the stop codon that today's ~3nt-wide annotation can't supply."""
    full = _full_tx_map(gene, ref_fasta, flank_5p=0, flank_3p=0)
    return {gpos: tx for tx, (gpos, _base) in full.items()}


def collect_stop_relative_positions(bam_paths, genes, ref_fasta,
                                    target_lengths=TARGET_LENGTHS, window=WINDOW):
    """
    {orientation: {length: [pos_relative_to_stop, ...]}} pooled across every
    gene and every BAM in bam_paths. pos_relative_to_stop = tx - cds_length
    (0 = first nt of the stop codon, negative = upstream/within the CDS).
    orientation in ("sense", "antisense"). Only uniquely-mapped reads
    (NH==1) are kept -- multi-mappers can't be unambiguously placed on one
    gene's tx coordinate.
    """
    positions = {"sense": collections.defaultdict(list),
                "antisense": collections.defaultdict(list)}
    lo, hi = window

    # per-gene tx maps + cds length, built once, reused across every BAM
    gene_info = {}
    for gname, gene in genes.items():
        cl = cds_length(gene)
        if cl < MIN_CDS_LEN:
            continue
        gpos_to_tx = _gpos_to_tx_full(gene, ref_fasta)
        if gpos_to_tx:
            gene_info[gname] = (gpos_to_tx, cl)

    for bam_path in bam_paths:
        print(f"  scanning {bam_path} ...", file=sys.stderr)
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for gname, (gpos_to_tx, cl) in gene_info.items():
                gene = genes[gname]
                for read in bam.fetch(gene["chrom"], gene["gene_start"], gene["gene_end"]):
                    if read.is_unmapped:
                        continue
                    if read.query_length not in target_lengths:
                        continue
                    if read.has_tag("NH") and read.get_tag("NH") != 1:
                        continue

                    if gene["strand"] == "+":
                        five_prime_gpos = read.reference_start
                    else:
                        five_prime_gpos = read.reference_end - 1
                    tx = gpos_to_tx.get(five_prime_gpos)
                    if tx is None:
                        continue
                    rel = tx - cl
                    if not (lo <= rel <= hi):
                        continue

                    read_is_sense = (read.is_reverse == (gene["strand"] == "-"))
                    orientation = "sense" if read_is_sense else "antisense"
                    positions[orientation][read.query_length].append(rel)

    return positions


def find_peak_and_phase(rel_positions, window=WINDOW):
    """
    (peak_pos, peak_prominence, phase_counts, dominant_phase, phase_purity)
    for one (orientation, length)'s pooled list of stop-relative positions.

    peak_pos: the single most common position in the window -- the
    termination-queueing pileup's location, so P-site offset = -peak_pos.
    peak_prominence: peak height / median height of the other positions --
    a rough confidence signal (much >1 means a real, sharp pileup; near 1
    means no discernible peak, i.e. too little/noisy data for this length).

    phase_counts / dominant_phase / phase_purity: histogram of RAW
    (untransformed) positions by (position mod 3) across the WHOLE window
    -- unlike the broken offset-sweep this replaces, this has no free
    parameter, so it's a valid, independent periodicity signal. Its
    dominant_phase should match peak_pos % 3 if the peak is real and not
    noise; disagreement is a red flag to look at the plot before trusting
    either number.
    """
    lo, hi = window
    counts = collections.Counter(rel_positions)
    if not counts:
        return None, 0.0, (0, 0, 0), None, 0.0

    peak_pos = max(counts, key=counts.get)
    other_vals = [counts.get(p, 0) for p in range(lo, hi + 1) if p != peak_pos]
    median_other = float(np.median(other_vals)) if other_vals else 0.0
    prominence = counts[peak_pos] / median_other if median_other > 0 else float("inf")

    phase_counts = [0, 0, 0]
    for pos, n in counts.items():
        phase_counts[pos % 3] += n
    total = sum(phase_counts)
    dominant_phase = int(np.argmax(phase_counts))
    phase_purity = phase_counts[dominant_phase] / total if total else 0.0

    return peak_pos, prominence, tuple(phase_counts), dominant_phase, phase_purity


def plot_calibration(positions, pdf_prefix, window=WINDOW, target_lengths=TARGET_LENGTHS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = window
    edges = list(range(lo, hi + 1))
    summary = {}

    for orientation in ("sense", "antisense"):
        by_length = positions[orientation]
        fig, axes = plt.subplots(len(target_lengths), 1,
                                 figsize=(8, 2.6 * len(target_lengths)), squeeze=False)
        axes = axes[:, 0]
        summary[orientation] = {}

        for ax, length in zip(axes, target_lengths):
            rel_list = by_length.get(length, [])
            peak_pos, prom, phase_counts, dom_phase, phase_purity = find_peak_and_phase(
                rel_list, window)
            summary[orientation][length] = {
                "n": len(rel_list), "peak_pos": peak_pos, "prominence": prom,
                "phase_counts": phase_counts, "dominant_phase": dom_phase,
                "phase_purity": phase_purity,
                "offset_from_peak": (-peak_pos if peak_pos is not None else None),
            }

            counts = collections.Counter(rel_list)
            n_total = len(rel_list) or 1
            ys = [1e6 * counts.get(e, 0) / n_total for e in edges]
            ax.plot(edges, ys, color="tab:blue", linewidth=1)
            ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
            if peak_pos is not None:
                ax.axvline(peak_pos, color="tab:red", linewidth=0.8, linestyle=":")
            title = (f"len={length}nt (n={len(rel_list)})  "
                    f"peak@{peak_pos} (offset={-peak_pos if peak_pos is not None else 'n/a'}, "
                    f"prom={prom:.1f}x)  phase{dom_phase} purity={phase_purity:.2f}")
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("5' position rel. stop codon (0 = first nt of stop)")
            ax.set_ylabel("RPM")

        fig.suptitle(f"P-site calibration (stop-anchored) -- {orientation} orientation",
                    fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(f"{pdf_prefix}.{orientation}.png", dpi=150)
        plt.close(fig)
        print(f"Wrote {pdf_prefix}.{orientation}.png", file=sys.stderr)

    return summary


def main(args):
    bamListPath, gtfPath, outPrefix = args[0], args[1], args[2]
    refPath = args[3] if len(args) > 3 else "/data16/liam/genomes/210524_sacCer/210524_allChrs.fa"

    bam_groups = load_ribo_bam_groups(bamListPath)
    print(f"{len(bam_groups)} condition group(s): "
          f"{ {c: len(p) for c, p in bam_groups.items()} }", file=sys.stderr)

    genes = parse_gtf(gtfPath)
    print(f"{len(genes):,} genes parsed from GTF.", file=sys.stderr)
    ref_fasta = pysam.FastaFile(refPath)

    for condition, bam_paths in bam_groups.items():
        print(f"\n=== condition: {condition} ({len(bam_paths)} BAM(s)) ===", file=sys.stderr)
        print(f"  {bam_paths}", file=sys.stderr)

        positions = collect_stop_relative_positions(bam_paths, genes, ref_fasta)
        for orientation in ("sense", "antisense"):
            for length in TARGET_LENGTHS:
                n = len(positions[orientation].get(length, []))
                print(f"  {orientation} len={length}: n={n} reads in window", file=sys.stderr)

        summary = plot_calibration(positions, f"{outPrefix}.{condition}")

        print(f"\nPer (orientation, length) offset call (offset = -peak_pos) -- "
              f"condition={condition}:", file=sys.stderr)
        for orientation in ("sense", "antisense"):
            for length in sorted(summary[orientation]):
                s = summary[orientation][length]
                off = s['offset_from_peak']
                off_mod3 = (off % 3) if off is not None else 'n/a'
                print(f"  {orientation:10s} len={length}: n={s['n']:6d}  "
                      f"peak_pos={s['peak_pos']}  offset={off}  "
                      f"prominence={s['prominence']:.1f}x  "
                      f"dominant_phase={s['dominant_phase']} (purity={s['phase_purity']:.2f})  "
                      f"offset%3={off_mod3}",
                      file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
