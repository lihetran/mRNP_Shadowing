'''
October 5, 2025 LT

Dorado is now the recommended basecaller for Oxford Nanopore data. Guppy is no longer supported or maintained by ONT.
Dorado's output after basecalling is now an unaligned bam file which retains base modification information. This script has been updated to process Dorado output.
My previous scripts for aligning highly modified reads by TadA or APOBEC worked with Guppy fastq files. This script has been updated to work with Dorado unaligned bam files.

input: unaligned bam file from Dorado basecalling
output: aligned bam file to reference sequence

This script mutates all A's to G's in the reads and reference sequence to allow for better alignment of TadA edited reads.

Genes live on both strands of the genome, but a single-direction A->G mutation of the
reference only preserves alignability in the orientation it was mutated in: reverse-
complementing an A->G-collapsed sequence turns former-A positions into 'C' (not back into
'A'), while the reference's real T's (from complementing the gene's real A's) were never
touched. So comparing across that reverse-complement boundary produces a mismatch at every
real or edited A, and reads whose sense orientation requires a reverse-complement match
against the once-mutated reference are effectively unalignable (confirmed on real data:
minus-strand-gene reads: ~92% unmapped, ~8% mapped to the wrong strand, 0% correct).

Fix (mirrors the standard bisulfite-aligner two-conversion trick): build a second,
same-coordinate reference where T's are mutated to C's instead of A's to G's. A real A->G
edit on a minus-strand gene's sense (Crick) strand shows up as T->C when read off the
Watson/plus reference, so aligning the raw (not sense-flipped) read's T->C-mutated form
against this T->C reference correctly captures minus-strand-gene reads via a plain forward
match -- no reverse-complement/CIGAR surgery needed, since both mutated references share the
exact coordinate system of the original genome. Each read is aligned via both pathways and
whichever one actually mapped (usually only one does, per the asymmetry above) is kept, with
is_reverse forced to match which pathway won so downstream edit-calling (which branches on
is_reverse to know whether to look for A->G or T->C mismatches) sees the correct strand.

Splicing: for genomic DNA, alignment uses minimap2's plain map-ont preset (dorado_align_raw).
For spliced mRNA/cDNA reads that may span introns, pass splice_aware=True to align_reads
instead, which uses minimap2_align_splice_aware -- this calls dorado's own bundled minimap2
binary directly (bypassing `dorado aligner`'s CLI, whose --mm2-opts only forwards a fixed
allow-list of flags) with minimap2's splice preset AND a validated max-intron-length cap
(-G, default 3000bp for this yeast genome). The cap is not optional: `dorado aligner
-x splice` alone was confirmed on a real sequencing run to fabricate splice junctions at a
huge rate (24% of primary alignments, many with 5-19 fake "introns" each, at lengths
nowhere near real yeast biology) because minimap2's splice preset defaults -G to 200kb
(sized for mammalian genomes) and this script's own A->G/T->C mutation collapses sequence
complexity enough that minimap2 cheaply chains unrelated, distant, repeat-driven matches
together instead of leaving them unmapped. See minimap2_align_splice_aware's docstring for
the validated numbers and for build_junction_bed, the (optional but recommended) way to also
bias minimap2 toward real, GTF-annotated junctions via --junc-bed.
'''

import sys
import shutil
import pysam
import subprocess
import argparse
import numpy as np
import matplotlib.pyplot as plt
from logJosh import Tee

from Bio import SeqIO, Seq, SeqRecord
from pathlib import Path

def replaceCharacter(seq, char1, char2):
    n = len(seq)
    res = ""
    positions = []
    for i in range(n):
        if seq[i] != char1:
            res += seq[i]
        else:
            res += char2
            positions.append(i)

    return res, positions

def reverseComplement(seq):

    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    reverse_complement = "".join(complement.get(base, base) for base in reversed(seq.upper()))

    return reverse_complement

def mutateBamSenseAG(input_bam):
    """
    Reverse-complement each raw dorado read to its sense (mRNA-matching)
    orientation, then mutate A's to G's. Aligning this against the A->G
    mutated reference (mutateFastaAG) correctly captures reads from '+'
    strand genes via a plain forward match.

    read_dict stores the sense (unmutated) sequence, used to restore the
    correct query_sequence for '+'-strand hits after alignment.
    """
    mutated_bam = input_bam.replace('.bam', '_mutated_AG.bam')
    read_dict = {}
    with pysam.AlignmentFile(input_bam, "rb", check_sq=False) as in_bam:
        with pysam.AlignmentFile(mutated_bam, "wb", template=in_bam) as out_bam:
            for read in in_bam:
                seq = read.query_sequence
                if seq is not None:
                    sense_seq = reverseComplement(seq) # reverse complement the sequence bc of cDNA sequencing
                    new_seq, positions = replaceCharacter(sense_seq, 'A', 'G')
                    read.query_qualities = None
                    read.query_sequence = new_seq
                    out_bam.write(read)
                    read_dict[read.query_name] = sense_seq

    return mutated_bam, read_dict

def mutateBamAntisenseTC(input_bam):
    """
    Mutate each raw (as-sequenced, NOT reverse-complemented) dorado read's
    T's to C's. Aligning this against the T->C mutated reference
    (mutateFastaTC) correctly captures reads from '-' strand genes via a
    plain forward match -- see module docstring for why this is needed in
    addition to the A->G pathway above.

    read_dict stores the raw (unmutated) sequence. Note reverseComplement of
    the sense sequence used in the AG pathway equals this raw sequence, so
    this is exactly the correct query_sequence to restore for a read that
    ends up flagged is_reverse=True (minus-strand-gene) in the final output.
    """
    mutated_bam = input_bam.replace('.bam', '_mutated_TC.bam')
    read_dict = {}
    with pysam.AlignmentFile(input_bam, "rb", check_sq=False) as in_bam:
        with pysam.AlignmentFile(mutated_bam, "wb", template=in_bam) as out_bam:
            for read in in_bam:
                seq = read.query_sequence
                if seq is not None:
                    new_seq, positions = replaceCharacter(seq, 'T', 'C')
                    read.query_qualities = None
                    read.query_sequence = new_seq
                    out_bam.write(read)
                    read_dict[read.query_name] = seq

    return mutated_bam, read_dict

def _output_is_stale(input_path, output_path):
    """True if output_path needs to be (re)built: missing, empty, or older
    than input_path (the source was updated since output_path was built)."""
    if not output_path.exists() or output_path.stat().st_size == 0:
        return True
    return output_path.stat().st_mtime < input_path.stat().st_mtime

def mutateFastaAG(input_fasta):
    # create a mutated fasta file where all A's are changed to G's, same coordinates as input_fasta
    input_fasta = Path(input_fasta)
    output_fasta = input_fasta.with_name(input_fasta.stem + '_mutated_AG.fasta')
    if not _output_is_stale(input_fasta, output_fasta):
        print(f'  {output_fasta} already exists and is up to date; skipping.')
        return output_fasta
    with open(output_fasta, 'w') as out_fasta:
        for record in SeqIO.parse(input_fasta, 'fasta'):
            seq = str(record.seq)
            new_seq, positions = replaceCharacter(seq, 'A', 'G')
            new_record = SeqRecord.SeqRecord(Seq.Seq(new_seq), id=record.id, description=record.description)
            SeqIO.write(new_record, out_fasta, 'fasta')

    return output_fasta

def mutateFastaTC(input_fasta):
    # create a mutated fasta file where all T's are changed to C's, same coordinates as input_fasta
    input_fasta = Path(input_fasta)
    output_fasta = input_fasta.with_name(input_fasta.stem + '_mutated_TC.fasta')
    if not _output_is_stale(input_fasta, output_fasta):
        print(f'  {output_fasta} already exists and is up to date; skipping.')
        return output_fasta
    with open(output_fasta, 'w') as out_fasta:
        for record in SeqIO.parse(input_fasta, 'fasta'):
            seq = str(record.seq)
            new_seq, positions = replaceCharacter(seq, 'T', 'C')
            new_record = SeqRecord.SeqRecord(Seq.Seq(new_seq), id=record.id, description=record.description)
            SeqIO.write(new_record, out_fasta, 'fasta')

    return output_fasta

def build_junction_bed(gtf_path):
    """
    Convert a GTF/GFF3's gene models to BED12 via paftools.js gff2bed, for use
    as minimap2's --junc-bed. minimap2 pulls the exon/intron block structure
    out of the BED12 itself, so this needs no coordinate math of its own.

    Cached next to gtf_path (same staleness check as the mutated fastas):
    rebuilt only if missing/empty, or older than gtf_path.
    """
    gtf_path = Path(gtf_path)
    output_bed = gtf_path.with_suffix('.junctions.bed')
    if not _output_is_stale(gtf_path, output_bed):
        print(f'  {output_bed} already exists and is up to date; skipping.')
        return output_bed
    cmd = f'paftools.js gff2bed {gtf_path} > {output_bed}'
    print(cmd)
    subprocess.call(cmd, shell=True)
    return output_bed

def dorado_align_raw(mutated_bam, ref_fasta, tmp_bam, threads=0):
    """
    Run dorado aligner (minimap2 under the hood) and return the path to the
    raw (unfiltered) output bam. Non-splice-aware: -x map-ont, plain genomic
    read-vs-reference alignment, no intron-gap model. Correct for genomic
    DNA; for spliced mRNA/cDNA reads see minimap2_align_splice_aware and
    align_reads's splice_aware flag.

    threads: dorado aligner's own --threads (a top-level flag, separate from
    --mm2-opts), 0 = unlimited (dorado's own default). Set explicitly on a
    shared machine -- unlimited means dorado will grab as many of the box's
    cores as it can, which has previously monopolized a 48-core host
    entirely (~44 cores) while a labmate's job was also running on it.
    """
    cmd = f'dorado aligner {ref_fasta} {mutated_bam} --threads {threads} --mm2-opts "-x map-ont --secondary=no" > {tmp_bam}'
    print(cmd)
    subprocess.call(cmd, shell=True)
    return tmp_bam

def _find_dorado_bundled_minimap2():
    """
    Locate the minimap2 binary bundled alongside dorado's own executable
    (installed in the same bin/ directory), NOT whatever `minimap2` resolves
    to on PATH. `dorado aligner` is a thin wrapper around this exact bundled
    binary, so it's the only way to guarantee alignment behavior actually
    matches `dorado aligner`'s when adding flags its --mm2-opts wrapper
    won't forward (like -G). Confirmed on this environment: dorado bundles
    minimap2 2.28, while the system `minimap2` on PATH is 2.17 -- an
    11-version gap that produced materially different splice-mode results
    in testing (a real read that dorado aligner mapped with 7 fabricated
    splice junctions came back cleanly unmapped under 2.28 + -G3000, but
    aligned as a totally different, still-wrong shape under system 2.17 --
    the two builds are not interchangeable for this).
    """
    dorado_path = shutil.which('dorado')
    if dorado_path:
        bundled = Path(dorado_path).resolve().parent / 'minimap2'
        if bundled.exists():
            return str(bundled)
    raise FileNotFoundError(
        "Could not find the minimap2 binary bundled next to the `dorado` "
        "executable. Splice-aware alignment requires this exact binary -- "
        "see _find_dorado_bundled_minimap2's docstring for why the system "
        "`minimap2` on PATH isn't a safe substitute."
    )

def _discover_bam_tags(bam_path, sample_size=50):
    """
    Union of all two-letter tag names seen on the first sample_size reads of
    bam_path. Used to build samtools fastq's -T TAGLIST argument, since the
    installed samtools (1.15.1) doesn't support -T '*' (wildcard "all tags")
    -- confirmed it silently copies nothing when given '*'. Sampling rather
    than hardcoding a fixed tag list because which tags exist varies by run
    (e.g. barcode-demultiplexed bams carry cI/cS; plain dorado output
    doesn't) and by dorado/basecall-model version.
    """
    tags = set()
    with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
        for i, read in enumerate(bam):
            if i >= sample_size:
                break
            tags.update(tag for tag, _ in read.get_tags())
    return sorted(tags)

def minimap2_align_splice_aware(mutated_bam, ref_fasta, tmp_bam, junc_bed=None, max_intron_length=3000, threads=3):
    """
    Splice-aware alignment via dorado's own bundled minimap2 binary, called
    directly (bypassing `dorado aligner`'s CLI wrapper) as:
        samtools fastq -T <tags> mutated_bam | minimap2 -a -y -x splice
            -G<max_intron_length> [--junc-bed=...] ref_fasta -
        | samtools view -b -o tmp_bam
    `-y` copies the fastq header's tag comment (written by `-T`) back onto
    the output SAM records, so bam tags round-trip through the fastq step
    (verified: cI/cS and dorado's own qs/du/ns/... metadata tags all survive).

    Why bypass dorado aligner: -x splice alone (dorado aligner's own
    --mm2-opts, see dorado_align_raw's docstring for why the plain map-ont
    default is wrong for spliced reads) opens up spurious splice junctions
    at a huge rate on real data -- confirmed on an actual sequencing run
    (JANP-146): 24% of primary alignments (664k/2.7M reads) had >=1 'N'
    CIGAR op, many with 5-19 per read, at lengths (median 39kb, mean 63kb,
    max 855kb) nowhere near real yeast biology (real annotated introns:
    median 116bp, max 2483bp, n=380). Root cause: minimap2's splice preset
    defaults -G (max intron length) to 200kb, sized for mammalian genomes;
    combined with this script's own A->G/T->C mutation collapsing sequence
    complexity (worst in repeat-rich regions -- one confirmed case was
    FLO9, a notoriously repetitive subtelomeric gene), minimap2 cheaply
    chains together marginal, scattered local matches into fake
    "multi-exon" alignments spanning huge, implausible distances.

    max_intron_length=3000 is the validated cap: covers the real annotated
    max (2483bp) with margin, while being ~2 orders of magnitude below the
    fabricated junction lengths seen above. Validated on a 300-read sample
    of reads that had spurious junctions under -x splice alone: capping to
    -G3000 dropped 269/300 (90%) to zero junctions, left at most one
    junction (at a plausible 1867-2954bp length) for another 39/300, and
    correctly returned 180/300 as unmapped instead of a fabricated
    alignment. -u n/--splice-flank=no (disabling GT-AG motif matching --
    the earlier hypothesis for what caused the "everywhere" behavior) was
    also re-tested here against dorado's actual bundled minimap2 2.28 (the
    earlier test disproving it ran against the wrong, outdated system
    2.17 binary) and made results marginally *worse*, not better -- it is
    NOT part of this fix. --junc-bed on top of -G3000 didn't change the
    300-read sample's results either way.

    threads: minimap2's own -t (default 3, quite conservative -- unlike
    dorado aligner's own default of unlimited). Safe to raise when there's
    headroom (idle cores, free memory); this is the CPU-bound alignment
    step, so more threads roughly linearly speeds it up without multiplying
    the dual-pathway alignment's per-run memory footprint the way running
    multiple libraries concurrently would.
    """
    mm2 = _find_dorado_bundled_minimap2()
    tags = _discover_bam_tags(mutated_bam)
    tag_arg = f"-T {','.join(tags)}" if tags else ""

    opts = f'-a -y -x splice -G{max_intron_length} -t{threads} --secondary=no'
    if junc_bed:
        opts += f' --junc-bed={junc_bed}'

    cmd = (
        f"samtools fastq {tag_arg} {mutated_bam} | "
        f"{mm2} {opts} {ref_fasta} - | "
        f"samtools view -b -o {tmp_bam} -"
    )
    print(cmd)
    subprocess.call(cmd, shell=True)
    return tmp_bam

def _max_n_op_length(read):
    """Longest 'N' (intron/skip) CIGAR op on this read, or 0 if it has none."""
    if not read.cigartuples:
        return 0
    return max((length for op, length in read.cigartuples if op == 3), default=0)

def merge_dual_pathway_alignments(tmp_ag_bam, tmp_tc_bam, read_dict_ag, read_dict_tc, out_bam,
                                   flag_intron_threshold=2500):
    """
    Merge the A->G-pathway and T->C-pathway alignments into one final bam
    with correct strand flags and query sequences.

    Where a read has a primary hit in only one pathway (the normal/expected
    case given the asymmetry described in the module docstring), that hit is
    kept as-is, with is_reverse forced to match the pathway (False for AG,
    True for TC) and query_sequence restored from the matching read_dict.

    Where a read has a primary hit in both pathways (rare -- e.g. a repeat
    region), the higher-mapping-quality hit wins.

    Memory: only the smaller pathway's bam (by file size, a cheap proxy for
    read count that avoids a separate counting pass) is loaded into an
    in-memory index; the larger pathway is streamed against it one read at a
    time, popping matched entries out of the index as they're consumed. This
    holds at most one pathway's full hit set in memory at once instead of
    both simultaneously -- on real data, holding both was the dominant
    driver of peak RSS (tens of GB for a full sequencing run).

    flag_intron_threshold: reads are tagged 'XJ:i:<length>' (the longest N
    op on that read) whenever an N op exceeds this length. This is a soft
    flag, not a filter -- the read is still written -- because minimap2_
    align_splice_aware's own -G cap is itself only a soft chaining
    heuristic, not a hard ceiling: confirmed on real data that ~5% of N ops
    still exceed the nominal -G3000 cap (up to 9824bp) even with the cap
    applied, several clustering at a subtelomeric locus with no annotated
    multi-exon gene at all (i.e. still-spurious, repeat-driven junctions,
    just smaller-scale than before the cap). 2500bp matches the real
    annotated max intron length for this genome (2483bp), so any read
    tagged here has at least one junction longer than any known real yeast
    intron -- downstream scripts can filter on has_tag('XJ') (or the tag's
    value, for a custom threshold) per-analysis rather than losing the read
    outright. Set to None to disable tagging entirely.
    """
    if Path(tmp_ag_bam).stat().st_size <= Path(tmp_tc_bam).stat().st_size:
        index_bam, index_pathway, index_dict = tmp_ag_bam, 'ag', read_dict_ag
        stream_bam, stream_pathway, stream_dict = tmp_tc_bam, 'tc', read_dict_tc
    else:
        index_bam, index_pathway, index_dict = tmp_tc_bam, 'tc', read_dict_tc
        stream_bam, stream_pathway, stream_dict = tmp_ag_bam, 'ag', read_dict_ag

    index_hits = {}
    with pysam.AlignmentFile(index_bam, "rb") as bam:
        for read in bam:
            if not read.is_unmapped and not read.is_secondary and not read.is_supplementary:
                index_hits[read.query_name] = read

    ambiguous = 0
    with pysam.AlignmentFile(index_bam, "rb") as template_bam:
        with pysam.AlignmentFile(out_bam, "wb", template=template_bam) as out:

            def write_chosen(chosen, pathway, seq_dict):
                chosen.reference_id = template_bam.get_tid(chosen.reference_name)
                chosen.query_qualities = None
                chosen.is_reverse = (pathway != 'ag')
                chosen.query_sequence = seq_dict[chosen.query_name]
                if flag_intron_threshold is not None:
                    max_n = _max_n_op_length(chosen)
                    if max_n > flag_intron_threshold:
                        chosen.set_tag('XJ', max_n, value_type='i')
                out.write(chosen)

            with pysam.AlignmentFile(stream_bam, "rb") as sbam:
                for stream_read in sbam:
                    if (stream_read.is_unmapped or stream_read.is_secondary
                            or stream_read.is_supplementary):
                        continue
                    index_read = index_hits.pop(stream_read.query_name, None)
                    if index_read is None:
                        write_chosen(stream_read, stream_pathway, stream_dict)
                    else:
                        ambiguous += 1
                        if index_read.mapping_quality >= stream_read.mapping_quality:
                            write_chosen(index_read, index_pathway, index_dict)
                        else:
                            write_chosen(stream_read, stream_pathway, stream_dict)

            # whatever's left in index_hits mapped only via the index pathway
            for index_read in index_hits.values():
                write_chosen(index_read, index_pathway, index_dict)

    if ambiguous:
        print(f'  {ambiguous} reads mapped via both the A->G and T->C pathways; '
              f'kept the higher-mapq hit.', file=sys.stderr)

    sorted_bam = out_bam.replace('.bam', '_sorted.bam')
    subprocess.call('samtools sort ' + out_bam + ' > ' + sorted_bam, shell=True)
    subprocess.call('samtools index ' + sorted_bam, shell=True)
    return sorted_bam

def get_editing_efficiency(bam_file, ref_dict):
    bam_file = pysam.AlignmentFile(bam_file, "rb")
    efficiencies = []
    for read in bam_file:
        if not read.is_unmapped and not read.is_secondary and not read.is_supplementary:
            if read.reference_name not in ['XII']: # leave these out, they're rRNA
                read_seq = read.query_sequence.upper()
                # ref_seq = ref_seq.upper()
                ref_seq = ref_dict[read.reference_name].seq.upper()
                aligned_pairs = read.get_aligned_pairs()
                edits = 0
                numAs = 0
                for read_pos, ref_pos in aligned_pairs:
                    if ref_pos is not None and read_pos is not None:
                        if read.is_reverse:
                            if read_seq[read_pos] == 'C' and ref_seq[ref_pos] == 'T':
                                edits += 1
                                numAs += 1
                            elif ref_seq[ref_pos] == 'T':
                                numAs += 1
                        else:
                            if read_seq[read_pos] == 'G' and ref_seq[ref_pos] == 'A':
                                edits += 1
                                numAs += 1
                            elif ref_seq[ref_pos] == 'A':
                                numAs += 1

                efficiencies.append(edits / numAs if numAs > 0 else 0)

    return efficiencies

def plot_editing_efficiency(efficiencies, output_file):

    figureHeight = 5
    figureWidth = 5

    plt.figure(figsize=(figureWidth, figureHeight))
    panelHeight = 4 / figureHeight
    panelWidth = 4 / figureWidth

    panel = plt.axes([0.15, 0.1, panelWidth, panelHeight])

    counts, bin_edges = np.histogram(efficiencies)
    pdf = counts/sum(counts)
    cdf = np.cumsum(pdf)
    panel.plot(bin_edges[1:], cdf, color='blue')

    plt.title('Editing Efficiency per Read')
    plt.xlabel('Editing Efficiency')
    plt.ylabel('CDF')
    plt.savefig(output_file, dpi=300)


def align_reads(reads_bam, ref_fasta, out_bam, keep_intermediates=False, gtf=None,
                 splice_aware=False, max_intron_length=3000, flag_intron_threshold=2500,
                 threads=3):
    """
    Run the full dual-pathway (A->G / T->C) alignment strategy described in
    this module's docstring against reads_bam/ref_fasta. Returns the path to
    the final sorted, indexed bam. Importable so other scripts (e.g. a
    combined align+parquet pipeline) can call it directly.

    keep_intermediates: if False (default), delete this run's per-run
    intermediates (mutated bams, per-pathway tmp alignment bams, and the
    pre-sort merged bam) once the final sorted+indexed bam exists. The
    mutated reference fastas are never deleted here -- they're cached and
    reused across runs against the same genome (see mutateFastaAG/
    mutateFastaTC's staleness check), not owned by this one run.

    splice_aware: False (default) aligns with dorado_align_raw's plain
    map-ont preset -- correct for genomic DNA, but reads spanning a real
    intron get clipped or split (see dorado_align_raw's docstring). Set
    True for spliced mRNA/cDNA reads, which uses
    minimap2_align_splice_aware instead (dorado's own bundled minimap2,
    called directly with a validated -G cap -- see that function's
    docstring for why plain `dorado aligner -x splice` fabricates masses of
    false splice junctions on real data, and why the cap fixes it).

    gtf: optional annotation used to build a --junc-bed (see
    build_junction_bed) that biases minimap2 toward real annotated splice
    junctions. Only used when splice_aware=True. Genes whose true introns
    are missing from the GTF don't benefit from this (there's nothing to
    bias toward).

    max_intron_length: only used when splice_aware=True -- see
    minimap2_align_splice_aware's docstring for why 3000 is the validated
    default for this (yeast) genome; pass something else for organisms with
    larger real introns.

    flag_intron_threshold: forwarded to merge_dual_pathway_alignments -- see
    its docstring. Tags (doesn't drop) reads whose longest N op exceeds this
    length, since max_intron_length's -G cap is a soft chaining heuristic in
    minimap2 and real data still slips a residual few percent past it.

    threads: forwarded to dorado_align_raw/minimap2_align_splice_aware.
    Default (3) is conservative/shared-machine-safe; raise it when there's
    known headroom (idle cores, free memory) -- the AG and TC pathways run
    sequentially here, so this is the thread count for whichever one is
    currently running, not a combined total.
    """
    mutated_bam_ag, read_dict_ag = mutateBamSenseAG(reads_bam)
    mutated_bam_tc, read_dict_tc = mutateBamAntisenseTC(reads_bam)
    print(f'Mutated bam files: {mutated_bam_ag}, {mutated_bam_tc}')

    mutated_ref_ag = mutateFastaAG(ref_fasta)
    mutated_ref_tc = mutateFastaTC(ref_fasta)
    print(f'Mutated reference fasta files: {mutated_ref_ag}, {mutated_ref_tc}')

    tmp_ag_bam = out_bam.replace('.bam', '_AG_tmp.bam')
    tmp_tc_bam = out_bam.replace('.bam', '_TC_tmp.bam')

    if splice_aware:
        junc_bed = build_junction_bed(gtf) if gtf else None
        if junc_bed:
            print(f'Junction bed file: {junc_bed}')
        minimap2_align_splice_aware(mutated_bam_ag, mutated_ref_ag, tmp_ag_bam,
                                     junc_bed=junc_bed, max_intron_length=max_intron_length, threads=threads)
        minimap2_align_splice_aware(mutated_bam_tc, mutated_ref_tc, tmp_tc_bam,
                                     junc_bed=junc_bed, max_intron_length=max_intron_length, threads=threads)
    else:
        dorado_align_raw(mutated_bam_ag, mutated_ref_ag, tmp_ag_bam, threads=threads)
        dorado_align_raw(mutated_bam_tc, mutated_ref_tc, tmp_tc_bam, threads=threads)

    aligned_bam = merge_dual_pathway_alignments(
        tmp_ag_bam, tmp_tc_bam, read_dict_ag, read_dict_tc, out_bam,
        flag_intron_threshold=flag_intron_threshold)
    print(f'Aligned bam file: {aligned_bam}')

    if not keep_intermediates:
        intermediates = [mutated_bam_ag, mutated_bam_tc, tmp_ag_bam, tmp_tc_bam, out_bam]
        removed = 0
        for path in intermediates:
            if Path(path).exists():
                Path(path).unlink()
                removed += 1
        print(f'Removed {removed} intermediate file(s): {", ".join(intermediates)}')

    return aligned_bam


def main():
    parser = argparse.ArgumentParser(description='Map TadA edited reads to reference sequence')
    parser.add_argument('--reads_bam', required=True, help='Input unaligned bam file from Dorado basecalling')
    parser.add_argument('--ref_fasta', required=True, help='Reference fasta file')
    parser.add_argument('--out_bam', required=True, help='Output aligned bam file')
    parser.add_argument('--keep_intermediates', action='store_true',
                        help='Keep per-run intermediate bams (mutated bams, per-pathway '
                             'tmp alignments, pre-sort merged bam) instead of deleting them')
    parser.add_argument('--gtf', default=None,
                        help='Optional GTF/GFF3 annotation. With --splice_aware, its gene '
                             'models are converted to a --junc-bed to bias minimap2 toward '
                             'real splice junctions.')
    parser.add_argument('--splice_aware', action='store_true',
                        help='Align as spliced mRNA/cDNA (minimap2 splice preset via dorado\'s '
                             'own bundled minimap2 binary, with a validated max-intron-length '
                             'cap) instead of plain genomic map-ont alignment. Use this for '
                             'RNA-seq/cDNA reads that may span introns -- see '
                             'minimap2_align_splice_aware\'s docstring for why plain '
                             '`dorado aligner -x splice` isn\'t safe to use directly.')
    parser.add_argument('--max_intron_length', type=int, default=3000,
                        help='Only used with --splice_aware: cap (bp) on minimap2 -G (max '
                             'intron length). Default 3000 is validated for this yeast genome '
                             '(real annotated max is 2483bp); pass something larger for '
                             'organisms with bigger introns.')
    parser.add_argument('--flag_intron_threshold', type=int, default=2500,
                        help='Tag (not drop) reads whose longest N op exceeds this length '
                             '(bp) with XJ:i:<length>, since --max_intron_length\'s -G cap is '
                             'a soft chaining heuristic in minimap2 and a residual few percent '
                             'of junctions still slip past it on real data. Default 2500 '
                             'matches this genome\'s real annotated max intron (2483bp). Pass '
                             '0 (or a negative value) to disable tagging.')
    parser.add_argument('--threads', type=int, default=3,
                        help='Threads for the alignment step (default 3, conservative for a '
                             'shared machine). Raise this when there\'s known headroom -- see '
                             'align_reads docstring for why this is safe to increase without '
                             'multiplying memory usage, unlike running multiple libraries '
                             'concurrently.')
    args = parser.parse_args()

    flag_intron_threshold = args.flag_intron_threshold if args.flag_intron_threshold > 0 else None
    align_reads(args.reads_bam, args.ref_fasta, args.out_bam,
                keep_intermediates=args.keep_intermediates, gtf=args.gtf,
                splice_aware=args.splice_aware, max_intron_length=args.max_intron_length,
                flag_intron_threshold=flag_intron_threshold, threads=args.threads)
    # # calculate editing efficiency
    # ref_dict = SeqIO.to_dict(SeqIO.parse(args.ref_fasta, 'fasta'))
    # e = get_editing_efficiency(aligned_bam, ref_dict)
    # print(f'Average editing efficiency: {np.mean(e)*100:.2f}%')
    # plot_editing_efficiency(e, args.out_bam.replace('.bam', '_editingEfficiencyPerRead.png'))

if __name__ == '__main__':
    Tee()
    main()
