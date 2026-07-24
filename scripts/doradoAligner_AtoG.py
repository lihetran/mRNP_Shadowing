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
'''

import sys
import pysam
import subprocess
import argparse
import numpy as np
import matplotlib.pyplot as plt

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

def mutateFastaAG(input_fasta):
    # create a mutated fasta file where all A's are changed to G's, same coordinates as input_fasta
    input_fasta = Path(input_fasta)
    output_fasta = input_fasta.with_name(input_fasta.stem + '_mutated_AG.fasta')
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
    with open(output_fasta, 'w') as out_fasta:
        for record in SeqIO.parse(input_fasta, 'fasta'):
            seq = str(record.seq)
            new_seq, positions = replaceCharacter(seq, 'T', 'C')
            new_record = SeqRecord.SeqRecord(Seq.Seq(new_seq), id=record.id, description=record.description)
            SeqIO.write(new_record, out_fasta, 'fasta')

    return output_fasta

def dorado_align_raw(mutated_bam, ref_fasta, tmp_bam):
    """Run dorado aligner and return the path to the raw (unfiltered) output bam."""
    cmd = f'dorado aligner {ref_fasta} {mutated_bam} --mm2-opts "-x map-ont --secondary=no" > {tmp_bam}'
    print(cmd)
    subprocess.call(cmd, shell=True)
    return tmp_bam

def merge_dual_pathway_alignments(tmp_ag_bam, tmp_tc_bam, read_dict_ag, read_dict_tc, out_bam):
    """
    Merge the A->G-pathway and T->C-pathway alignments into one final bam
    with correct strand flags and query sequences.

    Where a read has a primary hit in only one pathway (the normal/expected
    case given the asymmetry described in the module docstring), that hit is
    kept as-is, with is_reverse forced to match the pathway (False for AG,
    True for TC) and query_sequence restored from the matching read_dict.

    Where a read has a primary hit in both pathways (rare -- e.g. a repeat
    region), the higher-mapping-quality hit wins.
    """
    def load_primary(bam_path):
        hits = {}
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for read in bam:
                if not read.is_unmapped and not read.is_secondary and not read.is_supplementary:
                    hits[read.query_name] = read
        return hits

    ag_hits = load_primary(tmp_ag_bam)
    tc_hits = load_primary(tmp_tc_bam)

    ambiguous = 0
    with pysam.AlignmentFile(tmp_ag_bam, "rb") as template_bam:
        with pysam.AlignmentFile(out_bam, "wb", template=template_bam) as out:
            for read_id in set(ag_hits) | set(tc_hits):
                ag_read = ag_hits.get(read_id)
                tc_read = tc_hits.get(read_id)

                if ag_read is not None and tc_read is not None:
                    ambiguous += 1
                    pathway = 'ag' if ag_read.mapping_quality >= tc_read.mapping_quality else 'tc'
                    chosen = ag_read if pathway == 'ag' else tc_read
                elif ag_read is not None:
                    chosen, pathway = ag_read, 'ag'
                else:
                    chosen, pathway = tc_read, 'tc'

                chosen.reference_id = template_bam.get_tid(chosen.reference_name)
                chosen.query_qualities = None
                if pathway == 'ag':
                    chosen.is_reverse = False
                    chosen.query_sequence = read_dict_ag[read_id]
                else:
                    chosen.is_reverse = True
                    chosen.query_sequence = read_dict_tc[read_id]

                out.write(chosen)

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


def main():
    parser = argparse.ArgumentParser(description='Map APOBEC edited reads to reference sequence')
    parser.add_argument('--reads_bam', required=True, help='Input unaligned bam file from Dorado basecalling')
    parser.add_argument('--ref_fasta', required=True, help='Reference fasta file')
    parser.add_argument('--out_bam', required=True, help='Output aligned bam file')
    args = parser.parse_args()

    mutated_bam_ag, read_dict_ag = mutateBamSenseAG(args.reads_bam)
    mutated_bam_tc, read_dict_tc = mutateBamAntisenseTC(args.reads_bam)
    print(f'Mutated bam files: {mutated_bam_ag}, {mutated_bam_tc}')

    mutated_ref_ag = mutateFastaAG(args.ref_fasta)
    mutated_ref_tc = mutateFastaTC(args.ref_fasta)
    print(f'Mutated reference fasta files: {mutated_ref_ag}, {mutated_ref_tc}')

    tmp_ag_bam = args.out_bam.replace('.bam', '_AG_tmp.bam')
    tmp_tc_bam = args.out_bam.replace('.bam', '_TC_tmp.bam')
    dorado_align_raw(mutated_bam_ag, mutated_ref_ag, tmp_ag_bam)
    dorado_align_raw(mutated_bam_tc, mutated_ref_tc, tmp_tc_bam)

    aligned_bam = merge_dual_pathway_alignments(
        tmp_ag_bam, tmp_tc_bam, read_dict_ag, read_dict_tc, args.out_bam)
    print(f'Aligned bam file: {aligned_bam}')
    # # calculate editing efficiency
    # ref_dict = SeqIO.to_dict(SeqIO.parse(args.ref_fasta, 'fasta'))
    # e = get_editing_efficiency(aligned_bam, ref_dict)
    # print(f'Average editing efficiency: {np.mean(e)*100:.2f}%')
    # plot_editing_efficiency(e, args.out_bam.replace('.bam', '_editingEfficiencyPerRead.png'))

if __name__ == '__main__':
    main()
