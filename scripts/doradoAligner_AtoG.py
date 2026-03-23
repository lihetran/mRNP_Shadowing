'''
October 5, 2025 LT

Dorado is now the recommended basecaller for Oxford Nanopore data. Guppy is no longer supported or maintained by ONT. 
Dorado's output after basecalling is now an unaligned bam file which retains base modification information. This script has been updated to process Dorado output. 
My previous scripts for aligning highly modified reads by TadA or APOBEC worked with Guppy fastq files. This script has been updated to work with Dorado unaligned bam files.

input: unaligned bam file from Dorado basecalling
output: aligned bam file to reference sequence

This script mutates all A's to G's in the reads and reference sequence to allow for better alignment of TadA edited reads.
'''

import pysam
import mappy as mp
import subprocess
import os
import shutil

from tqdm import tqdm
from Bio import SeqIO, Seq, SeqRecord
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def mutateBam(input_bam):
    # create a mutated bam file where all C's are changed to T's
    # this will allow for better alignment of APOBEC edited reads to the reference sequence
    # output is a bam file with the same header as the input bam file

    mutated_bam = input_bam.replace('.bam', '_mutated.bam')
    read_dict = {}
    with pysam.AlignmentFile(input_bam, "rb", check_sq=False) as in_bam:
        with pysam.AlignmentFile(mutated_bam, "wb", template=in_bam) as out_bam:
            for read in in_bam:
                seq = read.query_sequence
                if seq is not None:
                    rc = reverseComplement(seq) # reverse complement the sequence bc of cDNA sequencing
                    new_seq, positions = replaceCharacter(rc, 'A', 'G')
                    read.query_sequence = new_seq
                    # # need to update the qualities too
                    # if read.query_qualities is not None:
                    #     quals = read.query_qualities
                    #     new_quals = [quals[i] for i in range(len(quals)) if i not in positions]
                    #     read.query_qualities = new_quals
                    out_bam.write(read)
                    read_dict[read.query_name] = rc # store the original read sequence for later
        
    return mutated_bam, read_dict

def mutateFasta(input_fasta):
    # create a mutated fasta file where all C's are changed to T's
    # this will allow for better alignment of APOBEC edited reads to the reference sequence
    # output is a fasta file
    input_fasta = Path(input_fasta)
    output_fasta = input_fasta.with_name(input_fasta.stem + '_mutated_AG.fasta')
    with open(output_fasta, 'w') as out_fasta:
        for record in SeqIO.parse(input_fasta, 'fasta'):
            seq = str(record.seq)
            new_seq, positions = replaceCharacter(seq, 'A', 'G')
            new_record = SeqRecord.SeqRecord(Seq.Seq(new_seq), id=record.id, description=record.description)
            SeqIO.write(new_record, out_fasta, 'fasta')
    
    return output_fasta

def dorado_aligner(mutated_bam, ref_fasta, read_dict, out_bam):
    
    tmp_bam = out_bam.replace('.bam', '_tmp.bam')
    # output is a bam file
    cmd = 'dorado aligner ' + str(ref_fasta) + ' ' + str(mutated_bam) + ' > ' + tmp_bam + ' --mm2-opts "-x map-ont --secondary=no"'
    print(cmd)
    # run the command
    subprocess.call(cmd, shell=True)
    # subprocess.run(['dorado aligner', ref_fasta, mutated_bam, '-o', out_bam], shell=True)

    with pysam.AlignmentFile(tmp_bam, "rb") as bam:
         with pysam.AlignmentFile(out_bam, "wb", template=bam) as out_bam_file:
            for read in bam:
                if not read.is_unmapped and not read.is_secondary and not read.is_supplementary:
                    read_id = read.query_name
                    if read_id in read_dict:
                        original_seq = read_dict[read_id]
                        read.query_sequence = original_seq
                        out_bam_file.write(read)


    bam.close()
    # sort and index the bam file
    sorted_bam = out_bam.replace('.bam', '_sorted.bam')
    subprocess.call('samtools sort ' + out_bam + ' > ' + sorted_bam, shell=True)
    subprocess.call('samtools index ' + sorted_bam, shell=True)

    return sorted_bam

def get_editing_efficiency(bam_file, ref_dict):
    bam_file = pysam.AlignmentFile(bam_file, "rb")
    efficiencies = []
    for read in bam_file:
        if not read.is_unmapped and not read.is_secondary and not read.is_supplementary:
            # if read.reference_name not in ['I', 'V', 'cerENO2']: # leave these out, they're rRNA and spike-in controls
            read_seq = read.query_sequence.upper()
            # ref_seq = ref_seq.upper()
            ref_seq = ref_dict[read.reference_name].seq.upper()
            aligned_pairs = read.get_aligned_pairs()
            edits = 0
            numAs = 0
            for read_pos, ref_pos in aligned_pairs:
                if ref_pos is not None and read_pos is not None:
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
    
    # path = 'tmp'
    # if not os.path.exists(path):
    #     os.mkdir(path)
    # mutate the bam file and reference fasta file

    mutated_bam, read_dict = mutateBam(args.reads_bam)
    mutated_ref = mutateFasta(args.ref_fasta)
    print(f'Mutated bam file: {mutated_bam}')
    print(f'Mutated reference fasta file: {mutated_ref}')
    aligned_bam = dorado_aligner(mutated_bam, mutated_ref, read_dict, args.out_bam)
    print(f'Aligned bam file: {aligned_bam}')
    # # calculate editing efficiency
    ref_dict = SeqIO.to_dict(SeqIO.parse(args.ref_fasta, 'fasta'))
    e = get_editing_efficiency(aligned_bam, ref_dict)
    print(f'Average editing efficiency: {np.mean(e)*100:.2f}%')
    plot_editing_efficiency(e, args.out_bam.replace('.bam', '_editingEfficiencyPerRead.png'))

if __name__ == '__main__':
    main()
