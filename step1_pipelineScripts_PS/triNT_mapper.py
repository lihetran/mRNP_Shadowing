'''
Script for mapping Tad8A.20 reads in 3 nucleotide space. Mutates reads and reference from A-G and T-C and maps to mutated reference.
This script then finds A-G or T-C mismatches in the alignments and calculates the efficiency of the editing enzyme.

Usage: python3 triNT_mapper.py -r <reads.fastq> -g <reference.fasta> -o <outFilePrefix>
'''

from FastaReader import FastAreader
from FastqReader import FastQreader
import math

import pysam
import mappy as mp
import subprocess
import os
import shutil

from tqdm import tqdm
from Bio import SeqIO, Seq, SeqRecord
import argparse

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

'''
This method is adapted from Marcus Viscardi's cigar string parsing code.
'''
def FindInosines(read_seq, ref_seq, strand: bool, cigar, ref_start, ref_end, q_st, q_en):
    import re
    parsed_cigar = re.findall(rf'(\d+)([MDNSIX])', cigar)
    parsed_cigar = [(int(num), char) for num, char in parsed_cigar]
    ref_seq = ref_seq[ref_start: ref_end].upper()
    ref_pos = 0
    read_seq = read_seq[q_st: q_en].upper()
    read_pos = 0

    top_line = ""
    middle_line = ""
    bottom_line = ""

    for length, code in parsed_cigar:
        if code == "M":  # Map (Read & Ref Match)
            read_map_piece = read_seq[read_pos:read_pos + length]
            ref_map_piece = ref_seq[ref_pos:ref_pos + length]
            perfect_matches = ""
            for index, char in enumerate(read_map_piece):
                try:
                    if char == ref_map_piece[index]:
                        perfect_matches += "|"
                    else:
                        perfect_matches += "•"
                except IndexError:
                    perfect_matches += " "
            top_line += read_map_piece
            middle_line += perfect_matches
            bottom_line += ref_map_piece
            ref_pos += length
            read_pos += length

        elif code == "I":  # Insert (Gap in Ref)
            top_line += read_seq[read_pos:read_pos + length]
            middle_line += " " * length
            bottom_line += " " * length
            read_pos += length

        elif code == "D" or code == "N":  # Delete (Gap in Read)
            top_line += " " * length
            middle_line += " " * length
            bottom_line += ref_seq[ref_pos:ref_pos + length]
            ref_pos += length
    # print(top_line[0:15])
    # print(middle_line[0:15])
    # print(bottom_line[0:15])
    inosines = []
    numSites = 0
    if strand == True: #reads are reverse complemented and mutated from A-G
        for i in range(len(top_line)):
            if top_line[i] == 'G' and bottom_line[i] == 'A':
                numSites += 1
                inosines.append(i)
            elif bottom_line[i] == 'A':
                numSites += 1
    elif strand == False: #reads are not reverse complemented and mutated from C-T
        for i in range(len(top_line)):
            if top_line[i] == 'C' and bottom_line[i] == 'T':
                numSites += 1
                inosines.append(i)
            elif bottom_line[i] == 'T':
                numSites += 1

    #### Create custom tag for bam file ####
    ct = 0
    tag = 'A,'
    for i in range(len(top_line)):
        
        if top_line[i] == 'G' and i not in inosines:
            ct += 1
        elif top_line[i] == 'G' and i in inosines:
            tag += str(ct) + ','
            ct = 0
    tag = tag[:-1]        
    #tag += ';'

    if bottom_line.find('A') == -1 and strand == True:
        return None
    elif bottom_line.find('T') == -1 and strand == False:
        return None
    else:
        return len(inosines)/numSites*100 

def main():
    parser = argparse.ArgumentParser(description='Find A-I edits in nanopore reads',
                                     add_help=True,
                                     prefix_chars='-')
    parser.add_argument('-fq', '--reads_fastq', help="reads in fastq format")
    parser.add_argument('-fa', '--reads_fasta', help="reads in fasta format")

    parser.add_argument('-g', '--fasta', help="reference in fasta format")
    # parser.add_argument('-j', '--junction_bed', help="junction bed file")
    parser.add_argument('-o', '--outFilePrefix', help="output file name without extension")

    args = parser.parse_args()

    # if args.reads:
    #     fastq = FastQreader(args.reads)
    # elif args.fasta:
    #     fasta = FastAreader(args.fasta)
    # fastq = FastQreader(args.reads)

    # Leaf directory
    directory = "tmp"
    
    # Parent Directories
    parent_dir = os.getcwd()
    
    # Path
    path = os.path.join(parent_dir, directory)

    os.makedirs(path)
    mut_AG = path + '/mut_AG.fastq'

    #### mutate and reverse complement cDNA reads from A-G ########
    if args.reads_fastq:
        originalReadDict_AG = {}
        reads_fastq = FastQreader(args.reads_fastq)
        with open(mut_AG, 'w') as f_out:
            for record in reads_fastq.readFastq():
                header = record[0]
                seq = record[1]
                strand = record[2]
                qual = record[3]

                rc = reverseComplement(seq)
                newSeq, positions = replaceCharacter(rc, 'A', 'G')
                originalReadDict_AG[header.split()[0][1:]] = rc
                f_out.write(header + '\n' + newSeq + '\n' + strand + '\n' + qual + '\n')

    elif args.reads_fasta:
        originalReadDict_AG = {}
        # reads_fasta = FastAreader(args.reads_fasta)
        with open(mut_AG, 'w') as f_out:
            for record in SeqIO.parse(args.reads_fasta, 'fasta'):
                header = record.id
                seq = str(record.seq)
                rc = reverseComplement(seq)
                newSeq, positions = replaceCharacter(rc, 'A', 'G')
                originalReadDict_AG[header] = rc
                f_out.write('>' + header + '\n' + newSeq + '\n')
                
    #### mutate reference from A-G ########
    mutated_genome_AG = 'tmp' + '/mutated_genome_AG.fa'
    refDictAG = {}
    with open(mutated_genome_AG, 'w') as f_out:
        for record in SeqIO.parse(args.fasta, 'fasta'):
            header = record.id
            seq = str(record.seq)
            newSeq, positions = replaceCharacter(seq, 'A', 'G')
            refDictAG[header] = seq
            f_out.write('>' + header + '\n' + newSeq + '\n') 

    #### Align mutated reads to mutated reference (A-G) ########
    
    # cmd1 = 'minimap2 -ax map-ont -t 20 --junc-bed ' + args.junction_bed + ' --secondary=no --for-only --MD --sam-hit-only ' + mutated_genome_AG + ' ' + mut_AG + ' > tmp/alignments.AtoG.sam'
    cmd1 = 'minimap2 -ax map-ont -t 20' + ' --secondary=no --for-only --cs="long" --sam-hit-only ' + mutated_genome_AG + ' ' + mut_AG + ' > tmp/alignments.AtoG.sam' 
    # cmd1 = 'minimap2 -ax map-ont -t 20' + ' --secondary=no --cs="long" --sam-hit-only ' + mutated_genome_AG + ' ' + mut_AG + ' > tmp/alignments.AtoG.sam' 
    print('Mapping A-G reads to A-G reference')
    subprocess.call(cmd1, shell=True)

    #### remove supplementary alignments ########
    subprocess.call('samtools view -F 2048 -bo tmp/alignments.AtoG.bam tmp/alignments.AtoG.sam', shell=True)

    # subprocess.call('samtools view -hbS tmp/alignments.AtoG.sam > tmp/alignments.AtoG.bam', shell=True)

    #### Find inosines in A-G alignments ########
    #inosine_positions_AG = []
    AtoGbam = args.outFilePrefix+'.AtoG.bam'
    numReads = 0
    e = []
    with pysam.AlignmentFile('tmp/alignments.AtoG.bam','rb') as bamFile:
    # with pysam.AlignmentFile('tmp/alignments.AtoG.sam','rb') as samFile:
        #change to wb if you want to write to bam file
        with pysam.AlignmentFile(AtoGbam, "wb", template=bamFile) as outFile:
            for read in tqdm(bamFile,desc='Finding inosines'):
                # if read.is_unmapped == False and read.reference_name != 'cerENO2':
                if read.is_unmapped == False:
                    numReads += 1
                    read_id = read.query_name
                    seq = read.query_sequence
                    original_seq = originalReadDict_AG[read_id]
                    original_ref = refDictAG[read.reference_name]
                    q_st = read.query_alignment_start
                    q_en = read.query_alignment_end
                    ref_start = read.reference_start
                    ref_end = read.reference_end
                    cigar = read.cigarstring
                    # cs = read.get_tag('cs')
                    inosine_positions_AG = FindInosines(original_seq, original_ref, True, cigar, ref_start, ref_end, q_st, q_en)
                    # read.set_tag('cs', cs)
                    read.query_sequence = original_seq
                    e.append(inosine_positions_AG)
                    
                    outFile.write(read)

    # close files
    bamFile.close()
    outFile.close()
    
    #### Sort and index bam file ########
    subprocess.call('samtools sort ' + AtoGbam + ' -o ' + AtoGbam[:-3] + 'sorted' + '.bam', shell=True)
    subprocess.call('samtools index ' + AtoGbam[:-3] + 'sorted.bam', shell=True)

    eff = [x for x in e if x is not None]
    print('Tad8A.20 Efficiency: ' + str(sum(eff)/len(eff)))

    shutil.rmtree(path)
    
    
if __name__ == '__main__':
    main()






