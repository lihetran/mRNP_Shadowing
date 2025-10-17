'''
August 12, 2025 Liam Tran

ONT seems to be moving away from guppy for basecalling and now dorado is the best supported basecalling software. By default, Dorado will output basecalled reads in a BAM to preserve 
base modification data. These are usually stored in the read's tags. 

The way Dorado outputs fastq files is mildly annoying, I want write a script that take in an unaligned BAM, converts A's to G's in the read sequences, and outputs a new fastq file.

input: unaligned.bam
output: modified.fastq
'''

import sys
import pysam
import os

def modify_bam(input_bam, output_fastq):
    with pysam.AlignmentFile(input_bam, "rb") as bam_file, open(output_fastq, "w") as fastq_file:
        for read in bam_file:
            if not read.is_unmapped:
                # Convert A's to G's in the read sequence
                seq = read.query_sequence()
                modified_seq = seq.replace("A", "G")
                # Write the modified read to the fastq file
                fastq_file.write(f"@{read.qname}\n{modified_seq}\n+\n{read.qual}\n")

def dorado_aligner():
    pass

def main(args):
    if len(args) != 2:
        print("Usage: script.py <input.bam> <output.fastq>")
        return
    input_bam = args[0]
    output_fastq = args[1]
    modify_bam(input_bam, output_fastq)

if __name__ == "__main__":
    main(sys.argv[1:])