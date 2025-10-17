import matplotlib.pyplot as plt
import seaborn as sns
import pysam
import argparse
import numpy as np
from Bio import SeqIO

def plot_reads_per_contig(bam, output_file):
    '''
    Plot the number of reads per contig in a BAM file.
    '''
    contig_counts = {}
    with pysam.AlignmentFile(bam, "rb") as bamfile:
        for read in bamfile.fetch():
            if not read.is_unmapped:
                contig = read.reference_name
                if contig not in contig_counts:
                    contig_counts[contig] = 0
                contig_counts[contig] += 1

    contigs = list(contig_counts.keys())
    counts = list(contig_counts.values())

    plt.figure(figsize=(8, 6))
    sns.barplot(x=contigs, y=counts)
    plt.xlabel('Contigs')
    plt.ylabel('Number of Reads')
    plt.title('Number of Reads per Contig')

    plt.savefig(output_file)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot number of reads per contig from a BAM file.')
    parser.add_argument('-b', '--bam', type=str, help='Input BAM file')
    parser.add_argument('-o', '--output', type=str, help='Output plot file (e.g., output.png)')

    args = parser.parse_args()
    
    plot_reads_per_contig(args.bam, args.output)

if __name__ == "__main__":
    main()