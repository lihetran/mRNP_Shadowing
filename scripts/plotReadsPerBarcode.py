'''
August 7, 2025 Liam Tran

This script will plot the number of reads per barcode from a BAM file.

Usage:
    python plotReadsPerContigPerBarcode.py <input.bam> <output.png>

input: BAM file containing reads with barcodes, usually an output from mapBased_barcodeSplitting_Hamming.py
output: a plot of the number of reads per contig per barcode
'''

import sys
import pysam
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_reads_per_barcode(bam_file):
    bam = pysam.AlignmentFile(bam_file, "rb")
    barcode_dict = {}

    for read in bam.fetch():
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        
        barcode = read.get_tag('cI')
        if barcode not in barcode_dict:
            barcode_dict[barcode] = 1 
        else:
            barcode_dict[barcode] += 1

    return barcode_dict

def plot_barcode_data(barcode_dict, output_file):

    # plot
    df = pd.DataFrame(barcode_dict.items(), columns=['Barcode', 'Count'])
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Barcode', y='Count')
    plt.title('Reads per Barcode')
    
    if output_file.endswith('.png'):
        plt.savefig(output_file, dpi=300)
    elif output_file.endswith('.svg'):
        plt.savefig(output_file, format='svg')
    plt.savefig(output_file)
    plt.close()

def main(args):
    if len(args) != 2:
        print("Usage: python plotReadsPerContigPerBarcode.py <input.bam> <output.png>")
        sys.exit(1)

    bam_file = args[0]
    output_file = args[1]

    barcode_dict = plot_reads_per_barcode(bam_file)
    plot_barcode_data(barcode_dict, output_file)
if __name__ == "__main__":
    main(sys.argv[1:])