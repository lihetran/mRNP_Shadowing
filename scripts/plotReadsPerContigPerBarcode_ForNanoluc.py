'''
August 7, 2025 Liam Tran

This script will plot the number of reads per contig per barcode in a given BAM file. Barcodes are encoded by the 'cI' tag.
I'll also restrict to reads that include the shine dalgarno and start codon sequence. Contigs in the nanoluc reference file are otherwise identical besides the mutated start and shine dalgarno sequences.

input: barcoded bam file, usually an output from mapBased_barcodeSplitting_Hamming.py
output: a plot of the number of reads per contig per barcode
'''

import sys
import pysam
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def plot_reads_per_contig_per_barcode(bam_file):

    bam = pysam.AlignmentFile(bam_file, "rb")
    barcode_dict = {}

    for read in bam.fetch(start=70, end=733): # start codon and end of barcode
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        
        barcode = read.get_tag('cI')
        if barcode not in barcode_dict:
            barcode_dict[barcode] = {}
        
        contig = read.reference_name
        if contig not in barcode_dict[barcode]:
            barcode_dict[barcode][contig] = 1
        else:
            barcode_dict[barcode][contig] += 1

    bam.close()
    return barcode_dict

def plot_barcode_data(bamDict, output_file):
    '''
    Plot a grouped bar chart of the number of reads per contig per barcode.
    '''
    # for each bam, plot the number of reads that align to each nanoluc version
    plotDict = {}
    for b in bamDict:
        readDict = {}
        total = 0
        for chrom in bamDict[b]:
            readDict[chrom] = bamDict[b][chrom]
            total += bamDict[b][chrom]

        readDict['total'] = total
        plotDict[b] = readDict

    # normalise by total reads
    for b in plotDict:
        for chrom in plotDict[b]:
            plotDict[b][chrom] = plotDict[b][chrom] / plotDict[b]['total']


    # plot
    df = pd.DataFrame(plotDict).T

    # rename rows
    # df = df.rename(index={input: 'input', minusBn: 'minusBn', plusBn: 'plusBn'})
    

    df = df.reset_index()
    df = df.rename(columns={'index': 'Library'})
    # drop total row
    df = df.drop('total', axis=1)
    df = df.melt(id_vars='Library', var_name='nanoluc_version', value_name='Proportion')

    # plot
    # # df.plot(kind='bar')
    figureHeight = 5
    figureWidth = 5

    plt.figure(figsize=(figureWidth, figureHeight))

    panelHeight = 4 / figureHeight
    panelWidth = 4 / figureWidth

    panel = plt.axes([0.15, 0.15, panelWidth, panelHeight])
    panel = sns.barplot(data=df,x='Library', y='Proportion', palette='Grays', edgecolor='black', errorbar=None, hue='nanoluc_version')
    # shrink the legend
    panel.legend(loc='upper left')
    plt.setp(panel.get_legend().get_texts(), fontsize='8')
    panel.set_ylabel('Proportion of reads')
    panel.set_xlabel('Nanoluc version')
    panel.set_title('Proportion of Nanoluc Txts per Library')
    

    if output_file.endswith('.png'):
        plt.savefig(output_file, dpi=300)
    elif output_file.endswith('.svg'):
        plt.savefig(output_file)

def main(args):
    if len(args) != 2:
        print("Usage: python plotReadsPerContigPerBarcode_ForNanoluc.py <input.bam> <output.png/svg>")
        sys.exit(1)

    bam_file = args[0]
    output_file = args[1]

    barcode_dict = plot_reads_per_contig_per_barcode(bam_file)
    print(barcode_dict)
    plot_barcode_data(barcode_dict, output_file)

if __name__ == "__main__":
    main(sys.argv[1:])
