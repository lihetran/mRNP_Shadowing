import matplotlib.pyplot as plt
import seaborn as sns
import pysam
import argparse
import numpy as np
from Bio import SeqIO

def get_editing_efficiency(bam_file, ref_dict):
    bam_file = pysam.AlignmentFile(bam_file, "rb")
    barcode_dict = {}
    
    for read in bam_file:
        if not read.is_unmapped:
            # barcode = read.get_tag('bI')
            barcode = read.get_tag('cI')
            read_seq = read.query_sequence.upper()
            # ref_seq = ref_seq.upper()
            ref_seq = ref_dict[read.reference_name].seq.upper()
            aligned_pairs = read.get_aligned_pairs()
            edits = 0
            for read_pos, ref_pos in aligned_pairs:
                if ref_pos is not None and read_pos is not None:
                    if read_seq[read_pos] == 'G' and ref_seq[ref_pos] == 'A':
                        edits += 1
            
            if barcode not in barcode_dict:
                barcode_dict[barcode] = {read.query_name: edits / ref_seq[aligned_pairs[1][0]: aligned_pairs[-1][0]].count('A')}
            else:
                barcode_dict[barcode][read.query_name] = edits / ref_seq[aligned_pairs[1][0]: aligned_pairs[-1][0]].count('A')

    
    return barcode_dict

def plot_editing_efficiency(barcode_dict, output_file):
    
    colors = sns.color_palette("dark", len(barcode_dict))
    figureHeight = 5
    figureWidth = 5

    plt.figure(figsize=(figureWidth, figureHeight))
    panelHeight = 4 / figureHeight
    panelWidth = 4 / figureWidth

    panel = plt.axes([0.15, 0.1, panelWidth, panelHeight])

    for b in barcode_dict:
        m = []
        for read in barcode_dict[b]:
            m.append(barcode_dict[b][read])

        counts, bin_edges = np.histogram(m)
        pdf = counts/sum(counts)
        cdf = np.cumsum(pdf)
        panel.plot(bin_edges[1:], cdf, label=b, color=colors[list(barcode_dict.keys()).index(b)])

    plt.title('Editing Efficiency per Barcode')
    plt.xlabel('Editing Efficiency')
    plt.ylabel('CDF')
    plt.legend()
    plt.savefig(output_file, dpi=300)

def main():
    parser = argparse.ArgumentParser(description='Plot editing efficiency per barcode')
    parser.add_argument('-b', '--bam', required=True, help='BAM file')
    parser.add_argument('-r', '--ref', required=True, help='Reference fasta')
    parser.add_argument('-o', '--output', required=True, help='Output file')
    args = parser.parse_args()

    
    # ref_seq = SeqIO.read(args.ref, 'fasta').seq
    ref_dict = SeqIO.to_dict(SeqIO.parse(args.ref, 'fasta'))
    barcode_dict = get_editing_efficiency(args.bam, ref_dict)
    plot_editing_efficiency(barcode_dict, args.output)

if __name__ == '__main__':
    main()
    

    


    
    