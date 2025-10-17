import matplotlib.pyplot as plt
import seaborn as sns
import pysam
import argparse
import numpy as np
from Bio import SeqIO
import tqdm

def get_editing_efficiency(bam_file, ref_dict):
    bam_file = pysam.AlignmentFile(bam_file, "rb")
    efficiencies = []
    for read in tqdm.tqdm(bam_file):
        if not read.is_unmapped:
            if read.reference_name not in ['I', 'V', 'cerENO2']: # leave these out, they're rRNA and spike-in controls
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
    parser = argparse.ArgumentParser(description='Plot editing efficiency per Read')
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