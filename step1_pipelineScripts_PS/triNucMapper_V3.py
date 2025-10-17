'''
December 3, 2024 LT
Going to test a mapping approach set forth by the following paper: https://www.bioif args.subsample is not None and i >= args.subsample:
                break
            if read.is_unmapped:
                continue
            pairs = read.get_aligned_pairs(with_seq=False)
            extracted_seq = extract_seq_with_ref_positions_and_aligned_pairs(pairs,
                                                                             args.ref_start,
                                                                             args.ref_end,
                                                                             read.query_sequence)
            barcode, distance = pick_best_barcode(extracted_seq, barcodes, args.dist)

            # read.set_tag('bI', barcode)
            # read.set_tag('bS', extracted_seq)
            # read.set_tag('bD', distance)

            # LT changed tag on 8/26/24 so that we can also demultiplex PCR pol conditions
            read.set_tag('CB', barcode, 'Z')
            # read.set_tag('cS', extracted_seq)
            # read.set_tag('cD', distance)
 
            barcode_reads_dict[barcode].append(read)
            barcode_counter[barcode] += 1
            top_barcodes = sorted(barcode_counter.items(),
                                  key=lambda x: x[1],
                                  reverse=True)[:args.top_bar_count]
            if not args.quiet and i % 100 == 0:
                bam_iterator.desc = (f"Extracting barcodes | {i} reads | "
                                     + " | ".join([f"{bar}: {count}" for bar, count in top_barcodes]))
                rxiv.org/content/10.1101/2024.11.06.622310v1

Instead of mapping in 3nt space, they do this:
1. Map with minimap2 "--MD -Y -y -ax map-ont"
    i. Count A-G mismatches and T-C mismatches. If >90% of the mismatches are A-G => top strand
    ii. If >90% of the mismatches are T-C => bottom strand
    iii. Discard secondary and supplementary alignments
    iv. replace G's in the top strand with R's and C's with Y's in the bottom strand
2. Remap with minimap2 "--MD -Y -y -ax map-ont" using the modified sequences
'''

import pysam
import mappy as mp
import subprocess
import os
import shutil

from tqdm import tqdm
from Bio import SeqIO, Seq, SeqRecord
import argparse
from icecream import ic

class Mapper:
    def __init__(self, read_file, reference_file, output_prefix):
        self.read_file = read_file
        self.reference_file = reference_file
        self.output_prefix = output_prefix

        self.ref_dict = SeqIO.to_dict(SeqIO.parse(self.reference_file, 'fasta'))
        # ic(self.ref_dict)

    def map_reads(self):
        # Map reads to reference
        minimap2_command = f"minimap2 --MD -Y -y -ax map-ont {self.reference_file} {self.read_file} > tmp.sam"
        subprocess.run(minimap2_command, shell=True)

        # Parse SAM file
        efficiency = []
        samfile = pysam.AlignmentFile(f"tmp.sam", "r")
        newSam = "mutated.sam"
        with pysam.AlignmentFile(newSam, "w", template=samfile) as f_out:
            for read in samfile:
                if not read.is_secondary and not read.is_supplementary:
                    # ic(ref_seq)
                    if read.reference_name in self.ref_dict:
                        ref_seq = self.ref_dict[read.reference_name].seq.upper()
                        strand = self.count_mismatches(read.query_sequence.upper(), ref_seq, read.get_aligned_pairs(matches_only=True))
                        if strand == "top":
                            # newSeq, positions = self.replace_chars(read.query_sequence, "G", "R") # replace all G's with R's or replace only the mismatches?
                            # read.query_sequence = newSeq
                            read.is_reverse = False
                        
                        elif strand == "bottom":
                            # newSeq, positions = self.replace_chars(read.query_sequence, "C", "Y") # replace all C's with Y's or replace only the mismatches?
                            # read.query_sequence = newSeq
                            read.is_reverse = True
                        else:
                            continue
                        # Write modified reads to file
                        e = self.get_efficiency(read.query_sequence, ref_seq, read.get_aligned_pairs(), strand)
                        efficiency.append(e)
                        f_out.write(read)
        newSam.close()
        samfile.close()

        # Remap modified reads
        minimap2_command = f"minimap2 --MD -Y -y -ax map-ont {self.reference_file} mutated.sam > {self.output_prefix}.sam"
        subprocess.run(minimap2_command, shell=True)
        # Clean up
        os.remove("tmp.sam")
        eff = [x for x in efficiency if x is not None]
        return eff

    def count_mismatches(self, read_seq, ref_seq, aligned_pairs):
        # Count A-G and T-C mismatches
        a_g = 0
        t_c = 0
        total = 0
        read_seq = read_seq.upper()

        ref_seq = ref_seq.upper()
        new_read_seq = "" 
        ag_positions = []
        tc_positions = []
        for read_pos, ref_pos in aligned_pairs:
            if read_pos is not None and ref_pos is not None:
                # ic(read_seq[read_pos], ref_seq[ref_pos])
                if read_seq[read_pos] == "G" and ref_seq[ref_pos] == "A":
                    a_g += 1
                    total += 1
                    ag_positions.append(read_pos)
                    # new_read_seq += "R"
                elif read_seq[read_pos] == "C" and ref_seq[ref_pos] == "T":
                    t_c += 1
                    total += 1
                    # new_read_seq += "Y"
                elif read_seq[read_pos] != ref_seq[ref_pos]:
                    total += 1
                    new_read_seq += read_seq[read_pos]
            elif read_pos is not None:
                new_read_seq += read_seq[read_pos]

                    
        # ic(a_g, t_c, total)
        # ic(read_seq, new_read_seq)
        if a_g / total > 0.5 and total > 0:
            return "top"
        elif t_c / total > 0.5 and total > 0:
            return "bottom"
        else:
            return "unknown"
        
    def replace_chars(self, seq, char1, char2):
        n = len(seq)
        res = ""
        positions = []
        seq = seq.upper()
        for i in range(n):
            if seq[i] != char1:
                res += seq[i]
            else:
                res += char2
                positions.append(i)

        return res, positions
    
    def get_efficiency(self, read_seq, ref_seq, aligned_pairs, strand):
        # Calculate efficiency of conversion
        mods = 0
        total = 0
        read_seq = read_seq.upper()
        ref_seq = ref_seq.upper()

        if strand == "top":
            for read_pos, ref_pos in aligned_pairs:
                if read_pos is not None and ref_pos is not None:
                    if read_seq[read_pos] == "G" and ref_seq[ref_pos] == "A":
                        mods += 1
                        total += 1
                    elif ref_seq[ref_pos] == "A":
                        total += 1
        elif strand == "bottom":
            for read_pos, ref_pos in aligned_pairs:
                if read_pos is not None and ref_pos is not None:
                    if read_seq[read_pos] == "C" and ref_seq[ref_pos] == "T":
                        mods += 1
                        total += 1
                    elif ref_seq[ref_pos] == "T":
                        total += 1

        return mods / total
    

def main():
    parser = argparse.ArgumentParser(description="Map reads to reference and modify sequences based on mismatches")
    parser.add_argument("-r", "--read_file", help="Path to read file")
    parser.add_argument("-g", "--reference_file", help="Path to reference file")
    parser.add_argument("-o", "--output_prefix", help="Output file prefix")
    args = parser.parse_args()

    mapper = Mapper(args.read_file, args.reference_file, args.output_prefix)
    mapper.map_reads()

if __name__ == "__main__":
    main()

            