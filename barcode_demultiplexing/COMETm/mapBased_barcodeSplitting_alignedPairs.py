"""
mapBased_barcodeSplitting.py
Marcus Viscardi,    March 15, 2024
Liam Tran, modifying Marcus Viscardi's code, June 24, 2024
    Liam added functionality to have aligned pairs being the method of sequence extraction rather than cs_tag parsing.
    The use of aligned pairs will make the barcode splitting more robust to different alignment strategies!

This script will take in a BAM file and spit out several BAM files split on the basis of their contained barcodes.

Additionally, the user can specify where in the reference to pull the barcode from, which barcodes to use,
and how close barcode matches have to be.

TODO: Add a helper tool to check that the contig positions are correct for the barcode splitting.
- This would just require a simple function that accepts the reference fasta file and the contig positions,
   then it would extract the sequence and print it out. Good way to affirm that the positions are correct.
"""
from typing import Dict, Tuple, List

import subprocess

import argparse
import stringdist
from Bio import SeqIO
from pathlib import Path

import pysam

# I just added these functions to this file down below to make it self-contained:
# from COMETm.mapBased.cs_parsing import extract_ref_orient_query_region

from icecream import ic
from datetime import datetime
import re

from tqdm.auto import tqdm

from contextlib import ExitStack

from pprint import pprint


def __time_formatter__():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"ic: {now} | > "


ic.configureOutput(prefix=__time_formatter__)


def extract_seq_with_ref_positions_and_aligned_pairs(aligned_pairs, start, end, query_seq) -> str:
    """
    Extract the query positions from a list of aligned pairs that fall within a given range.
    This replaces the cs_tag based extraction!

    Fixed by Liam Tran 6/24/24

    Args:
        aligned_pairs (list): List of aligned pairs.
        start (int): Start of the range.
        end (int): End of the range.

    Returns:
        The query sequence that aligned between the start and end params
        with deletions denoted with '.'s and insertions retained
    """
    result = []
    for query_pos, ref_pos in aligned_pairs:
        if ref_pos is None:
            if result:  # if result list is not empty, append None
                result.append(query_pos)
        elif start <= ref_pos <= end:
            result.append(query_pos)
        elif ref_pos > end:
            break  # exit loop if position is beyond the end of the range
    region_query_seq_list = [query_seq[i] if i else "." for i in result]
    region_query_sequence = "".join(region_query_seq_list)
    return region_query_sequence


# TODO: Finish this:
def check_contig_positions(ref_fasta: Path, contig_pos_map: Path):
    """
    This function will take in a reference fasta file and a contig position map file and extract the sequences
    from the reference fasta file based on the positions given in the contig position map file. This is useful
    for checking that the contig positions are correct for barcode splitting.
    :param ref_fasta: Path to the reference fasta file
    :param contig_pos_map: Path to the contig position map file
    :return: None
    """
    contig_positions_dict = parse_contig_pos_map(contig_pos_map)
    with open(ref_fasta, "r") as ref:
        ref_seqs = SeqIO.parse(ref, "fasta")
        for rec in ref_seqs:
            if rec.id in contig_positions_dict.keys():
                start, end = contig_positions_dict[rec.id]
                print(f"Extracting sequence for {rec.id} from {start} to {end}")
                print(rec.seq[start:end])
            else:
                print(f"Contig {rec.id} not found in contig position map file.")


def get_barcodes(barcodes_fasta=Path("/data16/marcus/working"
                                     "/231201_mutagenesisSequencing_TAD"
                                     "/initialPrimerDesign/ONT_sequences"
                                     "/ONT_barcodes_SQK-NBD114.24.fasta"),
                 rev_compliment=True) -> dict:
    # First lets load the barcodes fasta:
    print("Starting barcode load...", end="")
    if not rev_compliment:
        barcodes = {rec.id: str(rec.seq) for rec in SeqIO.parse(barcodes_fasta, "fasta")}
    else:
        barcodes = {rec.id: str(rec.seq.reverse_complement()) for rec in SeqIO.parse(barcodes_fasta, "fasta")}
    print("Done!")
    return barcodes


def pick_best_barcode(input_seq: str,
                      barcodes_dict: dict,
                      better_than_second_best_by: int = 4):
    dist_dict = {}
    seq_upper = input_seq.upper()
    for name, bar in barcodes_dict.items():
        dist_dict[name] = stringdist.levenshtein(seq_upper, bar.upper())
    closest = min(dist_dict, key=dist_dict.get)
    closest_dist = dist_dict[closest]
    second_best = sorted(dist_dict, key=dist_dict.get)[1]
    second_best_dist = dist_dict[second_best]

    if second_best_dist - closest_dist > better_than_second_best_by:
        return closest, closest_dist
    else:
        return "NotDistinct", closest_dist


def process_bam(args: argparse.Namespace, barcodes: dict, output_bam_paths: dict, output_dir: Path):
    contig = args.contig_target
    if not args.quiet:
        print("Starting BAM file parsing...")
    with (pysam.AlignmentFile(args.bam_file, "rb") as in_bam):
        iterator_total = in_bam.count(contig, args.ref_start, args.ref_end)
        if args.subsample is not None and 0 < args.subsample < iterator_total:
            iterator_total = args.subsample
        if not args.quiet:
            bam_iterator = tqdm(enumerate(in_bam.fetch(contig, args.ref_start, args.ref_end)),
                                total=iterator_total,
                                desc="Extracting barcodes")
        else:
            bam_iterator = enumerate(in_bam.fetch(contig, args.ref_start, args.ref_end))
        barcode_counter = {name: 0 for name in barcodes.keys()}
        barcode_counter["NotDistinct"] = 0
        # header = in_bam.header
        output_bam_paths["NotDistinct"] = output_dir / "NotDistinct.bam"

        with ExitStack() as stack:
            out_bams = {name: stack.enter_context(pysam.AlignmentFile(path, "wb", header=in_bam.header))
                        for name, path in output_bam_paths.items()}
            # out_bams["NotDistinct"] = stack.enter_context(pysam.AlignmentFile(output_dir / "NotDistinct.bam",
            #                                                                   "wb", header=header))
            
            # check that NotDistinct header is the same as the others
            assert out_bams["NotDistinct"].header == in_bam.header

            for i, read in bam_iterator:
                if args.subsample is not None and i >= args.subsample:
                    break
                if read.is_unmapped:
                    continue
                pairs = read.get_aligned_pairs(with_seq=False)
                read_seq = read.query_sequence
                extracted_seq = extract_seq_with_ref_positions_and_aligned_pairs(pairs, args.ref_start, args.ref_end, read_seq)
                barcode, distance = pick_best_barcode(extracted_seq, barcodes, args.dist)
                read.set_tag('bI', barcode)
                read.set_tag('bD', distance)
                out_bams[barcode].write(read)
                barcode_counter[barcode] += 1
                top_barcodes = sorted(barcode_counter.items(), key=lambda x: x[1], reverse=True)[:args.top_bar_count]
                if not args.quiet and i % 100 == 0:
                    bam_iterator.desc = (f"Extracting barcodes | {i} reads | "
                                         + " | ".join([f"{bar}: {count}" for bar, count in top_barcodes]))
    return barcode_counter


def process_bam2(args: argparse.Namespace, barcodes: dict, output_bam_paths: dict, output_dir: Path):
    contig = args.contig_target
    if not args.quiet:
        print("Starting BAM file parsing...")
    with (pysam.AlignmentFile(args.bam_file, "rb") as in_bam):
        iterator_total = in_bam.count(contig, args.ref_start, args.ref_end)
        if args.subsample is not None and 0 < args.subsample < iterator_total:
            iterator_total = args.subsample
        if not args.quiet:
            bam_iterator = tqdm(enumerate(in_bam.fetch(contig, args.ref_start, args.ref_end)),
                                total=iterator_total,
                                desc="Extracting barcodes")
        else:
            bam_iterator = enumerate(in_bam.fetch(contig, args.ref_start, args.ref_end))
        
        # Counter for easy tracking of barcode counts
        barcode_counter = {name: 0 for name in barcodes.keys()}
        barcode_counter["NotDistinct"] = 0
        # Dictionary to hold reads in memory before writing to output BAMs
        barcode_reads_dict = {name: [] for name in barcodes.keys()}
        barcode_reads_dict["NotDistinct"] = []
        for i, read in bam_iterator:
            if args.subsample is not None and i >= args.subsample:
                break
            if read.is_unmapped:
                continue
            pairs = read.get_aligned_pairs(with_seq=False)
            extracted_seq = extract_seq_with_ref_positions_and_aligned_pairs(pairs,
                                                                             args.ref_start,
                                                                             args.ref_end,
                                                                             read.query_sequence)
            barcode, distance = pick_best_barcode(extracted_seq, barcodes, args.dist)
            read.set_tag('bI', barcode)
            read.set_tag('bD', distance)
            barcode_reads_dict[barcode].append(read)
            barcode_counter[barcode] += 1
            top_barcodes = sorted(barcode_counter.items(),
                                  key=lambda x: x[1],
                                  reverse=True)[:args.top_bar_count]
            if not args.quiet and i % 100 == 0:
                bam_iterator.desc = (f"Extracting barcodes | {i} reads | "
                                     + " | ".join([f"{bar}: {count}" for bar, count in top_barcodes]))
                
        # Let's just never make empty BAM files, that seems unnecessary:
        barcode_counter_trimmed = {name: count for name, count in barcode_counter.items() if count > 0}
        total_mapped_reads = sum(barcode_counter_trimmed.values())
        if args.verbose:
            print(f"Total reads: {total_mapped_reads}")
            print(f"Barcodes with more than 0 reads assigned to them (total of {len(barcode_counter_trimmed)}):")
            pprint(barcode_counter_trimmed)
        # Dropping barcodes that had nothing mapped to them:
        barcode_reads_dict = {name: reads for name, reads in barcode_reads_dict.items() if len(reads) > 0}
        output_bam_paths_trimmed = {name: output_dir / f"{name}.bam" for name, out_path in
                                    output_bam_paths.items()
                                    if name in barcode_reads_dict.keys()}
        output_bam_paths_trimmed["NotDistinct"] = output_dir / "NotDistinct.bam"
        # Write the reads out of memory into the appropriate BAM files:
        with ExitStack() as stack:
            out_bams = {name: stack.enter_context(pysam.AlignmentFile(path, "wb", header=in_bam.header))
                        for name, path in output_bam_paths_trimmed.items()}
            for name, reads in barcode_reads_dict.items():
                if args.quiet:
                    bam_writing_iterator = reads
                else:
                    bam_writing_iterator = tqdm(reads, desc=f"Writing reads to {name} BAM")
                for read in bam_writing_iterator:
                    out_bams[name].write(read)
    return barcode_counter


def process_bam_contig_agnostic(args: argparse.Namespace, barcodes: dict, output_bam_paths: dict, output_dir: Path):
    """
    This method allows for the use of multiple contigs (chromosomes or whatever the reference fasta had),
    and it requires a contig position map file to specify the start and end positions for each contig.
    """

    if not args.quiet:
        print("Starting BAM file parsing...")
    if args.contig_pos_map is not None:
        contig_positions_dict = parse_contig_pos_map(args.contig_pos_map)
    else:
        raise ValueError("You must provide --contig_pos_map for contig agnostic barcode splitting.")

    barcode_reads_dict = {name: [] for name in barcodes.keys()}
    barcode_reads_dict["NotDistinct"] = []
    barcode_counter = {name: 0 for name in barcodes.keys()}
    barcode_counter["NotDistinct"] = 0
    total_mapped_reads = 0
    with (pysam.AlignmentFile(args.bam_file, "rb") as in_bam):
        contigs = in_bam.references
        if not args.quiet:
            print(f"Found {len(contigs)} contigs in the BAM file: {', '.join(contigs)}")

        for contig in contigs:
            try:
                contig_start, contig_end = contig_positions_dict[contig]
            except KeyError:
                print(f"Contig {contig} not found in contig position map file. Moving on.")
                continue
            
            barcode_sub_counter = {name: 0 for name in barcodes.keys()}
            barcode_sub_counter["NotDistinct"] = 0
            
            if not args.quiet:
                print(f"Processing contig {contig}, targeting positions "
                      f"{contig_start}-{contig_end} to extract barcodes.")
                
            if not args.quiet:
                iterator_total = in_bam.count(contig,
                                              contig_start,
                                              contig_end)
                if args.subsample is not None and 0 < args.subsample < iterator_total:
                    iterator_total = args.subsample
                bam_iterator = tqdm(enumerate(in_bam.fetch(contig,
                                                           contig_start,
                                                           contig_end)),
                                    total=iterator_total,
                                    desc=f"Barcodes for {contig}")
            else:
                bam_iterator = enumerate(in_bam.fetch(contig,
                                                      contig_start,
                                                      contig_end))

            for i, read in bam_iterator:
                if args.subsample is not None and i >= args.subsample:
                    break
                if read.is_unmapped:
                    continue
                aligned_pairs = read.get_aligned_pairs(with_seq=False)
                
                extracted_seq = extract_seq_with_ref_positions_and_aligned_pairs(aligned_pairs,
                                                                                 contig_start,
                                                                                 contig_end,
                                                                                 read.query_sequence)
                barcode, distance = pick_best_barcode(extracted_seq, barcodes, args.dist)
                read.set_tag('bI', barcode)
                read.set_tag('bD', distance)
                barcode_reads_dict[barcode].append(read)
                barcode_counter[barcode] += 1
                barcode_sub_counter[barcode] += 1
                total_mapped_reads += 1

                top_barcodes = sorted(barcode_sub_counter.items(),
                                      key=lambda x: x[1],
                                      reverse=True)[:args.top_bar_count]
                if not args.quiet and i % 100 == 0:
                    bam_iterator.desc = (f"Barcodes for {contig} | {i} reads | "
                                         + " | ".join([f"{bar}: {count}" for bar, count in top_barcodes]))
        # Let's just never make empty BAM files, that seems unnecessary:
        barcode_counter_trimmed = {name: count for name, count in barcode_counter.items() if count > 0}
        if args.verbose:
            print(f"Total reads: {total_mapped_reads}")
            print(f"Barcodes with more than 0 reads assigned to them (total of {len(barcode_counter_trimmed)}):")
            pprint(barcode_counter_trimmed)
        barcode_reads_dict = {name: reads for name, reads in barcode_reads_dict.items() if len(reads) > 0}
        output_bam_paths_trimmed = {name: output_dir / f"{name}.bam" for name, out_path in output_bam_paths.items()
                                    if name in barcode_reads_dict.keys()}
        output_bam_paths_trimmed["NotDistinct"] = output_dir / "NotDistinct.bam"
    
        # This ExitStack context manager is sweet. Lets us open a bunch of files and then close them all at once.
        with ExitStack() as stack:
            out_bams = {name: stack.enter_context(pysam.AlignmentFile(path, "wb", header=in_bam.header))
                        for name, path in output_bam_paths_trimmed.items()}
            for name, reads in barcode_reads_dict.items():
                for read in tqdm(reads, desc=f"Writing reads to {name} BAM"):
                    out_bams[name].write(read)
    return barcode_counter


def process_bam_file(args):
    if args.barcodes is None:
        barcodes = get_barcodes(rev_compliment=(not args.not_rev_compliment))
    else:
        barcodes = get_barcodes(args.barcodes, rev_compliment=(not args.not_rev_compliment))

    if args.output_dir is None:
        output_dir = args.bam_file.parent / "demultiplexed_bams"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    output_bam_paths = {name: output_dir / f"{name}.bam" for name in barcodes.keys()}
    output_bam_paths["NotDistinct"] = output_dir / "NotDistinct.bam"
    if args.verbose:
        print("Outputs will be saved to the following paths")
        pprint(output_bam_paths)

    if args.contig_target is not None:
        barcode_counter = process_bam2(args, barcodes, output_bam_paths, output_dir)
    elif args.contig_pos_map is not None:
        barcode_counter = process_bam_contig_agnostic(args, barcodes, output_bam_paths, output_dir)
    else:
        raise ValueError("You must provide either --contig_target or --contig_pos_map")
    
    successful_barcode_counter = {name: count for name, count in barcode_counter.items() if count > 0}
    if not args.quiet:
        print(f"Barcodes with more than 0 reads assigned to them (total of {len(successful_barcode_counter)}):")
        pprint(successful_barcode_counter)
    for name, count in barcode_counter.items():
        if count < args.min_reads:
            if args.verbose:
                print(f"Removing BAM file for barcode {name} with {count} reads")
            try:
                output_bam_paths[name].unlink()
            except FileNotFoundError:
                if args.verbose:
                    print(f"No BAM file found for barcode {name}")
        else:
            if args.verbose:
                print(f"Keeping BAM file for barcode {name} with {count} reads")
            if args.samtools:
                if args.verbose:
                    print(f"Indexing BAM file and producing SAM file for barcode {name}")
                try:
                    pysam.index(str(output_bam_paths[name]))
                except pysam.utils.SamtoolsError as e:
                    print(f"Could not index {output_bam_paths[name]}\n{e}")
                try:
                    call = f"samtools view -h -o {output_bam_paths[name].with_suffix('.sam')} {output_bam_paths[name]}"
                    subprocess.run(call, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Could not produce SAM file for {output_bam_paths[name]}\n{e}")
    if args.merged_bam:
        if args.verbose:
            print("Merging all barcode matched reads into a single BAM file.")
        merged_bam_path = output_dir / "merged_barcodes.bam"
        merge_targets_txt_file = output_dir / "merge_targets.txt"
        with open(merge_targets_txt_file, "w") as merge_targets:
            for name, count in barcode_counter.items():
                if count >= args.min_reads and name != "NotDistinct":
                    merge_targets.write(f"{output_bam_paths[name]}\n")
        try:
            call = f"samtools merge -frb {merge_targets_txt_file} -o {merged_bam_path}"
            subprocess.run(call, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Could not merge barcodes into {merged_bam_path}\n{e}")
        if args.samtools:
            try:
                pysam.index(str(merged_bam_path))
            except pysam.utils.SamtoolsError as e:
                print(f"Could not index {merged_bam_path}\n{e}")
            try:
                call = f"samtools view -h -o {merged_bam_path.with_suffix('.sam')} {merged_bam_path}"
                subprocess.run(call, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Could not produce SAM file for {merged_bam_path}\n{e}")
    if not args.verbose and not args.quiet:
        dropped_barcodes = [name for name, count in barcode_counter.items() if count < args.min_reads]
        kept_barcodes = [name for name, count in barcode_counter.items() if count >= args.min_reads]
        print(f"Removed BAM files for barcodes with fewer than {args.min_reads} reads: {dropped_barcodes}")
        print(f"Kept BAM files for barcodes with at least {args.min_reads} reads: {kept_barcodes}")


def parse_contig_pos_map(contig_pos_map: Path) -> Dict[str, Tuple[int, int]]:
    """
    Parse a contig position map file into a dictionary. The general format of the file should be:
    contig_name    start_position    end_position
    W/ tabs rather than spaces.
    
    The positions should be integers.
    :param contig_pos_map: path to the contig position map file
    :return: dictionary with contig names as keys and tuples of start and end positions as values
    """
    contig_pos_dict = {}
    with open(contig_pos_map, "r") as cpm:
        for line in cpm:
            contig, start, end = line.strip().split("\t")
            contig_pos_dict[contig] = (int(start), int(end))
    return contig_pos_dict


def arg_parse():
    parser = argparse.ArgumentParser(description="Split a BAM file on the basis of the barcodes contained within.")
    parser.add_argument("bam_file", type=Path, help="The BAM file to be split.")
    
    parser.add_argument("--contig_target", type=str, default=None,
                        help="The name of the contig to target for barcode splitting. If not specified, "
                             "all contigs will be attempted but a contig_pos_map file must be provided.")
    parser.add_argument("--ref_start", type=int, default=779,
                        help="The start position in the reference to pull the barcode from.")
    parser.add_argument("--ref_end", type=int, default=803,
                        help="The end position in the reference to pull the barcode from.")
    parser.add_argument("--contig_pos_map", type=Path, default=None,
                        help="A tab-seperated file containing contig names, start, and end "
                             "positions of where the barcodes are. This is mutually exclusive with "
                             "contig_target, ref_start, and ref_end.")
    
    parser.add_argument("-b", "--barcodes", type=Path, default=None,
                        help="The fasta file containing the barcodes to be used. Default is the ONT barcodes.")
    parser.add_argument("-r", "--ref_name", type=str, default=None,
                        help="The fastq file of the reference that the BAM file was aligned to.")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Subsample the BAM file to this number of reads before splitting.")
    
    parser.add_argument("-o", "--output_dir", type=Path, default=None,
                        help="The directory to output the split BAM files to. Default is the current directory.")
    
    parser.add_argument("-d", "--dist", type=int, default=4,
                        help="The maximum distance a barcode can be from the second closest barcode "
                             "to be considered distinct.")
    
    parser.add_argument("-v", "--verbose", action="store_true", help="Print out extra information.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Do not print out any information.")
    parser.add_argument("--not_rev_compliment", action="store_true",
                        help="Use this flag if the barcodes in the fasta file are not reverse compliments of "
                             "the actual barcodes.")
    
    parser.add_argument("--top_bar_count", type=int, default=4,
                        help="The number of top barcodes to show in the progress bar.")
    parser.add_argument("--min_reads", type=int, default=100,
                        help="The minimum number of reads for a barcodes bam file to be retained.")
    parser.add_argument("--samtools", action="store_true",
                        help="Use samtools to index the final BAMs and produce SAM files.")
    parser.add_argument("--merged_bam", action="store_true",
                        help="This flag will cause the script to also output a single BAM file will all "
                             "barcode matched reads. Each read will have a tag indicating the barcode it "
                             "was matched to.")
    args = parser.parse_args()
    
    # Custom logic to check if we have contig and barcode positions:
    if args.contig_pos_map is None and (args.contig_target is None or args.ref_start is None or args.ref_end is None):
        parser.error("Either --contig_pos_map or all of --contig_target, --ref_start, and --ref_end must be provided")

    if args.verbose and args.quiet:
        raise ValueError("Cannot specify both verbose and quiet!")

    return args


def main():
    args = arg_parse()
    process_bam_file(args)


if __name__ == "__main__":
    main()
