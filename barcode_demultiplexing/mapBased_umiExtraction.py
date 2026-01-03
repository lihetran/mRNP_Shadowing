'''
July 25, 2024 LT
Putting Marcus Viscardi's code from mapBased_umiExtraction.ipynb into a script for ease of use.

This script will extract UMIs from a BAM file based on a reference sequence and UMI positions.
Input:
    - BAM file
    - Reference sequence file
    - Project directory
    - Output directory for the tagged BAM file
Output:
    - Tagged BAM file with UMI information
'''

import os
import seaborn
import sys
import subprocess
import pandas as pd
import numpy as np

import pickle

from pathlib import Path
from icecream import ic
from datetime import datetime

from pprint import pprint

from tqdm.auto import tqdm

from Bio import SeqIO
import pysam

import seaborn as sea
import matplotlib.pyplot as plt
import argparse

from typing import Dict, List

# from cs_parsing import extract_ref_and_query_region2, bam_to_umi_df2, bam_to_tagged_bam2, bam_to_df

import resource
import sys

IUPAC_DNA = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "R": "AG",
    "Y": "CT",
    "S": "GC",
    "W": "AT",
    "K": "GT",
    "M": "AC",
    "B": "CGT",
    "D": "AGT",
    "H": "ACT",
    "V": "ACG",
    "N": "ACGT"
}

def memory_limit_half():
    """Limit max memory usage to half."""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    # Convert KiB to bytes, and divide in two to half
    resource.setrlimit(resource.RLIMIT_AS, (int(get_memory() * 1024 / 2), hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory  # KiB


def extract_positions_within_range(positions, start, end):
    """
    Extract positions within a given range from a list of positions.
    None values, representing deletions, are also included if they fall within the range.

    Args:
        positions (list): List of positions.
        start (int): Start of the range.
        end (int): End of the range.

    Returns:
        list: Positions within the given range.
    """
    result = []
    for pos in positions:
        if pos is None:
            if result:  # if result list is not empty, append None
                result.append(pos)
        elif start <= pos <= end:
            result.append(pos)
        elif pos > end:
            break  # exit loop if position is beyond the end of the range
    return result

def extract_query_positions_from_ref(aligned_pairs, start, end):
    """
    Extract the query positions from a list of aligned pairs that fall within a given range.

    Args:
        aligned_pairs (list): List of aligned pairs.
        start (int): Start of the range.
        end (int): End of the range.

    Returns:
        list: Query positions within the given range.
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
    return result

def extract_ref_and_query_region2(target_entry: pysam.AlignedSegment, ref_seq: str,
                                  region_start: int, region_end: int) -> dict:
    """
    Extracts reference and query regions from a given aligned segment.

    This function extracts the reference and query positions and sequences from a given aligned segment that fall within a specified range.
    It also provides the option to write the output to a file and print the output.

    Args:
        target_entry (pysam.AlignedSegment): The aligned segment from which to extract the reference and query regions.
        ref_seq (str): The reference sequence.
        region_start (int): The start of the range within which to extract the reference and query regions.
        region_end (int): The end of the range within which to extract the reference and query regions.

    Returns:
        dict:
            ref_positions: The positions along the reference that the query aligned to (with deletions marked)
            ref_sequence: The sequence of the reference that the query aligned to (with deletions marked)
            query_positions: The positions along the query that aligned to the reference (with insertions marked)
            query_sequence: The sequence of the query that aligned to the reference (with insertions marked)
            ins_count: The number of insertions in the query sequence
            del_count: The number of deletions in the query sequence
            mismatch_count: The number of mismatches between the query and reference sequences
            perfect_match: Whether the query sequence perfectly matches the reference sequence
    """
    real_ref_seq = ref_seq
    region_ref_positions = extract_positions_within_range(target_entry.get_reference_positions(full_length=True),
                                                          region_start, region_end)
    region_ref_sequence = [real_ref_seq[i] if i else "." for i in region_ref_positions]
    # Now we need to be able to pull out the same nucleotides for the actual query sequence
    aligned_pairs = target_entry.get_aligned_pairs(with_seq=False)
    region_query_positions = extract_query_positions_from_ref(aligned_pairs, region_start, region_end)
    region_query_sequence = [target_entry.query_sequence[i] if i else "." for i in region_query_positions]

    region_ref_sequence_matched, region_query_sequence_matched = [], []
    ins_count, del_count, was_perfect, mismatch_count = 0, 0, True, 0
    if not region_ref_sequence or not region_query_sequence:
        was_perfect = False
    for ref, query in zip(region_ref_sequence, region_query_sequence):
        ref, query = ref.upper(), query.upper()
        if ref == query:
            region_ref_sequence_matched.append(ref)
            region_query_sequence_matched.append(query)
        elif ref == ".":
            region_ref_sequence_matched.append(ref.lower())
            region_query_sequence_matched.append(query.lower())
            ins_count += 1
            was_perfect = False
        elif query == ".":
            region_ref_sequence_matched.append(ref.lower())
            region_query_sequence_matched.append(query.lower())
            del_count += 1
            was_perfect = False
        elif query in IUPAC_DNA[ref]:
            region_ref_sequence_matched.append(ref)
            region_query_sequence_matched.append(query)
        else:
            region_ref_sequence_matched.append(ref.lower())
            region_query_sequence_matched.append(query.lower())
            was_perfect = False
            mismatch_count += 1
    return_dict = {'ref_positions': region_ref_positions,
                   'ref_sequence': ''.join(region_ref_sequence_matched),
                   'query_positions': region_query_positions,
                   'query_sequence': ''.join(region_query_sequence_matched),
                   'ins_count': ins_count, 'del_count': del_count, 'mismatch_count': mismatch_count,
                   'perfect_match': was_perfect}
    return return_dict

def umi_alignment_checker(extracted_dict: dict, max_del_in_umi: int, max_ins_in_umi: int, 
                          max_iupac_mismatches: int, umi_length: int, umi_seq = 'ABBBBAABBBBAABBBBAABBBBAA') -> bool:
    '''
    This function will check if the extracted UMI sequence passes the following criteria:
    - Number of deletions <= max_del_in_umi
    - Number of insertions <= max_ins_in_umi
    - Number of mismatches <= max_iupac_mismatches
    - Length of UMI == umi_length

    Since UMIs are expected to have the sequence: ABBBBAABBBBAABBBBAABBBBAA
    We want to make sure that the extracted_seq aligning here has A's in the right places.
    This will allow us to relax the mismatch, deletion, insertion count a bit.
    '''
    extracted_seq = extracted_dict['query_sequence'].upper()
    ins_count = extracted_dict['ins_count']
    del_count = extracted_dict['del_count']
    mismatch_count = extracted_dict['mismatch_count']
    ref_seq = extracted_dict['ref_sequence'].upper()

    A_count = umi_seq.count('A')
    ct = 0
    for index, (q, r) in enumerate(zip(extracted_seq, ref_seq)):
        if ref_seq[index] == 'A' and extracted_seq == 'A':
            ct += 1
    if ct != A_count:
        return False
    if len(extracted_seq) != umi_length:
        return False
    if ins_count > max_ins_in_umi:
        return False
    if del_count > max_del_in_umi:
        return False
    if mismatch_count > max_iupac_mismatches:
        return False
    else:
        return True


def bam_to_tagged_bam2(bam_file_path: Path,
                       target_chr: str,
                       ref_seq: str,
                       umi_ref_start: int,
                       umi_ref_end: int,
                       flanking_seq_to_capture=0,
                       save_dir=None,
                       mkdir=True,
                       read_id_suffix="",
                       add_read_id_suffix_from_tag=None,
                       save_suffix=".tagged.bam",
                       max_del_in_umi=2,
                       max_ins_in_umi=2,
                       max_iupac_mismatches=2,
                       restrict_to_length=True,
                       bam_tag_dict: Dict[str, str] = None,
                       subset_count=-1) -> Path:
    """
    Extract UMIs from a BAM file and write them to a new BAM file with the UMI sequence
    and deletion count stored in tags.
    :param bam_file_path: 
    :param target_chr: 
    :param ref_seq: 
    :param umi_ref_start: 
    :param umi_ref_end: 
    :param flanking_seq_to_capture: 
    :param save_dir: 
    :param mkdir: 
    :param read_id_suffix: 
    :param add_read_id_suffix_from_tag: 
    :param save_suffix: 
    :param max_del_in_umi: 
    :param max_ins_in_umi: 
    :param max_iupac_mismatches: 
    :param restrict_to_length: 
    :param bam_tag_dict: 
    :param subset_count: 
    :return: 
    """
    assert bam_file_path.exists(), f"The provided bam file {bam_file_path} does not exist"
    
    bam_tag_dict_required_keys = [
        "umi_sequence",
        "deletion_count",
        "insertion_count",
        "mismatch_count",
        "umi_length"]
    if bam_tag_dict is None:
        bam_tag_dict = {
            "umi_sequence": "uM",
            "deletion_count": "uD",
            "insertion_count": "uI",
            "mismatch_count": "um",
            "umi_length": "uL",
        }
    else:
        for key in bam_tag_dict_required_keys:
            if key not in bam_tag_dict:
                raise ValueError(f"The provided bam_tag_dict is missing the required key: {key}")

    with pysam.AlignmentFile(bam_file_path, 'rb') as input_bam:
        iterator_total = input_bam.count(target_chr, umi_ref_start, umi_ref_end)
        if 0 < subset_count < iterator_total:
            iterator_total = subset_count
        bam_iterator = tqdm(enumerate(input_bam.fetch(target_chr,
                                                      umi_ref_start,
                                                      umi_ref_end)),
                            total=iterator_total,
                            desc="Extracting UMIs")
        
        if save_dir is None:
            output_bam_path = bam_file_path.with_suffix(save_suffix)
        else:
            if not Path(save_dir).exists() and not mkdir:
                raise FileNotFoundError(f"The save directory {save_dir} does not exist and mkdir is set to False")
            elif not Path(save_dir).exists() and mkdir:
                Path(save_dir).mkdir(exist_ok=True)
            output_bam_path = Path(save_dir) / bam_file_path.with_suffix(save_suffix).name
        umi_success_count = 0
        umi_drop_count = 0
        dropped_for_del, dropped_for_ins, dropped_for_mismatch = 0, 0, 0
        dropped_for_length = 0
        umi_output_set = set()
        with pysam.AlignmentFile(output_bam_path, "wb", header=input_bam.header) as out_bam:
            for i, entry in bam_iterator:
                try:
                    entry_dict = extract_ref_and_query_region2(entry, ref_seq,
                                                               umi_ref_start - flanking_seq_to_capture,
                                                               umi_ref_end + flanking_seq_to_capture)
                    extracted_seq: str = entry_dict["query_sequence"]
                    ins_count: int = entry_dict["ins_count"]
                    del_count: int = entry_dict["del_count"]
                    mismatch_count: int = entry_dict["mismatch_count"]
                    if restrict_to_length:
                        if len(extracted_seq) != umi_ref_end - umi_ref_start + 1: 
                            # ic(len(extracted_seq), umi_ref_end - umi_ref_start)
                            length_cutoff_passed = False
                        else:
                            length_cutoff_passed = True
                    else:
                        length_cutoff_passed = True
                    if del_count <= max_del_in_umi \
                            and ins_count <= max_ins_in_umi \
                            and mismatch_count <= max_iupac_mismatches \
                            and length_cutoff_passed:
                        umi_output_set.add(extracted_seq)
                        entry.set_tag(bam_tag_dict['umi_sequence'], extracted_seq, value_type='Z')
                        entry.set_tag(bam_tag_dict['deletion_count'], del_count, value_type='i')
                        entry.set_tag(bam_tag_dict['insertion_count'], ins_count, value_type='i')
                        entry.set_tag(bam_tag_dict['mismatch_count'], mismatch_count, value_type='i')
                        entry.set_tag(bam_tag_dict['umi_length'], len(extracted_seq), value_type='i')
                        entry.query_name += read_id_suffix
                        if add_read_id_suffix_from_tag:
                            entry.query_name += f"_{entry.get_tag(add_read_id_suffix_from_tag)}"
                        out_bam.write(entry)
                        umi_success_count += 1
                    else:
                        umi_drop_count += 1
                        if del_count > max_del_in_umi:
                            dropped_for_del += 1
                        if ins_count > max_ins_in_umi:
                            dropped_for_ins += 1
                        if mismatch_count > max_iupac_mismatches:
                            dropped_for_mismatch += 1
                        if not length_cutoff_passed:
                            dropped_for_length += 1

                    if i % 100 == 0:
                        bam_iterator.desc = (f"Extracting UMIs | {i:,} reads | "
                                             f"{len(umi_output_set):,} unique UMIs | "
                                             f"{umi_drop_count:,} dropped UMIs")
                    if 0 < subset_count <= i:
                        break
                except IndexError as e:
                    print(entry.reference_start, entry.reference_end, entry.get_tag('cs'), e)
                    continue
                except ValueError as e:
                    print(entry.reference_start, entry.reference_end, entry.get_tag('cs'), e)
                    continue
                except Exception as e:
                    print(entry.reference_start, entry.reference_end, entry.get_tag('cs'), e)
                    raise e
    summary_string = (
        f"\n"
        f"Extracted {len(umi_output_set):>8,} unique UMIs from {i + 1:>8,} reads\n"
        f"Wrote     {umi_success_count:>8,} reads to {output_bam_path.name}\n"
        f"Dropped   {umi_drop_count:>8,} UMIs for having too many deletions, insertions, or mismatches\n"
        f"Breakdown (reads can fit into multiple categories): \n"
        f"\tDropped {dropped_for_del:>8,} UMIs for having too many deletions (>{max_del_in_umi})\n"
        f"\tDropped {dropped_for_ins:>8,} UMIs for having too many insertions (>{max_ins_in_umi})\n"
        f"\tDropped {dropped_for_mismatch:>8,} UMIs for having too many mismatches (>{max_iupac_mismatches})\n")
    if restrict_to_length:
        summary_string += (f"\tDropped {dropped_for_length:>8,} UMIs for not being the expected length "
                           f"({umi_ref_end - umi_ref_start + 1} nts)\n")
    print(summary_string)
    return output_bam_path

def bam_to_tagged_bam3(bam_file_path: Path,
                       ref_seq: str,
                       umi_ref_start: int,
                       umi_ref_end: int,
                       flanking_seq_to_capture=0,
                       save_dir=None,
                       mkdir=True,
                       read_id_suffix="",
                       add_read_id_suffix_from_tag=None,
                       save_suffix=".tagged.bam",
                       max_del_in_umi=2,
                       max_ins_in_umi=2,
                       max_iupac_mismatches=2,
                       restrict_to_length=True,
                       bam_tag_dict: Dict[str, str] = None,
                       subset_count=-1) -> Path:
    """
    240807
    LT editing MV's function to work with multiple contigs

    Extract UMIs from a BAM file and write them to a new BAM file with the UMI sequence
    and deletion count stored in tags.
    :param bam_file_path: 
    :param target_chr: 
    :param ref_seq: 
    :param umi_ref_start: 
    :param umi_ref_end: 
    :param flanking_seq_to_capture: 
    :param save_dir: 
    :param mkdir: 
    :param read_id_suffix: 
    :param add_read_id_suffix_from_tag: 
    :param save_suffix: 
    :param max_del_in_umi: 
    :param max_ins_in_umi: 
    :param max_iupac_mismatches: 
    :param restrict_to_length: 
    :param bam_tag_dict: 
    :param subset_count: 
    :return: 
    """
    assert bam_file_path.exists(), f"The provided bam file {bam_file_path} does not exist"
    
    bam_tag_dict_required_keys = [
        "umi_sequence",
        "deletion_count",
        "insertion_count",
        "mismatch_count",
        "umi_length"]
    if bam_tag_dict is None:
        bam_tag_dict = {
            "umi_sequence": "uM",
            "deletion_count": "uD",
            "insertion_count": "uI",
            "mismatch_count": "um",
            "umi_length": "uL",
        }
    else:
        for key in bam_tag_dict_required_keys:
            if key not in bam_tag_dict:
                raise ValueError(f"The provided bam_tag_dict is missing the required key: {key}")

    with pysam.AlignmentFile(bam_file_path, 'rb') as input_bam:
        contigs = input_bam.references
        for target_chr in contigs:
            iterator_total = input_bam.count(target_chr, umi_ref_start, umi_ref_end)
            if 0 < subset_count < iterator_total:
                iterator_total = subset_count
            bam_iterator = tqdm(enumerate(input_bam.fetch(target_chr,
                                                        umi_ref_start,
                                                        umi_ref_end)),
                                total=iterator_total,
                                desc="Extracting UMIs")
            
        if save_dir is None:
            output_bam_path = bam_file_path.with_suffix(save_suffix)
        else:
            if not Path(save_dir).exists() and not mkdir:
                raise FileNotFoundError(f"The save directory {save_dir} does not exist and mkdir is set to False")
            elif not Path(save_dir).exists() and mkdir:
                Path(save_dir).mkdir(exist_ok=True)
            output_bam_path = Path(save_dir) / bam_file_path.with_suffix(save_suffix).name
        umi_success_count = 0
        umi_drop_count = 0
        dropped_for_del, dropped_for_ins, dropped_for_mismatch = 0, 0, 0
        dropped_for_length = 0
        umi_output_set = set()
        with pysam.AlignmentFile(output_bam_path, "wb", header=input_bam.header) as out_bam:
            for i, entry in bam_iterator:
                try:
                    entry_dict = extract_ref_and_query_region2(entry, ref_seq,
                                                            umi_ref_start - flanking_seq_to_capture,
                                                            umi_ref_end + flanking_seq_to_capture)
                    extracted_seq: str = entry_dict["query_sequence"]
                    ins_count: int = entry_dict["ins_count"]
                    del_count: int = entry_dict["del_count"]
                    mismatch_count: int = entry_dict["mismatch_count"]
                    if restrict_to_length:
                        if len(extracted_seq) != umi_ref_end - umi_ref_start + 1: 
                            # ic(len(extracted_seq), umi_ref_end - umi_ref_start)
                            length_cutoff_passed = False
                        else:
                            length_cutoff_passed = True
                    else:
                        length_cutoff_passed = True
                    if del_count <= max_del_in_umi \
                            and ins_count <= max_ins_in_umi \
                            and mismatch_count <= max_iupac_mismatches \
                            and length_cutoff_passed:
                        umi_output_set.add(extracted_seq)
                        entry.set_tag(bam_tag_dict['umi_sequence'], extracted_seq, value_type='Z')
                        entry.set_tag(bam_tag_dict['deletion_count'], del_count, value_type='i')
                        entry.set_tag(bam_tag_dict['insertion_count'], ins_count, value_type='i')
                        entry.set_tag(bam_tag_dict['mismatch_count'], mismatch_count, value_type='i')
                        entry.set_tag(bam_tag_dict['umi_length'], len(extracted_seq), value_type='i')
                        entry.query_name += read_id_suffix
                        if add_read_id_suffix_from_tag:
                            entry.query_name += f"_{entry.get_tag(add_read_id_suffix_from_tag)}"
                        out_bam.write(entry)
                        umi_success_count += 1
                    else:
                        umi_drop_count += 1
                        if del_count > max_del_in_umi:
                            dropped_for_del += 1
                        if ins_count > max_ins_in_umi:
                            dropped_for_ins += 1
                        if mismatch_count > max_iupac_mismatches:
                            dropped_for_mismatch += 1
                        if not length_cutoff_passed:
                            dropped_for_length += 1

                    if i % 100 == 0:
                        bam_iterator.desc = (f"Extracting UMIs | {i:,} reads | "
                                            f"{len(umi_output_set):,} unique UMIs | "
                                            f"{umi_drop_count:,} dropped UMIs")
                    if 0 < subset_count <= i:
                        break
                except IndexError as e:
                    print(entry.reference_start, entry.reference_end, entry.get_tag('cs'), e)
                    continue
                except ValueError as e:
                    print(entry.reference_start, entry.reference_end, entry.get_tag('cs'), e)
                    continue
                except Exception as e:
                    print(entry.reference_start, entry.reference_end, entry.get_tag('cs'), e)
                    raise e
    summary_string = (
        f"\n"
        f"Extracted {len(umi_output_set):>8,} unique UMIs from {i + 1:>8,} reads\n"
        f"Wrote     {umi_success_count:>8,} reads to {output_bam_path.name}\n"
        f"Dropped   {umi_drop_count:>8,} UMIs for having too many deletions, insertions, or mismatches\n"
        f"Breakdown (reads can fit into multiple categories): \n"
        f"\tDropped {dropped_for_del:>8,} UMIs for having too many deletions (>{max_del_in_umi})\n"
        f"\tDropped {dropped_for_ins:>8,} UMIs for having too many insertions (>{max_ins_in_umi})\n"
        f"\tDropped {dropped_for_mismatch:>8,} UMIs for having too many mismatches (>{max_iupac_mismatches})\n")
    if restrict_to_length:
        summary_string += (f"\tDropped {dropped_for_length:>8,} UMIs for not being the expected length "
                        f"({umi_ref_end - umi_ref_start + 1} nts)\n")
    print(summary_string)
    return output_bam_path



def main():
    parser = argparse.ArgumentParser(description="Extract UMIs from a BAM file based on a reference sequence and UMI positions")
    parser.add_argument("-b", "--bam_file", type=str, help="Path to the BAM file")
    parser.add_argument("-r", "--reference_file", type=str, help="Path to the reference sequence file")
    parser.add_argument("-p", "--project_dir", type=str, help="Path to the project directory")


    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    if not project_dir.exists():
        os.makedirs(project_dir)
    COMETm_dir = project_dir / "COMETm"
    if not COMETm_dir.exists():
        os.makedirs(COMETm_dir)
    mapBased_dir = COMETm_dir / "mapBased"
    if not mapBased_dir.exists():
        os.makedirs(mapBased_dir)
    contig_umi_positions_file = mapBased_dir / "contig_umi_positions.tsv"
    if not contig_umi_positions_file.exists():
        raise FileNotFoundError(f"contig_umi_positions.tsv not found in {mapBased_dir}")

    
    target_lib_dir = project_dir / "output_dir"  
    if not target_lib_dir.exists():
        os.makedirs(target_lib_dir)
    bam_dir = target_lib_dir / "mapReads"
    if not bam_dir.exists():
        os.makedirs(bam_dir)
    bam_file = Path(args.bam_file)
    umi_dir = target_lib_dir / "umi"
    if not umi_dir.exists():
        os.makedirs(umi_dir)

    # First let's quickly ID the contigs that we are interested in
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        contigs = bam.references
        ic(len(contigs), contigs)
        # if len(contigs) > 1:
        #     raise NotImplementedError("More than one contig in BAM file, need to update code to handle this!!")
        # rename contig to match reference sequence name
        # contig = contigs[0]
        reference_path = ""
        for entry in bam.header['PG']:
            id = entry['ID']
            if id == 'minimap2' and not reference_path:
                # reference_path = entry['CL'].split()[-2]
                ic(entry['CL'])
                reference_path = args.reference_file
                break
        ic(reference_path)
        ref_dict = SeqIO.to_dict(SeqIO.parse(reference_path, "fasta"))
        # ref = Path(reference_path)
        # ref_seq = SeqIO.read(ref, "fasta")
        # assert ref_seq.id == contig, f"Contig name in BAM file ({contig}) does not match reference sequence name ({ref_seq.id})"
        # ref_seq = str(ref_seq.seq)
        ic(ref_dict)

    # Now let's check that the needed contig is in the contig_umi_positions.tsv file:
    contig_df = pd.read_csv(contig_umi_positions_file, sep="\t")
    ic(contig_df)
    contig_df['umi_positions'] = contig_df['umi_positions'].apply(eval)
    contig_df['umi_positions'] = contig_df['umi_positions'].apply(lambda x: [tuple(i) for i in x])
    contigs_umi_positions = contig_df.set_index("contig_name").to_dict()['umi_positions']
    # ic(contigs_umi_positions)
    for contig in contigs:
        try:
            contig_umi_positions = contigs_umi_positions[contig]
            print(f"Found position(s) for UMI(s) in contig {contig}\n"
                f"\tPositions: {contig_umi_positions}")
        except KeyError:
            raise KeyError(f"Contig {contig} not found in contig_umi_positions.tsv")
    

    flanking_seq_to_capture = 0
    umi_set_target = 0  # 0 indexed!
    # umi_positions = contig_umi_positions[umi_set_target]
    contig_bam_list = []
    for contig in contigs_umi_positions: # need to modify so that bam_to_tagged_bam2 is called for each contig. Then we can merge the BAM files
        # ic(contig)
        umi_pos = contigs_umi_positions[contig][0]
        # ic(umi_pos)
        ref_seq = ref_dict[contig].seq
        ic(umi_pos, ref_seq[umi_pos[0]-flanking_seq_to_capture:umi_pos[1]+flanking_seq_to_capture])
        subprocess.run(["samtools", "index", str(bam_file)])
        tagged_bam1 = bam_to_tagged_bam2(bam_file, contig, ref_seq,
                                        umi_pos[0], umi_pos[1],
                                        save_dir=umi_dir,
                                        flanking_seq_to_capture=flanking_seq_to_capture,
                                        bam_tag_dict={"umi_sequence": f"u{umi_set_target+1}", "deletion_count": f"d{umi_set_target+1}", "insertion_count": f"i{umi_set_target+1}", "mismatch_count": f"m{umi_set_target+1}", "umi_length": f"l{umi_set_target+1}"},
                                        # subset_count=1000,
                                        )
        # add contig name to tagged BAM file
        tagged_bam1 = Path(tagged_bam1) 
        ic(tagged_bam1)
        tagged_bam1 = tagged_bam1.rename(tagged_bam1.with_name(f"{tagged_bam1.stem}_{contig}{tagged_bam1.suffix}"))
        ic(tagged_bam1)

        subprocess.run(["samtools", "index", str(tagged_bam1)])
        # contig_bam_list.append(tagged_bam1)
    # print("Merging BAM files")
    # subprocess.run(["samtools", "merge", str(bam_dir / "merged_contigs.bam"), *[str(bam) for bam in contig_bam_list]])

if __name__ == "__main__":
    memory_limit_half()
    try:
        main()
    except MemoryError:
        print("Memory Error")
        sys.exit(1)
