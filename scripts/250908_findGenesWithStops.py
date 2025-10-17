'''
September 8, 2025 LT

I need to ensure that my scripts "shadowBamToPickle2.py" and "editingAroundStops.py" are working as expected.
To do this, this script will find all reads that have a stop codon by checking if 'T' is present in the 'gene_features' string.
I will also compute alignments to make sure that the feature string, query sequence, and edit string all look good. I'll write this out
to a separate file so that I can easily inspect the results.

input:
    - longjam file
    - bam file
    - reference genome (fasta)

output:
    - .txt file with read ID
                     query_seq_aligned
                     edit_string
                     feature_string
                     ref_string
'''

import pysam
import sys
from Bio import SeqIO

def extract_features_from_longjam(file):
    

    read_feature_dict = {}
    nextLine = ''
    print("Extracting features from longjam file...")
    with open(file, 'r') as f:
        for line in f:
            if not line.startswith('@'):
                read_id = line.strip().split('\t')[0]
                nextLine = next(f)
                featureString = nextLine.strip().split('\t')[0]
                read_feature_dict[read_id] = featureString

    return read_feature_dict

def get_absolute_positions(read):
    '''need to calculate absolute position of the read in the reference sequence, will do this by getting the start of 
    the alignment (4th field in the sam file) and the CIGAR string (5th field in the sam file) to get the absolute positions.'''
    # get the start position of the read
    start = read.query_alignment_start
    # get the CIGAR string
    cigar_string = read.cigarstring
    # get the aligned positions
    aligned_positions = []
    ref_pos = start
    for i in range(len(cigar_string)):
        if cigar_string[i].isdigit():
            continue
        else:
            length = int(cigar_string[:i])
            if cigar_string[i] == 'M':  # match or mismatch
                aligned_positions.extend(range(ref_pos, ref_pos + length))
                ref_pos += length
            elif cigar_string[i] == 'I':  # insertion
                aligned_positions.extend([None] * length)  # None for insertion
            elif cigar_string[i] == 'D':  # deletion
                ref_pos += length  # skip these positions in the reference
            cigar_string = cigar_string[i + 1:]
            break
    # continue processing the remaining CIGAR string
    while cigar_string:
        for i in range(len(cigar_string)):
            if cigar_string[i].isdigit():
                continue
            else:
                length = int(cigar_string[:i])
                if cigar_string[i] == 'M':  # match or mismatch
                    aligned_positions.extend(range(ref_pos, ref_pos + length))
                    ref_pos += length
                elif cigar_string[i] == 'I':  # insertion
                    aligned_positions.extend([None] * length)  # None for insertion
                elif cigar_string[i] == 'D':  # deletion
                    ref_pos += length  # skip these positions in the reference
                cigar_string = cigar_string[i + 1:]
                break
    # print(f"Read {read.query_name} aligned positions: {aligned_positions}")
    

    return aligned_positions

def get_read_dict(bam, ref_dict, feature_dict):
    read_dict = {}
    with pysam.AlignmentFile(bam, "rb") as bamfile:
        for read in bamfile:
            if read.is_unmapped:
                continue
            read_id = read.query_name
            if read_id in feature_dict:
                feature_seq = feature_dict[read_id]
                ref_seq = ref_dict[read.reference_name].seq.upper()
                read_seq = read.query_sequence.upper()
                # print("ref")
                # print(ref_seq, len(ref_seq))
                # print("read")
                # print(read_seq, len(read_seq))
                # print("feature")
                # print(feature_seq, len(feature_seq))
                aligned_pairs = read.get_aligned_pairs()
                edits = []
                read_string = []
                ref_string = []
                feature_string = []
                # feature seq needs to be lined up with read seq so add ' ' before query_alignment_start and after query_alignment_end
                f_st = ' ' * read.query_alignment_start
                f_end = ' ' * (len(read_seq) - read.query_alignment_end)
                feature_seq = f_st + feature_seq + f_end
                absolute_indices = get_absolute_positions(read)
                for read_pos, ref_pos in aligned_pairs:
                    if ref_pos is not None and read_pos is not None:
                        
                        if ref_seq[ref_pos] == 'A' and read_seq[read_pos] == 'G':
                            edits.append(1)
                            read_string.append(read_seq[read_pos])
                            ref_string.append(ref_seq[ref_pos])
                            try:
                                feature_string.append(feature_seq[read_pos])
                            except IndexError:
                                feature_string.append(' ')
                        else:
                            edits.append(0)
                            read_string.append(read_seq[read_pos])
                            ref_string.append(ref_seq[ref_pos])
                            try:
                                feature_string.append(feature_seq[read_pos])
                            except IndexError:
                                feature_string.append(' ')
                    elif ref_pos is None: # if insertion in read
                        edits.append(2)
                        read_string.append(read_seq[read_pos])
                        ref_string.append(' ')
                        try:
                            feature_string.append(feature_seq[read_pos])
                        except IndexError:
                            feature_string.append(' ')
                    elif read_pos is None: # if deletion in read
                        edits.append(2)
                        read_string.append(' ')
                        ref_string.append(ref_seq[ref_pos])
                        feature_string.append(' ')
                # print("ref")
                # print(''.join(i for i in ref_string))   
                # print("read")
                # print(''.join(i for i in read_string))   
                # print("feature")
                # print(''.join(i for i in feature_string))   
                # print("edits")
                # print(''.join(str(i) for i in edits))
                read_dict[read_id] = {
                    "query_seq": read_seq,
                    "feature_string": feature_seq,
                    "ref_string": ref_seq,
                    "edit_string": edits,
                    "query_seq_aligned": ''.join(i for i in read_string),
                    "ref_seq_aligned": ''.join(i for i in ref_string),
                    "feature_string_aligned": ''.join(i for i in feature_string),
                    "absolute_indices": absolute_indices
                }

    return read_dict


def main(args):
    bam_file = args[0]
    fasta_file = args[1]
    long_jam_file = args[2]
    

    ref_dict = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    feature_dict = extract_features_from_longjam(long_jam_file)
    read_dict = get_read_dict(bam_file, ref_dict, feature_dict)

    # Output the results

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)  