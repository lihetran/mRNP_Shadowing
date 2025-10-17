"""
Loosely based on the code from the following paper: 

Delahaye, Clara, and Jacques Nicolas. “Sequencing DNA with nanopores: Troubles and biases.” PloS one vol. 16,10 e0257521. 1 Oct. 2021, doi:10.1371/journal.pone.0257521

This script analyzes the substitution errors in a given set of reads for mock and Tad8A.20 treatment. 
It takes in a SAM file and a reference genome, and outputs a plot of the substitution errors.
"""

# --------------------------------------------------------------------------------------------------
# Packages

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

def init_dict_substitutions():
    """
    Initialize a dictionary that will store substitution error occurrences for each species.
    Dictionary of form: dictionary[substitution_type][condition] = substitution_occurrences

    """
    L_bases = ["A", "C", "G", "T"]
    L_substitutions = [base1+base2 for base1 in L_bases for base2 in L_bases if base1!=base2]
    dictionary = {}
    for substitution in L_substitutions:
        dictionary[substitution] = 0 # will store occurrences for each species
    return dictionary

def getSubstitutionErrors(read_seq, ref_seq, cigar, ref_start, ref_end, q_st, q_en):
    import re
    parsed_cigar = re.findall(rf'(\d+)([MDNSIX])', cigar)
    parsed_cigar = [(int(num), char) for num, char in parsed_cigar]
    ref_seq = ref_seq[ref_start: ref_end].upper()
    ref_pos = 0
    read_seq = read_seq[q_st: q_en].upper()
    read_pos = 0

    top_line = ""
    middle_line = ""
    bottom_line = ""

    for length, code in parsed_cigar:
        if code == "M":  # Map (Read & Ref Match)
            read_map_piece = read_seq[read_pos:read_pos + length]
            ref_map_piece = ref_seq[ref_pos:ref_pos + length]
            perfect_matches = ""
            for index, char in enumerate(read_map_piece):
                try:
                    if char == ref_map_piece[index]:
                        perfect_matches += "|"
                    else:
                        perfect_matches += "•"
                except IndexError:
                    perfect_matches += " "
            top_line += read_map_piece
            middle_line += perfect_matches
            bottom_line += ref_map_piece
            ref_pos += length
            read_pos += length

        elif code == "I":  # Insert (Gap in Ref)
            top_line += read_seq[read_pos:read_pos + length]
            middle_line += " " * length
            bottom_line += "-" * length
            read_pos += length

        elif code == "D" or code == "N":  # Delete (Gap in Read)
            top_line += "-" * length
            middle_line += " " * length
            bottom_line += ref_seq[ref_pos:ref_pos + length]
            ref_pos += length
    
    substitutionDict = init_dict_substitutions()
    
    
    for i in range(len(top_line)):
        if bottom_line[i] != top_line[i] and top_line[i] != "-" and bottom_line[i] != "-":    
            substitutionDict[bottom_line[i]+top_line[i]] += 1
            

            
    return substitutionDict
                
            
    
        
