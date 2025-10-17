'''.
This script is for preprocessing reads from a directory of fastq files. It will filter out reads that are too short and 
it will also filter out reads that have a quality score below a certain threshold.
'''

import os
import sys
import argparse
from tqdm import tqdm
import multiprocessing as mp
import shutil
from glob import glob
from FastqReader import FastQreader

class Preprocess:

    def __init__(self, workingdir, outFile, min_length, min_qual):
        self.workingdir = workingdir
        self.ascii_to_quality = {
        '!': 0,
        '"': 1,
        '#': 2,
        '$': 3,
        '%': 4,
        '&': 5,
        "'": 6,
        '(': 7,
        ')': 8,
        '*': 9,
        '+': 10,
        ',': 11,
        '-': 12,
        '.': 13,
        '/': 14,
        '0': 15,
        '1': 16,
        '2': 17,
        '3': 18,
        '4': 19,
        '5': 20,
        '6': 21,
        '7': 22,
        '8': 23,
        '9': 24,
        ':': 25,
        ';': 26,
        '<': 27,
        '=': 28,
        '>': 29,
        '?': 30,
        '@': 31,
        'A': 32,
        'B': 33,
        'C': 34,
        'D': 35,
        'E': 36,
        'F': 37,
        'G': 38,
        'H': 39,
        'I': 40,
        'J': 41,
        'K': 42,
        'L': 43,
        'M': 44,
        'N': 45,
        'O': 46,
        'P': 47,
        'Q': 48,
        'R': 49,
        'S': 50,
        'T': 51,
        'U': 52,
        'V': 53,
        'W': 54,
        'X': 55,
        'Y': 56,
        'Z': 57,
        '[': 58,
        '\\': 59,
        ']': 60,
        '^': 61,
        '_': 62,
        '`': 63,
        'a': 64,
        'b': 65,
        'c': 66,
        'd': 67,
        'e': 68,
        'f': 69,
        'g': 70,
        'h': 71,
        'i': 72,
        'j': 73,
        'k': 74,
        'l': 75,
        'm': 76,
        'n': 77,
        'o': 78,
        'p': 79,
        'q': 80,
        'r': 81,
        's': 82,
        't': 83,
        'u': 84,
        'v': 85,
        'w': 86,
        'x': 87,
        'y': 88,
        'z': 89,
        '{': 90,
        '|': 91,
        '}': 92,
        '~': 93
        }
        for file in os.listdir(workingdir):
            f = os.path.join(workingdir, file)
            # checking if it is a file
            if os.path.isfile(f):
                if f.endswith('.fastq') or f.endswith('.fq'):
                    record = self.filter_reads(f, min_length, min_qual)
                    with open(outFile, 'a') as out:
                        for r in record:
                            out.write(r[0] + '\n' + r[1] + '\n' + r[2] + '\n' + r[3] + '\n')


    def filter_reads(self, fastq_file, min_length, min_qual):

        fastq = FastQreader(fastq_file)
        for record in tqdm(fastq.readFastq()):
            read = record[1].upper()
            Q_string = record[3]
            qScore = self.get_Q(Q_string)
            #print(qScore)
            if len(read) >= min_length and qScore >= min_qual:
                header = record[0]
                read = record[1]
                strand = record[2]
                Q_string = record[3]
                yield record

    def get_Q(self, Q_string):
        scores = []
        for a in Q_string:
            scores.append(int(self.ascii_to_quality[a]))

        return sum(scores)/len(scores)
    
def main():
    parser = argparse.ArgumentParser(usage="fasta or fastq") # , version="%prog 0.1")
    parser.add_argument('-r', '--workingdir', help="directory of fastq files")
    parser.add_argument('-o','--outFile',help='output fastq file')
    parser.add_argument('-l', '--min_length', help='minimum read length')
    parser.add_argument('-q', '--min_qual', help='minimum average quality score')
    
    args = parser.parse_args()

    Preprocess(args.workingdir, args.outFile, int(args.min_length), int(args.min_qual))
if __name__ == "__main__":
    main()








