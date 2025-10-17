'''
Script for mutating a percentage or all A's to G's in a fasta or fastq file. This to determine mappability issues
that may arise from treating RNA with the tad8A.20 enzyme. 
'''

from FastaReader import FastAreader
from FastqReader import FastQreader
import pickle
import random
import argparse

class Mutate:
    def __init__(self,header,sequence,k):
        self.header = header
        self.sequence = sequence
        self.k = k
        #self.mutated_locs = {}

    def mutate(self):
        AInd = [i for i,a in enumerate(self.sequence) if a == 'A']
        gInd = random.sample(AInd, round(len(AInd)*self.k))
        new_character = 'G'
        temp = list(self.sequence)
        for i in gInd:
            temp[i] = new_character
        newString = ''.join(temp)
        #self.mutated_locs[self.header] = gInd
        return newString, gInd
    
def main():
    parser = argparse.ArgumentParser(usage="fasta or fastq") # , version="%prog 0.1")
    parser.add_argument('-fa', '--fasta', help="fasta file to be mutated")
    parser.add_argument('-fq','--fastq',help='fastq file to be mutated')
    parser.add_argument('-m','--mutation_frequency', help='mutate percentage of As',
    default='all')
    parser.add_argument('-o', '--outFile', help='output file name')

    args = parser.parse_args()

    mutated_locs = {}
    if args.fasta:
        with open(args.outFile,'w') as o:
            fasta = FastAreader(args.fasta)
            for header,sequence in fasta.readFasta():
                h = header.split()
                print(h[0][1:])
                seq = Mutate(header,sequence.upper(),float(args.mutation_frequency))
                synthetic_seq, mutated_locs[h[0][1:]] = seq.mutate()
                o.write(header+'\n')
                o.write(synthetic_seq+'\n')
        # with open(args.fasta+'.mutated.indices.pickle', 'wb') as p:
        #     pickle.dump(mutated_locs, p, protocol=pickle.HIGHEST_PROTOCOL)

    if args.fastq:
        with open(args.outFile,'w') as o:
            fastq = FastQreader(args.fastq)
            i = []
            j = []
            for record in fastq.readFastq():
                header = record[0]
                h = header.split()[0][1:]
                sequence = record[1].upper()
                i.append(sequence)
                seq = Mutate(header,sequence,float(args.mutation_frequency))
                sep = record[2]
                qScore = record[3]
                print(h)
                synthetic_seq, mutated_locs[h] = seq.mutate()
                j.append(sequence)
                #print(synthetic_seq)
                o.write(header+'\n')
                o.write(synthetic_seq+'\n')
                o.write(sep+'\n')
                o.write(qScore+'\n')
            print(len(i)==len(j))
        # with open(args.fastq+'.mutated.indices.pickle', 'wb') as p:
        #     pickle.dump(mutated_locs, p, protocol=pickle.HIGHEST_PROTOCOL) 
    
               
if __name__ == '__main__':
    main()
