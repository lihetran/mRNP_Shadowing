from FastqReader import FastQreader
from FastaReader import FastAreader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import pysam  
import statistics
from cigar import Cigar
import argparse

class KmerScores:

    def __init__(self, file):
        '''
        This class will break down nano3P reads into kmers and assign a quality score to each kmer.
        base off mismatches in the kmer and the quality scores of the bases in the kmer.

        input: list of reads 
        output: dictionary of kmers and their quality scores, arrow format?
        '''
        self.kmerDict = {}
        self.ascii_dict = {}
        for num in range(33, 127):
            self.ascii_dict[chr(num)] = num
        '''
        Read dictionary will be a dictionary of reads and their corresponding kmers and quality scores.
        '''
        #self.readDict = {}
        self.kmerDict = {}
        file = FastQreader(file)
        # for record in file.readFastq():
        #     header = record[0]
        #     read = record[1]
        #     strand = record[2]
        #     quality = record[3]
        #     #kmerDict = {}
        #     for i in range(0, len(read) - k + 1):
        #         kmer = read[i:i+k]
        #         if kmer not in self.kmerDict:
        #             self.kmerDict[kmer] = [(self.kmer_mean(list(quality[i:i+k])), self.kmer_median(list(quality[i:i+k])))]
        #         else:
        #             self.kmerDict[kmer].append([self.kmer_quality(list(quality[i:i+k]))])
        #             self.kmerDict[kmer] = [sum(self.kmerDict[kmer])/len(self.kmerDict[kmer])]
            #self.readDict[header] = kmerDict
    
    def kmer_mean(self, ascii_scores):
        '''
        This function will take in a list ascii scores corresponding to bases in a kmer and return the 
        average quality score of that kmer.
        '''
        scores = []
        for a in ascii_scores:
            scores.append(int(self.ascii_dict[a])-33)

        return sum(scores)/len(scores)
    
    def kmer_median(self, ascii_scores):
        scores = []
        for a in ascii_scores:
            scores.append(int(self.ascii_dict[a])-33)

        return statistics.median(scores)
    
    def alignment_error(self, samFile):
        '''
        This function will take in a kmer and a sam file and return the error rate of that kmer.
        '''
        alignmentDict = {}
        with open(samFile) as f:
            for line in f:
                indelCount = 0
                if not line.startswith('@'):
                    #line = f.readline()
                    line = line.split('\t')
                    readID = line[0]
                    mapq = line[4] #mapping quality
                    #ASscore = line[11] #alignment score
                    cigar = line[5]
                    c = Cigar(cigar)
                    alignmentLength = len(c)
                    qScore = line[10]
                    mean = self.kmer_mean(qScore)
                    median = self.kmer_median(qScore)
                    for char in cigar:
                        if char == 'I' or char == 'D' or char == 'S':
                            indelCount += 1
                    alignmentDict[readID] = [indelCount/alignmentLength, mean, median]
        
        return alignmentDict

                
    
    def plot(self, alignmentDict, outFileName):
        '''
        This function will plot the quality scores of each kmer.
        '''
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        fig.set_dpi(300)

        df = pd.DataFrame.from_dict(alignmentDict, orient='index')
        df = df.reset_index()

        df.columns = ['readID', 'error_rate', 'Mean_Phred', 'Median_Phred']
        df = df.sort_values(by=['error_rate'])

        sns.scatterplot(y = 'error_rate', x = 'Mean_Phred', data=df, color = 'blue', ax=ax)
        sns.scatterplot(y = 'error_rate', x = 'Median_Phred', data=df, color = 'red', ax=ax)

        ax.set_xlabel('Mean and Median Phred Score')
        ax.set_ylabel('Error Rate')

        ax.legend(['Mean', 'Median'])

        plt.savefig(outFileName, dpi=300)

def main():
    parser = argparse.ArgumentParser(description='Find A-I edits in nanopore reads',
                                     add_help=True,
                                     prefix_chars='-')
    parser.add_argument('-a', '--alignmentFile', help="alignment file in SAM format")
    parser.add_argument('-o', '--outFile', help="output file name")

    args = parser.parse_args()

    alignments = KmerScores(args.alignmentFile)

    alignmentDict = alignments.alignment_error(args.alignmentFile)
    alignments.plot(alignmentDict, args.outFile)

if __name__ == '__main__':
    main()

        




    





