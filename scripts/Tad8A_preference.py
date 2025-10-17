from FastaReader import FastAreader
from FastqReader import FastQreader
import pysam
import pandas as pd
import logomaker as lm 
from tqdm import tqdm
import matplotlib.pyplot as plt

from math import log, prod
from random import randint, choices
import random
import math

class RandomizedMotifSearch:
    '''
    Inputs: list of sequences from a fasta file and commandline arguments kmerSize, number of iterations, and pseudocount
    Outputs: A consensuse motif found in common in each sequence. Also outputs an entropy score for the best 
    consensus motif as an indicator of how well the consensus represents the the individual motifs found.
    
    This class generates a consensus motif sequence from a list of sequences taken from a fasta file.
    In this project, we are aiming to uncover a hidden promoter element in each of these sequences 
    that we can target with CRISPR. 
    
    '''

    def __init__(self, DNA, kmerSize, iterations, pseudoCount):
        '''
        This method instantiates global variables needed for methods in this class.
        '''

        self.kmerSize = int(kmerSize)
        self.iterations = int(iterations)
        self.DNA = DNA
        self.p = int(pseudoCount)
        self.gSize = 0 
        for i in self.DNA:
            self.gSize += len(i) #get total size of fasta in bases
        self.bestMotifs = [] # empty list to hold best motifs found
        self.bestScore = -1 #set best score to something negative because best score is going to be the highest score
       
    def randomMotifs(self):
        '''
        This method generates random kmers of length kmerSize. Kmers are then added to a random motif "matrix".
        This matrix is really a list of strings.
        '''
        randomKmersList = [] #List of random kmers
        for seq in self.DNA:
            #get random position in the sequence, subtract kmer size to make sure you don't get a position such that the kmer is less than the given size
            startPos = random.randint(0, len(seq) - self.kmerSize) 
            r = seq[startPos:startPos+self.kmerSize] #get random kmer from sequence
            randomKmersList.append(r) 
        
        return randomKmersList


    def nullProfile(self):
        """
        This method creates a null distribution (Q) representative of the input list of sequences from a 
        fasta file. 
        """
        Q = {'A':0,'T':0, 'G':0, 'C':0}
        pTotal = (self.p*len(self.DNA))
        countA=countT=countG=countC=0
        for i in self.DNA:
            countA += i.count('A') 
            countT += i.count('T')
            countG += i.count('G') 
            countC += i.count('C')
        
        Q['A'] = ((countA+self.p)/(self.gSize+pTotal)) 
        Q['T'] = ((countT+self.p)/(self.gSize+pTotal)) 
        Q['G'] = ((countG+self.p)/(self.gSize+pTotal)) 
        Q['C'] = ((countC+self.p)/(self.gSize+pTotal))
        
        return Q

    def profile(self, motifs):
        '''Calculates the profile from a set of motifs'''
        # Init the profile with pseudocounts
        profile = {"A": [],"G": [],"T": [],"C": []}
        tot = len(motifs)+(4*self.p)
        # Count bases in each position
        for i in range(self.kmerSize):
            for base in profile.keys():
                profile[base].append(self.p)
            for motif in motifs:
                profile[motif[i]][i] += 1
        # Convert counts to probabilities
        for base in profile.keys():
            profile[base] = [count/tot for count in profile[base]]

        return profile 
            
    def score(self, motifProfile):
        '''
        This method computes the entropy score for each probability value in the profile matrix using this formula:
        score = p*math.log2(p/q) where p is each individual probability value and q is the null model. Each
        of these probabililty values is then summed up to get a score for the profile.
        '''
        
        Q = self.nullProfile() #get the null model
        score = 0
        
        for base in motifProfile:
            for prob in motifProfile[base]:
                if prob != 0:
                    score += prob*math.log2(prob/Q[base])
                    
                else:
                    score += 0
        return score

    def nextMotifs(self, profile):
        '''
        This method takes in a profile matrix and generates a new kmer matrix. It compares all kmers in the
        sequences in DNA to the profile matrix by position and computes the probability that base occurs in that 
        position. These probabilites are multiplied together and if the probabiliy is greater than the previous one,
        That kmer is added to a new motif matrix. 
        '''
        
        newMotifs = [None]*len(self.DNA) #empty list for new kmers
        for i in range(0, len(self.DNA)): #iterate through each seq
            bestProb = 0
            seq = self.DNA[i]
            for j in range(0, len(seq) - self.kmerSize+1): 
                kmer = seq[j:j+self.kmerSize] #grab each kmer in sequence
                prob = 1
                for pos in range(0,len(kmer)):
                    base = kmer[pos]
                    prob *= profile[base][pos]
                if prob > bestProb: #if probability of this profile is bigger than the best prob score
                    bestProb = prob #set prob as new best score
                    newMotifs[i] = kmer
       
        return newMotifs
    
    def findMotif(self, gibbs):
        '''
        This method gets the best fit list of motifs for a list of sequences. 
        '''
        if gibbs:
            x = self.gibbsSearch()
            self.bestScore = x[0]
            self.bestProfile = x[1]
            self.bestMotifs = x[2]
        else:
            for i in tqdm(range(0,self.iterations)):
                motifs = self.randomMotifs()
                bestMotifs = motifs
                bestProfile = self.profile(bestMotifs)
                
                while True:
                    nextMotifs = self.nextMotifs(bestProfile)
                    nextProfile = self.profile(nextMotifs)
                    
                    newScore = self.score(nextProfile)
                    bestScore = self.score(bestProfile)
                    
                    if newScore > bestScore:
                        bestProfile = nextProfile
                        bestMotifs = nextMotifs
                        if newScore > self.bestScore:
                            self.bestScore = newScore
                            self.bestMotifs = bestMotifs
                        
                    else:
                        break

            self.bestProfile = bestProfile

            
    def getConsensus(self):
        '''
        This method gets a consensus sequence from the best motif profile found in the method above. Looks at the highest
        probability in each column of profile matrix and gets the corresponding base. Returns a consensus sequence.
        '''
        #self.findMotif() #generate best motif matrix
        return "".join([max(self.bestProfile.keys(), key=lambda k: self.bestProfile[k][i]) for i in range(self.kmerSize)])
    
    def gibbsSearch(self):
        # Initialize with random motifs
        motifs = self.randomMotifs()
        score = self.score(self.profile(motifs))
        for i in tqdm(range(self.iterations)):
            # Remove a random kmer and generate a profile
            profile = self.profile([motif for motif in motifs if motif!=motifs[randint(0, len(motifs)-1)]])
            # Get a new set of kmers from this profile
            newMotifs = [self.getProbMotif(seq, profile) for seq in self.DNA]
            newScore = self.score(self.profile(newMotifs))
            # Update the existing motifs if this set is better
            if newScore > score:
                score = newScore
                motifs = newMotifs

        return (score, self.profile(motifs), motifs)
    
    def getProbMotif(self, seq, profile):
        '''Chooses a motif from a sequence using kmer weights generated from a profile'''
        kmers = []
        probs = []
        for i in range(len(seq)-self.kmerSize+1):
            probs.append(prod([profile[seq[i+j]][j] for j in range(self.kmerSize)]))
            kmers.append(seq[i:i+self.kmerSize])
        return(choices(kmers, weights=probs)[0])
    
    def plot_logo(self, profile):
        '''
        This method plots a sequence logo from a profile matrix.
        '''
        df = pd.DataFrame.from_dict(profile)
        df.index.name = 'pos'
        #df = df.transpose()
        
        fig, ax = plt.subplots(figsize=(10,4))
        lm.Logo(df, ax=ax)
        ax.set_ylabel('bits')
        ax.set_xlabel('position')
        plt.savefig('logo.png', dpi=300)
                

class CommandLine():
    '''
    Handle the command line, usage and help requests.

    CommandLine uses argparse, now standard in 2.7 and beyond. 
    it implements a standard command line argument parser with various argument options,
    a standard usage and help, and an error termination mechanism do-usage_and_die.

    attributes:
    all arguments received from the commandline using .add_argument will be
    avalable within the .args attribute of object instantiated from CommandLine.
    For example, if myCommandLine is an object of the class, and requiredbool was
    set as an option using add_argument, then myCommandLine.args.requiredbool will
    name that option.

    '''

    def __init__(self, inOpts=None):
        '''
        CommandLine constructor.
        Implements a parser to interpret the command line argv string using argparse.
        '''

        import argparse
        self.parser = argparse.ArgumentParser(
            description='Program prolog - a brief description of what this thing does',
            epilog='Program epilog - some other stuff you feel compelled to say',
            add_help=True,  # default is True
            prefix_chars='-',
            usage='%(prog)s [options] -option1[default] <input >output'
        )

        self.parser.add_argument('-i', '--iterations', nargs='?', default=1000, action='store',
                                 help='Number of iterations ')
        self.parser.add_argument('-k', '--motifLength', nargs='?', default=8, action='store',
                                 help='kMer Size ')
        self.parser.add_argument('-p', '--pseudoCount', nargs='?', type=float, default=.01, action='store',
                                 help='pseudoCount')
        self.parser.add_argument('-g', '--gibbs', action='store_true',
                                help='Use Gibbs algorithm')
        self.parser.add_argument('-m', '--printMotif', action='store_true',
                                help='Print the motifs and contributing sequence name')


        self.parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
        if inOpts is None:
            self.args = self.parser.parse_args()
        else:
            self.args = self.parser.parse_args(inOpts)


def FindInosines(refString, queryString, qStart, aligned_pairs):
    '''
    This method looks for A-G mismatches between a reference and query sequence alignment.
    '''
    #print(len(refString) == len(queryString))
    inosinePositions = []
    for q,r in aligned_pairs:
        if q is not None and r is not None:
            
            if queryString[q] == 'G' and refString[r] == 'A':
                pos = q
                inosinePositions.append(pos)
            

    return inosinePositions



def main(inFile = '', fastq = '', fastaFile= '', options = None):
    print(inFile)
    cl = CommandLine(options)
    #load fastq file
    fastq = FastQreader(fastq)
    queryDict = {}
    for record in tqdm(fastq.readFastq(), desc='Loading Fastq File'):
        header = record[0]
        #print(header.split()[0][1:])
        queryDict[header.split()[0][1:]] = record[1].upper()

    #load fasta file
    fasta = FastAreader(fastaFile)
    refDict = {}
    for header, sequence in tqdm(fasta.readFasta(), desc='Loading Fasta File'):
        refDict[header.split()[0][1:]] = sequence.upper()
    #get inosine positions by finding A-G mismatches on forward mapped reads
    inosinePositions = {}

    with pysam.AlignmentFile(inFile,'rb') as samFile:
        for read in tqdm(samFile, desc='Getting Inosine Positions'):
            if not read.is_unmapped:
                #print(read.query_name)
                #get matches only
                aligned_pairs = read.get_aligned_pairs(matches_only=True)
                #get original sequences
                query = queryDict[read.query_name]
                ref = refDict[read.reference_name]
                #get alignment start position
                qStart = read.query_alignment_start
                #get strand info  
                if not read.is_reverse:
                    inosinePositions[read.query_name] = FindInosines(ref, query, qStart, aligned_pairs)
    #get 15mers around inosine positions
    DNA = []
    for read in tqdm(inosinePositions, desc='Getting Inosine kmers'):
        for inosine in inosinePositions[read]:
            if int(inosine)-7 > 0 and int(inosine)+8 < len(queryDict[read]):
                #print(queryDict[read][int(inosine)-15:int(inosine)+15])
                DNA.append(queryDict[read][int(inosine)-7:int(inosine)+8])
    #print(len(DNA))
    DNA_sample = random.sample(DNA, 100)
    #print(DNA_sample)
            
            
    #Find consensus 6mer motif and profile    
    motifFinder = RandomizedMotifSearch(DNA, cl.args.motifLength, cl.args.iterations, cl.args.pseudoCount)
    if cl.args.gibbs:
        motifFinder.findMotif(True)
    else:
        motifFinder.findMotif(False)
    bestScore = motifFinder.bestScore
    bestProfile = motifFinder.bestProfile
    consensus = motifFinder.getConsensus()
         
    print('Consensus sequence: ' + consensus)
    print('Best Score: ' + str(bestScore))
    
    motifFinder.plot_logo(bestProfile)

if __name__ == "__main__":
    main(inFile="/data16/marcus/working/230613_nanoporeRun_sMV025-RNAStds_50-50_LT_TAD_Nano3P/output_dir/cat_files/cat.sorted.mappedAndPrimary.bam", 
         fastq="/data16/marcus/working/230613_nanoporeRun_sMV025-RNAStds_50-50_LT_TAD_Nano3P/output_dir/cat_files/cat.fastq", 
         fastaFile="/data16/marcus/genomes/plus_cerENO2_elegansRelease100/230327_allChrs_plus-cerENO2.allChrs.fa",
         options=["--iterations=1000" , "--motifLength=6", "--pseudoCount=1"])