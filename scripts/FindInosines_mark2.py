'''
Script for finding A-I edits in nanopore reads

instead of a 3nt genome mapping approach, this script utilized a modified version of 
minimap2 that doesn't penalize A-G mismatches in finding minimizers.
'''

from FastaReader import FastAreader
from FastqReader import FastQreader
import argparse
import pysam
from tqdm import tqdm



class InosineFinder:
    '''
    Inputs: alignment file in SAM format from minimap2_AG
    Outputs: list of inosine positions in each read
    '''
    def __init__(self):
        
        self.ct = 0
     
    def FindInosines(self,refString, queryString, qStart, aligned_pairs, revStrand):
        '''
        This method looks for A-G mismatches between a reference and query sequence alignment.
        '''
        
        inosinePositions = []
        ct = 0
        for q,r in aligned_pairs:
            if q is not None and r is not None:
                if revStrand == True: #if mapped to reverse strand
                    if queryString[q] == 'G' and refString[r] == 'A':
                        pos = q
                        ct += 1
                        inosinePositions.append(pos)
                    
                elif revStrand == False: #if mapped to forward strand
                    if queryString[q] == 'C' and refString[r] == 'T':
                        pos = q
                        ct += 1
                        inosinePositions.append(pos)

        return inosinePositions, ct
    
    def get_Q(self, Q_string):
        scores = []
        for a in Q_string:
            scores.append(int(self.ascii_to_quality[a]))

        return scores
    
    def error_rate(self, num_aligned_pairs, alignment_length, num_indels):
        '''
        This method calculates the error rate of an alignment.

        error rate = (number of insertions, deletions, mismatches/number of aligned pairs)/alignment length
        '''
        e = (num_indels/num_aligned_pairs)/alignment_length
        return e
        
    
    def score_mods(self, tad_eff, baseQScore, global_error_rate):
        '''
        This method scores the probability of an A-G mismatch being an inosine modification
        '''
        p = 10**(-baseQScore)
        X = 1-p
        prob = (1-global_error_rate) - (tad_eff * X)
        return prob
    
    def plot_mod_coverage(self,bam_file):
        import modbamtools as mbt

        mbt.plot(bam_file)
    
def main():
    parser = argparse.ArgumentParser(description='Find A-I edits in nanopore reads',
                                     add_help=True,
                                     prefix_chars='-')
    parser.add_argument('-a', '--alignmentFile', help="alignment file in SAM format from minimap2_AG")
    parser.add_argument('-g', '--fasta', help="reference in fasta format")
    #parser.add_argument('-q', '--fastq', help="reads in fastq format")
    parser.add_argument('-o', '--outFile', help="output file name")

    args = parser.parse_args()

    fasta = FastAreader(args.fasta)
    refDict = {}
    for header, sequence in fasta.readFasta():
        refDict[header.split()[0][1:]] = sequence.upper()

    # fastq = FastQreader(args.fastq)
    # queryDict = {}
    # numAs = 0
    # for record in fastq.readFastq():
    #     seq = record[1].upper()
    #     queryDict[header.split()[0][1:]] = seq
    #     numAs += seq.count('A')

    alignments = InosineFinder()
    inosinePositions = {}
    numInosines = 0
    numAs = 0

    with pysam.AlignmentFile(args.alignmentFile,'rb') as samFile:
        with pysam.AlignmentFile(args.outFile, "wb", template=samFile) as outFile:
            for read in tqdm(samFile,desc='Finding inosines'):
                if read.is_unmapped == False:
                    
                    refString = refDict[read.reference_name].upper()
                    
                    queryString = read.query_sequence
                    qStart = read.query_alignment_start
                    qEnd = read.query_alignment_end
                    #print(read.get_reference_sequence())
                    numAs += refString[qStart:qEnd].count('A')
                    
                    aligned_pairs = read.get_aligned_pairs()
                    revStrand = read.is_reverse
                    
                    if queryString != None:
                        inosinePositions[read.query_name], count = alignments.FindInosines(refString, queryString, qStart, aligned_pairs, revStrand)
                    numInosines += count
                    inosineTag = 'AtoI'
                    
                    for i in inosinePositions[read.query_name]:
                        inosineTag += ','+str(i)

                    if len(inosinePositions[read.query_name]) > 0:
                        read.set_tag('MM',inosineTag)
                        outFile.write(read)
                    else:
                        outFile.write(read)
                
    
    print('Tad8A.20 Efficency: ', numInosines/numAs*100)
if __name__ == '__main__':
    main()