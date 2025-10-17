
'''
Script for mutating a percentage or all A's to G's in a fasta or fastq file. This to determine mappability issues
that may arise from treating RNA with the tad8A.20 enzyme. 
'''

from FastaReader import FastAreader
from FastqReader import FastQreader
import random
import argparse
import subprocess
import pysam
import mappy as mp

import os
from tqdm import tqdm
import multiprocessing as mp
import shutil
from glob import glob

from pathlib import Path
from pprint import pprint

from Bio import SeqIO, Seq, SeqRecord
import nanoporePipelineCommon as npC


class Mutate:
    def __init__(self,header,sequence,k):
        self.header = header
        self.sequence = sequence
        self.k = k
        self.mutated_locs = {}
    
    def replaceCharacter(self, seq, char1, char2):
        n = len(seq)
        res = ""
        positions = []
        for i in range(n):
            if seq[i] != char1:
                res += seq[i]    
            else:
                res += char2
                positions.append(i)
        # self.mutated_locs[self.header] = positions
        return res, positions
    
    def reverseComplement(self, seq):

        complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
        reverse_complement = "".join(complement.get(base, base) for base in reversed(seq.upper()))

        return reverse_complement
    
'''
Class for unmutating tad8A.20 reads and identifying A-to-I edits

Inputs: alignment file in SAM format, pickle file with dictionary of computationally mutated sites in reads
Outputs: Bam file with list of inosine positions in each read
'''
 
class InosineFinder:
    '''
    Inputs: alignment file in SAM format, pickle file with dictionary of mutated sites in reads
    Outputs: list of inosine positions in each read
    '''
    def __init__(self):
        
        self.numIs = 0
        self.numPos = 0


    def unmutate_ref(self, header, refDict, start, end):
        '''
        Unmutates a read based on the dictionary of mutated sites in reads
        '''
        refSequence = refDict[header][start:end]
        return refSequence
    
    def unmutate_query(self, header, read, mutDict, start, end):
        '''
        Unmutates a read based on the dictionary of mutated sites in reads
        '''
        unmutated_string = ''
        if read:
            for i in mutDict[header]:
                read = read[:i] + 'A' + read[i+1:]
            unmutated_string = read
            
        return unmutated_string
     
    def FindInosines(self,refString, queryString, qStart, aligned_pairs, strand, qualityString):
        '''
        This method looks for A-G mismatches between a reference and query sequence alignment.
        '''
        #print(len(refString) == len(queryString))
        #scores = self.get_Q(qualityString)
        inosinePositions = []

        for q,r in aligned_pairs:
            if q is not None:
                if strand == True: #if mapped to forward strand
                    if queryString[q] == 'G' and refString[r] == 'A':
                        pos = q + qStart
                        inosinePositions.append(pos)
                        self.numIs += 1
                        self.numPos += 1
                elif strand == False: #if mapped to reverse strand
                    if queryString[q] == 'C' and refString[r] == 'T':
                        pos = q + qStart
                        self.numIs += 1
                        self.numPos += 1
                        inosinePositions.append(pos)

        return inosinePositions
    
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


def main():
    parser = argparse.ArgumentParser(description='Find A-I edits in nanopore reads',
                                     add_help=True,
                                     prefix_chars='-')
    parser.add_argument('-r', '--reads', help="reads in fastq format")
    parser.add_argument('-g', '--fasta', help="reference in fasta format")
    parser.add_argument('-j', '--junction_bed', help="junction bed file")
    parser.add_argument('-o', '--outFilePrefix', help="output file name without extension")

    args = parser.parse_args()

    # Leaf directory
    directory = "tmp"
    
    # Parent Directories
    parent_dir = os.getcwd()
    
    # Path
    path = os.path.join(parent_dir, directory)

    os.makedirs(path)

    #### filter fastq to remove reads < 100 bp and with average quality < 9 ########
    fastq = FastQreader(args.reads)

    #### mutate and reverse complement cDNA reads from A-G ########
    AGmutDict = {}
    with open('tmp/mutated_reads_RC.AtoG.fastq', 'w') as mutated_reads_RC:
        for record in tqdm(fastq.readFastq(), desc='Mutating reads A-G'):
            header = record[0]
            read = record[1].upper()
            strand = record[2]
            qualityString = record[3]

            mutated = Mutate(header.split()[0][1:], read, 1)
            rcRead = mutated.reverseComplement(read)
            newRead, positions = mutated.replaceCharacter(rcRead, 'A', 'G')

            mutated_reads_RC.write(header+'\n')
            mutated_reads_RC.write(newRead+'\n')
            mutated_reads_RC.write(strand+'\n')
            mutated_reads_RC.write(qualityString+'\n')
            AGmutDict[header.split()[0][1:]] = positions

    with open('tmp/RCreads.fastq', 'w') as RCreads:
        for record in fastq.readFastq():
            header = record[0]
            read = record[1].upper()
            strand = record[2]
            qualityString = record[3]

            mutated = Mutate(header.split()[0][1:], read, 1)
            rcRead = mutated.reverseComplement(read)

            RCreads.write(header+'\n')
            RCreads.write(rcRead+'\n')
            RCreads.write(strand+'\n')
            RCreads.write(qualityString[::-1]+'\n')
            
    #### mutate cDNA reads from T-C ########
    TCmutDict = {}
    with open('tmp/mutated_reads.TtoC.fastq', 'w') as mutated_reads:
        for record in tqdm(fastq.readFastq(), desc='Mutating reads T-C'):
            header = record[0]
            read = record[1].upper()
            strand = record[2]
            qualityString = record[3]

            mutated = Mutate(header.split()[0][1:], read, 1)
            newRead, positions = mutated.replaceCharacter(read, 'T', 'C')

            mutated_reads.write(header+'\n')
            mutated_reads.write(newRead+'\n')
            mutated_reads.write(strand+'\n')
            mutated_reads.write(qualityString+'\n')
            TCmutDict[header.split()[0][1:]] = positions


    AtoGfastq = 'tmp/RCreads.fastq'
    AG = FastQreader(AtoGfastq)
    queryDictAG = {}
    for record in AG.readFastq():
        header = record[0]
        queryDictAG[header.split()[0][1:]] = [record[1].upper(),record[3]] 

    TtoCfastq = args.reads
    TC = FastQreader(TtoCfastq)
    queryDictTC = {}
    for record in TC.readFastq():
        header = record[0]
        queryDictTC[header.split()[0][1:]] = [record[1].upper(),record[3]]

    ###### mutate and reverse complement reference from A-G ########
    fasta = FastAreader(args.fasta)
    
    with open('tmp/mutated_reference.AtoG.fasta', 'w') as mutated_reference:
        for header, sequence in tqdm(fasta.readFasta(),desc='Mutating reference A-G'):
            
            mutated = Mutate(header, sequence.upper(), 1)
            newSequence, positions = mutated.replaceCharacter(sequence.upper(), 'A', 'G')

            mutated_reference.write(header+'\n')
            mutated_reference.write(newSequence+'\n')

    ##### mutate reference from T-C ########
    with open('tmp/mutated_reference_RC.TtoC.fasta', 'w') as mutated_reference:
        for header, sequence in tqdm(fasta.readFasta(),desc='Mutating reference T-C'):
            mutated = Mutate(header.split()[0][1:], sequence.upper(), 1)
            rc = mutated.reverseComplement(sequence.upper())
            newSequence, positions = mutated.replaceCharacter(rc, 'T', 'C')

            mutated_reference.write(header+'\n')
            mutated_reference.write(newSequence+'\n')
    
    AGfasta = FastAreader('tmp/mutated_reference.AtoG.fasta')
    refDictAG = {}
    for header, sequence in AGfasta.readFasta():
        refDictAG[header.split()[0][1:]] = sequence.upper()

    TCfasta = FastAreader('tmp/mutated_reference_RC.TtoC.fasta')
    refDictTC = {}
    for header, sequence in TCfasta.readFasta():
        refDictTC[header.split()[0][1:]] = sequence.upper()

    ###### align mutated reads to mutated reference ########
    
    cmd1 = 'minimap2 -ax map-ont -t 20 --junc-bed ' + args.junction_bed + ' --secondary=no --for-only --MD --sam-hit-only tmp/mutated_reference.AtoG.fasta tmp/mutated_reads_RC.AtoG.fastq > tmp/alignments.AtoG.sam'
    cmd2 = 'minimap2 -ax map-ont -t 20 --junc-bed ' + args.junction_bed + ' --secondary=no --for-only --MD --sam-hit-only tmp/mutated_reference_RC.TtoC.fasta tmp/mutated_reads.TtoC.fastq > tmp/alignments.TtoC.sam'
    
    print('Mapping A-G reads to A-G reference')
    subprocess.call(cmd1, shell=True)
    print('Mapping T-C reads to T-C reference')
    subprocess.call(cmd2, shell=True)
    
    ###### find inosines ########
    alignments = InosineFinder()
    inosinePositions = {}
    AtoGbam = args.outFilePrefix+'.AtoG.bam'
    with pysam.AlignmentFile('tmp/alignments.AtoG.sam','rb') as samFile:
        #change to wb if you want to write to bam file
        with pysam.AlignmentFile(AtoGbam, "wb", template=samFile) as outFile:
            for read in tqdm(samFile,desc='Finding inosines'):
                # #inosinePositions = {}
                # print('is read unmapped? ' + str(read.is_unmapped))
                if read.is_unmapped == False:
                    #get matches only
                    aligned_pairs = read.get_aligned_pairs(matches_only=True)
                    #get original sequences
                    query = queryDictAG[read.query_name][0]
                    # qualityString = queryDict[read.query_name][1]
                    ref = refDictAG[read.reference_name]
                    #get alignment start position
                    qStart = read.query_alignment_start
                    qEnd = read.query_alignment_end
                    # print(query)
                    # print(read.query_sequence)

                    print(len(read.query_sequence) == len(queryDictAG[read.query_name][0]))
                    #Find A-G mismatches in orignal sequences
                    inosinePositions[read.query_name] = alignments.FindInosines(ref, query, qStart, aligned_pairs, strand==True, qualityString)
                    #create custom tag for bam file
                    inosineTag = 'A+m'
                    for i in inosinePositions[read.query_name]:
                        tag = i-qStart
                        inosineTag += ','+str(tag)
                    ####write to new bam file#######
                    a = pysam.AlignedSegment()
                    if read.query_name in inosinePositions:
                        a.query_name = read.query_name
                        seq = InosineFinder()
                        # unmutated_query = seq.unmutate_query(read.query_name, read.query_sequence, AGmutDict, read.query_alignment_start, read.query_alignment_end)
                        # a.query_sequence = unmutated_query
                        a.query_sequence = read.query_sequence
                        a.flag = read.flag
                        a.reference_id = read.reference_id
                        a.reference_start = read.reference_start
                        a.mapping_quality = read.mapping_quality
                        a.cigar = read.cigar
                        a.next_reference_id = read.next_reference_id
                        a.next_reference_start= read.next_reference_start
                        a.template_length=read.template_length
                        #a.query_qualities = read.query_qualities
                        a.tags = read.tags
                        a.set_tag('MM',inosineTag)
                        #print(len(read.query_sequence) == len(newQuerySeq))
                        outFile.write(a)
    print(inosinePositions)
    TtoCbam = args.outFilePrefix + '.TtoC.bam'
    with pysam.AlignmentFile('tmp/alignments.TtoC.sam','rb') as samFile:
        #change to wb if you want to write to bam file
        with pysam.AlignmentFile(TtoCbam, "w", template=samFile) as outFile:
            for read in tqdm(samFile,desc='Finding inosines'):
                #inosinePositions = {}
                # print('is read unmapped? ' + str(read.is_unmapped))
                if read.is_unmapped == False:
                    #get matches only
                    aligned_pairs = read.get_aligned_pairs(matches_only=True)
                    #get original sequences
                    query = queryDictTC[read.query_name][0]
                    # qualityString = queryDict[read.query_name][1]
                    ref = refDictTC[read.reference_name]
                    #get alignment start position
                    qStart = read.query_alignment_start
                    #Find T-C mismatches in alignments
                    inosinePositions[read.query_name] = alignments.FindInosines(ref, query, qStart, aligned_pairs, strand==False, qualityString)
                    #create custom tag for bam file
                    inosineTag = 'T+m'
                    for i in inosinePositions[read.query_name]:
                        tag = i-qStart
                        inosineTag += ','+str(tag)
                    ####write to new bam file#######
                    a = pysam.AlignedSegment()
                    if read.query_name in inosinePositions:
                        a.query_name = read.query_name
                        seq = InosineFinder()
                        # unmutated_query = seq.unmutate_query(read.query_name, read.query_sequence, TCmutDict, read.query_alignment_start, read.query_alignment_end)
                        # a.query_sequence = unmutated_query
                        a.query_sequence = read.query_sequence
                        a.flag = read.flag
                        a.reference_id = read.reference_id
                        a.reference_start = read.reference_start
                        a.mapping_quality = read.mapping_quality
                        a.cigar = read.cigar
                        a.next_reference_id = read.next_reference_id
                        a.next_reference_start= read.next_reference_start
                        a.template_length=read.template_length
                        #a.query_qualities = read.query_qualities
                        a.tags = read.tags
                        a.set_tag('MM',inosineTag)
                        #print(len(read.query_sequence) == len(newQuerySeq))
                        outFile.write(a)

    
    subprocess.call('samtools sort ' + AtoGbam + ' -o ' + AtoGbam[:-3] + 'sorted' + '.bam', shell=True)
    subprocess.call('samtools sort ' + TtoCbam + ' -o ' + TtoCbam[:-3] + 'sorted' + '.bam', shell=True) 
    # subprocess.call('samtools merge ' + '-o ' + args.outFilePrefix + '.bam' + ' ' + AtoGbam + '.sorted' + ' ' 
    #                 + TtoCbam + '.sorted', shell=True)
    subprocess.call('samtools index ' + AtoGbam[:-3] + 'sorted.bam', shell=True)
    subprocess.call('samtools index ' + TtoCbam[:-3] + 'sorted.bam', shell=True)  
    
    print('A-G mismatches scored as Inosines: ' + str(alignments.numIs/alignments.numPos))
    shutil.rmtree(path)
       
if __name__ == '__main__':
    main()