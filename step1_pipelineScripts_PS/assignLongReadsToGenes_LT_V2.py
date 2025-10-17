"""
Joshua Arribere, Aug 2, 2023

Script to deal with the assignment of long reads and their positions to
    txts/genes and CDS locations therein. The main goal of this script is
    to solve the information-parsing problem, and reduce subsequent coding.
    The goal is therefore to keep as much info as is relevant to the
    assignment problem. This script will output that. It will also make
    other, more compact formats designed for easy downstream applications,
    but that make some assumptions that may not be appropriate in every
    use case (YMMV).

Input: inFile.sam - reads mapped in sam format.
    inFile.allChrs.txt - output of prepareReadAssignment3.py

Output: outFile.longjam of the format
    first row: read info from sam file
    second row: txt information, a list of all entries where the position
        of that read overlaps w/ a gene/txt, and the associated txt info.

run as python3 assignLongReadsToGenes.py inFile.sam inFile.allChrs.txt 
    outPrefix

Liam Tran, Sep 20, 2023
    - added geneType to output file
    - modified prepareReadAssignment3.py to include geneType
    - added FindInosines function to find A-G mismatches in alignment
    - added binary string to output file to represent inosine positions

    run as python3 assignLongReadsToGenes_LT_V2.py inFile.sam inFile.allChrs.txt genome.fa outPrefix
"""
import sys, common, collections, re
from logJosh import Tee
import pysam
from Bio import SeqIO, Seq, SeqRecord

def parseChrFile(chrFile):
    """
    will parse the output of prepareReadAssignments3.py to a dict of format
    {chr:{position:info]}}
    """
    aa=collections.defaultdict(dict)
    with open(chrFile,'r') as f:
        for line in f:
            line=line.strip().split()
            theChr,position=line[0].split('_') # ex. chrI_123
            aa[theChr][int(position)]=line[1:] # ex. F40D4.17.1:289:113
    return dict(aa)

def parseCigar(alignmentStart,cigar):
    """
    Given an alignmentStart and a cigar, will determine all positions that
    the alignment covers in genomic space.
    """
    ##
    aa=[]
    ##
    parsed_cigar = re.findall(rf'(\d+)([MDNSIX])', cigar)
    parsed_cigar = [(int(num), char) for num, char in parsed_cigar]
    ##
    ref_pos=0
    read_pos=0
    ##
    for length,code in parsed_cigar:
        if code == "M":
            for entry in range(alignmentStart,alignmentStart+length):
                aa.append(entry)
            alignmentStart+=length
        elif code=="I":#gap in read, next X bases missing from genome
            for entry in range(length):
                aa.append(0)
        elif code in ["D","N"]:#gap in genome, skip X bases before next aligned nt
            alignmentStart+=length
    ##
    return aa

def FindInosines(read_seq, ref_seq, strand, cigar, ref_start, ref_end, q_st, q_en):
    '''
    This method will find edits based on A-G mismatches in the alignment
    '''
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
            bottom_line += " " * length
            read_pos += length

        elif code == "D" or code == "N":  # Delete (Gap in Read)
            top_line += " " * length
            middle_line += " " * length
            bottom_line += ref_seq[ref_pos:ref_pos + length]
            ref_pos += length

    inosinePos = []
    binary = [] #initialize binary array of 0s and 1s to represent inosine positions. 1's represent inosine positions, 0's represent non-inosine positions
    
    if strand == '+': #reads are reverse complemented and mutated from A-G
        for i in range(len(top_line)):
            if top_line[i] == 'G' and bottom_line[i] == 'A':
                inosinePos.append(i)
                binary.append(1)
            elif bottom_line[i] == ' ':
                binary.append(2)
            else:
                binary.append(0)
                
    elif strand == '-': #reads are not reverse complemented and mutated from C-T
        for i in range(len(top_line)):
            if top_line[i] == 'C' and bottom_line[i] == 'T':
                inosinePos.append(i)
                binary.append(1)
            elif bottom_line[i] == ' ':
                binary.append(2)
            else:
                binary.append(0)

    return binary

def printN(seq,N):
    aa=''
    for ii in range(0,len(seq),N):
        aa+=seq[ii:ii+N]
        aa+=' '
    print(aa)

def getAnnotations(alignmentPositions,innerDict,strand):
    aa=[]
    
    for entry in alignmentPositions:
        
        if entry==0:
            aa.append('na')#no align
        elif entry in innerDict:
            if innerDict[entry][0].split(':')[1]==strand:
                aa.append(innerDict[entry])
            else:
                aa.append('ng')
        else:
            aa.append('ng')#no gene
    return aa

def formatAnnotations(annotations):
    aa=[]
    for entry in annotations:
        if type(entry)==list:
            aa.append(','.join(entry))
        elif type(entry)==str:
            aa.append(entry)
        else:
            print('Here be dragons...')
            sys.exit()
    return '\t'.join(aa)

def getChar(string):
    starts=[entry.split(':')[1] for entry in string.split('|')]
    stops=[entry.split(':')[2] for entry in string.split('|')]
    if 'NA' not in starts and 'NA' not in stops:
        starts=[int(entry) for entry in starts]
        stops=[int(entry) for entry in stops]
        ##
        inUTR=0
        inCDS=0
        for entry in zip(starts,stops):
            if entry[0]>=0 and entry[1]<=0:
                inCDS+=1
            else:
                inUTR+=1
        ##
        if inCDS*inUTR!=0:
            return 'd'#disagreement
        else:
            starts=list(set(starts))
            if len(starts)==1:
                if starts[0]==0:
                    return 'S'
            stops=list(set(stops))
            if len(stops)==1:
                if stops[0]==1:
                    return 'T'
            ##
            if inCDS>=1 and inUTR==0:
                return 'C'
            elif inCDS==0 and inUTR>=1:
                return 'U'
            print('You made a logical mistake.')
            print(string,inCDS,inUTR)
            sys.exit()
    else:
        return 'x' # no cds annotation

def formatAnnotations2(annotations):
    aa=[]
    for entry in annotations:
        if type(entry)==list:
            theChar=getChar(entry[1])
            aa.append(theChar)
        elif type(entry)==str:
            if entry=='na':
                aa.append('a')
            elif entry=='ng':
                aa.append('g')
            else:
                print('AYYYEEEEEE!!!!')
                sys.exit()
        else:
            print('Here be dragons...')
            sys.exit()
    return ''.join(aa)

def main(args):
    samFile,chrFile,genomeFile,outPrefix=args[0:]
    ##
    chrDict=parseChrFile(chrFile)
    ##
    refDict = {}
    for record in SeqIO.parse(genomeFile, 'fasta'):
        name, sequence = record.id, str(record.seq)
        refDict[name] = sequence

    

    with open('%s.sham'%(outPrefix),'w') as f:
        with open(samFile,'r') as g:
            for line in g:
                if line.startswith('@'):
                    f.write(line)

        with pysam.AlignmentFile(samFile, 'r') as s:
            for read in s:
                if read.is_unmapped:
                    continue
                else:
                    # get strand info
                    if read.is_reverse:
                        strand = '-'
                    else:
                        strand = '+'
                    # write original alignment info
                    r = str(read).split('\t')
                    r[2] = str(read.reference_name)
                    f.write('\t'.join(x for x in r)+'\n')
                    alignmentPositions=parseCigar(read.reference_start, read.cigarstring)
                    annotations=getAnnotations(alignmentPositions, chrDict[read.reference_name], strand)
                    formatted2=formatAnnotations2(annotations)
                    # write coding region string
                    f.write(formatted2+'\n')
                    # write gene type
                    try:
                        geneInfo=chrDict[read.reference_name][read.reference_start]
                        geneType = geneInfo[0].split(':')[2]
                        f.write(geneType+'\n')
                    except:
                        geneType = 'NG'
                        f.write(geneType+'\n')
                    # write edit string
                    binary = FindInosines(read.query_sequence, refDict[read.reference_name], strand, read.cigarstring, read.reference_start, read.reference_end, read.query_alignment_start, read.query_alignment_end)
                    f.write(''.join(str(x) for x in binary)+'\n')

                    # write reference sequence, it's difficult to rebuild the original reference sequence from the cigar string because of 3nt mapping
                    # might as well just write in into the sham file
                    # f.write(refDict[read.reference_name][read.reference_start:read.reference_end]+'\n')
                    
                    # write alignment coordinates
                    f.write(str(read.reference_start)+'\t'+str(read.reference_end)+'\t'+str(read.query_alignment_start)+'\t'+str(read.query_alignment_end)+'\n')
                
                    
if __name__=='__main__':
    Tee()
    main(sys.argv[1:])
