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
"""
import sys, common, collections, re
from logJosh import Tee

def parseChrFile(chrFile):
    """
    will parse the output of prepareReadAssignments3.py to a dict of format
    {chr:{position:info]}}
    """
    aa=collections.defaultdict(dict)
    with open(chrFile,'r') as f:
        for line in f:
            line=line.strip().split()
            theChr,position=line[0].split('_')
            aa[theChr][int(position)]=line[1:]
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
    starts=[int(entry.split(':')[1]) for entry in string.split('|')]
    stops=[int(entry.split(':')[2]) for entry in string.split('|')]
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
    samFile,chrFile,outPrefix=args[0:]
    ##
    chrDict=parseChrFile(chrFile)
    ##
    with open('%s.longjam'%(outPrefix),'w') as f:
        with open(samFile,'r') as g:
            for line in g:
                if line.startswith('@'):
                    f.write(line)
                else:
                    f.write(line)
                    ##
                    line2=line.strip().split('\t')
                    ##
                    readID=line2[0]
                    strand=int(line2[1])#need to check the modulo 16
                    ##
                    if strand & 16 !=0:
                        strand='-'
                    else:
                        strand='+'
                    ##
                    theChr=line2[2]
                    alignmentStart=int(line2[3])
                    cigar=line2[5]
                    seq=line2[9]
                    #print(readID,theChr,alignmentStart)
                    #printN(seq,10)
                    #print(cigar)
                    ##now get the reference seq positions from the cigar
                    ##and the alignmentStart
                    alignmentPositions=parseCigar(alignmentStart,cigar)
                    #print(alignmentPositions)
                    ##
                    annotations=getAnnotations(alignmentPositions,
                        chrDict[theChr],strand)
                    ##
                    # formatted=formatAnnotations(annotations)
                    # f.write(formatted+'\n')
                    ##
                    formatted2=formatAnnotations2(annotations)
                    #printN(formatted2,10)
                    f.write(formatted2+'\n')

if __name__=='__main__':
    Tee()
    main(sys.argv[1:])
