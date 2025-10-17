"""
Joshua Arribere Mar 30, 2023

Script to edit a fasta file and change one or more bases to others.

Input: inFile.fasta
    inSeqs - one or more nts, comma-separated
    outSeqs - one or more nts, comma-separated

Output: outPrefix.fasta - fasta file w/ inSeqs mutated to outSeqs

run as python3 completelyEditFastaFile.py inFile.fastq A,C G,T outPrefix
^This will mutate all As to Gs and all Cs to Ts
"""
import sys, common
#from logJosh import Tee

def main(args):
    inFile,inSeqs,outSeqs,outPrefix=args[0:]
    ##
    inSeqs=inSeqs.split(',')
    outSeqs=outSeqs.split(',')
    mutateDict=dict(zip(inSeqs,outSeqs))
    ##
    with open(inFile,'r') as f:
        with open(outPrefix+'.fa','w') as g:
            for line in f:
                if line.startswith('>'):
                    g.write(line)
                else:
                    for k,v in mutateDict.items():
                        line=line.replace(k,v)
                    g.write(line)

if __name__=='__main__':
    #Tee()
    main(sys.argv[1:])
