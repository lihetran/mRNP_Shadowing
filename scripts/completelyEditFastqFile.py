"""
Joshua Arribere, Mar 30, 2023

Script to completely edit the nts of a fastq file. For example, mutate all
    the As to Gs.

Input: inFile.fastq - fastqFile
    inSeqs - comma-separated list of nts to mutate
    outSeqs - comma-separated list of nts to mutate to

Output: outPrefix.fastq

run as python3 completelyEditFastqFile.py inFile.fastq A,C G,T
^This would mutate all the As to Gs and all the Cs to Ts.
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
    ct=0
    with open(inFile,'r') as f:
        with open(outPrefix+'.fastq','w') as g:
            ##
            for line in f:
                if ct!=1:
                    g.write(line)
                    if ct==3:
                        ct=0
                    else:
                        ct+=1
                else:
                    line=line.strip()
                    for k,v in mutateDict.items():
                        line=line.replace(k,v)
                    g.write(line+'\n')
                    ct+=1

if __name__=='__main__':
    #Tee()
    main(sys.argv[1:])
