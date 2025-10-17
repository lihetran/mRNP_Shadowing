"""

################## Original Documentation ##################
Joshua Arribere, March 6, 2014
Updated to python3, revamped output so that it only writes lines for chromosomal positions
    that contain a txt.

Read assignment with assignReadsToGenes.py is taking an interminable amount of time. Part
    of this is because for each read, I calculate whether a gene overlaps with it, whether
    one and only one gene overlaps with it, and then the position of that read in all
    transcripts for that gene. This is a lot of calculations, for each read.
    
    An alternative that I'm going to try out here, is to calculate this for every possible
    position in the genome. Then when I have my reads, loop through them and simply
    perform a look up in that data structure.
    
    This script will create that data structure. It will, for every position in a provided
    genome annotation, return all positions which overlap with exactly one gene. For each
    such position, it will provide the position relative to annotated start/stop codons
    for each transcript overlapping that position.

Input: annots.gtf - gtf-formatted annotations. Will also get output name from this

Output: a txt file where each line contains a position on a chr, and the txts at that site
    as:
        position \t gene:+ \t txt1:posRelStart:posRelStop|txt2:posRelStart:posRelStop|...

run as python prepareReadAssignmentFile.py annots.gtf

################## Liam's Documentation ##################
Liam Tran, September 1, 2023
Updating this to include gene type as to assign rRNA, protein-coding, etc to long read data downstream.
"""
import sys, common, time, collections, csv, subprocess, pickle
from logJosh import Tee
import multiprocessing

def getDist(exon_list,position):
    """Will return distance from the first position of the first exon to position, in mRNA space"""
    dist=0
    for exon in exon_list:
        if position in range(exon[0],exon[1]+1):
            return dist+position-exon[0]
        else:
            dist+=exon[1]-exon[0]+1#Edit July 28, 2014 added +1.
    print(exon_list, position, dist)
    print('Error: position not found in txt'), sys.exit()

def parseAnnots(annots):#borrowed and adapted from parseAnnots
    """Will output readPositions, a dictionary of {chr:position:{}} where the innermost dictionary
    contains information about all genes/transcripts overlapping with that position."""
    
    ########the below lines are just to estimate time the script will take...
    t=time.time()
    ct=0.
    p=subprocess.Popen(['wc','-l',annots],shell=False,stdout=subprocess.PIPE)
    aa=p.stdout.read()
    aa=aa.split()
    total=int(aa[0])
    ########the above lines are just to estimate time the script will take...
    # dictionary to hold transcripts with CDS annotations
    a={} 
    l = {}
    readPositions=collections.defaultdict(lambda:collections.defaultdict(dict))
    with open(annots,'r') as f:
        freader=csv.reader(f,delimiter='\t')
        for row in freader:
            # if row contains trancript information
            if len(row)>1: 
                ct+=1
                featureType=row[2]
                if featureType == 'exon':# and row[1]=='protein_coding':
                    Chr=row[0]
                    strand=row[6]
                    gene_id=row[8].split('"')[1]
                    geneType=row[8].split('gene_biotype')[1].split('"')[1]
                    transcript_id = row[8].split('transcript_id')[1].split('"')[1]
                    for ii in range(int(row[3]),int(row[4])+1):#edit July 29, 2014 josh added +1 to include right exon boundary
                        if gene_id not in readPositions[Chr][ii]:
                            readPositions[Chr][ii][gene_id]={'strand':strand,'transcript_id':[],'gene_type':geneType}
                        readPositions[Chr][ii][gene_id]['transcript_id'].append(transcript_id)
                
                if featureType in ['exon','CDS']:# and row[1]=='protein_coding':
                    transcript_id = row[8].split('transcript_id')[1].split('"')[1]
                    if transcript_id not in a:
                        a[transcript_id]={'strand':row[6],'exon':[],'CDS':[]}
                    a[transcript_id][featureType].append([int(row[3]),int(row[4])])
    
    b=dict((txt,a[txt]) for txt in a if a[txt]['CDS']!=[])
    
    #print len(b), ' number of annotated CDSs'
    print('Done parsing annotations...')
    return b,readPositions

def getTxtRelPositions(transcript_id,txt_annot,readPosition):#modified so that it returns strand +/- instead of S/AS
    """Given txt_annot={strand,exon,CDS},readStrand,readPosiiton, will return a colon-separated string of
    transcript_id, readPosition_rel_to_CDSstart, readPosition_rel_to_CDSEnd, S/AS where S/AS indicates whether
    the read is on the same strand or not"""
    #Great, now figure out position relative to CDSStart
    #To do this, I will calculate the distance from the txtStart to readPosition, and txtStart to CDSStart. I'll subtract the two.
    cdsStart=min([entry[0] for entry in txt_annot['CDS']])
    exons=txt_annot['exon']
    exonStarts=[exon[0] for exon in exons]
    exonEnds=[exon[1] for exon in exons]
    exonStarts.sort(),exonEnds.sort()
    #the zip function works differently in python2 and python3.
    #So changed the next line of code to work via list comprehension
    #https://stackoverflow.com/questions/31683959/the-zip-function-in-python-3
    #exons=zip(exonStarts,exonEnds)
    exons=[(exonStarts[ii],exonEnds[ii]) for ii in range(len(exonStarts))]
    #Edit: Apparently the exons are not necessarily orderd in the gtf file.
    
    txtStart_cdsStart_dist=getDist(exons,cdsStart)
    txtStart_readPosition_dist=getDist(exons,readPosition)
    readPosition_rel_to_CDSstart=txtStart_readPosition_dist-txtStart_cdsStart_dist
    
    #now do the same thing with the cdsEnd
    cdsEnd=max([entry[1] for entry in txt_annot['CDS']])
    txtStart_cdsEnd_dist=getDist(exons,cdsEnd)
    #already determined txtStart_readPosition_dist
    readPosition_rel_to_CDSend=txtStart_readPosition_dist-txtStart_cdsEnd_dist
    
    #stranded issues. Although ensembl defines start_codon and stop_codon as those exact locations on the - and + strand, here I find the start/stop by taking min/max of CDS exon boundaries, so I need to flip it
    if txt_annot['strand']=='+':
        return ':'.join([transcript_id,str(readPosition_rel_to_CDSstart),str(readPosition_rel_to_CDSend)])
    else:
        return ':'.join([transcript_id,str(-readPosition_rel_to_CDSend),str(-readPosition_rel_to_CDSstart)])
    
def getTxtRelPositions2(transcript_id,txt_annot,readPosition):#modified so that it returns strand +/- instead of S/AS
    pass

def writeOutput(someTuple):
    """
    Will write the output file. For every position from 1 to the largest key in dict1,
    will write a line. If the key is in dict1, will tab, then write the value.
    """
    Chr,dict1,annotFileName=someTuple[0],someTuple[1],someTuple[2]
    ct=0
    with open('.'.join(annotFileName.split('.')[:-1])+'.%s.txt'%Chr,'w') as f:
        for ii in range(1,max(dict1)+1):
            if ii not in dict1:
                #f.write('\n')
                pass
            else:
                if ct==0:
                    f.write('%s|'%ii)
                    ct+=1
                elif ct!=0:
                    f.write('\n%s|'%ii)
                for entry in dict1[ii]:
                    f.write('%s\t'%entry)

def writeOutput2(someDict,annotFileName):
    """
    Will write the output file. For every position from 1 to the largest key in dict1,
    will write a line. If the key is in dict1, will tab, then write the value.
    """
    ct=0
    with open('.'.join(annotFileName.split('.')[:-1])+'.allChrs.txt','w') as f:
        for Chr in someDict:
            dict1=someDict[Chr]
            try:
                for ii in range(1,max(dict1)+1):
                    if ii not in dict1:
                        #f.write('\n')
                        pass
                    else:
                        if ct==0:
                            f.write('%s_%s\t'%(Chr,ii))
                            ct+=1
                        elif ct!=0:
                            f.write('\n%s_%s\t'%(Chr,ii))
                        temp='\t'.join([dict1[ii][0],'|'.join(dict1[ii][1:])])
                        f.write('%s'%temp)
            except ValueError:
                print(dict1)
                pass

def positionReadsOnTxts(annots,readPositions,annotFileName):
    """Given annots={txt_ID:{strand,CDS,exon}}, readPositions={chr:position:gene_id:{strand,txt_ids}},
    will make a file with
    chr position strand read_seq Aligned_length gene_id txt_id:relToCDSStart:relToCDSEnd:S/AS
    where the last column has potentially multiple comma-separated entries"""
    
    #print len(annots), ' number of annotated txts.'
    #print sum([len(readPositions[Chr]) for Chr in readPositions]), ' number of reads'
    
    outputPositions={}
    print('Starting positioning of theoretical reads on txts...')
    
    for Chr in readPositions:
        outputPositions[Chr]=collections.defaultdict(list)
        
        for position in readPositions[Chr]:
            if readPositions[Chr][position]=={}:#then the read never mapped within to an annotated feature, Should not exist in this version.
                pass#for now I'll ignore these reads
            elif len(readPositions[Chr][position])==1 and readPositions[Chr][position]!={}:#omiting reads that are ambiguously assignable to genes b/c genes overlap
                #the next line is a python2/3 issue. This line contains python2 code, and
                #below it is the python3 code
                #gene_id=readPositions[Chr][position].keys()[0]
                gene_id=list(readPositions[Chr][position])[0]
                positionInfo=[]
                for transcript_id in readPositions[Chr][position][gene_id]['transcript_id']:
                    if transcript_id in annots:
                        if annots[transcript_id]['CDS']!=[]:
                            positionInfo+=[getTxtRelPositions(transcript_id,annots[transcript_id],position)]
                            txtStrand=annots[transcript_id]['strand']

                    else:
                        positionInfo+=[transcript_id+':NA:NA']
                        txtStrand=readPositions[Chr][position][gene_id]['strand']

                            
        
                if len(positionInfo)==0:#then it's not assignable to a single txt--maybe b/c no txts are annotated for that gene, or b/c they all lack a CDS annotation
                    # pass#don't care about it if there's no CDS annotated for the txts
                    txtStrand=readPositions[Chr][position][gene_id]['strand']
                    positionInfo=[gene_id+':'+readPositions[Chr][position][gene_id]['strand']+':'+readPositions[Chr][position][gene_id]['gene_type']]
                    outputPositions[Chr][position]=positionInfo
                else:#then we've got a gene and at least one transcript
                    #positionInfo.append(gene_id+':'+txtStrand)
                    positionInfo.insert(0,gene_id+':'+txtStrand+':'+readPositions[Chr][position][gene_id]['gene_type'])
                    outputPositions[Chr][position]=positionInfo
            elif len(readPositions[Chr][position])>1:#then it's multiply mapping
                pass
    
    print('Done positioning theoretical reads on txts and now pickling output...')
    #in python2 > python3 'w' must be 'wb'
    with open('.'.join(annotFileName.split('.')[:-1])+'.processed.p','wb') as f:
        pickle.dump(outputPositions,f)
    
    
    #If you already have a processed.p file. comment out 137 to 157, and uncomment 160 to 162.
    
    print('Unpickling...')
    tempFile='.'.join(annotFileName.split('.')[:-1])+'.processed.p'
    with open(tempFile,'rb') as f:
        outputPositions=pickle.load(f)
    print('Done unpickling!')
    
    ##now output .txt file
    #p=multiprocessing.Pool(len(outputPositions)) #do not do multiprocessing because it will max out memory
    #p.map(writeOutput,[(Chr,outputPositions[Chr],annotFileName) for Chr in outputPositions])
    """
    for Chr in outputPositions:
        if len(outputPositions[Chr])>0:
            print('Working on %s...'%(Chr))
            writeOutput((Chr,outputPositions[Chr],annotFileName))
            print('Finished %s.'%(Chr))
    """
    #added this to see if I can later process everything in one pd DataFrame
    print('Working on all chromsome-file...')
    print(outputPositions)
    writeOutput2(outputPositions,annotFileName)

def main(args):
    annotFileName=args[0]
    
    annots,readPositions=parseAnnots(annotFileName)  
    # print(readPositions)
    print(annots)
    #annots='blag'
    #readPositions='abas'
    
    positionReadsOnTxts(annots,readPositions,annotFileName)



if __name__=='__main__':
    Tee()
    main(sys.argv[1:])
