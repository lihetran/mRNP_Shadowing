"""
Joshua Arribere, April 24, 2026

Script to plot the meta-edit distribution about start/stop codons.

Input: inFile.gtf - gtf-formatted file containing genome annotations.
        Will use this to build a dict of format:
            {strand:{chr:[absIndx:(txtName,relStart,relStop)]}}
            where relStart is the index relative to the start codon,
            and relStop is the index relative to the stop codon.
    N - the minimum number of reads per transcript_id to be included in the analysis.
    inFiles.parquet - a list of parquet files

Output: outPrefix, which will be where a pickled dict is stored, as well as
    a graph saved.

run as python3 metaStartStop.py inFile.gtf outPrefix inFiles.parquet
"""
import collections
import sys, common
from logJosh import Tee
import pandas as pd
from pyx import *

def parseGTF(gtfFile):
    """
    Parse a gtf file to get dict of format:
    {strand:{chr:{absIndx:(txtName,relStart,relStop)]}}}
    where relStart is the index relative to the start codon,
    and relStop is the index relative to the stop codon.
    """
    ##
    print('\nrelStart=0 is the A of the ATG.')
    print('relStop=0 is the T of the TAA/TAG/TGA.\n')
    ##
    gtfDict = {'+': {}, '-': {}}
    ##
    with open(gtfFile, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            chrom, source, feature_type, start, end, score, strand, frame, attributes = fields
            if feature_type != 'CDS':
                continue
            #if the script made it thus far, it's found a line that's a CDS.

            # Extract txtID from attributes
            transcript_id = None
            for attr in attributes.split(';'):
                if attr.strip().startswith('transcript_id'):
                    transcript_id = attr.split('"')[1]
                    break
            
            if transcript_id is None:
                continue
            
            ##now check that transcript_biotype is "protein_coding"
            biotype = None
            for attr in attributes.split(';'):
                if attr.strip().startswith('transcript_biotype'):
                    biotype = attr.split('"')[1]
                    break

            if biotype != "protein_coding":
                continue

            start, end = int(start), int(end)
            #abs_index = (start + end) // 2  # Approximate center of CDS
            
            if chrom not in gtfDict[strand]:
                gtfDict[strand][chrom] = {}
            
            if transcript_id not in gtfDict[strand][chrom]:
                gtfDict[strand][chrom][transcript_id]=[]
            gtfDict[strand][chrom][transcript_id].append((start,end))
    
    ##gtfDict is now of format {strand:{chr:{txtID:[(start,end)]}}}
    ##where there may be multiple (start,end) pairs for a given txtID
    ##if there are multiple exons.
    #print(gtfDict['+'])
    txtCounter=0
    positionCounter=0
    bb={'+':{},'-':{}}
    for strand in gtfDict:
        for chrom in gtfDict[strand]:
            bb[strand][chrom]=collections.defaultdict(list)
            for txtID in gtfDict[strand][chrom]:
                txtCounter+=1
                ##
                if txtCounter%1000==0:
                    print('Placed %d transcripts from gtf file in a genomic position dict.'%(txtCounter))
                ##
                relStart=0
                relStop=-sum([end-start+1 for start,end in gtfDict[strand][chrom][txtID]])
                ##add these two lines to start 100nts upstream of the start codon.
                relStart-=100
                relStop-=100
                #print(relStart,relStop)
                #print(gtfDict[strand][chrom][txtID])
                exons=gtfDict[strand][chrom][txtID]
                exons.sort(key=lambda x:x[0]) #sort by start position, which will ensure that the first exon is the one with the start codon, and the last exon is the
                ##these next lines were for restricting to multi-exon transcripts.
                #if len(exons)==1:
                #    counter-=1
                #    continue
                ##
                if strand=='+':
                    for ii,exon in enumerate(exons):
                        start, end = exon
                        if ii==0:##then it's the first exon
                            start-=100
                        if ii==len(exons)-1:##then it's the last exon
                            end+=100

                        for ii in range(start,end+1):
                            ##
                            bb[strand][chrom][ii-1].append([txtID,relStart,relStop])
                            ##leave for debugging.
                            #if len(exons)==1 and chrom!='Mito':
                            #    print(strand,chrom,ii,txtID,relStart,relStop,len(exons))
                            ##
                            relStart+=1
                            relStop+=1
                            positionCounter+=1
                elif strand=='-':
                    for ii,exon in enumerate(reversed(exons)):
                        start, end = exon
                        if ii==0:
                            end+=100
                        if ii==len(exons)-1:
                            start-=100
                        for ii in range(end,start-1,-1):
                            ##
                            bb[strand][chrom][ii-1].append([txtID,relStart,relStop])
                            ##leave for debugging.
                            #if len(exons)==1 and chrom!='Mito':
                            #    print(strand,chrom,ii,txtID,relStart,relStop,len(exons))
                            ##
                            relStart+=1
                            relStop+=1
                            positionCounter+=1
    ##
    ##print(bb)
    ##bb is now of format {strand:{chr:{absIndx:[(txtID,relStart,relStop)]}}}
    print('\nPlaced %d transcripts from gtf file into genomic position dict.'%(txtCounter))
    print('Placed %s positions.\n'%(positionCounter))

    return bb

def prepData(dataDict):
    """
    dataDict is of format {relPos:freq} where relPos is the position relative to the start/stop codon, and freq is the mean edit frequency at that position.
    Will convert this to a format that's easier to plot. Will be a list of tuples of format (relPos,freq) ordered from most negative relPos to most positive relPos.
    """
    dataList = sorted([(relPos, freq) for relPos, freq in dataDict.items()], key=lambda x: x[0])
    return dataList

def mkPlot(metaDict,outPrefix):
    """
    metaDict is of the format:
    {'starts': {relPos: [editInts]}, 'stops': {relPos: [editInts]}}
    Will plot a graph for each of starts and stops.
    The graph will have relPos on the x-axis, and the mean of editInts on the y-axis.
    Each point on the graph will be the sum of the editInts divided by the number of editInts at that relPos.
    Will also add a plot at the top showing the number of transcripts that contribute to that position.
    """
    ##first conver the input dict to a format that's easier to plot. Will be of format:
    # {start/stop:[relPos:freq]} where freq is the mean of edit ints at that relPos,
    #and relPos is the position relative to the start/stop codon and is ordered from most negative to most positive.
    plotDict=collections.defaultdict(dict)
    plot2Dict=collections.defaultdict(dict)
    for key in metaDict:
        ##
        avgIntCt=[]
        ##
        for relPos in metaDict[key]:
            editInts=metaDict[key][relPos]
            #print(relPos,editInts)
            if len(editInts)>0:
                freq=sum(editInts)/len(editInts)
                plotDict[key][relPos]=freq
                plot2Dict[key][relPos]=len(editInts)
                ##
                if relPos in range(-100,100+1):
                    avgIntCt.append(len(editInts))
        print('\nAverage number of edit ints per position for %s: %f'%(key,sum(avgIntCt)/len(avgIntCt)))
    #print(plotDict)
    ##plotDict is now of format {start/stop:{relPos:freq}}
    ##now we can plot the graph.
    startData=prepData(plotDict['starts'])
    start2Data=prepData(plot2Dict['starts'])
    stopData=prepData(plotDict['stops'])
    stop2Data=prepData(plot2Dict['stops'])
    ##
    start = graph.graphxy(width=8, height=8,
                              x=graph.axis.linear(min=-100, max=100,
                                                      title='Position Relative to Start Codon'),
                              y=graph.axis.linear(min=-0.1,max=1.1,
                                                  title='Average Edit Frequency'))
    start2 = graph.graphxy(width=8,height=2,ypos=start.height+0.5,
                            x=graph.axis.linkedaxis(start.axes["x"]),
                            y=graph.axis.log(title='Position Count'))
    stop = graph.graphxy(width=8, height=8, xpos=start.width * 1.1,
                             x=graph.axis.linear(min=-100, max=100,
                                                     title='Position Relative to Stop Codon'),
                             y=graph.axis.linkedaxis(start.axes["y"]))
    stop2 = graph.graphxy(width=8,height=2,xpos=start.width * 1.1,ypos=start.height+0.5,
                            x=graph.axis.linkedaxis(stop.axes["x"]),
                            y=graph.axis.linkedaxis(start2.axes["y"]))
    ##plot the data
    start.plot(graph.data.points(startData, x=1, y=2),
                   [graph.style.line([common.colors(0)])])
    start2.plot(graph.data.points(start2Data,x=1,y=2),
                    [graph.style.line([common.colors(0)])])
    stop.plot(graph.data.points(stopData, x=1, y=2),
                  [graph.style.line([common.colors(0)])])
    stop2.plot(graph.data.points(stop2Data,x=1,y=2),
                    [graph.style.line([common.colors(0)])])
    
    ##save the graph
    c = canvas.canvas()
    c.insert(start)
    c.insert(start2)
    c.insert(stop)
    c.insert(stop2)
    c.writePDFfile(outPrefix)

def processMeta(metaDict,readCt,N):
    """
    metaDict is of the format:
    {starts/stops:{txtID:{position:[editInts]}}}}
    readCt is of the format:
    {txtID:set(reads)} where reads is a list of reads that all map to txtID and
    survived filters
    Will process metaDict to the format: {start/stop:{relPos:[editInts]}} 
    where relPos is the position relative to the start/stop codon, and editInts 
    is the average of the editInts for transcript_ids that overlap with that position.
    Every transcript_id is weighted the same, as long as there is at least N
    reads per transcript_id
    """
    metaDict2={}
    passed=set()
    for startOrStop in metaDict:
        ##doing this to avoid limitations around pickling lambda
        if startOrStop not in metaDict2:
            metaDict2[startOrStop]=collections.defaultdict(list)
        ##
        for txtID in metaDict[startOrStop]:
            if len(readCt[txtID])>=N:
                passed.add(txtID)
                for pos in metaDict[startOrStop][txtID]:
                    editInts=metaDict[startOrStop][txtID][pos]
                    if len(editInts)>0:
                        avgEdit=sum(editInts)/len(editInts)
                        metaDict2[startOrStop][pos].append(avgEdit)
    ##
    print('\n%s transcripts passed read count cutoffs.'%(len(passed)))
    ##
    return dict(metaDict2)

def main(args):
    gtfFile,N,outPrefix,parquetFiles=args[0],args[1],args[2],args[3:]
    ## parse gtf to get dict of format:
    # {strand:{chr:{absIndx:[(txtName,relStart,relStop)]}}}
    gtfDict=parseGTF(gtfFile)
    ##metaDict will be of the format:
    ##{starts/stops:{txtID:{position:[editInts]}}}}
    metaDict={'starts':collections.defaultdict(lambda:collections.defaultdict(list)),
              'stops':collections.defaultdict(lambda:collections.defaultdict(list))}
    ##To keep track of how many reads per transcript_id:
    readCt=collections.defaultdict(set)
    ##
    for parquetFile in parquetFiles:
        #
        keptReadCount=0
        ntsProcessed=0
        aContainingPosition=0
        #
        print('\nAnalyzing %s...'%(parquetFile))
        ##read the parquet file into a dataframe.
        df=pd.read_parquet(parquetFile)
        ##
        print('Parquet File has %d rows'%(len(df)))
        #df=df[:1000]#useful for subsetting the data during troubleshooting.
        ##now loop through each row in the dataframe, which is a different read.
        for index,row in df.iterrows():
            chrom=row['chrom']
            #absIndx=row['absolute_indices']
            strand=row['gene_strand']
            ##leaving for debugging later
            #if strand=='-' and row['transcript_id'] not in ['YAL049C_mRNA','YAL048C_mRNA']:
            #    print(index)
            #    print(row)
            #else:
            #    continue
            ##
            #if strand=='+':
            #    print(index)
            #    print(row)
            #print(len(row['read_sequence']),len(row['read_sequence_aligned']),len(row['ref_sequence_aligned']))
            ##
            transcript_id=row['transcript_id']
            read_id=row['read_id']
            ##now check if this read overlaps with any of the positions in gtfDict.
            if chrom in gtfDict[strand] and transcript_id and 'RDN' not in transcript_id:
                #
                keptReadCount+=1
                #
                for index2,(refNt,readNt,edit,absIdx) in enumerate(zip(row['ref_sequence_aligned'],row['read_sequence_aligned'],row['edit_string'],row['absolute_indices'])):
                    #
                    ntsProcessed+=1
                    '''
                    if index2>1:
                        if abs(absIdx-row['absolute_indices'][index2-1])>10:
                            ##then we just spanned an intron or another large gap in the ref.
                            print(read_id)
                            sys.exit()
                    '''
                    #
                    hasA=False
                    if strand=='+':
                        if refNt=='A':
                            hasA=True
                    elif strand=='-':
                        if refNt=='T':
                            hasA=True
                    ##
                    #
                    #if strand=='+':
                    #    print(index,index2,chrom,absIdx,refNt,readNt,edit,strand,hasA)
                    ##
                    #if absIdx in gtfDict[strand][chrom]:
                    #    print(index,index2,chrom,absIdx,refNt,readNt,edit,strand,hasA)
                    ##
                    if hasA:
                        absIdx=int(absIdx)
                        edit=int(edit)
                        if edit in [0,1] and absIdx in gtfDict[strand][chrom] and len(gtfDict[strand][chrom][absIdx])==1:##restriction for uniquely assignable
                            #
                            aContainingPosition+=1
                            #
                            for txtID,relStart,relStop in gtfDict[strand][chrom][absIdx]:
                                #print(txtID,relStart,relStop)
                                metaDict['starts'][transcript_id][relStart].append(edit)
                                metaDict['stops'][transcript_id][relStop].append(edit)
                                #
                                readCt[transcript_id].add(read_id)
                                ##
                                '''
                                if relStart in [1,2] and hasA:
                                    print('A nt found in TG of ATG...: %s'%(relStart))
                                    print(row)
                                    print(row['ref_sequence_aligned'])
                                    print(row['read_sequence_aligned'])
                                    sys.exit()
                                '''
                                ##
                                ##
                                '''
                                if relStop==0 and hasA and read_id=='c05248ea-bf81-4e47-9ff1-c6d9c876a763':
                                    print('A found in T of TAA/TAG/TGA...: %s'%(relStop))
                                    print('btw, relStart: %s'%(relStart))
                                    print(refNt,readNt,edit,absIdx)
                                    print(row['read_sequence_aligned'][index2:])
                                    print(row)
                                    print(row['ref_sequence_aligned'])
                                    print(row['read_sequence_aligned'])
                                '''
                                ##
            else:
                if transcript_id and 'RDN' in transcript_id:
                    continue
                elif chrom not in gtfDict[strand]:
                    print('Chromosome %s not found in gtfDict for strand %s...'%(chrom,strand))
        #
        print('%s reads were processed after filters.'%(keptReadCount))
        print('%s nucleotides were processed after filters.'%(ntsProcessed))
        print('%s positions within those nts contained an A.'%(aContainingPosition))
    ##metaDict is now of format {starts/stops:{txtID:{position:[editInts]}}}}
    ##now process metaDict to the format: {start/stop:{relPos:[editInts]}} where relPos is the position relative to the start/stop codon, and editInts is a list of 0/1 integers for reads that overlap with that position.
    metaDictProcessed=processMeta(metaDict,readCt,int(N))
    ##
    '''
    for ii in range(-1,2+2):
        print('\nPosition ii: %d'%(ii))
        print('Number of start edits: %d'%(len(metaDictProcessed['starts'][ii])))
        print('Number of stop edits: %d'%(len(metaDictProcessed['stops'][ii])))
    '''
    ##metaDict is now of format {start/stop:{relPos:[editStrings]}} where relPos is the position relative to the start/stop codon, and editStrings is a list of edit strings for reads that overlap with that position.
    ##now we can plot the meta-edit distribution about start/stop codons.
    #print(metaDict)
    ##metaDict is of the format:
    ##{'starts': {relPos: [editInts]}, 'stops': {relPos: [editInts]}}
    ##where relPos is the position relative to the start/stop codon, and editInts is a
    ##list of 0/1 integers.
    ##
    ##will save as pickled dict
    print('Saving meta-edit distribution about start/stop codons as pickled dict...')
    common.rePickle(metaDictProcessed,outPrefix+'.metaDict.pkl')
    #metaDictProcessed=common.unPickle(outPrefix+'.metaDict.pkl')
    ##will now plot the meta-edit distribution about start/stop codons.
    print('Plotting meta-edit distribution about start/stop codons...')
    mkPlot(metaDictProcessed,outPrefix+'.pdf')

if __name__=='__main__':
    Tee()
    main(sys.argv[1:])