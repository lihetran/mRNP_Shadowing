"""
October 20, 2025 - Liam Tran

Script to perform Multiple Correspondence Analysis on an oligo shadowing
    expt.

Input: pickle2 - a pickled file from Liam
    numReads - will pick this many reads from each bc. Rec: 50.
        Will select longer reads first.
    minEdit - minimum edit frequency, e.g., 0.7 for 70%

Output: plot of first two components from PCA, colored by bc of the
    library they come from.

run as python3 pcaPrinceOligoShadowing.py pickle2 numReads minEdit 
    outPrefix
"""
import sys, common, pickle, collections, numpy
import pandas as pd
import matplotlib.pyplot as plt
from logJosh import Tee
import prince

def parsePickleFile(pickleFile):
    """
    Given Liam's pickle file, will output a dict of the format:
    {bc:{readID:{position:edit}}} where position has to be an A in the ref
    """
    with open(pickleFile,'rb') as f:
        dataDict=pickle.load(f)
    ##
    bb=collections.defaultdict(lambda:collections.defaultdict(dict))      
    ##
    for readID,subDict in dataDict.items():
        barCode=subDict['barcode']
        #print(barCode,readID)
        editString=subDict['edit_string']
        #print(editString,len(editString))
        readSeq=subDict['read_sequence_aligned']
        #print(readSeq,len(readSeq))
        refSeq=subDict['ref_sequence_aligned']
        #print(refSeq,len(refSeq))
        alignedPairs=subDict['aligned_pairs']
        #print(alignedPairs,len(alignedPairs))
        #sys.exit()
        for ii in range(len(alignedPairs)-1):
            entry=alignedPairs[ii]
            idx=entry[0]
            absIdx=entry[1]
            if idx!=None and absIdx!=None:
                seq=refSeq[ii]
                edit=editString[ii]
                if seq=='A' and edit!='2' and absIdx<=695:
                    ##695 is where the RT primer binds--anything past this
                    ##is artifact.
                    bb[barCode][readID][absIdx]=int(edit)
    ##
    return bb

def formatForMCA(dataDict,numReads,minEditFreq):
    """
    dataDict={bc:{readID:{position:1/0}}}
    numReads is an integer.
    minEditFreq is a frequency (e.g., 0.7)
    Will sort each bc's readIDs by longest -> shortest, and then select the
    numReads longest reads with at least minEditFreq of 1's
    Will then convert to a list-of-lists like:
    [[0,1,1,0,...,1],
     [1,1,0,1,...,1],
     ...
     [1,0,0,0,...,0]]
     where each row represents a different readID. Will also return a list
     of bcs, which are the row labels.
    """
    aa=[]
    ##This part just gets the numReads reads from each bc
    for bc,readDict in dataDict.items():
        temp=[]
        for readID,positDict in readDict.items():
            if sum(positDict.values())/len(positDict)>=minEditFreq:
                temp.append(len(positDict))
        ##
        temp.sort()
        temp.reverse()
        ##
        temp2=[]
        for readID,positDict in readDict.items():
            if len(positDict) in temp[:numReads]:
                if sum(positDict.values())/len(positDict)>=minEditFreq:
                    temp2.append((bc,positDict))
            if len(temp2)==numReads:
                break
        ##
        aa+=temp2
    ##
    ##now get all the positions
    positions=[]
    for entry in aa:
        for k in entry[1]:
            positions.append(k)
    ##now get unique positions, and then sort them
    positions=list(set(positions))
    positions.sort()
    ##now prepare outputs
    matrixOfEdits=[]
    listOfBCs=[]
    for entry in aa:
        ##record the bc
        listOfBCs.append(entry[0])
        ##now convert the dict to a vector/list
        temp=[]
        for k in positions:
            if k in entry[1]:
                v=entry[1][k]
            else:
                v=numpy.nan##in case no value
            temp.append(v)
        ##
        matrixOfEdits.append(temp)
    ##
    return matrixOfEdits,listOfBCs

def doMCAandPlot(matrixOfEdits,listOfBCs,outPrefix):
    """
    matrixOfEdits is a list of lists, where each list is a list of 1/0/nan
    indicated edit status. listOfBCs are the labels associated with that.
    Will run MCA and plot the first two components.
    """
    ##convert to DF
    X = pd.DataFrame(matrixOfEdits,\
        columns=['A%s'%(ii) for ii in range(len(matrixOfEdits[0]))])
    ##initialize mca
    mca = prince.MCA(n_components=5, n_iter=5, \
        copy=True, check_input=True, engine='sklearn', \
        random_state=42)
    ##
    mca_fit = mca.fit(X)
    X_mca = mca_fit.transform(X)
    ##create labels, but don't actually attached them to the DF
    labels = pd.Series(listOfBCs, name='label')
    ##add the labels
    X_mca['label'] = labels.values
    ##print some results
    print(mca.eigenvalues_summary)
    print(mca.column_coordinates(X))
    ##
    for ii in range(4):
        ##plot the data
        fig, ax = plt.subplots(figsize=(6, 6))
        for label in X_mca['label'].unique():
            subset = X_mca[X_mca['label'] == label]
            ax.scatter(subset[ii], subset[ii+1], label=label)
        ##set axis labels
        ax.set_title('MCA of Binary Vectors')
        ax.set_xlabel('MCA Dimension %s'%(ii+1))##+1 b/c python 0-based
        ax.set_ylabel('MCA Dimension %s'%(ii+2))
        ax.legend()
        ##save output
        plt.savefig('%s.%s.%s.png'%(outPrefix,ii,ii+1))
        plt.close()

def main(args):
    pickleFile,numReads,minEdit,outPrefix=args[0:]
    ##
    dataDict=parsePickleFile(pickleFile)
    print(len(dataDict))
    ##
    matrixOfEdits,listOfBCs=formatForMCA(dataDict,int(numReads),\
        float(minEdit))
    print(len(matrixOfEdits))
    ##
    doMCAandPlot(matrixOfEdits,listOfBCs,outPrefix)

if __name__=='__main__':
    Tee()
    main(sys.argv[1:])
