"""
Liam Tran, October 7 2025

Script to perform Multiple Correspondence Analysis on TadA edited reads. This is based on JA's script mcaPrinceOligoShadowing.py

Input: pickle2 - a pickled file from Liam
    numReads - will pick this many reads from each bc. Rec: 50.
        Will select longer reads first.
    minEdit - minimum edit frequency, e.g., 0.7 for 70%

Output: plot of first two components from MCA, colored by bc of the
    library they come from.

run as python3 mcaPrinceOligoShadowing.py pickle2 numReads minEdit 
    outPrefix
"""
import sys, pickle, collections, numpy
import pandas as pd
import matplotlib.pyplot as plt
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
                #     ##815 is where the RT primer binds--anything past this
                #     ##is artifact.
                #     if absIdx == 342: # make sure we have the first Ile
                        
                #     bb[barCode][readID][absIdx]=int(edit)
                # if seq=='A' and edit!='2' and 245<=absIdx<=444: # get positions around Ile
                    bb[barCode][readID][absIdx]=int(edit)
    ##
    return bb

def formatForMCA(dataDict,numReads,minEditFreq,maxEditFreq):
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
            # if sum(positDict.values())/len(positDict)>=minEditFreq: 
            if minEditFreq<=sum(positDict.values())/len(positDict)<=maxEditFreq:
                temp.append(len(positDict))
        ##
        temp.sort()
        temp.reverse()
        ##
        temp2=[]
        for readID,positDict in readDict.items():
            if len(positDict) in temp[:numReads]:
                # if sum(positDict.values())/len(positDict)>=minEditFreq: 
                if minEditFreq<=sum(positDict.values())/len(positDict)<=maxEditFreq:
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
    ##print index of Ile
    print(positions.index(342))
    ##now prepare outputs
    matrixOfEdits=[]
    listOfBCs=[]
    for entry in aa:
        ##record the bc
        # listOfBCs.append(entry[0])
        ##now convert the dict to a vector/list
        temp=[]
        for k in positions:
            if k in entry[1]:
                v=entry[1][k]
            else:
                v=numpy.nan##in case no value
            temp.append(v)
        # ##
        # if numpy.nan not in temp: ##make sure no nans
        matrixOfEdits.append(temp)
        listOfBCs.append(entry[0])

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
    ## LT added 10/8/25##
    # Add labels to the PCA DataFrame
    X_mca_df = X_mca.copy()
    X_mca_df.columns = [f'Dim_{i+1}' for i in range(X_mca_df.shape[1])]

    # Threshold filter on Dimension 1 (PC1)
    pc1 = X_mca_df['Dim_1']
    mask = pc1.var() <= 0.2  # If you meant global variance check (not likely)
    # Or, if you meant: keep rows with small values along PC1 (i.e., not extreme)
    mask = pc1.abs() <= 0.2  # Absolute deviation filter

    # Filter the PCA data and labels
    X_mca_filtered = X_mca_df[mask].copy()
    labels_filtered = pd.Series(listOfBCs, name='label')[mask].reset_index(drop=True)

    # Add labels
    X_mca_filtered['label'] = labels_filtered.values
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


    return mca.column_contributions_

def heatmapOfContributions(contributions,outPrefix):
    """
    contributions is the output from mca.column_contributions_
    Will plot a heatmap of the contributions.
    """
    import seaborn as sns
    # subset contributions so that every other row is taken
    contributions = contributions.iloc[::2, :]
    # get the other half of the rows
    other_half = contributions.iloc[1::2, :]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(other_half, cmap='viridis', ax=ax)
    ax.set_title('MCA Column Contributions')
    plt.savefig('%s.heatmap.png'%(outPrefix))
    plt.close()

def mkDataPCAable(matrixOfEdits, window):
    '''
    Make the data PCA-able by converting the binary strings into continuous data. I'll do this by using a sliding window approach, where each position is replaced by the average of the edits (1s) in the window centered on that position.
    '''
    smoothedData = []
    for edits in matrixOfEdits:
        smoothedEdits = []
        for i in range(len(edits)):
            start = max(0, i - window // 2)
            end = min(len(edits), i + window // 2 + 1)
            smoothedEdits.append(numpy.nanmean(edits[start:end]))
        smoothedData.append(smoothedEdits)
    return smoothedData

def doPCAandPlot(matrixOfEdits, listOfBCs, outPrefix):
    '''
    Perform PCA on the smoothed data and plot the first two principal components.
    '''
    ##convert to DF
    X = pd.DataFrame(matrixOfEdits,\
        columns=['A%s'%(ii) for ii in range(len(matrixOfEdits[0]))])
    ##initialize pca
    pca = prince.PCA(n_components=5, n_iter=5, \
        copy=True, check_input=True, engine='sklearn', \
        random_state=42)
    ##
    pca_fit = pca.fit(X)
    X_pca = pca.fit_transform(X)
    ##create labels, but don't actually attached them to the DF
    labels = pd.Series(listOfBCs, name='label')
    ##add the labels
    X_pca['label'] = labels.values
    ##print some results
    print(pca.eigenvalues_summary)
    print(pca.column_coordinates_)

    ##
    for ii in range(4):
        ##plot the data
        fig, ax = plt.subplots(figsize=(6, 6))
        for label in X_pca['label'].unique():
            subset = X_pca[X_pca['label'] == label]
            ax.scatter(subset[ii], subset[ii+1], label=label)
        ##set axis labels
        ax.set_title('PCA of TadA Edited Reads')
        ax.set_xlabel('PCA Dimension %s'%(ii+1))##+1 b/c python 0-based
        ax.set_ylabel('PCA Dimension %s'%(ii+2))
        ax.legend()
        ##save output
        plt.savefig('%s.%s.%s.png'%(outPrefix,ii,ii+1))
        plt.close()


def main(args):
    pickleFile,numReads,minEdit,maxEdit,outPrefix=args[0:]
    ##
    dataDict=parsePickleFile(pickleFile)
    # print(dataDict['bc1']['22582f0b-4f38-4cc2-b7e6-8ad176217f0e'])
    ##
    matrixOfEdits,listOfBCs=formatForMCA(dataDict,int(numReads),\
        float(minEdit),float(maxEdit))
    print(len(matrixOfEdits[0]))
    ##
    contributions = doMCAandPlot(matrixOfEdits,listOfBCs,outPrefix)

    heatmapOfContributions(contributions,outPrefix)

    # smoothedData = mkDataPCAable(matrixOfEdits, 10)
    # print(smoothedData)
    # doPCAandPlot(smoothedData, listOfBCs, outPrefix)

if __name__=='__main__':
    main(sys.argv[1:])
