"""
Joshua Arribere, May 15, 2026

Script to analyze the distribution of inter-edit distances from polysome shadowing data about
    each in-frame AA

Input: inFile.gtf - to filter sites based on CDS/transcript position
    inFileParquet.txt - line-delimited 'fileName rep parquetPath', same
        convention as calculateProtectionAcrossParquets.py / polysomeShadowHMMQC.py.
        Each row becomes its own library, labeled 'fileName-rep'; parquetPath may be
        a directory of *.parquet chunk files (globbed and read as one library) or a
        single parquet file.
    color_map.txt (optional) - manuscript color TSV 'name rep path hex_color' (no
        leading '#'). Labels are looked up as 'name-rep'/'name_rep' or bare 'name';
        unmatched libraries fall back to common.colors(idx). Only used to color-code
        the per-library overlay in mkPlotAllAA -- mkPlot/mkPlotsPerAA stay pooled
        across every library, same as before, since their own color scheme is
        already spent on AA identity.

Output: histogram of inter-edit distances.

run as python3 interEditDistancePerAA.py inFile.gtf outPrefix inFileParquet.txt
    [color_map.txt]
"""
import sys, common, collections, random, metaStartStop, numpy
from pathlib import Path
import pyarrow.parquet as pq
from logJosh import Tee
from pyx import *

##Only these columns are ever read from a row (see main()'s per-row loop
##body) -- projecting down to just these, AND streaming one row group at a
##time (see main()) rather than pd.read_parquet()'ing a whole file/chunk at
##once, keeps memory bounded by one row group's worth of data regardless of
##how large a library's parquet file(s) are.
REQUIRED_COLUMNS = ['chrom', 'gene_strand', 'transcript_id', 'edit_string',
                    'ref_sequence_aligned', 'read_sequence_aligned', 'absolute_indices']


def load_color_map(path: str) -> dict:
    """
    Parse a manuscript color-map TSV with columns:
        sample_name, rep, path, hex_color (no leading '#')
    Returns a dict keyed by "name_rep", "name-rep" (this script's own
    libraryID convention, see parse_inFileParquet_libs), and bare "name"
    (first match wins for the bare key) mapping to "#RRGGBB".
    """
    color_map = {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 4:
                continue
            name, rep, _path, hexcol = fields[0], fields[1], fields[2], fields[3]
            hexcol = "#" + hexcol.strip().lstrip("#")
            if rep:
                color_map.setdefault(f"{name}_{rep}", hexcol)
                color_map.setdefault(f"{name}-{rep}", hexcol)
            color_map.setdefault(name, hexcol)
    return color_map


def hex_to_pyx_color(hexcol: str):
    hexcol = hexcol.lstrip("#")
    r = int(hexcol[0:2], 16) / 255.0
    g = int(hexcol[2:4], 16) / 255.0
    b = int(hexcol[4:6], 16) / 255.0
    return color.rgb(r, g, b)


def resolve_color(color_map, label, idx):
    """Look up label's manuscript color; fall back to common.colors(idx)."""
    hexcol = color_map.get(label) if color_map else None
    if hexcol:
        return hex_to_pyx_color(hexcol)
    return common.colors(idx)


def parse_inFileParquet_libs(path: str) -> list:
    """
    Parse a line-delimited inFileParquet.txt file of format:
        fileName    rep    parquetPath
    (same convention as calculateProtectionAcrossParquets.py's
    parse_parquet_libs_file / polysomeShadowHMMQC.py), and return a list of
    (libraryID, parquetFiles) tuples with libraryID = 'fileName-rep' and
    parquetFiles = the sorted list of every *.parquet file found under
    parquetPath if it's a directory, or [parquetPath] itself if it's already
    a single file.
    """
    libs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            fileName, rep, parquetPath = parts[0], parts[1], parts[2]
            p = Path(parquetPath)
            if p.is_dir():
                parquetFiles = [str(x) for x in sorted(p.glob("*.parquet"))]
            else:
                parquetFiles = [str(p)]
            libs.append((f"{fileName}-{rep}", parquetFiles))
    return libs

def prepData(dataDict):
    """
    dataDict is of format {position:ct} 
    Will convert this to a format that's easier to plot.
    Will be a list of tuples of format (relPos,freq) ordered
    from most negative position to most positive position.
    """
    dataList1 = sorted([(position, ct) for position, ct in dataDict.items()], key=lambda x: x[0])
    total=sum(dataDict.values())
    dataList2 = sorted([(position, ct/total) for position, ct in dataDict.items() if ct>0], key=lambda x: x[0])

    return dataList1,dataList2

def getTheColor(AA,ct,total):
    if AA=='H':
        return color.cmyk(0,0.8,1,0)#red
    else:
        frac=ct/total
        return color.cmyk(1-frac*.03,.5-0.5*frac,.75*frac,0)#blue>green

def mkPlot(interEditDistances,outPrefix):
    """
    interEditDistances is of the format:
    {AA:[list]}
    Will plot a histogram of the values in list.
    """
    #print(interEditDistances)
    dataToPlot1={}
    dataToPlot2={}
    for k,v in interEditDistances.items():
        tempDict=dict(collections.Counter(v))
        dataToPlot1[k],dataToPlot2[k]=prepData(tempDict)
    #print(dataToPlot1)
    #dataToPlot1 is of the format {UTR5/CDS/UTR3:[dataList]}
    #dataToPlot2 is the same, but dataList if freq instead of ct

    ##now plot everything on the same plot
    g=graph.graphxy(width=8,height=8,
                    key=graph.key.key(pos='tr',hinside=0),
                    x=graph.axis.linear(min=0,max=150,
                                        title='Interedit Distance (nt)'),
                        y=graph.axis.log(title='Frequency'))
    ##
    #dataToPlot2 is of the format {AA:[dataList]}
    total=len(dataToPlot2)
    ct=0
    for AA,freqList in dataToPlot2.items():
        if AA!='H':
            theColor=getTheColor(AA,ct,total)
            g.plot(graph.data.points(freqList,x=1,y=2,title=AA),
                   [graph.style.line([theColor])])
        ct+=1
    ##draw this one last so that it's on top
    g.plot(graph.data.points(dataToPlot2['H'],x=1,y=2,title='H'),
                [graph.style.line([getTheColor('H',total,total)])])
    ##
    g.writePDFfile(outPrefix+'.allTogether')

def mkPlotAllAA(distancesByLibrary,outPrefix,lib_colors=None):
    """
    distancesByLibrary is of the format {libraryID:[list]} (see main()) --
    every AA's distances already pooled together within each library (no
    per-AA split at all, for a coarser look at the raw inter-edit-distance
    distribution on its own), one line PER LIBRARY so different
    samples/replicates can be compared directly. Same normalization
    (prepData) and log-y-axis style as mkPlot. lib_colors is
    {libraryID:pyx color} (see resolve_color); a library missing from it
    falls back to common.colors(idx), cycled in sorted(distancesByLibrary)
    order.
    """
    lib_colors=lib_colors or {}
    g=graph.graphxy(width=8,height=8,
                    key=graph.key.key(pos='tr',hinside=0),
                    x=graph.axis.linear(min=0,max=150,
                                        title='Interedit Distance (nt)'),
                        y=graph.axis.log(min=1e-5,max=1,title='Frequency'))
    for idx,label in enumerate(sorted(distancesByLibrary.keys())):
        tempDict=dict(collections.Counter(distancesByLibrary[label]))
        _,freqList=prepData(tempDict)
        theColor=lib_colors.get(label,common.colors(idx))
        g.plot(graph.data.points(freqList,x=1,y=2,title=label),
               [graph.style.line([theColor,style.linewidth.Thick])])
    g.writePDFfile(outPrefix+'.allDistances')

def getTheData(boundsList,leftMost,rightMost):
    """
    boundsList is a list of the format
    [(left,right),...]
    Will return an ordered list in range [leftMost,rightMost]
    counting the number of entries from boundsList that span a
    given coordinate within that range, scaled s.t. the max is 1.
    """
    aa=collections.defaultdict(int)
    ##
    for entry in boundsList:
        ##
        left=entry[0]
        right=entry[1]
        ##
        for ii in range(left,right+1):
            aa[ii]+=1
    ##
    bb=[]
    theMax=max(aa.values())
    for ii in range(leftMost,rightMost+1):
        if theMax>0:
            bb.append([ii,aa[ii]/theMax])
        else:
            bb.append([ii,0])
    ##
    return bb

def mkPlotsPerAA(inDict,outPrefix):
    """
    inDict is of the format:
    {AA:{length:[(left,right),...]}}
    Will create a plot for every AA and save it in the appropriate filename.
    For every list of (left,right), will assume that the [left,right] is a shadown, and
    tally shadowed nts across all (left,right) for that length. Will then make a plot scaled
    by the max shadowed value (the nt that's shadowed the most). Will plot for just that length.
    Will array the plots vertically.
    """
    ##
    leftMostBound=-100
    rightMostBound=100
    ##
    for AA,lengthToBoundsDict in inDict.items():
        print('\nPlotting %s...'%(AA))
        ##
        ##lengthToBoundsDict is of the format:
        ##{length:[(left,right)]}
        ##
        tempDict={}
        for length,boundsList in lengthToBoundsDict.items():
            theData=getTheData(boundsList,leftMostBound,rightMostBound)
            ##theData is an ordere list of the format [(x,y),...]
            ##with max_y=1
            tempDict[length]=theData
        ##
        #theShortestLength=min(tempDict.keys())
        theShortestLength=10
        #theLongestLength=max(tempDict.keys())
        theLongestLength=150
        temp2=[]
        for length in range(theLongestLength,theShortestLength,-1):
            if length in tempDict:
                temp2.append((length,tempDict[length]))
            else:
                temp2.append((length,[(0,0)]))
        ##temp2 is now an ordered list of [length,theData]
        ##
        ##initialize canvas
        c=canvas.canvas()
        for ii,entry in enumerate(temp2):
            length=entry[0]
            theData=entry[1]
            numberOfShadows=len(lengthToBoundsDict[length])
            
            ##now plot everything on the same plot
            if ii==0:
                g1=graph.graphxy(width=8,height=0.5,
                                x=graph.axis.linear(min=leftMostBound,max=rightMostBound,
                                                    title='Interedit Distance (nt)'),
                                    y=graph.axis.linear(min=0,max=1))
                ##plot the data
                g1.plot(graph.data.points(theData,x=1,y=2),
                       [graph.style.line([color.cmyk.black])])
                ##
                g1.text(g1.width+1,g1.ypos+g1.height/2.,'%s nt (n=%s)'%(length,numberOfShadows),
                        [text.valign.middle,text.halign.boxleft])
                c.insert(g1)
            else:
                g2=graph.graphxy(width=8,height=0.5,ypos=ii*0.5,
                                x=graph.axis.linkedaxis(g1.axes["x"],painter=None),
                                    y=graph.axis.linear(min=0,max=1))
                ##plot the data
                g2.plot(graph.data.points(theData,x=1,y=2),
                       [graph.style.line([color.cmyk.black])])
                ##
                g2.text(g2.width+1,g2.ypos+g2.height/2.,'%s nt (n=%s)'%(length,numberOfShadows),
                        [text.valign.middle,text.halign.boxleft])
                ##
                c.insert(g2)
        ##
        print('Saving plot...')
        c.writePDFfile(outPrefix+'.'+AA)
        print('Done with %s.'%(AA))

def setUpDict():
    """
    Will output a dict of
    {AA:[]}
    """
    return {'A':[],
            'C':[],
            'D':[],
            'E':[],
            'F':[],
            'G':[],
            'H':[],
            'I':[],
            'K':[],
            'L':[],
            'M':[],
            'N':[],
            'P':[],
            'Q':[],
            'R':[],
            'S':[],
            'T':[],
            'V':[],
            'W':[],
            'Y':[],
            '*':[]}

def setUpDict2():
    """
    Will output a dict of
    {AA:[]}
    """
    return {'A':collections.defaultdict(list),
            'C':collections.defaultdict(list),
            'D':collections.defaultdict(list),
            'E':collections.defaultdict(list),
            'F':collections.defaultdict(list),
            'G':collections.defaultdict(list),
            'H':collections.defaultdict(list),
            'I':collections.defaultdict(list),
            'K':collections.defaultdict(list),
            'L':collections.defaultdict(list),
            'M':collections.defaultdict(list),
            'N':collections.defaultdict(list),
            'P':collections.defaultdict(list),
            'Q':collections.defaultdict(list),
            'R':collections.defaultdict(list),
            'S':collections.defaultdict(list),
            'T':collections.defaultdict(list),
            'V':collections.defaultdict(list),
            'W':collections.defaultdict(list),
            'Y':collections.defaultdict(list),
            '*':collections.defaultdict(list)}

def getNonEditedLength(index2,refSequenceAligned,editString):
    """
    Will return the length of an unedited stretch centered around index2
    """
    ##a bunch of print statements are here for checking behavior. All commented out.
    #print(index2)
    #print(refSequenceAligned)
    #print(editString)
    #print(len(editString))
    #print(editString.count('0'),editString.count('1'))
    leftLength=0
    #print('left')
    for ii in range(index2,0,-1):
        #print(ii,editString[ii])
        if editString[ii]=='0':
            leftLength+=1
        elif editString[ii]=='1':
            break
    ##
    rightLength=0
    #print('right')
    for ii in range(index2+1,len(editString),1):
        #print(ii,editString[ii])
        if editString[ii]=='0':
            rightLength+=1
        elif editString[ii]=='1':
            break
    ##
    #print(leftLength+rightLength)
    #sys.exit()
    return leftLength+rightLength, (-leftLength,rightLength)

def getAA(index2,refSeqAligned,strand):
    codon=None
    if strand=='+':
        if len(refSeqAligned)>index2+3:
            codon=refSeqAligned[index2:index2+3]
    elif strand=='-':
        if index2-3>0:
            codon=common.revCompl(refSeqAligned[index2-3:index2])
    if codon:
        if len(set(codon))==len(set(codon).intersection(set(['T','A','G','C']))):##check for canonical nts
            return common.translate(codon)[1]

    return 'na'

def main(args):
    ##
    gtfFile=args[0]
    outPrefix=args[1]
    inFileParquetPath=args[2]
    colorMapPath=args[3] if len(args)>3 else None
    ##
    gtfDict=metaStartStop.parseGTF(gtfFile)
    ##gtfDict is of the format {strand:{chr:{absIndx:(txtName,relStart,relStop)]}}}
    ##
    color_map=load_color_map(colorMapPath) if colorMapPath else {}
    libs=parse_inFileParquet_libs(inFileParquetPath)
    if not libs:
        sys.exit('No libraries found in %s; exiting.'%(inFileParquetPath))
    if colorMapPath:
        unmatched=[label for label,_ in libs if label not in color_map]
        if unmatched:
            print('  WARNING: no color found in %s for librar%s %s; '
                  'falling back to the default palette.'%(
                  colorMapPath,'y' if len(unmatched)==1 else 'ies',unmatched))
    lib_colors={label:resolve_color(color_map,label,idx) for idx,(label,_) in enumerate(libs)}
    ##
    interEditDistances=setUpDict()
    #interEditDistances is of the format:
    #{AA:[]}
    leftAndRightBounds=setUpDict2()
    ##leftAndRightBounds is of the format:
    #{AA:collections.defaultdict(list)}
    ##pooled across every AA within a library, for mkPlotAllAA's per-library
    ##overlay -- see that function's docstring.
    distancesByLibrary=collections.defaultdict(list)
    ##
    readCt=0
    for label,parquetFiles in libs:
      for parquetFile in parquetFiles:
        ##
        print('\nAnalyzing %s (%s)...'%(parquetFile,label))
        ##stream one row group at a time (pf.read_row_group), projected down
        ##to REQUIRED_COLUMNS, instead of pd.read_parquet()'ing the whole
        ##file into memory at once -- keeps peak memory bounded by one row
        ##group's worth of data regardless of how large the file is.
        pf=pq.ParquetFile(parquetFile)
        print('Parquet file has %d row(s) across %d row group(s)'%(
              pf.metadata.num_rows,pf.metadata.num_row_groups))
        for rg in range(pf.metadata.num_row_groups):
          df=pf.read_row_group(rg,columns=REQUIRED_COLUMNS).to_pandas()
          #df=df[:1000]#useful for subsetting the data during troubleshooting.
          ##now loop through each row in the dataframe, which is a different read.
          for index,row in df.iterrows():
            ##
            chrom=row['chrom']
            strand=row['gene_strand']
            transcript_id=row['transcript_id']
            ##
            ##now check if this read overlaps with any of the positions in gtfDict.
            if chrom in gtfDict[strand] and transcript_id and 'RDN' not in transcript_id:
                readCt+=1
                ##
                #lastEdit=0
                editString=row['edit_string']
                #listEditString=list(editString)
                #random.shuffle(listEditString)
                #editString=''.join(listEditString)
                ##
                for index2,(refNt,readNt,edit,absIdx) in enumerate(zip(row['ref_sequence_aligned'],row['read_sequence_aligned'],editString,row['absolute_indices'])):
                    ##
                    edit=int(edit)
                    ##
                    if edit in [0,1] and absIdx in gtfDict[strand][chrom] and len(gtfDict[strand][chrom][absIdx])==1:##restriction for uniquely assignable
                        ##
                        txtID,relStart,relStop=gtfDict[strand][chrom][absIdx][0]
                        #print(txtID,relStart,relStop)
                        ##
                        if not relStart%3==0:##then it's out-of-frame, continue
                            continue
                        #
                        if relStart<3 or relStop>-3:#restriction for in CDS
                            continue
                        ##
                        if edit==0:##only focused on unedited nts--otherwise the interedit window is 0
                            nonEditedLength,asTuple=getNonEditedLength(index2,row['ref_sequence_aligned'],editString)
                            ##asTuple is (-leftLength,rightLength), and nonEditedLength is just leftLength+rightLength
                            ##
                            AA=getAA(index2,row['ref_sequence_aligned'],strand)
                            ##
                            if AA in interEditDistances:
                                interEditDistances[AA].append(nonEditedLength)
                                leftAndRightBounds[AA][nonEditedLength].append(asTuple)
                                distancesByLibrary[label].append(nonEditedLength)
                            #print(interEditDistances)
    for k,v in interEditDistances.items():
        print(k,numpy.average(v),numpy.median(v))
    ##
    print('%s reads passed filters and were analyzed.'%(readCt))
    ##interEditDistances is now a dict with keys AAs
    ##and values a list of distances between edits (in txt space).
    ##will plot a histogram for each AA
    mkPlot(interEditDistances,outPrefix)
    ##
    mkPlotAllAA(distancesByLibrary,outPrefix,lib_colors=lib_colors)
    ##
    mkPlotsPerAA(leftAndRightBounds,outPrefix)

if __name__=='__main__':
    Tee()
    main(sys.argv[1:])