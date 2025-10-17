import sys

def main(args):
    inFile = args[0]
    
    readIDs = []
    with open(inFile, 'r') as f:
        for line in f:
            if not line.startswith('@'):
                line = line.rstrip().split('\t')
                if line[0] not in readIDs:
                    readIDs.append(line[0])
    print(len(readIDs))

if __name__=='__main__':
    main(sys.argv[1:])
