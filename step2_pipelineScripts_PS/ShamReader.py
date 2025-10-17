import sys

class shamReader:
    def __init__(self, fname=''):
        '''contructor: saves attribute fname '''
        self.fname = fname

    def doOpen(self):
        if self.fname == '':
            return sys.stdin
        else:
            return open(self.fname)

    def readSham(self):

        with self.doOpen() as fileH:
            samInfo = []
            annotString = ''
            geneType = ''
            editString = ''
            alignmentCoords = [] # [ref_start, ref_end, q_st, q_en]
            
            ct = 0
            for line in fileH:
                if not line.startswith('@'):
                    line = line.rstrip()
                    # print(line)
                    if ct == 0:
                        samInfo += line.split('\t')
                        ct += 1
                    elif ct == 1:
                        annotString = line
                        ct += 1
                    elif ct == 2:
                        geneType = line
                        ct += 1
                    elif ct == 3:
                        editString = line
                        ct += 1
                    elif ct == 4:
                        alignmentCoords += line.split('\t')
                        yield samInfo, annotString, geneType, editString, alignmentCoords
                        ct = 0
                        samInfo = []
                        annotString = ''
                        geneType = ''
                        editString = ''
                        alignmentCoords = []
########################################################################
# Main
# Here is the main program
#
########################################################################

def main(inCL=None):
    ''' '''
    pass


if __name__ == "__main__":
    main()            




                

    
