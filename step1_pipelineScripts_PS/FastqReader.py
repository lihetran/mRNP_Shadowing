#!/usr/bin/env python3
# Name: Liam Tran (lihetran)
# Group Members: “None”

import sys


class FastQreader:

    def __init__(self, fname=''):
        '''contructor: saves attribute fname '''
        self.fname = fname

    def doOpen(self):
        if self.fname == '':
            return sys.stdin
        else:
            return open(self.fname)

    def readFastq(self):
        header = ''
        sequence = ''
        sep = ''
        qScore = '' 
        ct=0

        with self.doOpen() as fileH:
            record = []
            n =0
            for line in fileH:
                n += 1
                record.append(line.rstrip())
                #record[1].upper()
                if n == 4:
                    yield record
                    n = 0
                    record = []




            

                    
                        
                    


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
