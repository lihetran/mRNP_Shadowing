#!/usr/bin/env python3
# Name: Liam Tran (lihetran)
# Group Members: “None”

import sys


class FastAreader:

    def __init__(self, fname=''):
        '''contructor: saves attribute fname '''
        self.fname = fname

    def doOpen(self):
        if self.fname == '':
            return sys.stdin
        else:
            return open(self.fname)

    def readFasta(self):

        header = ''
        sequence = ''

        with self.doOpen() as fileH:

            header = ''
            sequence = ''

            # skip to first fasta header
            line = fileH.readline()
            while not line.startswith('>'):
                if not line:  # we are at EOF
                    return header, sequence
                line = fileH.readline()
            header = line.rstrip()

            for line in fileH:
                if line.startswith('>'):
                    yield header, sequence
                    header = line.rstrip()
                    sequence = ''
                else:
                    sequence += ''.join(line.rstrip().split()).upper()

        yield header, sequence


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
