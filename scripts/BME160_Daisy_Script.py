#!/usr/bin/env python3
#Group Members: Emma Arbil (earbil), Daisy Lopez (dlopez87), Amanda Ferguson (ammfergu)

import sys
import matplotlib.pyplot as plt
import subprocess
import pysam 
from Bio import SeqIO

""" This script converts all A bases to G bases in the references and reads files, aligns these converted reads
to the converted reference using teh minimap2 tool, then calculates the A to G editing efficiency for each read.
The script then plots the empirical cumulitive distribution function (eCDF) to be visualized.
This is to be used for nanopore reads of RNA treated with TadA8.20. """

class AGConverter :
    """ Convert all A to G in the reference (FASTA) and reads (FASTQ) files.
        Output all of the converted sequences to new files for the downstream alignment."""
    def __init__(self, refFile, readsFile, outRef, outReads) :
        self.refFile = refFile
        self.readsFile = readsFile
        self.outRef = outRef
        self.outReads = outReads

    def convertFasta(self):
        """ For the FASTA file from the CLI, write converted sequence to new file """
        with open(self.outRef, "w") as out :
            for record in SeqIO.parse(self.refFile, "fasta"):
                record.seq = record.seq.upper().replace("A", "G")
                SeqIO.write(record, out, "fasta")

    def convertFastq(self):
        """ For the FASTQ file from the CLI, write converted sequence to new file """
        with open(self.outReads, "w") as out :
            for record in SeqIO.parse(self.readsFile, "fastq"):
                record.seq = record.seq.upper().replace("A","G")
                SeqIO.write(record, out, "fastq")

class AtoGAligner:
    """ Use minimap2 to align the converted reads to a converted reference and produce a SAM file of aligments """
    def __init__(self, refPath):
        self.refPath = refPath # use the set reference path

    def alignReads(self, readsPath, outSam):
        cmd = [
            "minimap2",
            "-ax", "map-ont", # ONT settings
            "-t", "20", # Use 20 threads
            "--secondary=no", # Only use primary alignments
            "--for-only", # Only use fwd strand
            "--cs=long", # Include the cs tag
            "--sam-hit-only", # Only report mapped reads
            self.refPath,
            readsPath
        ]
        print(cmd) # Print the command for debugging
        with open(outSam, "w") as samOut:
            subprocess.run(cmd, stdout=samOut, check=True)


class AGAnalyzer :
    """ Analyze the editiing efficiency from the aligned reads and plot the empirical CDF of mismatch rates per read."""
    def __init__(self, samPath) :
        self.samfile = pysam.AlignmentFile(samPath,"r")
        self.rates = [] # List to hold the mismatch rate per read for plotting

    def calcEditRates(self) :
        """ For each primary, mapped read, count the A bases in the ref, then find the mismatch rate in the reads."""
        for read in self.samfile :
            # Only align primary reads that are mapped using, otherwise stop
            if read.is_unmapped or not read.is_primary:
                continue
            # Use pysam tool to align the pairs
            alignedPairs = read.get_aligned_pairs(with_seq = True) # align with the bases using with_seq parameter on
            totalA = 0
            mismatches = 0
            
            for qpos, rpos, seq in alignedPairs :
                if qpos is not None and rpos is not None and seq is not None:
                    qbase, rbase = seq
                    if rbase == "A": # Check for A in the reference
                        totalA += 1 # Add to total if so
                        if qbase == "G": # Check for G in the read
                            mismatches += 1 # If yes it is mismatch

            rate = mismatches / totalA if totalA > 0 else 0 
            self.rates.append(rate)
        self.samfile.close() # Close when done

    def plotCDF(self, title = "eCDF of A to G Editing Efficiency using TadA8.20") :
        """Plot the eCDF of A to G editing rates for all reads."""
        rates = sorted(self.rates)
        n = len(rates)
        yVals = [i / n for i in range(1, n + 1)] # Create a list of the y values from 1/n to 1

        plt.figure (figsize=(10,6))
        plt.plot(rates, yVals, linestyle="solid")
        plt.xlabel("Editing Efficiency (A-G mismatch / A bases)")
        plt.ylabel("Empirical CDF")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        

class CommandLine :
    """ Parse command line arguments for reference, reads, and output files."""
    def __init__(self, inOpts = None) :
        import argparse
        self.parser = argparse.ArgumentParser(
            description = "AG mismatch analyzer for long Oxford-Nanopore reads treated with TAD",
            epilog = "Converts A to G in reference and reads files, aligns using mappy, and finds mismatch rates.",
            add_help = True,
            prefix_chars = "-" )
                        
        self.parser.add_argument("-r", "--reference", required = True, help = "Path to original reference FASTA file")
        self.parser.add_argument("-q", "--reads", required = True, help = "Path to original reads FASTQ file")
        self.parser.add_argument("--refOut", default = "mutatedRefAG.fasta", help = "Output for mutated reference FASTA")
        self.parser.add_argument("--readsOut", default = "mutatedReadsAG.fastq", help = "Output for mutated reads FASTQ")
        self.parser.add_argument("--sam", default = "alignments.AtoG.sam", help = "Output SAM for aligned reads")

        if inOpts is None :
            self.args = self.parser.parse_args()
        else :
            self.args = self.parser.parse_args(inOpts)

def main() :
    """ Run the A to E efficiency pipeline: convert, align, analyze, plot."""
    cmd = CommandLine()
    args = cmd.args

    # convert A to G in reference and reads files, write to new files
    converter = AGConverter(args.reference, args.reads, args.refOut, args.readsOut)
    converter.convertFasta()
    converter.convertFastq()

    # align reads with minimap2
    aligner = AtoGAligner(args.refOut)
    aligner.alignReads(args.readsOut, args.sam)

    # calculate A-G mismatch rate per read
    analyzer = AGAnalyzer(args.sam)
    analyzer.calcEditRates()
    analyzer.plotCDF()

if __name__ == "__main__" : 
    main()
                
