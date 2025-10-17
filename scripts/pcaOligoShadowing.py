"""
Liam Tran, September 3, 2025

Based on JA's script mcaPrinceOligoShadowing.py. One thing about doing an 
MCA is that it assumes edits are independent. Tad editing could be processive and
in that case, edits would not be indepedent. One way to solve this is to use a sliding window approach to group edits together.


Input: pickle2 - a pickled file from Liam
    numReads - will pick this many reads from each bc. Rec: 50.
        Will select longer reads first.
    minEdit - minimum edit frequency, e.g., 0.7 for 70%

Output: plot of first two components from MCA, colored by bc of the
    library they come from.

run as python3 mcaPrinceOligoShadowing.py pickle2 numReads minEdit 
    outPrefix
"""
import sys, common, pickle, collections, numpy
import pandas as pd
import matplotlib.pyplot as plt
import prince

