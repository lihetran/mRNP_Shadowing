'''
January 23, 2026 LT

This script will perform PCA on the indicator matrix generated from shadowingBamToIndicatorMatrix.py.
It reads the parquet files containing the indicator matrix, performs PCA, and saves the results. Functionality to do this in
chunks is included to handle large datasets.

input: parquet files with indicator matrix
       output_dir - directory to save PCA results
output: PCA results saved as numpy arrays and plots
'''

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def perform_pca_on_indicator_matrix(parquet_file, output_dir, n_components=10, chunk_size=10000):