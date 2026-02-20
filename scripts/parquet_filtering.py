'''
February 20, 2026 LT

This script performs preprocessing of long-read sequencing data stored in parquet files. 
For example, if we don't care about certain barcodes, we can filter them out before performing PCA.
'''

import pandas as pd
import numpy as np
import sys
import os

def remove_barcodes(df, barcodes_to_remove):
    """
    Remove specified barcodes from the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing a 'barcode' column.
    barcodes_to_remove (list): A list of barcodes to be removed from the DataFrame.

    Returns:
    pd.DataFrame: A new DataFrame with the specified barcodes removed.
    """
    return df[~df['barcode'].isin(barcodes_to_remove)]

def read_parquet_file(file_path):
    """
    Read a parquet file into a DataFrame.

    Parameters:
    file_path (str): The path to the parquet file.

    Returns:
    pd.DataFrame: The DataFrame containing the data from the parquet file.
    """
    return pd.read_parquet(file_path)

def main():
    # read parquet directory from command line argument
    parquet_dir = sys.argv[1]
    barcodeList = sys.argv[2].split(",")  # list of barcodes to remove, passed as a comma-separated string
    # iterate over parquet files in directory
    for file in os.listdir(parquet_dir):
        if file.endswith(".parquet"):
            file_path = os.path.join(parquet_dir, file)
            df = read_parquet_file(file_path)
            df_filtered = remove_barcodes(df, barcodeList)
            # save the filtered DataFrame back to a new parquet file
            output_file_path = os.path.join(parquet_dir, f"filtered_{file}")
            df_filtered.to_parquet(output_file_path)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

