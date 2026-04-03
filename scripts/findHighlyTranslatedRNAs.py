'''
April 1, 2026 LT

I'm not observing a strong signal of ribosome protection from TadA out of yeast lysates. I'm going to narrow down the search space by
finding RNAs that are highly translated. I have Ribo-Seq data and Ribo-depleted RNA-Seq data from PMID: 26876183. I'll find RNAs that are highly
translated by first dividing the RPF data by the RNA expression data. Then I'll plot a histogram of the resulting data (what I'll call a TE score) and
generate a list of genes of the top decile of this distribution.

inputs:
    -Ribo-Seq data (RPFs) from PMID: 26876183, txt file with gene names and RPF RPKM
    -RNA-Seq data (Ribo-depleted) from PMID: 26876183, txt file with gene names and RNA RPKM
    -Mapping of gene systematic names to gene names (ex. YAL001C to TCF3)

outputs:
    -Histogram of TE scores
    -List of genes in the top decile of TE scores
'''

import sys
import pandas as pd
import matplotlib.pyplot as plt

def parseRPFData(rpf_file):
    rpf_df = pd.read_csv(rpf_file, sep='\t')
    names = ['systematic_name', 'RPF_RPKM']
    rpf_df.columns = names
    return rpf_df

def parseRNAData(rna_file):
    rna_df = pd.read_csv(rna_file, sep='\t')
    names = ['systematic_name', 'RNA_RPKM']
    rna_df.columns = names
    return rna_df

def parseMappingFile(mappingFile):
    mapping_df = pd.read_csv(mappingFile, sep='\t')
    names = ['DBID', 'systematic_name', 'organism', 'gene_name', 'description']
    mapping_df.columns = names
    filtered_df = mapping_df[['systematic_name', 'gene_name']]
    return filtered_df

def join_dataframes(rpf_df, rna_df, mapping_df):
    # first lets join the rpf and rna dataframes on systematic name
    merged_df = pd.merge(rpf_df, rna_df, on='systematic_name')
    # now let's join the merged dataframe with the mapping dataframe to get gene names
    final_df = pd.merge(merged_df, mapping_df, on='systematic_name')
    # filter out RNA_RPKM < 1
    final_df = final_df[(final_df['RNA_RPKM'] > 0)]
    final_df = final_df.dropna()
    # compute TE score
    final_df['TE_score'] = final_df['RPF_RPKM'] / final_df['RNA_RPKM']
    # get deciles
    final_df['decile_rank'] = pd.qcut(final_df['TE_score'], 10, labels = False)
    top = final_df[final_df['decile_rank'] == 9]
    top_list = top['gene_name'].tolist()
    return final_df, top_list

def plot_distribution(final_df, outfile):
    '''
    Plot the distribution of TE scores and get a list of the genes in the top decile of TE scores
    '''
    plot_df = final_df[['TE_score', 'gene_name', 'decile_rank']]
    # plot histogram of scores
    figureWidth = 5
    figureHeight = 5

    plt.figure(figsize=(figureWidth, figureHeight))
    panelWidth = 4 / figureWidth
    panelHeight = 4 / figureHeight

    panel = plt.axes([0.15, 0.1, panelWidth, panelHeight])
    panel.hist(plot_df['TE_score'], bins=50, color='blue', edgecolor='black')
    panel.set_xlabel('TE Score (RPF RPKM / mRNA RPKM)', fontsize=14)
    panel.set_ylabel('Frequency', fontsize=14)
    panel.set_title('Distribution of TE Scores', fontsize=16)
    plt.savefig(outfile, dpi=300)


def main(args):
    rpf_file  = args[1]
    rna_file = args[2]
    mappingFile = args[3]
    outFile = args[4]
    listFile = args[5]

    rpf_df = parseRPFData(rpf_file)
    print(rpf_df.head())
    rna_df = parseRNAData(rna_file)
    print(rna_df.head())
    mapping_df = parseMappingFile(mappingFile)
    print(mapping_df.head())

    final_df, top_list = join_dataframes(rpf_df, rna_df, mapping_df)
    print(final_df.head())
    plot_distribution(final_df, outFile)
    # write genes to txt file
    with open(listFile, 'a') as f:
        for gene in top_list:
            f.write(gene + '\n')
    f.close()


if __name__ == "__main__":
    main(sys.argv)