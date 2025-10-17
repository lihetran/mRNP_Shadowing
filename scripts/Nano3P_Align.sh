#!/bin/bash
#get access to appropriate scripts
scripts=/home/liam/bin/
export PATH=$PATH:$scripts
# Requires minimap2
minimap2=/home/liam/bin/minimap2
export PATH=$PATH:$minimap2


# Reference Genome File
reference=$1
#query fasta or fastq
query=$2
junction_bed=$3
mutation_frequency=$4 #suggested to always go for 100% A-to-G
outdir=/home/liam/outdir 
mkdir mutated_files
#mutate query, record mutated indices 
if [[ $query == *fq || *fastq ]] ; then
    python3 /home/liam/bin/Mutate.py -fq $query -m $mutation_frequency --outFile mutated_files/query_mutated.fq;
fi

#mutate reference, record mutated indices 
if [[ "$reference" == *fa || *fasta ]]; then
    python3 /home/liam/bin/Mutate.py -fa $reference -m $mutation_frequency --outFile mutated_files/reference_mutated.fa;
fi 

#######Align unmodified reads to unmodifed genome##########
#minimap2 -ax splice -uf -k14 $reference $query -t 20 --junc-bed $junction_bed --sam-hit-only > $outdir/unmod_mapping.sam

#######Align modified reads to modifed genome##########
minimap2 -ax splice -uf -k14 -t 20 --junc-bed $junction_bed --MD --sam-hit-only mutated_files/reference_mutated.fa mutated_files/query_mutated.fq > $outdir/mod_mapping.sam

#######Unmodify Reads and Find Inosines##########
#python3 /home/liam/bin/FindInosines.py -s $outdir/mod_mapping.sam -r mutated_files/$reference.mutated.indices.pickle -q mutated_files/$query.mutated.indices.pickle -fa $reference -fq $query -o $outdir/mod_mapping.bam  



#minimap2 -ax splice -uf -k14 /data16/joshua/genomes/210303_elegans/200430_allChrs.fa /data16/marcus/nanoporeSoftLinks/210719_nanoporeRun_polyA_0639_L3_replicate/output_dir/cat_files/cat.fastq -t 15 --junc-bed /data16/joshua/genomes/210303_elegans/Caenorhabditis_elegans.WBcel235.100.bed --sam-hit-only > 230406out/230330_cat.out.sam
# minimap2 -ax splice -uf -k14 -t 15 --junc-bed /data16/joshua/genomes/210303_elegans/Caenorhabditis_elegans.WBcel235.100.bed --sam-hit-only --MD testing/reference_mutated.fa testing/query_test.fastq > test.sam