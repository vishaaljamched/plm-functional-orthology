#!/bin/bash
#SBATCH -J blastp_reverse
#SBATCH -N 1 --ntasks-per-node=8
#SBATCH -t 4:00:00
#SBATCH --mem=16G

module load gcc/12.3.0
module load blast-plus/2.13.0

blastp -query yeast_reviewed.fasta -db human_db \
  -out yeast_vs_human.tsv \
  -outfmt "6 qseqid sseqid pident length evalue bitscore" \
  -evalue 1e-5 -num_threads 8
