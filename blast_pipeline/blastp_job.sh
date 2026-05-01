#!/bin/bash
#SBATCH -J blastp_run
#SBATCH -N 1 --ntasks-per-node=8
#SBATCH -t 4:00:00
#SBATCH --mem=16G

module load gcc/12.3.0
module load blast-plus/2.13.0

blastp -query human_reviewed.fasta -db yeast_db \
  -out human_vs_yeast.tsv \
  -outfmt "6 qseqid sseqid pident length evalue bitscore" \
  -evalue 1e-5 -num_threads 8
