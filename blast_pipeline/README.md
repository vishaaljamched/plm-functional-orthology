## BLAST Baseline Pipeline

Runs BLASTp between H. sapiens and S. cerevisiae Swiss-Prot reviewed proteomes to establish a sequence-based orthology baseline for comparison against the ESM-C + MLP approach.

## Files

- `blastp_job.sh` — SLURM script for forward BLASTp (human vs yeast)
- `blastp_reverse.sh` — SLURM script for reverse BLASTp (yeast vs human)
- `rbh.py` — computes Reciprocal Best Hit pairs from both directions
- `human_vs_yeast.tsv` / `yeast_vs_human.tsv` — raw BLAST output
- `rbh_pairs.tsv` — final RBH pairs (2,304 pairs)

## Usage

```bash
makeblastdb -in yeast_reviewed.fasta -dbtype prot -out yeast_db
makeblastdb -in human_reviewed.fasta -dbtype prot -out human_db
sbatch blastp_job.sh
sbatch blastp_reverse.sh
python3.9 rbh.py
```
