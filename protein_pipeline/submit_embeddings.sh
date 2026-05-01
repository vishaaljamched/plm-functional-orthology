#!/bin/bash
# =============================================================================
# SLURM Job Script — Step 2: ESM Embedding Precomputation
# Target cluster: Georgia Tech PACE ICE
# =============================================================================
#
# Submit with:
#   sbatch submit_embeddings.sbatch
#
# Monitor with:
#   squeue -u $USER
#   tail -f logs/embed_<jobid>.log

# ---------------------------------------------------------------------------
# Resource requests
# ---------------------------------------------------------------------------
#SBATCH --job-name=esm_embed
#SBATCH --partition=ice-gpu              # PACE ICE GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8               # CPU cores for DataLoader workers
#SBATCH --mem=32G                        # System RAM (model + tokenizer + data)
#SBATCH --gres=gpu:A100:1               # Request 1 A100 GPU (40 or 80 GB VRAM)
#SBATCH --time=04:00:00                 # Wall-clock limit (HH:MM:SS)
#SBATCH --output=logs/embed_%j.log      # stdout + stderr merged into one file
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vjamched3@gatech.edu

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
echo "======================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPU(s)       : $CUDA_VISIBLE_DEVICES"
echo "Start time   : $(date)"
echo "Working dir  : $(pwd)"
echo "======================================================"

# Load PACE ICE software modules
module purge
module load anaconda3/2023.09   # provides conda; version may vary on ICE

# Activate your conda environment (create it once with the instructions below)
conda activate plm-orthology

# Ensure logs directory exists
mkdir -p logs

# ---------------------------------------------------------------------------
# Run the embedding script
# ---------------------------------------------------------------------------
# Adjust --batch_size based on available VRAM:
#   A100 40GB  →  batch_size=64  (650M model)
#   V100 32GB  →  batch_size=32
#   Smaller GPU → batch_size=8 or 16

python precompute_embeddings.py \
    --model       facebook/esm2_t33_650M_UR50D \
    --batch_size  64 \
    --num_workers 4 \
    --triplets    data/training_triplets.tsv \
    --output      data/precomputed_embeddings.pt

echo "======================================================"
echo "End time : $(date)"
echo "======================================================"
