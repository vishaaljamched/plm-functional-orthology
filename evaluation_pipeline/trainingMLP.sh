#!/bin/bash
#SBATCH --job-name=mlp_training          # Job name
#SBATCH --output=mlpTj%j.out                # Standard output file
#SBATCH --error=mlpT%j.err                 # Error file
#SBATCH --partition=ice-gpu              # Partition name
#SBATCH -N1 --gres=gpu:1                 # Request 1 GPU (not used, but kept as requested)
#SBATCH --cpus-per-task=4                # Request 4 CPU cores
#SBATCH --mem-per-gpu=64GB              # Request 16GB RAM
#SBATCH --time=03:00:00                  # Max job runtime
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ychauhan9@gatech.edu

set -euo pipefail

module load python/3.11
module load anaconda3
module load uv || true

# ===== SCRATCH REDIRECT (DO NOT REMOVE) =====
export SCRATCH_ROOT="/home/hice1/ychauhan9/scratch/mlb_project"

export HF_HOME="$SCRATCH_ROOT/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

export XINFERENCE_HOME="$SCRATCH_ROOT/xinference"
export VLLM_CACHE_ROOT="$SCRATCH_ROOT/vllm"

export TORCH_HOME="$SCRATCH_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$SCRATCH_ROOT/torch_extensions"

export XDG_CACHE_HOME="$SCRATCH_ROOT/.cache"
export PYTHONPYCACHEPREFIX="$SCRATCH_ROOT/pycache"
export PIP_CACHE_DIR="$SCRATCH_ROOT/pip"

export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p \
  "$SCRATCH_ROOT"/{hf,xinference,vllm,torch,torch_extensions,.cache,pip,runs}

# ===== Activate environment =====
cd "$SCRATCH_ROOT"
source mlb/bin/activate

# Create output directory for final artifacts
mkdir -p final_results

# Define the path to your best model weights from the grid search
# UPDATE THIS PATH to your winning run (e.g., outputs/run_lr1e-4_margin1.0/functional_mlp_weights.pth)
WEIGHTS_PATH="outputs/functional_mlp_weights.pth"

echo "=========================================================="
echo "Starting Final Biological Evaluation Pipeline"
echo "Using model weights: $WEIGHTS_PATH"
echo "=========================================================="

# 2. Part 1: Project All Embeddings (Requires GPU)
# We pass the weights path as an environment variable so the python script can catch it
echo "[1/3] Projecting 1280-D embeddings to 512-D functional space..."
export BEST_WEIGHTS=$WEIGHTS_PATH
python project_all_embeddings.py

echo "----------------------------------------------------------"

# 3. Part 2: FAISS & Reciprocal Best Hits (CPU & High RAM)
echo "[2/3] Running FAISS Indexing and enforcing Reciprocal Best Hits..."
python run_faiss_rbh.py
# Move the output to our final folder
mv outputs/rbh_functional_pairs.tsv final_results/

echo "----------------------------------------------------------"

# 4. Part 3: Bio-Statistical Validation & UMAP (CPU & High RAM)
echo "[3/3] Calculating IC-Jaccard statistics and generating UMAPs..."
python evaluate_biology.py
# Move the plot to our final folder
mv outputs/umap_comparison.png final_results/

echo "=========================================================="
echo "Pipeline Complete! All artifacts are in the final_results/ folder."
echo "=========================================================="