"""
Step 2: ESM-C Embedding Precomputation
=======================================
Reads the triplet dataset produced by Step 1, extracts every unique protein,
runs each through an ESM language model (via HuggingFace Transformers), applies
masked Global Average Pooling (GAP) over the sequence dimension, and persists
the resulting embedding dictionary to disk as a .pt file.

Usage
-----
    python precompute_embeddings.py                   # default settings
    python precompute_embeddings.py --batch_size 8    # smaller batches for limited VRAM
    python precompute_embeddings.py --model facebook/esm2_t6_8M_UR50D  # lightweight model
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
TRIPLETS_PATH = DATA_DIR / "training_triplets.tsv"
OUTPUT_PATH = DATA_DIR / "precomputed_embeddings.pt"

DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"
DEFAULT_BATCH_SIZE = 16


# ---------------------------------------------------------------------------
# 1. Extract Unique Sequences
# ---------------------------------------------------------------------------

def extract_unique_proteins(triplets_path: Path) -> List[Tuple[str, str]]:
    """
    Read the triplet TSV and return a deduplicated list of (UniProt_ID, sequence)
    pairs across all three roles (anchor, positive, negative).

    Parameters
    ----------
    triplets_path:
        Path to the tab-separated triplet file produced by Step 1.

    Returns
    -------
    List of (id, sequence) tuples, one per unique protein.
    """
    log.info("Reading triplets from %s …", triplets_path)
    df = pd.read_csv(triplets_path, sep="\t")

    seen: Dict[str, str] = {}
    for role in ("anchor", "positive", "negative"):
        id_col, seq_col = f"{role}_id", f"{role}_seq"
        for uid, seq in zip(df[id_col], df[seq_col]):
            if uid not in seen:
                seen[uid] = seq

    unique_proteins = list(seen.items())
    log.info(
        "Triplets: %d rows  |  unique proteins: %d", len(df), len(unique_proteins)
    )
    return unique_proteins


# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------

class ProteinDataset(Dataset):
    """
    Minimal PyTorch Dataset wrapping a list of (UniProt_ID, sequence) pairs.
    Sequences are returned as raw strings; tokenisation happens inside the
    collate function so padding is applied per-batch.
    """

    def __init__(self, proteins: List[Tuple[str, str]]) -> None:
        self.proteins = proteins

    def __len__(self) -> int:
        return len(self.proteins)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.proteins[idx]


# ---------------------------------------------------------------------------
# 3. Model Initialization
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, device: torch.device):
    """
    Download (or load from cache) an ESM model and its tokenizer from
    HuggingFace Hub.  The model is moved to *device* and set to eval mode.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier, e.g. 'facebook/esm2_t33_650M_UR50D'.
    device:
        Target torch device (CPU or CUDA).

    Returns
    -------
    (tokenizer, model) tuple ready for inference.
    """
    log.info("Loading tokenizer: %s …", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    log.info("Loading model: %s …", model_name)
    model = EsmModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    log.info("Model loaded on %s.", device)
    return tokenizer, model


# ---------------------------------------------------------------------------
# 4. Global Average Pooling
# ---------------------------------------------------------------------------

def masked_gap(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Apply masked Global Average Pooling over the sequence dimension.

    Padding tokens (attention_mask == 0) are excluded from the average so
    they do not dilute the representation of shorter sequences.

    Parameters
    ----------
    last_hidden_state:
        Float tensor of shape [batch_size, seq_len, hidden_dim].
    attention_mask:
        Integer tensor of shape [batch_size, seq_len] with 1 for real tokens
        and 0 for padding.

    Returns
    -------
    Float tensor of shape [batch_size, hidden_dim].
    """
    # Expand mask to match hidden_dim so broadcasting works cleanly.
    mask = attention_mask.unsqueeze(-1).float()          # [B, L, 1]
    sum_embeddings = (last_hidden_state * mask).sum(dim=1)  # [B, H]
    token_counts = mask.sum(dim=1).clamp(min=1e-9)       # [B, 1]  — avoid /0
    return sum_embeddings / token_counts                  # [B, H]


# ---------------------------------------------------------------------------
# 5. Batch Inference Loop
# ---------------------------------------------------------------------------

def embed_proteins(
    proteins: List[Tuple[str, str]],
    tokenizer,
    model: EsmModel,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
) -> Dict[str, Tensor]:
    """
    Run every protein through the ESM model and collect GAP embeddings.

    Sequences are processed in mini-batches.  All intermediate tensors live on
    *device* during forward pass; pooled embeddings are immediately moved to CPU
    to prevent VRAM accumulation.

    Parameters
    ----------
    proteins:
        List of (UniProt_ID, sequence) pairs from :func:`extract_unique_proteins`.
    tokenizer:
        HuggingFace tokenizer for the chosen ESM model.
    model:
        ESM model in eval mode, already on *device*.
    device:
        Torch device used for inference.
    batch_size:
        Number of sequences per GPU forward pass.

    Returns
    -------
    Dict mapping UniProt_ID → 1-D CPU tensor of shape [hidden_dim].
    """
    dataset = ProteinDataset(proteins)

    def collate(batch: List[Tuple[str, str]]):
        ids, seqs = zip(*batch)
        return list(ids), list(seqs)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),  # speeds up CPU→GPU transfer on HPC
    )

    embeddings: Dict[str, Tensor] = {}

    with torch.no_grad():
        for batch_ids, batch_seqs in tqdm(loader, desc="Embedding batches", unit="batch"):
            encoded = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = masked_gap(outputs.last_hidden_state, attention_mask)

            # Detach and move each vector to CPU before storing.
            for uid, vec in zip(batch_ids, pooled):
                embeddings[uid] = vec.cpu()

    return embeddings


# ---------------------------------------------------------------------------
# 6. Output Storage
# ---------------------------------------------------------------------------

def save_embeddings(embeddings: Dict[str, Tensor], out_path: Path) -> None:
    """
    Persist the embedding dictionary to disk using torch.save.

    Parameters
    ----------
    embeddings:
        Dict of UniProt_ID → 1-D CPU tensor.
    out_path:
        Destination .pt file path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, out_path)
    log.info("Saved %d embeddings → %s", len(embeddings), out_path)

    # Quick sanity check: report embedding dimensionality.
    sample_key = next(iter(embeddings))
    log.info("Embedding shape: %s", tuple(embeddings[sample_key].shape))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute ESM embeddings for all proteins in the triplet dataset."
    )
    parser.add_argument(
        "--triplets",
        type=Path,
        default=TRIPLETS_PATH,
        help=f"Path to training_triplets.tsv (default: {TRIPLETS_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output .pt file path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace ESM model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Sequences per forward pass (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker processes for prefetching (default: 0). "
             "Set to 4-8 on HPC nodes with many CPU cores.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=" * 60)
    log.info("Step 2: ESM Embedding Precomputation")
    log.info("=" * 60)
    log.info("  Model       : %s", args.model)
    log.info("  Batch size  : %d", args.batch_size)
    log.info("  Num workers : %d", args.num_workers)
    log.info("  Triplets    : %s", args.triplets)
    log.info("  Output      : %s", args.output)

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("CUDA GPU detected: %s", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Apple MPS detected.")
    else:
        device = torch.device("cpu")
        log.warning("No GPU detected — running on CPU (this will be slow for large datasets).")

    # ------------------------------------------------------------------
    # Step 1 — Extract unique proteins
    # ------------------------------------------------------------------
    log.info("\n[Step 1] Extracting unique proteins …")
    proteins = extract_unique_proteins(args.triplets)

    # ------------------------------------------------------------------
    # Step 2 — Load model
    # ------------------------------------------------------------------
    log.info("\n[Step 2] Initialising model …")
    tokenizer, model = load_model_and_tokenizer(args.model, device)

    # ------------------------------------------------------------------
    # Step 3 — Batch inference
    # ------------------------------------------------------------------
    log.info("\n[Step 3] Running batch inference …")
    embeddings = embed_proteins(proteins, tokenizer, model, device, args.batch_size, args.num_workers)

    # ------------------------------------------------------------------
    # Step 4 — Save
    # ------------------------------------------------------------------
    log.info("\n[Step 4] Saving embeddings …")
    save_embeddings(embeddings, args.output)

    log.info("\nAll done.")


if __name__ == "__main__":
    main()
