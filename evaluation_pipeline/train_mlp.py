import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Tuple, Dict

class TripletEmbeddingDataset(Dataset):
    """Custom Dataset to load triplet IDs and fetch their corresponding ESM-C embeddings."""
    def __init__(self, tsv_path: str, embeddings_path: str):
        print(f"Loading dataset from: {tsv_path}")
        self.triplets_df = pd.read_csv(tsv_path, sep='\t')
        
        print(f"Loading precomputed embeddings from: {embeddings_path}")
        self.embeddings: Dict[str, torch.Tensor] = torch.load(
            embeddings_path, 
            map_location='cpu', 
            weights_only=True
        )

        # --- THE FIX: Data Validation & Filtering ---
        initial_len = len(self.triplets_df)
        valid_keys = set(self.embeddings.keys())

        # Keep only rows where anchor, positive, AND negative are all in the dictionary
        self.triplets_df = self.triplets_df[
            self.triplets_df['anchor_id'].isin(valid_keys) &
            self.triplets_df['positive_id'].isin(valid_keys) &
            self.triplets_df['negative_id'].isin(valid_keys)
        ].reset_index(drop=True)

        dropped = initial_len - len(self.triplets_df)
        if dropped > 0:
            print(f"⚠️ Warning: Dropped {dropped} triplets because one or more proteins were missing from the embeddings dictionary.")
        print(f"Final valid triplets for training: {len(self.triplets_df)}")

    def __len__(self) -> int:
        return len(self.triplets_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.triplets_df.iloc[idx]
        
        anchor_id = row['anchor_id']
        positive_id = row['positive_id']
        negative_id = row['negative_id']

        # We can safely remove the try/except block now, 
        # as we guaranteed these keys exist in __init__
        anchor_tensor = self.embeddings[anchor_id]
        positive_tensor = self.embeddings[positive_id]
        negative_tensor = self.embeddings[negative_id]

        return anchor_tensor, positive_tensor, negative_tensor
class FunctionalMLP(nn.Module):
    """Multi-Layer Perceptron projecting 1280-D ESM-C embeddings to 512-D."""
    def __init__(self):
        super(FunctionalMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        return F.normalize(x, p=2, dim=1)

def train_model(args):
    """Main training loop using parsed arguments."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")

    # Initialize Dataset and HPC-Optimized DataLoader
    dataset = TripletEmbeddingDataset(
        tsv_path=args.tsv_path,
        embeddings_path=args.embeddings_path
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False # Speeds up CPU to GPU transfer
    )

    model = FunctionalMLP().to(device)
    criterion = nn.TripletMarginLoss(margin=args.margin, p=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            anchor = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)

            optimizer.zero_grad()

            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = criterion(anchor_out, positive_out, negative_out)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] | Average Training Loss: {avg_loss:.4f}")

    # Save the trained model weights
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "functional_mlp_weights.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model weights saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Functional Projection Head on PACE")
    
    # File Paths
    parser.add_argument('--tsv_path', type=str, required=True, help="Path to training_triplets.tsv")
    parser.add_argument('--embeddings_path', type=str, required=True, help="Path to precomputed_embeddings.pt")
    parser.add_argument('--output_dir', type=str, default="./outputs", help="Directory to save the trained model")
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for AdamW")
    parser.add_argument('--margin', type=float, default=1.0, help="Margin for Triplet Margin Loss")
    
    # System settings
    parser.add_argument('--num_workers', type=int, default=4, help="Number of CPU threads for data loading")
    
    args = parser.parse_args()
    train_model(args)