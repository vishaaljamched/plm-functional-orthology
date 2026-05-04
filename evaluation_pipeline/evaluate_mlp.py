import argparse
import torch
from torch.utils.data import DataLoader
from train_mlp import FunctionalMLP, TripletEmbeddingDataset

def evaluate_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the Test Data
    test_dataset = TripletEmbeddingDataset(
        tsv_path='data/test_triplets.tsv', 
        embeddings_path='data/precomputed_embeddings.pt'
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Initialize Model and dynamically load weights
    model = FunctionalMLP().to(device)
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.eval()

    correct_triplets = 0
    total_triplets = 0

    with torch.no_grad():
        for anchor, positive, negative in test_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            a_out = model(anchor)
            p_out = model(positive)
            n_out = model(negative)

            sim_positive = torch.sum(a_out * p_out, dim=1)
            sim_negative = torch.sum(a_out * n_out, dim=1)

            correct = (sim_positive > sim_negative).sum().item()
            correct_triplets += correct
            total_triplets += anchor.size(0)

    accuracy = (correct_triplets / total_triplets) * 100
    # Print a clean string that our bash script can easily capture
    print(f"Test Accuracy: {accuracy:.2f}% (Total Triplets: {total_triplets})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str, required=True)
    args = parser.parse_args()
    evaluate_model(args)