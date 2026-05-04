import torch
import os
from train_mlp import FunctionalMLP

def project_embeddings():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Projecting on: {device}")

    # 1. Load the raw 1280-D embeddings and the trained model
    raw_embeddings = torch.load('data/precomputed_embeddings.pt', map_location='cpu', weights_only=True)
    model = FunctionalMLP().to(device)
    weights_path = os.environ.get('BEST_WEIGHTS', 'outputs/functional_mlp_weights.pth')
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    projected_embeddings = {}

    # 2. Pass every protein through the MLP
    with torch.no_grad():
        for protein_id, tensor in raw_embeddings.items():
            # Add batch dimension, move to device, project, remove batch dim, move to CPU
            tensor_batch = tensor.unsqueeze(0).to(device)
            out = model(tensor_batch)
            projected_embeddings[protein_id] = out.squeeze(0).cpu()

    # 3. Save the new 512-D functional space
    torch.save(projected_embeddings, 'outputs/functional_512d_embeddings.pt')
    print(f"Successfully projected {len(projected_embeddings)} proteins into 512-D space.")

if __name__ == "__main__":
    project_embeddings()