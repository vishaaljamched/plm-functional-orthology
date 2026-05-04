import torch
import faiss
import numpy as np
import pandas as pd  # <-- Added pandas to read the TSV

def run_faiss_search():
    print("Loading 512-D embeddings...")
    embeddings = torch.load('outputs/functional_512d_embeddings.pt', weights_only=True)
    
    # --- THE FIX: Use the TSV to figure out which ID is which species ---
    print("Sorting species using training_triplets.tsv...")
    df = pd.read_csv('data/training_triplets.tsv', sep='\t')
    
    # Extract unique sets of IDs
    known_human = set(df['anchor_id'].unique())
    known_yeast = set(df['positive_id'].unique()).union(set(df['negative_id'].unique()))
    
    # Sort the dictionary keys based on those sets
    human_ids = [k for k in embeddings.keys() if k in known_human]
    yeast_ids = [k for k in embeddings.keys() if k in known_yeast]
    
    print(f"Successfully separated: {len(human_ids)} Human proteins and {len(yeast_ids)} Yeast proteins.")
    
    if len(human_ids) == 0 or len(yeast_ids) == 0:
        raise ValueError("Critical Error: Separation failed. One of the species lists is empty.")

    # Convert to contiguous float32 numpy arrays for FAISS
    print("Building FAISS matrices...")
    human_matrix = np.vstack([embeddings[k].numpy() for k in human_ids]).astype('float32')
    yeast_matrix = np.vstack([embeddings[k].numpy() for k in yeast_ids]).astype('float32')

    # 2. Build FAISS Indices (IndexFlatIP for Cosine Similarity since vectors are L2 normalized)
    d = 512
    yeast_index = faiss.IndexFlatIP(d)
    yeast_index.add(yeast_matrix)
    
    human_index = faiss.IndexFlatIP(d)
    human_index.add(human_matrix)

    # 3. Perform Bi-directional Search
    print("Querying Human -> Yeast...")
    h2y_distances, h2y_indices = yeast_index.search(human_matrix, 1) # Top 1 match
    
    print("Querying Yeast -> Human...")
    y2h_distances, y2h_indices = human_index.search(yeast_matrix, 1) # Top 1 match

    # 4. Enforce Reciprocal Best Hits (RBH)
    print("Enforcing strict Reciprocal Best Hits...")
    rbh_pairs = []
    for h_idx, human_id in enumerate(human_ids):
        # Best yeast match for this human
        best_yeast_idx = h2y_indices[h_idx][0]
        best_yeast_id = yeast_ids[best_yeast_idx]
        
        # Check if this human is ALSO the best match for that yeast
        if y2h_indices[best_yeast_idx][0] == h_idx:
            score = h2y_distances[h_idx][0]
            rbh_pairs.append((human_id, best_yeast_id, score))

    # Save results for Anish
    with open('outputs/rbh_functional_pairs.tsv', 'w') as f:
        f.write("Human_ID\tYeast_ID\tCosine_Score\n")
        for h, y, s in sorted(rbh_pairs, key=lambda x: x[2], reverse=True):
            f.write(f"{h}\t{y}\t{s:.4f}\n")
            
    print(f"Success! Found {len(rbh_pairs)} strict Reciprocal Best Hits!")

if __name__ == "__main__":
    run_faiss_search()