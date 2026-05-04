import numpy as np
import pandas as pd
from scipy.stats import ranksums
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import math
import os

def load_go_annotations(gaf_paths):
    """Parses GAF files to create a mapping of UniProt ID -> Set of GO Terms (Experimental only)."""
    print("Parsing GO annotation files (Experimental Evidence Only)...")
    go_mapping = {}
    exp_codes = {'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP'}
    
    for path in gaf_paths:
        if not os.path.exists(path):
            print(f"⚠️ Warning: Could not find {path}.")
            continue
            
        df = pd.read_csv(path, sep='\t', comment='!', header=None, dtype=str, on_bad_lines='skip')
        df = df[df[6].isin(exp_codes)]
        
        for uid, group in df.groupby(1):
            if uid not in go_mapping:
                go_mapping[uid] = set()
            go_mapping[uid].update(group[4].tolist())
            
    return go_mapping

def calculate_ic_dict(go_mapping):
    """Calculates Information Content (IC) of each GO term."""
    print("Calculating Information Content for GO terms...")
    term_counts = {}
    total_proteins = len(go_mapping)
    
    for terms in go_mapping.values():
        for term in terms:
            term_counts[term] = term_counts.get(term, 0) + 1
            
    ic_dict = {}
    for term, count in term_counts.items():
        prob = count / total_proteins
        ic_dict[term] = -math.log2(prob)
        
    return ic_dict

def calculate_ic_jaccard(go_set_1, go_set_2, ic_dict):
    """Calculates IC-weighted Jaccard score between two sets of GO terms."""
    if not go_set_1 or not go_set_2: return 0.0 
    intersection = go_set_1.intersection(go_set_2)
    union = go_set_1.union(go_set_2)
    if not union: return 0.0
    intersection_ic = sum(ic_dict.get(go, 1.0) for go in intersection)
    union_ic = sum(ic_dict.get(go, 1.0) for go in union)
    return intersection_ic / union_ic

def load_blast_pairs(blast_file_path):
    """Loads BLAST pairs from tabular format."""
    blast_pairs = []
    if os.path.exists(blast_file_path):
        df = pd.read_csv(blast_file_path, sep='\t', header=None)
        for _, row in df.iterrows():
            try:
                h_id = row[0].split('|')[1]
                y_id = row[1].split('|')[1]
                blast_pairs.append((h_id, y_id))
            except IndexError: continue
    return blast_pairs

def plot_correlation_scatter(merged_tsv_path, go_mapping, ic_dict):
    """Generates the Scatter Plot tracking Sequence Identity vs Biological Function."""
    print("\nGenerating Correlation Scatter Plot...")
    if not os.path.exists(merged_tsv_path):
        print(f"⚠️ Warning: Could not find {merged_tsv_path}. Skipping scatter plot.")
        return
        
    df = pd.read_csv(merged_tsv_path, sep='\t')
    # Coerce to numeric and fill blank BLAST pident values with 0
    df['pident'] = pd.to_numeric(df['pident'], errors='coerce').fillna(0.0)
    
    jaccard_scores = []
    for _, row in df.iterrows():
        h_go = go_mapping.get(str(row['Human_ID']), set())
        y_go = go_mapping.get(str(row['Yeast_ID']), set())
        score = calculate_ic_jaccard(h_go, y_go, ic_dict)
        jaccard_scores.append(score)
        
    df['IC_Jaccard'] = jaccard_scores
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='pident', y='IC_Jaccard', data=df, alpha=0.6, color='purple', edgecolor=None)
    plt.axvline(x=30.0, color='red', linestyle='--', label='30% Sequence Identity Threshold')
    plt.axvspan(0, 30, color='red', alpha=0.05, label='The Twilight Zone')
    
    plt.title('Sequence Identity vs. Biological Functionality (MLP Predictions)', fontsize=14)
    plt.xlabel('Sequence Identity % (BLASTp pident)', fontsize=12)
    plt.ylabel('Information Content Jaccard Score', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig('final_results/correlation_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Scatter plot successfully saved to final_results/correlation_scatter.png")

def run_statistics_and_umap():
    # 1. Setup GO Infrastructure
    gaf_files = ['data/goa_human.gaf', 'data/goa_yeast.gaf']
    go_mapping = load_go_annotations(gaf_files)
    ic_dict = calculate_ic_dict(go_mapping)

    # 2. Calculate Real Scores for MLP
    print("\nCalculating IC-Jaccard scores for MLP predictions...")
    rbh_file = 'final_results/rbh_functional_pairs.tsv'
    if not os.path.exists(rbh_file):
        rbh_file = 'outputs/rbh_functional_pairs.tsv' # Fallback
        
    rbh_df = pd.read_csv(rbh_file, sep='\t')
    mlp_scores = []
    for _, row in rbh_df.iterrows():
        h_go = go_mapping.get(row['Human_ID'], set())
        y_go = go_mapping.get(row['Yeast_ID'], set())
        mlp_scores.append(calculate_ic_jaccard(h_go, y_go, ic_dict))

# =====================================================================
    # 3. BLAST EVALUATION (NOW ACTIVE)
    # =====================================================================
    print("\nCalculating IC-Jaccard scores for BLAST baseline...")
    blast_scores = []
    
    # Extract the BLAST pairs directly from Vishaal's merged file
    blast_df = pd.read_csv('data/esmc_with_blast.tsv', sep='\t')
    
    # Drop rows where BLAST found no match (the blank spaces we saw earlier)
    valid_blast = blast_df.dropna(subset=['qseqid', 'sseqid'])
    
    for _, row in valid_blast.iterrows():
        # qseqid = Human, sseqid = Yeast
        h_go = go_mapping.get(str(row['qseqid']), set())
        y_go = go_mapping.get(str(row['sseqid']), set())
        blast_scores.append(calculate_ic_jaccard(h_go, y_go, ic_dict))

    if blast_scores:
        print("\n--- Final Biological Statistics ---")
        statistic, p_value = ranksums(mlp_scores, blast_scores)
        print(f"Mean MLP RBH Score:   {np.mean(mlp_scores):.4f}")
        print(f"Mean BLAST Score:     {np.mean(blast_scores):.4f}")
        print(f"Wilcoxon Statistic:   {statistic:.4f}")
        print(f"P-value:              {p_value:.2e}")
    # =====================================================================

    # 4. Save Scores and Generate Violin Plot
    print("\nSaving scores and generating Violin Plot...")
    os.makedirs('final_results', exist_ok=True)
    with open('final_results/jaccard_scores.tsv', 'w') as f:
        f.write("Method\tIC_Jaccard_Score\n")
        for s in mlp_scores: f.write(f"MLP_RBH\t{s:.4f}\n")
        for s in blast_scores: f.write(f"BLAST\t{s:.4f}\n") # <-- This line is now active!

    df_scores = pd.read_csv('final_results/jaccard_scores.tsv', sep='\t')
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Method', y='IC_Jaccard_Score', data=df_scores, palette='muted', inner='quartile')
    plt.title('Distribution of IC-Weighted Functional Similarity', fontsize=14)
    plt.ylabel('Information Content Jaccard Score', fontsize=12)
    plt.xlabel('Orthology Prediction Method', fontsize=12)
    plt.savefig('final_results/violin_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Generate the Correlation Scatter Plot
    plot_correlation_scatter('data/esmc_with_blast.tsv', go_mapping, ic_dict) 

    # 6. UMAP Visualization
    print("\nGenerating UMAPs...")
    raw_emb = torch.load('data/precomputed_embeddings.pt', weights_only=True)
    proj_emb = torch.load('outputs/functional_512d_embeddings.pt', weights_only=True)
    
    df_triplets = pd.read_csv('data/training_triplets.tsv', sep='\t')
    known_human = set(df_triplets['anchor_id'].unique())
    known_yeast = set(df_triplets['positive_id'].unique()).union(set(df_triplets['negative_id'].unique()))
    
    valid_human = [k for k in raw_emb.keys() if k in known_human]
    valid_yeast = [k for k in raw_emb.keys() if k in known_yeast]

    human_ids, yeast_ids = valid_human[:2000], valid_yeast[:2000]
    all_ids = human_ids + yeast_ids

    raw_matrix = np.vstack([raw_emb[k].numpy() for k in all_ids])
    proj_matrix = np.vstack([proj_emb[k].numpy() for k in all_ids])

    reducer = umap.UMAP(metric='cosine', random_state=42)
    raw_umap = reducer.fit_transform(raw_matrix)
    proj_umap = reducer.fit_transform(proj_matrix)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.scatter(raw_umap[:len(human_ids), 0], raw_umap[:len(human_ids), 1], c='blue', label='Human', alpha=0.5, s=10)
    ax1.scatter(raw_umap[len(human_ids):, 0], raw_umap[len(human_ids):, 1], c='red', label='Yeast', alpha=0.5, s=10)
    ax1.set_title("Raw ESM-C Space (1280-D)")
    ax1.legend()
    
    ax2.scatter(proj_umap[:len(human_ids), 0], proj_umap[:len(human_ids), 1], c='blue', label='Human', alpha=0.5, s=10)
    ax2.scatter(proj_umap[len(human_ids):, 0], proj_umap[len(human_ids):, 1], c='red', label='Yeast', alpha=0.5, s=10)
    ax2.set_title("MLP Projected Space (512-D)")
    ax2.legend()
    
    plt.savefig('final_results/umap_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Pipeline Complete! Check final_results/ folder.")

if __name__ == "__main__":
    run_statistics_and_umap()