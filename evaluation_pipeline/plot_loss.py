import argparse
import re
import matplotlib.pyplot as plt

def plot_training_loss(args):
    epochs = []
    losses = []
    
    with open(args.log_file, 'r') as f:
        for line in f:
            match = re.search(r'Epoch \[(\d+)/\d+\] \| Average Training Loss: ([\d.]+)', line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                
    if not epochs:
        print(f"No loss data found in {args.log_file}.")
        return

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='b', linewidth=2)
    plt.title('Functional MLP Training Loss (Triplet Margin)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(epochs) 
    
    plt.savefig(args.output_image, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, required=True)
    parser.add_argument('--output_image', type=str, required=True)
    args = parser.parse_args()
    plot_training_loss(args)