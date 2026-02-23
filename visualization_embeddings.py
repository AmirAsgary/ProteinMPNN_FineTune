#!/usr/bin/env python3
"""
t-SNE Visualization Pipeline for Protein Embeddings
Reads h5 embeddings and creates colored t-SNE visualization
"""

import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from pathlib import Path


def load_embeddings(h5_path):
    """Load embeddings from h5 file"""
    print(f"Loading embeddings from {h5_path}...")

    with h5py.File(h5_path, 'r') as f:
        h_v = f['h_V'][:]  # [N_proteins, hidden_dim]
        h_e = f['h_E'][:]  # [N_proteins, hidden_dim]
        names = [n.decode() for n in f['names'][:]]

    # Concatenate node and edge embeddings
    embed = np.concatenate([h_v, h_e], axis=-1)

    print(f"  Loaded {embed.shape[0]} samples with {embed.shape[1]} dimensions")
    print(f"  h_V shape: {h_v.shape}, h_E shape: {h_e.shape}")

    return embed, names


def extract_labels(names):
    """Extract labels from sample names (first part before underscore)"""
    labels = [int(i.split('_')[0]) for i in names]

    unique_labels = np.unique(labels)
    print(f"\nExtracted labels from names:")
    print(f"  Number of unique labels: {len(unique_labels)}")
    print(f"  Label range: {min(labels)} - {max(labels)}")

    return np.array(labels)


def run_tsne(embed, perplexity=30, random_state=42):
    """Perform t-SNE dimensionality reduction"""
    print(f"\nRunning t-SNE (perplexity={perplexity})...")

    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity,
        max_iter=1000,
        verbose=1
    )

    embed_2d = tsne.fit_transform(embed)

    print(f"  t-SNE completed")
    print(f"  Output shape: {embed_2d.shape}")
    print(f"  X range: [{embed_2d[:, 0].min():.2f}, {embed_2d[:, 0].max():.2f}]")
    print(f"  Y range: [{embed_2d[:, 1].min():.2f}, {embed_2d[:, 1].max():.2f}]")

    return embed_2d


def create_visualization(embed_2d, labels, names, output_path):
    """Create and save colored t-SNE visualization"""
    print(f"\nCreating visualization...")

    # Create output directory if needed
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare data
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)

    # Choose colormap based on number of labels
    if n_labels <= 10:
        cmap = plt.cm.tab10
    elif n_labels <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.viridis

    colors = cmap(np.linspace(0, 1, n_labels))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot each label with different color
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        count = np.sum(mask)
        ax.scatter(
            embed_2d[mask, 0],
            embed_2d[mask, 1],
            c=[colors[i]],
            label=f'{lbl} (n={count})',
            alpha=0.6,
            s=20,
            edgecolors='none'
        )

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('t-SNE Visualization of Protein Embeddings', fontsize=14, fontweight='bold')

    # Legend handling based on number of labels
    if n_labels <= 30:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        # Too many labels for legend, use colorbar instead
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(unique_labels), vmax=max(unique_labels)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Label', fontsize=12)

    plt.tight_layout()

    # Save plot
    plot_path = output_path / 'tsne_visualization.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved plot to {plot_path}")
    plt.close()

    # Save results to CSV
    df = pd.DataFrame({
        'name': names,
        'label': labels,
        'tsne_1': embed_2d[:, 0],
        'tsne_2': embed_2d[:, 1]
    })

    csv_path = output_path / 'tsne_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"  Saved results to {csv_path}")

    # Save summary statistics
    summary_path = output_path / 'tsne_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("t-SNE Visualization Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples: {len(names)}\n")
        f.write(f"Embedding dimensions: {embed_2d.shape[1]}D -> 2D\n")
        f.write(f"Number of unique labels: {n_labels}\n\n")
        f.write("Label distribution:\n")
        for lbl in unique_labels:
            count = np.sum(labels == lbl)
            f.write(f"  Label {lbl}: {count} samples ({100*count/len(labels):.1f}%)\n")

    print(f"  Saved summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='t-SNE visualization pipeline for protein embeddings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to input embeddings.h5 file'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output directory for results'
    )

    parser.add_argument(
        '-p', '--perplexity',
        type=int,
        default=30,
        help='t-SNE perplexity parameter (5-50 recommended)'
    )

    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Run pipeline
    print("=" * 60)
    print("t-SNE Visualization Pipeline")
    print("=" * 60)

    # Load data
    embed, names = load_embeddings(args.input)

    # Extract labels
    labels = extract_labels(names)

    # Run t-SNE
    embed_2d = run_tsne(embed, perplexity=args.perplexity, random_state=args.seed)

    # Create visualization
    create_visualization(embed_2d, labels, names, args.output)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
