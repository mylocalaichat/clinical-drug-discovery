#!/usr/bin/env python3
"""
Visualize drug and disease embeddings using t-SNE or UMAP.

This script loads the saved embeddings and creates 2D visualizations
to understand what the model learned.

Usage:
    python scripts/visualize_embeddings.py --node-type drug --method tsne
    python scripts/visualize_embeddings.py --node-type disease --method umap
    python scripts/visualize_embeddings.py --node-type drug --highlight "Topiramate,Tamoxifen,Paclitaxel"
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_embeddings(node_type: str, embeddings_dir: Path):
    """Load embeddings for a specific node type."""
    embeddings_path = embeddings_dir / f"embeddings_{node_type}.csv"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {embeddings_path}\n"
            f"Please run the offlabel_model_embeddings asset first:\n"
            f"  dagster asset materialize --select offlabel_model_embeddings"
        )

    logger.info(f"Loading embeddings from {embeddings_path}")
    df = pd.read_csv(embeddings_path)

    # Separate metadata from embeddings
    node_ids = df['node_id'].values
    node_names = df['node_name'].values
    embeddings = df.iloc[:, 2:].values  # Skip node_id and node_name columns

    logger.info(f"Loaded {len(node_ids)} nodes with {embeddings.shape[1]}-dimensional embeddings")

    return embeddings, node_ids, node_names


def reduce_dimensions(embeddings: np.ndarray, method: str = 'tsne', random_state: int = 42):
    """Reduce embeddings to 2D using t-SNE or UMAP."""
    logger.info(f"Reducing dimensions using {method.upper()}...")

    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=30, n_iter=1000)
        embeddings_2d = reducer.fit_transform(embeddings)

    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
            embeddings_2d = reducer.fit_transform(embeddings)
        except ImportError:
            logger.error("UMAP not installed. Install with: pip install umap-learn")
            logger.info("Falling back to t-SNE...")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=random_state, perplexity=30)
            embeddings_2d = reducer.fit_transform(embeddings)

    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=random_state)
        embeddings_2d = reducer.fit_transform(embeddings)
        logger.info(f"  Explained variance: {reducer.explained_variance_ratio_.sum():.2%}")

    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne', 'umap', or 'pca'")

    logger.info(f"Reduced to 2D: {embeddings_2d.shape}")
    return embeddings_2d


def visualize_embeddings(
    embeddings_2d: np.ndarray,
    node_names: np.ndarray,
    node_ids: np.ndarray,
    node_type: str,
    method: str,
    highlight_names: list = None,
    output_path: Path = None,
    max_labels: int = 20
):
    """Create 2D scatter plot of embeddings."""
    logger.info("Creating visualization...")

    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 12))

    # Determine which nodes to highlight
    highlight_mask = np.zeros(len(node_names), dtype=bool)
    if highlight_names:
        for name in highlight_names:
            mask = np.array([name.lower() in str(n).lower() for n in node_names])
            highlight_mask |= mask

    # Plot non-highlighted points
    plt.scatter(
        embeddings_2d[~highlight_mask, 0],
        embeddings_2d[~highlight_mask, 1],
        alpha=0.3,
        s=20,
        c='lightgray',
        label=f'Other {node_type}s'
    )

    # Plot highlighted points
    if highlight_mask.any():
        plt.scatter(
            embeddings_2d[highlight_mask, 0],
            embeddings_2d[highlight_mask, 1],
            alpha=0.8,
            s=100,
            c='red',
            edgecolors='black',
            linewidths=1.5,
            label=f'Highlighted {node_type}s'
        )

        # Add labels for highlighted points
        for i, (x, y, name) in enumerate(zip(
            embeddings_2d[highlight_mask, 0],
            embeddings_2d[highlight_mask, 1],
            node_names[highlight_mask]
        )):
            plt.annotate(
                name,
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )
    else:
        # If no highlights, label a random sample
        if len(node_names) > max_labels:
            indices = np.random.choice(len(node_names), max_labels, replace=False)
        else:
            indices = np.arange(len(node_names))

        for idx in indices:
            plt.annotate(
                node_names[idx],
                (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                xytext=(3, 3),
                textcoords='offset points',
                fontsize=7,
                alpha=0.6
            )

    plt.xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    plt.title(
        f'{node_type.capitalize()} Embeddings ({method.upper()} Projection)\n'
        f'Total: {len(node_names)} {node_type}s',
        fontsize=14,
        fontweight='bold'
    )
    plt.legend(fontsize=10)
    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def find_similar_nodes(
    target_name: str,
    embeddings: np.ndarray,
    node_names: np.ndarray,
    node_ids: np.ndarray,
    top_k: int = 10
):
    """Find most similar nodes to a target node using cosine similarity."""
    from sklearn.metrics.pairwise import cosine_similarity

    # Find target node
    target_mask = np.array([target_name.lower() in str(n).lower() for n in node_names])

    if not target_mask.any():
        logger.warning(f"Node '{target_name}' not found")
        return

    target_idx = np.where(target_mask)[0][0]
    target_embedding = embeddings[target_idx].reshape(1, -1)

    # Compute similarities
    similarities = cosine_similarity(target_embedding, embeddings)[0]

    # Get top k (excluding self)
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]

    logger.info(f"\nTop {top_k} most similar nodes to '{node_names[target_idx]}':")
    logger.info("=" * 80)
    for rank, idx in enumerate(top_indices, 1):
        logger.info(f"{rank:2d}. {node_names[idx]:50s} | Similarity: {similarities[idx]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize node embeddings")
    parser.add_argument(
        '--node-type',
        type=str,
        default='drug',
        help='Node type to visualize (drug, disease, protein, etc.)'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='tsne',
        choices=['tsne', 'umap', 'pca'],
        help='Dimensionality reduction method'
    )
    parser.add_argument(
        '--highlight',
        type=str,
        default=None,
        help='Comma-separated list of node names to highlight (e.g., "Topiramate,Tamoxifen")'
    )
    parser.add_argument(
        '--similar-to',
        type=str,
        default=None,
        help='Find nodes similar to this node (e.g., "Topiramate")'
    )
    parser.add_argument(
        '--max-labels',
        type=int,
        default=20,
        help='Maximum number of labels to show (if not highlighting)'
    )
    parser.add_argument(
        '--embeddings-dir',
        type=Path,
        default=Path('data/06_models/offlabel'),
        help='Directory containing embeddings files'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output file path (if not specified, will show plot)'
    )

    args = parser.parse_args()

    # Load embeddings
    embeddings, node_ids, node_names = load_embeddings(args.node_type, args.embeddings_dir)

    # Find similar nodes if requested
    if args.similar_to:
        find_similar_nodes(args.similar_to, embeddings, node_names, node_ids)

    # Reduce dimensions
    embeddings_2d = reduce_dimensions(embeddings, method=args.method)

    # Parse highlight names
    highlight_names = None
    if args.highlight:
        highlight_names = [name.strip() for name in args.highlight.split(',')]
        logger.info(f"Highlighting: {highlight_names}")

    # Set default output path if not specified
    output_path = args.output
    if output_path is None:
        output_dir = Path('data/07_model_output/offlabel/visualizations')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{args.node_type}_embeddings_{args.method}.png'

    # Visualize
    visualize_embeddings(
        embeddings_2d,
        node_names,
        node_ids,
        args.node_type,
        args.method,
        highlight_names,
        output_path,
        args.max_labels
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
