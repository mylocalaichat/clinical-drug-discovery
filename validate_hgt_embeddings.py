"""
Validate HGT embeddings to check if they're learning meaningful representations.

This script checks:
1. Embedding variance (are all embeddings too similar?)
2. Cosine similarity distribution
3. Per-dimension statistics
4. Node type separation
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


def load_embeddings(csv_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load embeddings from CSV."""
    print(f"Loading embeddings from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Parse embedding column (stored as string representation of list)
    embeddings = []
    for emb_str in df['embedding']:
        emb = json.loads(emb_str)
        embeddings.append(emb)

    embeddings = np.array(embeddings)
    print(f"✓ Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

    return df, embeddings


def analyze_variance(embeddings: np.ndarray, node_types: List[str] = None) -> Dict:
    """Analyze variance in embeddings."""
    print("\n" + "=" * 80)
    print("VARIANCE ANALYSIS")
    print("=" * 80)

    stats = {}

    # Overall statistics
    mean_emb = np.mean(embeddings, axis=0)
    std_emb = np.std(embeddings, axis=0)

    stats['mean_across_nodes'] = np.mean(embeddings, axis=1)
    stats['std_across_nodes'] = np.std(embeddings, axis=1)

    print(f"\nOverall Statistics:")
    print(f"  Mean of all values: {np.mean(embeddings):.6f}")
    print(f"  Std of all values: {np.std(embeddings):.6f}")
    print(f"  Min value: {np.min(embeddings):.6f}")
    print(f"  Max value: {np.max(embeddings):.6f}")

    # Per-dimension variance
    print(f"\nPer-Dimension Statistics:")
    print(f"  Mean variance across dimensions: {np.mean(np.var(embeddings, axis=0)):.6f}")
    print(f"  Std of variance across dimensions: {np.std(np.var(embeddings, axis=0)):.6f}")
    print(f"  Min variance: {np.min(np.var(embeddings, axis=0)):.6f}")
    print(f"  Max variance: {np.max(np.var(embeddings, axis=0)):.6f}")

    # Per-node statistics
    print(f"\nPer-Node Statistics:")
    print(f"  Mean of node means: {np.mean(stats['mean_across_nodes']):.6f}")
    print(f"  Std of node means: {np.std(stats['mean_across_nodes']):.6f}")
    print(f"  Mean of node stds: {np.mean(stats['std_across_nodes']):.6f}")
    print(f"  Std of node stds: {np.std(stats['std_across_nodes']):.6f}")

    # Check if embeddings are too similar
    node_mean_range = np.max(stats['mean_across_nodes']) - np.min(stats['mean_across_nodes'])
    print(f"\n  Range of node means: {node_mean_range:.6f}")

    if node_mean_range < 0.01:
        print("  ⚠️  WARNING: Very small range of node means - embeddings may be too similar!")

    if np.std(stats['std_across_nodes']) < 0.001:
        print("  ⚠️  WARNING: Very low variance in node stds - embeddings may be too similar!")

    return stats


def analyze_similarity(embeddings: np.ndarray, df: pd.DataFrame, sample_size: int = 1000) -> Dict:
    """Analyze cosine similarity between embeddings."""
    print("\n" + "=" * 80)
    print("COSINE SIMILARITY ANALYSIS")
    print("=" * 80)

    # Sample if too large
    if len(embeddings) > sample_size:
        print(f"\nSampling {sample_size} embeddings for similarity analysis...")
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[indices]
        sample_df = df.iloc[indices]
    else:
        sample_embeddings = embeddings
        sample_df = df

    # Compute cosine similarity
    print("Computing cosine similarity matrix...")
    cos_sim = cosine_similarity(sample_embeddings)

    # Get upper triangle (excluding diagonal)
    upper_triangle = np.triu(cos_sim, k=1)
    similarities = upper_triangle[upper_triangle != 0]

    print(f"\nCosine Similarity Statistics:")
    print(f"  Mean: {np.mean(similarities):.6f}")
    print(f"  Median: {np.median(similarities):.6f}")
    print(f"  Std: {np.std(similarities):.6f}")
    print(f"  Min: {np.min(similarities):.6f}")
    print(f"  Max: {np.max(similarities):.6f}")
    print(f"  25th percentile: {np.percentile(similarities, 25):.6f}")
    print(f"  75th percentile: {np.percentile(similarities, 75):.6f}")

    # Check if too similar
    if np.mean(similarities) > 0.95:
        print("\n  🚨 CRITICAL: Mean cosine similarity > 0.95 - embeddings are nearly identical!")
    elif np.mean(similarities) > 0.9:
        print("\n  ⚠️  WARNING: Mean cosine similarity > 0.9 - embeddings are very similar!")
    elif np.mean(similarities) > 0.7:
        print("\n  ⚠️  CAUTION: Mean cosine similarity > 0.7 - embeddings may be too similar")
    else:
        print("\n  ✓ Cosine similarity distribution looks reasonable")

    # Analyze by node type
    if 'node_type' in sample_df.columns:
        print("\nSimilarity Within vs Across Node Types:")
        node_types = sample_df['node_type'].unique()

        for node_type in node_types[:5]:  # Limit to first 5 types
            mask = (sample_df['node_type'] == node_type).values
            if mask.sum() < 2:
                continue

            # Within-type similarity
            type_indices = np.where(mask)[0]
            within_sim = []
            for i in range(len(type_indices)):
                for j in range(i+1, len(type_indices)):
                    within_sim.append(cos_sim[type_indices[i], type_indices[j]])

            # Across-type similarity
            across_sim = []
            for i in type_indices:
                for j in np.where(~mask)[0][:100]:  # Sample 100 from other types
                    across_sim.append(cos_sim[i, j])

            if within_sim and across_sim:
                print(f"  {node_type}:")
                print(f"    Within-type:  {np.mean(within_sim):.4f} ± {np.std(within_sim):.4f}")
                print(f"    Across-type:  {np.mean(across_sim):.4f} ± {np.std(across_sim):.4f}")
                print(f"    Separation:   {np.mean(within_sim) - np.mean(across_sim):.4f}")

    return {
        'mean': np.mean(similarities),
        'median': np.median(similarities),
        'std': np.std(similarities),
        'min': np.min(similarities),
        'max': np.max(similarities),
        'similarities': similarities
    }


def analyze_pca(embeddings: np.ndarray, df: pd.DataFrame) -> Dict:
    """Analyze PCA components to check information content."""
    print("\n" + "=" * 80)
    print("PCA ANALYSIS")
    print("=" * 80)

    print("\nFitting PCA...")
    pca = PCA(n_components=min(50, embeddings.shape[1], embeddings.shape[0]))
    pca.fit(embeddings)

    print(f"\nExplained Variance Ratio (first 10 components):")
    for i, ratio in enumerate(pca.explained_variance_ratio_[:10]):
        print(f"  PC{i+1}: {ratio:.6f} ({ratio*100:.2f}%)")

    # Cumulative variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components_90 = np.argmax(cumsum >= 0.90) + 1
    n_components_95 = np.argmax(cumsum >= 0.95) + 1

    print(f"\nCumulative Variance:")
    print(f"  90% variance explained by: {n_components_90} components")
    print(f"  95% variance explained by: {n_components_95} components")
    print(f"  Total components: {pca.n_components_}")

    # Check if information is concentrated in few components
    if n_components_90 < 5:
        print("\n  ⚠️  WARNING: 90% variance in < 5 components - embeddings may be collapsing!")
    elif n_components_90 < 10:
        print("\n  ⚠️  CAUTION: 90% variance in < 10 components - limited information content")
    else:
        print("\n  ✓ Information spread across multiple components")

    return {
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'n_components_90': n_components_90,
        'n_components_95': n_components_95
    }


def check_specific_nodes(embeddings: np.ndarray, df: pd.DataFrame, node_type: str = 'disease') -> None:
    """Check specific nodes to see if they're truly different."""
    print("\n" + "=" * 80)
    print(f"SPECIFIC NODE ANALYSIS ({node_type})")
    print("=" * 80)

    # Filter to specific node type
    type_mask = df['node_type'] == node_type
    if type_mask.sum() == 0:
        print(f"No nodes of type '{node_type}' found")
        return

    type_df = df[type_mask].reset_index(drop=True)
    type_embeddings = embeddings[type_mask]

    print(f"\nFound {len(type_embeddings)} {node_type} nodes")

    # Sample a few nodes
    sample_size = min(5, len(type_embeddings))
    sample_indices = np.random.choice(len(type_embeddings), sample_size, replace=False)

    print(f"\nSample {node_type} embeddings:")
    for idx in sample_indices:
        node_name = type_df.iloc[idx]['node_name']
        emb = type_embeddings[idx]
        print(f"  {node_name}:")
        print(f"    Mean: {np.mean(emb):.6f}, Std: {np.std(emb):.6f}")
        print(f"    First 5 dims: {emb[:5]}")

    # Check pairwise similarities
    if len(type_embeddings) >= 2:
        print(f"\nPairwise similarities between sampled {node_type} nodes:")
        for i in range(sample_size):
            for j in range(i+1, sample_size):
                sim = cosine_similarity(
                    type_embeddings[sample_indices[i]].reshape(1, -1),
                    type_embeddings[sample_indices[j]].reshape(1, -1)
                )[0, 0]
                name_i = type_df.iloc[sample_indices[i]]['node_name']
                name_j = type_df.iloc[sample_indices[j]]['node_name']
                print(f"  {name_i[:30]:<30} <-> {name_j[:30]:<30}: {sim:.6f}")


def plot_analysis(embeddings: np.ndarray, df: pd.DataFrame, output_dir: str = "./validation_plots"):
    """Create visualization plots."""
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Distribution of embedding values
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(embeddings.flatten(), bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Embedding Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of All Embedding Values')

    # 2. Per-dimension variance
    plt.subplot(1, 3, 2)
    variances = np.var(embeddings, axis=0)
    plt.plot(variances)
    plt.xlabel('Dimension')
    plt.ylabel('Variance')
    plt.title('Variance per Dimension')

    # 3. Mean embedding value per node
    plt.subplot(1, 3, 3)
    node_means = np.mean(embeddings, axis=1)
    plt.hist(node_means, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Mean Embedding Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Node Mean Values')

    plt.tight_layout()
    plot_path = Path(output_dir) / 'embedding_distributions.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()

    # 4. PCA visualization
    if len(embeddings) > 10:
        print("Computing PCA for visualization...")
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings[:min(1000, len(embeddings))])

        plt.figure(figsize=(10, 8))

        # Color by node type if available
        if 'node_type' in df.columns:
            node_types = df['node_type'][:len(embeddings_2d)].values
            unique_types = np.unique(node_types)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))

            for i, node_type in enumerate(unique_types[:10]):  # Limit to 10 types
                mask = node_types == node_type
                plt.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c=[colors[i]],
                    label=node_type,
                    alpha=0.6,
                    s=20
                )
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, s=20)

        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.title('PCA Visualization of Embeddings')

        plot_path = Path(output_dir) / 'pca_visualization.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {plot_path}")
        plt.close()


def validate_hgt_embeddings(csv_path: str, output_dir: str = "./validation_plots") -> Dict:
    """Complete validation pipeline."""
    print("=" * 80)
    print("HGT EMBEDDING VALIDATION")
    print("=" * 80)

    # Load embeddings
    df, embeddings = load_embeddings(csv_path)

    results = {}

    # 1. Variance analysis
    results['variance'] = analyze_variance(embeddings, df['node_type'].values if 'node_type' in df.columns else None)

    # 2. Similarity analysis
    results['similarity'] = analyze_similarity(embeddings, df, sample_size=1000)

    # 3. PCA analysis
    results['pca'] = analyze_pca(embeddings, df)

    # 4. Check specific nodes
    for node_type in ['disease', 'drug']:
        if node_type in df['node_type'].values:
            check_specific_nodes(embeddings, df, node_type)

    # 5. Generate plots
    plot_analysis(embeddings, df, output_dir)

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    issues = []

    if results['similarity']['mean'] > 0.95:
        issues.append("🚨 CRITICAL: Cosine similarity > 0.95 - embeddings are nearly identical")
    elif results['similarity']['mean'] > 0.9:
        issues.append("⚠️  WARNING: Cosine similarity > 0.9 - embeddings are very similar")

    if results['pca']['n_components_90'] < 5:
        issues.append("🚨 CRITICAL: 90% variance in < 5 PCA components - embeddings are collapsing")
    elif results['pca']['n_components_90'] < 10:
        issues.append("⚠️  WARNING: 90% variance in < 10 PCA components")

    if np.std(results['variance']['std_across_nodes']) < 0.001:
        issues.append("⚠️  WARNING: Very low variance in embeddings")

    if issues:
        print("\nISSUES DETECTED:")
        for issue in issues:
            print(f"  {issue}")
        print("\n❌ HGT embeddings are NOT learning meaningful representations")
        print("\nPossible causes:")
        print("  1. Model is not training properly (check loss convergence)")
        print("  2. Learning rate too high/low")
        print("  3. Not enough training epochs")
        print("  4. Graph structure issues (disconnected components)")
        print("  5. Initialization problems")
        print("  6. Edge sampling too aggressive")
    else:
        print("\n✓ HGT embeddings appear to be learning meaningful representations")

    return results


if __name__ == "__main__":
    import sys

    # Default to the main HGT embeddings file
    default_path = "./data/06_models/embeddings/hgt_embeddings.csv"

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = default_path

    print(f"Validating: {csv_path}\n")

    results = validate_hgt_embeddings(csv_path)
