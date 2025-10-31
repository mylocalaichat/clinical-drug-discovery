"""
Validate disease embedding quality by checking if diseases sharing pathways/genes cluster together.

This is the key validation for off-label drug discovery:
- Diseases with similar mechanisms should have similar embeddings
- Example: Different cancers sharing oncogenic pathways
- Example: Autoimmune diseases sharing inflammatory pathways
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import ast
from collections import defaultdict


def load_embeddings(embedding_csv: str) -> pd.DataFrame:
    """Load and parse embeddings from CSV."""
    print(f"Loading embeddings from: {embedding_csv}")

    df = pd.read_csv(embedding_csv)
    print(f"Loaded {len(df)} total embeddings")

    # Parse embedding column
    df['embedding'] = df['embedding'].apply(ast.literal_eval).apply(np.array)

    return df


def load_knowledge_graph(kg_csv: str) -> pd.DataFrame:
    """Load knowledge graph to find disease-gene/pathway relationships."""
    print(f"\nLoading knowledge graph from: {kg_csv}")

    kg_df = pd.read_csv(kg_csv, low_memory=False)
    print(f"Loaded {len(kg_df):,} edges")

    return kg_df


def build_disease_gene_map(kg_df: pd.DataFrame, diseases_df: pd.DataFrame) -> Dict[str, set]:
    """
    Build mapping of disease -> genes/pathways.

    Uses 'associated with' and 'expression present' edges.
    """
    print("\nBuilding disease-gene/pathway associations...")

    # Get disease node IDs
    disease_ids = set(diseases_df['node_id'].values)

    # Filter to disease-gene edges
    disease_gene_edges = kg_df[
        (kg_df['x_type'] == 'disease') &
        (kg_df['y_type'].isin(['gene/protein', 'pathway', 'biological_process'])) &
        (kg_df['relation'].isin(['associated with', 'expression present', 'linked to']))
    ]

    print(f"Found {len(disease_gene_edges):,} disease-gene/pathway edges")

    # Build mapping
    disease_to_features = defaultdict(set)

    for _, row in disease_gene_edges.iterrows():
        disease_id = str(row['x_id'])
        feature_id = str(row['y_id'])

        if disease_id in disease_ids:
            disease_to_features[disease_id].add(feature_id)

    print(f"Mapped {len(disease_to_features)} diseases to genes/pathways")

    # Print stats
    feature_counts = [len(features) for features in disease_to_features.values()]
    if feature_counts:
        print(f"  Mean features per disease: {np.mean(feature_counts):.1f}")
        print(f"  Median features per disease: {np.median(feature_counts):.1f}")

    return disease_to_features


def compute_disease_jaccard_similarity(
    disease_to_features: Dict[str, set],
    disease_ids: List[str]
) -> np.ndarray:
    """
    Compute Jaccard similarity matrix based on shared genes/pathways.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    """
    print("\nComputing Jaccard similarity based on shared features...")

    n = len(disease_ids)
    jaccard_matrix = np.zeros((n, n))

    for i, disease_i in enumerate(disease_ids):
        features_i = disease_to_features.get(disease_i, set())

        for j, disease_j in enumerate(disease_ids):
            if i > j:  # Use symmetry
                jaccard_matrix[i, j] = jaccard_matrix[j, i]
                continue

            features_j = disease_to_features.get(disease_j, set())

            if len(features_i) == 0 or len(features_j) == 0:
                jaccard_matrix[i, j] = 0.0
            else:
                intersection = len(features_i & features_j)
                union = len(features_i | features_j)
                jaccard_matrix[i, j] = intersection / union if union > 0 else 0.0

    # Print distribution
    upper_triangle = jaccard_matrix[np.triu_indices_from(jaccard_matrix, k=1)]
    print(f"Jaccard similarity distribution:")
    print(f"  Mean: {np.mean(upper_triangle):.4f}")
    print(f"  Median: {np.median(upper_triangle):.4f}")
    print(f"  Max: {np.max(upper_triangle):.4f}")
    print(f"  Pairs with similarity > 0.1: {np.sum(upper_triangle > 0.1):,} ({100*np.mean(upper_triangle > 0.1):.1f}%)")
    print(f"  Pairs with similarity > 0.3: {np.sum(upper_triangle > 0.3):,} ({100*np.mean(upper_triangle > 0.3):.1f}%)")

    return jaccard_matrix


def validate_embedding_preserves_mechanism_similarity(
    diseases_df: pd.DataFrame,
    jaccard_matrix: np.ndarray,
    disease_ids: List[str]
) -> Dict:
    """
    Validate that embedding similarity correlates with mechanism similarity.

    Good embeddings: high cosine similarity when Jaccard similarity is high.
    """
    print("\n" + "="*80)
    print("CORRELATION: Embedding Similarity vs. Mechanism Similarity")
    print("="*80)

    # Compute embedding cosine similarity
    embeddings = np.vstack(diseases_df['embedding'].values)
    cosine_matrix = cosine_similarity(embeddings)

    # Get upper triangles
    upper_idx = np.triu_indices_from(jaccard_matrix, k=1)
    jaccard_values = jaccard_matrix[upper_idx]
    cosine_values = cosine_matrix[upper_idx]

    # Compute correlation
    from scipy.stats import spearmanr, pearsonr

    spearman_corr, spearman_p = spearmanr(jaccard_values, cosine_values)
    pearson_corr, pearson_p = pearsonr(jaccard_values, cosine_values)

    print(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.2e})")
    print(f"Pearson correlation:  {pearson_corr:.4f} (p={pearson_p:.2e})")

    # Binned analysis
    print("\nEmbedding similarity by mechanism similarity bins:")
    print(f"{'Jaccard Range':<20} {'Count':>8} {'Mean Cosine':>12} {'Std Cosine':>12}")
    print("-"*60)

    bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0)]
    for low, high in bins:
        mask = (jaccard_values >= low) & (jaccard_values < high)
        if np.sum(mask) > 0:
            mean_cosine = np.mean(cosine_values[mask])
            std_cosine = np.std(cosine_values[mask])
            print(f"{f'[{low:.1f}, {high:.1f})':<20} {np.sum(mask):>8,} {mean_cosine:>12.4f} {std_cosine:>12.4f}")

    # Find best and worst preserved similarities
    print("\n" + "="*80)
    print("EXAMPLES: Well-Preserved Disease Pairs")
    print("="*80)

    # Find disease pairs with high Jaccard and high cosine
    high_jaccard_mask = jaccard_values > 0.2
    if np.sum(high_jaccard_mask) > 0:
        high_jaccard_idx = np.where(high_jaccard_mask)[0]

        # Sort by cosine similarity
        sorted_idx = high_jaccard_idx[np.argsort(-cosine_values[high_jaccard_mask])]

        print("Top 10 disease pairs with shared mechanisms and similar embeddings:")
        print(f"{'Disease 1':<30} {'Disease 2':<30} {'Jaccard':>8} {'Cosine':>8}")
        print("-"*80)

        for idx in sorted_idx[:10]:
            i, j = upper_idx[0][idx], upper_idx[1][idx]
            disease_1 = diseases_df.iloc[i]['node_name']
            disease_2 = diseases_df.iloc[j]['node_name']
            print(f"{disease_1[:28]:<30} {disease_2[:28]:<30} "
                  f"{jaccard_values[idx]:>8.4f} {cosine_values[idx]:>8.4f}")

    # Find poorly preserved pairs
    print("\n" + "="*80)
    print("EXAMPLES: Poorly-Preserved Disease Pairs")
    print("="*80)

    if np.sum(high_jaccard_mask) > 0:
        # Sort by cosine similarity (ascending)
        sorted_idx = high_jaccard_idx[np.argsort(cosine_values[high_jaccard_mask])]

        print("Disease pairs with shared mechanisms but dissimilar embeddings:")
        print(f"{'Disease 1':<30} {'Disease 2':<30} {'Jaccard':>8} {'Cosine':>8}")
        print("-"*80)

        for idx in sorted_idx[:10]:
            i, j = upper_idx[0][idx], upper_idx[1][idx]
            disease_1 = diseases_df.iloc[i]['node_name']
            disease_2 = diseases_df.iloc[j]['node_name']
            print(f"{disease_1[:28]:<30} {disease_2[:28]:<30} "
                  f"{jaccard_values[idx]:>8.4f} {cosine_values[idx]:>8.4f}")

    return {
        'spearman_correlation': spearman_corr,
        'pearson_correlation': pearson_corr,
        'spearman_pvalue': spearman_p,
        'pearson_pvalue': pearson_p,
    }


def visualize_disease_similarity_correlation(
    jaccard_matrix: np.ndarray,
    diseases_df: pd.DataFrame,
    output_dir: str,
    model_name: str
):
    """Create scatter plot showing embedding similarity vs. mechanism similarity."""

    embeddings = np.vstack(diseases_df['embedding'].values)
    cosine_matrix = cosine_similarity(embeddings)

    # Sample for visualization (too many points otherwise)
    upper_idx = np.triu_indices_from(jaccard_matrix, k=1)
    jaccard_values = jaccard_matrix[upper_idx]
    cosine_values = cosine_matrix[upper_idx]

    # Sample if too many points
    if len(jaccard_values) > 10000:
        sample_idx = np.random.choice(len(jaccard_values), 10000, replace=False)
        jaccard_values = jaccard_values[sample_idx]
        cosine_values = cosine_values[sample_idx]

    # Create scatter plot
    fig = px.scatter(
        x=jaccard_values,
        y=cosine_values,
        opacity=0.3,
        labels={'x': 'Mechanism Similarity (Jaccard)', 'y': 'Embedding Similarity (Cosine)'},
        title=f'Disease Embedding Quality ({model_name})<br>Correlation between Mechanism and Embedding Similarity',
        width=800,
        height=600
    )

    # Add trend line
    z = np.polyfit(jaccard_values, cosine_values, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(0, 1, 100)
    y_trend = p(x_trend)

    fig.add_trace(go.Scatter(
        x=x_trend, y=y_trend,
        mode='lines',
        name='Trend Line',
        line=dict(color='red', width=2)
    ))

    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plot_file = output_path / f"{model_name}_disease_similarity_correlation.html"
    fig.write_html(str(plot_file))

    print(f"\n✓ Saved correlation plot to: {plot_file}")
    print(f"  Open in browser: file://{plot_file.absolute()}")


def main():
    """Run disease clustering validation."""

    print("="*80)
    print("DISEASE MECHANISM CLUSTERING VALIDATION")
    print("For Off-Label Drug Discovery")
    print("="*80)

    # Load knowledge graph
    kg_csv = "data/01_raw/primekg/kg.csv"
    if not Path(kg_csv).exists():
        print(f"❌ Knowledge graph not found: {kg_csv}")
        return

    kg_df = load_knowledge_graph(kg_csv)

    # Check which embedding files exist
    gnn_path = "data/06_models/embeddings/gnn_embeddings.csv"
    hgt_path = "data/06_models/embeddings/hgt_embeddings.csv"

    embeddings_to_compare = []
    if Path(gnn_path).exists():
        embeddings_to_compare.append(("graphsage", gnn_path))
    if Path(hgt_path).exists():
        embeddings_to_compare.append(("hgt", hgt_path))

    if not embeddings_to_compare:
        print("❌ No embedding files found!")
        return

    comparison_results = []

    for model_name, embedding_path in embeddings_to_compare:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {model_name.upper()}")
        print(f"{'='*80}")

        # Load embeddings
        all_embeddings_df = load_embeddings(embedding_path)

        # Filter to diseases
        diseases_df = all_embeddings_df[all_embeddings_df['node_type'] == 'disease'].copy()
        print(f"\nFound {len(diseases_df)} disease embeddings")

        if len(diseases_df) < 10:
            print(f"⚠️  Too few diseases in {model_name} embeddings")
            continue

        # Sample diseases if too many (for computational efficiency)
        if len(diseases_df) > 500:
            print(f"Sampling 500 diseases for analysis...")
            diseases_df = diseases_df.sample(n=500, random_state=42)

        disease_ids = diseases_df['node_id'].astype(str).tolist()

        # Build disease-gene/pathway map
        disease_to_features = build_disease_gene_map(kg_df, diseases_df)

        # Compute ground truth similarity (Jaccard)
        jaccard_matrix = compute_disease_jaccard_similarity(disease_to_features, disease_ids)

        # Validate correlation
        metrics = validate_embedding_preserves_mechanism_similarity(
            diseases_df, jaccard_matrix, disease_ids
        )

        # Visualize
        visualize_disease_similarity_correlation(
            jaccard_matrix, diseases_df,
            output_dir="data/06_models/embeddings/validation",
            model_name=model_name
        )

        comparison_results.append({
            'model': model_name.upper(),
            'spearman_corr': metrics['spearman_correlation'],
            'pearson_corr': metrics['pearson_correlation'],
            'num_diseases': len(diseases_df),
        })

    # Final comparison
    if len(comparison_results) > 1:
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)

        comp_df = pd.DataFrame(comparison_results)
        print(comp_df.to_string(index=False))

        print("\nInterpretation:")
        print("  - Higher correlation = embeddings better preserve disease mechanisms")
        print("  - Good embeddings: diseases sharing pathways have similar embeddings")
        print("  - This enables off-label discovery: find diseases similar to known indication")

        best_model = comp_df.loc[comp_df['spearman_corr'].idxmax(), 'model']
        print(f"\n🏆 Best disease clustering: {best_model}")
        print(f"    → Use this for off-label drug discovery!")


if __name__ == "__main__":
    main()
