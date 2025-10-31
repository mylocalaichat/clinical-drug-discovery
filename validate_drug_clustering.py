"""
Validate drug embedding quality by checking if drugs with similar mechanisms cluster together.

Examples of drug mechanism classes to validate:
- Beta blockers (e.g., Atenolol, Metoprolol, Propranolol)
- Statins (e.g., Atorvastatin, Simvastatin, Lovastatin)
- ACE inhibitors (e.g., Lisinopril, Enalapril, Ramipril)
- SSRIs (e.g., Fluoxetine, Sertraline, Citalopram)
- Proton pump inhibitors (e.g., Omeprazole, Pantoprazole, Lansoprazole)
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


# Known drug mechanism classes for validation
DRUG_CLASSES = {
    'Beta Blockers': [
        'Atenolol', 'Metoprolol', 'Propranolol', 'Carvedilol', 'Bisoprolol',
        'Labetalol', 'Nadolol', 'Timolol'
    ],
    'Statins': [
        'Atorvastatin', 'Simvastatin', 'Lovastatin', 'Pravastatin', 'Rosuvastatin',
        'Fluvastatin', 'Pitavastatin'
    ],
    'ACE Inhibitors': [
        'Lisinopril', 'Enalapril', 'Ramipril', 'Captopril', 'Benazepril',
        'Fosinopril', 'Quinapril', 'Perindopril'
    ],
    'SSRIs': [
        'Fluoxetine', 'Sertraline', 'Citalopram', 'Escitalopram', 'Paroxetine',
        'Fluvoxamine'
    ],
    'Proton Pump Inhibitors': [
        'Omeprazole', 'Pantoprazole', 'Lansoprazole', 'Esomeprazole', 'Rabeprazole',
        'Dexlansoprazole'
    ],
    'Benzodiazepines': [
        'Diazepam', 'Lorazepam', 'Alprazolam', 'Clonazepam', 'Temazepam',
        'Midazolam', 'Triazolam'
    ],
    'NSAIDs': [
        'Ibuprofen', 'Naproxen', 'Diclofenac', 'Indomethacin', 'Celecoxib',
        'Meloxicam', 'Piroxicam', 'Ketorolac'
    ],
    'Calcium Channel Blockers': [
        'Amlodipine', 'Nifedipine', 'Diltiazem', 'Verapamil', 'Felodipine',
        'Nicardipine', 'Isradipine'
    ],
}


def load_embeddings(embedding_csv: str) -> pd.DataFrame:
    """Load and parse embeddings from CSV."""
    print(f"Loading embeddings from: {embedding_csv}")

    df = pd.read_csv(embedding_csv)
    print(f"Loaded {len(df)} total embeddings")

    # Parse embedding column
    df['embedding'] = df['embedding'].apply(ast.literal_eval).apply(np.array)

    # Filter to drugs only
    drugs_df = df[df['node_type'] == 'drug'].copy()
    print(f"Found {len(drugs_df)} drug embeddings")

    return drugs_df


def find_drugs_in_embeddings(
    drugs_df: pd.DataFrame,
    drug_classes: Dict[str, List[str]]
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Find which drugs from our classes are in the embeddings."""

    # Create lookup dictionary (case-insensitive)
    drug_name_to_class = {}
    for class_name, drug_list in drug_classes.items():
        for drug in drug_list:
            drug_name_to_class[drug.lower()] = class_name

    # Find matching drugs
    found_drugs = []
    found_classes = {}

    for _, row in drugs_df.iterrows():
        drug_name = row['node_name']
        drug_name_lower = drug_name.lower()

        if drug_name_lower in drug_name_to_class:
            class_name = drug_name_to_class[drug_name_lower]
            found_drugs.append({
                'node_id': row['node_id'],
                'drug_name': drug_name,
                'class': class_name,
                'embedding': row['embedding']
            })

            if class_name not in found_classes:
                found_classes[class_name] = []
            found_classes[class_name].append(drug_name)

    found_df = pd.DataFrame(found_drugs)

    print(f"\nFound {len(found_df)} drugs from {len(found_classes)} mechanism classes:")
    for class_name, drugs in sorted(found_classes.items()):
        print(f"  - {class_name}: {len(drugs)} drugs - {', '.join(drugs[:3])}{'...' if len(drugs) > 3 else ''}")

    return found_df, found_classes


def compute_within_vs_between_class_similarity(found_df: pd.DataFrame) -> Dict:
    """
    Compute similarity metrics:
    - Within-class similarity (drugs in same mechanism class)
    - Between-class similarity (drugs in different mechanism classes)

    Good embeddings should have: within > between
    """

    embeddings = np.vstack(found_df['embedding'].values)
    classes = found_df['class'].values

    # Compute all pairwise similarities
    similarity_matrix = cosine_similarity(embeddings)

    within_class_sims = []
    between_class_sims = []

    for i in range(len(found_df)):
        for j in range(i + 1, len(found_df)):
            sim = similarity_matrix[i, j]

            if classes[i] == classes[j]:
                within_class_sims.append(sim)
            else:
                between_class_sims.append(sim)

    results = {
        'within_class_mean': np.mean(within_class_sims) if within_class_sims else 0,
        'within_class_std': np.std(within_class_sims) if within_class_sims else 0,
        'between_class_mean': np.mean(between_class_sims) if between_class_sims else 0,
        'between_class_std': np.std(between_class_sims) if between_class_sims else 0,
        'within_count': len(within_class_sims),
        'between_count': len(between_class_sims),
    }

    results['separation_score'] = results['within_class_mean'] - results['between_class_mean']

    print("\n" + "="*80)
    print("CLUSTERING QUALITY METRICS")
    print("="*80)
    print(f"Within-class similarity:  {results['within_class_mean']:.4f} ± {results['within_class_std']:.4f}")
    print(f"Between-class similarity: {results['between_class_mean']:.4f} ± {results['between_class_std']:.4f}")
    print(f"\nSeparation score: {results['separation_score']:.4f}")
    print(f"  (Positive = good, drugs in same class are more similar)")
    print(f"\nComparisons: {results['within_count']} within-class, {results['between_count']} between-class")

    return results


def compute_per_class_cohesion(found_df: pd.DataFrame) -> pd.DataFrame:
    """Compute cohesion score for each drug class."""

    embeddings = np.vstack(found_df['embedding'].values)
    similarity_matrix = cosine_similarity(embeddings)

    class_scores = []

    for class_name in found_df['class'].unique():
        class_mask = found_df['class'] == class_name
        class_indices = np.where(class_mask)[0]

        if len(class_indices) < 2:
            continue

        # Average similarity within this class
        within_sims = []
        for i in class_indices:
            for j in class_indices:
                if i < j:
                    within_sims.append(similarity_matrix[i, j])

        # Average similarity to other classes
        other_indices = np.where(~class_mask)[0]
        between_sims = []
        for i in class_indices:
            for j in other_indices:
                between_sims.append(similarity_matrix[i, j])

        class_scores.append({
            'class': class_name,
            'num_drugs': len(class_indices),
            'within_similarity': np.mean(within_sims),
            'between_similarity': np.mean(between_sims) if between_sims else 0,
            'cohesion_score': np.mean(within_sims) - (np.mean(between_sims) if between_sims else 0)
        })

    scores_df = pd.DataFrame(class_scores).sort_values('cohesion_score', ascending=False)

    print("\n" + "="*80)
    print("PER-CLASS COHESION SCORES")
    print("="*80)
    print(f"{'Class':<30} {'Drugs':>6} {'Within':>8} {'Between':>8} {'Cohesion':>8}")
    print("-"*80)

    for _, row in scores_df.iterrows():
        print(f"{row['class']:<30} {row['num_drugs']:>6} "
              f"{row['within_similarity']:>8.4f} {row['between_similarity']:>8.4f} "
              f"{row['cohesion_score']:>8.4f}")

    return scores_df


def visualize_drug_clustering(found_df: pd.DataFrame, output_dir: str, model_name: str):
    """Create t-SNE visualization of drug embeddings colored by mechanism class."""

    print("\n" + "="*80)
    print("CREATING t-SNE VISUALIZATION")
    print("="*80)

    embeddings = np.vstack(found_df['embedding'].values)

    # t-SNE dimensionality reduction
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(found_df) - 1))
    tsne_coords = tsne.fit_transform(embeddings)

    # Create visualization DataFrame
    viz_df = found_df.copy()
    viz_df['tsne_1'] = tsne_coords[:, 0]
    viz_df['tsne_2'] = tsne_coords[:, 1]

    # Create interactive plot
    fig = px.scatter(
        viz_df,
        x='tsne_1',
        y='tsne_2',
        color='class',
        hover_data=['drug_name'],
        title=f'Drug Mechanism Clustering ({model_name})',
        labels={'tsne_1': 't-SNE Component 1', 'tsne_2': 't-SNE Component 2'},
        width=1000,
        height=800
    )

    # Update layout
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        font=dict(size=12),
        legend=dict(title='Drug Mechanism Class', font=dict(size=10))
    )

    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plot_file = output_path / f"{model_name}_drug_mechanism_clustering.html"
    fig.write_html(str(plot_file))

    print(f"✓ Saved visualization to: {plot_file}")
    print(f"  Open in browser: file://{plot_file.absolute()}")

    return fig


def find_similar_drugs(
    drug_name: str,
    drugs_df: pd.DataFrame,
    top_k: int = 10
) -> pd.DataFrame:
    """Find most similar drugs to a given drug."""

    # Find the query drug
    query_mask = drugs_df['node_name'].str.lower() == drug_name.lower()

    if not query_mask.any():
        print(f"Drug '{drug_name}' not found in embeddings")
        return None

    query_drug = drugs_df[query_mask].iloc[0]
    query_emb = query_drug['embedding'].reshape(1, -1)

    # Compute similarities
    all_embs = np.vstack(drugs_df['embedding'].values)
    similarities = cosine_similarity(query_emb, all_embs)[0]

    # Get top-k (excluding self)
    drugs_df_copy = drugs_df.copy()
    drugs_df_copy['similarity'] = similarities

    # Exclude the query drug itself
    results = drugs_df_copy[drugs_df_copy['node_name'] != query_drug['node_name']]
    results = results.sort_values('similarity', ascending=False).head(top_k)

    return results[['drug_name', 'similarity']]


def main():
    """Run validation pipeline."""

    print("="*80)
    print("DRUG MECHANISM CLUSTERING VALIDATION")
    print("="*80)

    # Check which embedding files exist
    gnn_path = "data/06_models/embeddings/gnn_embeddings.csv"
    hgt_path = "data/06_models/embeddings/hgt_embeddings.csv"

    embeddings_to_compare = []
    if Path(gnn_path).exists():
        embeddings_to_compare.append(("GraphSAGE", gnn_path))
    if Path(hgt_path).exists():
        embeddings_to_compare.append(("HGT", hgt_path))

    if not embeddings_to_compare:
        print("❌ No embedding files found!")
        print(f"   Expected: {gnn_path} or {hgt_path}")
        return

    comparison_results = []

    for model_name, embedding_path in embeddings_to_compare:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {model_name}")
        print(f"{'='*80}")

        # Load embeddings
        drugs_df = load_embeddings(embedding_path)

        # Find drugs from known mechanism classes
        found_df, found_classes = find_drugs_in_embeddings(drugs_df, DRUG_CLASSES)

        if len(found_df) == 0:
            print(f"⚠️  No drugs from mechanism classes found in {model_name} embeddings")
            continue

        # Compute clustering metrics
        metrics = compute_within_vs_between_class_similarity(found_df)

        # Per-class cohesion
        class_scores = compute_per_class_cohesion(found_df)

        # Visualize
        visualize_drug_clustering(
            found_df,
            output_dir="data/06_models/embeddings/validation",
            model_name=model_name.lower()
        )

        # Store results for comparison
        comparison_results.append({
            'model': model_name,
            'separation_score': metrics['separation_score'],
            'within_similarity': metrics['within_class_mean'],
            'between_similarity': metrics['between_class_mean'],
            'num_drugs': len(found_df),
            'num_classes': len(found_classes)
        })

        # Example: Find similar drugs to Atorvastatin (a statin)
        print("\n" + "="*80)
        print("EXAMPLE: Drugs similar to Atorvastatin (a statin)")
        print("="*80)

        similar = find_similar_drugs("Atorvastatin", drugs_df, top_k=10)
        if similar is not None:
            print(similar.to_string(index=False))

    # Final comparison
    if len(comparison_results) > 1:
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)

        comp_df = pd.DataFrame(comparison_results)
        print(comp_df.to_string(index=False))

        print("\nInterpretation:")
        print("  - Higher separation_score = better clustering by mechanism")
        print("  - Higher within_similarity = drugs in same class are more similar")
        print("  - Lower between_similarity = drugs in different classes are more different")

        best_model = comp_df.loc[comp_df['separation_score'].idxmax(), 'model']
        print(f"\n🏆 Best clustering: {best_model}")


if __name__ == "__main__":
    main()
