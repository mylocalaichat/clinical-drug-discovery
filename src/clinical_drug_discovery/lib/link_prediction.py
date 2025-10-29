"""
Link prediction module for drug-disease relationship prediction.

This module generates all possible drug-disease pairs (Cartesian product),
removes training pairs to find missing edges, and ranks predictions.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


def generate_all_drug_disease_pairs(
    drugs_df: pd.DataFrame,
    diseases_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate ALL possible drug-disease combinations (Cartesian product).

    This is the core of the link prediction approach: create every possible pairing and then
    score them to find missing edges (potential new treatments).

    Args:
        drugs_df: DataFrame with drug information (must have 'node_id' column)
        diseases_df: DataFrame with disease information (must have 'node_id' column)

    Returns:
        DataFrame with all drug-disease pairs
    """
    print("\n=== Generating Cartesian Product ===")

    drugs_list = drugs_df['node_id'].tolist()
    diseases_list = diseases_df['node_id'].tolist()

    n_drugs = len(drugs_list)
    n_diseases = len(diseases_list)
    total_pairs = n_drugs * n_diseases

    print(f"Drugs: {n_drugs:,}")
    print(f"Diseases: {n_diseases:,}")
    print(f"Total possible pairs: {total_pairs:,}")

    # Generate all combinations
    all_pairs = []

    for disease_id in tqdm(diseases_list, desc="Generating pairs"):
        for drug_id in drugs_list:
            all_pairs.append({
                'drug_id': drug_id,
                'disease_id': disease_id,
            })

    pairs_df = pd.DataFrame(all_pairs)

    print(f"✓ Generated {len(pairs_df):,} drug-disease pairs")

    return pairs_df


def remove_training_pairs(
    all_pairs_df: pd.DataFrame,
    known_pairs_df: pd.DataFrame,
    drug_col: str = 'drug_id',
    disease_col: str = 'disease_id',
) -> pd.DataFrame:
    """
    Remove known training pairs from all pairs to find missing edges.

    This is the key to link prediction: the remaining pairs after removal
    are the "missing edges" - drug-disease relationships that don't exist
    in the knowledge graph but might be valid treatments.

    Args:
        all_pairs_df: DataFrame with all possible pairs
        known_pairs_df: DataFrame with known (training) pairs
        drug_col: Name of drug ID column
        disease_col: Name of disease ID column

    Returns:
        DataFrame with unknown pairs only
    """
    print("\n=== Finding Missing Edges ===")

    # Create set of known pairs for fast lookup
    known_set = set(zip(
        known_pairs_df[drug_col],
        known_pairs_df[disease_col]
    ))

    print(f"Known pairs (to remove): {len(known_set):,}")
    print(f"All pairs (before): {len(all_pairs_df):,}")

    # Filter out known pairs
    is_unknown = ~all_pairs_df.apply(
        lambda row: (row[drug_col], row[disease_col]) in known_set,
        axis=1
    )

    unknown_pairs_df = all_pairs_df[is_unknown].copy()

    print(f"Unknown pairs (after): {len(unknown_pairs_df):,}")
    print(f"Removed: {len(all_pairs_df) - len(unknown_pairs_df):,} known pairs")

    return unknown_pairs_df


def predict_with_ensemble(
    pairs_df: pd.DataFrame,
    models: List,
    embeddings: Dict[str, np.ndarray],
    drug_col: str = 'drug_id',
    disease_col: str = 'disease_id',
    batch_size: int = 10000,
) -> pd.DataFrame:
    """
    Make predictions using ensemble of models.

    Averages predictions from multiple models (XGBoost, LightGBM, RandomForest)
    to get final probability scores.

    Args:
        pairs_df: DataFrame with drug-disease pairs
        models: List of trained models
        embeddings: Dictionary of node embeddings
        drug_col: Name of drug ID column
        disease_col: Name of disease ID column
        batch_size: Number of pairs to process at once

    Returns:
        DataFrame with prediction scores
    """
    print("\n=== Making Predictions with Ensemble ===")

    results = []

    # Process in batches
    n_batches = (len(pairs_df) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(n_batches), desc="Predicting"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(pairs_df))
        batch_df = pairs_df.iloc[start_idx:end_idx].copy()

        # Get embeddings for this batch
        drug_embeddings = np.stack([
            embeddings.get(drug_id, np.zeros(512))
            for drug_id in batch_df[drug_col]
        ])

        disease_embeddings = np.stack([
            embeddings.get(disease_id, np.zeros(512))
            for disease_id in batch_df[disease_col]
        ])

        # Concatenate features: [512 drug] + [512 disease] = 1024
        X = np.concatenate([drug_embeddings, disease_embeddings], axis=1)

        # Get predictions from each model
        all_predictions = []
        for model in models:
            # predict_proba returns shape (n_samples, 3) for 3 classes
            probs = model.predict_proba(X)
            all_predictions.append(probs)

        # Average predictions across models
        ensemble_probs = np.mean(all_predictions, axis=0)

        # Extract probabilities for each class
        batch_df['not_treat_score'] = ensemble_probs[:, 0]  # P(y=0)
        batch_df['treat_score'] = ensemble_probs[:, 1]      # P(y=1) ← KEY!
        batch_df['unknown_score'] = ensemble_probs[:, 2]    # P(y=2)

        results.append(batch_df)

    # Combine all batches
    final_df = pd.concat(results, ignore_index=True)

    print(f"✓ Generated predictions for {len(final_df):,} pairs")
    print("\nScore statistics:")
    print(f"  Treat score: mean={final_df['treat_score'].mean():.4f}, "
          f"std={final_df['treat_score'].std():.4f}, "
          f"max={final_df['treat_score'].max():.4f}")

    return final_df


def rank_predictions(
    predictions_df: pd.DataFrame,
    score_column: str = 'treat_score',
) -> pd.DataFrame:
    """
    Rank predictions by treatment score.

    Args:
        predictions_df: DataFrame with prediction scores
        score_column: Name of the score column to rank by

    Returns:
        DataFrame sorted and ranked by score
    """
    print("\n=== Ranking Predictions ===")

    # Sort by score (descending)
    ranked_df = predictions_df.sort_values(score_column, ascending=False).copy()

    # Add rank
    ranked_df['rank'] = range(1, len(ranked_df) + 1)

    # Add quantile rank (0 to 1)
    ranked_df['quantile_rank'] = ranked_df['rank'] / len(ranked_df)

    print(f"✓ Ranked {len(ranked_df):,} predictions")
    print("\nTop 10 predictions:")
    print(ranked_df[['drug_id', 'disease_id', score_column, 'rank']].head(10).to_string(index=False))

    return ranked_df


def filter_top_predictions(
    ranked_df: pd.DataFrame,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
) -> pd.DataFrame:
    """
    Filter to top K predictions or minimum score threshold.

    Args:
        ranked_df: DataFrame with ranked predictions
        top_k: Return top K predictions (optional)
        min_score: Minimum treat_score threshold (optional)

    Returns:
        Filtered DataFrame
    """
    result = ranked_df.copy()

    if top_k is not None:
        result = result.head(top_k)
        print(f"Filtered to top {top_k} predictions")

    if min_score is not None:
        result = result[result['treat_score'] >= min_score]
        print(f"Filtered to predictions with score >= {min_score}")

    return result


def add_node_metadata(
    predictions_df: pd.DataFrame,
    drugs_df: pd.DataFrame,
    diseases_df: pd.DataFrame,
    drug_col: str = 'drug_id',
    disease_col: str = 'disease_id',
) -> pd.DataFrame:
    """
    Add drug and disease names to predictions.

    Args:
        predictions_df: DataFrame with predictions
        drugs_df: DataFrame with drug information (node_id, node_name)
        diseases_df: DataFrame with disease information (node_id, node_name)
        drug_col: Name of drug ID column
        disease_col: Name of disease ID column

    Returns:
        DataFrame with added drug_name and disease_name columns
    """
    print("Adding node metadata...")

    result = predictions_df.copy()

    # Merge drug names
    result = result.merge(
        drugs_df[['node_id', 'node_name']].rename(columns={'node_id': drug_col, 'node_name': 'drug_name'}),
        on=drug_col,
        how='left'
    )

    # Merge disease names
    result = result.merge(
        diseases_df[['node_id', 'node_name']].rename(columns={'node_id': disease_col, 'node_name': 'disease_name'}),
        on=disease_col,
        how='left'
    )

    print(f"✓ Added metadata for {len(result):,} predictions")

    return result


def save_predictions(
    predictions_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Save predictions to CSV.

    Args:
        predictions_df: DataFrame with predictions
        output_path: Path to save the CSV file
    """
    from pathlib import Path

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(output_file, index=False)

    print(f"✓ Saved predictions to: {output_file}")
