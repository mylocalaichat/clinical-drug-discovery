"""
Predict drug candidates for a specific disease.

This provides fast inference for single disease queries.
"""

import logging
from typing import Dict
import pandas as pd
import torch

from .offlabel_model import OffLabelRGCN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_drugs_for_disease(
    disease_name_or_id: str,
    model: OffLabelRGCN,
    train_graph,
    node_mapping: Dict[str, Dict[str, int]],
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    device: torch.device,
    top_k: int = 100,
    min_drug_degree: int = 5,
    exclude_known: bool = True
) -> pd.DataFrame:
    """
    Predict drug candidates for a specific disease.

    Args:
        disease_name_or_id: Disease name (e.g., "Castleman") or ID
        model: Trained R-GCN model
        train_graph: HeteroData graph
        node_mapping: Node ID to index mapping
        nodes_df: Node metadata DataFrame
        edges_df: Edges DataFrame (to filter known relationships)
        device: Device to run on
        top_k: Number of top drugs to return
        min_drug_degree: Minimum connectivity for drugs
        exclude_known: Whether to exclude known drug-disease relationships

    Returns:
        DataFrame with top drug candidates ranked by score
    """
    logger.info(f"Finding drug candidates for: {disease_name_or_id}")

    # Find the disease
    disease_nodes = nodes_df[nodes_df['node_type'] == 'disease']

    # Search by name or ID
    matching_diseases = disease_nodes[
        (disease_nodes['node_name'].str.contains(disease_name_or_id, case=False, na=False)) |
        (disease_nodes['node_id'] == disease_name_or_id)
    ]

    if len(matching_diseases) == 0:
        logger.error(f"No disease found matching '{disease_name_or_id}'")
        return pd.DataFrame()

    if len(matching_diseases) > 1:
        logger.warning(f"Multiple diseases found matching '{disease_name_or_id}':")
        for _, row in matching_diseases.iterrows():
            logger.warning(f"  - {row['node_id']}: {row['node_name']}")
        logger.warning("Using the first match")

    disease_id = matching_diseases.iloc[0]['node_id']
    disease_full_name = matching_diseases.iloc[0]['node_name']
    logger.info(f"Disease: {disease_full_name} (ID: {disease_id})")

    # Check if disease is in node_mapping
    if disease_id not in node_mapping.get('disease', {}):
        logger.error(f"Disease '{disease_id}' not in node_mapping (not in training graph)")
        return pd.DataFrame()

    disease_idx = node_mapping['disease'][disease_id]

    # Get all drugs in the graph
    all_drugs = nodes_df[nodes_df['node_type'] == 'drug']['node_id'].tolist()

    # Filter by minimum degree if specified
    if min_drug_degree > 0:
        logger.info(f"Filtering drugs with degree >= {min_drug_degree}...")
        drug_degrees = {}

        drug_source_mask = edges_df['source_type'] == 'drug'
        drug_target_mask = edges_df['target_type'] == 'drug'

        drug_source_counts = edges_df[drug_source_mask]['source_id'].value_counts().to_dict()
        drug_target_counts = edges_df[drug_target_mask]['target_id'].value_counts().to_dict()

        for drug_id in set(list(drug_source_counts.keys()) + list(drug_target_counts.keys())):
            drug_degrees[drug_id] = drug_source_counts.get(drug_id, 0) + drug_target_counts.get(drug_id, 0)

        filtered_drugs = [d for d in all_drugs if drug_degrees.get(d, 0) >= min_drug_degree]
        logger.info(f"Filtered drugs: {len(filtered_drugs):,} / {len(all_drugs):,}")
    else:
        filtered_drugs = all_drugs

    # Only consider drugs that are in node_mapping
    candidate_drugs = [d for d in filtered_drugs if d in node_mapping.get('drug', {})]
    logger.info(f"Candidate drugs in node_mapping: {len(candidate_drugs):,}")

    # Exclude known relationships if requested
    if exclude_known:
        logger.info("Excluding drugs with known relationships to this disease...")
        known_drugs = set()

        for relation in ['indication', 'off-label use', 'contraindication']:
            rel_edges = edges_df[edges_df['relation'] == relation]
            for _, row in rel_edges.iterrows():
                if row['source_type'] == 'drug' and row['target_type'] == 'disease' and row['target_id'] == disease_id:
                    known_drugs.add(row['source_id'])
                elif row['source_type'] == 'disease' and row['target_type'] == 'drug' and row['source_id'] == disease_id:
                    known_drugs.add(row['target_id'])

        candidate_drugs = [d for d in candidate_drugs if d not in known_drugs]
        logger.info(f"After excluding {len(known_drugs)} known drugs: {len(candidate_drugs):,} candidates")

    if len(candidate_drugs) == 0:
        logger.error("No candidate drugs remaining after filtering")
        return pd.DataFrame()

    # Pre-compute embeddings
    logger.info("Computing embeddings...")
    model = model.to(device)
    train_graph = train_graph.to(device)
    model.eval()

    with torch.no_grad():
        all_embeddings = model.encode(train_graph)

    # Get disease embedding (same for all pairs)
    disease_emb = all_embeddings['disease'][disease_idx].unsqueeze(0)  # [1, dim]

    # Predict scores for all candidate drugs
    logger.info(f"Predicting scores for {len(candidate_drugs):,} drugs...")
    drug_indices = [node_mapping['drug'][drug_id] for drug_id in candidate_drugs]
    drug_embs = all_embeddings['drug'][drug_indices]  # [N, dim]

    # Repeat disease embedding for all drugs
    disease_embs = disease_emb.repeat(len(candidate_drugs), 1)  # [N, dim]

    # Predict with MLP
    with torch.no_grad():
        predictions = model.link_predictor(drug_embs, disease_embs)

    scores = predictions.cpu().numpy()

    # Create results DataFrame
    drug_names = nodes_df[nodes_df['node_type'] == 'drug'].set_index('node_id')['node_name'].to_dict()

    results_df = pd.DataFrame({
        'drug_id': candidate_drugs,
        'drug_name': [drug_names.get(d, d) for d in candidate_drugs],
        'disease_id': disease_id,
        'disease_name': disease_full_name,
        'prediction_score': scores
    })

    # Sort by score (highest first)
    results_df = results_df.sort_values('prediction_score', ascending=False).reset_index(drop=True)

    # Add rank
    results_df['rank'] = range(1, len(results_df) + 1)

    # Reorder columns
    results_df = results_df[['rank', 'drug_id', 'drug_name', 'disease_id', 'disease_name', 'prediction_score']]

    # Return top K
    logger.info(f"Top {min(top_k, len(results_df))} predictions:")
    for _, row in results_df.head(min(10, len(results_df))).iterrows():
        logger.info(f"  {int(row['rank']):3d}. {row['drug_name'][:50]:50s} | Score: {row['prediction_score']:.4f}")

    return results_df.head(top_k)


if __name__ == "__main__":
    logger.info("Disease-specific prediction utility loaded")
