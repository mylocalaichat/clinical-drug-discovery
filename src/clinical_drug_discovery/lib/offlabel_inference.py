"""
Off-Label Drug Discovery: Fast Inference Utilities

This module provides optimized inference methods:
- Pre-computed embedding approach (1000x faster)
- Cosine similarity baseline (instant, for comparison)
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from .offlabel_model import OffLabelRGCN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_embeddings_once(
    model: OffLabelRGCN,
    data: HeteroData,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Pre-compute embeddings for all nodes in the graph.

    This should be called ONCE, then embeddings can be reused
    for all predictions.

    Args:
        model: Trained R-GCN model
        data: HeteroData graph
        device: Device to run on

    Returns:
        Dictionary of embeddings {node_type: tensor}
    """
    logger.info("Pre-computing node embeddings...")

    model = model.to(device)
    data = data.to(device)
    model.eval()

    with torch.no_grad():
        embeddings = model.encode(data)

    logger.info(f"Embeddings computed:")
    for node_type, emb in embeddings.items():
        logger.info(f"  {node_type}: {emb.shape}")

    return embeddings


def predict_with_mlp(
    model: OffLabelRGCN,
    drug_embeddings: torch.Tensor,
    disease_embeddings: torch.Tensor,
    batch_size: int = 10000
) -> np.ndarray:
    """
    Predict using pre-computed embeddings + MLP head.

    This uses the trained MLP head, preserving model accuracy.

    Args:
        model: Trained R-GCN model
        drug_embeddings: Pre-computed drug embeddings [N, dim]
        disease_embeddings: Pre-computed disease embeddings [N, dim]
        batch_size: Batch size for prediction

    Returns:
        Array of prediction scores [N]
    """
    model.eval()
    predictions = []

    device = drug_embeddings.device
    num_pairs = len(drug_embeddings)

    with torch.no_grad():
        for i in range(0, num_pairs, batch_size):
            batch_drug_embs = drug_embeddings[i:i+batch_size]
            batch_disease_embs = disease_embeddings[i:i+batch_size]

            batch_preds = model.link_predictor(batch_drug_embs, batch_disease_embs)
            predictions.extend(batch_preds.cpu().numpy())

    return np.array(predictions)


def predict_with_cosine(
    drug_embeddings: torch.Tensor,
    disease_embeddings: torch.Tensor
) -> np.ndarray:
    """
    Predict using cosine similarity between embeddings.

    This is a simple baseline that doesn't use the MLP head.
    Cosine similarity ranges from -1 to 1; we rescale to [0, 1].

    Args:
        drug_embeddings: Drug embeddings [N, dim]
        disease_embeddings: Disease embeddings [N, dim]

    Returns:
        Array of cosine similarity scores [N], scaled to [0, 1]
    """
    # Normalize embeddings
    drug_norm = F.normalize(drug_embeddings, p=2, dim=1)
    disease_norm = F.normalize(disease_embeddings, p=2, dim=1)

    # Compute cosine similarity
    cosine_sim = (drug_norm * disease_norm).sum(dim=1)

    # Scale from [-1, 1] to [0, 1]
    scores = (cosine_sim + 1) / 2

    return scores.cpu().numpy()


def compute_all_pairwise_scores(
    drug_embeddings: torch.Tensor,
    disease_embeddings: torch.Tensor,
    method: str = 'cosine'
) -> torch.Tensor:
    """
    Compute scores for ALL drug-disease pairs at once.

    This creates a full matrix [num_drugs, num_diseases].
    Only feasible for cosine similarity (instant).

    Args:
        drug_embeddings: All drug embeddings [num_drugs, dim]
        disease_embeddings: All disease embeddings [num_diseases, dim]
        method: 'cosine' or 'mlp' (cosine is much faster for full matrix)

    Returns:
        Score matrix [num_drugs, num_diseases]
    """
    if method == 'cosine':
        # Normalize
        drug_norm = F.normalize(drug_embeddings, p=2, dim=1)
        disease_norm = F.normalize(disease_embeddings, p=2, dim=1)

        # Compute pairwise cosine similarity
        # [num_drugs, dim] @ [dim, num_diseases] = [num_drugs, num_diseases]
        similarity_matrix = torch.mm(drug_norm, disease_norm.t())

        # Scale to [0, 1]
        scores = (similarity_matrix + 1) / 2

        return scores

    else:
        raise ValueError("Full pairwise MLP prediction is too expensive. Use batch prediction instead.")


def find_top_k_diseases_for_drug(
    drug_idx: int,
    drug_embeddings: torch.Tensor,
    disease_embeddings: torch.Tensor,
    k: int = 10,
    method: str = 'cosine'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find top K diseases for a specific drug.

    Args:
        drug_idx: Index of the drug
        drug_embeddings: All drug embeddings
        disease_embeddings: All disease embeddings
        k: Number of top diseases to return
        method: 'cosine' or 'mlp'

    Returns:
        (top_k_indices, top_k_scores)
    """
    drug_emb = drug_embeddings[drug_idx].unsqueeze(0)  # [1, dim]

    if method == 'cosine':
        # Normalize
        drug_norm = F.normalize(drug_emb, p=2, dim=1)
        disease_norm = F.normalize(disease_embeddings, p=2, dim=1)

        # Compute similarity with all diseases
        scores = torch.mm(drug_norm, disease_norm.t()).squeeze(0)  # [num_diseases]

        # Scale to [0, 1]
        scores = (scores + 1) / 2

    else:
        raise NotImplementedError("MLP method for top-k not implemented yet")

    # Get top K
    top_k_scores, top_k_indices = torch.topk(scores, k)

    return top_k_indices.cpu().numpy(), top_k_scores.cpu().numpy()


if __name__ == "__main__":
    # Example usage
    logger.info("Fast inference utilities loaded")
