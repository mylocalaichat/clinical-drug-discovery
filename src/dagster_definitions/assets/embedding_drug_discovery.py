"""
Enhanced drug discovery using graph embeddings.

This module creates drug discovery assets that leverage graph embeddings
to find drug similarities and enhance discovery results.
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
from dagster import AssetExecutionContext, asset
from sklearn.metrics.pairwise import cosine_similarity

from clinical_drug_discovery.lib.drug_discovery import (
    query_base_drug_discovery,
)


@asset(group_name="drug_discovery", compute_kind="ml")
def drug_similarity_matrix(
    context: AssetExecutionContext,
    flattened_embeddings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute drug similarity matrix using Node2Vec graph embeddings.
    
    HOW IT WORKS:
    1. Extracts drug embeddings from the flattened Node2Vec embeddings
    2. Computes pairwise cosine similarity between all drug embedding vectors
    3. Creates a symmetric matrix where each cell [i,j] represents similarity between drug i and drug j
    
    MATHEMATICAL FOUNDATION:
    - Uses cosine similarity: sim(A,B) = (A·B) / (||A|| × ||B||)
    - Values range from -1 (opposite) to 1 (identical)
    - Graph embeddings capture structural neighborhood information from PrimeKG
    
    BIOLOGICAL INTERPRETATION:
    - High similarity (>0.8): Drugs with similar mechanisms or targets
    - Medium similarity (0.5-0.8): Potentially related therapeutic classes
    - Low similarity (<0.5): Structurally/functionally distinct drugs
    
    USE CASES:
    - Drug repurposing: Find drugs similar to known treatments
    - Mechanism discovery: Identify drugs with similar action patterns
    - Safety profiling: Drugs with similar embeddings may have similar side effects
    """
    context.log.info("Computing drug similarity matrix from Node2Vec embeddings...")
    
    # STEP 1: Filter to drug nodes only
    # DrugBank IDs typically start with 'DB' (e.g., DB00001, DB00002)
    # This separates drugs from diseases, proteins, and other node types
    drug_embeddings = flattened_embeddings[
        flattened_embeddings['node_id'].str.startswith('DB')
    ].copy()
    
    if len(drug_embeddings) == 0:
        context.log.warning("No drug embeddings found!")
        return pd.DataFrame()
    
    context.log.info(f"Found {len(drug_embeddings)} drug embeddings")
    
    # STEP 2: Extract embedding feature vectors
    # Each drug has a 512-dimensional embedding (from Node2Vec hyperparameters)
    # These are flattened into columns named 'emb_0', 'emb_1', ..., 'emb_511'
    embedding_cols = [col for col in drug_embeddings.columns if col.startswith('emb_')]
    X = drug_embeddings[embedding_cols].values  # Shape: (n_drugs, 512)
    
    context.log.info(f"Using {len(embedding_cols)}-dimensional embeddings for similarity computation")
    
    # STEP 3: Compute pairwise cosine similarity
    # Creates an n×n matrix where n = number of drugs
    # Each cell [i,j] = cosine similarity between drug i and drug j
    # Diagonal = 1.0 (perfect self-similarity)
    # Matrix is symmetric: similarity(A,B) = similarity(B,A)
    similarity_matrix = cosine_similarity(X)
    
    # STEP 4: Create labeled DataFrame for easy access
    # Rows and columns are drug IDs, making it easy to look up drug-drug similarities
    drug_ids = drug_embeddings['node_id'].tolist()
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=drug_ids,
        columns=drug_ids
    )
    
    # STEP 5: Save sample for inspection and analysis
    # Full matrix can be huge (n² entries), so we save a manageable sample
    output_path = "data/07_model_output/drug_similarity_matrix_sample.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save top 100x100 sample (full matrix would be too large for CSV)
    sample_df = similarity_df.iloc[:100, :100]
    sample_df.to_csv(output_path)
    
    # Log some interesting statistics about the similarity distribution
    similarity_values = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]  # Upper triangle, no diagonal
    context.log.info("Similarity statistics:")
    context.log.info(f"  Mean similarity: {np.mean(similarity_values):.3f}")
    context.log.info(f"  Std similarity: {np.std(similarity_values):.3f}")
    context.log.info(f"  Max similarity: {np.max(similarity_values):.3f}")
    context.log.info(f"  Min similarity: {np.min(similarity_values):.3f}")
    
    # Find most similar drug pairs (excluding self-similarity)
    flat_indices = np.unravel_index(np.argsort(similarity_values)[-5:], similarity_matrix.shape)
    context.log.info("Top 5 most similar drug pairs:")
    for i in range(5):
        row_idx, col_idx = flat_indices[0][-i-1], flat_indices[1][-i-1]
        if row_idx < len(drug_ids) and col_idx < len(drug_ids):
            drug1, drug2 = drug_ids[row_idx], drug_ids[col_idx]
            similarity = similarity_matrix[row_idx, col_idx]
            context.log.info(f"  {drug1} ↔ {drug2}: {similarity:.3f}")
    
    context.add_output_metadata({
        "num_drugs": len(drug_ids),
        "similarity_matrix_shape": f"{similarity_matrix.shape[0]}x{similarity_matrix.shape[1]}",
        "mean_similarity": float(np.mean(similarity_values)),
        "std_similarity": float(np.std(similarity_values)),
        "max_similarity": float(np.max(similarity_values)),
        "min_similarity": float(np.min(similarity_values)),
        "sample_saved_to": output_path,
        "embedding_dimensions": len(embedding_cols),
        "total_comparisons": int(len(similarity_values)),
    })
    
    return similarity_df


@asset(group_name="drug_discovery", compute_kind="ml")
def embedding_enhanced_drug_discovery(
    context: AssetExecutionContext,
    drug_similarity_matrix: pd.DataFrame,
    clinical_validation_stats: Dict,
) -> pd.DataFrame:
    """
    Enhanced drug discovery using embedding-based drug similarity.
    
    HOW IT DISCOVERS NEW DRUG-DISEASE RELATIONSHIPS:
    1. START: Query graph for known drug-disease connections using traditional paths
    2. EXPAND: For each known effective drug, find structurally similar drugs
    3. HYPOTHESIS: Similar drugs (by embedding) may treat similar diseases
    4. SCORE: Use similarity score as confidence measure for new drug candidates
    
    SCIENTIFIC RATIONALE:
    - Drugs with similar graph embeddings often share:
      * Similar molecular targets (proteins, enzymes)
      * Similar mechanisms of action
      * Similar therapeutic pathways
    - If Drug A treats Disease X, and Drug B is highly similar to Drug A,
      then Drug B might also treat Disease X (drug repurposing hypothesis)
    
    EXAMPLE WORKFLOW:
    - Known: Aspirin treats cardiovascular disease (from graph query)
    - Similar: Embedding finds Clopidogrel is 0.85 similar to Aspirin
    - Hypothesis: Clopidogrel might also treat cardiovascular disease
    - Score: 0.85 confidence based on structural similarity
    
    ADVANTAGES OVER TRADITIONAL METHODS:
    - Discovers drugs not directly connected in knowledge graph
    - Leverages implicit structural patterns in biomedical networks
    - Provides quantitative similarity scores for ranking candidates
    - Can find novel drug repurposing opportunities
    """
    context.log.info("Running embedding-enhanced drug discovery...")
    context.log.info("Strategy: Find known treatments, then discover similar drugs as new candidates")
    
    if len(drug_similarity_matrix) == 0:
        context.log.warning("Empty similarity matrix, returning empty results")
        return pd.DataFrame()
    
    all_results = []
    
    # Test diseases (same as in drug_discovery.py)
    TEST_DISEASES = [
        {"disease_id": "15564", "name": "Castleman disease"},
        {"disease_id": "8170", "name": "Ovarian cancer"},
    ]
    
    for disease_info in TEST_DISEASES:
        disease_id = disease_info['disease_id']
        disease_name = disease_info['name']
        
        context.log.info(f"Processing: {disease_name} (ID: {disease_id})")
        
        # STEP 1: Get base results from traditional graph query
        # This finds drugs that are DIRECTLY connected to the disease in PrimeKG
        # Examples: known treatments, drugs in clinical trials, etc.
        base_results = query_base_drug_discovery(
            disease_id=disease_id,
            memgraph_uri=os.getenv("MEMGRAPH_URI"),
            memgraph_user=os.getenv("MEMGRAPH_USER"),
            memgraph_password=os.getenv("MEMGRAPH_PASSWORD"),
            database=os.getenv("MEMGRAPH_DATABASE"),
        )
        
        if len(base_results) == 0:
            context.log.warning(f"No base results for {disease_name}")
            continue
        
        context.log.info(f"Found {len(base_results)} known drugs for {disease_name}")
            
        # STEP 2: Enhance with embedding similarity
        # For each known drug, find similar drugs that might ALSO treat this disease
        enhanced_results = _enhance_with_similarity(
            base_results, drug_similarity_matrix, context
        )
        
        # Add disease information
        enhanced_results['disease_id'] = disease_id
        enhanced_results['disease_name'] = disease_name
        enhanced_results['discovery_method'] = 'embedding_enhanced'
        
        all_results.append(enhanced_results)
    
    if not all_results:
        return pd.DataFrame()
    
    # Combine all results
    final_results = pd.concat(all_results, ignore_index=True)
    
    # Save results
    output_path = "data/07_model_output/embedding_enhanced_discovery.csv"
    final_results.to_csv(output_path, index=False)
    
    context.add_output_metadata({
        "total_drugs_found": len(final_results),
        "diseases_processed": len(TEST_DISEASES),
        "avg_similarity_score": float(final_results['similarity_score'].mean()) if 'similarity_score' in final_results.columns else 0,
        "output_saved_to": output_path,
    })
    
    return final_results


def _enhance_with_similarity(
    base_results: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    context: AssetExecutionContext,
    top_k_similar: int = 5,
) -> pd.DataFrame:
    """
    Enhance base discovery results with embedding similarity scores.
    
    DRUG REPURPOSING LOGIC:
    1. Take known effective drugs for a disease (from graph queries)
    2. Find drugs that are structurally similar (high embedding similarity)
    3. Hypothesize: similar drugs may treat the same disease
    4. Return expanded candidate list with similarity scores
    
    SCORING METHODOLOGY:
    - Original drugs: score = 1.0 (known to work)
    - Similar drugs: score = cosine similarity (0.7-1.0)
    - Higher similarity = higher confidence in therapeutic potential
    
    EXAMPLE:
    - Known: Drug A treats Disease X (from knowledge graph)
    - Similar: Drug B has 0.85 similarity to Drug A (from embeddings)
    - Hypothesis: Drug B might treat Disease X with 0.85 confidence
    - Output: Both Drug A (1.0) and Drug B (0.85) as candidates
    
    FILTERING:
    - Only includes highly similar drugs (>0.7 threshold)
    - Prevents low-confidence false positives
    - Focuses on most promising repurposing candidates
    """
    enhanced_rows = []
    
    for _, row in base_results.iterrows():
        drug_id = row['drug_id']
        
        # STEP 1: Add original known drug with perfect score
        enhanced_row = row.copy()
        enhanced_row['similarity_score'] = 1.0  # Perfect score for known treatments
        enhanced_row['discovery_source'] = 'original_known_treatment'
        enhanced_rows.append(enhanced_row)
        
        # STEP 2: Find structurally similar drugs for repurposing
        if drug_id in similarity_matrix.index:
            # Get similarity scores for this drug against all other drugs
            similarities = similarity_matrix.loc[drug_id].sort_values(ascending=False)
            
            # Get top K most similar drugs (excluding the drug itself)
            similar_drugs = similarities.iloc[1:top_k_similar+1]  # Skip self at index 0
            
            context.log.debug(f"Finding similar drugs to {drug_id}:")
            
            for similar_drug_id, similarity_score in similar_drugs.items():
                # Only include highly similar drugs (threshold = 0.7)
                if similarity_score > 0.7:
                    similar_row = row.copy()
                    similar_row['drug_id'] = similar_drug_id
                    similar_row['similarity_score'] = float(similarity_score)
                    similar_row['discovery_source'] = f'similar_to_{drug_id}'
                    enhanced_rows.append(similar_row)
                    
                    context.log.debug(f"  → {similar_drug_id}: {similarity_score:.3f} similarity")
    
    if enhanced_rows:
        enhanced_df = pd.DataFrame(enhanced_rows)
        original_count = len(base_results)
        total_count = len(enhanced_df)
        new_candidates = total_count - original_count
        
        context.log.info("Drug discovery enhancement:")
        context.log.info(f"  Original known drugs: {original_count}")
        context.log.info(f"  New similar candidates: {new_candidates}")
        context.log.info(f"  Total candidates: {total_count}")
        
        return enhanced_df
    else:
        return base_results