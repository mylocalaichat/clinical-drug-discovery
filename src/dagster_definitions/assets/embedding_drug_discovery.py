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
    Compute drug similarity matrix using embeddings.
    
    This asset creates a similarity matrix between drugs based on their
    graph embeddings, which can be used to find similar drugs for
    drug repurposing and discovery.
    """
    context.log.info("Computing drug similarity matrix from embeddings...")
    
    # Filter to only drug nodes (assuming drug IDs start with 'DB')
    drug_embeddings = flattened_embeddings[
        flattened_embeddings['node_id'].str.startswith('DB')
    ].copy()
    
    if len(drug_embeddings) == 0:
        context.log.warning("No drug embeddings found!")
        return pd.DataFrame()
    
    context.log.info(f"Found {len(drug_embeddings)} drug embeddings")
    
    # Extract embedding features (columns starting with 'emb_')
    embedding_cols = [col for col in drug_embeddings.columns if col.startswith('emb_')]
    X = drug_embeddings[embedding_cols].values
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(X)
    
    # Create DataFrame with drug IDs as indices/columns
    drug_ids = drug_embeddings['node_id'].tolist()
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=drug_ids,
        columns=drug_ids
    )
    
    # Save similarity matrix sample for inspection
    output_path = "data/07_model_output/drug_similarity_matrix_sample.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save top 100x100 sample (full matrix would be too large)
    sample_df = similarity_df.iloc[:100, :100]
    sample_df.to_csv(output_path)
    
    context.add_output_metadata({
        "num_drugs": len(drug_ids),
        "similarity_matrix_shape": f"{similarity_matrix.shape[0]}x{similarity_matrix.shape[1]}",
        "mean_similarity": float(np.mean(similarity_matrix)),
        "sample_saved_to": output_path,
        "embedding_dimensions": len(embedding_cols),
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
    
    This combines traditional graph queries with embedding similarity
    to find drugs that are topologically similar to known treatments.
    """
    context.log.info("Running embedding-enhanced drug discovery...")
    
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
        
        # Get base results from traditional graph query
        base_results = query_base_drug_discovery(
            disease_id=disease_id,
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_user=os.getenv("NEO4J_USER"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE"),
        )
        
        if len(base_results) == 0:
            context.log.warning(f"No base results for {disease_name}")
            continue
            
        # Enhance with embedding similarity
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
    
    For each drug found in base results, find the most similar drugs
    and add them as additional candidates.
    """
    enhanced_rows = []
    
    for _, row in base_results.iterrows():
        drug_id = row['drug_id']
        
        # Add original result
        enhanced_row = row.copy()
        enhanced_row['similarity_score'] = 1.0  # Self-similarity
        enhanced_row['discovery_source'] = 'original'
        enhanced_rows.append(enhanced_row)
        
        # Find similar drugs
        if drug_id in similarity_matrix.index:
            # Get similarity scores for this drug
            similarities = similarity_matrix.loc[drug_id].sort_values(ascending=False)
            
            # Get top K similar drugs (excluding self)
            similar_drugs = similarities.iloc[1:top_k_similar+1]  # Skip self at index 0
            
            for similar_drug_id, similarity_score in similar_drugs.items():
                if similarity_score > 0.7:  # Only high similarity
                    similar_row = row.copy()
                    similar_row['drug_id'] = similar_drug_id
                    similar_row['similarity_score'] = float(similarity_score)
                    similar_row['discovery_source'] = f'similar_to_{drug_id}'
                    enhanced_rows.append(similar_row)
    
    if enhanced_rows:
        enhanced_df = pd.DataFrame(enhanced_rows)
        context.log.info(f"Enhanced {len(base_results)} base results to {len(enhanced_df)} total candidates")
        return enhanced_df
    else:
        return base_results