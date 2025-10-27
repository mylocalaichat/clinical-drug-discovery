"""
Dagster assets for graph embeddings pipeline.

This module splits the graph embeddings process into multiple assets:
1. Load graph from Neo4j
2. Train Node2Vec embeddings 
3. Save embeddings to disk
4. Create embedding DataFrame
5. Flatten embeddings for ML models
"""

import os
from pathlib import Path
from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd
from dagster import AssetExecutionContext, asset

from clinical_drug_discovery.lib.graph_embeddings import (
    create_embedding_dataframe,
    flatten_embeddings_for_model,
    load_graph_from_neo4j,
    save_embeddings,
    train_node2vec_embeddings,
)


@asset(group_name="embeddings", compute_kind="neo4j")
def knowledge_graph(
    context: AssetExecutionContext,
    clinical_validation_stats: Dict,  # Ensure clinical evidence is in graph first
) -> nx.Graph:
    """Load knowledge graph from Neo4j into NetworkX format."""
    context.log.info("Loading knowledge graph from Neo4j...")
    
    # Log that clinical validation is complete
    context.log.info(f"Clinical validation completed with {len(clinical_validation_stats)} metrics")
    
    # Get Neo4j connection details from environment
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j")
    database = os.getenv("NEO4J_DATABASE", "primekg")
    
    # Exclude INDICATION edges to prevent data leakage in link prediction
    exclude_edges = ["INDICATION", "CONTRAINDICATION"]
    
    # Use a very small limit for testing (1 node)
    limit_nodes = 1
    context.log.info(f"Testing with minimal graph: {limit_nodes} node")
    
    graph = load_graph_from_neo4j(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        database=database,
        exclude_edge_types=exclude_edges,
        limit_nodes=limit_nodes,
    )
    
    context.add_output_metadata({
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "excluded_edge_types": exclude_edges,
        "clinical_validation_complete": True,
        "limit_nodes": limit_nodes,
        "is_testing": True,
    })
    
    return graph


@asset(group_name="embeddings", compute_kind="ml")
def node2vec_embeddings(
    context: AssetExecutionContext,
    knowledge_graph: nx.Graph,
) -> Dict[str, np.ndarray]:
    """Train Node2Vec embeddings on the knowledge graph."""
    context.log.info("Training Node2Vec embeddings...")
    
    # Node2Vec hyperparameters
    embedding_params = {
        "dimensions": 512,
        "walk_length": 30,
        "num_walks": 10,
        "window_size": 10,
        "workers": 4,
        "p": 1.0,
        "q": 1.0,
        "seed": 42,
    }
    
    embeddings = train_node2vec_embeddings(
        graph=knowledge_graph,
        **embedding_params
    )
    
    # Convert all keys to strings for Dagster type compliance
    string_keyed_embeddings = {str(k): v for k, v in embeddings.items()}
    
    context.add_output_metadata({
        "num_embeddings": len(string_keyed_embeddings),
        "embedding_dimension": string_keyed_embeddings[list(string_keyed_embeddings.keys())[0]].shape[0],
        "hyperparameters": embedding_params,
    })
    
    return string_keyed_embeddings


@asset(group_name="embeddings", compute_kind="io")
def saved_embeddings(
    context: AssetExecutionContext,
    node2vec_embeddings: Dict[str, np.ndarray],
) -> str:
    """Save embeddings to disk for persistence."""
    context.log.info("Saving embeddings to disk...")
    
    output_path = "data/06_models/embeddings/node2vec_embeddings.pkl"
    
    save_embeddings(
        embeddings=node2vec_embeddings,
        output_path=output_path,
    )
    
    output_file = Path(output_path)
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    
    context.add_output_metadata({
        "output_path": str(output_path),
        "file_size_mb": round(file_size_mb, 2),
        "num_embeddings": len(node2vec_embeddings),
    })
    
    return str(output_path)


@asset(group_name="embeddings", compute_kind="transform")
def embedding_dataframe(
    context: AssetExecutionContext,
    node2vec_embeddings: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Convert embeddings dictionary to DataFrame format."""
    context.log.info("Creating embedding DataFrame...")
    
    df = create_embedding_dataframe(node2vec_embeddings)
    
    context.add_output_metadata({
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "column_names": list(df.columns),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    })
    
    return df


@asset(group_name="embeddings", compute_kind="transform")
def flattened_embeddings(
    context: AssetExecutionContext,
    embedding_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Flatten embeddings for use in ML models."""
    context.log.info("Flattening embeddings for ML models...")
    
    flattened_df = flatten_embeddings_for_model(
        df=embedding_dataframe,
        embedding_column="embedding",
        prefix="emb_",
    )
    
    # Save flattened embeddings for inspection
    output_path = "data/06_models/embeddings/flattened_embeddings.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save a sample for inspection (full file would be too large)
    sample_df = flattened_df.head(1000)
    sample_df.to_csv(output_path, index=False)
    
    context.add_output_metadata({
        "num_rows": len(flattened_df),
        "num_columns": len(flattened_df.columns),
        "embedding_dimensions": len([col for col in flattened_df.columns if col.startswith("emb_")]),
        "sample_saved_to": output_path,
        "memory_usage_mb": round(flattened_df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    })
    
    return flattened_df


@asset(group_name="embeddings", compute_kind="validation")
def embedding_validation_stats(
    context: AssetExecutionContext,
    node2vec_embeddings: Dict[str, np.ndarray],
    flattened_embeddings: pd.DataFrame,
) -> Dict[str, float]:
    """Validate embeddings quality and compute statistics."""
    context.log.info("Computing embedding validation statistics...")
    
    # Sample a few embeddings for analysis
    sample_embeddings = list(node2vec_embeddings.values())[:100]
    embeddings_array = np.array(sample_embeddings)
    
    stats = {
        "mean_embedding_norm": float(np.mean([np.linalg.norm(emb) for emb in sample_embeddings])),
        "std_embedding_norm": float(np.std([np.linalg.norm(emb) for emb in sample_embeddings])),
        "mean_embedding_value": float(np.mean(embeddings_array)),
        "std_embedding_value": float(np.std(embeddings_array)),
        "min_embedding_value": float(np.min(embeddings_array)),
        "max_embedding_value": float(np.max(embeddings_array)),
        "num_zero_embeddings": int(np.sum(np.all(embeddings_array == 0, axis=1))),
        "coverage_ratio": len(node2vec_embeddings) / len(flattened_embeddings),
    }
    
    context.add_output_metadata(stats)
    
    context.log.info("Embedding validation completed:")
    for key, value in stats.items():
        context.log.info(f"  {key}: {value}")
    
    return stats