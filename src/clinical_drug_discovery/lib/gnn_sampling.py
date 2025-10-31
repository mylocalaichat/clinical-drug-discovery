"""
Subgraph sampling strategies for handling large-scale graphs efficiently.
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional
import random
import numpy as np
from torch_geometric.data import Data


def sample_connected_subgraph(
    edges_df: pd.DataFrame,
    seed_nodes: List[str],
    max_nodes: int = 10000,
    max_hops: int = 2,
    include_node_types: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Sample a connected subgraph starting from seed nodes.
    
    Args:
        edges_df: Full edge DataFrame
        seed_nodes: List of seed node IDs to start sampling from
        max_nodes: Maximum number of nodes in subgraph
        max_hops: Maximum number of hops from seed nodes
        include_node_types: Node types to include
    
    Returns:
        Subgraph edges DataFrame
    """
    print(f"Sampling connected subgraph from {len(seed_nodes)} seed nodes...")
    
    current_nodes = set(seed_nodes)
    all_nodes = set(seed_nodes)
    subgraph_edges = []
    
    for hop in range(max_hops):
        if len(all_nodes) >= max_nodes:
            break
            
        print(f"  Hop {hop + 1}: expanding from {len(current_nodes)} nodes...")
        
        # Find edges connected to current nodes
        connected_edges = edges_df[
            (edges_df['x_id'].isin(current_nodes)) | 
            (edges_df['y_id'].isin(current_nodes))
        ]
        
        # Filter by node types if specified
        if include_node_types:
            connected_edges = connected_edges[
                connected_edges['x_type'].isin(include_node_types) &
                connected_edges['y_type'].isin(include_node_types)
            ]
        
        # Get new nodes
        new_x_nodes = set(connected_edges['x_id'].unique()) - all_nodes
        new_y_nodes = set(connected_edges['y_id'].unique()) - all_nodes
        new_nodes = new_x_nodes | new_y_nodes
        
        # Limit new nodes if we're approaching max_nodes
        if len(all_nodes) + len(new_nodes) > max_nodes:
            remaining_slots = max_nodes - len(all_nodes)
            new_nodes = set(random.sample(list(new_nodes), remaining_slots))
        
        all_nodes.update(new_nodes)
        current_nodes = new_nodes
        subgraph_edges.append(connected_edges)
        
        print(f"    Added {len(new_nodes)} new nodes (total: {len(all_nodes)})")
        
        if not new_nodes:  # No more expansion possible
            break
    
    # Combine all edges and filter to final node set
    final_edges = pd.concat(subgraph_edges, ignore_index=True)
    final_edges = final_edges[
        final_edges['x_id'].isin(all_nodes) &
        final_edges['y_id'].isin(all_nodes)
    ].drop_duplicates().reset_index(drop=True)
    
    print(f"Sampled subgraph: {len(all_nodes)} nodes, {len(final_edges)} edges")
    return final_edges


def sample_drug_disease_subgraph(
    edges_df: pd.DataFrame,
    max_nodes: int = 15000,
    drug_sample_size: int = 500,
    disease_sample_size: int = 200
) -> pd.DataFrame:
    """
    Sample a subgraph focused on drug-disease relationships.
    
    Args:
        edges_df: Full edge DataFrame
        max_nodes: Maximum nodes in subgraph
        drug_sample_size: Number of drugs to sample as seeds
        disease_sample_size: Number of diseases to sample as seeds
    
    Returns:
        Drug-disease focused subgraph
    """
    print("Sampling drug-disease focused subgraph...")
    
    # Get drug and disease nodes
    drug_nodes = edges_df[edges_df['x_type'] == 'drug']['x_id'].unique()
    disease_nodes = edges_df[edges_df['x_type'] == 'disease']['x_id'].unique()
    
    # Add nodes from y side
    drug_nodes = np.concatenate([
        drug_nodes,
        edges_df[edges_df['y_type'] == 'drug']['y_id'].unique()
    ])
    disease_nodes = np.concatenate([
        disease_nodes,
        edges_df[edges_df['y_type'] == 'disease']['y_id'].unique()
    ])
    
    drug_nodes = np.unique(drug_nodes)
    disease_nodes = np.unique(disease_nodes)
    
    print(f"Available: {len(drug_nodes)} drugs, {len(disease_nodes)} diseases")
    
    # Sample seed nodes
    selected_drugs = random.sample(
        list(drug_nodes), 
        min(drug_sample_size, len(drug_nodes))
    )
    selected_diseases = random.sample(
        list(disease_nodes), 
        min(disease_sample_size, len(disease_nodes))
    )
    
    seed_nodes = selected_drugs + selected_diseases
    
    # Sample connected subgraph
    subgraph = sample_connected_subgraph(
        edges_df=edges_df,
        seed_nodes=seed_nodes,
        max_nodes=max_nodes,
        max_hops=2,
        include_node_types=[
            'drug', 'disease', 'gene/protein', 'pathway', 
            'biological_process', 'effect/phenotype'
        ]
    )
    
    return subgraph


def progressive_loading_strategy(
    edges_csv: str,
    target_sizes: List[int] = [1000, 5000, 15000, 50000],
    include_node_types: Optional[List[str]] = None
) -> List[Tuple[Data, Dict]]:
    """
    Progressive loading strategy that loads increasingly larger subgraphs.
    
    This is useful for:
    1. Quick experimentation with small graphs
    2. Gradually scaling up to larger graphs
    3. Performance testing and optimization
    
    Args:
        edges_csv: Path to edges CSV file
        target_sizes: List of target node counts for each progressive load
        include_node_types: Node types to include
    
    Returns:
        List of (Data, node_metadata) tuples for each size
    """
    from .gnn_embeddings import load_graph_from_csv
    
    print("Progressive loading strategy...")
    results = []
    
    for target_size in target_sizes:
        print(f"\n{'='*50}")
        print(f"Loading graph with ~{target_size:,} nodes")
        print(f"{'='*50}")
        
        data, metadata = load_graph_from_csv(
            edges_csv=edges_csv,
            limit_nodes=target_size,
            include_node_types=include_node_types,
            use_cache=True
        )
        
        results.append((data, metadata))
        
        print(f"✓ Loaded: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
    
    return results


def node_importance_sampling(
    edges_df: pd.DataFrame,
    target_nodes: int = 10000,
    importance_metric: str = 'degree'
) -> List[str]:
    """
    Sample nodes based on importance metrics (degree, betweenness, etc.).
    
    Args:
        edges_df: Edge DataFrame
        target_nodes: Number of nodes to sample
        importance_metric: 'degree', 'random', or 'type_balanced'
    
    Returns:
        List of selected node IDs
    """
    print(f"Sampling {target_nodes} nodes using {importance_metric} importance...")
    
    if importance_metric == 'degree':
        # Calculate degree centrality
        x_degrees = edges_df['x_id'].value_counts()
        y_degrees = edges_df['y_id'].value_counts()
        
        all_degrees = x_degrees.add(y_degrees, fill_value=0)
        
        # Sample top nodes by degree
        top_nodes = all_degrees.nlargest(target_nodes).index.tolist()
        return top_nodes
        
    elif importance_metric == 'type_balanced':
        # Sample balanced across node types
        x_nodes = edges_df[['x_id', 'x_type']].drop_duplicates()
        y_nodes = edges_df[['y_id', 'y_type']].drop_duplicates()
        y_nodes.columns = ['x_id', 'x_type']
        
        all_nodes = pd.concat([x_nodes, y_nodes]).drop_duplicates()
        
        # Sample proportionally from each type
        selected_nodes = []
        type_counts = all_nodes['x_type'].value_counts()
        
        for node_type, count in type_counts.items():
            type_nodes = all_nodes[all_nodes['x_type'] == node_type]['x_id'].tolist()
            sample_size = min(int(target_nodes * count / len(all_nodes)), len(type_nodes))
            selected_nodes.extend(random.sample(type_nodes, sample_size))
        
        return selected_nodes[:target_nodes]
    
    else:  # random
        all_nodes = set(edges_df['x_id'].unique()) | set(edges_df['y_id'].unique())
        return random.sample(list(all_nodes), min(target_nodes, len(all_nodes)))