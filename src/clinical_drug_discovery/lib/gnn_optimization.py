"""
Optimized edge index construction utilities for large-scale graphs.
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple, List, Optional
import time


def optimized_edge_index_construction(
    edges_df: pd.DataFrame,
    node_id_to_idx: Dict[str, int],
    chunk_size: int = 500000
) -> torch.Tensor:
    """
    Construct edge index in chunks to reduce memory usage and improve performance.
    
    Args:
        edges_df: DataFrame with x_id and y_id columns
        node_id_to_idx: Mapping from node IDs to indices
        chunk_size: Number of edges to process per chunk
    
    Returns:
        Edge index tensor [2, num_edges]
    """
    print(f"Building edge index in chunks of {chunk_size:,} edges...")
    
    total_edges = len(edges_df)
    num_chunks = (total_edges + chunk_size - 1) // chunk_size
    
    src_indices_list = []
    tgt_indices_list = []
    
    start_time = time.time()
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_edges)
        
        chunk = edges_df.iloc[start_idx:end_idx]
        
        # Vectorized mapping for chunk
        chunk_src = chunk['x_id'].map(node_id_to_idx).values
        chunk_tgt = chunk['y_id'].map(node_id_to_idx).values
        
        # Filter out any NaN values (missing mappings)
        valid_mask = ~(pd.isna(chunk_src) | pd.isna(chunk_tgt))
        
        if valid_mask.any():
            src_indices_list.append(chunk_src[valid_mask].astype(np.int64))
            tgt_indices_list.append(chunk_tgt[valid_mask].astype(np.int64))
        
        if (chunk_idx + 1) % 10 == 0 or (chunk_idx + 1) == num_chunks:
            elapsed = time.time() - start_time
            progress = (chunk_idx + 1) / num_chunks * 100
            rate = (chunk_idx + 1) * chunk_size / elapsed
            print(f"  Chunk {chunk_idx + 1}/{num_chunks} ({progress:.1f}%) - {rate:.0f} edges/sec")
    
    # Concatenate all chunks
    print("Concatenating chunks...")
    src_indices = np.concatenate(src_indices_list)
    tgt_indices = np.concatenate(tgt_indices_list)
    
    # Create edge index tensor
    edge_index = torch.tensor(
        np.stack([src_indices, tgt_indices], axis=0),
        dtype=torch.long
    )
    
    total_time = time.time() - start_time
    print(f"Edge index construction completed: {len(src_indices):,} edges in {total_time:.1f}s")
    
    return edge_index


def fast_node_extraction(
    edges_df: pd.DataFrame,
    include_node_types: List[str],
    limit_nodes: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Fast node extraction using set operations and vectorized pandas.
    
    Args:
        edges_df: DataFrame with edge data
        include_node_types: Node types to include
        limit_nodes: Maximum number of nodes to keep
    
    Returns:
        Filtered nodes DataFrame and node_id to index mapping
    """
    print("Fast node extraction using set operations...")
    
    # Use set operations for faster unique node discovery
    print("  Finding unique node IDs...")
    unique_x_ids = set(edges_df['x_id'].unique())
    unique_y_ids = set(edges_df['y_id'].unique())
    all_unique_ids = unique_x_ids | unique_y_ids
    
    print(f"  Found {len(all_unique_ids):,} unique node IDs")
    
    # Create a more efficient node DataFrame
    print("  Building node metadata...")
    
    # Get first occurrence of each node from both x and y sides
    x_nodes = edges_df[['x_id', 'x_name', 'x_type']].drop_duplicates('x_id')
    x_nodes.columns = ['id', 'name', 'type']
    
    y_nodes = edges_df[['y_id', 'y_name', 'y_type']].drop_duplicates('y_id')
    y_nodes.columns = ['id', 'name', 'type']
    
    # Combine and deduplicate (prioritize x_nodes in case of conflicts)
    nodes_df = pd.concat([x_nodes, y_nodes]).drop_duplicates('id', keep='first').reset_index(drop=True)
    
    print(f"  Created nodes DataFrame: {len(nodes_df):,} nodes")
    
    # Filter by node types
    if include_node_types:
        nodes_df = nodes_df[nodes_df['type'].isin(include_node_types)].reset_index(drop=True)
        print(f"  After type filtering: {len(nodes_df):,} nodes")
    
    # Apply limit
    if limit_nodes and len(nodes_df) > limit_nodes:
        nodes_df = nodes_df.head(limit_nodes).reset_index(drop=True)
        print(f"  After limit: {len(nodes_df):,} nodes")
    
    # Create ID mapping using dict comprehension (faster than enumerate)
    print("  Creating node ID mapping...")
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(nodes_df['id'].values)}
    
    return nodes_df, node_id_to_idx


def create_sparse_edge_index(
    edges_df: pd.DataFrame,
    valid_node_ids: set,
    node_id_to_idx: Dict[str, int],
    chunk_size: int = 1000000
) -> torch.Tensor:
    """
    Create edge index using sparse operations for memory efficiency.
    
    Args:
        edges_df: DataFrame with edge data
        valid_node_ids: Set of valid node IDs to include
        node_id_to_idx: Node ID to index mapping
        chunk_size: Chunk size for processing
    
    Returns:
        Edge index tensor
    """
    print("Creating sparse edge index...")
    
    # Pre-filter edges using set membership (very fast)
    print("  Pre-filtering edges...")
    start_time = time.time()
    
    valid_edges_mask = (
        edges_df['x_id'].isin(valid_node_ids) & 
        edges_df['y_id'].isin(valid_node_ids)
    )
    
    filtered_edges = edges_df[valid_edges_mask].reset_index(drop=True)
    filter_time = time.time() - start_time
    
    print(f"  Filtered to {len(filtered_edges):,} edges in {filter_time:.1f}s")
    
    # Build edge index in chunks
    edge_index = optimized_edge_index_construction(
        filtered_edges, 
        node_id_to_idx, 
        chunk_size=chunk_size
    )
    
    return edge_index