"""
Memory-efficient subgraph sampling for large graphs.
"""

import pandas as pd
import random
from typing import List, Optional, Tuple
from pathlib import Path


def memory_efficient_node_sampling(
    edges_csv: str,
    max_nodes: int = 10000,
    node_types_priority: Optional[List[str]] = None,
    sample_strategy: str = "balanced"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Sample nodes in a memory-efficient way for large graphs.
    
    Args:
        edges_csv: Path to edges CSV file
        max_nodes: Maximum number of nodes to sample
        node_types_priority: Prioritized node types
        sample_strategy: 'balanced', 'degree_weighted', or 'random'
    
    Returns:
        Tuple of (sampled_edges_df, selected_node_ids)
    """
    print(f"Memory-efficient sampling: max {max_nodes:,} nodes")
    
    if node_types_priority is None:
        node_types_priority = ['drug', 'disease', 'gene/protein', 'pathway']
    
    # Read edges in chunks to manage memory
    chunk_size = 100000
    selected_nodes = set()
    degree_counts = {}
    
    print("1. Analyzing graph structure in chunks...")
    
    for chunk_idx, chunk in enumerate(pd.read_csv(edges_csv, chunksize=chunk_size)):
        if chunk_idx % 10 == 0:
            print(f"   Processing chunk {chunk_idx}, found {len(selected_nodes):,} nodes")
        
        # Filter by node types early
        chunk_filtered = chunk[
            chunk['x_type'].isin(node_types_priority) & 
            chunk['y_type'].isin(node_types_priority)
        ]
        
        if len(chunk_filtered) == 0:
            continue
        
        # Count degrees for importance sampling
        if sample_strategy == "degree_weighted":
            for node_id in chunk_filtered['x_id']:
                degree_counts[node_id] = degree_counts.get(node_id, 0) + 1
            for node_id in chunk_filtered['y_id']:
                degree_counts[node_id] = degree_counts.get(node_id, 0) + 1
        
        # Collect unique nodes
        chunk_nodes = set(chunk_filtered['x_id'].tolist() + chunk_filtered['y_id'].tolist())
        selected_nodes.update(chunk_nodes)
        
        # Early termination if we have enough candidates
        if len(selected_nodes) > max_nodes * 3:  # 3x buffer for sampling
            break
    
    print(f"2. Found {len(selected_nodes):,} candidate nodes")
    
    # Sample nodes based on strategy
    if len(selected_nodes) <= max_nodes:
        final_nodes = list(selected_nodes)
        print(f"   Using all {len(final_nodes):,} nodes (under limit)")
    else:
        if sample_strategy == "degree_weighted" and degree_counts:
            # Sample by degree (importance)
            sorted_nodes = sorted(degree_counts.items(), key=lambda x: x[1], reverse=True)
            final_nodes = [node_id for node_id, _ in sorted_nodes[:max_nodes]]
            print(f"   Sampled {len(final_nodes):,} high-degree nodes")
        
        elif sample_strategy == "balanced":
            # Balance across node types
            final_nodes = []
            nodes_by_type = {}
            
            # Re-read to get node types (more memory efficient than storing all)
            print("   Balancing across node types...")
            temp_nodes = {}
            for chunk in pd.read_csv(edges_csv, chunksize=chunk_size):
                chunk_filtered = chunk[
                    chunk['x_type'].isin(node_types_priority) & 
                    chunk['y_type'].isin(node_types_priority)
                ]
                
                for _, row in chunk_filtered.iterrows():
                    if row['x_id'] in selected_nodes:
                        temp_nodes[row['x_id']] = row['x_type']
                    if row['y_id'] in selected_nodes:
                        temp_nodes[row['y_id']] = row['y_type']
                
                if len(temp_nodes) >= len(selected_nodes):
                    break
            
            # Group by type
            for node_id, node_type in temp_nodes.items():
                if node_type not in nodes_by_type:
                    nodes_by_type[node_type] = []
                nodes_by_type[node_type].append(node_id)
            
            # Sample proportionally
            nodes_per_type = max_nodes // len(nodes_by_type)
            for node_type, type_nodes in nodes_by_type.items():
                sample_size = min(nodes_per_type, len(type_nodes))
                final_nodes.extend(random.sample(type_nodes, sample_size))
            
            # Fill remaining slots
            remaining_slots = max_nodes - len(final_nodes)
            if remaining_slots > 0:
                remaining_nodes = [n for n in selected_nodes if n not in final_nodes]
                if remaining_nodes:
                    additional = random.sample(remaining_nodes, min(remaining_slots, len(remaining_nodes)))
                    final_nodes.extend(additional)
            
            print(f"   Balanced sampling: {len(final_nodes):,} nodes across {len(nodes_by_type)} types")
        
        else:  # random
            final_nodes = random.sample(list(selected_nodes), max_nodes)
            print(f"   Random sampling: {len(final_nodes):,} nodes")
    
    final_node_set = set(final_nodes)
    
    # Extract subgraph edges efficiently
    print("3. Extracting subgraph edges...")
    subgraph_edges = []
    
    for chunk_idx, chunk in enumerate(pd.read_csv(edges_csv, chunksize=chunk_size)):
        if chunk_idx % 5 == 0:
            print(f"   Processing edges chunk {chunk_idx}")
        
        # Filter edges to selected nodes
        valid_edges = chunk[
            chunk['x_id'].isin(final_node_set) & 
            chunk['y_id'].isin(final_node_set) &
            chunk['x_type'].isin(node_types_priority) & 
            chunk['y_type'].isin(node_types_priority)
        ]
        
        if len(valid_edges) > 0:
            subgraph_edges.append(valid_edges)
    
    if subgraph_edges:
        final_edges_df = pd.concat(subgraph_edges, ignore_index=True)
    else:
        # Create empty DataFrame with correct structure
        final_edges_df = pd.DataFrame(columns=['x_id', 'y_id', 'x_type', 'y_type', 'x_name', 'y_name'])
    
    print(f"4. Final subgraph: {len(final_nodes):,} nodes, {len(final_edges_df):,} edges")
    
    return final_edges_df, final_nodes


def create_memory_efficient_graph(
    edges_csv: str,
    max_nodes: int = 10000,
    output_edges_csv: Optional[str] = None
) -> str:
    """
    Create a memory-efficient subgraph and save to file.
    
    Args:
        edges_csv: Input edges CSV
        max_nodes: Maximum nodes in subgraph
        output_edges_csv: Output file path
    
    Returns:
        Path to saved subgraph file
    """
    if output_edges_csv is None:
        base_path = Path(edges_csv).parent
        output_edges_csv = base_path / f"subgraph_{max_nodes}_nodes.csv"
    
    print(f"Creating memory-efficient subgraph: {max_nodes:,} nodes")
    
    # Sample subgraph
    subgraph_edges, selected_nodes = memory_efficient_node_sampling(
        edges_csv=edges_csv,
        max_nodes=max_nodes,
        sample_strategy="balanced"
    )
    
    # Save subgraph
    print(f"Saving subgraph to: {output_edges_csv}")
    Path(output_edges_csv).parent.mkdir(parents=True, exist_ok=True)
    subgraph_edges.to_csv(output_edges_csv, index=False)
    
    print(f"✓ Saved subgraph: {len(selected_nodes):,} nodes, {len(subgraph_edges):,} edges")
    
    return str(output_edges_csv)