"""
Caching utilities for GNN edge index construction to avoid recomputation.
"""

import pickle
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import torch


def generate_cache_key(
    edges_csv: str, 
    include_node_types: List[str], 
    limit_nodes: Optional[int]
) -> str:
    """Generate a cache key based on input parameters."""
    # Create hash of parameters
    params = f"{edges_csv}_{sorted(include_node_types)}_{limit_nodes}"
    return hashlib.md5(params.encode()).hexdigest()


def get_cache_path(cache_key: str, cache_dir: str = "data/06_models/gnn_cache") -> Path:
    """Get the cache file path."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path / f"edge_index_{cache_key}.pkl"


def save_edge_index_cache(
    edge_index: torch.Tensor,
    node_metadata: Dict,
    node_features: torch.Tensor,
    cache_key: str,
    cache_dir: str = "data/06_models/gnn_cache"
) -> None:
    """Save computed edge index and metadata to cache."""
    cache_path = get_cache_path(cache_key, cache_dir)
    
    cache_data = {
        'edge_index': edge_index,
        'node_metadata': node_metadata,
        'node_features': node_features,
        'cache_key': cache_key
    }
    
    print(f"Saving edge index cache to: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)


def load_edge_index_cache(
    cache_key: str,
    cache_dir: str = "data/06_models/gnn_cache"
) -> Optional[Tuple[torch.Tensor, Dict, torch.Tensor]]:
    """Load cached edge index and metadata."""
    cache_path = get_cache_path(cache_key, cache_dir)
    
    if not cache_path.exists():
        return None
    
    print(f"Loading edge index cache from: {cache_path}")
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        return (
            cache_data['edge_index'],
            cache_data['node_metadata'], 
            cache_data['node_features']
        )
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None


def clear_cache(cache_dir: str = "data/06_models/gnn_cache") -> None:
    """Clear all cached edge indices."""
    cache_path = Path(cache_dir)
    if cache_path.exists():
        for cache_file in cache_path.glob("edge_index_*.pkl"):
            cache_file.unlink()
        print(f"Cleared cache directory: {cache_path}")