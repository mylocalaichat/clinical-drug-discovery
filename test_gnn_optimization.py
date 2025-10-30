#!/usr/bin/env python3
"""
Test script for optimized GNN edge index construction.
Compares original vs optimized approaches.
"""

import time
import pandas as pd
from pathlib import Path

def test_optimization_approaches():
    """Test different optimization approaches for edge index construction."""
    
    print("="*60)
    print("GNN EDGE INDEX OPTIMIZATION TEST")
    print("="*60)
    
    # Check if data exists
    edges_file = "data/01_raw/primekg/nodes.csv"
    if not Path(edges_file).exists():
        print(f"Error: {edges_file} not found. Please download PrimeKG data first.")
        return
    
    print(f"Loading sample data from: {edges_file}")
    
    # Test with progressively larger samples
    test_sizes = [1000, 5000, 15000]
    
    for test_size in test_sizes:
        print(f"\n{'='*40}")
        print(f"TESTING WITH {test_size:,} NODES")
        print(f"{'='*40}")
        
        # Test original approach (with cache disabled)
        print("\n1. Original approach (cache disabled):")
        start_time = time.time()
        
        try:
            from src.clinical_drug_discovery.lib.gnn_embeddings import load_graph_from_csv
            
            data_orig, metadata_orig = load_graph_from_csv(
                edges_csv=edges_file,
                limit_nodes=test_size,
                use_cache=False,  # Disable cache for fair comparison
                chunk_size=100000  # Use smaller chunks for testing
            )
            
            orig_time = time.time() - start_time
            print(f"  ✓ Original: {data_orig.num_nodes:,} nodes, {data_orig.num_edges:,} edges in {orig_time:.1f}s")
            
        except Exception as e:
            print(f"  ✗ Original failed: {e}")
            continue
        
        # Test optimized approach (with cache enabled)
        print("\n2. Optimized approach (with caching):")
        start_time = time.time()
        
        try:
            data_opt, metadata_opt = load_graph_from_csv(
                edges_csv=edges_file,
                limit_nodes=test_size,
                use_cache=True,  # Enable cache
                chunk_size=500000  # Larger chunks
            )
            
            opt_time = time.time() - start_time
            print(f"  ✓ Optimized: {data_opt.num_nodes:,} nodes, {data_opt.num_edges:,} edges in {opt_time:.1f}s")
            
            # Calculate speedup
            if orig_time > 0:
                speedup = orig_time / opt_time if opt_time > 0 else float('inf')
                print(f"  📈 Speedup: {speedup:.1f}x faster")
        
        except Exception as e:
            print(f"  ✗ Optimized failed: {e}")
        
        # Test subgraph sampling approach
        print("\n3. Subgraph sampling approach:")
        start_time = time.time()
        
        try:
            from src.clinical_drug_discovery.lib.gnn_sampling import sample_drug_disease_subgraph
            
            # Load edges for sampling
            edges_df = pd.read_csv(edges_file, nrows=500000)  # Sample for testing
            
            # Sample subgraph
            _ = sample_drug_disease_subgraph(
                edges_df=edges_df,
                max_nodes=test_size,
                drug_sample_size=min(100, test_size // 10),
                disease_sample_size=min(50, test_size // 20)
            )
            
            # Convert to graph
            data_sub, metadata_sub = load_graph_from_csv(
                edges_csv=edges_file,
                limit_nodes=test_size,
                use_cache=True
            )
            
            sub_time = time.time() - start_time
            print(f"  ✓ Subgraph: {data_sub.num_nodes:,} nodes, {data_sub.num_edges:,} edges in {sub_time:.1f}s")
            
        except Exception as e:
            print(f"  ✗ Subgraph sampling failed: {e}")


def test_progressive_loading():
    """Test progressive loading strategy."""
    
    print(f"\n{'='*60}")
    print("PROGRESSIVE LOADING STRATEGY TEST")
    print(f"{'='*60}")
    
    try:
        from src.clinical_drug_discovery.lib.gnn_sampling import progressive_loading_strategy
        
        results = progressive_loading_strategy(
            edges_csv="data/01_raw/primekg/nodes.csv",
            target_sizes=[1000, 5000, 15000],
            include_node_types=['drug', 'disease', 'gene/protein', 'pathway']
        )
        
        print("\n✓ Progressive loading completed successfully!")
        print(f"Generated {len(results)} graphs of increasing sizes")
        
        for i, (data, metadata) in enumerate(results):
            print(f"  Graph {i+1}: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
            
    except Exception as e:
        print(f"✗ Progressive loading failed: {e}")


def clear_test_cache():
    """Clear test cache files."""
    try:
        from src.clinical_drug_discovery.lib.gnn_cache import clear_cache
        clear_cache()
        print("✓ Test cache cleared")
    except Exception as e:
        print(f"Cache clear failed: {e}")


if __name__ == "__main__":
    try:
        # Clear cache first for fair testing
        clear_test_cache()
        
        # Run optimization tests
        test_optimization_approaches()
        
        # Test progressive loading
        test_progressive_loading()
        
        print(f"\n{'='*60}")
        print("✅ ALL TESTS COMPLETED!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()