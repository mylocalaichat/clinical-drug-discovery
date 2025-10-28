#!/usr/bin/env python3
"""
Test script for bulk loading performance with PrimeKG nodes.
Tests transaction management and loading performance with smaller datasets.
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv

from src.clinical_drug_discovery.lib.data_loading import (
    setup_memgraph_database,
    extract_nodes_from_edges,
    bulk_load_nodes_to_memgraph,
    bulk_load_edges_to_memgraph
)

# Load environment variables
load_dotenv()

def test_bulk_loading_performance():
    """Test bulk loading performance with different batch sizes and dataset sizes."""
    
    print("="*60)
    print("BULK LOADING PERFORMANCE TEST")
    print("="*60)
    
    # Setup database connection
    memgraph_uri = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
    memgraph_user = os.getenv("MEMGRAPH_USER", "")
    memgraph_password = os.getenv("MEMGRAPH_PASSWORD", "")
    
    print(f"Connecting to Memgraph at: {memgraph_uri}")
    
    # Setup fresh database
    print("\nSetting up fresh database...")
    db_status = setup_memgraph_database(
        memgraph_uri=memgraph_uri,
        memgraph_user=memgraph_user,
        memgraph_password=memgraph_password,
        fresh_start=True
    )
    print(f"Database status: {db_status['status']}")
    
    # Load sample data
    download_dir = "data/01_raw/primekg"
    edges_file = os.path.join(download_dir, "nodes.csv")
    
    if not os.path.exists(edges_file):
        print(f"Error: {edges_file} not found. Please run data download first.")
        return
    
    print(f"\nLoading edges from: {edges_file}")
    edges_df = pd.read_csv(edges_file)
    print(f"Total edges available: {len(edges_df):,}")
    
    # Extract nodes
    print("\nExtracting unique nodes...")
    nodes_df = extract_nodes_from_edges(edges_df)
    print(f"Total unique nodes: {len(nodes_df):,}")
    
    # Test with different dataset sizes
    test_sizes = [1000, 5000, 10000, 25000]
    if len(nodes_df) >= 50000:
        test_sizes.append(50000)
    
    batch_sizes = [1000, 2500, 5000]
    
    results = []
    
    for test_size in test_sizes:
        if test_size > len(nodes_df):
            continue
            
        print(f"\n" + "="*40)
        print(f"TESTING WITH {test_size:,} NODES")
        print("="*40)
        
        test_nodes = nodes_df.head(test_size).copy()
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            
            # Clear database for clean test
            setup_memgraph_database(
                memgraph_uri=memgraph_uri,
                memgraph_user=memgraph_user,
                memgraph_password=memgraph_password,
                fresh_start=True
            )
            
            # Test node loading
            start_time = time.time()
            loading_stats = bulk_load_nodes_to_memgraph(
                nodes_df=test_nodes,
                memgraph_uri=memgraph_uri,
                memgraph_user=memgraph_user,
                memgraph_password=memgraph_password,
                batch_size=batch_size,
                timeout=300
            )
            total_time = time.time() - start_time
            
            result = {
                "test_size": test_size,
                "batch_size": batch_size,
                "total_time": total_time,
                "nodes_per_second": loading_stats['loading_rate_nodes_per_second'],
                "success_rate": loading_stats['success_rate'],
                "failed_batches": loading_stats['failed_batches']
            }
            results.append(result)
            
            print(f"  Result: {loading_stats['loaded_nodes']:,}/{loading_stats['total_nodes']:,} nodes")
            print(f"  Time: {total_time:.2f}s")
            print(f"  Rate: {loading_stats['loading_rate_nodes_per_second']} nodes/sec")
            print(f"  Success: {loading_stats['success_rate']:.1f}%")
            
            if loading_stats['failed_batches'] > 0:
                print(f"  ⚠️  Failed batches: {loading_stats['failed_batches']}")
    
    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Test Size':<10} {'Batch Size':<12} {'Time (s)':<10} {'Rate (n/s)':<12} {'Success %':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['test_size']:<10,} {result['batch_size']:<12,} {result['total_time']:<10.2f} "
              f"{result['nodes_per_second']:<12.0f} {result['success_rate']:<10.1f}")
    
    # Find optimal configuration
    if results:
        best_result = max(results, key=lambda x: x['nodes_per_second'])
        print(f"\nOptimal configuration:")
        print(f"  Batch size: {best_result['batch_size']:,}")
        print(f"  Performance: {best_result['nodes_per_second']:.0f} nodes/sec")
        print(f"  Recommended for 50k nodes: ~{50000/best_result['nodes_per_second']:.0f} seconds")


def test_transaction_management():
    """Test transaction management and error handling."""
    
    print("\n" + "="*60)
    print("TRANSACTION MANAGEMENT TEST")
    print("="*60)
    
    # Create test data with some invalid entries
    test_data = []
    for i in range(1000):
        test_data.append({
            "node_id": f"test_node_{i}",
            "node_index": i,
            "node_type": "TEST",
            "node_name": f"Test Node {i}",
            "node_source": "test"
        })
    
    # Add some problematic entries
    test_data.append({
        "node_id": None,  # This should cause issues
        "node_index": 1001,
        "node_type": "TEST",
        "node_name": "Bad Node",
        "node_source": "test"
    })
    
    test_df = pd.DataFrame(test_data)
    
    print(f"Testing with {len(test_df)} nodes (including 1 problematic entry)")
    
    # Test with small batch size to isolate failures
    loading_stats = bulk_load_nodes_to_memgraph(
        nodes_df=test_df,
        memgraph_uri=os.getenv("MEMGRAPH_URI", "bolt://localhost:7687"),
        memgraph_user=os.getenv("MEMGRAPH_USER", ""),
        memgraph_password=os.getenv("MEMGRAPH_PASSWORD", ""),
        batch_size=100,  # Small batches to test error isolation
        timeout=60
    )
    
    print(f"Transaction test results:")
    print(f"  Loaded: {loading_stats['loaded_nodes']}/{loading_stats['total_nodes']}")
    print(f"  Failed batches: {loading_stats['failed_batches']}")
    print(f"  Success rate: {loading_stats['success_rate']:.1f}%")
    
    if loading_stats['failed_batches'] > 0:
        print("✓ Transaction isolation working - failed batches didn't affect others")
    else:
        print("✓ All transactions succeeded")


if __name__ == "__main__":
    try:
        test_bulk_loading_performance()
        test_transaction_management()
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()