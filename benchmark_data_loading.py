"""
Benchmark: Memgraph vs CSV loading for GNN training.

Compares performance of loading graph data from:
1. Memgraph (current approach)
2. CSV files from data_loading assets
"""

import time
import os
from typing import Dict, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data
from neo4j import GraphDatabase


def benchmark_memgraph_loading(
    memgraph_uri: str = "bolt://localhost:7687",
    include_node_types: list = None
) -> Tuple[Data, float, Dict]:
    """Benchmark loading graph from Memgraph."""

    if include_node_types is None:
        include_node_types = [
            'drug', 'disease', 'gene/protein', 'effect/phenotype',
            'pathway', 'biological_process', 'molecular_function', 'anatomy'
        ]

    start_time = time.time()

    driver = GraphDatabase.driver(memgraph_uri, auth=None)

    stats = {
        'method': 'memgraph',
        'query_times': {},
        'processing_times': {}
    }

    try:
        with driver.session() as session:
            # Query 1: Get nodes
            query_start = time.time()
            node_query = """
            MATCH (n:Node)
            WHERE n.node_type IN $include_types
            RETURN n.node_id as id,
                   n.node_name as name,
                   n.node_type as type
            """
            result = session.run(node_query, include_types=include_node_types)
            nodes_data = [
                {'id': r['id'], 'name': r['name'], 'type': r['type']}
                for r in result
            ]
            stats['query_times']['nodes'] = time.time() - query_start

            # Build node mapping
            process_start = time.time()
            nodes_df = pd.DataFrame(nodes_data)
            node_id_to_idx = {node_id: idx for idx, node_id in enumerate(nodes_df['id'].values)}
            stats['processing_times']['node_mapping'] = time.time() - process_start

            stats['num_nodes'] = len(nodes_df)

            # Query 2: Get edges
            query_start = time.time()
            node_ids = list(node_id_to_idx.keys())
            edges_query = """
            MATCH (a:Node)-[r:RELATES]->(b:Node)
            WHERE a.node_id IN $node_ids AND b.node_id IN $node_ids
            RETURN a.node_id as source, b.node_id as target
            """
            result = session.run(edges_query, node_ids=node_ids)
            edges_data = []
            for record in result:
                src_idx = node_id_to_idx.get(record['source'])
                tgt_idx = node_id_to_idx.get(record['target'])
                if src_idx is not None and tgt_idx is not None:
                    edges_data.append([src_idx, tgt_idx])
            stats['query_times']['edges'] = time.time() - query_start

            stats['num_edges'] = len(edges_data)

            # Build PyG Data
            process_start = time.time()
            edge_index = torch.tensor(edges_data, dtype=torch.long).t().contiguous()

            # One-hot encoding
            unique_types = sorted(nodes_df['type'].unique())
            type_to_idx = {node_type: idx for idx, node_type in enumerate(unique_types)}
            num_node_types = len(type_to_idx)

            x = torch.zeros((len(nodes_df), num_node_types), dtype=torch.float)
            for idx, node_type in enumerate(nodes_df['type'].values):
                type_idx = type_to_idx[node_type]
                x[idx, type_idx] = 1.0

            data = Data(x=x, edge_index=edge_index)
            stats['processing_times']['pyg_data'] = time.time() - process_start

    finally:
        driver.close()

    total_time = time.time() - start_time
    stats['total_time'] = total_time

    return data, total_time, stats


def benchmark_csv_loading(
    nodes_csv: str = "data/01_raw/primekg/nodes.csv",
    edges_csv: str = "data/01_raw/primekg/edges.csv",
    include_node_types: list = None
) -> Tuple[Data, float, Dict]:
    """Benchmark loading graph from CSV files."""

    if include_node_types is None:
        include_node_types = [
            'drug', 'disease', 'gene/protein', 'effect/phenotype',
            'pathway', 'biological_process', 'molecular_function', 'anatomy'
        ]

    start_time = time.time()

    stats = {
        'method': 'csv',
        'load_times': {},
        'processing_times': {}
    }

    # Load nodes CSV
    load_start = time.time()
    nodes_df = pd.read_csv(nodes_csv)
    stats['load_times']['nodes_csv'] = time.time() - load_start
    stats['nodes_csv_size_mb'] = round(os.path.getsize(nodes_csv) / (1024 * 1024), 2)

    # Filter nodes by type
    process_start = time.time()
    nodes_df = nodes_df[nodes_df['node_type'].isin(include_node_types)]
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(nodes_df['node_id'].values)}
    stats['processing_times']['node_filtering'] = time.time() - process_start
    stats['num_nodes'] = len(nodes_df)

    # Load edges CSV
    load_start = time.time()
    edges_df = pd.read_csv(edges_csv)
    stats['load_times']['edges_csv'] = time.time() - load_start
    stats['edges_csv_size_mb'] = round(os.path.getsize(edges_csv) / (1024 * 1024), 2)

    # Filter edges to only included nodes
    process_start = time.time()
    valid_node_ids = set(node_id_to_idx.keys())
    edges_df = edges_df[
        edges_df['x_id'].isin(valid_node_ids) &
        edges_df['y_id'].isin(valid_node_ids)
    ]

    # Map to PyG indices
    edges_data = []
    for _, row in edges_df.iterrows():
        src_idx = node_id_to_idx.get(row['x_id'])
        tgt_idx = node_id_to_idx.get(row['y_id'])
        if src_idx is not None and tgt_idx is not None:
            edges_data.append([src_idx, tgt_idx])

    stats['processing_times']['edge_filtering'] = time.time() - process_start
    stats['num_edges'] = len(edges_data)

    # Build PyG Data
    process_start = time.time()
    edge_index = torch.tensor(edges_data, dtype=torch.long).t().contiguous()

    # One-hot encoding
    unique_types = sorted(nodes_df['node_type'].unique())
    type_to_idx = {node_type: idx for idx, node_type in enumerate(unique_types)}
    num_node_types = len(type_to_idx)

    x = torch.zeros((len(nodes_df), num_node_types), dtype=torch.float)
    for idx, node_type in enumerate(nodes_df['node_type'].values):
        type_idx = type_to_idx[node_type]
        x[idx, type_idx] = 1.0

    data = Data(x=x, edge_index=edge_index)
    stats['processing_times']['pyg_data'] = time.time() - process_start

    total_time = time.time() - start_time
    stats['total_time'] = total_time

    return data, total_time, stats


def print_comparison(memgraph_stats: Dict, csv_stats: Dict):
    """Print detailed comparison of both methods."""

    print("\n" + "=" * 80)
    print("DATA LOADING BENCHMARK: MEMGRAPH vs CSV")
    print("=" * 80)

    print("\n📊 RESULTS SUMMARY")
    print("-" * 80)

    # Total time comparison
    memgraph_time = memgraph_stats['total_time']
    csv_time = csv_stats['total_time']
    speedup = csv_time / memgraph_time

    print(f"\nTotal Loading Time:")
    print(f"  Memgraph:  {memgraph_time:.2f}s")
    print(f"  CSV:       {csv_time:.2f}s")
    if speedup > 1:
        print(f"  Winner:    🏆 Memgraph is {speedup:.2f}x FASTER")
    else:
        print(f"  Winner:    🏆 CSV is {1/speedup:.2f}x FASTER")

    # Data loaded
    print(f"\nData Loaded:")
    print(f"  Nodes:     {memgraph_stats['num_nodes']:,} (both methods)")
    print(f"  Edges:     {memgraph_stats['num_edges']:,} (Memgraph) vs {csv_stats['num_edges']:,} (CSV)")

    print("\n⏱️  DETAILED BREAKDOWN")
    print("-" * 80)

    # Memgraph breakdown
    print("\nMemgraph Approach:")
    print(f"  Query nodes:         {memgraph_stats['query_times']['nodes']:.2f}s")
    print(f"  Build node mapping:  {memgraph_stats['processing_times']['node_mapping']:.2f}s")
    print(f"  Query edges:         {memgraph_stats['query_times']['edges']:.2f}s")
    print(f"  Build PyG Data:      {memgraph_stats['processing_times']['pyg_data']:.2f}s")
    print(f"  ────────────────────")
    print(f"  Total:               {memgraph_time:.2f}s")

    # CSV breakdown
    print("\nCSV Approach:")
    print(f"  Load nodes CSV:      {csv_stats['load_times']['nodes_csv']:.2f}s ({csv_stats['nodes_csv_size_mb']}MB)")
    print(f"  Filter nodes:        {csv_stats['processing_times']['node_filtering']:.2f}s")
    print(f"  Load edges CSV:      {csv_stats['load_times']['edges_csv']:.2f}s ({csv_stats['edges_csv_size_mb']}MB)")
    print(f"  Filter edges:        {csv_stats['processing_times']['edge_filtering']:.2f}s")
    print(f"  Build PyG Data:      {csv_stats['processing_times']['pyg_data']:.2f}s")
    print(f"  ────────────────────")
    print(f"  Total:               {csv_time:.2f}s")

    print("\n💡 ANALYSIS")
    print("-" * 80)

    # Bottlenecks
    print("\nBottlenecks:")
    memgraph_bottleneck = max(memgraph_stats['query_times'].items(), key=lambda x: x[1])
    csv_bottleneck = max(csv_stats['load_times'].items(), key=lambda x: x[1])

    print(f"  Memgraph: {memgraph_bottleneck[0]} ({memgraph_bottleneck[1]:.2f}s)")
    print(f"  CSV:      {csv_bottleneck[0]} ({csv_bottleneck[1]:.2f}s)")

    # Network overhead (Memgraph only)
    query_overhead = sum(memgraph_stats['query_times'].values())
    print(f"\nNetwork Overhead (Memgraph):")
    print(f"  Total query time:    {query_overhead:.2f}s ({query_overhead/memgraph_time*100:.1f}% of total)")

    # File I/O overhead (CSV only)
    io_overhead = sum(csv_stats['load_times'].values())
    print(f"\nFile I/O Overhead (CSV):")
    print(f"  Total load time:     {io_overhead:.2f}s ({io_overhead/csv_time*100:.1f}% of total)")

    print("\n✅ RECOMMENDATION")
    print("-" * 80)

    if speedup > 1.5:
        print(f"""
🏆 Use MEMGRAPH (Current Approach)

Memgraph is {speedup:.1f}x faster because:
  • Indexed queries on node_type (fast filtering)
  • Only returns relevant data (no post-filtering needed)
  • Cypher pattern matching is optimized for graphs
  • No need to load entire CSV into memory

CSV is slower because:
  • Must load ALL edges ({csv_stats['edges_csv_size_mb']}MB) before filtering
  • Pandas filtering on large DataFrames is slow
  • Sequential file I/O is limited by disk speed
""")
    elif speedup < 0.67:
        print(f"""
🏆 Use CSV FILES

CSV is {1/speedup:.1f}x faster because:
  • Sequential file reading is highly optimized
  • Pandas vectorized operations are efficient
  • No network overhead
  • Data is preprocessed and ready

Memgraph is slower because:
  • Network latency for queries
  • Multiple round-trips to database
  • Query parsing and execution overhead
""")
    else:
        print(f"""
⚖️ BOTH ARE COMPARABLE ({speedup:.2f}x difference)

Consider other factors:
  • Memgraph: Real-time data, complex queries, graph algorithms
  • CSV: Reproducibility, no database dependency, versioning

Current approach (Memgraph) is fine.
""")

    print("\n" + "=" * 80)


def main():
    """Run benchmark comparing both approaches."""

    print("Starting benchmark...")
    print("This will test loading ~124K nodes and millions of edges")
    print()

    # Check if files exist
    nodes_csv = "data/03_primary/primekg_nodes.csv"
    edges_csv = "data/03_primary/primekg_edges.csv"

    if not os.path.exists(nodes_csv):
        print(f"❌ Nodes CSV not found: {nodes_csv}")
        print("Run data_loading assets first to generate CSV files")
        return

    if not os.path.exists(edges_csv):
        print(f"❌ Edges CSV not found: {edges_csv}")
        print("Run data_loading assets first to generate CSV files")
        return

    # Benchmark Memgraph
    print("1️⃣  Benchmarking Memgraph loading...")
    try:
        data_memgraph, time_memgraph, stats_memgraph = benchmark_memgraph_loading()
        print(f"   ✓ Loaded {data_memgraph.num_nodes} nodes, {data_memgraph.num_edges} edges in {time_memgraph:.2f}s")
    except Exception as e:
        print(f"   ❌ Memgraph loading failed: {e}")
        return

    # Benchmark CSV
    print("\n2️⃣  Benchmarking CSV loading...")
    try:
        data_csv, time_csv, stats_csv = benchmark_csv_loading(nodes_csv, edges_csv)
        print(f"   ✓ Loaded {data_csv.num_nodes} nodes, {data_csv.num_edges} edges in {time_csv:.2f}s")
    except Exception as e:
        print(f"   ❌ CSV loading failed: {e}")
        return

    # Print comparison
    print_comparison(stats_memgraph, stats_csv)


if __name__ == "__main__":
    main()
