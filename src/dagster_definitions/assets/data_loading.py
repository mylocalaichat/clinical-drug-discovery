"""
Dagster assets for loading PrimeKG data into Memgraph.
"""

import os
from typing import Any, Dict

import pandas as pd
from dagster import AssetExecutionContext, asset
from dotenv import load_dotenv

from clinical_drug_discovery.lib.data_loading import (
    download_primekg_data,
    setup_memgraph_database,
)

# Load environment variables
load_dotenv()


@asset(group_name="data_loading", compute_kind="download")
def primekg_download_status(context: AssetExecutionContext) -> Dict[str, Any]:
    """Download PrimeKG dataset from Harvard Dataverse."""
    download_dir = "data/01_raw/primekg"
    context.log.info(f"Downloading PrimeKG data to {download_dir}")

    result = download_primekg_data(download_dir)

    context.log.info(f"Downloaded: {result['downloaded_files']}")
    context.log.info(f"Skipped: {result['skipped_files']}")

    return result


@asset(group_name="data_loading", compute_kind="database")
def memgraph_database_ready(context: AssetExecutionContext, primekg_download_status: Dict) -> Dict[str, str]:
    """Setup Memgraph database and clear all existing data."""
    from neo4j import GraphDatabase
    
    context.log.info("Setting up Memgraph database...")
    
    # Connect to Memgraph and delete all existing nodes and edges
    auth = None
    memgraph_user = os.getenv("MEMGRAPH_USER", "")
    memgraph_password = os.getenv("MEMGRAPH_PASSWORD", "")
    if memgraph_user or memgraph_password:
        auth = (memgraph_user, memgraph_password)

    driver = GraphDatabase.driver(os.getenv("MEMGRAPH_URI"), auth=auth)
    
    try:
        with driver.session() as session:
            context.log.info("Deleting all existing nodes and edges...")
            
            # Delete all relationships first
            result = session.run("MATCH ()-[r]->() DELETE r")
            context.log.info("Deleted all relationships")
            
            # Then delete all nodes
            result = session.run("MATCH (n) DELETE n")
            context.log.info("Deleted all nodes")
            
            # Verify cleanup
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            edge_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            context.log.info(f"Database cleanup complete - Nodes: {node_count}, Edges: {edge_count}")
            
    finally:
        driver.close()

    result = setup_memgraph_database(
        memgraph_uri=os.getenv("MEMGRAPH_URI"),
        memgraph_user=os.getenv("MEMGRAPH_USER"),
        memgraph_password=os.getenv("MEMGRAPH_PASSWORD"),
        fresh_start=True,
    )

    context.log.info("Database setup complete")
    return result


@asset(group_name="data_loading", compute_kind="database")
def primekg_nodes_loaded(
    context: AssetExecutionContext,
    memgraph_database_ready: Dict
) -> Dict[str, Any]:
    """Load PrimeKG nodes into Memgraph using optimized bulk loading operations."""
    from clinical_drug_discovery.lib.data_loading import extract_nodes_from_edges, bulk_load_nodes_to_memgraph

    download_dir = "data/01_raw/primekg"
    edges_file = os.path.join(download_dir, "nodes.csv")  # Actually contains edges

    context.log.info(f"Loading edges to extract nodes from {edges_file}")
    edges_df = pd.read_csv(edges_file)
    context.log.info(f"Loaded {len(edges_df):,} edges")

    # Extract unique nodes
    nodes_df = extract_nodes_from_edges(edges_df)
    context.log.info(f"Extracted {len(nodes_df):,} unique nodes")

    # Use optimized bulk loading with proper transaction management
    context.log.info("Starting optimized bulk node loading...")
    loading_stats = bulk_load_nodes_to_memgraph(
        nodes_df=nodes_df,
        memgraph_uri=os.getenv("MEMGRAPH_URI"),
        memgraph_user=os.getenv("MEMGRAPH_USER", ""),
        memgraph_password=os.getenv("MEMGRAPH_PASSWORD", ""),
        batch_size=10000,  # Increased from 50000 for better memory management
        timeout=600  # Increased to 10 minutes for large batches
    )

    context.log.info(
        f"Bulk loading complete: {loading_stats['loaded_nodes']:,}/{loading_stats['total_nodes']:,} nodes "
        f"({loading_stats['success_rate']:.1f}% success rate) "
        f"in {loading_stats['loading_time_seconds']}s "
        f"at {loading_stats['loading_rate_nodes_per_second']} nodes/sec"
    )

    if loading_stats['failed_batches'] > 0:
        context.log.warning(f"{loading_stats['failed_batches']} batches failed during loading")

    return {
        "nodes_count": loading_stats['loaded_nodes'],
        "node_types": nodes_df["node_type"].value_counts().to_dict(),
        "download_dir": download_dir,
        **loading_stats  # Include all loading statistics
    }


@asset(group_name="data_loading", compute_kind="database")
def primekg_edges_loaded(
    context: AssetExecutionContext,
    primekg_nodes_loaded: Dict
) -> Dict[str, Any]:
    """Load PrimeKG edges/relationships into Memgraph using optimized bulk loading operations."""
    from clinical_drug_discovery.lib.data_loading import bulk_load_edges_to_memgraph

    download_dir = primekg_nodes_loaded["download_dir"]
    edges_file = os.path.join(download_dir, "nodes.csv")  # Actually contains edges

    context.log.info(f"Loading edges from {edges_file}")
    edges_df = pd.read_csv(edges_file)
    context.log.info(f"Loaded {len(edges_df):,} edges")

    # Use optimized bulk loading with proper transaction management
    context.log.info("Starting optimized bulk edge loading...")
    loading_stats = bulk_load_edges_to_memgraph(
        edges_df=edges_df,
        memgraph_uri=os.getenv("MEMGRAPH_URI"),
        memgraph_user=os.getenv("MEMGRAPH_USER", ""),
        memgraph_password=os.getenv("MEMGRAPH_PASSWORD", ""),
        batch_size=10000,  # Increased from 5000 for better performance
        timeout=600  # Increased to 10 minutes for large batches
    )

    context.log.info(
        f"Bulk loading complete: {loading_stats['loaded_edges']:,}/{loading_stats['total_edges']:,} edges "
        f"({loading_stats['success_rate']:.1f}% success rate) "
        f"in {loading_stats['loading_time_seconds']}s "
        f"at {loading_stats['loading_rate_edges_per_second']} edges/sec"
    )

    if loading_stats['failed_batches'] > 0:
        context.log.warning(f"{loading_stats['failed_batches']} batches failed during loading")

    return {
        "edges_count": loading_stats['loaded_edges'],
        "relation_types": edges_df["relation"].value_counts().to_dict(),
        "download_dir": download_dir,
        **loading_stats  # Include all loading statistics
    }


@asset(group_name="data_loading", compute_kind="database")
def drug_features_loaded(
    context: AssetExecutionContext,
    primekg_edges_loaded: Dict
) -> Dict[str, Any]:
    """Load drug features into Memgraph."""
    from neo4j import GraphDatabase

    download_dir = primekg_edges_loaded["download_dir"]
    # Note: edges.csv actually contains drug features (see download function comments)
    drug_features_file = os.path.join(download_dir, "edges.csv")

    context.log.info(f"Loading drug features from {drug_features_file}")

    # Drug features file is tab-separated
    drug_df = pd.read_csv(drug_features_file, sep="\t")
    context.log.info(f"Loaded {len(drug_df):,} drug feature records")

    # Connect to Memgraph
    auth = None
    memgraph_user = os.getenv("MEMGRAPH_USER", "")
    memgraph_password = os.getenv("MEMGRAPH_PASSWORD", "")
    if memgraph_user or memgraph_password:
        auth = (memgraph_user, memgraph_password)

    driver = GraphDatabase.driver(os.getenv("MEMGRAPH_URI"), auth=auth)

    try:
        with driver.session() as session:
            # Add drug features to existing nodes
            batch_size = 1000
            loaded_count = 0

            for i in range(0, len(drug_df), batch_size):
                batch = drug_df.iloc[i:i+batch_size]

                for _, row in batch.iterrows():
                    # Match drug node and add features
                    node_id = str(row.iloc[0]) if len(row) > 0 else None
                    if node_id:
                        # Create properties dictionary from all columns
                        props = {f"drug_{col}": str(val) for col, val in row.items() if pd.notna(val)}

                        session.run("""
                            MATCH (n:Node {node_id: $node_id})
                            SET n += $props,
                                n.has_drug_features = true
                        """, {
                            "node_id": node_id,
                            "props": props
                        })
                        loaded_count += 1

                context.log.info(f"Processed {min(i+batch_size, len(drug_df)):,}/{len(drug_df):,} drug features")

        context.log.info(f"Successfully loaded drug features for {loaded_count:,} drugs")

        return {
            "drug_features_count": loaded_count,
            "total_records": len(drug_df)
        }

    finally:
        driver.close()


@asset(group_name="data_loading", compute_kind="database")
def disease_features_loaded(
    context: AssetExecutionContext,
    primekg_edges_loaded: Dict
) -> Dict[str, Any]:
    """Load disease features into Memgraph."""
    from neo4j import GraphDatabase

    download_dir = primekg_edges_loaded["download_dir"]
    # Note: drug_features.csv actually contains disease features (see download function comments)
    disease_features_file = os.path.join(download_dir, "drug_features.csv")

    context.log.info(f"Loading disease features from {disease_features_file}")

    # Disease features file is tab-separated
    disease_df = pd.read_csv(disease_features_file, sep="\t")
    context.log.info(f"Loaded {len(disease_df):,} disease feature records")

    # Connect to Memgraph
    auth = None
    memgraph_user = os.getenv("MEMGRAPH_USER", "")
    memgraph_password = os.getenv("MEMGRAPH_PASSWORD", "")
    if memgraph_user or memgraph_password:
        auth = (memgraph_user, memgraph_password)

    driver = GraphDatabase.driver(os.getenv("MEMGRAPH_URI"), auth=auth)

    try:
        with driver.session() as session:
            # Add disease features to existing nodes
            batch_size = 1000
            loaded_count = 0

            for i in range(0, len(disease_df), batch_size):
                batch = disease_df.iloc[i:i+batch_size]

                for _, row in batch.iterrows():
                    # Match disease node and add features
                    node_id = str(row.iloc[0]) if len(row) > 0 else None
                    if node_id:
                        # Create properties dictionary from all columns
                        props = {f"disease_{col}": str(val) for col, val in row.items() if pd.notna(val)}

                        session.run("""
                            MATCH (n:Node {node_id: $node_id})
                            SET n += $props,
                                n.has_disease_features = true
                        """, {
                            "node_id": node_id,
                            "props": props
                        })
                        loaded_count += 1

                context.log.info(f"Processed {min(i+batch_size, len(disease_df)):,}/{len(disease_df):,} disease features")

        context.log.info(f"Successfully loaded disease features for {loaded_count:,} diseases")

        return {
            "disease_features_count": loaded_count,
            "total_records": len(disease_df)
        }

    finally:
        driver.close()
