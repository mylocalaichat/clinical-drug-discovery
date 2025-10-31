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


@asset(
    group_name="data_loading",
    compute_kind="download",
    op_tags={"dagster/max_runtime": 1800}  # 30 minutes timeout for large file downloads
)
def download_data(context: AssetExecutionContext) -> Dict[str, Any]:
    """Download PrimeKG dataset from Harvard Dataverse."""
    from pathlib import Path

    download_dir = "data/01_raw/primekg"
    download_path = Path(download_dir)

    # Delete all existing CSV files before downloading
    if download_path.exists():
        context.log.info(f"Cleaning up existing files in {download_dir}")
        deleted_files = []
        for csv_file in download_path.glob("*.csv"):
            file_size_mb = csv_file.stat().st_size / (1024 * 1024)
            csv_file.unlink()
            deleted_files.append(f"{csv_file.name} ({file_size_mb:.1f} MB)")
            context.log.info(f"  Deleted: {csv_file.name} ({file_size_mb:.1f} MB)")

        if deleted_files:
            context.log.info(f"Cleaned up {len(deleted_files)} existing file(s)")
        else:
            context.log.info("No existing files to clean up")
    else:
        context.log.info("Download directory does not exist yet, will be created")

    context.log.info(f"Starting fresh download of PrimeKG data to {download_dir}")

    result = download_primekg_data(download_dir)

    context.log.info(f"Downloaded: {result['downloaded_files']}")
    context.log.info(f"Skipped: {result['skipped_files']}")

    return result


@asset(group_name="data_loading", compute_kind="database")
def memgraph_database_ready(context: AssetExecutionContext, download_data: Dict) -> Dict[str, str]:
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
            # First check what's currently in the database
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            edge_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            context.log.info(f"Found existing data - Nodes: {node_count:,}, Edges: {edge_count:,}")
            
            if node_count > 0 or edge_count > 0:
                context.log.info("Performing complete database cleanup...")
                
                # Method 1: Use DETACH DELETE for complete cleanup (faster)
                try:
                    context.log.info("Using DETACH DELETE for complete cleanup...")
                    session.run("MATCH (n) DETACH DELETE n", timeout=600)  # 10 minute timeout
                    context.log.info("✓ DETACH DELETE completed successfully")
                except Exception as e:
                    context.log.warning(f"DETACH DELETE failed: {e}")
                    context.log.info("Falling back to separate edge/node deletion...")
                    
                    # Method 2: Fallback - delete edges then nodes separately
                    try:
                        # Delete all relationships first
                        context.log.info("Deleting all relationships...")
                        session.run("MATCH ()-[r]->() DELETE r", timeout=300)
                        context.log.info("✓ All relationships deleted")
                        
                        # Then delete all nodes
                        context.log.info("Deleting all nodes...")
                        session.run("MATCH (n) DELETE n", timeout=300)
                        context.log.info("✓ All nodes deleted")
                    except Exception as e2:
                        context.log.error(f"Fallback deletion also failed: {e2}")
                        raise e2
            else:
                context.log.info("Database is already empty, no cleanup needed")
            
            # Verify cleanup is complete
            final_node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            final_edge_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            context.log.info(f"Database cleanup verified - Nodes: {final_node_count}, Edges: {final_edge_count}")
            
            if final_node_count > 0 or final_edge_count > 0:
                raise Exception(f"Database cleanup incomplete! Still found {final_node_count} nodes and {final_edge_count} edges")
            
    finally:
        driver.close()

    # Now setup the database (but skip the cleanup since we already did it)
    result = setup_memgraph_database(
        memgraph_uri=os.getenv("MEMGRAPH_URI"),
        memgraph_user=os.getenv("MEMGRAPH_USER"),
        memgraph_password=os.getenv("MEMGRAPH_PASSWORD"),
        fresh_start=False,  # Set to False since we already cleaned up
    )

    context.log.info("Database setup complete - ready for fresh data loading")
    return result


@asset(group_name="data_loading", compute_kind="database")
def primekg_nodes_loaded(
    context: AssetExecutionContext,
    memgraph_database_ready: Dict
) -> Dict[str, Any]:
    """Load PrimeKG nodes into Memgraph using optimized bulk loading operations."""
    from clinical_drug_discovery.lib.data_loading import extract_nodes_from_edges, bulk_load_nodes_to_memgraph
    from neo4j import GraphDatabase

    # Clean up existing nodes at the start
    context.log.info("Cleaning up existing nodes from previous runs...")
    memgraph_uri = os.getenv("MEMGRAPH_URI")
    memgraph_user = os.getenv("MEMGRAPH_USER", "")
    memgraph_password = os.getenv("MEMGRAPH_PASSWORD", "")

    auth = None
    if memgraph_user or memgraph_password:
        auth = (memgraph_user, memgraph_password)

    driver = GraphDatabase.driver(memgraph_uri, auth=auth)

    try:
        with driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            if node_count > 0:
                context.log.info(f"Deleting {node_count:,} existing nodes...")
                session.run("MATCH (n) DETACH DELETE n", timeout=600)
                context.log.info("✓ All nodes deleted")
            else:
                context.log.info("No existing nodes to clean up")
    finally:
        driver.close()

    download_dir = "data/01_raw/primekg"
    edges_file = os.path.join(download_dir, "kg.csv")  # Contains edge triplets

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
    from neo4j import GraphDatabase

    download_dir = primekg_nodes_loaded["download_dir"]
    edges_file = os.path.join(download_dir, "kg.csv")  # Contains edge triplets

    context.log.info(f"Loading edges from {edges_file}")
    edges_df = pd.read_csv(edges_file)
    context.log.info(f"Loaded {len(edges_df):,} edges")

    # Delete all existing edges before loading new ones
    context.log.info("Deleting all existing edges from Memgraph...")
    memgraph_uri = os.getenv("MEMGRAPH_URI")
    memgraph_user = os.getenv("MEMGRAPH_USER", "")
    memgraph_password = os.getenv("MEMGRAPH_PASSWORD", "")

    auth = None
    if memgraph_user or memgraph_password:
        auth = (memgraph_user, memgraph_password)

    driver = GraphDatabase.driver(memgraph_uri, auth=auth)
    deleted_count = 0

    try:
        with driver.session() as session:
            # Delete all relationships (edges) regardless of type
            # Note: Using DELETE (not DETACH DELETE) to keep nodes intact
            result = session.run("MATCH ()-[r]->() DELETE r RETURN count(r) as deleted_count")
            record = result.single()
            deleted_count = record["deleted_count"] if record else 0
            context.log.info(f"✓ Deleted {deleted_count:,} existing edges (all types)")
    finally:
        driver.close()

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

    context.add_output_metadata({
        "deleted_edges": deleted_count,
        "loaded_edges": loading_stats['loaded_edges'],
        "success_rate": f"{loading_stats['success_rate']:.1f}%",
        "loading_time_seconds": loading_stats['loading_time_seconds'],
    })

    return {
        "edges_count": loading_stats['loaded_edges'],
        "deleted_edges": deleted_count,
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

    # Clean up existing drug features at the start
    context.log.info("Cleaning up existing drug features from previous runs...")
    memgraph_uri = os.getenv("MEMGRAPH_URI")
    memgraph_user = os.getenv("MEMGRAPH_USER", "")
    memgraph_password = os.getenv("MEMGRAPH_PASSWORD", "")

    auth = None
    if memgraph_user or memgraph_password:
        auth = (memgraph_user, memgraph_password)

    driver = GraphDatabase.driver(memgraph_uri, auth=auth)

    try:
        with driver.session() as session:
            # Count nodes with drug features
            result = session.run("""
                MATCH (n:Node)
                WHERE n.has_drug_features = true
                RETURN count(n) as count
            """)
            count = result.single()["count"]

            if count > 0:
                context.log.info(f"Removing drug features from {count:,} nodes...")
                # Remove all drug_* properties and has_drug_features flag
                session.run("""
                    MATCH (n:Node)
                    WHERE n.has_drug_features = true
                    WITH n, [k IN keys(n) WHERE k STARTS WITH 'drug_' OR k = 'has_drug_features'] AS props_to_remove
                    UNWIND props_to_remove AS prop
                    REMOVE n[prop]
                """, timeout=300)
                context.log.info("✓ Drug features removed")
            else:
                context.log.info("No existing drug features to clean up")
    finally:
        driver.close()

    download_dir = primekg_edges_loaded["download_dir"]
    drug_features_file = os.path.join(download_dir, "drug_features.csv")

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
            batch_size = 10000  # Increased from 1000 for better performance
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

    # Clean up existing disease features at the start
    context.log.info("Cleaning up existing disease features from previous runs...")
    memgraph_uri = os.getenv("MEMGRAPH_URI")
    memgraph_user = os.getenv("MEMGRAPH_USER", "")
    memgraph_password = os.getenv("MEMGRAPH_PASSWORD", "")

    auth = None
    if memgraph_user or memgraph_password:
        auth = (memgraph_user, memgraph_password)

    driver = GraphDatabase.driver(memgraph_uri, auth=auth)

    try:
        with driver.session() as session:
            # Count nodes with disease features
            result = session.run("""
                MATCH (n:Node)
                WHERE n.has_disease_features = true
                RETURN count(n) as count
            """)
            count = result.single()["count"]

            if count > 0:
                context.log.info(f"Removing disease features from {count:,} nodes...")
                # Remove all disease_* properties and has_disease_features flag
                session.run("""
                    MATCH (n:Node)
                    WHERE n.has_disease_features = true
                    WITH n, [k IN keys(n) WHERE k STARTS WITH 'disease_' OR k = 'has_disease_features'] AS props_to_remove
                    UNWIND props_to_remove AS prop
                    REMOVE n[prop]
                """, timeout=300)
                context.log.info("✓ Disease features removed")
            else:
                context.log.info("No existing disease features to clean up")
    finally:
        driver.close()

    download_dir = primekg_edges_loaded["download_dir"]
    disease_features_file = os.path.join(download_dir, "disease_features.csv")

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
            batch_size = 10000  # Increased from 1000 for better performance
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
