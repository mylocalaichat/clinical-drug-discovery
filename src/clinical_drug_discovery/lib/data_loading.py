"""
Data loading utilities for PrimeKG dataset with Memgraph.
"""

import os
import time
import urllib.request
from pathlib import Path
from typing import Dict

import pandas as pd
from neo4j import GraphDatabase


def get_database_session(driver, database: str = None):
    """
    Create a database session for Memgraph.
    The database parameter is ignored since Memgraph uses a single database.
    """
    return driver.session()


def setup_memgraph_database(
    memgraph_uri: str,
    memgraph_user: str = "",
    memgraph_password: str = "",
    database: str = "memgraph",
    fresh_start: bool = True,
) -> Dict[str, str]:
    """
    Setup Memgraph database (clear all data if fresh_start=True).

    Args:
        memgraph_uri: Memgraph connection URI
        memgraph_user: Memgraph username (often empty for local)
        memgraph_password: Memgraph password (often empty for local)
        database: Database name (ignored for Memgraph)
        fresh_start: If True, clear all data

    Returns:
        Dictionary with database status
    """
    # Handle empty credentials for local Memgraph
    auth = None
    if memgraph_user or memgraph_password:
        auth = (memgraph_user, memgraph_password)
    
    driver = GraphDatabase.driver(memgraph_uri, auth=auth)

    try:
        print("Connecting to Memgraph...")
        
        if fresh_start:
            print("Clearing all data from Memgraph...")
            with driver.session() as session:
                try:
                    # First try to get a count of existing data
                    result = session.run("MATCH (n) RETURN count(n) as count LIMIT 1")
                    node_count = result.single()['count']
                    print(f"Found {node_count:,} nodes in database")
                    
                    if node_count > 0:
                        print("Clearing all nodes and relationships...")
                        session.run("MATCH (n) DETACH DELETE n", timeout=300)  # 5 minute timeout
                        print("✓ All data cleared from Memgraph")
                    else:
                        print("✓ Memgraph is already empty")
                        
                except Exception as e:
                    print(f"Warning: Could not clear existing data: {e}")
                    print("Continuing anyway - this might be okay if database is empty")

        # Test final connection
        print("Testing Memgraph connection...")
        with driver.session() as session:
            session.run("RETURN 1")
            print("✓ Memgraph is ready!")

        return {
            "status": "ready",
            "database": "memgraph",
            "database_type": "memgraph"
        }

    finally:
        driver.close()


def download_primekg_data(download_dir: str) -> Dict[str, str]:
    """
    Download PrimeKG dataset from Harvard Dataverse if not already present.

    Args:
        download_dir: Directory to download files to

    Returns:
        Dictionary with download status and file paths
    """
    # Create download directory if it doesn't exist
    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)

    # PrimeKG files and their download URLs from Harvard Dataverse
    # Files are saved with descriptive names that match their actual content
    files = {
        "kg.csv": "https://dataverse.harvard.edu/api/access/datafile/6180620",  # Edge triplets (relation, x_node, y_node)
        "drug_features.csv": "https://dataverse.harvard.edu/api/access/datafile/6180619",  # Drug features (tab-separated)
        "disease_features.csv": "https://dataverse.harvard.edu/api/access/datafile/6180618",  # Disease features (tab-separated)
        "nodes.csv": "https://dataverse.harvard.edu/api/access/datafile/6180617",  # Basic node info (not used - we extract from kg.csv)
    }

    downloaded = []
    skipped = []

    print(f"\nChecking PrimeKG files in: {download_path}")

    for filename, url in files.items():
        filepath = download_path / filename

        if filepath.exists():
            file_size = filepath.stat().st_size / (1024 * 1024)  # MB
            print(f"✓ {filename} already exists ({file_size:.1f} MB)")
            skipped.append(filename)
        else:
            print(f"Downloading {filename}...")

            # Retry logic for downloads
            max_retries = 3
            retry_delay = 5  # seconds

            for attempt in range(max_retries):
                try:
                    # Create request with proper headers
                    req = urllib.request.Request(
                        url,
                        headers={
                            'User-Agent': 'PrimeKG-Clinical-Discovery/1.0',
                            'Accept': 'text/csv,application/octet-stream,*/*',
                        }
                    )

                    # Open URL with headers and save to file
                    with urllib.request.urlopen(req, timeout=60) as response:
                        with open(filepath, 'wb') as f:
                            total_size = int(response.headers.get('Content-Length', 0))
                            total_mb = total_size / (1024 * 1024)
                            block_size = 8192 * 128  # 1MB blocks for fewer updates
                            bytes_downloaded = 0
                            last_reported_percent = -1

                            # Only print progress every 10%
                            while True:
                                block = response.read(block_size)
                                if not block:
                                    break
                                f.write(block)
                                bytes_downloaded += len(block)

                                # Only report progress every 10% or at end
                                if total_size > 0:
                                    percent = int((bytes_downloaded * 100) / total_size)
                                    if percent >= last_reported_percent + 10 or bytes_downloaded >= total_size:
                                        downloaded_mb = bytes_downloaded / (1024 * 1024)
                                        print(f"  {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
                                        last_reported_percent = percent

                    file_size = filepath.stat().st_size / (1024 * 1024)
                    print(f"  ✓ Downloaded {filename} ({file_size:.1f} MB)")
                    downloaded.append(filename)
                    break  # Success, exit retry loop

                except urllib.error.HTTPError as e:
                    if e.code == 403:
                        print(f"  ⚠️  Access denied for {filename}. You may need to:")
                        print("     1. Visit https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM")
                        print("     2. Register an account and request access")
                        print(f"     3. Download {filename} manually to {download_path}")
                        print(f"  Skipping {filename} for now...")
                        skipped.append(filename)
                        break  # Don't retry on 403
                    else:
                        if attempt < max_retries - 1:
                            print(f"  ⚠️  HTTP Error {e.code}: {e.reason} (attempt {attempt + 1}/{max_retries})")
                            print(f"  Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                        else:
                            print(f"  ✗ HTTP Error {e.code}: {e.reason} (all retries failed)")
                            raise

                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"  ⚠️  Failed: {e} (attempt {attempt + 1}/{max_retries})")
                        print(f"  Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        # Clean up partial download
                        if filepath.exists():
                            filepath.unlink()
                    else:
                        print(f"  ✗ Failed to download {filename} after {max_retries} attempts: {e}")
                        raise

    return {
        "download_dir": str(download_path),
        "downloaded_files": downloaded,
        "skipped_files": skipped,
        "total_files": len(files),
    }




def extract_nodes_from_edges(edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract unique nodes from PrimeKG edges/triplets DataFrame.

    Args:
        edges_df: DataFrame with columns: x_index, x_id, x_type, x_name, x_source,
                  y_index, y_id, y_type, y_name, y_source

    Returns:
        DataFrame with columns: node_index, node_id, node_type, node_name, node_source
    """
    print("Extracting unique nodes from edges...")

    # Extract x (source) nodes
    x_nodes = edges_df[['x_index', 'x_id', 'x_type', 'x_name', 'x_source']].copy()
    x_nodes.columns = ['node_index', 'node_id', 'node_type', 'node_name', 'node_source']

    # Extract y (target) nodes
    y_nodes = edges_df[['y_index', 'y_id', 'y_type', 'y_name', 'y_source']].copy()
    y_nodes.columns = ['node_index', 'node_id', 'node_type', 'node_name', 'node_source']

    # Combine and get unique nodes
    all_nodes = pd.concat([x_nodes, y_nodes], ignore_index=True)
    unique_nodes = all_nodes.drop_duplicates(subset=['node_index']).reset_index(drop=True)

    # Sort by node_index for consistency
    unique_nodes = unique_nodes.sort_values('node_index').reset_index(drop=True)

    print(f"Extracted {len(unique_nodes):,} unique nodes from {len(edges_df):,} edges")

    return unique_nodes





def bulk_load_nodes_to_memgraph(
    nodes_df: pd.DataFrame,
    memgraph_uri: str,
    memgraph_user: str = "",
    memgraph_password: str = "",
    batch_size: int = 5000,
    timeout: int = 300
) -> Dict[str, any]:
    """
    Efficiently load nodes into Memgraph using bulk operations with proper transaction management.
    
    Args:
        nodes_df: DataFrame with columns: node_index, node_id, node_type, node_name, node_source
        memgraph_uri: Memgraph connection URI
        memgraph_user: Memgraph username
        memgraph_password: Memgraph password
        batch_size: Number of nodes to process per batch (default: 5000)
        timeout: Transaction timeout in seconds (default: 300)
        
    Returns:
        Dictionary with loading statistics
    """
    import time
    from neo4j import GraphDatabase
    
    # Handle empty credentials for local Memgraph
    auth = None
    if memgraph_user or memgraph_password:
        auth = (memgraph_user, memgraph_password)
    
    driver = GraphDatabase.driver(memgraph_uri, auth=auth)
    
    try:
        start_time = time.time()
        total_loaded = 0
        failed_batches = 0
        
        print(f"Starting bulk node loading: {len(nodes_df):,} nodes in batches of {batch_size}")
        
        with driver.session() as session:
            # Create index for faster lookups
            try:
                session.run("CREATE INDEX ON :Node(node_id)")
                print("✓ Created index on Node(node_id)")
            except Exception as e:
                print(f"Index may already exist: {e}")
            
            for i in range(0, len(nodes_df), batch_size):
                batch_start = time.time()
                batch = nodes_df.iloc[i:i+batch_size]
                
                try:
                    # Convert batch to list of dictionaries for bulk operation
                    batch_data = []
                    for _, row in batch.iterrows():
                        batch_data.append({
                            "node_id": row["node_id"],
                            "node_index": int(row["node_index"]),
                            "node_type": row["node_type"],
                            "node_name": row["node_name"],
                            "node_source": row["node_source"]
                        })
                    
                    # Use CREATE instead of MERGE for better performance (assumes clean database)
                    bulk_query = """
                    UNWIND $batch_data AS node_data
                    CREATE (n:Node {
                        node_id: node_data.node_id,
                        node_index: node_data.node_index,
                        node_type: node_data.node_type,
                        node_name: node_data.node_name,
                        node_source: node_data.node_source
                    })
                    """
                    
                    # Retry logic: attempt up to 3 times
                    max_retries = 3
                    retry_count = 0
                    batch_successful = False
                    
                    while retry_count < max_retries and not batch_successful:
                        try:
                            # Execute with timeout and transaction management
                            session.run(bulk_query, {"batch_data": batch_data}, timeout=timeout)
                            batch_successful = True
                            
                            total_loaded += len(batch)
                            batch_time = time.time() - batch_start
                            
                            retry_msg = f" (retry {retry_count})" if retry_count > 0 else ""
                            print(
                                f"✓ Batch {i//batch_size + 1}{retry_msg}: {total_loaded:,}/{len(nodes_df):,} nodes "
                                f"({batch_time:.2f}s, {len(batch)/batch_time:.0f} nodes/sec)"
                            )
                            
                        except Exception as e:
                            retry_count += 1
                            if retry_count < max_retries:
                                print(f"⚠️  Batch {i//batch_size + 1} failed (attempt {retry_count}), retrying: {e}")
                                time.sleep(2 ** retry_count)  # Exponential backoff: 2s, 4s, 8s
                            else:
                                failed_batches += 1
                                print(f"✗ Batch {i//batch_size + 1} failed after {max_retries} attempts: {e}")
                                break
                    
                except Exception as e:
                    failed_batches += 1
                    print(f"✗ Critical error in batch {i//batch_size + 1}: {e}")
                    # Continue with next batch rather than failing completely
                    continue
        
        total_time = time.time() - start_time
        avg_rate = total_loaded / total_time if total_time > 0 else 0
        
        print(
            f"Bulk loading complete: {total_loaded:,}/{len(nodes_df):,} nodes loaded "
            f"in {total_time:.2f}s (avg rate: {avg_rate:.0f} nodes/sec)"
        )
        
        if failed_batches > 0:
            print(f"⚠️  {failed_batches} batches failed during loading")
        
        return {
            "total_nodes": len(nodes_df),
            "loaded_nodes": total_loaded,
            "failed_batches": failed_batches,
            "loading_time_seconds": round(total_time, 2),
            "loading_rate_nodes_per_second": round(avg_rate, 0),
            "batch_size": batch_size,
            "success_rate": round(total_loaded / len(nodes_df) * 100, 1) if len(nodes_df) > 0 else 0
        }
        
    finally:
        driver.close()


def bulk_load_edges_to_memgraph(
    edges_df: pd.DataFrame,
    memgraph_uri: str,
    memgraph_user: str = "",
    memgraph_password: str = "",
    batch_size: int = 5000,
    timeout: int = 300
) -> Dict[str, any]:
    """
    Efficiently load edges into Memgraph using bulk operations with proper transaction management.
    
    Args:
        edges_df: DataFrame with columns: x_id, y_id, relation, display_relation
        memgraph_uri: Memgraph connection URI
        memgraph_user: Memgraph username
        memgraph_password: Memgraph password
        batch_size: Number of edges to process per batch (default: 5000)
        timeout: Transaction timeout in seconds (default: 300)
        
    Returns:
        Dictionary with loading statistics
    """
    from neo4j import GraphDatabase
    
    # Handle empty credentials for local Memgraph
    auth = None
    if memgraph_user or memgraph_password:
        auth = (memgraph_user, memgraph_password)
    
    driver = GraphDatabase.driver(memgraph_uri, auth=auth)
    
    try:
        start_time = time.time()
        total_loaded = 0
        failed_batches = 0
        
        print(f"Starting bulk edge loading: {len(edges_df):,} edges in batches of {batch_size}")
        
        with driver.session() as session:
            for i in range(0, len(edges_df), batch_size):
                batch_start = time.time()
                batch = edges_df.iloc[i:i+batch_size]
                
                try:
                    # Convert batch to list of dictionaries for bulk operation
                    batch_data = []
                    for _, row in batch.iterrows():
                        batch_data.append({
                            "x_id": row["x_id"],
                            "y_id": row["y_id"],
                            "relation": row["relation"],
                            "display_relation": row["display_relation"]
                        })
                    
                    # Optimized bulk relationship creation - uses CREATE for performance
                    bulk_query = """
                    UNWIND $batch_data AS edge_data
                    MATCH (x:Node {node_id: edge_data.x_id})
                    MATCH (y:Node {node_id: edge_data.y_id})
                    CREATE (x)-[r:RELATES {
                        relation: edge_data.relation,
                        display_relation: edge_data.display_relation
                    }]->(y)
                    """
                    
                    # Retry logic: attempt up to 3 times
                    max_retries = 3
                    retry_count = 0
                    batch_successful = False
                    
                    while retry_count < max_retries and not batch_successful:
                        try:
                            # Execute with timeout and transaction management
                            session.run(bulk_query, {"batch_data": batch_data}, timeout=timeout)
                            batch_successful = True
                            
                            total_loaded += len(batch)
                            batch_time = time.time() - batch_start
                            
                            retry_msg = f" (retry {retry_count})" if retry_count > 0 else ""
                            print(
                                f"✓ Batch {i//batch_size + 1}{retry_msg}: {total_loaded:,}/{len(edges_df):,} edges "
                                f"({batch_time:.2f}s, {len(batch)/batch_time:.0f} edges/sec)"
                            )
                            
                        except Exception as e:
                            retry_count += 1
                            if retry_count < max_retries:
                                print(f"⚠️  Batch {i//batch_size + 1} failed (attempt {retry_count}), retrying: {e}")
                                time.sleep(2 ** retry_count)  # Exponential backoff: 2s, 4s, 8s
                            else:
                                failed_batches += 1
                                print(f"✗ Batch {i//batch_size + 1} failed after {max_retries} attempts: {e}")
                                break
                    
                except Exception as e:
                    failed_batches += 1
                    print(f"✗ Critical error in batch {i//batch_size + 1}: {e}")
                    # Continue with next batch rather than failing completely
                    continue
        
        total_time = time.time() - start_time
        avg_rate = total_loaded / total_time if total_time > 0 else 0
        
        print(
            f"Bulk loading complete: {total_loaded:,}/{len(edges_df):,} edges loaded "
            f"in {total_time:.2f}s (avg rate: {avg_rate:.0f} edges/sec)"
        )
        
        if failed_batches > 0:
            print(f"⚠️  {failed_batches} batches failed during loading")
        
        return {
            "total_edges": len(edges_df),
            "loaded_edges": total_loaded,
            "failed_batches": failed_batches,
            "loading_time_seconds": round(total_time, 2),
            "loading_rate_edges_per_second": round(avg_rate, 0),
            "batch_size": batch_size,
            "success_rate": round(total_loaded / len(edges_df) * 100, 1) if len(edges_df) > 0 else 0
        }
        
    finally:
        driver.close()
