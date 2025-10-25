"""
Data loading utilities for PrimeKG dataset.
"""

import os
import time
import urllib.request
from pathlib import Path
from typing import Dict

import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm


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
    # WARNING: The Harvard Dataverse file naming is confusing/swapped!
    # Here's what each file actually contains:
    #   - nodes.csv (ID 6180620) → kg.csv: edge triplets (relation, x_node, y_node)
    #   - edges.csv (ID 6180619) → drug features (tab-separated)
    #   - drug_features.csv (ID 6180618) → disease features (tab-separated)
    #   - disease_features.csv (ID 6180617) → basic node info (not used - we extract from kg.csv)
    files = {
        "nodes.csv": "https://dataverse.harvard.edu/api/access/datafile/6180620",  # Actually: kg.csv
        "edges.csv": "https://dataverse.harvard.edu/api/access/datafile/6180619",  # Actually: drug features
        "drug_features.csv": "https://dataverse.harvard.edu/api/access/datafile/6180618",  # Actually: disease features
        "disease_features.csv": "https://dataverse.harvard.edu/api/access/datafile/6180617",  # Actually: node list
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
            try:
                # Download with progress bar and proper headers
                def report_progress(block_num, block_size, total_size):
                    downloaded_mb = (block_num * block_size) / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    if total_size > 0:
                        percent = min(100, (block_num * block_size * 100) / total_size)
                        print(f"  {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end='\r')

                # Create request with proper headers
                req = urllib.request.Request(
                    url,
                    headers={
                        'User-Agent': 'PrimeKG-Clinical-Discovery/1.0',
                        'Accept': 'text/csv,application/octet-stream,*/*',
                    }
                )
                
                # Open URL with headers and save to file
                with urllib.request.urlopen(req) as response:
                    with open(filepath, 'wb') as f:
                        total_size = int(response.headers.get('Content-Length', 0))
                        block_size = 8192
                        bytes_downloaded = 0
                        
                        while True:
                            block = response.read(block_size)
                            if not block:
                                break
                            f.write(block)
                            bytes_downloaded += len(block)
                            report_progress(bytes_downloaded // block_size, block_size, total_size)
                
                file_size = filepath.stat().st_size / (1024 * 1024)
                print(f"  ✓ Downloaded {filename} ({file_size:.1f} MB)")
                downloaded.append(filename)

            except urllib.error.HTTPError as e:
                if e.code == 403:
                    print(f"  ⚠️  Access denied for {filename}. You may need to:")
                    print("     1. Visit https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM")
                    print("     2. Register an account and request access")
                    print(f"     3. Download {filename} manually to {download_path}")
                    print(f"  Skipping {filename} for now...")
                    skipped.append(filename)
                else:
                    print(f"  ✗ HTTP Error {e.code}: {e.reason}")
                    raise
            except Exception as e:
                print(f"  ✗ Failed to download {filename}: {e}")
                raise

    return {
        "download_dir": str(download_path),
        "downloaded_files": downloaded,
        "skipped_files": skipped,
        "total_files": len(files),
    }


def setup_neo4j_database(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    database: str = "primekg",
    fresh_start: bool = True,
) -> Dict[str, str]:
    """
    Setup Neo4j database (drop and recreate if fresh_start=True).

    Args:
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        database: Database name
        fresh_start: If True, drop and recreate database

    Returns:
        Dictionary with database status
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    try:
        with driver.session(database="system") as session:
            # Check if database exists
            result = session.run("SHOW DATABASES")
            databases = [record["name"] for record in result]

            if fresh_start and database in databases:
                print(f"Dropping existing database '{database}'...")
                try:
                    session.run(f"STOP DATABASE `{database}`")
                except Exception as e:
                    print(f"Note: {e}")

                session.run(f"DROP DATABASE `{database}` IF EXISTS")
                print(f"Dropped database '{database}'")

            # Create database if it doesn't exist
            if database not in databases or fresh_start:
                print(f"Creating database '{database}'...")
                session.run(f"CREATE DATABASE `{database}`")
                print(f"Database '{database}' created successfully!")

        # Wait for database to be ready
        print("Waiting for database to be ready...")
        max_retries = 30
        for i in range(max_retries):
            try:
                with driver.session(database=database) as session:
                    session.run("RETURN 1")
                    print("Database is ready!")
                    break
            except Exception as e:
                if i < max_retries - 1:
                    time.sleep(1)
                else:
                    raise Exception(
                        f"Database did not become ready after {max_retries} seconds: {e}"
                    )

        return {"status": "success", "database": database, "action": "created" if fresh_start else "verified"}

    finally:
        driver.close()


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


def load_nodes_to_neo4j(
    nodes_df: pd.DataFrame,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    database: str = "primekg",
) -> Dict[str, int]:
    """
    Load nodes from DataFrame into Neo4j.

    Args:
        nodes_df: DataFrame with columns: node_index, node_id, node_type, node_name, node_source
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        database: Database name

    Returns:
        Dictionary with loading statistics
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    try:
        print(f"\nTotal nodes to load: {len(nodes_df):,}")
        print("\nNode types distribution:")
        print(nodes_df['node_type'].value_counts())

        with driver.session(database=database) as session:
            # Create indices
            print("\nCreating indices...")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:PrimeKGNode) ON (n.node_index)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:PrimeKGNode) ON (n.node_id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:PrimeKGNode) ON (n.node_type)")

            # Load nodes in batches
            batch_size = 50000
            total_processed = 0
            print("\nLoading nodes in batches...")

            for i in tqdm(range(0, len(nodes_df), batch_size)):
                batch = nodes_df.iloc[i : i + batch_size]

                query = """
                UNWIND $rows AS row
                MERGE (n:PrimeKGNode {node_index: row.node_index})
                SET n.node_id = row.node_id,
                    n.node_type = row.node_type,
                    n.node_name = row.node_name,
                    n.node_source = row.node_source
                """
                result = session.run(query, {"rows": batch.to_dict("records")})
                result.consume()
                total_processed += len(batch)

            # Get final count
            result = session.run("MATCH (n:PrimeKGNode) RETURN count(n) as count").single()
            final_count = result['count']
            print(f"\nTotal nodes loaded: {final_count:,}")

            return {
                "nodes_processed": total_processed,
                "nodes_in_db": final_count
            }

    finally:
        driver.close()


def load_edges_to_neo4j(
    edges_df: pd.DataFrame,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    database: str = "primekg",
) -> Dict[str, int]:
    """
    Load edges from DataFrame into Neo4j.

    Args:
        edges_df: DataFrame with columns: x_index, y_index, relation, display_relation
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        database: Database name

    Returns:
        Dictionary with loading statistics
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    try:
        print(f"\nTotal edges to load: {len(edges_df):,}")
        print("\nRelation types distribution:")
        print(edges_df['relation'].value_counts())

        total_processed = 0
        relation_types = edges_df['relation'].unique()
        print(f"\nFound {len(relation_types)} unique relationship types")

        for rel_type in tqdm(relation_types, desc="Relationship types"):
            rel_edges = edges_df[edges_df['relation'] == rel_type]
            neo4j_rel_type = rel_type.upper().replace("-", "_").replace(" ", "_")

            batch_size = 5000
            for i in range(0, len(rel_edges), batch_size):
                batch = rel_edges.iloc[i : i + batch_size]

                with driver.session(database=database) as session:
                    query = f"""
                    UNWIND $rows AS row
                    MATCH (source:PrimeKGNode {{node_index: row.x_index}})
                    MATCH (target:PrimeKGNode {{node_index: row.y_index}})
                    MERGE (source)-[r:`{neo4j_rel_type}`]->(target)
                    SET r.display_relation = row.display_relation
                    """
                    result = session.run(query, {"rows": batch.to_dict("records")})
                    result.consume()
                    total_processed += len(batch)

        # Get final count
        with driver.session(database=database) as session:
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()
            final_count = result['count']
            print(f"\nTotal edges loaded: {final_count:,}")

        return {
            "edges_processed": total_processed,
            "edges_in_db": final_count
        }

    finally:
        driver.close()


def load_drug_features_to_neo4j(
    drug_features_df: pd.DataFrame,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    database: str = "primekg",
) -> Dict[str, int]:
    """
    Load drug features and attach to existing drug nodes.

    Args:
        drug_features_df: DataFrame with drug feature columns
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        database: Database name

    Returns:
        Dictionary with loading statistics
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    try:
        print(f"\nTotal drug features to load: {len(drug_features_df):,}")

        with driver.session(database=database) as session:
            batch_size = 1000
            total_processed = 0

            print("\nAttaching drug features to nodes...")
            for i in tqdm(range(0, len(drug_features_df), batch_size)):
                batch = drug_features_df.iloc[i : i + batch_size]
                batch_dict = batch.where(pd.notnull(batch), None).to_dict("records")

                query = """
                UNWIND $rows AS row
                MATCH (n:PrimeKGNode {node_index: row.node_index, node_type: 'drug'})
                SET n.description = row.description,
                    n.half_life = row.half_life,
                    n.indication = row.indication,
                    n.mechanism_of_action = row.mechanism_of_action,
                    n.protein_binding = row.protein_binding,
                    n.pharmacodynamics = row.pharmacodynamics,
                    n.state = row.state,
                    n.atc_1 = row.atc_1,
                    n.atc_2 = row.atc_2,
                    n.atc_3 = row.atc_3,
                    n.atc_4 = row.atc_4,
                    n.category = row.category,
                    n.group = row.group,
                    n.pathway = row.pathway,
                    n.molecular_weight = row.molecular_weight,
                    n.tpsa = row.tpsa,
                    n.clogp = row.clogp
                """
                result = session.run(query, {"rows": batch_dict})
                result.consume()
                total_processed += len(batch)

            print(f"\nTotal drug features loaded: {total_processed:,}")

            return {"drug_features_processed": total_processed}

    finally:
        driver.close()


def load_disease_features_to_neo4j(
    disease_features_df: pd.DataFrame,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    database: str = "primekg",
) -> Dict[str, int]:
    """
    Load disease features and attach to existing disease nodes.

    Args:
        disease_features_df: DataFrame with disease feature columns
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        database: Database name

    Returns:
        Dictionary with loading statistics
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    try:
        print(f"\nTotal disease features to load: {len(disease_features_df):,}")

        with driver.session(database=database) as session:
            batch_size = 1000
            total_processed = 0

            print("\nAttaching disease features to nodes...")
            for i in tqdm(range(0, len(disease_features_df), batch_size)):
                batch = disease_features_df.iloc[i : i + batch_size]
                batch_dict = batch.where(pd.notnull(batch), None).to_dict("records")

                query = """
                UNWIND $rows AS row
                MATCH (n:PrimeKGNode {node_index: row.node_index, node_type: 'disease'})
                SET n.mondo_id = row.mondo_id,
                    n.mondo_name = row.mondo_name,
                    n.group_id_bert = row.group_id_bert,
                    n.group_name_bert = row.group_name_bert,
                    n.mondo_definition = row.mondo_definition,
                    n.umls_description = row.umls_description,
                    n.orphanet_definition = row.orphanet_definition,
                    n.orphanet_prevalence = row.orphanet_prevalence,
                    n.orphanet_epidemiology = row.orphanet_epidemiology,
                    n.orphanet_clinical_description = row.orphanet_clinical_description,
                    n.orphanet_management_and_treatment = row.orphanet_management_and_treatment,
                    n.mayo_symptoms = row.mayo_symptoms,
                    n.mayo_causes = row.mayo_causes,
                    n.mayo_risk_factors = row.mayo_risk_factors,
                    n.mayo_complications = row.mayo_complications,
                    n.mayo_prevention = row.mayo_prevention,
                    n.mayo_see_doc = row.mayo_see_doc
                """
                result = session.run(query, {"rows": batch_dict})
                result.consume()
                total_processed += len(batch)

            print(f"\nTotal disease features loaded: {total_processed:,}")

            return {"disease_features_processed": total_processed}

    finally:
        driver.close()