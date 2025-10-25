"""
Dagster assets for loading PrimeKG data into Neo4j.
"""

import os
from typing import Any, Dict

import pandas as pd
from dagster import AssetExecutionContext, asset
from dotenv import load_dotenv

from clinical_drug_discovery.lib.data_loading import (
    download_primekg_data,
    load_disease_features_to_neo4j,
    load_drug_features_to_neo4j,
    load_edges_to_neo4j,
    load_nodes_to_neo4j,
    setup_neo4j_database,
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
def neo4j_database_ready(context: AssetExecutionContext, primekg_download_status: Dict) -> Dict[str, str]:
    """Setup Neo4j database."""
    context.log.info("Setting up Neo4j database...")

    result = setup_neo4j_database(
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE"),
        fresh_start=True,
    )

    context.log.info(f"Database setup: {result['action']}")
    return result


@asset(group_name="data_loading", compute_kind="neo4j")
def primekg_nodes_loaded(
    context: AssetExecutionContext,
    neo4j_database_ready: Dict,
    primekg_download_status: Dict,
) -> Dict[str, int]:
    """Load PrimeKG nodes into Neo4j by extracting them from the edges file."""
    context.log.info("Loading PrimeKG nodes...")

    # Read the kg.csv file (currently named nodes.csv) which contains edge triplets
    context.log.info("Reading PrimeKG edges file...")
    edges_df = pd.read_csv("data/01_raw/primekg/nodes.csv")
    context.log.info(f"Read {len(edges_df):,} edges")

    # Extract unique nodes from the edges
    from clinical_drug_discovery.lib.data_loading import extract_nodes_from_edges
    nodes_df = extract_nodes_from_edges(edges_df)
    context.log.info(f"Extracted {len(nodes_df):,} unique nodes")

    # Load nodes to Neo4j
    result = load_nodes_to_neo4j(
        nodes_df=nodes_df,
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE"),
    )

    context.log.info(f"Loaded {result['nodes_in_db']} nodes")
    return result


@asset(group_name="data_loading", compute_kind="neo4j")
def primekg_edges_loaded(
    context: AssetExecutionContext,
    primekg_nodes_loaded: Dict,
) -> Dict[str, int]:
    """Load PrimeKG edges into Neo4j."""
    context.log.info("Loading PrimeKG edges...")

    # Read the kg.csv file (currently named nodes.csv) which contains edge triplets
    context.log.info("Reading PrimeKG kg.csv (edges) file...")
    edges_df = pd.read_csv("data/01_raw/primekg/nodes.csv")
    context.log.info(f"Read {len(edges_df):,} edges")

    result = load_edges_to_neo4j(
        edges_df=edges_df,
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE"),
    )

    context.log.info(f"Loaded {result['edges_in_db']} edges")
    return result


@asset(group_name="data_loading", compute_kind="neo4j")
def drug_features_loaded(
    context: AssetExecutionContext,
    primekg_edges_loaded: Dict,
) -> Dict[str, int]:
    """Load drug features into Neo4j."""
    context.log.info("Loading drug features...")

    # Read drug features from edges.csv (tab-separated)
    # Note: Harvard Dataverse file naming is swapped - edges.csv contains drug features
    drug_features_df = pd.read_csv("data/01_raw/primekg/edges.csv", sep='\t')

    result = load_drug_features_to_neo4j(
        drug_features_df=drug_features_df,
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE"),
    )

    context.log.info(f"Loaded {result['drug_features_processed']} drug features")
    return result


@asset(group_name="data_loading", compute_kind="neo4j")
def disease_features_loaded(
    context: AssetExecutionContext,
    primekg_edges_loaded: Dict,
) -> Dict[str, int]:
    """Load disease features into Neo4j."""
    context.log.info("Loading disease features...")

    # Read disease features from drug_features.csv (tab-separated)
    # Note: Harvard Dataverse file naming is swapped - drug_features.csv contains disease features
    disease_features_df = pd.read_csv("data/01_raw/primekg/drug_features.csv", sep='\t')

    result = load_disease_features_to_neo4j(
        disease_features_df=disease_features_df,
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE"),
    )

    context.log.info(f"Loaded {result['disease_features_processed']} disease features")
    return result