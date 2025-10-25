"""
Dagster assets for drug discovery queries and comparison with MLflow tracking.
"""

import os
from typing import Dict

import pandas as pd
from dagster import AssetExecutionContext, asset
from dotenv import load_dotenv

from clinical_drug_discovery.lib.drug_discovery import (
    compare_discovery_results,
    log_results_to_mlflow,
    query_base_drug_discovery,
    query_enhanced_drug_discovery,
)

# Load environment variables
load_dotenv()


# Test diseases configuration
TEST_DISEASES = [
    {"disease_id": "15564", "name": "Castleman disease"},
    {"disease_id": "8170", "name": "Ovarian cancer"},
]


@asset(group_name="drug_discovery", compute_kind="neo4j")
def drug_discovery_results(
    context: AssetExecutionContext,
    clinical_validation_stats: Dict,
    clinical_enrichment_stats: Dict,
) -> pd.DataFrame:
    """Run drug discovery for multiple diseases and aggregate results."""
    context.log.info("Running drug discovery queries...")

    all_results = []

    for disease_info in TEST_DISEASES:
        disease_id = disease_info['disease_id']
        disease_name = disease_info['name']

        context.log.info(f"Processing: {disease_name} (ID: {disease_id})")

        # Run base query (graph topology only)
        base_results = query_base_drug_discovery(
            disease_id=disease_id,
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_user=os.getenv("NEO4J_USER"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE"),
        )

        # Run enhanced query (with clinical evidence)
        enhanced_results = query_enhanced_drug_discovery(
            disease_id=disease_id,
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_user=os.getenv("NEO4J_USER"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE"),
        )

        # Compare results
        comparison = compare_discovery_results(base_results, enhanced_results)

        # Log to MLflow
        mlflow_info = log_results_to_mlflow(
            disease_id=disease_id,
            disease_name=disease_name,
            base_results=base_results,
            enhanced_results=enhanced_results,
            comparison=comparison,
            clinical_enrichment_stats=clinical_enrichment_stats,
            clinical_validation_stats=clinical_validation_stats,
        )

        # Add metadata
        if len(comparison) > 0:
            comparison['disease_id'] = disease_id
            comparison['disease_name'] = disease_name
            comparison['mlflow_run_id'] = mlflow_info['run_id']
            all_results.append(comparison)

            context.log.info(
                f"Found {len(comparison)} candidate drugs for {disease_name}"
            )

    # Aggregate all results
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)

        # Save to CSV
        output_path = "data/07_model_output/drug_discovery_results.csv"
        final_results.to_csv(output_path, index=False)
        context.log.info(f"Saved results to {output_path}")

        return final_results
    else:
        context.log.warning("No results found")
        return pd.DataFrame()
