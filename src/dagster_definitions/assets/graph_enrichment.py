"""
Dagster assets for enriching the Neo4j graph with clinical evidence.
"""

import os
from typing import Dict

import mlflow
import pandas as pd
from dagster import AssetExecutionContext, asset
from dotenv import load_dotenv

from clinical_drug_discovery.lib.graph_enrichment import (
    add_clinical_evidence_to_graph,
    validate_clinical_evidence,
)

# Load environment variables
load_dotenv()


@asset(group_name="graph_enrichment", compute_kind="neo4j")
def clinical_enrichment_stats(
    context: AssetExecutionContext,
    clinical_drug_disease_pairs: pd.DataFrame,
    disease_features_loaded: Dict,
    drug_features_loaded: Dict,
) -> Dict[str, int]:
    """Add CLINICAL_EVIDENCE relationships to Neo4j graph."""
    context.log.info("Adding clinical evidence to Neo4j...")

    # Set MLflow experiment
    mlflow.set_experiment("clinical-drug-discovery")

    with mlflow.start_run(run_name="graph_enrichment"):
        # Log parameters
        mlflow.log_params({
            "min_score": 0.1,
            "num_input_pairs": len(clinical_drug_disease_pairs),
        })

        result = add_clinical_evidence_to_graph(
            clinical_pairs=clinical_drug_disease_pairs,
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_user=os.getenv("NEO4J_USER"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE"),
            min_score=0.1,
        )

        # Log metrics to MLflow
        mlflow.log_metrics({
            "clinical_relationships_added": result.get("clinical_relationships_added", 0),
            "relationships_with_positive_score": result.get("relationships_with_positive_score", 0),
            "relationships_with_negative_score": result.get("relationships_with_negative_score", 0),
        })

        context.log.info(
            f"Added {result['clinical_relationships_added']} clinical evidence relationships"
        )
        context.log.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    return result


@asset(group_name="graph_enrichment", compute_kind="neo4j")
def clinical_validation_stats(
    context: AssetExecutionContext,
    clinical_enrichment_stats: Dict,
) -> Dict[str, int]:
    """Validate clinical evidence relationships in Neo4j."""
    context.log.info("Validating clinical evidence...")

    result = validate_clinical_evidence(
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE"),
    )

    context.log.info(f"Validation: {result['total_clinical_evidence']} relationships found")
    return result
