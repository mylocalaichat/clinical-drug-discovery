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
def clinical_pairs_loaded(
    context: AssetExecutionContext,
    clinical_drug_disease_pairs: pd.DataFrame,
    disease_features_loaded: Dict,
    drug_features_loaded: Dict,
) -> Dict[str, int]:
    """Load clinical drug-disease pairs as CLINICAL_EVIDENCE relationships into Neo4j graph."""
    context.log.info("Loading clinical evidence pairs into Neo4j...")

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
            memgraph_uri=os.getenv("MEMGRAPH_URI"),
            memgraph_user=os.getenv("MEMGRAPH_USER"),
            memgraph_password=os.getenv("MEMGRAPH_PASSWORD"),
            database=os.getenv("MEMGRAPH_DATABASE"),
            min_score=0.1,
        )

        # Log metrics to MLflow
        mlflow.log_metrics({
            "clinical_relationships_added": result.get("clinical_relationships_added", 0),
            "relationships_with_positive_score": result.get("relationships_with_positive_score", 0),
            "relationships_with_negative_score": result.get("relationships_with_negative_score", 0),
        })

        context.log.info(
            f"Loaded {result['clinical_relationships_added']} clinical evidence relationships into Neo4j"
        )
        context.log.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    return result


@asset(group_name="graph_enrichment", compute_kind="neo4j")
def clinical_validation_stats(
    context: AssetExecutionContext,
    clinical_pairs_loaded: Dict,
) -> Dict[str, int]:
    """Validate clinical evidence relationships in Neo4j."""
    context.log.info("Validating clinical evidence...")

    result = validate_clinical_evidence(
        memgraph_uri=os.getenv("MEMGRAPH_URI"),
        memgraph_user=os.getenv("MEMGRAPH_USER"),
        memgraph_password=os.getenv("MEMGRAPH_PASSWORD"),
        database=os.getenv("MEMGRAPH_DATABASE"),
    )

    context.log.info(f"Validation: {result['total_clinical_evidence']} relationships found")
    return result
