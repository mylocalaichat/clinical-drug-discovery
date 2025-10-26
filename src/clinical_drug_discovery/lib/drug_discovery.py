"""
Drug discovery utilities for querying Neo4j and comparing results.
"""

from typing import Dict, List

import mlflow
import pandas as pd
from neo4j import GraphDatabase


def query_base_drug_discovery(
    disease_id: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    database: str = "primekg",
) -> pd.DataFrame:
    """
    Run base topology-only drug discovery query.

    Args:
        disease_id: Target disease ID
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        database: Database name

    Returns:
        DataFrame with drug discovery results
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    try:
        print(f"\nRunning base discovery query for disease ID: {disease_id}")

        query = """
        WITH $disease_id as disease_id

        MATCH (target_disease:PrimeKGNode {node_id: disease_id})
        MATCH (target_disease)-[:DISEASE_PROTEIN]-(shared_protein:PrimeKGNode)
        WHERE shared_protein.node_type = 'gene/protein'

        MATCH (similar_disease:PrimeKGNode)-[:DISEASE_PROTEIN]-(shared_protein)
        WHERE similar_disease.node_type = 'disease'
          AND similar_disease.node_id <> target_disease.node_id

        MATCH (candidate_drug:PrimeKGNode)-[:INDICATION]-(similar_disease)
        WHERE candidate_drug.node_type = 'drug'
          AND NOT EXISTS {
            MATCH (candidate_drug)-[:INDICATION]-(target_disease)
          }

        WITH candidate_drug,
             count(DISTINCT similar_disease) as similar_disease_count,
             count(DISTINCT shared_protein) as shared_protein_count,
             (count(DISTINCT shared_protein) * 10) + count(DISTINCT similar_disease) as base_score

        RETURN candidate_drug.node_name as drug_name,
               base_score as score,
               shared_protein_count,
               similar_disease_count
        ORDER BY base_score DESC
        LIMIT 20
        """

        with driver.session(database=database) as session:
            result = session.run(query, disease_id=disease_id)
            df = pd.DataFrame([dict(record) for record in result])

        print(f"Found {len(df)} candidate drugs (base query)")
        if len(df) > 0:
            print("Top 5 drugs (base):")
            print(df.head().to_string(index=False))

        return df

    finally:
        driver.close()


def query_enhanced_drug_discovery(
    disease_id: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    database: str = "primekg",
) -> pd.DataFrame:
    """
    Run enhanced drug discovery query with clinical evidence.

    Args:
        disease_id: Target disease ID
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        database: Database name

    Returns:
        DataFrame with drug discovery results including clinical evidence
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    try:
        print(f"\nRunning enhanced discovery query for disease ID: {disease_id}")

        query = """
        WITH $disease_id as disease_id

        MATCH (target_disease:PrimeKGNode {node_id: disease_id})
        MATCH (target_disease)-[:DISEASE_PROTEIN]-(shared_protein:PrimeKGNode)
        WHERE shared_protein.node_type = 'gene/protein'

        MATCH (similar_disease:PrimeKGNode)-[:DISEASE_PROTEIN]-(shared_protein)
        WHERE similar_disease.node_type = 'disease'
          AND similar_disease.node_id <> target_disease.node_id

        MATCH (candidate_drug:PrimeKGNode)-[:INDICATION]-(similar_disease)
        WHERE candidate_drug.node_type = 'drug'
          AND NOT EXISTS {
            MATCH (candidate_drug)-[:INDICATION]-(target_disease)
          }

        WITH candidate_drug,
             count(DISTINCT similar_disease) as similar_disease_count,
             count(DISTINCT shared_protein) as shared_protein_count,
             (count(DISTINCT shared_protein) * 10) + count(DISTINCT similar_disease) as base_score

        OPTIONAL MATCH (candidate_drug)-[clinical:CLINICAL_EVIDENCE]->(target_disease)

        WITH candidate_drug,
             base_score,
             shared_protein_count,
             similar_disease_count,
             COALESCE(clinical.score, 0.0) as clinical_score,
             COALESCE(clinical.confidence, 0.0) as clinical_confidence,
             COALESCE(clinical.evidence_strength, 'none') as evidence_strength,
             // Enhanced score: base score + clinical evidence boost
             // Positive clinical scores get significant boost, negative scores get penalty
             base_score + (COALESCE(clinical.score, 0.0) * 50) as enhanced_score

        RETURN candidate_drug.node_name as drug_name,
               enhanced_score as score,
               base_score,
               clinical_score,
               clinical_confidence,
               evidence_strength,
               shared_protein_count,
               similar_disease_count,
               CASE 
                 WHEN clinical_score > 0 THEN 'Graph + Positive Clinical' 
                 WHEN clinical_score < 0 THEN 'Graph + Negative Clinical'
                 ELSE 'Graph Only' 
               END as evidence_type
        ORDER BY enhanced_score DESC
        LIMIT 20
        """

        with driver.session(database=database) as session:
            result = session.run(query, disease_id=disease_id)
            df = pd.DataFrame([dict(record) for record in result])

        print(f"Found {len(df)} candidate drugs (enhanced query)")
        if len(df) > 0:
            print("Top 5 drugs (enhanced):")
            print(df[['drug_name', 'score', 'base_score', 'clinical_score', 'evidence_type']].head().to_string(index=False))

        return df

    finally:
        driver.close()


def compare_discovery_results(
    base_results: pd.DataFrame,
    enhanced_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare base and enhanced discovery results.

    Args:
        base_results: Results from base query
        enhanced_results: Results from enhanced query

    Returns:
        DataFrame with comparison
    """
    print("\nComparing base vs enhanced results...")

    if len(enhanced_results) == 0:
        print("No enhanced results to compare")
        return pd.DataFrame()

    comparison = enhanced_results.merge(
        base_results[['drug_name', 'score']],
        on='drug_name',
        how='left',
        suffixes=('_enhanced', '_base')
    )

    comparison['score_improvement'] = comparison['score_enhanced'] - comparison['score_base'].fillna(0)
    comparison['improvement_pct'] = (comparison['score_improvement'] / comparison['score_base'].fillna(1) * 100).round(1)

    comparison = comparison.sort_values('score_enhanced', ascending=False)

    print("Comparison Summary:")
    print(f"Total drugs in enhanced results: {len(comparison)}")
    drugs_with_clinical = len(comparison[comparison['clinical_score'] != 0])
    print(f"Drugs with clinical evidence: {drugs_with_clinical}")

    if drugs_with_clinical > 0:
        print("Top 5 drugs with clinical evidence:")
        clinical_df = comparison[comparison['clinical_score'] != 0][
            ['drug_name', 'score_enhanced', 'score_improvement', 'clinical_score', 'evidence_strength']
        ].head()
        print(clinical_df.to_string(index=False))

    return comparison


def log_results_to_mlflow(
    disease_id: str,
    disease_name: str,
    base_results: pd.DataFrame,
    enhanced_results: pd.DataFrame,
    comparison: pd.DataFrame,
    clinical_enrichment_stats: Dict,
    clinical_validation_stats: Dict,
) -> Dict[str, str]:
    """
    Log discovery results to MLflow.

    Args:
        disease_id: Target disease ID
        disease_name: Target disease name
        base_results: Base query results
        enhanced_results: Enhanced query results
        comparison: Comparison DataFrame
        clinical_enrichment_stats: Stats from graph enrichment
        clinical_validation_stats: Validation stats

    Returns:
        Dictionary with MLflow run info
    """
    print("\nLogging results to MLflow...")

    mlflow.set_experiment("clinical-drug-discovery")

    with mlflow.start_run(run_name=f"discovery_{disease_name}") as run:
        # Log parameters
        mlflow.log_param("disease_id", disease_id)
        mlflow.log_param("disease_name", disease_name)

        # Log clinical enrichment metrics
        mlflow.log_metrics({
            "clinical_relationships_added": clinical_enrichment_stats.get("clinical_relationships_added", 0),
            "total_clinical_evidence": clinical_validation_stats.get("total_clinical_evidence", 0),
        })

        # Log discovery metrics
        mlflow.log_metrics({
            "base_candidates_found": len(base_results),
            "enhanced_candidates_found": len(enhanced_results),
            "candidates_with_clinical_evidence": len(comparison[comparison['clinical_score'] != 0]) if len(comparison) > 0 else 0,
        })

        if len(comparison) > 0:
            mlflow.log_metrics({
                "max_score_improvement": float(comparison['score_improvement'].max()),
                "avg_score_improvement": float(comparison['score_improvement'].mean()),
            })

        # Log result tables as artifacts
        base_results.to_csv("base_results.csv", index=False)
        mlflow.log_artifact("base_results.csv")

        enhanced_results.to_csv("enhanced_results.csv", index=False)
        mlflow.log_artifact("enhanced_results.csv")

        if len(comparison) > 0:
            comparison.to_csv("comparison.csv", index=False)
            mlflow.log_artifact("comparison.csv")

        print(f"MLflow run ID: {run.info.run_id}")
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "tracking_uri": mlflow.get_tracking_uri(),
        }


def run_multi_disease_discovery(
    disease_list: List[Dict[str, str]],
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    database: str,
    clinical_enrichment_stats: Dict,
    clinical_validation_stats: Dict,
) -> pd.DataFrame:
    """
    Run discovery for multiple diseases and aggregate results.

    Args:
        disease_list: List of dicts with disease_id and name
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        database: Database name
        clinical_enrichment_stats: Stats from enrichment
        clinical_validation_stats: Validation stats

    Returns:
        DataFrame with aggregated results
    """
    all_results = []

    for disease_info in disease_list:
        disease_id = disease_info['disease_id']
        disease_name = disease_info['name']

        print(f"\n{'='*60}")
        print(f"Processing: {disease_name} (ID: {disease_id})")
        print(f"{'='*60}")

        # Run queries
        base_results = query_base_drug_discovery(disease_id, neo4j_uri, neo4j_user, neo4j_password, database)
        enhanced_results = query_enhanced_drug_discovery(disease_id, neo4j_uri, neo4j_user, neo4j_password, database)
        comparison = compare_discovery_results(base_results, enhanced_results)

        # Log to MLflow
        mlflow_info = log_results_to_mlflow(
            disease_id,
            disease_name,
            base_results,
            enhanced_results,
            comparison,
            clinical_enrichment_stats,
            clinical_validation_stats
        )

        # Aggregate
        if len(comparison) > 0:
            comparison['disease_id'] = disease_id
            comparison['disease_name'] = disease_name
            comparison['mlflow_run_id'] = mlflow_info['run_id']
            all_results.append(comparison)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()