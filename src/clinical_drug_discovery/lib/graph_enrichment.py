"""
Graph enrichment utilities for adding clinical evidence to Neo4j.
"""

from typing import Dict

import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm


def add_clinical_evidence_to_graph(
    clinical_pairs: pd.DataFrame,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    database: str = "primekg",
    min_frequency: int = 2,
) -> Dict[str, int]:
    """
    Add CLINICAL_EVIDENCE relationships to Neo4j graph.

    Args:
        clinical_pairs: DataFrame with columns: drug_name, disease_name, frequency
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        database: Database name
        min_frequency: Minimum frequency to create relationship

    Returns:
        Dictionary with enrichment statistics
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    added = 0
    skipped = 0

    try:
        print(f"\nAdding clinical evidence to Neo4j...")
        print(f"Total pairs to process: {len(clinical_pairs):,}")
        print(f"Minimum frequency threshold: {min_frequency}")

        for _, row in tqdm(clinical_pairs.iterrows(), total=len(clinical_pairs), desc="Adding clinical evidence"):
            if row['frequency'] < min_frequency:
                skipped += 1
                continue

            # Calculate confidence score (normalize frequency to 0-1 range)
            confidence = min(row['frequency'] / 10.0, 1.0)

            with driver.session(database=database) as session:
                query = """
                MATCH (drug:PrimeKGNode)
                WHERE drug.node_type = 'drug'
                  AND toLower(drug.node_name) CONTAINS toLower($drug_name)

                MATCH (disease:PrimeKGNode)
                WHERE disease.node_type = 'disease'
                  AND toLower(disease.node_name) CONTAINS toLower($disease_name)

                MERGE (drug)-[r:CLINICAL_EVIDENCE]->(disease)
                SET r.frequency = $frequency,
                    r.source = 'MTSamples',
                    r.confidence = $confidence
                RETURN drug.node_name as drug, disease.node_name as disease
                """

                try:
                    result = session.run(
                        query,
                        drug_name=row['drug_name'],
                        disease_name=row['disease_name'],
                        frequency=int(row['frequency']),
                        confidence=float(confidence)
                    )

                    if result.single():
                        added += 1
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1

        stats = {
            "clinical_relationships_added": added,
            "skipped": skipped,
            "total_processed": len(clinical_pairs)
        }

        print(f"\nClinical Evidence Added: {added:,}")
        print(f"Skipped: {skipped:,}")

        return stats

    finally:
        driver.close()


def validate_clinical_evidence(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    database: str = "primekg",
) -> Dict[str, int]:
    """
    Validate clinical evidence relationships in Neo4j.

    Args:
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        database: Database name

    Returns:
        Dictionary with validation statistics
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    try:
        with driver.session(database=database) as session:
            # Count clinical evidence relationships
            result = session.run("""
                MATCH ()-[r:CLINICAL_EVIDENCE]->()
                RETURN count(r) as total_clinical_evidence
            """).single()

            total_evidence = result['total_clinical_evidence']

            # Get statistics
            result = session.run("""
                MATCH ()-[r:CLINICAL_EVIDENCE]->()
                RETURN
                    min(r.frequency) as min_freq,
                    max(r.frequency) as max_freq,
                    avg(r.frequency) as avg_freq,
                    min(r.confidence) as min_conf,
                    max(r.confidence) as max_conf,
                    avg(r.confidence) as avg_conf
            """).single()

            stats = {
                "total_clinical_evidence": total_evidence,
                "min_frequency": int(result['min_freq']) if result['min_freq'] else 0,
                "max_frequency": int(result['max_freq']) if result['max_freq'] else 0,
                "avg_frequency": round(result['avg_freq'], 2) if result['avg_freq'] else 0,
                "min_confidence": round(result['min_conf'], 3) if result['min_conf'] else 0,
                "max_confidence": round(result['max_conf'], 3) if result['max_conf'] else 0,
                "avg_confidence": round(result['avg_conf'], 3) if result['avg_conf'] else 0,
            }

            print("\nClinical Evidence Validation:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

            # Get top pairs by frequency
            print("Top 10 Clinical Evidence pairs:")
            result = session.run("""
                MATCH (drug:PrimeKGNode)-[r:CLINICAL_EVIDENCE]->(disease:PrimeKGNode)
                RETURN drug.node_name as drug, disease.node_name as disease,
                       r.frequency as freq, r.confidence as conf
                ORDER BY r.frequency DESC
                LIMIT 10
            """)

            for record in result:
                print(f"  {record['drug']:20s} -> {record['disease']:30s} (freq: {record['freq']}, conf: {record['conf']:.2f})")

            return stats

    finally:
        driver.close()