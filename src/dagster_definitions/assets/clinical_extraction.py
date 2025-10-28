"""
Dagster assets for extracting clinical evidence from medical notes.
"""

import os
from pathlib import Path
from typing import Dict

import mlflow
import pandas as pd
from dagster import AssetExecutionContext, MetadataValue, asset

from clinical_drug_discovery.lib.clinical_extraction import (
    download_mtsamples,
    extract_and_normalize_drug_disease_pairs,
    get_extraction_stats,
)


@asset(group_name="clinical_extraction", compute_kind="download")
def mtsamples_raw(context: AssetExecutionContext) -> pd.DataFrame:
    """Download MTSamples clinical notes dataset."""
    context.log.info("Downloading MTSamples clinical notes...")

    df = download_mtsamples()

    context.log.info(f"Downloaded {len(df):,} clinical notes")
    return df


@asset(group_name="clinical_extraction", compute_kind="nlp")
def clinical_drug_disease_pairs(
    context: AssetExecutionContext,
    mtsamples_raw: pd.DataFrame,
    drug_features_loaded: Dict,  # Ensure drugs are loaded in Neo4j
    disease_features_loaded: Dict,  # Ensure diseases are loaded in Neo4j
) -> pd.DataFrame:
    """Extract and normalize drug-disease co-occurrences using scispaCy NER."""
    context.log.info("Extracting and normalizing drug-disease pairs using NER...")
    
    # Log that data loading is complete
    context.log.info(f"Data loading complete - drugs: {len(drug_features_loaded)} entities, diseases: {len(disease_features_loaded)} entities")

    # Add environment diagnostics for spaCy model issues
    import sys
    import spacy
    context.log.info(f"Python executable: {sys.executable}")
    context.log.info(f"spaCy version: {spacy.__version__}")
    
    try:
        import spacy.util
        installed_models = spacy.util.get_installed_models()
        context.log.info(f"Available spaCy models: {installed_models}")
        
        # Test model loading in Dagster context
        if "en_ner_bc5cdr_md" in installed_models:
            context.log.info("✓ en_ner_bc5cdr_md found in installed models")
        else:
            context.log.error(f"✗ en_ner_bc5cdr_md NOT found. Available: {installed_models}")
            
    except Exception as e:
        context.log.error(f"Error checking spaCy models: {e}")

    # Set MLflow experiment
    mlflow.set_experiment("clinical-drug-discovery")

    with mlflow.start_run(run_name="clinical_extraction"):
        # Log parameters
        mlflow.log_params({
            "ner_model": "en_ner_bc5cdr_md",
            "min_frequency": 1,
            "max_note_length": 10000,
            "num_input_notes": len(mtsamples_raw),
        })

        result, stats = extract_and_normalize_drug_disease_pairs(
            notes_df=mtsamples_raw,
            memgraph_uri=os.getenv("MEMGRAPH_URI"),
            memgraph_user=os.getenv("MEMGRAPH_USER"),
            memgraph_password=os.getenv("MEMGRAPH_PASSWORD"),
            database=os.getenv("MEMGRAPH_DATABASE"),
            ner_model="en_ner_bc5cdr_md",
            min_frequency=1,
            max_note_length=10000,
        )

        context.log.info(f"Extracted and normalized {len(result):,} drug-disease pairs")

        # Log metrics to MLflow
        mlflow.log_metrics({
            "num_extracted_pairs": len(result),
            "num_unique_drugs": result['drug_id'].nunique() if len(result) > 0 else 0,
            "num_unique_diseases": result['disease_id'].nunique() if len(result) > 0 else 0,
        })

        # Log extraction stats
        if "extraction_stats" in stats:
            mlflow.log_metrics({
                f"extraction_{k}": v for k, v in stats["extraction_stats"].items()
                if isinstance(v, (int, float))
            })

        # Save to CSV for inspection
        output_file = "data/03_primary/clinical_drug_disease_pairs.csv"
        result.to_csv(output_file, index=False)

        # Log artifact to MLflow
        mlflow.log_artifact(output_file)

        # Get absolute path for display
        output_path = Path(output_file).resolve()
        context.log.info(f"Saved to: {output_path}")
        context.log.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    # Add metadata to show in Dagster UI
    context.add_output_metadata({
        "num_pairs": len(result),
        "extraction_stats": MetadataValue.json(stats.get("extraction_stats", {})),
        "normalization_stats": MetadataValue.json(stats.get("normalization_stats", {})),
        "output_file": MetadataValue.path(str(output_path)),
        "preview": MetadataValue.md(result.head(10).to_markdown()),
    })

    return result


@asset(group_name="clinical_extraction", compute_kind="stats")
def clinical_extraction_stats(
    context: AssetExecutionContext,
    clinical_drug_disease_pairs: pd.DataFrame,
) -> Dict[str, int]:
    """Get statistics about extracted clinical pairs."""
    context.log.info("Computing extraction statistics...")

    stats = get_extraction_stats(clinical_drug_disease_pairs)

    context.log.info(f"Stats: {stats}")
    return stats
