"""
Dagster assets for extracting clinical evidence from medical notes.
"""

import os
from pathlib import Path
from typing import Dict

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
) -> pd.DataFrame:
    """Extract and normalize drug-disease co-occurrences using scispaCy NER."""
    context.log.info("Extracting and normalizing drug-disease pairs using NER...")

    result, stats = extract_and_normalize_drug_disease_pairs(
        notes_df=mtsamples_raw,
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE"),
        ner_model="en_ner_bc5cdr_md",
        min_frequency=1,  # Lowered from 2 to see if we get any pairs at all
        max_note_length=10000,
    )

    context.log.info(f"Extracted and normalized {len(result):,} drug-disease pairs")

    # Save to CSV for inspection
    output_file = "data/03_primary/clinical_drug_disease_pairs.csv"
    result.to_csv(output_file, index=False)

    # Get absolute path for display
    output_path = Path(output_file).resolve()
    context.log.info(f"Saved to: {output_path}")

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
