"""
Dagster assets for extracting clinical evidence from medical notes.
"""

from typing import Dict

import pandas as pd
from dagster import AssetExecutionContext, asset

from clinical_drug_discovery.lib.clinical_extraction import (
    download_mtsamples,
    extract_drug_disease_pairs,
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
    """Extract drug-disease co-occurrences using scispaCy NER."""
    context.log.info("Extracting drug-disease pairs using NER...")

    result = extract_drug_disease_pairs(
        notes_df=mtsamples_raw,
        ner_model="en_ner_bc5cdr_md",
        min_frequency=1,  # Lowered from 2 to see if we get any pairs at all
        max_note_length=10000,
    )

    context.log.info(f"Extracted {len(result):,} unique drug-disease pairs")

    # Save to CSV for inspection
    result.to_csv("data/03_primary/clinical_drug_disease_pairs.csv", index=False)

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
