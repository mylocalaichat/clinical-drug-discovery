"""
Clinical extraction utilities for drug-disease co-occurrences.
"""

from collections import Counter
from typing import Dict

import pandas as pd
import spacy
from tqdm import tqdm


def download_mtsamples():
    """
    Download MTSamples clinical notes dataset.

    Returns:
        DataFrame with clinical notes
    """
    print("\nDownloading MTSamples clinical notes...")
    # Try primary source first, fallback to alternative
    urls = [
        "https://raw.githubusercontent.com/yemiobolo/MTSamples/main/mtsamples.csv",
        "https://raw.githubusercontent.com/biolab/datasets/master/mtsamples.csv",
        "https://huggingface.co/datasets/mtsamples/mtsamples/raw/main/mtsamples.csv"
    ]
    
    df = None
    for url in urls:
        try:
            print(f"Trying: {url}")
            df = pd.read_csv(url)
            print(f"✓ Successfully downloaded from: {url}")
            break
        except Exception as e:
            print(f"✗ Failed to download from {url}: {e}")
            continue
    
    if df is None:
        print("⚠️ All MTSamples sources failed. Creating minimal sample data...")
        # Create a minimal sample dataset for testing
        df = pd.DataFrame({
            'description': [
                'Patient presents with chest pain and shortness of breath. Prescribed aspirin and nitroglycerin.',
                'Diabetic patient with high blood pressure. Administering metformin and lisinopril.',
                'Patient with severe depression. Starting treatment with sertraline.',
                'Chronic pain management with ibuprofen and acetaminophen.',
                'Heart failure patient receiving digoxin and furosemide therapy.'
            ],
            'medical_specialty': ['Cardiology', 'Endocrinology', 'Psychiatry', 'Pain Management', 'Cardiology'],
            'sample_name': ['Chest Pain Case', 'Diabetes Case', 'Depression Case', 'Pain Case', 'Heart Failure Case']
        })
        print("✓ Created minimal sample dataset with 5 clinical notes")

    print(f"Downloaded {len(df):,} clinical notes")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nSample categories:")
    if 'medical_specialty' in df.columns:
        print(df['medical_specialty'].value_counts().head(10))

    return df


def extract_drug_disease_pairs(
    notes_df: pd.DataFrame,
    ner_model: str = "en_ner_bc5cdr_md",
    min_frequency: int = 2,
    max_note_length: int = 10000,
) -> pd.DataFrame:
    """
    Extract drug-disease co-occurrences using scispaCy NER.

    Args:
        notes_df: DataFrame with clinical notes
        ner_model: Name of spaCy NER model to use
        min_frequency: Minimum frequency for a pair to be included
        max_note_length: Maximum characters to process per note

    Returns:
        DataFrame with columns: drug_name, disease_name, frequency
    """
    print(f"\nLoading NER model: {ner_model}...")
    try:
        nlp = spacy.load(ner_model)
    except OSError:
        print(f"Model {ner_model} not found. Please run: ./install_models.sh")
        raise

    pairs = []
    notes_processed = 0
    notes_with_entities = 0

    # Determine which column to use for text
    text_column = 'description' if 'description' in notes_df.columns else 'transcription'

    print(f"\nExtracting drug-disease pairs from {len(notes_df):,} notes...")
    print(f"Using column: {text_column}")

    for idx, row in tqdm(notes_df.iterrows(), total=len(notes_df), desc="Processing notes"):
        note = row[text_column]
        if pd.notna(note):
            notes_processed += 1

            # Limit text length to avoid processing very long documents
            note_text = str(note)[:max_note_length]

            # Run NER
            doc = nlp(note_text)

            # Extract entities
            drugs = [ent.text for ent in doc.ents if ent.label_ == "CHEMICAL"]
            diseases = [ent.text for ent in doc.ents if ent.label_ == "DISEASE"]

            if drugs and diseases:
                notes_with_entities += 1

            # Create co-occurrence pairs
            for drug in drugs:
                for disease in diseases:
                    pairs.append((drug.lower().strip(), disease.lower().strip()))

    print(f"\nNotes processed: {notes_processed:,}")
    print(f"Notes with drug-disease entities: {notes_with_entities:,}")
    print(f"Total drug-disease pairs (with duplicates): {len(pairs):,}")

    # Count frequencies
    pair_counts = Counter(pairs)
    print(f"Unique drug-disease pairs: {len(pair_counts):,}")

    # Convert to DataFrame
    filtered_pairs = [
        {"drug_name": drug, "disease_name": disease, "frequency": count}
        for (drug, disease), count in pair_counts.items()
        if count >= min_frequency
    ]

    if filtered_pairs:
        df = pd.DataFrame(filtered_pairs)
        df = df.sort_values('frequency', ascending=False)

        print(f"\nPairs with frequency >= {min_frequency}: {len(df):,}")
        print(f"\nTop 10 most frequent pairs:")
        print(df.head(10).to_string(index=False))
    else:
        # Create empty DataFrame with correct columns
        df = pd.DataFrame(columns=["drug_name", "disease_name", "frequency"])
        print(f"\nWarning: No pairs found with frequency >= {min_frequency}")

    return df


def get_extraction_stats(clinical_pairs_df: pd.DataFrame) -> Dict[str, int]:
    """
    Get statistics about extracted clinical pairs.

    Args:
        clinical_pairs_df: DataFrame with drug-disease pairs

    Returns:
        Dictionary with statistics
    """
    if len(clinical_pairs_df) == 0:
        # Handle empty DataFrame
        stats = {
            "total_pairs": 0,
            "unique_drugs": 0,
            "unique_diseases": 0,
            "total_occurrences": 0,
            "median_frequency": 0,
            "max_frequency": 0,
        }
        print("\nClinical Extraction Statistics:")
        print("  Warning: No drug-disease pairs found")
        for key, value in stats.items():
            print(f"  {key}: {value:,}")
    else:
        stats = {
            "total_pairs": len(clinical_pairs_df),
            "unique_drugs": int(clinical_pairs_df['drug_name'].nunique()),
            "unique_diseases": int(clinical_pairs_df['disease_name'].nunique()),
            "total_occurrences": int(clinical_pairs_df['frequency'].sum()),
            "median_frequency": int(clinical_pairs_df['frequency'].median()),
            "max_frequency": int(clinical_pairs_df['frequency'].max()),
        }

        print("\nClinical Extraction Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:,}")

    return stats