"""
Clinical extraction utilities for drug-disease co-occurrences.
"""

from collections import Counter
from typing import Dict, Tuple

import pandas as pd
import spacy
from tqdm import tqdm

from .name_matching import create_name_matcher


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
                'Patient presents with chest pain and shortness of breath. Prescribed aspirin and nitroglycerin. Patient improved significantly with treatment.',
                'Diabetic patient with high blood pressure. Administering metformin and lisinopril. Blood sugar well controlled and BP normalized.',
                'Patient with severe depression. Starting treatment with sertraline. Patient responded well to medication with marked improvement.',
                'Chronic pain management with ibuprofen and acetaminophen. Patient reports good pain relief and functional improvement.',
                'Heart failure patient receiving digoxin and furosemide therapy. Excellent response with reduced symptoms.',
                'Patient had adverse reaction to aspirin causing stomach bleeding. Discontinued immediately.',
                'Metformin caused severe nausea and was ineffective for glucose control. Switched to alternative therapy.',
                'Sertraline provided some improvement in depression but patient experienced side effects.',
                'Ibuprofen helps with pain but patient developed gastric irritation. Monitoring closely.',
                'Patient with chest pain treated with aspirin. Significant improvement in symptoms and recovery.'
            ],
            'medical_specialty': ['Cardiology', 'Endocrinology', 'Psychiatry', 'Pain Management', 'Cardiology', 
                                'Cardiology', 'Endocrinology', 'Psychiatry', 'Pain Management', 'Cardiology'],
            'sample_name': ['Chest Pain Case 1', 'Diabetes Case 1', 'Depression Case 1', 'Pain Case 1', 'Heart Failure Case',
                          'Chest Pain Case 2', 'Diabetes Case 2', 'Depression Case 2', 'Pain Case 2', 'Chest Pain Case 3']
        })
        print("✓ Created sample dataset with varied association types (10 clinical notes)")

    print(f"Downloaded {len(df):,} clinical notes")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nSample categories:")
    if 'medical_specialty' in df.columns:
        print(df['medical_specialty'].value_counts().head(10))

    return df


def classify_association(context: str, drug: str, disease: str) -> str:
    """
    Classify the drug-disease association as positive, negative, or neutral.

    Args:
        context: Text context containing the drug and disease mentions
        drug: Drug name
        disease: Disease name

    Returns:
        'positive', 'negative', or 'neutral'
    """
    context_lower = context.lower()

    # Positive indicators (drug helped/improved condition)
    positive_keywords = [
        'improved', 'resolved', 'responded', 'successful', 'effective',
        'helped', 'relief', 'recovery', 'better', 'cured', 'controlled',
        'stabilized', 'decreased', 'reduced', 'alleviated', 'managed',
        'remission', 'improvement', 'benefited', 'favorable', 'treated successfully'
    ]

    # Negative indicators (adverse effect, didn't work, worsened)
    negative_keywords = [
        'adverse', 'failed', 'ineffective', 'worsened', 'allergic', 'allergy',
        'reaction', 'side effect', 'toxicity', 'discontinued', 'stopped',
        'no response', 'unresponsive', 'resistant', 'intolerant', 'contraindicated',
        'exacerbated', 'complications', 'deteriorated', 'aggravated', 'did not respond',
        'no improvement', 'no benefit', 'withdrew', 'ceased due to'
    ]

    # Count indicators
    positive_score = sum(1 for kw in positive_keywords if kw in context_lower)
    negative_score = sum(1 for kw in negative_keywords if kw in context_lower)

    # Classify based on scores
    if positive_score > negative_score and positive_score > 0:
        return 'positive'
    elif negative_score > positive_score and negative_score > 0:
        return 'negative'
    else:
        return 'neutral'


def extract_drug_disease_pairs(
    notes_df: pd.DataFrame,
    ner_model: str = "en_ner_bc5cdr_md",
    min_frequency: int = 2,
    max_note_length: int = 10000,
) -> pd.DataFrame:
    """
    Extract drug-disease co-occurrences with effectiveness scores using scispaCy NER.

    Args:
        notes_df: DataFrame with clinical notes
        ner_model: Name of spaCy NER model to use
        min_frequency: Minimum frequency for a pair to be included
        max_note_length: Maximum characters to process per note

    Returns:
        DataFrame with columns: drug, disease, score
        where score is 1 if positive association, 0 if negative/neutral
    """
    print(f"\nLoading NER model: {ner_model}...")
    try:
        nlp = spacy.load(ner_model)
    except OSError:
        print(f"Model {ner_model} not found. Please run: ./install_models.sh")
        raise

    # Store pairs with association types: (drug, disease, association_type)
    pairs_with_association = []
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

            # Extract entities with their positions
            drugs = [(ent.text, ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ == "CHEMICAL"]
            diseases = [(ent.text, ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ == "DISEASE"]

            if drugs and diseases:
                notes_with_entities += 1

            # Create co-occurrence pairs with context-based classification
            for drug_text, drug_start, drug_end in drugs:
                for disease_text, disease_start, disease_end in diseases:
                    # Extract context window around both entities
                    # Get the sentence(s) containing both entities
                    min_pos = min(drug_start, disease_start)
                    max_pos = max(drug_end, disease_end)

                    # Expand context window (200 chars before and after)
                    context_start = max(0, min_pos - 200)
                    context_end = min(len(note_text), max_pos + 200)
                    context = note_text[context_start:context_end]

                    # Classify the association
                    association_type = classify_association(context, drug_text, disease_text)

                    # Store the pair with association type
                    pairs_with_association.append((
                        drug_text.lower().strip(),
                        disease_text.lower().strip(),
                        association_type
                    ))

    print(f"\nNotes processed: {notes_processed:,}")
    print(f"Notes with drug-disease entities: {notes_with_entities:,}")
    print(f"Total drug-disease pairs (with duplicates): {len(pairs_with_association):,}")

    # Count frequencies by (drug, disease, association_type)
    association_counts = Counter(pairs_with_association)
    print(f"Unique drug-disease-association triplets: {len(association_counts):,}")

    # Aggregate counts by drug-disease pair
    pair_data = {}
    for (drug, disease, assoc_type), count in association_counts.items():
        key = (drug, disease)
        if key not in pair_data:
            pair_data[key] = {
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        pair_data[key][f'{assoc_type}_count'] += count

    # Convert to DataFrame with proportional scoring
    filtered_pairs = []
    for (drug, disease), counts in pair_data.items():
        total_count = counts['positive_count'] + counts['negative_count'] + counts['neutral_count']

        if total_count >= min_frequency:
            # Calculate proportional score based on evidence strength
            # Score ranges from -1 to +1, proportional to evidence magnitude
            positive_weight = counts['positive_count']
            negative_weight = counts['negative_count']
            neutral_weight = counts['neutral_count'] * 0.1  # Neutral has minimal impact
            
            # Net evidence score (positive - negative, with neutral slightly negative)
            net_evidence = positive_weight - negative_weight - neutral_weight
            
            # Normalize by total evidence to get proportional strength
            # Score range: approximately -1 to +1
            if total_count > 0:
                score = net_evidence / total_count
                # Ensure score is within reasonable bounds
                score = max(-1.0, min(1.0, score))
            else:
                score = 0.0

            filtered_pairs.append({
                "drug": drug,
                "disease": disease,
                "score": round(score, 3)  # Round to 3 decimal places for readability
            })

    if filtered_pairs:
        df = pd.DataFrame(filtered_pairs)
        df = df.sort_values('score', ascending=False)

        print(f"\nPairs with frequency >= {min_frequency}: {len(df):,}")
        print("\nScore distribution:")
        print(f"  Positive associations (score > 0): {(df['score'] > 0).sum():,}")
        print(f"  Negative associations (score < 0): {(df['score'] < 0).sum():,}")
        print(f"  Neutral associations (score = 0): {(df['score'] == 0).sum():,}")
        print(f"  Score range: {df['score'].min():.3f} to {df['score'].max():.3f}")
        print(f"  Average score: {df['score'].mean():.3f}")

        print("\nTop 10 pairs (by score):")
        print(df.head(10).to_string(index=False))
    else:
        # Create empty DataFrame with correct columns
        df = pd.DataFrame(columns=["drug", "disease", "score"])
        print(f"\nWarning: No pairs found with frequency >= {min_frequency}")

    return df


def get_extraction_stats(clinical_pairs_df: pd.DataFrame) -> Dict[str, int]:
    """
    Get statistics about extracted clinical pairs with proportional scores.

    Args:
        clinical_pairs_df: DataFrame with drug-disease pairs and proportional scores

    Returns:
        Dictionary with statistics
    """
    if len(clinical_pairs_df) == 0:
        # Handle empty DataFrame
        stats = {
            "total_pairs": 0,
            "unique_drugs": 0,
            "unique_diseases": 0,
            "positive_associations": 0,
            "negative_associations": 0,
            "neutral_associations": 0,
        }
        print("\nClinical Extraction Statistics:")
        print("  Warning: No drug-disease pairs found")
        for key, value in stats.items():
            print(f"  {key}: {value:,}")
    else:
        # Handle both normalized (drug_id/disease_id) and raw (drug/disease) formats
        if 'drug_id' in clinical_pairs_df.columns:
            # Normalized format with node IDs
            drug_col = 'drug_id'
            disease_col = 'disease_id'
            data_type = "normalized"
        else:
            # Raw format with names
            drug_col = 'drug'
            disease_col = 'disease'
            data_type = "raw"
            
        stats = {
            "total_pairs": len(clinical_pairs_df),
            "unique_drugs": int(clinical_pairs_df[drug_col].nunique()),
            "unique_diseases": int(clinical_pairs_df[disease_col].nunique()),
            "positive_associations": int((clinical_pairs_df['score'] > 0).sum()),
            "negative_associations": int((clinical_pairs_df['score'] < 0).sum()),
            "neutral_associations": int((clinical_pairs_df['score'] == 0).sum()),
        }

        print(f"\nClinical Extraction Statistics ({data_type} format):")
        for key, value in stats.items():
            print(f"  {key}: {value:,}")

        print("\nScore Statistics:")
        print(f"  Average score: {clinical_pairs_df['score'].mean():.3f}")
        print(f"  Score range: {clinical_pairs_df['score'].min():.3f} to {clinical_pairs_df['score'].max():.3f}")
        print(f"  Standard deviation: {clinical_pairs_df['score'].std():.3f}")

    return stats


def extract_and_normalize_drug_disease_pairs(
    notes_df: pd.DataFrame,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    database: str = "primekg",
    ner_model: str = "en_ner_bc5cdr_md",
    min_frequency: int = 2,
    max_note_length: int = 10000,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Extract drug-disease pairs from clinical notes and normalize to PrimeKG node IDs.
    
    Args:
        notes_df: DataFrame with clinical notes
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        database: Database name
        ner_model: Name of spaCy NER model to use
        min_frequency: Minimum frequency for a pair to be included
        max_note_length: Maximum characters to process per note
        
    Returns:
        Tuple of (normalized_df with node IDs, combined_stats)
    """
    print("\n" + "="*60)
    print("CLINICAL EXTRACTION WITH NAME NORMALIZATION")
    print("="*60)
    
    # Step 1: Extract raw clinical pairs
    print("\nStep 1: Extracting drug-disease pairs from clinical notes...")
    raw_pairs = extract_drug_disease_pairs(
        notes_df=notes_df,
        ner_model=ner_model,
        min_frequency=min_frequency,
        max_note_length=max_note_length,
    )
    
    if len(raw_pairs) == 0:
        print("Warning: No clinical pairs extracted")
        return pd.DataFrame(), {"extraction_stats": {}, "normalization_stats": {}}
    
    # Step 2: Normalize names to PrimeKG node IDs
    print(f"\nStep 2: Normalizing {len(raw_pairs)} pairs to PrimeKG node IDs...")
    name_matcher = create_name_matcher(neo4j_uri, neo4j_user, neo4j_password, database)
    normalized_pairs, normalization_stats = name_matcher.normalize_clinical_pairs(raw_pairs)
    
    # Step 3: Get extraction stats
    extraction_stats = get_extraction_stats(raw_pairs)
    
    # Combine stats
    combined_stats = {
        "extraction_stats": extraction_stats,
        "normalization_stats": normalization_stats,
        "final_normalized_pairs": len(normalized_pairs)
    }
    
    print("\n" + "="*60)
    print("CLINICAL EXTRACTION SUMMARY")
    print("="*60)
    print(f"Raw pairs extracted: {len(raw_pairs)}")
    print(f"Pairs after normalization: {len(normalized_pairs)}")
    print(f"Retention rate: {len(normalized_pairs)/len(raw_pairs)*100:.1f}%" if len(raw_pairs) > 0 else "0%")
    
    return normalized_pairs, combined_stats