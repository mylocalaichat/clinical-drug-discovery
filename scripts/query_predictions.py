#!/usr/bin/env python
"""
Query predictions for a specific disease from the full predictions file.

Usage:
    python scripts/query_predictions.py "Castleman"
    python scripts/query_predictions.py "Alzheimer" --top 50
    python scripts/query_predictions.py "cancer" --min-score 0.8
"""

import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Query drug predictions for a disease')
    parser.add_argument('disease', help='Disease name to search for')
    parser.add_argument('--top', type=int, default=20, help='Number of top drugs to show (default: 20)')
    parser.add_argument('--min-score', type=float, default=0.0, help='Minimum prediction score (default: 0.0)')
    parser.add_argument('--output', help='Save results to CSV file')

    args = parser.parse_args()

    # Load predictions
    predictions_file = Path("data/07_model_output/offlabel/10_all_predictions.csv")

    if not predictions_file.exists():
        print(f"Error: Predictions file not found at {predictions_file}")
        print("\nPlease materialize the 'offlabel_novel_predictions' asset in Dagster first.")
        return

    print(f"Loading predictions from {predictions_file}...")
    df = pd.read_csv(predictions_file)
    print(f"Loaded {len(df):,} total predictions")

    # Filter by disease name (case-insensitive)
    disease_mask = df['disease_name'].str.contains(args.disease, case=False, na=False)
    filtered = df[disease_mask]

    if len(filtered) == 0:
        print(f"\nNo diseases found matching '{args.disease}'")
        print("\nTry searching in the disease summary:")
        summary_file = Path("data/07_model_output/offlabel/10_disease_summary.csv")
        if summary_file.exists():
            summary = pd.read_csv(summary_file)
            matches = summary[summary['disease_name'].str.contains(args.disease, case=False, na=False)]
            if len(matches) > 0:
                print("\nSimilar diseases found:")
                for _, row in matches.iterrows():
                    print(f"  - {row['disease_name']}")
        return

    # Filter by minimum score
    if args.min_score > 0:
        filtered = filtered[filtered['prediction_score'] >= args.min_score]
        print(f"Filtered to {len(filtered):,} predictions with score >= {args.min_score}")

    # Group by disease (in case multiple diseases match)
    unique_diseases = filtered['disease_name'].unique()

    for disease_name in unique_diseases:
        disease_preds = filtered[filtered['disease_name'] == disease_name].head(args.top)

        print("\n" + "=" * 80)
        print(f"Top {len(disease_preds)} Drug Candidates for: {disease_name}")
        print("=" * 80)

        for _, row in disease_preds.iterrows():
            print(f"{int(row['rank']):5d}. {row['drug_name'][:50]:50s} | Score: {row['prediction_score']:.4f}")

        # Statistics
        disease_all = filtered[filtered['disease_name'] == disease_name]
        print("\n" + "-" * 80)
        print(f"Statistics for {disease_name}:")
        print(f"  Total candidate drugs: {len(disease_all):,}")
        print(f"  High confidence (>0.9): {(disease_all['prediction_score'] > 0.9).sum():,}")
        print(f"  Moderate confidence (0.7-0.9): {((disease_all['prediction_score'] > 0.7) & (disease_all['prediction_score'] <= 0.9)).sum():,}")
        print(f"  Mean score: {disease_all['prediction_score'].mean():.4f}")
        print(f"  Median score: {disease_all['prediction_score'].median():.4f}")

        # Save if requested
        if args.output:
            output_path = Path(args.output)
            disease_all.to_csv(output_path, index=False)
            print(f"\n✓ Saved {len(disease_all):,} predictions to: {output_path}")


if __name__ == "__main__":
    main()
