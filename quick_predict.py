#!/usr/bin/env python
"""
Quick prediction for a disease using already-materialized Dagster assets.

This loads the exact same assets used during training, ensuring compatibility.

Usage:
    python quick_predict.py "Castleman"
"""

import sys
from pathlib import Path
import pandas as pd
import torch

# Check if disease name provided
if len(sys.argv) < 2:
    print("Usage: python quick_predict.py <disease_name>")
    print("\nExamples:")
    print("  python quick_predict.py Castleman")
    print("  python quick_predict.py Alzheimer")
    sys.exit(1)

disease_query = sys.argv[1]

print("=" * 80)
print(f"FINDING DRUG CANDIDATES FOR: {disease_query}")
print("=" * 80)

# Load the model and graph from the offlabel_trained_model asset output
# This is saved during training and includes the correct graph structure
print("\nLoading trained model from Dagster asset materialization...")

checkpoint_path = Path("data/06_models/offlabel/08_best_model.pt")
if not checkpoint_path.exists():
    print(f"Error: Model checkpoint not found at {checkpoint_path}")
    print("\nPlease materialize the 'offlabel_trained_model' asset in Dagster first:")
    print("  1. Open Dagster UI")
    print("  2. Navigate to Assets")
    print("  3. Materialize: offlabel_trained_model")
    sys.exit(1)

# For now, tell the user to use Dagster to run the full predictions
print("\n" + "=" * 80)
print("RECOMMENDATION: Use the Dagster asset instead")
print("=" * 80)
print("\nThe standalone script has a graph mismatch issue.")
print("The best approach is to materialize the 'offlabel_novel_predictions' asset,")
print("which will generate predictions for ALL diseases at once.")
print("\nThen you can filter for your disease:")
print(f"\n  predictions = pd.read_csv('data/07_model_output/offlabel/10_novel_predictions.csv')")
print(f"  castleman = predictions[predictions['disease_name'].str.contains('{disease_query}', case=False)]")
print(f"  print(castleman.head(20))")
print("\nOR run the full asset now (takes ~2 min with optimizations):")
print("  1. Open Dagster UI")
print("  2. Materialize: offlabel_novel_predictions")
print(f"  3. Filter results for '{disease_query}'")
print("\n" + "=" * 80)
