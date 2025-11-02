#!/usr/bin/env python3
"""
Simple script to run the off-label drug discovery pipeline.

Usage:
    python run_offlabel_pipeline.py

Or with custom parameters:
    python run_offlabel_pipeline.py --num-epochs 50 --batch-size 32
"""

import os
import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import pipeline
from clinical_drug_discovery.lib.offlabel_pipeline import main

if __name__ == "__main__":
    print("=" * 80)
    print("OFF-LABEL DRUG DISCOVERY PIPELINE")
    print("=" * 80)
    print()
    print("This pipeline will:")
    print("  1. Load graph data from CSV files (default) or Memgraph")
    print("  2. Prune drug-drug and protein-protein edges")
    print("  3. Prepare training/validation/test splits")
    print("  4. Create heterogeneous graph structure")
    print("  5. Initialize R-GCN model with learnable embeddings")
    print("  6. Train model with early stopping")
    print("  7. Evaluate on test set")
    print()
    print("Output will be saved to: ./output/")
    print()
    print("To use Memgraph instead of CSV:")
    print("  python run_offlabel_pipeline.py --data-source memgraph")
    print()
    print("=" * 80)
    print()

    # Run pipeline
    main()
