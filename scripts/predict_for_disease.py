#!/usr/bin/env python
"""
Predict drug candidates for a specific disease.

Usage:
    python scripts/predict_for_disease.py "Castleman"
    python scripts/predict_for_disease.py "Alzheimer"
    python scripts/predict_for_disease.py "MONDO:0007915"  # By disease ID
"""

import sys
from pathlib import Path
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clinical_drug_discovery.lib.offlabel_model import OffLabelRGCN
from clinical_drug_discovery.lib.predict_for_disease import predict_drugs_for_disease


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict_for_disease.py <disease_name_or_id>")
        print("\nExamples:")
        print("  python scripts/predict_for_disease.py Castleman")
        print("  python scripts/predict_for_disease.py Alzheimer")
        print("  python scripts/predict_for_disease.py 'COVID-19'")
        sys.exit(1)

    disease_query = sys.argv[1]

    print("=" * 80)
    print(f"FINDING DRUG CANDIDATES FOR: {disease_query}")
    print("=" * 80)

    # Load model checkpoint
    checkpoint_path = Path("data/06_models/offlabel/08_best_model.pt")
    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Please train the model first by running the offlabel_trained_model asset")
        sys.exit(1)

    # Load node metadata
    nodes_path = Path("data/06_models/offlabel/01_node_metadata.csv")
    if not nodes_path.exists():
        print(f"Error: Node metadata not found at {nodes_path}")
        sys.exit(1)

    nodes_df = pd.read_csv(nodes_path)

    # Load edges
    edges_path = Path("data/06_models/offlabel/02_pruned_edges.csv")
    if not edges_path.exists():
        print(f"Error: Edges not found at {edges_path}")
        sys.exit(1)

    edges_df = pd.read_csv(edges_path, low_memory=False)
    print(f"Loaded {len(edges_df):,} edges")

    # The checkpoint was saved during training with the graph already loaded
    # We need to load it differently to avoid graph mismatch
    print("Loading model from checkpoint...")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # The checkpoint doesn't include the graph structure, so we need to rebuild it
    # from the training data to match exactly
    print("Loading training graph (this ensures compatibility with checkpoint)...")
    from clinical_drug_discovery.lib.offlabel_data_loading import create_heterogeneous_graph

    train_edges_path = Path("data/07_model_output/offlabel/03_train_edges.csv")
    train_edges = pd.read_csv(train_edges_path, low_memory=False)

    train_graph, graph_node_mapping = create_heterogeneous_graph(train_edges, nodes_df)

    # Initialize model with the SAME graph structure
    print("Initializing model...")
    from clinical_drug_discovery.lib.offlabel_model import initialize_model
    model = initialize_model(train_graph, device=device)

    # Now load the weights
    print("Loading model weights...")
    model.load_state_dict(checkpoint['model_state_dict'])

    # Use the graph's node mapping (which matches the checkpoint)
    node_mapping = graph_node_mapping

    # Predict
    print("\nRunning predictions...\n")
    results = predict_drugs_for_disease(
        disease_name_or_id=disease_query,
        model=model,
        train_graph=train_graph,
        node_mapping=node_mapping,
        nodes_df=nodes_df,
        edges_df=edges_df,
        device=device,
        top_k=100,
        min_drug_degree=5,
        exclude_known=True
    )

    if len(results) == 0:
        print("\nNo results found.")
        sys.exit(0)

    # Save results
    output_dir = Path("data/07_model_output/offlabel/disease_specific")
    output_dir.mkdir(parents=True, exist_ok=True)

    disease_name_clean = disease_query.replace(" ", "_").replace("/", "_")
    output_path = output_dir / f"predictions_{disease_name_clean}.csv"
    results.to_csv(output_path, index=False)

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 80}")

    print(f"\nTop 20 drug candidates for {results.iloc[0]['disease_name']}:")
    print(f"{'=' * 80}")
    for _, row in results.head(20).iterrows():
        print(f"{int(row['rank']):3d}. {row['drug_name'][:60]:60s} | Score: {row['prediction_score']:.4f}")


if __name__ == "__main__":
    main()
