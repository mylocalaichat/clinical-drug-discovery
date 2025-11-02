"""
Dagster assets for off-label drug discovery pipeline.

This module defines assets for the complete 9-step R-GCN pipeline:
1. Load and filter edges from Memgraph
2. Prune drug-drug and protein-protein edges
3. Prepare train/test/validation splits
4. Create heterogeneous graph
5-6. Train R-GCN model
7-9. Evaluate model
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from dagster import asset, AssetExecutionContext, Output, MetadataValue
from sqlalchemy import create_engine, text

from clinical_drug_discovery.lib.offlabel_data_loading import (
    CSVGraphLoader,
    GraphPruner,
    OffLabelDataPreparator,
    create_heterogeneous_graph,
)
from clinical_drug_discovery.lib.offlabel_model import initialize_model
from clinical_drug_discovery.lib.offlabel_training import (
    create_dataloaders,
    train_model,
    test_model,
)

logger = logging.getLogger(__name__)


@asset
def offlabel_edges_filtered(context: AssetExecutionContext, download_data: dict) -> Output[Dict[str, Any]]:
    """
    Step 1: Load edges from CSV files with selective edge type filtering.

    Reads data from the download_data asset (CSV files).

    Args:
        download_data: Dictionary containing download information from download_data asset

    Returns:
        Dictionary with filtered edges DataFrame and output file path
    """
    context.log.info("=" * 80)
    context.log.info("STEP 1: Loading graph with selective edge types from CSV")
    context.log.info("=" * 80)

    # Use edges file from download_data asset
    edges_file = download_data["edges_file"]
    context.log.info(f"Loading data from {edges_file}")

    # Load edges directly from the file path
    loader = CSVGraphLoader(data_dir=Path(edges_file).parent)

    try:
        # Load edges with filtering
        edges_df = loader.load_edges_with_filter()

        if len(edges_df) == 0:
            context.log.error("No edges loaded from CSV!")
            context.log.error("Please ensure:")
            context.log.error("  1. PrimeKG data has been downloaded (run download_data asset first)")
            context.log.error("  2. The edge types in INCLUSION_EDGE_TYPES exist in your data")
            raise ValueError("No edges found in CSV files. Cannot proceed with off-label pipeline.")

        context.log.info(f"Loaded {len(edges_df):,} edges with inclusion filter")

        # Get edge type distribution
        edge_type_counts = edges_df['relation'].value_counts().to_dict()

        # Save to data directory
        output_dir = Path("data/06_models/offlabel")
        output_dir.mkdir(parents=True, exist_ok=True)

        edges_path = output_dir / "01_filtered_edges.csv"
        edges_df.to_csv(edges_path, index=False)
        context.log.info(f"Saved filtered edges to {edges_path}")

        return Output(
            value={
                "edges_df": edges_df,
                "edges_file": str(edges_path),
                "num_edges": len(edges_df),
            },
            metadata={
                "num_edges": len(edges_df),
                "num_edge_types": len(edge_type_counts),
                "edge_types": MetadataValue.json(edge_type_counts),
                "output_file": str(edges_path),
                "preview": MetadataValue.md(edges_df.head(10).to_markdown()),
            },
        )

    finally:
        loader.close()


@asset
def offlabel_edges_pruned(
    context: AssetExecutionContext,
    offlabel_edges_filtered: Dict[str, Any],
) -> Output[Dict[str, Any]]:
    """
    Step 2: Prune drug-drug and protein-protein edges based on similarity scores.

    Drug-drug: Keep top 10 neighbors based on (5×shared_indications + 3×shared_proteins)
    Protein-protein: Keep top 20 neighbors based on (5×shared_bioprocesses + 3×shared_molfuncs + 4×shared_diseases)

    Args:
        offlabel_edges_filtered: Dictionary containing filtered edges DataFrame and file path

    Returns:
        Dictionary with pruned edges DataFrame and output file path
    """
    context.log.info("=" * 80)
    context.log.info("STEP 2: Pruning graph")
    context.log.info("=" * 80)

    # Get pruning parameters
    drug_top_k = int(os.getenv("OFFLABEL_DRUG_TOP_K", "10"))
    protein_top_k = int(os.getenv("OFFLABEL_PROTEIN_TOP_K", "20"))

    context.log.info(f"Pruning parameters: drug_top_k={drug_top_k}, protein_top_k={protein_top_k}")

    # Get filtered edges DataFrame from upstream asset
    edges_df = offlabel_edges_filtered["edges_df"]
    context.log.info(f"Using filtered edges from {offlabel_edges_filtered['edges_file']}")
    context.log.info(f"Input: {len(edges_df):,} edges")

    # Initialize CSV loader with pre-loaded filtered edges (much faster than reloading from CSV)
    loader = CSVGraphLoader(edges_df=edges_df)

    try:
        pruner = GraphPruner(loader)

        # Count initial edges
        initial_drug_drug = len(edges_df[edges_df['relation'] == 'drug_drug'])
        initial_protein_protein = len(edges_df[edges_df['relation'] == 'protein_protein'])

        # Prune drug-drug edges
        context.log.info(f"Pruning drug-drug edges (top {drug_top_k})...")
        pruned_df = pruner.prune_drug_drug_edges(edges_df, top_k=drug_top_k)

        # Prune protein-protein edges
        context.log.info(f"Pruning protein-protein edges (top {protein_top_k})...")
        pruned_df = pruner.prune_protein_protein_edges(pruned_df, top_k=protein_top_k)

        # Count final edges
        final_drug_drug = len(pruned_df[pruned_df['relation'] == 'drug_drug'])
        final_protein_protein = len(pruned_df[pruned_df['relation'] == 'protein_protein'])

        context.log.info("Pruning complete:")
        context.log.info(f"  Drug-drug: {initial_drug_drug:,} → {final_drug_drug:,}")
        context.log.info(f"  Protein-protein: {initial_protein_protein:,} → {final_protein_protein:,}")
        context.log.info(f"  Total: {len(edges_df):,} → {len(pruned_df):,}")

        # Save pruned edges
        output_dir = Path("data/06_models/offlabel")
        pruned_path = output_dir / "02_pruned_edges.csv"
        pruned_df.to_csv(pruned_path, index=False)
        context.log.info(f"Saved pruned edges to {pruned_path}")

        return Output(
            value={
                "edges_df": pruned_df,
                "edges_file": str(pruned_path),
                "num_edges": len(pruned_df),
            },
            metadata={
                "num_edges": len(pruned_df),
                "initial_edges": len(edges_df),
                "reduction_percent": (1 - len(pruned_df) / len(edges_df)) * 100,
                "drug_drug_before": initial_drug_drug,
                "drug_drug_after": final_drug_drug,
                "protein_protein_before": initial_protein_protein,
                "protein_protein_after": final_protein_protein,
                "output_file": str(pruned_path),
            },
        )

    finally:
        loader.close()


@asset
def offlabel_node_metadata(
    context: AssetExecutionContext,
    offlabel_edges_pruned: Dict[str, Any]
) -> Output[Dict[str, Any]]:
    """
    Extract node metadata from pruned edges.

    Args:
        offlabel_edges_pruned: Dictionary containing pruned edges DataFrame and file path

    Returns:
        Dictionary with node metadata DataFrame and output file path
    """
    context.log.info("Extracting node metadata from pruned edges...")

    # Get pruned edges DataFrame from upstream asset
    pruned_edges_df = offlabel_edges_pruned["edges_df"]
    context.log.info(f"Using pruned edges from {offlabel_edges_pruned['edges_file']}")
    context.log.info(f"Extracting nodes from {len(pruned_edges_df):,} pruned edges")

    # Use CSVGraphLoader with pruned edges to extract node metadata
    loader = CSVGraphLoader(edges_df=pruned_edges_df)

    try:
        nodes_df = loader.get_node_metadata()

        if len(nodes_df) == 0:
            context.log.error("No nodes extracted from pruned edges!")
            raise ValueError("No nodes found in pruned edges. Cannot proceed with off-label pipeline.")

        context.log.info(f"Extracted {len(nodes_df):,} unique nodes from pruned graph")

        # Get node type distribution
        node_type_counts = nodes_df['node_type'].value_counts().to_dict()

        # Save to data directory
        output_dir = Path("data/06_models/offlabel")
        output_dir.mkdir(parents=True, exist_ok=True)

        nodes_path = output_dir / "01_node_metadata.csv"
        nodes_df.to_csv(nodes_path, index=False)
        context.log.info(f"Saved node metadata to {nodes_path}")

        return Output(
            value={
                "nodes_df": nodes_df,
                "nodes_file": str(nodes_path),
                "num_nodes": len(nodes_df),
            },
            metadata={
                "num_nodes": len(nodes_df),
                "num_node_types": len(node_type_counts),
                "node_types": MetadataValue.json(node_type_counts),
                "output_file": str(nodes_path),
                "source_edges": offlabel_edges_pruned['edges_file'],
            },
        )

    finally:
        loader.close()


@asset
def offlabel_train_test_split(
    context: AssetExecutionContext,
    offlabel_edges_pruned: Dict[str, Any],
) -> Output[Dict[str, Any]]:
    """
    Step 3: Prepare train/validation/test splits for off-label drug discovery.

    Positive examples: off-label use edges
    Negative examples: contraindications + random drug-disease pairs

    Args:
        offlabel_edges_pruned: Dictionary containing pruned edges DataFrame and file path

    Returns:
        Dictionary with train/val/test splits
    """
    context.log.info("=" * 80)
    context.log.info("STEP 3: Preparing train/validation/test splits")
    context.log.info("=" * 80)

    # Get pruned edges DataFrame from upstream asset
    pruned_edges_df = offlabel_edges_pruned["edges_df"]
    context.log.info(f"Using pruned edges from {offlabel_edges_pruned['edges_file']}")
    context.log.info(f"Input: {len(pruned_edges_df):,} edges")

    # Get parameters
    test_size = float(os.getenv("OFFLABEL_TEST_SIZE", "0.2"))
    val_split = float(os.getenv("OFFLABEL_VAL_SPLIT", "0.2"))
    num_contraindication_samples = int(os.getenv("OFFLABEL_NUM_CONTRAINDICATIONS", "4800"))
    num_random_negatives = int(os.getenv("OFFLABEL_NUM_RANDOM_NEGATIVES", "1600"))
    random_seed = int(os.getenv("OFFLABEL_RANDOM_SEED", "42"))

    context.log.info(f"Parameters: test_size={test_size}, val_split={val_split}")
    context.log.info(f"Negative samples: {num_contraindication_samples} contraindications + {num_random_negatives} random")

    # Prepare data
    preparator = OffLabelDataPreparator(pruned_edges_df)
    datasets = preparator.prepare_link_prediction_data(
        test_size=test_size,
        num_contraindication_samples=num_contraindication_samples,
        num_random_negatives=num_random_negatives,
        random_seed=random_seed,
    )

    # Further split train into train/val
    from sklearn.model_selection import train_test_split

    train_pos, val_pos = train_test_split(
        datasets['train_positives'],
        test_size=val_split,
        random_state=random_seed
    )

    train_neg, val_neg = train_test_split(
        datasets['train_negatives'],
        test_size=val_split,
        random_state=random_seed
    )

    # Update datasets
    datasets['train_positives'] = train_pos
    datasets['train_negatives'] = train_neg
    datasets['val_positives'] = val_pos
    datasets['val_negatives'] = val_neg

    context.log.info("Dataset splits:")
    context.log.info(f"  Train: {len(train_pos):,} pos + {len(train_neg):,} neg = {len(train_pos) + len(train_neg):,}")
    context.log.info(f"  Val: {len(val_pos):,} pos + {len(val_neg):,} neg = {len(val_pos) + len(val_neg):,}")
    context.log.info(f"  Test: {len(datasets['test_positives']):,} pos + {len(datasets['test_negatives']):,} neg = "
                   f"{len(datasets['test_positives']) + len(datasets['test_negatives']):,}")

    # Save splits
    output_dir = Path("data/07_model_output/offlabel")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in datasets.items():
        if isinstance(split_data, pd.DataFrame):
            split_path = output_dir / f"03_{split_name}.csv"
            split_data.to_csv(split_path, index=False)
            context.log.info(f"Saved {split_name} to {split_path}")

    return Output(
        value=datasets,
        metadata={
            "train_positives": len(train_pos),
            "train_negatives": len(train_neg),
            "val_positives": len(val_pos),
            "val_negatives": len(val_neg),
            "test_positives": len(datasets['test_positives']),
            "test_negatives": len(datasets['test_negatives']),
            "total_train": len(train_pos) + len(train_neg),
            "total_val": len(val_pos) + len(val_neg),
            "total_test": len(datasets['test_positives']) + len(datasets['test_negatives']),
        },
    )


@asset
def offlabel_hetero_graph(
    context: AssetExecutionContext,
    offlabel_train_test_split: Dict[str, Any],
    offlabel_node_metadata: Dict[str, Any],
) -> Output[Dict[str, Any]]:
    """
    Step 4: Create heterogeneous graph structure for PyTorch Geometric.

    Args:
        offlabel_train_test_split: Dictionary containing train/test splits
        offlabel_node_metadata: Dictionary containing node metadata DataFrame

    Returns:
        Dictionary with HeteroData graph and node mapping
    """
    context.log.info("=" * 80)
    context.log.info("STEP 4: Creating heterogeneous graph")
    context.log.info("=" * 80)

    # Extract DataFrames from upstream assets
    nodes_df = offlabel_node_metadata["nodes_df"]
    context.log.info(f"Using node metadata from {offlabel_node_metadata['nodes_file']}")

    # Use training edges only (exclude test positives)
    train_graph, node_mapping = create_heterogeneous_graph(
        offlabel_train_test_split['train_edges'],
        nodes_df
    )

    context.log.info("Created heterogeneous graph:")
    context.log.info(f"  Node types: {len(train_graph.node_types)}")
    context.log.info(f"  Edge types: {len(train_graph.edge_types)}")

    # Save node mapping
    output_dir = Path("data/06_models/offlabel")
    node_mapping_path = output_dir / "04_node_mapping.json"

    # Convert to serializable format
    serializable_mapping = {
        node_type: {node_id: int(idx) for node_id, idx in mapping.items()}
        for node_type, mapping in node_mapping.items()
    }

    with open(node_mapping_path, 'w') as f:
        json.dump(serializable_mapping, f, indent=2)

    context.log.info(f"Saved node mapping to {node_mapping_path}")

    # Save graph statistics
    graph_stats = {
        'node_types': list(train_graph.node_types),
        'num_node_types': len(train_graph.node_types),
        'edge_types': [f"{src}-{rel}->{dst}" for src, rel, dst in train_graph.edge_types],
        'num_edge_types': len(train_graph.edge_types),
        'num_nodes_per_type': {
            node_type: train_graph[node_type].num_nodes
            for node_type in train_graph.node_types
        },
    }

    return Output(
        value={'graph': train_graph, 'node_mapping': node_mapping},
        metadata={
            "num_node_types": len(train_graph.node_types),
            "num_edge_types": len(train_graph.edge_types),
            "graph_stats": MetadataValue.json(graph_stats),
        },
    )


@asset
def offlabel_trained_model(
    context: AssetExecutionContext,
    offlabel_hetero_graph: Dict[str, Any],
    offlabel_train_test_split: Dict[str, pd.DataFrame],
) -> Output[Dict[str, Any]]:
    """
    Steps 5-8: Initialize model, train with mini-batch sampling and early stopping.

    Args:
        offlabel_hetero_graph: Heterogeneous graph and node mapping
        offlabel_train_test_split: Train/val/test splits

    Returns:
        Dictionary with trained model and training history
    """
    context.log.info("=" * 80)
    context.log.info("STEPS 5-8: Training R-GCN model")
    context.log.info("=" * 80)

    # Get hyperparameters from environment
    embedding_dim = int(os.getenv("OFFLABEL_EMBEDDING_DIM", "128"))
    hidden_dim = int(os.getenv("OFFLABEL_HIDDEN_DIM", "128"))
    num_layers = int(os.getenv("OFFLABEL_NUM_LAYERS", "2"))
    dropout = float(os.getenv("OFFLABEL_DROPOUT", "0.3"))
    batch_size = int(os.getenv("OFFLABEL_BATCH_SIZE", "64"))
    num_epochs = int(os.getenv("OFFLABEL_NUM_EPOCHS", "100"))
    learning_rate = float(os.getenv("OFFLABEL_LEARNING_RATE", "0.001"))
    early_stopping_patience = int(os.getenv("OFFLABEL_EARLY_STOPPING_PATIENCE", "10"))

    context.log.info("Model parameters:")
    context.log.info(f"  embedding_dim={embedding_dim}, hidden_dim={hidden_dim}")
    context.log.info(f"  num_layers={num_layers}, dropout={dropout}")
    context.log.info("Training parameters:")
    context.log.info(f"  batch_size={batch_size}, num_epochs={num_epochs}")
    context.log.info(f"  learning_rate={learning_rate}, patience={early_stopping_patience}")

    # Extract graph and node mapping
    train_graph = offlabel_hetero_graph['graph']
    node_mapping = offlabel_hetero_graph['node_mapping']

    # Auto-detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    context.log.info(f"Using device: {device}")

    # Initialize model
    model = initialize_model(
        data=train_graph,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        device=device
    )

    total_params = sum(p.numel() for p in model.parameters())
    context.log.info(f"Model initialized with {total_params:,} parameters")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_positives=offlabel_train_test_split['train_positives'],
        train_negatives=offlabel_train_test_split['train_negatives'],
        val_positives=offlabel_train_test_split['val_positives'],
        val_negatives=offlabel_train_test_split['val_negatives'],
        node_mapping=node_mapping,
        batch_size=batch_size,
        num_workers=0
    )

    # Checkpoint path
    output_dir = Path("data/06_models/offlabel")
    checkpoint_path = output_dir / "08_best_model.pt"

    # Train model
    history = train_model(
        model=model,
        data=train_graph,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=0.0,
        grad_clip_norm=1.0,
        early_stopping_patience=early_stopping_patience,
        checkpoint_path=str(checkpoint_path)
    )

    # Save training history
    history_path = output_dir / "08_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    context.log.info(f"Saved training history to {history_path}")

    # Best validation metrics
    best_val_auc = max(history['val_auc_roc'])
    best_epoch = history['val_auc_roc'].index(best_val_auc) + 1

    context.log.info("Training complete!")
    context.log.info(f"Best validation AUC-ROC: {best_val_auc:.4f} at epoch {best_epoch}")

    return Output(
        value={'model': model, 'history': history, 'device': device, 'node_mapping': node_mapping},
        metadata={
            "total_parameters": total_params,
            "num_epochs_trained": len(history['train_loss']),
            "best_val_auc_roc": best_val_auc,
            "best_epoch": best_epoch,
            "final_train_loss": history['train_loss'][-1],
            "final_val_loss": history['val_loss'][-1],
            "final_val_accuracy": history['val_accuracy'][-1],
            "training_history": MetadataValue.json(history),
        },
    )


@asset
def offlabel_model_evaluation(
    context: AssetExecutionContext,
    offlabel_trained_model: Dict[str, Any],
    offlabel_hetero_graph: Dict[str, Any],
    offlabel_train_test_split: Dict[str, pd.DataFrame],
) -> Output[Dict[str, float]]:
    """
    Step 9: Evaluate trained model on test set.

    Args:
        offlabel_trained_model: Trained model and history
        offlabel_hetero_graph: Heterogeneous graph
        offlabel_train_test_split: Train/test splits

    Returns:
        Dictionary with test metrics
    """
    context.log.info("=" * 80)
    context.log.info("STEP 9: Evaluating on test set")
    context.log.info("=" * 80)

    # Load best model checkpoint
    output_dir = Path("data/06_models/offlabel")
    checkpoint_path = output_dir / "08_best_model.pt"

    model = offlabel_trained_model['model']
    device = offlabel_trained_model['device']
    node_mapping = offlabel_trained_model['node_mapping']
    train_graph = offlabel_hetero_graph['graph']

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        context.log.info(f"Loaded best model from {checkpoint_path}")
    else:
        context.log.warning("No checkpoint found, using current model state")

    # Evaluate
    batch_size = int(os.getenv("OFFLABEL_BATCH_SIZE", "64"))

    test_metrics = test_model(
        model=model,
        data=train_graph,
        test_positives=offlabel_train_test_split['test_positives'],
        test_negatives=offlabel_train_test_split['test_negatives'],
        node_mapping=node_mapping,
        device=device,
        batch_size=batch_size
    )

    # Convert numpy types to Python native types for JSON/Dagster serialization
    # Convert all numeric values to float for Dagster type consistency
    serializable_metrics = {
        k: float(v) if isinstance(v, (np.integer, np.floating, int, float)) else v
        for k, v in test_metrics.items()
    }

    # Save test metrics
    metrics_path = Path("data/07_model_output/offlabel") / "09_test_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)

    context.log.info(f"Saved test metrics to {metrics_path}")

    context.log.info("=" * 80)
    context.log.info("OFF-LABEL DRUG DISCOVERY PIPELINE COMPLETE!")
    context.log.info("=" * 80)
    context.log.info(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
    context.log.info(f"Test AUC-PR: {test_metrics['auc_pr']:.4f}")
    context.log.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

    return Output(
        value=serializable_metrics,
        metadata={
            "auc_roc": serializable_metrics['auc_roc'],
            "auc_pr": serializable_metrics['auc_pr'],
            "accuracy": serializable_metrics['accuracy'],
            "precision": serializable_metrics['precision'],
            "recall": serializable_metrics['recall'],
            "f1": serializable_metrics['f1'],
            "sensitivity": serializable_metrics['sensitivity'],
            "specificity": serializable_metrics['specificity'],
            "true_positives": serializable_metrics['true_positives'],
            "true_negatives": serializable_metrics['true_negatives'],
            "false_positives": serializable_metrics['false_positives'],
            "false_negatives": serializable_metrics['false_negatives'],
            "all_metrics": MetadataValue.json(serializable_metrics),
        },
    )


@asset
def offlabel_novel_predictions(
    context: AssetExecutionContext,
    offlabel_trained_model: Dict[str, Any],
    offlabel_hetero_graph: Dict[str, Any],
    offlabel_node_metadata: Dict[str, Any],
    offlabel_edges_pruned: Dict[str, Any],
) -> Output[pd.DataFrame]:
    """
    Step 10: Predict novel off-label drug uses.

    This generates all possible drug-disease pairs, excludes known relationships,
    and predicts the likelihood of each being a valid off-label use.

    Args:
        offlabel_trained_model: Trained model
        offlabel_hetero_graph: Graph structure
        offlabel_node_metadata: Node information for generating pairs
        offlabel_edges_pruned: All edges to check for known relationships

    Returns:
        DataFrame with predicted novel off-label uses ranked by confidence
    """
    context.log.info("=" * 80)
    context.log.info("STEP 10: Predicting novel off-label drug uses")
    context.log.info("=" * 80)

    # Load model and data
    model = offlabel_trained_model['model']
    device = offlabel_trained_model['device']
    node_mapping = offlabel_trained_model['node_mapping']
    train_graph = offlabel_hetero_graph['graph']
    nodes_df = offlabel_node_metadata['nodes_df']
    edges_df = offlabel_edges_pruned['edges_df']

    # Load best model checkpoint
    output_dir = Path("data/06_models/offlabel")
    checkpoint_path = output_dir / "08_best_model.pt"

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        context.log.info(f"Loaded best model from {checkpoint_path}")

    # Get all drugs and diseases that are in the node_mapping
    all_drugs = [drug_id for drug_id in nodes_df[nodes_df['node_type'] == 'drug']['node_id']
                 if drug_id in node_mapping.get('drug', {})]
    all_diseases = [disease_id for disease_id in nodes_df[nodes_df['node_type'] == 'disease']['node_id']
                    if disease_id in node_mapping.get('disease', {})]

    context.log.info(f"Total drugs in graph: {len(all_drugs):,}")
    context.log.info(f"Total diseases in graph: {len(all_diseases):,}")
    context.log.info(f"Total possible pairs: {len(all_drugs) * len(all_diseases):,}")

    # Get known relationships to exclude
    known_pairs = set()

    # Collect all known drug-disease relationships
    for relation in ['indication', 'off-label use', 'contraindication']:
        rel_edges = edges_df[edges_df['relation'] == relation]
        for _, row in rel_edges.iterrows():
            # Normalize to (drug_id, disease_id)
            if row['source_type'] == 'drug' and row['target_type'] == 'disease':
                known_pairs.add((row['source_id'], row['target_id']))
            elif row['source_type'] == 'disease' and row['target_type'] == 'drug':
                known_pairs.add((row['target_id'], row['source_id']))

    context.log.info(f"Known drug-disease pairs to exclude: {len(known_pairs):,}")

    # Generate candidate pairs with intelligent sampling
    # Strategy: Only predict for drugs/diseases with at least some connectivity
    context.log.info("Generating candidate pairs with intelligent sampling...")

    # Get sampling strategy from env
    max_candidates = int(os.getenv("OFFLABEL_MAX_CANDIDATES", "100000"))
    min_drug_degree = int(os.getenv("OFFLABEL_MIN_DRUG_DEGREE", "5"))
    min_disease_degree = int(os.getenv("OFFLABEL_MIN_DISEASE_DEGREE", "5"))

    # Count node degrees (number of edges per node) - vectorized for speed
    context.log.info("Counting node degrees...")
    drug_degrees = {}
    disease_degrees = {}

    # Vectorized approach - much faster than iterrows
    drug_source_mask = edges_df['source_type'] == 'drug'
    drug_target_mask = edges_df['target_type'] == 'drug'
    disease_source_mask = edges_df['source_type'] == 'disease'
    disease_target_mask = edges_df['target_type'] == 'disease'

    # Count drug degrees
    drug_source_counts = edges_df[drug_source_mask]['source_id'].value_counts().to_dict()
    drug_target_counts = edges_df[drug_target_mask]['target_id'].value_counts().to_dict()

    for drug_id in set(list(drug_source_counts.keys()) + list(drug_target_counts.keys())):
        drug_degrees[drug_id] = drug_source_counts.get(drug_id, 0) + drug_target_counts.get(drug_id, 0)

    # Count disease degrees
    disease_source_counts = edges_df[disease_source_mask]['source_id'].value_counts().to_dict()
    disease_target_counts = edges_df[disease_target_mask]['target_id'].value_counts().to_dict()

    for disease_id in set(list(disease_source_counts.keys()) + list(disease_target_counts.keys())):
        disease_degrees[disease_id] = disease_source_counts.get(disease_id, 0) + disease_target_counts.get(disease_id, 0)

    # Filter drugs/diseases by minimum degree
    filtered_drugs = [d for d in all_drugs if drug_degrees.get(d, 0) >= min_drug_degree]
    filtered_diseases = [d for d in all_diseases if disease_degrees.get(d, 0) >= min_disease_degree]

    context.log.info(f"Filtered drugs (degree >= {min_drug_degree}): {len(filtered_drugs):,} / {len(all_drugs):,}")
    context.log.info(f"Filtered diseases (degree >= {min_disease_degree}): {len(filtered_diseases):,} / {len(all_diseases):,}")
    context.log.info(f"Potential pairs: {len(filtered_drugs) * len(filtered_diseases):,}")

    # Generate candidates (exclude known relationships)
    context.log.info(f"Generating candidate pairs from {len(filtered_drugs):,} drugs × {len(filtered_diseases):,} diseases...")
    candidate_pairs = []

    for drug_id in tqdm(filtered_drugs, desc="Generating pairs"):
        for disease_id in filtered_diseases:
            if (drug_id, disease_id) not in known_pairs:
                candidate_pairs.append((drug_id, disease_id))

    # Optional: Cap if explicitly set (set to 0 or very high to disable)
    if max_candidates > 0 and len(candidate_pairs) > max_candidates:
        context.log.info(f"Sampling {max_candidates:,} candidates from {len(candidate_pairs):,} total")
        import random
        random.seed(42)
        candidate_pairs = random.sample(candidate_pairs, max_candidates)
    else:
        context.log.info(f"Using ALL {len(candidate_pairs):,} candidate pairs (no sampling)")

    context.log.info(f"Final candidate novel pairs: {len(candidate_pairs):,}")

    # Predict on candidates
    context.log.info("Running predictions on candidate pairs...")
    context.log.info(f"Using device: {device}")

    model = model.to(device)
    train_graph = train_graph.to(device)
    model.eval()

    import time

    # OPTIMIZATION: Pre-compute all node embeddings once!
    context.log.info("Pre-computing node embeddings (this may take a minute)...")
    embedding_start = time.time()

    with torch.no_grad():
        all_embeddings = model.encode(train_graph)

    embedding_time = time.time() - embedding_start
    context.log.info(f"Embeddings computed in {embedding_time:.1f}s")
    context.log.info(f"  Drug embeddings: {all_embeddings['drug'].shape}")
    context.log.info(f"  Disease embeddings: {all_embeddings['disease'].shape}")

    # Now predict using pre-computed embeddings + MLP head
    batch_size = int(os.getenv("OFFLABEL_INFERENCE_BATCH_SIZE", "10000"))  # Much larger since we're just doing MLP
    context.log.info(f"Inference batch size: {batch_size}")

    predictions = []
    start_time = time.time()

    with torch.no_grad():
        for i in tqdm(range(0, len(candidate_pairs), batch_size), desc="Predicting"):
            batch_pairs = candidate_pairs[i:i+batch_size]

            drug_ids = [pair[0] for pair in batch_pairs]
            disease_ids = [pair[1] for pair in batch_pairs]

            # Get pre-computed embeddings
            drug_indices = [node_mapping['drug'][drug_id] for drug_id in drug_ids]
            disease_indices = [node_mapping['disease'][disease_id] for disease_id in disease_ids]

            drug_embs = all_embeddings['drug'][drug_indices]
            disease_embs = all_embeddings['disease'][disease_indices]

            # Use the MLP head only (super fast!)
            batch_predictions = model.link_predictor(drug_embs, disease_embs)
            predictions.extend(batch_predictions.cpu().numpy())

    elapsed_time = time.time() - start_time
    total_time = embedding_time + elapsed_time
    context.log.info(f"Prediction completed in {elapsed_time:.1f}s ({len(candidate_pairs)/elapsed_time:.0f} pairs/sec)")
    context.log.info(f"Total time (embedding + prediction): {total_time:.1f}s")

    # Create results DataFrame
    results_df = pd.DataFrame({
        'drug_id': [pair[0] for pair in candidate_pairs],
        'disease_id': [pair[1] for pair in candidate_pairs],
        'prediction_score': predictions
    })

    # Add drug and disease names if available
    drug_names = nodes_df[nodes_df['node_type'] == 'drug'].set_index('node_id')['node_name'].to_dict()
    disease_names = nodes_df[nodes_df['node_type'] == 'disease'].set_index('node_id')['node_name'].to_dict()

    results_df['drug_name'] = results_df['drug_id'].map(drug_names).fillna(results_df['drug_id'])
    results_df['disease_name'] = results_df['disease_id'].map(disease_names).fillna(results_df['disease_id'])

    # Sort by prediction score (highest first)
    results_df = results_df.sort_values('prediction_score', ascending=False).reset_index(drop=True)

    # Add rank
    results_df['rank'] = range(1, len(results_df) + 1)

    # Reorder columns
    results_df = results_df[['rank', 'drug_id', 'drug_name', 'disease_id', 'disease_name', 'prediction_score']]

    # Save ALL results
    output_path = Path("data/07_model_output/offlabel") / "10_all_predictions.csv"
    results_df.to_csv(output_path, index=False)
    context.log.info(f"Saved ALL {len(results_df):,} predictions to {output_path}")

    # Also save top N for quick reference
    top_n = int(os.getenv("OFFLABEL_TOP_N_PREDICTIONS", "1000"))
    top_predictions_path = Path("data/07_model_output/offlabel") / f"10_top_{top_n}_predictions.csv"
    results_df.head(top_n).to_csv(top_predictions_path, index=False)
    context.log.info(f"Saved top {top_n} predictions to {top_predictions_path}")

    # Create per-disease summary (top drugs per disease)
    context.log.info("Creating per-disease summary...")
    disease_groups = results_df.groupby('disease_id')

    disease_summary = []
    for disease_id, group in disease_groups:
        top_drugs = group.head(10)  # Top 10 drugs per disease
        disease_summary.append({
            'disease_id': disease_id,
            'disease_name': group.iloc[0]['disease_name'],
            'num_candidates': len(group),
            'top_drug': top_drugs.iloc[0]['drug_name'],
            'top_score': top_drugs.iloc[0]['prediction_score'],
            'mean_score': group['prediction_score'].mean(),
            'high_confidence_count': (group['prediction_score'] > 0.9).sum()
        })

    disease_summary_df = pd.DataFrame(disease_summary)
    disease_summary_df = disease_summary_df.sort_values('top_score', ascending=False)
    summary_path = Path("data/07_model_output/offlabel") / "10_disease_summary.csv"
    disease_summary_df.to_csv(summary_path, index=False)
    context.log.info(f"Saved disease summary to {summary_path}")

    # Log top 10 predictions
    context.log.info("\nTop 10 Novel Off-Label Predictions (Overall):")
    context.log.info("=" * 80)
    for _, row in results_df.head(10).iterrows():
        context.log.info(f"{int(row['rank']):3d}. {row['drug_name'][:40]:40s} -> {row['disease_name'][:40]:40s} | Score: {row['prediction_score']:.4f}")

    # Log top 5 diseases by best drug candidate
    context.log.info("\nTop 5 Diseases by Best Drug Candidate:")
    context.log.info("=" * 80)
    for _, row in disease_summary_df.head(5).iterrows():
        context.log.info(f"  {row['disease_name'][:50]:50s} | Best: {row['top_drug'][:30]:30s} ({row['top_score']:.4f})")

    return Output(
        value=results_df,
        metadata={
            "total_predictions": len(results_df),
            "total_candidates": len(candidate_pairs),
            "total_drugs": len(all_drugs),
            "total_diseases": len(all_diseases),
            "unique_diseases_predicted": len(disease_summary_df),
            "known_pairs_excluded": len(known_pairs),
            "top_score": float(results_df['prediction_score'].iloc[0]),
            "median_score": float(results_df['prediction_score'].median()),
            "mean_score": float(results_df['prediction_score'].mean()),
            "high_confidence_count": int((results_df['prediction_score'] > 0.9).sum()),
            "moderate_confidence_count": int((results_df['prediction_score'] > 0.7).sum()),
            "all_predictions_file": str(output_path),
            "top_predictions_file": str(top_predictions_path),
            "disease_summary_file": str(summary_path),
        },
    )


@asset(
    deps=["offlabel_novel_predictions"],
    group_name="offlabel_discovery",
    description="Load all predictions into PostgreSQL database for fast querying",
)
def offlabel_predictions_db(context: AssetExecutionContext) -> Output[None]:
    """
    Load the complete predictions dataset into PostgreSQL.

    Creates a table 'offlabel_predictions' with:
    - rank: integer
    - drug_id: text
    - drug_name: text (indexed for fast search)
    - disease_id: text
    - disease_name: text (indexed for fast search)
    - prediction_score: float (indexed for fast sorting)

    This enables fast SQL queries like:
        SELECT * FROM offlabel_predictions
        WHERE disease_name ILIKE '%Castleman%'
        ORDER BY prediction_score DESC
        LIMIT 10;
    """
    # Load environment variables
    postgres_user = os.getenv("DAGSTER_POSTGRES_USER", "dagster_user")
    postgres_password = os.getenv("DAGSTER_POSTGRES_PASSWORD", "dagster_password_123")
    postgres_host = os.getenv("DAGSTER_POSTGRES_HOST", "localhost")
    postgres_port = os.getenv("DAGSTER_POSTGRES_PORT", "5432")
    postgres_db = os.getenv("DAGSTER_POSTGRES_DB", "clinical_drug_discovery_db")
    postgres_schema = os.getenv("DAGSTER_POSTGRES_SCHEMA", "public")

    # Create connection string
    connection_string = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"

    context.log.info(f"Connecting to PostgreSQL database '{postgres_db}' on {postgres_host}:{postgres_port}")
    context.log.info(f"Target schema: {postgres_schema}")
    context.log.info(f"Target table: offlabel_predictions")

    # Try to connect and provide helpful error messages
    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            # Test connection
            conn.execute(text("SELECT 1"))
            context.log.info("✓ Database connection successful")
    except Exception as e:
        context.log.error("✗ Cannot connect to PostgreSQL database")
        context.log.error(f"   Error: {e}")
        context.log.error("")
        context.log.error("Please run the database setup script first:")
        context.log.error("   python scripts/setup_postgres.py")
        context.log.error("")
        context.log.error("Or manually create the database and user:")
        context.log.error(f"   createdb -U postgres {postgres_db}")
        context.log.error(f"   createuser -U postgres {postgres_user}")
        raise

    # Load predictions CSV
    predictions_file = Path("data/07_model_output/offlabel/10_all_predictions.csv")
    context.log.info(f"Streaming predictions from {predictions_file}...")

    # Create table and load data
    table_name = "offlabel_predictions"
    context.log.info(f"Loading data into PostgreSQL table '{table_name}'...")

    # Get raw connection for all operations
    raw_conn = engine.raw_connection()
    cursor = raw_conn.cursor()

    try:
        # Ensure schema exists
        if postgres_schema != 'public':
            context.log.info(f"Ensuring schema '{postgres_schema}' exists...")
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {postgres_schema}")
            context.log.info(f"✓ Schema '{postgres_schema}' ready")

        # Drop and create table
        context.log.info(f"Dropping table {postgres_schema}.{table_name} if it exists...")
        cursor.execute(f"DROP TABLE IF EXISTS {postgres_schema}.{table_name}")
        context.log.info("✓ Dropped existing table if it existed")

        # Create the table structure
        context.log.info(f"Creating table '{postgres_schema}.{table_name}'...")
        create_table_sql = f"""
        CREATE TABLE {postgres_schema}.{table_name} (
            rank INTEGER,
            drug_id TEXT,
            drug_name TEXT,
            disease_id TEXT,
            disease_name TEXT,
            prediction_score FLOAT
        )
        """
        cursor.execute(create_table_sql)
        context.log.info(f"✓ Table '{postgres_schema}.{table_name}' created successfully")

        # Load data using COPY with streaming batch reads
        from io import StringIO

        chunksize = 50000
        context.log.info(f"Streaming data in batches of {chunksize:,} rows...")

        # Stream CSV file in chunks and load directly to PostgreSQL
        rows_loaded = 0
        for chunk_num, chunk_df in enumerate(
            pd.read_csv(predictions_file, chunksize=chunksize), start=1
        ):
            # Data validation and cleaning
            if chunk_num == 1:  # Log sample data from first chunk
                context.log.info("Sample data from first chunk:")
                context.log.info(f"  Sample drug_id: {chunk_df['drug_id'].iloc[0]}")
                context.log.info(f"  Sample disease_id: {chunk_df['disease_id'].iloc[0]}")
                context.log.info(f"  Max drug_id length: {chunk_df['drug_id'].str.len().max()}")
                context.log.info(f"  Max disease_id length: {chunk_df['disease_id'].str.len().max()}")
                context.log.info(f"  Max drug_name length: {chunk_df['drug_name'].str.len().max()}")
                context.log.info(f"  Max disease_name length: {chunk_df['disease_name'].str.len().max()}")
                
                # Check for problematic IDs
                long_drug_ids = chunk_df[chunk_df['drug_id'].str.len() > 100]
                long_disease_ids = chunk_df[chunk_df['disease_id'].str.len() > 100]
                
                if len(long_drug_ids) > 0:
                    context.log.warning(f"Found {len(long_drug_ids)} drug_ids longer than 100 chars")
                    context.log.warning(f"  Example: {long_drug_ids['drug_id'].iloc[0][:200]}...")
                    
                if len(long_disease_ids) > 0:
                    context.log.warning(f"Found {len(long_disease_ids)} disease_ids longer than 100 chars")
                    context.log.warning(f"  Example: {long_disease_ids['disease_id'].iloc[0][:200]}...")

            # Convert chunk to CSV in memory
            output = StringIO()
            chunk_df.to_csv(output, sep='\t', header=False, index=False, na_rep='')
            output.seek(0)

            # Use COPY for fast bulk insert - NOTE: copy_from doesn't like qualified names
            # So we use the COPY SQL command instead
            try:
                context.log.info(f"  Loading batch {chunk_num}...")
                copy_sql = f"""
                COPY {postgres_schema}.{table_name} (rank, drug_id, drug_name, disease_id, disease_name, prediction_score)
                FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '')
                """
                cursor.copy_expert(copy_sql, output)
                raw_conn.commit()
                context.log.info(f"  ✓ Batch {chunk_num} loaded successfully")
            except Exception as e:
                raw_conn.rollback()
                context.log.error(f"  ✗ COPY failed for batch {chunk_num}")
                context.log.error(f"  Error: {e}")
                raise

            rows_loaded += len(chunk_df)
            if chunk_num % 10 == 0 or chunk_num == 1:
                context.log.info(f"  Progress: Loaded {rows_loaded:,} rows so far...")

        context.log.info(f"✓ Data loaded successfully - Total: {rows_loaded:,} rows")

    finally:
        cursor.close()
        raw_conn.close()

    context.log.info("Creating indexes for fast queries (this may take a few minutes)...")
    indexes = [
        ("idx_disease_name", "disease_name"),
        ("idx_drug_name", "drug_name"),
        ("idx_prediction_score", "prediction_score DESC"),
        ("idx_disease_score", "disease_name, prediction_score DESC"),
    ]

    with engine.connect() as conn:
        # Create indexes with progress indication
        for idx_name, idx_columns in tqdm(indexes, desc="Creating indexes", unit="index"):
            context.log.info(f"Creating index {idx_name}...")
            conn.execute(text(f"CREATE INDEX {idx_name} ON {postgres_schema}.{table_name} ({idx_columns})"))
            conn.commit()

        context.log.info("✓ All indexes created successfully")

    # Verify data
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {postgres_schema}.{table_name}"))
        row_count = result.scalar()

        result = conn.execute(text(f"SELECT MAX(prediction_score) FROM {postgres_schema}.{table_name}"))
        max_score = result.scalar()

        result = conn.execute(text(f"SELECT COUNT(DISTINCT disease_name) FROM {postgres_schema}.{table_name}"))
        unique_diseases = result.scalar()

        result = conn.execute(text(f"SELECT COUNT(DISTINCT drug_name) FROM {postgres_schema}.{table_name}"))
        unique_drugs = result.scalar()

    context.log.info(f"✓ Successfully loaded {row_count:,} predictions to {postgres_db}.{postgres_schema}.{table_name}")

    return Output(
        None,
        metadata={
            "database": postgres_db,
            "schema": postgres_schema,
            "table_name": table_name,
            "full_table_path": f"{postgres_db}.{postgres_schema}.{table_name}",
            "host": postgres_host,
            "port": postgres_port,
            "row_count": int(row_count),
            "unique_diseases": int(unique_diseases),
            "unique_drugs": int(unique_drugs),
            "max_score": float(max_score),
            "indexes_created": 4,
            "query_example": MetadataValue.md(
                f"""
**Database:** `{postgres_db}`
**Schema:** `{postgres_schema}`
**Table:** `{table_name}`
**Full Path:** `{postgres_db}.{postgres_schema}.{table_name}`

Example queries:

```sql
-- Top 10 drugs for a disease
SELECT rank, drug_name, prediction_score
FROM {postgres_schema}.{table_name}
WHERE disease_name ILIKE '%Castleman%'
ORDER BY prediction_score DESC
LIMIT 10;

-- All high-confidence predictions for a disease
SELECT drug_name, prediction_score
FROM {postgres_schema}.{table_name}
WHERE disease_name = 'Castleman disease'
  AND prediction_score > 0.9
ORDER BY prediction_score DESC;

-- Find diseases for a specific drug
SELECT disease_name, prediction_score
FROM {postgres_schema}.{table_name}
WHERE drug_name ILIKE '%Triamterene%'
ORDER BY prediction_score DESC
LIMIT 20;

-- Summary statistics by disease
SELECT disease_name,
       COUNT(*) as candidate_count,
       MAX(prediction_score) as top_score,
       AVG(prediction_score) as avg_score
FROM {postgres_schema}.{table_name}
GROUP BY disease_name
ORDER BY top_score DESC
LIMIT 10;
```
                """
            ),
        },
    )
