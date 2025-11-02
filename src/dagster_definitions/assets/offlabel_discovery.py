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

import pandas as pd
import torch
from dagster import asset, AssetExecutionContext, Output, MetadataValue

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
def offlabel_node_metadata(context: AssetExecutionContext, download_data: dict) -> Output[Dict[str, Any]]:
    """
    Load node metadata from CSV files.

    Args:
        download_data: Dictionary containing download information from download_data asset

    Returns:
        Dictionary with node metadata DataFrame and output file path
    """
    context.log.info("Loading node metadata from CSV...")

    # Use edges file from download_data asset to extract node metadata
    edges_file = download_data["edges_file"]
    context.log.info(f"Extracting node metadata from {edges_file}")

    loader = CSVGraphLoader(data_dir=Path(edges_file).parent)

    try:
        nodes_df = loader.get_node_metadata()

        if len(nodes_df) == 0:
            context.log.error("No nodes loaded from CSV!")
            context.log.error("Please ensure:")
            context.log.error("  1. PrimeKG data has been downloaded (run download_data asset first)")
            raise ValueError("No nodes found in CSV files. Cannot proceed with off-label pipeline.")

        context.log.info(f"Loaded {len(nodes_df):,} nodes")

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
def offlabel_train_test_split(
    context: AssetExecutionContext,
    offlabel_edges_pruned: Dict[str, Any],
) -> Output[Dict[str, Any]]:
    """
    Step 3: Prepare train/validation/test splits for off-label drug discovery.

    Positive examples: off-label_use edges
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

    # Save test metrics
    metrics_path = Path("data/07_model_output/offlabel") / "09_test_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)

    context.log.info(f"Saved test metrics to {metrics_path}")

    context.log.info("=" * 80)
    context.log.info("OFF-LABEL DRUG DISCOVERY PIPELINE COMPLETE!")
    context.log.info("=" * 80)
    context.log.info(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
    context.log.info(f"Test AUC-PR: {test_metrics['auc_pr']:.4f}")
    context.log.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

    return Output(
        value=test_metrics,
        metadata={
            "auc_roc": test_metrics['auc_roc'],
            "auc_pr": test_metrics['auc_pr'],
            "accuracy": test_metrics['accuracy'],
            "precision": test_metrics['precision'],
            "recall": test_metrics['recall'],
            "f1": test_metrics['f1'],
            "sensitivity": test_metrics['sensitivity'],
            "specificity": test_metrics['specificity'],
            "true_positives": test_metrics['true_positives'],
            "true_negatives": test_metrics['true_negatives'],
            "false_positives": test_metrics['false_positives'],
            "false_negatives": test_metrics['false_negatives'],
            "all_metrics": MetadataValue.json(test_metrics),
        },
    )
