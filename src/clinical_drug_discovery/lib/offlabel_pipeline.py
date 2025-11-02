"""
Off-Label Drug Discovery: Complete Pipeline

This script orchestrates the complete 9-step pipeline:
1. Load graph with selective edge types
2. Prune drug-drug and protein-protein edges
3. Prepare train/test data
4. Create heterogeneous graph
5. Initialize node embeddings
6. Build R-GCN model
7. Mini-batch sampling
8. Train with early stopping
9. Evaluate on test set
"""

import argparse
import logging
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import pandas as pd
from dotenv import load_dotenv

from .offlabel_data_loading import (
    MemgraphGraphLoader,
    CSVGraphLoader,
    GraphPruner,
    OffLabelDataPreparator,
    create_heterogeneous_graph
)
from .offlabel_model import initialize_model
from .offlabel_training import (
    create_dataloaders,
    train_model,
    test_model
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OffLabelPipeline:
    """Complete pipeline for off-label drug discovery."""

    def __init__(
        self,
        data_source: str = "csv",
        data_dir: str = "data/01_raw/primekg",
        memgraph_uri: str = None,
        memgraph_user: str = "",
        memgraph_password: str = "",
        memgraph_database: str = "memgraph",
        output_dir: str = "./output",
        device: Optional[torch.device] = None
    ):
        """
        Initialize pipeline.

        Args:
            data_source: Data source - "csv" or "memgraph"
            data_dir: Directory containing CSV files (for CSV mode)
            memgraph_uri: Memgraph connection URI (for Memgraph mode)
            memgraph_user: Memgraph username
            memgraph_password: Memgraph password
            memgraph_database: Memgraph database name
            output_dir: Directory to save outputs
            device: Device to use for training (auto-detected if None)
        """
        self.data_source = data_source
        self.data_dir = data_dir
        self.memgraph_uri = memgraph_uri
        self.memgraph_user = memgraph_user
        self.memgraph_password = memgraph_password
        self.memgraph_database = memgraph_database
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        logger.info(f"Initialized pipeline with device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")

        # Pipeline data
        self.loader = None
        self.edges_df = None
        self.node_metadata_df = None
        self.pruned_edges_df = None
        self.datasets = None
        self.train_graph = None
        self.node_mapping = None
        self.model = None
        self.history = None

    def run_step_1_load_graph(self):
        """Step 1: Load graph with selective edge types (from CSV or Memgraph)."""
        logger.info("=" * 80)
        logger.info(f"STEP 1: Loading graph with selective edge types from {self.data_source.upper()}")
        logger.info("=" * 80)

        # Initialize appropriate loader based on data source
        if self.data_source == "csv":
            self.loader = CSVGraphLoader(data_dir=self.data_dir)
        elif self.data_source == "memgraph":
            if not self.memgraph_uri:
                raise ValueError("Memgraph URI is required when data_source='memgraph'")
            self.loader = MemgraphGraphLoader(
                uri=self.memgraph_uri,
                user=self.memgraph_user,
                password=self.memgraph_password,
                database=self.memgraph_database
            )
        else:
            raise ValueError(f"Invalid data_source: {self.data_source}. Must be 'csv' or 'memgraph'")

        # Load edges with filtering
        self.edges_df = self.loader.load_edges_with_filter()
        logger.info(f"Loaded {len(self.edges_df):,} edges")

        # Load node metadata
        self.node_metadata_df = self.loader.get_node_metadata()
        logger.info(f"Loaded {len(self.node_metadata_df):,} nodes")

        # Save raw data
        edges_path = self.output_dir / "01_raw_edges.csv"
        nodes_path = self.output_dir / "01_raw_nodes.csv"
        self.edges_df.to_csv(edges_path, index=False)
        self.node_metadata_df.to_csv(nodes_path, index=False)
        logger.info(f"Saved raw data to {edges_path} and {nodes_path}")

    def run_step_2_prune_graph(self, drug_top_k: int = 10, protein_top_k: int = 20):
        """
        Step 2: Prune drug-drug and protein-protein edges.

        Args:
            drug_top_k: Number of top drug neighbors to keep
            protein_top_k: Number of top protein neighbors to keep
        """
        logger.info("=" * 80)
        logger.info("STEP 2: Pruning graph")
        logger.info("=" * 80)

        pruner = GraphPruner(self.loader)

        # Prune drug-drug edges
        logger.info(f"Pruning drug-drug edges (top {drug_top_k})...")
        pruned_df = pruner.prune_drug_drug_edges(self.edges_df, top_k=drug_top_k)

        # Prune protein-protein edges
        logger.info(f"Pruning protein-protein edges (top {protein_top_k})...")
        self.pruned_edges_df = pruner.prune_protein_protein_edges(pruned_df, top_k=protein_top_k)

        logger.info(f"Final edge count: {len(self.pruned_edges_df):,}")

        # Save pruned data
        pruned_path = self.output_dir / "02_pruned_edges.csv"
        self.pruned_edges_df.to_csv(pruned_path, index=False)
        logger.info(f"Saved pruned edges to {pruned_path}")

    def run_step_3_prepare_data(
        self,
        test_size: float = 0.2,
        num_contraindication_samples: int = 4800,
        num_random_negatives: int = 1600,
        val_split: float = 0.2,
        random_seed: int = 42
    ):
        """
        Step 3: Prepare train/test/validation splits.

        Args:
            test_size: Fraction for test set
            num_contraindication_samples: Number of contraindications to sample
            num_random_negatives: Number of random negatives
            val_split: Fraction of train set to use for validation
            random_seed: Random seed
        """
        logger.info("=" * 80)
        logger.info("STEP 3: Preparing train/test/validation data")
        logger.info("=" * 80)

        preparator = OffLabelDataPreparator(self.pruned_edges_df)
        self.datasets = preparator.prepare_link_prediction_data(
            test_size=test_size,
            num_contraindication_samples=num_contraindication_samples,
            num_random_negatives=num_random_negatives,
            random_seed=random_seed
        )

        # Further split train into train/val
        train_positives = self.datasets['train_positives']
        train_negatives = self.datasets['train_negatives']

        from sklearn.model_selection import train_test_split

        # Split positives
        train_pos, val_pos = train_test_split(
            train_positives, test_size=val_split, random_state=random_seed
        )

        # Split negatives
        train_neg, val_neg = train_test_split(
            train_negatives, test_size=val_split, random_state=random_seed
        )

        # Update datasets
        self.datasets['train_positives'] = train_pos
        self.datasets['train_negatives'] = train_neg
        self.datasets['val_positives'] = val_pos
        self.datasets['val_negatives'] = val_neg

        logger.info(f"Final dataset splits:")
        logger.info(f"  Train: {len(train_pos):,} pos + {len(train_neg):,} neg = {len(train_pos) + len(train_neg):,}")
        logger.info(f"  Val: {len(val_pos):,} pos + {len(val_neg):,} neg = {len(val_pos) + len(val_neg):,}")
        logger.info(f"  Test: {len(self.datasets['test_positives']):,} pos + "
                   f"{len(self.datasets['test_negatives']):,} neg = "
                   f"{len(self.datasets['test_positives']) + len(self.datasets['test_negatives']):,}")

        # Save splits
        for split_name, split_data in self.datasets.items():
            if isinstance(split_data, pd.DataFrame):
                split_path = self.output_dir / f"03_{split_name}.csv"
                split_data.to_csv(split_path, index=False)
                logger.info(f"Saved {split_name} to {split_path}")

    def run_step_4_create_graph(self):
        """Step 4: Create heterogeneous graph structure."""
        logger.info("=" * 80)
        logger.info("STEP 4: Creating heterogeneous graph")
        logger.info("=" * 80)

        # Use training edges only (exclude test positives)
        self.train_graph, self.node_mapping = create_heterogeneous_graph(
            self.datasets['train_edges'],
            self.node_metadata_df
        )

        logger.info(f"Created heterogeneous graph:")
        logger.info(f"  Node types: {len(self.train_graph.node_types)}")
        logger.info(f"  Edge types: {len(self.train_graph.edge_types)}")

        # Save node mapping
        node_mapping_path = self.output_dir / "04_node_mapping.json"
        # Convert to serializable format
        serializable_mapping = {
            node_type: {node_id: int(idx) for node_id, idx in mapping.items()}
            for node_type, mapping in self.node_mapping.items()
        }
        with open(node_mapping_path, 'w') as f:
            json.dump(serializable_mapping, f, indent=2)
        logger.info(f"Saved node mapping to {node_mapping_path}")

    def run_step_5_6_initialize_model(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Steps 5-6: Initialize node embeddings and build R-GCN model.

        Args:
            embedding_dim: Dimension of node embeddings
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            dropout: Dropout probability
        """
        logger.info("=" * 80)
        logger.info("STEPS 5-6: Initializing node embeddings and R-GCN model")
        logger.info("=" * 80)

        self.model = initialize_model(
            data=self.train_graph,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            device=self.device
        )

        # Log model statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model statistics:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Embedding dimension: {embedding_dim}")
        logger.info(f"  Hidden dimension: {hidden_dim}")
        logger.info(f"  Number of layers: {num_layers}")

    def run_step_7_8_train_model(
        self,
        batch_size: int = 64,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        grad_clip_norm: float = 1.0,
        early_stopping_patience: int = 10
    ):
        """
        Steps 7-8: Mini-batch training with early stopping.

        Args:
            batch_size: Batch size
            num_epochs: Maximum number of epochs
            learning_rate: Learning rate
            weight_decay: Weight decay
            grad_clip_norm: Gradient clipping norm
            early_stopping_patience: Early stopping patience
        """
        logger.info("=" * 80)
        logger.info("STEPS 7-8: Training model with mini-batch sampling")
        logger.info("=" * 80)

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_positives=self.datasets['train_positives'],
            train_negatives=self.datasets['train_negatives'],
            val_positives=self.datasets['val_positives'],
            val_negatives=self.datasets['val_negatives'],
            node_mapping=self.node_mapping,
            batch_size=batch_size,
            num_workers=0
        )

        # Save checkpoint path
        checkpoint_path = self.output_dir / "08_best_model.pt"

        # Train model
        self.history = train_model(
            model=self.model,
            data=self.train_graph,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            early_stopping_patience=early_stopping_patience,
            checkpoint_path=str(checkpoint_path)
        )

        # Save training history
        history_path = self.output_dir / "08_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")

    def run_step_9_evaluate(self, batch_size: int = 64):
        """
        Step 9: Evaluate on test set.

        Args:
            batch_size: Batch size for evaluation
        """
        logger.info("=" * 80)
        logger.info("STEP 9: Evaluating on test set")
        logger.info("=" * 80)

        # Load best model checkpoint
        checkpoint_path = self.output_dir / "08_best_model.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from {checkpoint_path}")
        else:
            logger.warning("No checkpoint found, using current model state")

        # Evaluate
        test_metrics = test_model(
            model=self.model,
            data=self.train_graph,
            test_positives=self.datasets['test_positives'],
            test_negatives=self.datasets['test_negatives'],
            node_mapping=self.node_mapping,
            device=self.device,
            batch_size=batch_size
        )

        # Save test metrics
        metrics_path = self.output_dir / "09_test_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        logger.info(f"Saved test metrics to {metrics_path}")

        return test_metrics

    def run_complete_pipeline(
        self,
        # Data parameters
        drug_top_k: int = 10,
        protein_top_k: int = 20,
        test_size: float = 0.2,
        val_split: float = 0.2,
        # Model parameters
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        # Training parameters
        batch_size: int = 64,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
    ):
        """
        Run the complete 9-step pipeline.

        Args:
            drug_top_k: Number of top drug neighbors to keep
            protein_top_k: Number of top protein neighbors to keep
            test_size: Fraction for test set
            val_split: Fraction of train for validation
            embedding_dim: Dimension of node embeddings
            hidden_dim: Hidden dimension for GNN
            num_layers: Number of GNN layers
            dropout: Dropout probability
            batch_size: Batch size
            num_epochs: Maximum epochs
            learning_rate: Learning rate
            early_stopping_patience: Early stopping patience

        Returns:
            Dictionary with test metrics
        """
        start_time = datetime.now()
        logger.info("=" * 80)
        logger.info("STARTING COMPLETE OFF-LABEL DRUG DISCOVERY PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Start time: {start_time}")

        try:
            # Step 1: Load graph
            self.run_step_1_load_graph()

            # Step 2: Prune graph
            self.run_step_2_prune_graph(drug_top_k=drug_top_k, protein_top_k=protein_top_k)

            # Step 3: Prepare data
            self.run_step_3_prepare_data(test_size=test_size, val_split=val_split)

            # Step 4: Create graph
            self.run_step_4_create_graph()

            # Steps 5-6: Initialize model
            self.run_step_5_6_initialize_model(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            )

            # Steps 7-8: Train model
            self.run_step_7_8_train_model(
                batch_size=batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                early_stopping_patience=early_stopping_patience
            )

            # Step 9: Evaluate
            test_metrics = self.run_step_9_evaluate(batch_size=batch_size)

            # Save pipeline configuration
            config = {
                'data_params': {
                    'drug_top_k': drug_top_k,
                    'protein_top_k': protein_top_k,
                    'test_size': test_size,
                    'val_split': val_split,
                },
                'model_params': {
                    'embedding_dim': embedding_dim,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'dropout': dropout,
                },
                'training_params': {
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'learning_rate': learning_rate,
                    'early_stopping_patience': early_stopping_patience,
                },
                'device': str(self.device),
                'start_time': str(start_time),
                'end_time': str(datetime.now()),
            }

            config_path = self.output_dir / "pipeline_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved pipeline configuration to {config_path}")

            end_time = datetime.now()
            duration = end_time - start_time

            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETE!")
            logger.info("=" * 80)
            logger.info(f"End time: {end_time}")
            logger.info(f"Total duration: {duration}")
            logger.info(f"Final test AUC-ROC: {test_metrics['auc_roc']:.4f}")
            logger.info(f"Final test AUC-PR: {test_metrics['auc_pr']:.4f}")

            return test_metrics

        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}", exc_info=True)
            raise

        finally:
            if self.loader:
                self.loader.close()


def main():
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(description="Off-Label Drug Discovery Pipeline")

    # Data source parameters
    parser.add_argument("--data-source", type=str, default="csv", choices=["csv", "memgraph"],
                       help="Data source: 'csv' or 'memgraph' (default: csv)")
    parser.add_argument("--data-dir", type=str, default="data/01_raw/primekg",
                       help="Directory containing CSV files (for CSV mode)")

    # Memgraph connection parameters (for Memgraph mode)
    parser.add_argument("--memgraph-uri", type=str, default=None,
                       help="Memgraph connection URI (required for Memgraph mode)")
    parser.add_argument("--memgraph-user", type=str, default="",
                       help="Memgraph username")
    parser.add_argument("--memgraph-password", type=str, default="",
                       help="Memgraph password")
    parser.add_argument("--memgraph-database", type=str, default="memgraph",
                       help="Memgraph database name")

    # Output parameters
    parser.add_argument("--output-dir", type=str, default="./output",
                       help="Output directory")

    # Data parameters
    parser.add_argument("--drug-top-k", type=int, default=10,
                       help="Top K drug-drug neighbors")
    parser.add_argument("--protein-top-k", type=int, default=20,
                       help="Top K protein-protein neighbors")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Test set fraction")
    parser.add_argument("--val-split", type=float, default=0.2,
                       help="Validation set fraction")

    # Model parameters
    parser.add_argument("--embedding-dim", type=int, default=128,
                       help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=128,
                       help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2,
                       help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="Dropout probability")

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                       help="Early stopping patience")

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Get connection parameters (for Memgraph mode)
    memgraph_uri = args.memgraph_uri or os.getenv("MEMGRAPH_URI")

    # Initialize pipeline
    pipeline = OffLabelPipeline(
        data_source=args.data_source,
        data_dir=args.data_dir,
        memgraph_uri=memgraph_uri,
        memgraph_user=args.memgraph_user,
        memgraph_password=args.memgraph_password,
        memgraph_database=args.memgraph_database,
        output_dir=args.output_dir
    )

    # Run complete pipeline
    pipeline.run_complete_pipeline(
        drug_top_k=args.drug_top_k,
        protein_top_k=args.protein_top_k,
        test_size=args.test_size,
        val_split=args.val_split,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
    )


if __name__ == "__main__":
    main()
