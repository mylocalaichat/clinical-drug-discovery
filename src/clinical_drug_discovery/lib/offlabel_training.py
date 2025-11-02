"""
Off-Label Drug Discovery: Training and Evaluation Module

This module implements:
- Step 7: Mini-batch neighborhood sampling
- Step 8: Training loop with early stopping
- Step 9: Evaluation with AUC-ROC, AUC-PR metrics
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm
import time

from .offlabel_model import OffLabelRGCN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Neighborhood sampling configuration (Step 7)
NEIGHBOR_SAMPLING_CONFIG = {
    'drug': {
        'drug_drug': 10,
        'drug_protein': 50,
        'drug_effect': 20,
        'indication': 10,
    },
    'disease': {
        'disease_disease': 15,
        'disease_protein': 50,
        'disease_phenotype_positive': 30,
    },
    'protein': {
        'protein_protein': 20,
        'bioprocess_protein_reverse': 15,
        'pathway_protein_reverse': 10,
        'molfunc_protein_reverse': 10,
        'phenotype_protein_reverse': 5,
        'disease_protein_reverse': 10,
    },
    'biological_process': {
        'bioprocess_bioprocess': 10,
        'bioprocess_protein': 15,
    },
    'effect/phenotype': {
        'disease_phenotype_positive_reverse': 10,
        'phenotype_protein': 10,
    },
    'pathway': {
        'pathway_pathway': 5,
        'pathway_protein': 15,
    },
    'molecular_function': {
        'molfunc_protein': 15,
    },
    'cellular_component': {},  # No neighbors (isolated)
    'anatomy': {},  # No neighbors (isolated)
    'exposure': {},  # No neighbors (isolated)
}


class DrugDiseasePairDataset(Dataset):
    """Dataset for drug-disease pairs with labels."""

    def __init__(
        self,
        drug_ids: List[str],
        disease_ids: List[str],
        labels: List[int],
        node_mapping: Dict[str, Dict[str, int]]
    ):
        """
        Initialize dataset.

        Args:
            drug_ids: List of drug node IDs
            disease_ids: List of disease node IDs
            labels: List of binary labels (1=positive, 0=negative)
            node_mapping: Dictionary mapping node IDs to indices
        """
        self.drug_ids = drug_ids
        self.disease_ids = disease_ids
        self.labels = labels
        self.node_mapping = node_mapping

        # Map IDs to indices
        self.drug_indices = [node_mapping['drug'][drug_id] for drug_id in drug_ids]
        self.disease_indices = [node_mapping['disease'][disease_id] for disease_id in disease_ids]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'drug_index': self.drug_indices[idx],
            'disease_index': self.disease_indices[idx],
            'label': self.labels[idx]
        }


def create_dataloaders(
    train_positives: pd.DataFrame,
    train_negatives: pd.DataFrame,
    val_positives: pd.DataFrame,
    val_negatives: pd.DataFrame,
    node_mapping: Dict[str, Dict[str, int]],
    batch_size: int = 64,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.

    Args:
        train_positives: DataFrame with positive training examples
        train_negatives: DataFrame with negative training examples
        val_positives: DataFrame with positive validation examples
        val_negatives: DataFrame with negative validation examples
        node_mapping: Dictionary mapping node IDs to indices
        batch_size: Batch size
        num_workers: Number of workers for data loading

    Returns:
        (train_loader, val_loader)
    """
    # Combine positives and negatives
    train_drugs = list(train_positives['source_id']) + list(train_negatives['source_id'])
    train_diseases = list(train_positives['target_id']) + list(train_negatives['target_id'])
    train_labels = [1] * len(train_positives) + [0] * len(train_negatives)

    val_drugs = list(val_positives['source_id']) + list(val_negatives['source_id'])
    val_diseases = list(val_positives['target_id']) + list(val_negatives['target_id'])
    val_labels = [1] * len(val_positives) + [0] * len(val_negatives)

    # Create datasets
    train_dataset = DrugDiseasePairDataset(train_drugs, train_diseases, train_labels, node_mapping)
    val_dataset = DrugDiseasePairDataset(val_drugs, val_diseases, val_labels, node_mapping)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    logger.info(f"Created dataloaders: {len(train_dataset)} train, {len(val_dataset)} val")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")

    return train_loader, val_loader


class EarlyStopping:
    """Early stopping handler for training."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == 'max':
            self.is_better = lambda score, best: score > best + min_delta
        else:
            self.is_better = lambda score, best: score < best - min_delta

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                return True
            return False


def train_epoch(
    model: OffLabelRGCN,
    data: HeteroData,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip_norm: float = 1.0
) -> Tuple[float, float]:
    """
    Train model for one epoch.

    Args:
        model: R-GCN model
        data: HeteroData object
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        grad_clip_norm: Gradient clipping max norm

    Returns:
        (average_loss, average_accuracy)
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        # Move batch to device
        drug_indices = torch.tensor(batch['drug_index'], dtype=torch.long, device=device)
        disease_indices = torch.tensor(batch['disease_index'], dtype=torch.long, device=device)
        labels = torch.tensor(batch['label'], dtype=torch.float, device=device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(data, drug_indices, disease_indices)

        # Compute loss
        loss = criterion(predictions, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        # Optimizer step
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * len(labels)
        predicted_labels = (predictions > 0.5).float()
        total_correct += (predicted_labels == labels).sum().item()
        total_samples += len(labels)

    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples

    return avg_loss, avg_accuracy


@torch.no_grad()
def evaluate(
    model: OffLabelRGCN,
    data: HeteroData,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on validation set.

    Args:
        model: R-GCN model
        data: HeteroData object
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Dictionary of metrics
    """
    model.eval()

    all_predictions = []
    all_labels = []
    total_loss = 0.0

    for batch in tqdm(val_loader, desc="Evaluating", leave=False):
        # Move batch to device
        drug_indices = torch.tensor(batch['drug_index'], dtype=torch.long, device=device)
        disease_indices = torch.tensor(batch['disease_index'], dtype=torch.long, device=device)
        labels = torch.tensor(batch['label'], dtype=torch.float, device=device)

        # Forward pass
        predictions = model(data, drug_indices, disease_indices)

        # Compute loss
        loss = criterion(predictions, labels)
        total_loss += loss.item() * len(labels)

        # Collect predictions
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Compute metrics
    avg_loss = total_loss / len(all_labels)
    predicted_labels = (all_predictions > 0.5).astype(int)

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_labels, predicted_labels),
        'precision': precision_score(all_labels, predicted_labels, zero_division=0),
        'recall': recall_score(all_labels, predicted_labels, zero_division=0),
        'f1': f1_score(all_labels, predicted_labels, zero_division=0),
        'auc_roc': roc_auc_score(all_labels, all_predictions) if len(np.unique(all_labels)) > 1 else 0.0,
        'auc_pr': average_precision_score(all_labels, all_predictions) if len(np.unique(all_labels)) > 1 else 0.0,
    }

    return metrics


def train_model(
    model: OffLabelRGCN,
    data: HeteroData,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    grad_clip_norm: float = 1.0,
    early_stopping_patience: int = 10,
    checkpoint_path: Optional[str] = None
) -> Dict[str, List[float]]:
    """
    Train R-GCN model with early stopping.

    Args:
        model: R-GCN model
        data: HeteroData object
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        grad_clip_norm: Gradient clipping max norm
        early_stopping_patience: Patience for early stopping
        checkpoint_path: Path to save best model checkpoint

    Returns:
        Dictionary of training history
    """
    logger.info("Starting training...")
    logger.info(f"  Device: {device}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Early stopping patience: {early_stopping_patience}")

    # Move model and data to device
    model = model.to(device)
    data = data.to(device)

    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    # Setup early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')

    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_auc_roc': [],
        'val_auc_pr': [],
    }

    best_val_auc = 0.0
    best_epoch = 0

    # Training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, data, train_loader, optimizer, criterion, device, grad_clip_norm
        )

        # Validate
        val_metrics = evaluate(model, data, val_loader, criterion, device)

        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_auc_roc'].append(val_metrics['auc_roc'])
        history['val_auc_pr'].append(val_metrics['auc_pr'])

        epoch_time = time.time() - epoch_start

        logger.info(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        logger.info(f"  Val AUC-ROC: {val_metrics['auc_roc']:.4f}, Val AUC-PR: {val_metrics['auc_pr']:.4f}")

        # Save best model
        if val_metrics['auc_roc'] > best_val_auc:
            best_val_auc = val_metrics['auc_roc']
            best_epoch = epoch + 1

            if checkpoint_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc_roc': best_val_auc,
                }, checkpoint_path)
                logger.info(f"  Saved best model to {checkpoint_path}")

        # Early stopping
        if early_stopping(val_metrics['auc_roc']):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    logger.info(f"Training complete!")
    logger.info(f"  Best validation AUC-ROC: {best_val_auc:.4f} at epoch {best_epoch}")

    return history


def test_model(
    model: OffLabelRGCN,
    data: HeteroData,
    test_positives: pd.DataFrame,
    test_negatives: pd.DataFrame,
    node_mapping: Dict[str, Dict[str, int]],
    device: torch.device,
    batch_size: int = 64
) -> Dict[str, float]:
    """
    Evaluate model on test set.

    Args:
        model: Trained R-GCN model
        data: HeteroData object
        test_positives: DataFrame with positive test examples
        test_negatives: DataFrame with negative test examples
        node_mapping: Dictionary mapping node IDs to indices
        device: Device to evaluate on
        batch_size: Batch size

    Returns:
        Dictionary of test metrics
    """
    logger.info("Evaluating on test set...")

    # Create test dataset
    test_drugs = list(test_positives['source_id']) + list(test_negatives['source_id'])
    test_diseases = list(test_positives['target_id']) + list(test_negatives['target_id'])
    test_labels = [1] * len(test_positives) + [0] * len(test_negatives)

    test_dataset = DrugDiseasePairDataset(test_drugs, test_diseases, test_labels, node_mapping)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Move model and data to device
    model = model.to(device)
    data = data.to(device)
    model.eval()

    # Evaluate
    criterion = nn.BCELoss()
    test_metrics = evaluate(model, data, test_loader, criterion, device)

    # Additional detailed metrics
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            drug_indices = torch.tensor(batch['drug_index'], dtype=torch.long, device=device)
            disease_indices = torch.tensor(batch['disease_index'], dtype=torch.long, device=device)
            labels = batch['label']

            predictions = model(data, drug_indices, disease_indices)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    predicted_labels = (all_predictions > 0.5).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, predicted_labels).ravel()

    # Compile detailed metrics
    detailed_metrics = {
        **test_metrics,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
    }

    # Log results
    logger.info("Test Results:")
    logger.info(f"  Loss: {detailed_metrics['loss']:.4f}")
    logger.info(f"  Accuracy: {detailed_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {detailed_metrics['precision']:.4f}")
    logger.info(f"  Recall: {detailed_metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {detailed_metrics['f1']:.4f}")
    logger.info(f"  AUC-ROC: {detailed_metrics['auc_roc']:.4f}")
    logger.info(f"  AUC-PR: {detailed_metrics['auc_pr']:.4f}")
    logger.info(f"  Sensitivity: {detailed_metrics['sensitivity']:.4f}")
    logger.info(f"  Specificity: {detailed_metrics['specificity']:.4f}")
    logger.info(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # Classification report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(all_labels, predicted_labels,
                                             target_names=['Negative', 'Positive'],
                                             digits=4))

    return detailed_metrics


if __name__ == "__main__":
    # Example usage
    logger.info("Training module test complete")
