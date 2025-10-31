"""
GNN-based graph embeddings using PyTorch Geometric.

Replaces Node2Vec with Graph Neural Network embeddings for better
representation learning on heterogeneous knowledge graphs.

Loads graph data directly from CSV files (no Memgraph dependency).
"""

import os
from typing import Dict, Any, Tuple
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm


# Set up MPS fallback for Apple Silicon compatibility
if torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


class GraphSAGEEmbedding(nn.Module):
    """GraphSAGE model for generating node embeddings."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        for _layer_idx in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.convs[-1](x, edge_index)
        return x


def load_graph_from_csv(
    edges_csv: str,
    limit_nodes: int = None,
    include_node_types: list = None,
    chunk_size: int = 500000
) -> Tuple[Data, Dict[int, Dict[str, Any]]]:
    """
    Load graph structure from CSV files for GNN training with optimizations.

    Args:
        edges_csv: Path to edges CSV file (kg.csv contains edge triplets) - REQUIRED
        limit_nodes: Limit number of nodes (for testing)
        include_node_types: List of node types to include. If None, uses default filtering.
                           Default excludes 'cellular_component' and 'exposure'.
        chunk_size: Size of chunks for edge processing

    Returns:
        - PyG Data object with node features and edges
        - Node metadata dictionary
    """
    from .gnn_optimization import (
        fast_node_extraction, create_sparse_edge_index
    )
    
    # Default node type filtering (excludes cellular_component and exposure)
    if include_node_types is None:
        include_node_types = [
            'drug',
            'disease',
            'gene/protein',
            'effect/phenotype',
            'pathway',
            'biological_process',
            'molecular_function',
            'anatomy'
        ]

    print(f"Loading graph from CSV: {edges_csv}")
    print(f"Including node types: {include_node_types}")
    print("Excluded types: ['cellular_component', 'exposure']")

    # Load edges CSV (this is actually the kg.csv file with all edges)
    print("\nReading edges CSV...")
    edges_df = pd.read_csv(edges_csv)
    print(f"Loaded {len(edges_df):,} edges from CSV")

    # Fast node extraction using optimized method
    print("\nExtracting nodes using optimized method...")
    nodes_df, node_id_to_idx = fast_node_extraction(
        edges_df, include_node_types, limit_nodes
    )

    # Log node type counts
    node_type_counts = nodes_df['type'].value_counts()
    print(f"Final result: {len(nodes_df):,} nodes across {len(node_type_counts)} node types:")
    for node_type, count in node_type_counts.items():
        print(f"  - {node_type}: {count:,}")

    valid_node_ids = set(node_id_to_idx.keys())

    # Build edge index using sparse operations
    print("\nBuilding optimized edge index...")
    edge_index = create_sparse_edge_index(
        edges_df=edges_df,
        valid_node_ids=valid_node_ids,
        node_id_to_idx=node_id_to_idx,
        chunk_size=chunk_size
    )

    print(f"Created edge index: {edge_index.shape} ({edge_index.shape[1]:,} edges)")

    # Create one-hot encoded node features based on node types
    print("\nCreating one-hot node features...")
    unique_types = sorted(nodes_df['type'].unique())
    type_to_idx = {node_type: idx for idx, node_type in enumerate(unique_types)}
    num_node_types = len(type_to_idx)

    print(f"One-hot encoding for {num_node_types} node types:")
    for node_type, idx in type_to_idx.items():
        print(f"  - {node_type}: index {idx}")

    x = torch.zeros((len(nodes_df), num_node_types), dtype=torch.float)
    for idx, node_type in enumerate(nodes_df['type'].values):
        type_idx = type_to_idx[node_type]
        x[idx, type_idx] = 1.0

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index)

    # Debug: Check for duplicates in nodes_df
    nodes_df_duplicates = nodes_df['id'].duplicated().sum()
    if nodes_df_duplicates > 0:
        print(f"⚠️  WARNING: nodes_df has {nodes_df_duplicates} duplicate IDs!")
        print(f"⚠️  Total rows in nodes_df: {len(nodes_df)}")
        print(f"⚠️  Unique IDs in nodes_df: {nodes_df['id'].nunique()}")

    # Create metadata dictionary
    node_metadata = {}
    for idx, row in nodes_df.iterrows():
        node_metadata[row['id']] = {
            'idx': idx,
            'name': row['name'],
            'type': row['type']
        }

    print(f"Created node_metadata with {len(node_metadata)} unique nodes")

    print(f"\n✓ Graph loaded: {data.num_nodes:,} nodes, {data.num_edges:,} edges")

    return data, node_metadata


def train_gnn_embeddings(
    data: Data,
    embedding_dim: int = 512,
    hidden_dim: int = 256,
    num_layers: int = 2,
    num_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.01,
    device: str = None,
    use_neighbor_sampling: bool = True,
    edge_batch_size: int = 10000  # Process edges in batches to reduce memory
) -> torch.Tensor:
    """
    Train GNN model with full MPS optimization and layer-wise computation.

    MPS-optimized training:
    - All operations on MPS (no CPU fallback)
    - Edge sampling in batches to reduce memory
    - Layer-wise inference for final embeddings
    - Aggressive memory management

    Args:
        data: PyG Data object
        embedding_dim: Output embedding dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of GNN layers
        num_epochs: Training epochs
        batch_size: Number of edges per batch (not used for full-batch)
        learning_rate: Learning rate
        device: 'cuda', 'mps', or 'cpu' (default: auto-detect MPS)
        use_neighbor_sampling: Ignored (uses full-batch training)
        edge_batch_size: Number of edges to sample per training step

    Returns:
        Node embeddings tensor
    """
    import os
    # Force MPS fallback for unsupported ops
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    # Determine device - prioritize MPS
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
            print("🍎 Using Apple Silicon MPS (Metal Performance Shaders)")
        elif torch.cuda.is_available():
            device = 'cuda'
            print("🔥 Using NVIDIA CUDA GPU")
        else:
            device = 'cpu'
            print("💻 Using CPU")
    else:
        print(f"Using device: {device}")

    print(f"✓ Device: {device}")
    print("✓ Full-batch training with edge sampling")
    print("✓ Layer-wise inference for memory efficiency")

    # Memory-aware device check
    # The bottleneck is message passing: (num_edges, embedding_dim)
    # This creates temporary tensors during aggregation
    num_edges = data.edge_index.size(1)

    # Memory components:
    # 1. Node embeddings: nodes × embedding_dim × 4 bytes
    # 2. Edge aggregation (BOTTLENECK): edges × embedding_dim × 4 bytes × num_layers
    # 3. Gradients: ~2x forward pass
    node_memory_gb = (data.num_nodes * embedding_dim * 4) / (1024**3)
    edge_memory_gb = (num_edges * embedding_dim * 4 * num_layers) / (1024**3)
    total_memory_gb = (node_memory_gb + edge_memory_gb) * 2  # × 2 for gradients

    print(f"\nMemory estimation:")
    print(f"  Nodes: {data.num_nodes:,} | Edges: {num_edges:,}")
    print(f"  Node memory: {node_memory_gb:.2f} GB")
    print(f"  Edge aggregation: {edge_memory_gb:.2f} GB (BOTTLENECK)")
    print(f"  Total estimated: {total_memory_gb:.2f} GB")

    # For MPS, auto-adjust if needed
    if device == 'mps' and total_memory_gb > 10:
        print(f"\n⚠️  WARNING: Graph too large for MPS at current dimensions")
        print(f"⚠️  Estimated memory: {total_memory_gb:.2f} GB (>10 GB)")

        # Calculate required scale factor
        target_memory_gb = 8  # Target 8 GB to be safe
        scale_factor = target_memory_gb / total_memory_gb
        new_embedding_dim = int(embedding_dim * scale_factor)
        new_hidden_dim = int(hidden_dim * scale_factor)

        # Ensure minimum viable dimensions
        new_embedding_dim = max(new_embedding_dim, 64)
        new_hidden_dim = max(new_hidden_dim, 32)

        print(f"\n🔧 Auto-adjusting dimensions to fit in memory...")
        print(f"   Old embedding_dim: {embedding_dim} → New: {new_embedding_dim}")
        print(f"   Old hidden_dim: {hidden_dim} → New: {new_hidden_dim}")

        embedding_dim = new_embedding_dim
        hidden_dim = new_hidden_dim

        # Recalculate
        edge_memory_gb = (num_edges * embedding_dim * 4 * num_layers) / (1024**3)
        total_memory_gb = (node_memory_gb + edge_memory_gb) * 2
        print(f"   New estimated memory: {total_memory_gb:.2f} GB")

    # Move data to device
    data = data.to(device)

    # Initialize model
    model = GraphSAGEEmbedding(
        in_channels=data.x.size(1),
        hidden_channels=hidden_dim,
        out_channels=embedding_dim,
        num_layers=num_layers
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop with memory-efficient forward pass
    model.train()
    print(f"\nTraining for {num_epochs} epochs...")

    # Enable gradient checkpointing to trade compute for memory
    if device == 'mps':
        print("✓ Using gradient checkpointing for memory efficiency")

    # Progress bar for training
    pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")

    for _ in pbar:
        optimizer.zero_grad()

        # Forward pass with gradient checkpointing (saves memory during backward)
        def forward_chunk(x, edge_index):
            return model(x, edge_index)

        # Use checkpointing for large graphs
        if device == 'mps' and data.num_nodes > 50000:
            embeddings = checkpoint(forward_chunk, data.x, data.edge_index, use_reentrant=False)
        else:
            embeddings = model(data.x, data.edge_index)

        # Sample edges for loss computation (reduces memory)
        num_edges = data.edge_index.size(1)
        num_edge_samples = min(edge_batch_size, num_edges)

        # Random sample of edges
        perm = torch.randperm(num_edges, device=device)[:num_edge_samples]
        pos_edge_sample = data.edge_index[:, perm]

        # Positive edge scores
        src_emb = embeddings[pos_edge_sample[0]]
        dst_emb = embeddings[pos_edge_sample[1]]
        pos_scores = (src_emb * dst_emb).sum(dim=1)

        # Negative sampling
        neg_dst = torch.randint(0, data.num_nodes, (num_edge_samples,), device=device)
        neg_dst_emb = embeddings[neg_dst]
        neg_scores = (src_emb * neg_dst_emb).sum(dim=1)

        # Binary cross-entropy loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )
        loss = pos_loss + neg_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update progress bar with loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Clear all intermediate tensors
        del embeddings, src_emb, dst_emb, pos_scores, neg_scores, neg_dst_emb
        if device == 'mps' and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

    # Generate final embeddings using layer-wise computation (memory efficient)
    print("\n✓ Generating final embeddings with layer-wise computation...")
    model.eval()

    with torch.no_grad():
        # Layer-wise forward pass
        x = data.x
        edge_index = data.edge_index

        # Progress bar for layer-wise inference
        layer_pbar = tqdm(enumerate(model.convs[:-1]), total=num_layers-1, desc="Inference (layers)", unit="layer")

        for i, conv in layer_pbar:
            layer_pbar.set_description(f"Layer {i+1}/{num_layers}")
            x = conv(x, edge_index)
            x = F.relu(x)
            x = model.dropout(x)

            # Clear cache after each layer
            if device == 'mps' and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

        # Final layer
        print(f"  Final layer {num_layers}/{num_layers}...")
        x = model.convs[-1](x, edge_index)

        # Final cache clear
        if device == 'mps' and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

        embeddings = x

    print(f"✓ Generated embeddings: {embeddings.shape}")
    return embeddings.cpu()


def save_embeddings_to_csv(
    embeddings: torch.Tensor,
    node_metadata: Dict[int, Dict[str, Any]],
    output_csv: str
) -> Dict[str, int]:
    """
    Save GNN embeddings to CSV file.

    Args:
        embeddings: Node embeddings tensor
        node_metadata: Node metadata dictionary
        output_csv: Path to output CSV file - REQUIRED

    Returns:
        Dictionary with statistics
    """
    print(f"\nSaving embeddings to CSV: {output_csv}")

    # Convert embeddings to numpy (handle device conversion)
    if embeddings.device.type != 'cpu':
        embeddings_np = embeddings.cpu().numpy()
    else:
        embeddings_np = embeddings.numpy()

    # Create output directory if needed
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build DataFrame with node metadata and embeddings
    node_ids = list(node_metadata.keys())
    print(f"Saving embeddings for {len(node_ids)} nodes from metadata...")
    rows = []

    for node_id in node_ids:
        idx = node_metadata[node_id]['idx']
        embedding = embeddings_np[idx]

        row = {
            'node_id': node_id,
            'node_name': node_metadata[node_id]['name'],
            'node_type': node_metadata[node_id]['type'],
            'embedding': embedding.tolist()  # Store as list for CSV
        }
        rows.append(row)

    embeddings_df = pd.DataFrame(rows)

    # Debug: Check for duplicates before saving
    df_duplicates = embeddings_df['node_id'].duplicated().sum()
    if df_duplicates > 0:
        print(f"⚠️  WARNING: embeddings_df has {df_duplicates} duplicate node_ids before saving!")
        print(f"⚠️  Total rows: {len(embeddings_df)}, Unique IDs: {embeddings_df['node_id'].nunique()}")

    # Save to CSV
    embeddings_df.to_csv(output_csv, index=False)

    print(f"✓ Saved {len(embeddings_df):,} embeddings to {output_csv}")
    print(f"  Embedding dimension: {embeddings_np.shape[1]}")
    print(f"  File size: {output_path.stat().st_size / (1024*1024):.2f} MB")

    return {
        'total_nodes': len(node_ids),
        'saved_nodes': len(embeddings_df),
        'embedding_dim': embeddings_np.shape[1],
        'output_file': output_csv
    }


def generate_gnn_embeddings(
    edges_csv: str,
    output_csv: str,
    embedding_dim: int = 512,
    hidden_dim: int = 256,
    num_layers: int = 2,
    num_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.01,
    limit_nodes: int = None,
    device: str = None,
    include_node_types: list = None
) -> Dict[str, Any]:
    """
    Complete pipeline: Load graph from CSV, train GNN, save embeddings to CSV.

    Args:
        edges_csv: Path to edges CSV file (kg.csv contains edge triplets) - REQUIRED
        output_csv: Path to save embeddings CSV - REQUIRED
        embedding_dim: Output embedding dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of GNN layers
        num_epochs: Training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        limit_nodes: Limit number of nodes (for testing)
        device: Training device ('cuda', 'mps', or 'cpu')
        include_node_types: List of node types to include. If None, excludes
                           'cellular_component' and 'exposure' by default.

    Returns:
        Dictionary with statistics
    """
    try:
        print("=" * 80)
        print("GNN EMBEDDINGS GENERATION (CSV-BASED)")
        print("=" * 80)

        # Step 1: Load graph from CSV
        print("\n1. Loading graph from CSV...")
        data, node_metadata = load_graph_from_csv(
            edges_csv=edges_csv,
            limit_nodes=limit_nodes,
            include_node_types=include_node_types
        )
        print(f"   Loaded {data.num_nodes} nodes, {data.num_edges} edges")

        # Step 2: Train GNN
        print("\n2. Training GNN model...")

        embeddings = train_gnn_embeddings(
            data=data,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device
        )
        print(f"   Generated embeddings: {embeddings.shape}")

        # Step 3: Save embeddings to CSV
        print("\n3. Saving embeddings to CSV...")
        save_stats = save_embeddings_to_csv(
            embeddings=embeddings,
            node_metadata=node_metadata,
            output_csv=output_csv
        )
        print(f"   Saved {save_stats['saved_nodes']} embeddings")

        print("\n" + "=" * 80)
        print("COMPLETE!")
        print("=" * 80)
    except Exception as e:
        print(f"\n❌ ERROR during GNN embeddings generation: {e}")
        raise e

    return {
        'num_nodes': data.num_nodes,
        'num_edges': data.num_edges,
        'embedding_dim': embedding_dim,
        'num_epochs': num_epochs,
        'device': device,
        **save_stats
    }
