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
from torch_geometric.loader import NeighborLoader


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
    edges_csv: str = "data/01_raw/primekg/nodes.csv",
    limit_nodes: int = None,
    include_node_types: list = None,
    use_cache: bool = True,
    chunk_size: int = 500000
) -> Tuple[Data, Dict[int, Dict[str, Any]]]:
    """
    Load graph structure from CSV files for GNN training with optimizations.

    Note: The PrimeKG 'nodes.csv' file actually contains the edge list (kg.csv).

    Args:
        edges_csv: Path to edges CSV file (kg.csv, named nodes.csv in PrimeKG)
        limit_nodes: Limit number of nodes (for testing)
        include_node_types: List of node types to include. If None, uses default filtering.
                           Default excludes 'cellular_component' and 'exposure'.
        use_cache: Whether to use cached edge index if available
        chunk_size: Size of chunks for edge processing

    Returns:
        - PyG Data object with node features and edges
        - Node metadata dictionary
    """
    from .gnn_cache import (
        generate_cache_key, load_edge_index_cache, save_edge_index_cache
    )
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
    
    # Check cache first
    if use_cache:
        cache_key = generate_cache_key(edges_csv, include_node_types, limit_nodes)
        cached_data = load_edge_index_cache(cache_key)
        
        if cached_data is not None:
            edge_index, node_metadata, x = cached_data
            data = Data(x=x, edge_index=edge_index)
            print(f"\n✓ Loaded from cache: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
            return data, node_metadata

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

    # Create metadata dictionary
    node_metadata = {}
    for idx, row in nodes_df.iterrows():
        node_metadata[row['id']] = {
            'idx': idx,
            'name': row['name'],
            'type': row['type']
        }

    # Save to cache if enabled
    if use_cache:
        save_edge_index_cache(edge_index, node_metadata, x, cache_key)

    print(f"\n✓ Graph loaded: {data.num_nodes:,} nodes, {data.num_edges:,} edges")

    return data, node_metadata


def train_gnn_embeddings(
    data: Data,
    embedding_dim: int = 512,
    hidden_dim: int = 256,
    num_layers: int = 2,
    num_epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 0.01,
    device: str = None,
    use_neighbor_sampling: bool = True
) -> torch.Tensor:
    """
    Train GNN model to generate node embeddings.

    Uses unsupervised training with link prediction as the objective.
    Supports both full-batch and neighbor sampling approaches.

    Args:
        data: PyG Data object
        embedding_dim: Output embedding dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of GNN layers
        num_epochs: Training epochs
        batch_size: Batch size for neighbor sampling (if used)
        learning_rate: Learning rate
        device: 'cuda', 'mps', or 'cpu'
        use_neighbor_sampling: Whether to use neighbor sampling (requires pyg-lib or torch-sparse)

    Returns:
        Node embeddings tensor
    """
    # Determine device with MPS fallback handling
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
            print("🍎 Using Apple M1 Neural Engine (MPS)")
        elif torch.cuda.is_available():
            device = 'cuda'
            print("🔥 Using NVIDIA CUDA GPU")
        else:
            device = 'cpu'
            print("💻 Using CPU")
    else:
        print(f"Training on device: {device}")

    print(f"Device: {device}")

    # Handle MPS compatibility issues with PyTorch Geometric
    use_cpu_for_sampling = False
    if device == 'mps':
        try:
            # Test if MPS supports the required operations
            import os
            # Enable MPS fallback for unsupported operations
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            print("⚠️  Enabled MPS fallback for unsupported operations")
            
            # For NeighborLoader, we'll use CPU for sampling but GPU for training
            use_cpu_for_sampling = True
            print("📝 Using CPU for graph sampling, MPS for model training")
            
        except Exception as e:
            print(f"⚠️  MPS compatibility issue detected: {e}")
            print("🔄 Falling back to CPU for full training")
            device = 'cpu'

    # Move data to training device
    data = data.to(device)

    # Initialize model
    model = GraphSAGEEmbedding(
        in_channels=data.x.size(1),
        hidden_channels=hidden_dim,
        out_channels=embedding_dim,
        num_layers=num_layers
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create neighbor sampler for batch training with MPS compatibility
    try:
        # Try to create NeighborLoader with original data
        if use_cpu_for_sampling:
            # Use CPU data for sampling to avoid MPS compatibility issues
            data_for_sampling = data.to('cpu')
            train_loader = NeighborLoader(
                data_for_sampling,
                num_neighbors=[10, 5],  # 2-hop neighbors
                batch_size=batch_size,
                shuffle=True,
                num_workers=0  # For laptop compatibility
            )
            print("✓ Created NeighborLoader on CPU for MPS compatibility")
        else:
            train_loader = NeighborLoader(
                data,
                num_neighbors=[10, 5],  # 2-hop neighbors
                batch_size=batch_size,
                shuffle=True,
                num_workers=0  # For laptop compatibility
            )
            print(f"✓ Created NeighborLoader on {device}")
    
    except Exception as e:
        print(f"⚠️  NeighborLoader creation failed on {device}: {e}")
        print("🔄 Falling back to CPU for sampling...")
        data_for_sampling = data.to('cpu')
        train_loader = NeighborLoader(
            data_for_sampling,
            num_neighbors=[10, 5],  # 2-hop neighbors
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # For laptop compatibility
        )
        use_cpu_for_sampling = True

    # Training loop with device handling
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            # Move batch to training device (may be different from sampling device)
            batch = batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            embeddings = model(batch.x, batch.edge_index)

            # Link prediction loss (unsupervised)
            # Sample positive and negative edges
            pos_edge_index = batch.edge_index
            num_nodes = batch.x.size(0)

            # Positive edge scores
            src_emb = embeddings[pos_edge_index[0]]
            dst_emb = embeddings[pos_edge_index[1]]
            pos_scores = (src_emb * dst_emb).sum(dim=1)

            # Negative sampling
            neg_dst = torch.randint(0, num_nodes, (pos_edge_index.size(1),), device=device)
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

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Generate final embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)

    return embeddings.cpu()


def save_embeddings_to_csv(
    embeddings: torch.Tensor,
    node_metadata: Dict[int, Dict[str, Any]],
    output_csv: str = "data/06_models/embeddings/gnn_embeddings.csv"
) -> Dict[str, int]:
    """
    Save GNN embeddings to CSV file.

    Args:
        embeddings: Node embeddings tensor
        node_metadata: Node metadata dictionary
        output_csv: Path to output CSV file

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
    edges_csv: str = "data/01_raw/primekg/nodes.csv",
    output_csv: str = "data/06_models/embeddings/gnn_embeddings.csv",
    embedding_dim: int = 512,
    hidden_dim: int = 256,
    num_layers: int = 2,
    num_epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 0.01,
    limit_nodes: int = None,
    device: str = None,
    include_node_types: list = None
) -> Dict[str, Any]:
    """
    Complete pipeline: Load graph from CSV, train GNN, save embeddings to CSV.

    Args:
        edges_csv: Path to edges CSV file (PrimeKG nodes.csv = kg.csv)
        output_csv: Path to save embeddings CSV
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
    
    # Use MPS-compatible simple training for Apple Silicon
    try:
        from .gnn_simple import train_gnn_embeddings_simple
        use_simple = True
        print("   Using MPS-compatible full-batch training")
    except ImportError:
        use_simple = False
        print("   Using original neighbor sampling training")
    
    if use_simple:
        embeddings = train_gnn_embeddings_simple(
            data=data,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device
        )
    else:
        embeddings = train_gnn_embeddings(
            data=data,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            use_neighbor_sampling=False  # Disable to avoid dependency issues
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

    return {
        'num_nodes': data.num_nodes,
        'num_edges': data.num_edges,
        'embedding_dim': embedding_dim,
        'num_epochs': num_epochs,
        'device': device,
        **save_stats
    }
