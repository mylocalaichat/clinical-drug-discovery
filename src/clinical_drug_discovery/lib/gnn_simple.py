"""
Fixed GNN embeddings with full-batch training for MPS compatibility.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


class GraphSAGEEmbedding(nn.Module):
    """GraphSAGE model for generating node embeddings."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        # Import here to avoid issues if torch_geometric.nn is not available
        from torch_geometric.nn import SAGEConv

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


class GraphAutoEncoder(nn.Module):
    """
    Multi-task Graph Autoencoder for mechanism-based drug repurposing.

    Learns embeddings that capture:
    1. Node semantics (drug, disease, gene, pathway types)
    2. Graph structure (connectivity patterns)

    Key insight: Diseases sharing genes/pathways will have similar neighborhoods
    in the graph, leading to similar embeddings - even without direct drug connections.

    Example:
        Drug A → Gene X → Pathway P → Disease X (known)
        Disease Y → also involves Gene X/Pathway P (shared mechanism)
        → Drug A embedding will be close to Disease Y embedding (repurposing opportunity)
    """

    def __init__(self, in_channels: int, hidden_channels: int, embedding_dim: int, num_layers: int = 3):
        super().__init__()

        # Encoder: Multi-layer GNN to capture multi-hop paths
        # 3 layers captures: Drug → Gene → Pathway → Disease
        self.encoder = GraphSAGEEmbedding(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=embedding_dim,
            num_layers=num_layers
        )

        # Task 1: Reconstruct node types (semantic decoder)
        self.feature_decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, in_channels)
        )

    def encode(self, x, edge_index):
        """Encode node features into embeddings."""
        return self.encoder(x, edge_index)

    def decode_features(self, embeddings):
        """Task 1: Reconstruct node types from embeddings."""
        return self.feature_decoder(embeddings)

    def decode_edges(self, embeddings, edge_sample):
        """
        Task 2: Predict edge existence from embeddings.

        Uses dot product similarity - nodes with similar embeddings are likely connected.
        This encourages: drugs targeting same genes → similar embeddings
                        diseases sharing pathways → similar embeddings
        """
        src = embeddings[edge_sample[0]]
        dst = embeddings[edge_sample[1]]
        return torch.sum(src * dst, dim=1)

    def forward(self, x, edge_index, edge_sample=None):
        """
        Multi-task forward pass.

        Args:
            x: Node features (one-hot types)
            edge_index: Full graph edges
            edge_sample: Sampled edges for link prediction (optional)

        Returns:
            embeddings: Learned representations
            reconstructed: Reconstructed node types
            edge_scores: Predicted edge probabilities (if edge_sample provided)
        """
        embeddings = self.encode(x, edge_index)
        reconstructed = self.decode_features(embeddings)

        edge_scores = None
        if edge_sample is not None:
            edge_scores = self.decode_edges(embeddings, edge_sample)

        return embeddings, reconstructed, edge_scores


def train_gnn_embeddings_simple(
    data: Data,
    embedding_dim: int = 512,
    hidden_dim: int = 256,
    num_layers: int = 3,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    device: str = None,
    memory_efficient: bool = False,
    max_nodes_mps: int = 5000,  # Conservative limit for MPS full-batch training
    alpha: float = 0.3,  # Weight for feature reconstruction loss
    beta: float = 0.7    # Weight for link prediction loss
) -> torch.Tensor:
    """
    Train GNN model using multi-task autoencoder for mechanism-based drug repurposing.

    Multi-task learning:
    - Task 1 (α=0.3): Reconstruct node types (semantic information)
    - Task 2 (β=0.7): Predict graph edges (structural/mechanistic patterns)

    This learns embeddings where:
    - Diseases with similar gene/pathway involvement are close together
    - Drugs targeting similar mechanisms are grouped
    - Enables finding repurposing opportunities via shared mechanisms

    Args:
        data: PyG Data object
        embedding_dim: Output embedding dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of GNN layers (3 captures Drug→Gene→Pathway→Disease)
        num_epochs: Training epochs
        learning_rate: Learning rate
        device: 'cuda', 'mps', or 'cpu'
        memory_efficient: Enable memory optimizations
        max_nodes_mps: Max nodes for MPS device (fallback to CPU if exceeded)
        alpha: Weight for feature reconstruction loss (0.3 = keep type info)
        beta: Weight for link prediction loss (0.7 = emphasize structure)

    Returns:
        Node embeddings tensor
    """
    # Set up MPS fallback
    if torch.backends.mps.is_available():
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Memory-aware device selection
    if device is None:
        if torch.backends.mps.is_available() and data.num_nodes <= max_nodes_mps:
            device = 'mps'
            print(f"🍎 Using Apple M1 Neural Engine (MPS) - {data.num_nodes} nodes")
        elif torch.backends.mps.is_available() and data.num_nodes > max_nodes_mps:
            device = 'cpu'
            print(f"💻 Using CPU (MPS memory limit: {data.num_nodes} > {max_nodes_mps} nodes)")
        elif torch.cuda.is_available():
            device = 'cuda'
            print("🔥 Using NVIDIA CUDA GPU")
        else:
            device = 'cpu'
            print("💻 Using CPU")
    else:
        # Override device selection if graph is too large for MPS
        if device == 'mps' and data.num_nodes > max_nodes_mps:
            print(f"⚠️  Graph too large for MPS ({data.num_nodes} > {max_nodes_mps}), using CPU")
            device = 'cpu'
        print(f"Training on device: {device}")

    print(f"Device: {device}")
    print("📋 Using multi-task autoencoder (feature + link prediction)")
    print(f"📐 Dimensions: embedding={embedding_dim}, hidden={hidden_dim}, layers={num_layers}")
    print(f"⚖️  Loss weights: α={alpha} (features), β={beta} (structure)")

    # Move data to device
    data = data.to(device)

    # Initialize autoencoder model
    model = GraphAutoEncoder(
        in_channels=data.x.size(1),
        hidden_channels=hidden_dim,
        embedding_dim=embedding_dim,
        num_layers=num_layers
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Memory-efficient training
    model.train()
    print("Starting multi-task training...")

    # Use gradient checkpointing for memory efficiency
    if memory_efficient:
        torch.backends.cudnn.benchmark = False  # Reduce memory fragmentation

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        try:
            # Sample edges for link prediction (memory efficient)
            num_edges = data.edge_index.size(1)
            max_edge_samples = min(1000 if device == 'mps' else 2000, num_edges)
            edge_sample_idx = torch.randperm(num_edges, device=device)[:max_edge_samples]
            edge_sample = data.edge_index[:, edge_sample_idx]

            # Sample negative edges (non-existent connections)
            neg_dst = torch.randint(0, data.num_nodes, (max_edge_samples,), device=device)
            neg_edge_sample = torch.stack([edge_sample[0], neg_dst], dim=0)

            # Forward pass
            embeddings, reconstructed, pos_scores = model(data.x, data.edge_index, edge_sample)
            _, _, neg_scores = model(data.x, data.edge_index, neg_edge_sample)

            # Task 1: Feature reconstruction loss
            feature_loss = F.mse_loss(reconstructed, data.x)

            # Task 2: Link prediction loss (positive + negative)
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores)
            )
            link_loss = pos_loss + neg_loss

            # Combined multi-task loss
            loss = alpha * feature_loss + beta * link_loss

            loss.backward()

            # Gradient clipping for stability
            if memory_efficient:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Aggressive memory cleanup for MPS (prevents fragmentation)
            if device == 'mps':
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()

            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}, Total: {loss.item():.4f}, Feature: {feature_loss.item():.4f}, Link: {link_loss.item():.4f}")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️  Memory error at epoch {epoch}: {e}")
                print("🔄 Switching to CPU for remaining epochs...")
                # Move to CPU and continue
                data = data.to('cpu')
                model = model.to('cpu')
                device = 'cpu'
                continue
            else:
                raise e
    
    # Generate final embeddings using encoder only
    model.eval()
    with torch.no_grad():
        final_embeddings = model.encode(data.x, data.edge_index)

    # Final cleanup
    if device == 'mps' and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()

    print(f"✓ Training completed. Generated embeddings shape: {final_embeddings.shape}")
    return final_embeddings