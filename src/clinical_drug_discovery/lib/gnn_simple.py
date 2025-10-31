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


def train_gnn_embeddings_simple(
    data: Data,
    embedding_dim: int = 512,
    hidden_dim: int = 256,
    num_layers: int = 2,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    device: str = None,
    memory_efficient: bool = False,
    max_nodes_mps: int = 5000  # Conservative limit for MPS full-batch training
) -> torch.Tensor:
    """
    Train GNN model using full-batch training (no neighbor sampling).
    This avoids dependency issues and MPS compatibility problems.

    Args:
        data: PyG Data object
        embedding_dim: Output embedding dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of GNN layers
        num_epochs: Training epochs
        learning_rate: Learning rate
        device: 'cuda', 'mps', or 'cpu'
        memory_efficient: Enable memory optimizations
        max_nodes_mps: Max nodes for MPS device (fallback to CPU if exceeded).
                       Default 50k is conservative for full-batch training.
                       Larger graphs (>50k nodes) automatically use CPU.

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
    print("📋 Using full-batch training (node-level)")
    print(f"📐 Dimensions: embedding={embedding_dim}, hidden={hidden_dim}")

    # Move data to device
    data = data.to(device)

    # Initialize model with memory-efficient settings
    model = GraphSAGEEmbedding(
        in_channels=data.x.size(1),
        hidden_channels=hidden_dim,
        out_channels=embedding_dim,
        num_layers=min(num_layers, 2)  # Limit layers for memory
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Memory-efficient training
    model.train()
    print("Starting memory-efficient training...")
    
    # Use gradient checkpointing for memory efficiency
    if memory_efficient:
        torch.backends.cudnn.benchmark = False  # Reduce memory fragmentation
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        try:
            # Forward pass - full batch for all nodes
            embeddings = model(data.x, data.edge_index)

            # Edge sampling for loss calculation
            if data.edge_index.size(1) > 0:
                # Sample edges for training (larger batches = better gradient estimates)
                max_samples = min(1000 if device == 'mps' else 2000, data.edge_index.size(1))
                edge_indices = torch.randperm(data.edge_index.size(1), device=device)[:max_samples]
                sampled_edges = data.edge_index[:, edge_indices]
                
                # Positive samples
                src_embeddings = embeddings[sampled_edges[0]]
                dst_embeddings = embeddings[sampled_edges[1]]
                pos_scores = torch.sum(src_embeddings * dst_embeddings, dim=1)
                
                # Negative samples
                neg_dst = torch.randint(0, data.num_nodes, (max_samples,), device=device)
                neg_embeddings = embeddings[neg_dst]
                neg_scores = torch.sum(src_embeddings * neg_embeddings, dim=1)
                
                # Binary classification loss
                pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
                neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
                loss = pos_loss + neg_loss
                
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
                    print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}, Batch: {max_samples} edges")
            else:
                print("⚠️  No edges found in graph")
                break
                
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
    
    # Generate final embeddings
    model.eval()
    with torch.no_grad():
        final_embeddings = model(data.x, data.edge_index)

    # Final cleanup
    if device == 'mps' and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()

    print(f"✓ Training completed. Generated embeddings shape: {final_embeddings.shape}")
    return final_embeddings