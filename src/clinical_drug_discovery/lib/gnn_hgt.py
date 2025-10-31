"""
HGT (Heterogeneous Graph Transformer) embeddings with contrastive learning.

Optimized for off-label drug discovery: learns embeddings where diseases with
similar pathways/genes cluster together, even if not directly connected.

Key features:
- Handles all 19 edge types from PrimeKG explicitly
- Attention mechanism learns which relationships matter most
- Contrastive learning: diseases sharing genes/pathways = positive pairs
"""

import os
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from collections import defaultdict
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear
from tqdm import tqdm


def ensure_divisible_by_heads(embedding_dim: int, num_heads: int) -> int:
    """
    Ensure embedding dimension is divisible by number of heads.
    
    Args:
        embedding_dim: Desired embedding dimension
        num_heads: Number of attention heads
        
    Returns:
        Adjusted embedding dimension that's divisible by num_heads
    """
    if embedding_dim % num_heads == 0:
        return embedding_dim
    
    # Round up to next multiple of num_heads
    adjusted_dim = ((embedding_dim // num_heads) + 1) * num_heads
    print(f"📐 Adjusted embedding_dim from {embedding_dim} to {adjusted_dim} (divisible by {num_heads} heads)")
    return adjusted_dim


# Set up MPS fallback for Apple Silicon compatibility
if torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


class HGTEmbedding(nn.Module):
    """Heterogeneous Graph Transformer for multi-relational knowledge graphs."""

    def __init__(
        self,
        node_types: Dict[str, int],  # node_type -> num_nodes
        edge_types: List[Tuple[str, str, str]],  # (src_type, edge_type, dst_type)
        hidden_channels: int = 256,
        out_channels: int = 512,
        num_heads: int = 8,
        num_layers: int = 2
    ):
        super().__init__()
        
        # CRITICAL VALIDATION: Ensure both hidden_channels and out_channels are divisible by num_heads
        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"'hidden_channels' ({hidden_channels}) must be divisible by 'num_heads' ({num_heads}). "
                f"Suggested fix: use hidden_channels={((hidden_channels // num_heads) + 1) * num_heads} "
                f"or change num_heads to a divisor of {hidden_channels}."
            )
            
        if out_channels % num_heads != 0:
            raise ValueError(
                f"'out_channels' ({out_channels}) must be divisible by 'num_heads' ({num_heads}). "
                f"Suggested fix: use out_channels={((out_channels // num_heads) + 1) * num_heads} "
                f"or change num_heads to a divisor of {out_channels}."
            )
        
        self.node_types = list(node_types.keys())
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        # Linear projections for each node type to hidden dimension
        self.node_lin = nn.ModuleDict()
        for node_type in self.node_types:
            # Input is one-hot encoded node type (1 dim per node type)
            self.node_lin[node_type] = Linear(-1, hidden_channels)

        # HGT convolutional layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=(self.node_types, edge_types),
                heads=num_heads
            )
            self.convs.append(conv)

        # Output projections for each node type
        self.out_lin = nn.ModuleDict()
        for node_type in self.node_types:
            self.out_lin[node_type] = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # Project all node types to hidden dimension
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.node_lin[node_type](x)

        # Apply HGT layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            # ReLU activation (applied to all node types)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Project to output dimension
        out_dict = {}
        for node_type, x in x_dict.items():
            out_dict[node_type] = self.out_lin[node_type](x)

        return out_dict


def load_hetero_graph_from_csv(
    edges_csv: str,
    limit_nodes: int = None,
    include_node_types: list = None,
    chunk_size: int = 500000,  # Process edges in chunks
) -> Tuple[HeteroData, Dict[str, Dict[int, Dict[str, Any]]]]:
    """
    Load heterogeneous graph from CSV, preserving edge types.
    Optimized version with chunked processing and vectorized operations.

    Args:
        edges_csv: Path to kg.csv with edge triplets
        limit_nodes: Limit number of nodes (for testing)
        include_node_types: Node types to include (default excludes cellular_component, exposure)
        chunk_size: Size of chunks for edge processing

    Returns:
        - HeteroData object with node features and typed edges
        - Node metadata dictionary: {node_type: {node_id: {idx, name, type}}}
    """
    from .gnn_optimization import fast_node_extraction

    # Default node type filtering
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

    print(f"Loading heterogeneous graph from: {edges_csv}")
    print(f"Including node types: {include_node_types}")
    print(f"Using chunk size: {chunk_size:,}")

    # Load edges
    print("\nReading edges CSV...")
    start_time = time.time()
    edges_df = pd.read_csv(edges_csv, low_memory=False)
    print(f"Loaded {len(edges_df):,} edges from CSV in {time.time() - start_time:.1f}s")

    # Extract nodes
    print("\nExtracting nodes...")
    start_time = time.time()
    nodes_df, node_id_to_idx = fast_node_extraction(
        edges_df, include_node_types, limit_nodes
    )
    print(f"Node extraction completed in {time.time() - start_time:.1f}s")

    # Log node type counts
    node_type_counts = nodes_df['type'].value_counts()
    print(f"Total: {len(nodes_df):,} nodes across {len(node_type_counts)} node types:")
    for node_type, count in node_type_counts.items():
        print(f"  - {node_type}: {count:,}")

    valid_node_ids = set(node_id_to_idx.keys())

    # Create HeteroData object
    data = HeteroData()

    # OPTIMIZATION: Use vectorized operations for node type grouping
    print("\nGrouping nodes by type...")
    start_time = time.time()
    
    # Create node type -> local index mapping more efficiently
    node_type_to_nodes = {}
    global_to_local_idx = {}
    
    for node_type in nodes_df['type'].unique():
        type_mask = nodes_df['type'] == node_type
        type_nodes = nodes_df[type_mask]['id'].values
        node_type_to_nodes[node_type] = sorted(type_nodes)
        
        # Create local index mapping
        global_to_local_idx[node_type] = {
            node_id: local_idx
            for local_idx, node_id in enumerate(node_type_to_nodes[node_type])
        }
    
    print(f"Node grouping completed in {time.time() - start_time:.1f}s")

    # Create one-hot features for each node type (simple: 1-dim feature = 1.0)
    print("\nCreating node features...")
    start_time = time.time()
    for node_type, node_ids in node_type_to_nodes.items():
        num_nodes = len(node_ids)
        # Simple feature: all ones (will be projected by model)
        data[node_type].x = torch.ones((num_nodes, 1), dtype=torch.float)
    print(f"Feature creation completed in {time.time() - start_time:.1f}s")

    # OPTIMIZATION: Build edge indices with chunked processing and vectorized operations
    print("\nBuilding edge indices by edge type...")
    start_time = time.time()
    
    # Filter edges to valid nodes ONCE at the beginning
    valid_edges_mask = (
        edges_df['x_id'].isin(valid_node_ids) &
        edges_df['y_id'].isin(valid_node_ids)
    )
    valid_edges_df = edges_df[valid_edges_mask].copy()
    print(f"Filtered to {len(valid_edges_df):,} valid edges")

    # Group by relation for processing
    edge_type_counts = {}
    relation_groups = valid_edges_df.groupby('relation')
    
    for relation, group_df in relation_groups:
        # Get unique type combinations for this relation
        type_combinations = group_df[['x_type', 'y_type']].drop_duplicates()
        
        for _, row in type_combinations.iterrows():
            src_type, dst_type = row['x_type'], row['y_type']
            
            # Skip if node types were filtered out
            if src_type not in global_to_local_idx or dst_type not in global_to_local_idx:
                continue
            
            # Filter edges for this specific type combination
            type_edges = group_df[
                (group_df['x_type'] == src_type) &
                (group_df['y_type'] == dst_type)
            ]
            
            if len(type_edges) == 0:
                continue
            
            # OPTIMIZATION: Vectorized index mapping
            src_ids = type_edges['x_id'].values
            dst_ids = type_edges['y_id'].values
            
            # Map to local indices using vectorized operations
            src_local_map = global_to_local_idx[src_type]
            dst_local_map = global_to_local_idx[dst_type]
            
            # Use list comprehension for faster mapping
            src_indices = [src_local_map[sid] for sid in src_ids if sid in src_local_map]
            dst_indices = [dst_local_map[did] for did in dst_ids if did in dst_local_map]
            
            if len(src_indices) != len(dst_indices) or len(src_indices) == 0:
                continue
            
            # Create edge index
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
            
            # Store in HeteroData
            edge_type_key = (src_type, relation, dst_type)
            data[edge_type_key].edge_index = edge_index
            edge_type_counts[edge_type_key] = len(src_indices)

    print(f"Edge index building completed in {time.time() - start_time:.1f}s")
    print(f"Created {len(edge_type_counts)} edge types:")
    for edge_type_key, count in sorted(edge_type_counts.items(), key=lambda x: -x[1])[:20]:
        src, rel, dst = edge_type_key
        print(f"  ({src}, {rel}, {dst}): {count:,} edges")

    # OPTIMIZATION: Create metadata dictionary more efficiently
    print("\nCreating metadata...")
    start_time = time.time()
    node_metadata = {}
    
    for node_type, node_ids in node_type_to_nodes.items():
        node_metadata[node_type] = {}
        
        # Get all rows for this type at once
        type_mask = nodes_df['type'] == node_type
        type_df = nodes_df[type_mask].set_index('id')
        
        for node_id in node_ids:
            row = type_df.loc[node_id]
            local_idx = global_to_local_idx[node_type][node_id]
            node_metadata[node_type][node_id] = {
                'idx': local_idx,
                'name': row['name'],
                'type': node_type,
                'global_id': node_id
            }
    
    print(f"Metadata creation completed in {time.time() - start_time:.1f}s")

    print("\n✓ Heterogeneous graph loaded:")
    for node_type in data.node_types:
        print(f"  - {node_type}: {data[node_type].num_nodes:,} nodes")
    print(f"  - Total edges: {sum(edge_type_counts.values()):,}")

    return data, node_metadata


def build_disease_similarity_matrix(
    edges_df: pd.DataFrame,
    disease_ids: List[int],
    node_metadata: Dict[int, Dict[str, Any]]
) -> torch.Tensor:
    """
    Build disease similarity matrix based on shared genes.

    Similarity = number of shared genes via 'associated with' or 'expression present' edges.

    Args:
        edges_df: Full edges DataFrame
        disease_ids: List of disease node IDs
        node_metadata: Metadata for disease nodes

    Returns:
        Similarity matrix (num_diseases x num_diseases)
    """
    print("\nBuilding disease similarity matrix based on shared genes...")

    # Get edges connecting diseases to genes
    # NOTE: In PrimeKG, the relation is 'disease_protein', not 'associated with'
    disease_gene_edges = edges_df[
        (edges_df['x_type'] == 'disease') &
        (edges_df['y_type'] == 'gene/protein') &
        (edges_df['relation'] == 'disease_protein')
    ]

    # Build disease -> gene set mapping
    disease_to_genes = defaultdict(set)
    for _, row in disease_gene_edges.iterrows():
        disease_id = row['x_id']
        gene_id = row['y_id']
        if disease_id in disease_ids:
            disease_to_genes[disease_id].add(gene_id)

    print(f"Found {len(disease_to_genes)} diseases with gene associations")

    # Build similarity matrix
    num_diseases = len(disease_ids)
    disease_id_to_idx = {disease_id: idx for idx, disease_id in enumerate(disease_ids)}

    similarity_matrix = torch.zeros((num_diseases, num_diseases), dtype=torch.float)

    for i, disease_i in enumerate(disease_ids):
        genes_i = disease_to_genes.get(disease_i, set())
        if len(genes_i) == 0:
            continue

        for j, disease_j in enumerate(disease_ids):
            if i >= j:  # Only compute upper triangle
                continue

            genes_j = disease_to_genes.get(disease_j, set())
            if len(genes_j) == 0:
                continue

            # Jaccard similarity
            intersection = len(genes_i & genes_j)
            union = len(genes_i | genes_j)

            if union > 0:
                similarity = intersection / union
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  # Symmetric

    # Set diagonal to 1.0
    similarity_matrix.fill_diagonal_(1.0)

    print(f"Similarity matrix: {similarity_matrix.shape}")
    print(f"Non-zero similarities: {(similarity_matrix > 0).sum().item()}")
    print(f"Mean similarity (excluding diagonal): {similarity_matrix[similarity_matrix < 1.0].mean().item():.4f}")

    return similarity_matrix


def contrastive_loss(
    embeddings: torch.Tensor,
    similarity_matrix: torch.Tensor,
    temperature: float = 0.07,
    threshold: float = 0.1
) -> torch.Tensor:
    """
    Contrastive loss for disease embeddings.

    Positive pairs: diseases with similarity > threshold
    Negative pairs: diseases with similarity <= threshold

    Args:
        embeddings: Disease embeddings (num_diseases, embedding_dim)
        similarity_matrix: Precomputed disease similarity (num_diseases, num_diseases)
        temperature: Temperature for softmax
        threshold: Similarity threshold for positive pairs

    Returns:
        Contrastive loss value
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Compute cosine similarity between all pairs
    cosine_sim = torch.mm(embeddings, embeddings.t()) / temperature

    # Create positive and negative masks
    positive_mask = (similarity_matrix > threshold).float()
    negative_mask = (similarity_matrix <= threshold).float()

    # Remove self-similarity (diagonal)
    positive_mask.fill_diagonal_(0)
    negative_mask.fill_diagonal_(0)

    # InfoNCE loss
    # For each disease, pull positive pairs closer, push negative pairs apart
    num_diseases = embeddings.size(0)
    total_loss = 0
    num_valid = 0

    for i in range(num_diseases):
        # Get positive and negative samples for disease i
        pos_mask_i = positive_mask[i] > 0
        neg_mask_i = negative_mask[i] > 0

        if pos_mask_i.sum() == 0:  # No positive pairs for this disease
            continue

        # Positive similarity scores
        pos_scores = cosine_sim[i][pos_mask_i]

        # Negative similarity scores
        neg_scores = cosine_sim[i][neg_mask_i]

        if neg_scores.size(0) == 0:  # No negative pairs
            continue

        # Simple contrastive loss: maximize positive similarities, minimize negative
        # Positive loss: want similarity close to 1
        pos_loss = F.mse_loss(pos_scores, torch.ones_like(pos_scores))

        # Negative loss: want similarity close to 0
        neg_loss = F.mse_loss(neg_scores, torch.zeros_like(neg_scores))

        # Combined loss for this disease
        loss_i = pos_loss + neg_loss

        total_loss += loss_i
        num_valid += 1

    if num_valid == 0:
        return torch.tensor(0.0, device=embeddings.device)

    return total_loss / num_valid


def train_hgt_embeddings(
    data: HeteroData,
    node_metadata: Dict[str, Dict[int, Dict[str, Any]]],
    edges_df: pd.DataFrame,
    embedding_dim: int = 512,
    hidden_dim: int = 256,
    num_layers: int = 2,
    num_heads: int = 8,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = None,
    contrastive_weight: float = 0.5,
    similarity_threshold: float = 0.1,
    edge_sample_size: int = 5000  # Edges to sample per edge type (increased from 1000)
) -> Dict[str, torch.Tensor]:
    """
    Train HGT model with full MPS optimization and layer-wise computation.

    MPS-optimized training:
    - All operations on MPS (no CPU fallback)
    - Edge sampling per type to reduce memory
    - Layer-wise inference for final embeddings
    - Aggressive memory management
    - Contrastive learning for disease similarity

    Args:
        data: HeteroData object
        node_metadata: Node metadata dict
        edges_df: Original edges DataFrame (for computing disease similarity)
        embedding_dim: Output embedding dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of HGT layers
        num_heads: Number of attention heads
        num_epochs: Training epochs
        learning_rate: Learning rate
        device: Training device (default: auto-detect MPS)
        contrastive_weight: Weight for contrastive loss (vs link prediction)
        similarity_threshold: Threshold for positive disease pairs
        edge_sample_size: Number of edges to sample per edge type

    Returns:
        Dictionary of embeddings: {node_type: tensor}
    """
    import os
    # Force MPS fallback for unsupported ops
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    # CRITICAL FIX: Ensure both embedding_dim and hidden_dim are compatible with num_heads
    embedding_dim = ensure_divisible_by_heads(embedding_dim, num_heads)
    hidden_dim = ensure_divisible_by_heads(hidden_dim, num_heads)
    print(f"✓ Validated embedding_dim={embedding_dim} (divisible by {num_heads} heads)")
    print(f"✓ Validated hidden_dim={hidden_dim} (divisible by {num_heads} heads)")

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
    print("✓ Full-batch training with edge sampling per type")
    print("✓ Layer-wise inference for memory efficiency")

    # Memory-aware configuration
    total_nodes = sum(data[node_type].num_nodes for node_type in data.node_types)

    # Count total edges across all edge types
    total_edges = sum(data[edge_type].edge_index.size(1) for edge_type in data.edge_types)

    # Memory components (similar to GNN but for heterogeneous graph):
    # 1. Node embeddings across all types
    # 2. Edge aggregation (BOTTLENECK): edges × embedding_dim × num_layers
    # 3. Gradients: ~2x forward pass
    node_memory_gb = (total_nodes * embedding_dim * 4) / (1024**3)
    edge_memory_gb = (total_edges * embedding_dim * 4 * num_layers) / (1024**3)
    total_memory_gb = (node_memory_gb + edge_memory_gb) * 2  # × 2 for gradients

    print(f"\nMemory estimation:")
    print(f"  Nodes: {total_nodes:,} | Edges: {total_edges:,}")
    print(f"  Node memory: {node_memory_gb:.2f} GB")
    print(f"  Edge aggregation: {edge_memory_gb:.2f} GB (BOTTLENECK)")
    print(f"  Total estimated: {total_memory_gb:.2f} GB")

    # For MPS, auto-adjust if needed (with improved minimum dimension preservation)
    if device == 'mps' and total_memory_gb > 10:
        print(f"\n⚠️  WARNING: Graph too large for MPS at current dimensions")
        print(f"⚠️  Estimated memory: {total_memory_gb:.2f} GB (>10 GB)")

        # Calculate required scale factor
        target_memory_gb = 8  # Target 8 GB to be safe
        scale_factor = target_memory_gb / total_memory_gb
        new_embedding_dim = int(embedding_dim * scale_factor)
        new_hidden_dim = int(hidden_dim * scale_factor)

        # IMPROVED: Ensure minimum viable dimensions (increased from 64/32 to 384/192)
        # This prevents severe embedding collapse
        new_embedding_dim = max(new_embedding_dim, 384)  # Was 64
        new_hidden_dim = max(new_hidden_dim, 192)        # Was 32

        # CRITICAL FIX: Ensure both dimensions are divisible by num_heads
        new_embedding_dim = ensure_divisible_by_heads(new_embedding_dim, num_heads)
        new_hidden_dim = ensure_divisible_by_heads(new_hidden_dim, num_heads)

        print(f"\n🔧 Auto-adjusting dimensions to fit in memory...")
        print(f"   Old embedding_dim: {embedding_dim} → New: {new_embedding_dim}")
        print(f"   Old hidden_dim: {hidden_dim} → New: {new_hidden_dim}")
        print(f"   ⚠️  Note: Preserving minimum 384 dims to prevent embedding collapse")

        embedding_dim = new_embedding_dim
        hidden_dim = new_hidden_dim

        # Recalculate
        edge_memory_gb = (total_edges * embedding_dim * 4 * num_layers) / (1024**3)
        total_memory_gb = (node_memory_gb + edge_memory_gb) * 2
        print(f"   New estimated memory: {total_memory_gb:.2f} GB")

    # Move data to device
    data = data.to(device)

    # Get node type information
    node_types = {node_type: data[node_type].num_nodes for node_type in data.node_types}
    edge_types = data.edge_types

    print(f"\nNode types: {list(node_types.keys())}")
    print(f"Edge types: {len(edge_types)}")

    # Initialize HGT model
    model = HGTEmbedding(
        node_types=node_types,
        edge_types=edge_types,
        hidden_channels=hidden_dim,
        out_channels=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Build disease similarity matrix
    if 'disease' in node_metadata:
        disease_ids = list(node_metadata['disease'].keys())
        print(f"\nFound {len(disease_ids)} diseases for contrastive learning")

        similarity_matrix = build_disease_similarity_matrix(
            edges_df, disease_ids, node_metadata['disease']
        ).to(device)
    else:
        print("⚠️  No diseases found, skipping contrastive learning")
        similarity_matrix = None

    # Training loop
    model.train()
    print(f"\nTraining for {num_epochs} epochs...")

    # Progress bar for training
    pbar = tqdm(range(num_epochs), desc="Training HGT", unit="epoch")

    for _ in pbar:
        optimizer.zero_grad()

        # Forward pass on full graph
        out_dict = model(data.x_dict, data.edge_index_dict)

        # Loss 1: Link prediction loss (sample edges per type)
        link_loss = 0
        num_edge_types = 0

        for edge_type in edge_types:
            src_type, rel, dst_type = edge_type
            edge_index = data[edge_type].edge_index

            if edge_index.size(1) == 0:
                continue

            # Sample edges for this type
            num_edges = min(edge_index.size(1), edge_sample_size)
            perm = torch.randperm(edge_index.size(1), device=device)[:num_edges]
            sampled_edge_index = edge_index[:, perm]

            # Positive edge scores
            src_emb = out_dict[src_type][sampled_edge_index[0]]
            dst_emb = out_dict[dst_type][sampled_edge_index[1]]
            pos_scores = (src_emb * dst_emb).sum(dim=1)

            # Negative sampling
            num_nodes = data[dst_type].num_nodes
            neg_dst = torch.randint(0, num_nodes, (num_edges,), device=device)
            neg_dst_emb = out_dict[dst_type][neg_dst]
            neg_scores = (src_emb * neg_dst_emb).sum(dim=1)

            # BCE loss
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores)
            )

            link_loss += (pos_loss + neg_loss)
            num_edge_types += 1

        link_loss = link_loss / max(num_edge_types, 1)

        # Loss 2: Contrastive loss for diseases
        if similarity_matrix is not None and 'disease' in out_dict:
            disease_embeddings = out_dict['disease']
            contr_loss = contrastive_loss(
                disease_embeddings,
                similarity_matrix,
                threshold=similarity_threshold
            )

            # Combined loss
            total_loss = (1 - contrastive_weight) * link_loss + contrastive_weight * contr_loss

            # Update progress bar with both losses
            pbar.set_postfix({
                'link': f'{link_loss.item():.4f}',
                'contr': f'{contr_loss.item():.4f}',
                'total': f'{total_loss.item():.4f}'
            })
        else:
            contr_loss = torch.tensor(0.0, device=device)
            total_loss = link_loss
            pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Aggressive memory cleanup
        del out_dict, link_loss, total_loss
        if contr_loss != 0.0:
            del contr_loss

        # Clear MPS cache
        if device == 'mps' and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

    # Generate final embeddings using layer-wise computation (memory efficient)
    print("\n✓ Generating final embeddings with layer-wise computation...")
    model.eval()

    with torch.no_grad():
        # Project all node types to hidden dimension
        x_dict = {}
        node_pbar = tqdm(data.x_dict.items(), desc="Projecting nodes", unit="type")
        for node_type, x in node_pbar:
            node_pbar.set_description(f"Projecting {node_type}")
            x_dict[node_type] = model.node_lin[node_type](x)

            # Clear cache
            if device == 'mps' and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

        # Apply HGT layers one at a time
        layer_pbar = tqdm(enumerate(model.convs), total=num_layers, desc="HGT layers", unit="layer")
        for layer_idx, conv in layer_pbar:
            layer_pbar.set_description(f"HGT Layer {layer_idx + 1}/{num_layers}")
            x_dict = conv(x_dict, data.edge_index_dict)
            # ReLU activation
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

            # Clear cache after each layer
            if device == 'mps' and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

        # Project to output dimension
        out_dict = {}
        for node_type, x in x_dict.items():
            print(f"  Output projection for {node_type}...")
            out_dict[node_type] = model.out_lin[node_type](x)

            # Clear cache
            if device == 'mps' and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

        embeddings_dict = out_dict

    # Move to CPU
    embeddings_dict = {k: v.cpu() for k, v in embeddings_dict.items()}

    print("\n✓ Training complete!")
    for node_type, emb in embeddings_dict.items():
        print(f"  - {node_type}: {emb.shape}")

    return embeddings_dict


def save_hgt_embeddings_to_csv(
    embeddings_dict: Dict[str, torch.Tensor],
    node_metadata: Dict[str, Dict[int, Dict[str, Any]]],
    output_csv: str
) -> Dict[str, int]:
    """
    Save HGT embeddings to CSV file.

    Args:
        embeddings_dict: Dictionary of embeddings by node type
        node_metadata: Node metadata dictionary
        output_csv: Path to output CSV file

    Returns:
        Statistics dictionary
    """
    print(f"\nSaving HGT embeddings to: {output_csv}")

    # Create output directory
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for node_type, embeddings in embeddings_dict.items():
        embeddings_np = embeddings.numpy()

        if node_type not in node_metadata:
            print(f"⚠️  Warning: {node_type} not in node_metadata, skipping")
            continue

        for node_id, metadata in node_metadata[node_type].items():
            local_idx = metadata['idx']
            embedding = embeddings_np[local_idx]

            row = {
                'node_id': node_id,
                'node_name': metadata['name'],
                'node_type': node_type,
                'embedding': embedding.tolist()
            }
            rows.append(row)

    embeddings_df = pd.DataFrame(rows)
    embeddings_df.to_csv(output_csv, index=False)

    print(f"✓ Saved {len(embeddings_df):,} embeddings to {output_csv}")
    print(f"  File size: {output_path.stat().st_size / (1024*1024):.2f} MB")

    return {
        'total_nodes': len(rows),
        'output_file': output_csv
    }


def generate_hgt_embeddings(
    edges_csv: str,
    output_csv: str,
    embedding_dim: int = 512,
    hidden_dim: int = 256,
    num_layers: int = 2,
    num_heads: int = 8,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    limit_nodes: int = None,
    device: str = None,
    include_node_types: list = None,
    contrastive_weight: float = 0.5,
    similarity_threshold: float = 0.1,
    edge_sample_size: int = 5000
) -> Dict[str, Any]:
    """
    Complete pipeline: Load heterogeneous graph, train HGT with contrastive learning, save embeddings.

    Args:
        edges_csv: Path to kg.csv
        output_csv: Output path for embeddings
        embedding_dim: Output embedding dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of HGT layers
        num_heads: Number of attention heads
        num_epochs: Training epochs
        learning_rate: Learning rate
        limit_nodes: Limit nodes for testing
        device: Training device
        include_node_types: Node types to include
        contrastive_weight: Weight for contrastive loss (0-1)
        similarity_threshold: Jaccard similarity threshold for positive pairs
        edge_sample_size: Number of edges to sample per edge type per epoch

    Returns:
        Statistics dictionary
    """
    print("=" * 80)
    print("HGT EMBEDDINGS WITH CONTRASTIVE LEARNING")
    print("Optimized for off-label drug discovery")
    print("=" * 80)

    # CRITICAL FIX: Ensure both embedding_dim and hidden_dim are compatible with num_heads
    embedding_dim = ensure_divisible_by_heads(embedding_dim, num_heads)
    hidden_dim = ensure_divisible_by_heads(hidden_dim, num_heads)
    print(f"✓ Validated embedding_dim={embedding_dim} (divisible by {num_heads} heads)")
    print(f"✓ Validated hidden_dim={hidden_dim} (divisible by {num_heads} heads)")

    # Step 1: Load heterogeneous graph
    print("\n1. Loading heterogeneous graph...")
    data, node_metadata = load_hetero_graph_from_csv(
        edges_csv=edges_csv,
        limit_nodes=limit_nodes,
        include_node_types=include_node_types
    )

    # Load full edges DataFrame for similarity computation
    edges_df = pd.read_csv(edges_csv, low_memory=False)

    # Step 2: Train HGT
    print("\n2. Training HGT model with contrastive learning...")
    embeddings_dict = train_hgt_embeddings(
        data=data,
        node_metadata=node_metadata,
        edges_df=edges_df,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        contrastive_weight=contrastive_weight,
        similarity_threshold=similarity_threshold,
        edge_sample_size=edge_sample_size
    )

    # Step 3: Save embeddings
    print("\n3. Saving embeddings...")
    save_stats = save_hgt_embeddings_to_csv(
        embeddings_dict=embeddings_dict,
        node_metadata=node_metadata,
        output_csv=output_csv
    )

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)

    return {
        'embedding_dim': embedding_dim,
        'num_epochs': num_epochs,
        'device': device,
        **save_stats
    }
