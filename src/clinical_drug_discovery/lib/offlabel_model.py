"""
Off-Label Drug Discovery: R-GCN Model Architecture

This module implements:
- Step 6: R-GCN architecture with heterogeneous graph convolutions
- Link prediction head for drug-disease pairs
- Node feature initialization with learnable embeddings
"""

import logging
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv
from torch_geometric.nn import Linear

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeterogeneousRGCN(nn.Module):
    """
    Relational Graph Convolutional Network (R-GCN) for heterogeneous graphs.

    Architecture:
    - Learnable node embeddings (128-dim per node)
    - 2 layers of heterogeneous graph convolutions
    - Separate weight matrices per edge type
    - Message passing aggregation by edge type
    """

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        num_nodes_dict: Dict[str, int],
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize R-GCN model.

        Args:
            node_types: List of node type names
            edge_types: List of (source_type, edge_type, target_type) tuples
            num_nodes_dict: Dictionary mapping node type to number of nodes
            embedding_dim: Dimension of node embeddings
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            dropout: Dropout probability
        """
        super().__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.num_nodes_dict = num_nodes_dict
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Initialize learnable node embeddings for each node type
        self.embeddings = nn.ModuleDict({
            node_type: nn.Embedding(num_nodes, embedding_dim)
            for node_type, num_nodes in num_nodes_dict.items()
        })

        # Initialize embeddings from normal distribution
        for node_type in node_types:
            nn.init.normal_(self.embeddings[node_type].weight, mean=0.0, std=0.01)

        # Build heterogeneous graph convolution layers
        self.convs = nn.ModuleList()

        for layer in range(num_layers):
            conv_dict = {}

            for src_type, edge_type, dst_type in edge_types:
                # Each edge type gets its own graph convolution
                if layer == 0:
                    conv_dict[(src_type, edge_type, dst_type)] = SAGEConv(
                        embedding_dim, hidden_dim, aggr='mean'
                    )
                else:
                    conv_dict[(src_type, edge_type, dst_type)] = SAGEConv(
                        hidden_dim, hidden_dim, aggr='mean'
                    )

            # Wrap in HeteroConv for automatic message passing
            hetero_conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(hetero_conv)

        logger.info(f"Initialized R-GCN with {num_layers} layers, {embedding_dim}-dim embeddings")
        logger.info(f"  Node types: {len(node_types)}")
        logger.info(f"  Edge types: {len(edge_types)}")

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"  Total parameters: {total_params:,}")

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict) -> Dict[str, torch.Tensor]:
        """
        Forward pass through R-GCN.

        Args:
            x_dict: Dictionary of node features (node_type -> features tensor)
            edge_index_dict: Dictionary of edge indices (edge_type -> edge_index tensor)

        Returns:
            Dictionary of updated node embeddings (node_type -> embeddings tensor)
        """
        # Apply heterogeneous graph convolutions
        for layer_idx, conv in enumerate(self.convs):
            # Apply convolution
            x_dict = conv(x_dict, edge_index_dict)

            # Apply activation and dropout (except last layer)
            if layer_idx < self.num_layers - 1:
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=self.dropout, training=self.training)
                         for key, x in x_dict.items()}

        return x_dict

    def get_initial_embeddings(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Get initial learnable embeddings for all nodes.

        Args:
            data: HeteroData object

        Returns:
            Dictionary of initial embeddings (node_type -> embeddings tensor)
        """
        x_dict = {}

        for node_type in self.node_types:
            if node_type in data.node_types:
                num_nodes = data[node_type].num_nodes
                # Create indices [0, 1, 2, ..., num_nodes-1]
                indices = torch.arange(num_nodes, device=self.embeddings[node_type].weight.device)
                x_dict[node_type] = self.embeddings[node_type](indices)

        return x_dict


class LinkPredictionHead(nn.Module):
    """
    MLP-based link prediction head for drug-disease pairs.

    Architecture:
    - Input: Concatenate [drug_embedding, disease_embedding] (256-dim)
    - Hidden: Linear(256 -> 128) + ReLU
    - Output: Linear(128 -> 1) + Sigmoid
    - Returns: Probability score [0, 1]
    """

    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 128):
        """
        Initialize link prediction head.

        Args:
            embedding_dim: Dimension of node embeddings
            hidden_dim: Hidden dimension for MLP
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # MLP layers
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        logger.info(f"Initialized LinkPredictionHead: {embedding_dim*2} -> {hidden_dim} -> 1")

    def forward(self, drug_embeddings: torch.Tensor, disease_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict link probability between drug-disease pairs.

        Args:
            drug_embeddings: Tensor of shape (batch_size, embedding_dim)
            disease_embeddings: Tensor of shape (batch_size, embedding_dim)

        Returns:
            Tensor of shape (batch_size,) with probabilities [0, 1]
        """
        # Concatenate drug and disease embeddings
        x = torch.cat([drug_embeddings, disease_embeddings], dim=1)

        # Hidden layer with ReLU
        x = F.relu(self.fc1(x))

        # Output layer with sigmoid
        x = torch.sigmoid(self.fc2(x))

        return x.squeeze(-1)


class OffLabelRGCN(nn.Module):
    """
    Complete R-GCN model for off-label drug discovery.

    Combines:
    - Heterogeneous R-GCN encoder
    - Link prediction head
    """

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        num_nodes_dict: Dict[str, int],
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize complete R-GCN model.

        Args:
            node_types: List of node type names
            edge_types: List of (source_type, edge_type, target_type) tuples
            num_nodes_dict: Dictionary mapping node type to number of nodes
            embedding_dim: Dimension of node embeddings
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            dropout: Dropout probability
        """
        super().__init__()

        self.encoder = HeterogeneousRGCN(
            node_types=node_types,
            edge_types=edge_types,
            num_nodes_dict=num_nodes_dict,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.link_predictor = LinkPredictionHead(
            embedding_dim=hidden_dim,  # Use hidden_dim from encoder
            hidden_dim=hidden_dim
        )

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

    def forward(
        self,
        data: HeteroData,
        drug_indices: torch.Tensor,
        disease_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for link prediction.

        Args:
            data: HeteroData object with graph structure
            drug_indices: Tensor of drug node indices (batch_size,)
            disease_indices: Tensor of disease node indices (batch_size,)

        Returns:
            Tensor of predicted probabilities (batch_size,)
        """
        # Get initial embeddings
        x_dict = self.encoder.get_initial_embeddings(data)

        # Apply R-GCN layers
        edge_index_dict = data.collect('edge_index')
        x_dict = self.encoder(x_dict, edge_index_dict)

        # Extract drug and disease embeddings for the batch
        drug_embeddings = x_dict['drug'][drug_indices]
        disease_embeddings = x_dict['disease'][disease_indices]

        # Predict link probability
        predictions = self.link_predictor(drug_embeddings, disease_embeddings)

        return predictions

    def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Encode all nodes to get final embeddings.

        Args:
            data: HeteroData object

        Returns:
            Dictionary of node embeddings (node_type -> embeddings tensor)
        """
        # Get initial embeddings
        x_dict = self.encoder.get_initial_embeddings(data)

        # Apply R-GCN layers
        edge_index_dict = data.collect('edge_index')
        x_dict = self.encoder(x_dict, edge_index_dict)

        return x_dict

    def predict_links(
        self,
        data: HeteroData,
        drug_indices: torch.Tensor,
        disease_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict link probabilities (inference mode).

        Args:
            data: HeteroData object
            drug_indices: Tensor of drug node indices
            disease_indices: Tensor of disease node indices

        Returns:
            Tensor of predicted probabilities
        """
        self.eval()
        with torch.no_grad():
            return self.forward(data, drug_indices, disease_indices)


def initialize_model(
    data: HeteroData,
    embedding_dim: int = 128,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    device: Optional[torch.device] = None
) -> OffLabelRGCN:
    """
    Initialize R-GCN model from HeteroData object.

    Args:
        data: HeteroData object with graph structure
        embedding_dim: Dimension of node embeddings
        hidden_dim: Hidden dimension for GNN layers
        num_layers: Number of GNN layers
        dropout: Dropout probability
        device: Device to place model on

    Returns:
        Initialized OffLabelRGCN model
    """
    # Extract node types and counts
    node_types = list(data.node_types)
    num_nodes_dict = {node_type: data[node_type].num_nodes for node_type in node_types}

    # Extract edge types
    edge_types = list(data.edge_types)

    # Initialize model
    model = OffLabelRGCN(
        node_types=node_types,
        edge_types=edge_types,
        num_nodes_dict=num_nodes_dict,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )

    # Move to device
    if device is not None:
        model = model.to(device)
        logger.info(f"Model moved to device: {device}")

    return model


if __name__ == "__main__":
    # Example usage
    from torch_geometric.data import HeteroData

    # Create dummy heterogeneous graph
    data = HeteroData()

    # Add nodes
    data['drug'].num_nodes = 100
    data['disease'].num_nodes = 200
    data['protein'].num_nodes = 500

    # Add edges
    data['drug', 'drug_protein', 'protein'].edge_index = torch.randint(0, 100, (2, 1000))
    data['protein', 'disease_protein', 'disease'].edge_index = torch.randint(0, 200, (2, 2000))
    data['drug', 'indication', 'disease'].edge_index = torch.randint(0, 100, (2, 500))

    # Add reverse edges
    data['protein', 'drug_protein_reverse', 'drug'].edge_index = data['drug', 'drug_protein', 'protein'].edge_index[[1, 0]]
    data['disease', 'disease_protein_reverse', 'protein'].edge_index = data['protein', 'disease_protein', 'disease'].edge_index[[1, 0]]
    data['disease', 'indication_reverse', 'drug'].edge_index = data['drug', 'indication', 'disease'].edge_index[[1, 0]]

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = initialize_model(data, device=device)

    # Test forward pass
    batch_size = 32
    drug_indices = torch.randint(0, 100, (batch_size,), device=device)
    disease_indices = torch.randint(0, 200, (batch_size,), device=device)

    predictions = model(data.to(device), drug_indices, disease_indices)
    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info(f"Predictions: {predictions[:5]}")
