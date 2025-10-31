"""
Dagster assets for GNN-based graph embeddings pipeline.

This module implements GNN embeddings using PyTorch:
1. Load graph structure from CSV files
2. Train GNN embeddings using PyTorch Geometric
3. Save embeddings to CSV for downstream tasks

No Memgraph dependency - uses CSV files directly.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from dagster import AssetExecutionContext, asset

from clinical_drug_discovery.lib.gnn_embeddings import generate_gnn_embeddings
from clinical_drug_discovery.lib.gnn_hgt import generate_hgt_embeddings
from clinical_drug_discovery.lib.gnn_hgt_batched import generate_hgt_embeddings_batched


# =====================================================================
# ASSET 1: GNN Embeddings (replaces Node2Vec)
# =====================================================================

@asset(group_name="embeddings", compute_kind="ml")
def gnn_embeddings(
    context: AssetExecutionContext,
    download_data: Dict,  # Ensure CSV files are downloaded
) -> Dict[str, Any]:
    """
    Train GNN embeddings on the knowledge graph using PyTorch.

    Replaces Node2Vec with Graph Neural Network (GraphSAGE) for better
    representation learning. Uses batch training for laptop compatibility.

    Data Source: Loads directly from PrimeKG CSV files (no Memgraph needed).

    Node Type Filtering:
    - Includes: drug, disease, gene/protein, effect/phenotype, pathway,
                biological_process, molecular_function, anatomy (8 types)
    - Excludes: cellular_component (too granular, cell biology focus)
                exposure (environmental toxins, not therapeutics)
    - Result: ~124,381 nodes (96.1% of graph) | Excludes ~4,994 nodes (3.9%)
    """
    # Clean up existing embeddings file before training
    output_csv = "data/06_models/embeddings/gnn_embeddings.csv"
    output_path = Path(output_csv)

    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        output_path.unlink()
        context.log.info(f"Deleted existing embeddings file: {output_csv} ({file_size_mb:.1f} MB)")
    else:
        context.log.info("No existing embeddings file to clean up")

    context.log.info("Training GNN embeddings from CSV files...")
    context.log.info(f"CSV files downloaded: {download_data.get('downloaded_files', [])}")
    context.log.info("Node filtering: Excluding 'cellular_component' and 'exposure' (3.9% of nodes)")

    # Get file paths from download_data output - REQUIRED, no defaults
    edges_csv = download_data['edges_file']  # Will fail if not present
    context.log.info(f"Using edges file: {edges_csv}")

    # GNN hyperparameters - optimized for mechanism-based drug repurposing
    embedding_params = {
        "edges_csv": edges_csv,
        "output_csv": output_csv,
        "embedding_dim": 512,
        "hidden_dim": 256,
        "num_layers": 3,       # 3 layers to capture Drug→Gene→Pathway→Disease paths
        "num_epochs": 10,
        "batch_size": 128,
        "learning_rate": 0.01,
        "device": None,        # Auto-detect (cuda/mps/cpu)
        # "limit_nodes": 15000,  # Removed - train on full dataset
        "include_node_types": [
            'drug', 'disease', 'gene/protein', 'pathway', 'biological_process'
        ]  # Focus on most important node types
    }

    context.log.info(f"GNN parameters: {embedding_params}")

    # Generate GNN embeddings from CSV
    stats = generate_gnn_embeddings(**embedding_params)

    context.add_output_metadata({
        "num_nodes": stats['num_nodes'],
        "num_edges": stats['num_edges'],
        "embedding_dim": stats['embedding_dim'],
        "num_epochs": stats['num_epochs'],
        "training_device": stats.get('device', 'cpu'),
        "saved_nodes": stats['saved_nodes'],
        "output_file": stats['output_file'],
        "data_source": "CSV (no Memgraph)",
        "node_filtering": "Excludes cellular_component and exposure (3.9% of graph)",
        "included_node_types": "8 types: drug, disease, gene/protein, effect/phenotype, pathway, biological_process, molecular_function, anatomy",
    })

    context.log.info(f"GNN training complete: {stats['num_nodes']} nodes embedded")
    context.log.info(f"Embeddings saved to: {stats['output_file']}")

    return stats


# =====================================================================
# ASSET 1B: HGT Embeddings with Contrastive Learning (for off-label discovery)
# =====================================================================

@asset(group_name="embeddings", compute_kind="ml")
def hgt_embeddings(
    context: AssetExecutionContext,
    download_data: Dict,  # Ensure CSV files are downloaded
) -> Dict[str, Any]:
    """
    Train HGT (Heterogeneous Graph Transformer) embeddings with contrastive learning.

    Optimized for off-label drug discovery:
    - Handles all 19 edge types from PrimeKG explicitly
    - Contrastive learning: diseases with similar pathways/genes cluster together
    - Better for finding diseases that share mechanisms even if not directly connected

    Key improvements over standard GNN:
    - Edge-type aware: distinguishes 'indication' from 'contraindication'
    - Attention mechanism: learns which relationships matter most
    - Contrastive loss: explicitly optimizes for disease similarity

    Data Source: Loads directly from PrimeKG CSV files (no Memgraph needed).
    """
    # Clean up existing embeddings file before training
    output_csv = "data/06_models/embeddings/hgt_embeddings.csv"
    output_path = Path(output_csv)

    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        output_path.unlink()
        context.log.info(f"Deleted existing HGT embeddings file: {output_csv} ({file_size_mb:.1f} MB)")
    else:
        context.log.info("No existing HGT embeddings file to clean up")

    context.log.info("Training HGT embeddings with contrastive learning...")
    context.log.info(f"CSV files downloaded: {download_data.get('downloaded_files', [])}")
    context.log.info("Node filtering: Excluding 'cellular_component' and 'exposure' (3.9% of nodes)")

    # Get file paths from download_data output - REQUIRED, no defaults
    edges_csv = download_data['edges_file']  # Will fail if not present
    context.log.info(f"Using edges file: {edges_csv}")

    # HGT hyperparameters - optimized for off-label drug discovery
    #
    # OPTION 1: Use batched training for MPS (faster, full 512 dims)
    # OPTION 2: Use CPU training (slower but more memory)
    # Uncomment your preferred option below:

    # DISABLED: Batched MPS has issues - use CPU for now
    use_batched_mps = False  # CPU is slower but reliable

    # use_batched_mps = True  # Set to False to use CPU instead
    # if use_batched_mps and not torch.backends.mps.is_available():
    #     context.log.warning("MPS not available, falling back to CPU")
    #     use_batched_mps = False

    embedding_params = {
        "edges_csv": edges_csv,
        "output_csv": output_csv,
        "embedding_dim": 512,
        "hidden_dim": 256,
        "num_layers": 2,       # 2 layers for HGT (more powerful than GraphSAGE)
        "num_heads": 8,        # Multi-head attention
        "num_epochs": 10,      # Quick test - increase to 100+ for production
        "learning_rate": 0.001,
        "device": None if use_batched_mps else "cpu",  # None = auto-detect MPS
        "contrastive_weight": 0.0,  # DISABLED: Contrastive loss has scaling issues - fix later
        "similarity_threshold": 0.1,  # Jaccard similarity threshold for positive pairs
        "edge_sample_size": 5000,  # Increased from 1000 to see more edges per epoch
        "include_node_types": [
            'drug', 'disease', 'gene/protein', 'pathway', 'biological_process'
        ]  # Focus on most important node types
    }

    # Add batching params if using MPS
    if use_batched_mps:
        embedding_params.update({
            "node_batch_size": 1024,     # Process 1024 seed nodes per batch
            "accumulation_steps": 8,     # Accumulate gradients over 8 batches
            "num_neighbors": [10, 10],   # Sample 10 neighbors per layer
        })

    context.log.info(f"HGT parameters: {embedding_params}")
    if use_batched_mps:
        context.log.info("Using BATCHED training for MPS (full 512 dims)")
    else:
        context.log.info("Using CPU training (full 512 dims)")
    context.log.info("Contrastive learning: diseases sharing genes/pathways = positive pairs")

    # Generate HGT embeddings from CSV (use batched version if MPS)
    if use_batched_mps:
        stats = generate_hgt_embeddings_batched(**embedding_params)
    else:
        stats = generate_hgt_embeddings(**embedding_params)

    context.add_output_metadata({
        "embedding_dim": stats['embedding_dim'],
        "num_epochs": stats['num_epochs'],
        "training_device": stats.get('device', 'cpu'),
        "total_nodes": stats['total_nodes'],
        "output_file": stats['output_file'],
        "model_type": "HGT (Heterogeneous Graph Transformer)",
        "optimization": "Contrastive learning for disease similarity",
        "edge_types_handled": "All 19 PrimeKG edge types explicitly modeled",
        "use_case": "Off-label drug discovery via pathway similarity",
    })

    context.log.info(f"HGT training complete: {stats['total_nodes']} nodes embedded")
    context.log.info(f"Embeddings saved to: {stats['output_file']}")

    return stats


# =====================================================================
# ASSET 2: Flattened Embeddings (for ML models)
# =====================================================================

@asset(group_name="embeddings", compute_kind="transform")
def flattened_embeddings(
    context: AssetExecutionContext,
    gnn_embeddings: Dict[str, Any],
) -> pd.DataFrame:
    """
    Load GNN embeddings from CSV and flatten for ML models.

    Reads embeddings from CSV file and converts them to a flattened DataFrame
    format suitable for XGBoost and other ML models.

    Uses GraphSAGE embeddings (baseline).
    """
    # Clean up existing flattened embeddings file before creating new one
    output_path = Path("data/06_models/embeddings/gnn_flattened_embeddings.csv")

    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        output_path.unlink()
        context.log.info(f"Deleted existing flattened embeddings: {output_path} ({file_size_mb:.1f} MB)")
    else:
        context.log.info("No existing flattened embeddings to clean up")

    context.log.info("Loading GNN embeddings from CSV...")
    context.log.info(f"GNN training generated {gnn_embeddings['num_nodes']} embeddings")

    # Get embeddings CSV path from gnn_embeddings output
    embeddings_csv = gnn_embeddings['output_file']
    context.log.info(f"Reading from: {embeddings_csv}")

    # Load embeddings CSV
    embeddings_df = pd.read_csv(embeddings_csv)

    context.log.info(f"Loaded {len(embeddings_df)} embeddings from CSV")

    # Debug: Check node type counts from CSV
    if 'node_type' in embeddings_df.columns:
        csv_node_counts = embeddings_df['node_type'].value_counts().to_dict()
        context.log.info(f"Node types from CSV: {csv_node_counts}")

    # Parse embedding column (stored as string representation of list)
    import ast

    context.log.info("Parsing embeddings using vectorized operations...")

    # Vectorized parsing - much faster than iterrows()
    embeddings_df['embedding'] = embeddings_df['embedding'].apply(ast.literal_eval).apply(np.array)
    embeddings_df['node_id'] = embeddings_df['node_id'].astype(str)

    # Keep only required columns
    embeddings_df = embeddings_df[['node_id', 'node_name', 'node_type', 'embedding']]

    context.log.info(f"Parsed {len(embeddings_df)} embeddings")

    # Debug: Check for duplicate node IDs
    duplicate_count = embeddings_df['node_id'].duplicated().sum()
    if duplicate_count > 0:
        context.log.warning(f"⚠️  Found {duplicate_count} duplicate node IDs in embeddings CSV!")
        duplicate_ids = embeddings_df[embeddings_df['node_id'].duplicated(keep=False)]['node_id'].unique()
        context.log.warning(f"⚠️  Number of unique IDs with duplicates: {len(duplicate_ids)}")
        context.log.warning(f"⚠️  Example duplicate IDs: {list(duplicate_ids[:5])}")

        # Remove duplicates before proceeding
        context.log.info("Removing duplicates, keeping first occurrence...")
        embeddings_df = embeddings_df.drop_duplicates(subset='node_id', keep='first')
        context.log.info(f"After deduplication: {len(embeddings_df)} unique nodes")

    # Flatten embeddings into separate columns
    context.log.info("Flattening embeddings into columns...")

    # Extract embedding dimensions
    embedding_matrix = np.vstack(embeddings_df['embedding'].values)
    embedding_dim = embedding_matrix.shape[1]

    # Create column names
    embedding_cols = [f"emb_{i}" for i in range(embedding_dim)]

    # Create flattened DataFrame
    flattened_df = pd.DataFrame(
        embedding_matrix,
        columns=embedding_cols
    )

    # Add metadata columns
    flattened_df['node_id'] = embeddings_df['node_id'].values
    flattened_df['node_name'] = embeddings_df['node_name'].values
    flattened_df['node_type'] = embeddings_df['node_type'].values

    # Reorder columns (metadata first, then embeddings)
    cols = ['node_id', 'node_name', 'node_type'] + embedding_cols
    flattened_df = flattened_df[cols]

    # Save flattened embeddings for inspection
    output_path = "data/06_models/embeddings/gnn_flattened_embeddings.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save a sample (full file would be too large)
    sample_df = flattened_df.head(1000)
    sample_df.to_csv(output_path, index=False)

    context.add_output_metadata({
        "num_rows": len(flattened_df),
        "num_columns": len(flattened_df.columns),
        "embedding_dimensions": embedding_dim,
        "sample_saved_to": output_path,
        "memory_usage_mb": float(round(flattened_df.memory_usage(deep=True).sum() / (1024 * 1024), 2)),
        "node_types": embeddings_df['node_type'].value_counts().to_dict(),
    })

    context.log.info(f"Flattened {len(flattened_df)} embeddings with {embedding_dim} dimensions")

    return flattened_df


# =====================================================================
# ASSET 2B: HGT Flattened Embeddings (for ML models)
# =====================================================================

@asset(group_name="embeddings", compute_kind="transform")
def hgt_flattened_embeddings(
    context: AssetExecutionContext,
    hgt_embeddings: Dict[str, Any],
) -> pd.DataFrame:
    """
    Load HGT embeddings from CSV and flatten for ML models.

    Reads embeddings from CSV file and converts them to a flattened DataFrame
    format suitable for XGBoost and other ML models.

    Uses HGT embeddings optimized for off-label drug discovery.
    """
    # Clean up existing flattened embeddings file before creating new one
    output_path = Path("data/06_models/embeddings/hgt_flattened_embeddings.csv")

    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        output_path.unlink()
        context.log.info(f"Deleted existing HGT flattened embeddings: {output_path} ({file_size_mb:.1f} MB)")
    else:
        context.log.info("No existing HGT flattened embeddings to clean up")

    context.log.info("Loading HGT embeddings from CSV...")
    context.log.info(f"HGT training generated {hgt_embeddings['total_nodes']} embeddings")

    # Get embeddings CSV path from hgt_embeddings output
    embeddings_csv = hgt_embeddings['output_file']
    context.log.info(f"Reading from: {embeddings_csv}")

    # Load embeddings CSV
    embeddings_df = pd.read_csv(embeddings_csv)

    context.log.info(f"Loaded {len(embeddings_df)} embeddings from CSV")

    # Debug: Check node type counts from CSV
    if 'node_type' in embeddings_df.columns:
        csv_node_counts = embeddings_df['node_type'].value_counts().to_dict()
        context.log.info(f"Node types from CSV: {csv_node_counts}")

    # Parse embedding column (stored as string representation of list)
    import ast

    context.log.info("Parsing embeddings using vectorized operations...")

    # Vectorized parsing - much faster than iterrows()
    embeddings_df['embedding'] = embeddings_df['embedding'].apply(ast.literal_eval).apply(np.array)
    embeddings_df['node_id'] = embeddings_df['node_id'].astype(str)

    # Keep only required columns
    embeddings_df = embeddings_df[['node_id', 'node_name', 'node_type', 'embedding']]

    context.log.info(f"Parsed {len(embeddings_df)} embeddings")

    # Debug: Check for duplicate node IDs
    duplicate_count = embeddings_df['node_id'].duplicated().sum()
    if duplicate_count > 0:
        context.log.warning(f"⚠️  Found {duplicate_count} duplicate node IDs in embeddings CSV!")
        duplicate_ids = embeddings_df[embeddings_df['node_id'].duplicated(keep=False)]['node_id'].unique()
        context.log.warning(f"⚠️  Number of unique IDs with duplicates: {len(duplicate_ids)}")
        context.log.warning(f"⚠️  Example duplicate IDs: {list(duplicate_ids[:5])}")

        # Remove duplicates before proceeding
        context.log.info("Removing duplicates, keeping first occurrence...")
        embeddings_df = embeddings_df.drop_duplicates(subset='node_id', keep='first')
        context.log.info(f"After deduplication: {len(embeddings_df)} unique nodes")

    # Flatten embeddings into separate columns
    context.log.info("Flattening embeddings into columns...")

    # Extract embedding dimensions
    embedding_matrix = np.vstack(embeddings_df['embedding'].values)
    embedding_dim = embedding_matrix.shape[1]

    # Create column names
    embedding_cols = [f"emb_{i}" for i in range(embedding_dim)]

    # Create flattened DataFrame
    flattened_df = pd.DataFrame(
        embedding_matrix,
        columns=embedding_cols
    )

    # Add metadata columns
    flattened_df['node_id'] = embeddings_df['node_id'].values
    flattened_df['node_name'] = embeddings_df['node_name'].values
    flattened_df['node_type'] = embeddings_df['node_type'].values

    # Reorder columns (metadata first, then embeddings)
    cols = ['node_id', 'node_name', 'node_type'] + embedding_cols
    flattened_df = flattened_df[cols]

    # Save flattened embeddings for inspection
    output_path = "data/06_models/embeddings/hgt_flattened_embeddings.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save the complete dataset (not just a sample)
    context.log.info(f"Saving complete flattened embeddings dataset ({len(flattened_df):,} rows)...")
    flattened_df.to_csv(output_path, index=False)
    
    # Calculate file size
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    context.log.info(f"Saved complete dataset to {output_path} ({file_size_mb:.1f} MB)")

    context.add_output_metadata({
        "num_rows": len(flattened_df),
        "num_columns": len(flattened_df.columns),
        "embedding_dimensions": embedding_dim,
        "sample_saved_to": output_path,
        "memory_usage_mb": float(round(flattened_df.memory_usage(deep=True).sum() / (1024 * 1024), 2)),
        "node_types": embeddings_df['node_type'].value_counts().to_dict(),
        "model_type": "HGT with contrastive learning",
    })

    context.log.info(f"Flattened {len(flattened_df)} HGT embeddings with {embedding_dim} dimensions")

    return flattened_df


# =====================================================================
# ASSET 3: Embedding Visualizations (optional)
# =====================================================================

@asset(group_name="embeddings", compute_kind="visualization")
def embedding_visualizations(
    context: AssetExecutionContext,
    flattened_embeddings: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Create visualizations of the GNN embeddings using PCA.

    Generates 2D and 3D scatter plots showing node embeddings colored by type.
    """
    # Clean up existing visualization files before creating new ones
    viz_dir = Path("data/06_models/embeddings/visualizations")

    if viz_dir.exists():
        deleted_files = []
        for html_file in viz_dir.glob("*.html"):
            file_size_mb = html_file.stat().st_size / (1024 * 1024)
            html_file.unlink()
            deleted_files.append(f"{html_file.name} ({file_size_mb:.1f} MB)")
            context.log.info(f"Deleted existing visualization: {html_file.name} ({file_size_mb:.1f} MB)")

        if deleted_files:
            context.log.info(f"Cleaned up {len(deleted_files)} visualization file(s)")
        else:
            context.log.info("No existing visualizations to clean up")
    else:
        context.log.info("Visualization directory does not exist yet, will be created")

    context.log.info("Creating GNN embedding visualizations...")

    try:
        from sklearn.decomposition import PCA
        import plotly.express as px
    except ImportError as e:
        context.log.warning(f"Visualization libraries not available: {e}")
        context.log.info("Install with: pip install scikit-learn plotly")
        return {"status": "skipped", "reason": "missing_dependencies"}

    # Extract embedding columns
    embedding_cols = [col for col in flattened_embeddings.columns if col.startswith('emb_')]
    embeddings_matrix = flattened_embeddings[embedding_cols].values

    context.log.info(f"Visualizing {len(embeddings_matrix)} embeddings of dimension {embeddings_matrix.shape[1]}")

    # Create visualization directory
    viz_dir = Path("data/06_models/embeddings/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)

    # PCA Analysis
    context.log.info("Computing PCA...")
    pca = PCA(n_components=min(3, embeddings_matrix.shape[1]))
    pca_embeddings = pca.fit_transform(embeddings_matrix)

    # Create PCA DataFrame
    pca_df = pd.DataFrame({
        'node_id': flattened_embeddings['node_id'].values,
        'node_name': flattened_embeddings['node_name'].values,
        'node_type': flattened_embeddings['node_type'].values,
        'PC1': pca_embeddings[:, 0],
        'PC2': pca_embeddings[:, 1],
        'PC3': pca_embeddings[:, 2] if pca_embeddings.shape[1] > 2 else 0
    })

    # Create interactive Plotly visualizations
    context.log.info("Creating interactive plots...")

    # PCA 2D scatter plot
    fig_2d = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color='node_type',
        hover_data=['node_name', 'node_id'],
        title='GNN Embeddings - PCA 2D Projection',
        width=800,
        height=600
    )

    # Save interactive plot
    plot_2d_path = viz_dir / "gnn_embeddings_pca_2d.html"
    fig_2d.write_html(str(plot_2d_path))

    # PCA 3D scatter plot
    fig_3d = px.scatter_3d(
        pca_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='node_type',
        hover_data=['node_name', 'node_id'],
        title='GNN Embeddings - PCA 3D Projection',
        width=800,
        height=600
    )

    # Save 3D plot
    plot_3d_path = viz_dir / "gnn_embeddings_pca_3d.html"
    fig_3d.write_html(str(plot_3d_path))

    # Compute statistics by node type
    type_stats = {}
    for node_type in pca_df['node_type'].unique():
        type_mask = pca_df['node_type'] == node_type
        type_embeddings = embeddings_matrix[type_mask]

        if len(type_embeddings) > 0:
            type_stats[node_type] = {
                'count': len(type_embeddings),
                'mean_norm': float(np.mean([np.linalg.norm(emb) for emb in type_embeddings])),
                'std_norm': float(np.std([np.linalg.norm(emb) for emb in type_embeddings])),
            }

    summary_data = {
        'pca_explained_variance': [float(x) for x in pca.explained_variance_ratio_.tolist()],
        'node_type_stats': type_stats,
        'total_nodes': len(pca_df),
        'embedding_dimension': embeddings_matrix.shape[1],
        'visualization_files': {
            'pca_2d': str(plot_2d_path),
            'pca_3d': str(plot_3d_path)
        }
    }

    context.add_output_metadata({
        "🖼️_2D_Plot_File": f"file://{plot_2d_path.absolute()}",
        "🖼️_3D_Plot_File": f"file://{plot_3d_path.absolute()}",
        "📊_nodes_visualized": len(pca_df),
        "📐_embedding_dimension": embeddings_matrix.shape[1],
        "2d_plot_file": str(plot_2d_path),
        "3d_plot_file": str(plot_3d_path)
    })

    # Simple output - just the essential file locations
    context.log.info(f"📊 Visualized {len(pca_df):,} nodes with {embeddings_matrix.shape[1]}D embeddings")
    context.log.info(f"🖼️  2D Plot: file://{plot_2d_path.absolute()}")
    context.log.info(f"🖼️  3D Plot: file://{plot_3d_path.absolute()}")

    return summary_data


# =====================================================================
# ASSET 3B: HGT Embedding Visualizations (optional)
# =====================================================================

@asset(group_name="embeddings", compute_kind="visualization")
def hgt_embedding_visualizations(
    context: AssetExecutionContext,
    hgt_flattened_embeddings: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Create visualizations of the HGT embeddings using PCA.

    Generates 2D and 3D scatter plots showing node embeddings colored by type.
    Uses HGT embeddings optimized for off-label drug discovery.
    """
    # Clean up existing visualization files before creating new ones
    viz_dir = Path("data/06_models/embeddings/visualizations")

    if viz_dir.exists():
        deleted_files = []
        for html_file in viz_dir.glob("hgt_*.html"):
            file_size_mb = html_file.stat().st_size / (1024 * 1024)
            html_file.unlink()
            deleted_files.append(f"{html_file.name} ({file_size_mb:.1f} MB)")
            context.log.info(f"Deleted existing HGT visualization: {html_file.name} ({file_size_mb:.1f} MB)")

        if deleted_files:
            context.log.info(f"Cleaned up {len(deleted_files)} HGT visualization file(s)")
        else:
            context.log.info("No existing HGT visualizations to clean up")
    else:
        context.log.info("Visualization directory does not exist yet, will be created")

    context.log.info("Creating HGT embedding visualizations...")

    try:
        from sklearn.decomposition import PCA
        import plotly.express as px
    except ImportError as e:
        context.log.warning(f"Visualization libraries not available: {e}")
        context.log.info("Install with: pip install scikit-learn plotly")
        return {"status": "skipped", "reason": "missing_dependencies"}

    # Extract embedding columns
    embedding_cols = [col for col in hgt_flattened_embeddings.columns if col.startswith('emb_')]
    embeddings_matrix = hgt_flattened_embeddings[embedding_cols].values

    context.log.info(f"Visualizing {len(embeddings_matrix)} HGT embeddings of dimension {embeddings_matrix.shape[1]}")

    # Create visualization directory
    viz_dir = Path("data/06_models/embeddings/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)

    # PCA Analysis
    context.log.info("Computing PCA...")
    pca = PCA(n_components=min(3, embeddings_matrix.shape[1]))
    pca_embeddings = pca.fit_transform(embeddings_matrix)

    # Create PCA DataFrame
    pca_df = pd.DataFrame({
        'node_id': hgt_flattened_embeddings['node_id'].values,
        'node_name': hgt_flattened_embeddings['node_name'].values,
        'node_type': hgt_flattened_embeddings['node_type'].values,
        'PC1': pca_embeddings[:, 0],
        'PC2': pca_embeddings[:, 1],
        'PC3': pca_embeddings[:, 2] if pca_embeddings.shape[1] > 2 else 0
    })

    # Create interactive Plotly visualizations
    context.log.info("Creating interactive plots...")

    # PCA 2D scatter plot
    fig_2d = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color='node_type',
        hover_data=['node_name', 'node_id'],
        title='HGT Embeddings - PCA 2D Projection (with Contrastive Learning)',
        width=800,
        height=600
    )

    # Save interactive plot
    plot_2d_path = viz_dir / "hgt_embeddings_pca_2d.html"
    fig_2d.write_html(str(plot_2d_path))

    # PCA 3D scatter plot
    fig_3d = px.scatter_3d(
        pca_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='node_type',
        hover_data=['node_name', 'node_id'],
        title='HGT Embeddings - PCA 3D Projection (with Contrastive Learning)',
        width=800,
        height=600
    )

    # Save 3D plot
    plot_3d_path = viz_dir / "hgt_embeddings_pca_3d.html"
    fig_3d.write_html(str(plot_3d_path))

    # Compute statistics by node type
    type_stats = {}
    for node_type in pca_df['node_type'].unique():
        type_mask = pca_df['node_type'] == node_type
        type_embeddings = embeddings_matrix[type_mask]

        if len(type_embeddings) > 0:
            type_stats[node_type] = {
                'count': len(type_embeddings),
                'mean_norm': float(np.mean([np.linalg.norm(emb) for emb in type_embeddings])),
                'std_norm': float(np.std([np.linalg.norm(emb) for emb in type_embeddings])),
            }

    summary_data = {
        'pca_explained_variance': [float(x) for x in pca.explained_variance_ratio_.tolist()],
        'node_type_stats': type_stats,
        'total_nodes': len(pca_df),
        'embedding_dimension': embeddings_matrix.shape[1],
        'model_type': 'HGT with contrastive learning',
        'visualization_files': {
            'pca_2d': str(plot_2d_path),
            'pca_3d': str(plot_3d_path)
        }
    }

    context.add_output_metadata({
        "🖼️_2D_Plot_File": f"file://{plot_2d_path.absolute()}",
        "🖼️_3D_Plot_File": f"file://{plot_3d_path.absolute()}",
        "📊_nodes_visualized": len(pca_df),
        "📐_embedding_dimension": embeddings_matrix.shape[1],
        "2d_plot_file": str(plot_2d_path),
        "3d_plot_file": str(plot_3d_path),
        "model_type": "HGT with contrastive learning",
    })

    # Simple output - just the essential file locations
    context.log.info(f"📊 Visualized {len(pca_df):,} HGT nodes with {embeddings_matrix.shape[1]}D embeddings")
    context.log.info(f"🖼️  2D Plot: file://{plot_2d_path.absolute()}")
    context.log.info(f"🖼️  3D Plot: file://{plot_3d_path.absolute()}")

    return summary_data
