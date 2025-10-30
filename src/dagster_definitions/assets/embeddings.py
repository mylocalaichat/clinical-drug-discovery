"""
Dagster assets for GNN-based graph embeddings pipeline.

This module implements GNN embeddings using PyTorch:
1. Load graph structure from CSV files
2. Train GNN embeddings using PyTorch Geometric
3. Save embeddings to CSV for downstream tasks

No Memgraph dependency - uses CSV files directly.
"""

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from dagster import AssetExecutionContext, asset

from clinical_drug_discovery.lib.gnn_embeddings import generate_gnn_embeddings


# =====================================================================
# ASSET 1: GNN Embeddings (replaces Node2Vec)
# =====================================================================

@asset(group_name="embeddings", compute_kind="ml")
def gnn_embeddings(
    context: AssetExecutionContext,
    primekg_download_status: Dict,  # Ensure CSV files are downloaded
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
    context.log.info("Training GNN embeddings from CSV files...")
    context.log.info(f"CSV files downloaded: {primekg_download_status.get('downloaded_files', [])}")
    context.log.info("Node filtering: Excluding 'cellular_component' and 'exposure' (3.9% of nodes)")

    # Define file paths
    edges_csv = "data/01_raw/primekg/nodes.csv"  # This is kg.csv (edge list)
    output_csv = "data/06_models/embeddings/gnn_embeddings.csv"

    # GNN hyperparameters - memory optimized for laptop/MPS
    embedding_params = {
        "edges_csv": edges_csv,
        "output_csv": output_csv,
        "embedding_dim": 512,  # Reduced from 512 for memory efficiency
        "hidden_dim": 256,     # Reduced from 256 for memory efficiency
        "num_layers": 2,
        "num_epochs": 50,      # Reduced for faster training on laptop
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
    """
    context.log.info("Loading GNN embeddings from CSV...")
    context.log.info(f"GNN training generated {gnn_embeddings['num_nodes']} embeddings")

    # Get embeddings CSV path from gnn_embeddings output
    embeddings_csv = gnn_embeddings['output_file']
    context.log.info(f"Reading from: {embeddings_csv}")

    # Load embeddings CSV
    embeddings_df = pd.read_csv(embeddings_csv)

    context.log.info(f"Loaded {len(embeddings_df)} embeddings from CSV")

    # Parse embedding column (stored as string representation of list)
    import ast
    embeddings_data = []
    for _, row in embeddings_df.iterrows():
        embedding = np.array(ast.literal_eval(row['embedding']))
        embeddings_data.append({
            'node_id': str(row['node_id']),
            'node_name': row['node_name'],
            'node_type': row['node_type'],
            'embedding': embedding
        })

    context.log.info(f"Parsed {len(embeddings_data)} embeddings")

    # Create DataFrame
    embeddings_df = pd.DataFrame(embeddings_data)

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
# ASSET 4: Visualization Report (Dagster UI Display)
# =====================================================================

@asset(group_name="embeddings", compute_kind="report")
def embedding_visualization_report(
    context: AssetExecutionContext,
    embedding_visualizations: Dict[str, Any],
) -> str:
    """
    Generate a markdown report with embedding visualizations for Dagster UI.
    
    This creates a summary report that displays in the Dagster UI with
    key statistics and links to interactive visualizations.
    """
    context.log.info("Generating visualization report for Dagster UI...")
    
    # Extract data from visualization results
    viz_data = embedding_visualizations
    
    if viz_data.get('status') == 'skipped':
        return "# Embeddings Visualization Report\n\n⚠️ **Visualizations skipped**: Missing dependencies (scikit-learn, plotly)"
    
    # Create markdown report
    report = f"""# 🧠 GNN Embeddings Visualization Report

## 📊 Dataset Overview
- **Total Nodes**: {viz_data['total_nodes']:,}
- **Embedding Dimension**: {viz_data['embedding_dimension']}
- **Node Types**: {len(viz_data['node_type_stats'])}

## 🎯 PCA Analysis
- **2D Explained Variance**: {sum(viz_data['pca_explained_variance'][:2]):.1%}
- **3D Explained Variance**: {sum(viz_data['pca_explained_variance'][:3]):.1%}

### Principal Components:
- **PC1**: {viz_data['pca_explained_variance'][0]:.1%} of variance
- **PC2**: {viz_data['pca_explained_variance'][1]:.1%} of variance  
- **PC3**: {viz_data['pca_explained_variance'][2]:.1%} of variance

## 📈 Node Type Statistics

| Node Type | Count | Avg Norm | Std Norm |
|-----------|-------|----------|----------|
"""
    
    # Add node type statistics table
    for node_type, stats in viz_data['node_type_stats'].items():
        report += f"| {node_type} | {stats['count']:,} | {stats['mean_norm']:.3f} | {stats['std_norm']:.3f} |\n"
    
    report += f"""
## 🖼️ Interactive Visualizations

### 2D PCA Projection
**File**: `{viz_data['visualization_files']['pca_2d']}`

### 3D PCA Projection  
**File**: `{viz_data['visualization_files']['pca_3d']}`

## 💡 Usage Instructions

1. **View Plots**: Open the HTML files in your browser
2. **Interactive Features**: 
   - Hover over points to see node details
   - Use plotly controls to zoom, pan, and rotate (3D)
   - Toggle node types on/off in the legend
3. **Analysis**: Look for clustering patterns by node type

## 🔍 Key Insights

- **Dimensionality**: {viz_data['embedding_dimension']}-dimensional embeddings reduced to 3D for visualization
- **Variance Capture**: First 3 components explain {sum(viz_data['pca_explained_variance'][:3]):.1%} of the variance
- **Node Distribution**: {len(viz_data['node_type_stats'])} different node types represented

---
*Generated by Dagster GNN Embeddings Pipeline*
"""

    # Save report to file
    report_path = Path("data/06_models/embeddings/visualization_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    context.add_output_metadata({
        "report_path": str(report_path),
        "report_length": len(report),
        "sections": 6,
        "node_types_analyzed": len(viz_data['node_type_stats']),
        "visualization_files": 2,
        "markdown_preview": report[:500] + "..." if len(report) > 500 else report
    })
    
    context.log.info(f"📄 Visualization report generated: {report_path}")
    context.log.info("🎉 Report will display in Dagster asset view")
    
    return report
