"""
Dagster assets for graph embeddings pipeline.

This module splits the graph embeddings process into multiple assets:
1. Load graph from Neo4j
2. Train Node2Vec embeddings 
3. Save embeddings to disk
4. Create embedding DataFrame
5. Flatten embeddings for ML models
"""

import os
from pathlib import Path
from typing import Any, Dict

import networkx as nx
import numpy as np
import pandas as pd
from dagster import AssetExecutionContext, asset

from clinical_drug_discovery.lib.graph_embeddings import (
    flatten_embeddings_for_model,
    train_node2vec_embeddings,
)


def create_ascii_graph_visualization(graph: nx.Graph, nodes_df: pd.DataFrame, max_nodes: int = 10) -> str:
    """Create ASCII art visualization of the graph structure."""
    if graph.number_of_nodes() == 0:
        return "Empty graph - no nodes to visualize"
    
    # Limit to a subset for readability
    nodes_to_show = list(graph.nodes())[:max_nodes]
    
    # Create a mapping of node IDs to short labels
    node_labels = {}
    node_types = {}
    
    for node_id in nodes_to_show:
        # Get node info from DataFrame
        node_info = nodes_df[nodes_df['node_id'] == int(node_id)]
        if not node_info.empty:
            node_name = str(node_info.iloc[0]['node_name'])[:15]  # Truncate long names
            node_type = str(node_info.iloc[0]['node_type'])[:8]
            node_labels[node_id] = f"{node_name}"
            node_types[node_id] = node_type
        else:
            node_labels[node_id] = f"Node_{node_id}"
            node_types[node_id] = "unknown"
    
    # Build ASCII representation
    lines = []
    lines.append("=" * 80)
    lines.append("📊 KNOWLEDGE GRAPH STRUCTURE (ASCII)")
    lines.append("=" * 80)
    lines.append("")
    
    # Show node types legend
    type_counts = {}
    for node_type in node_types.values():
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
    
    lines.append("🏷️  NODE TYPES LEGEND:")
    for node_type, count in type_counts.items():
        symbol = {"drug": "💊", "disease": "🦠", "protein": "🧬", "gene": "🧬"}.get(node_type, "⚫")
        lines.append(f"   {symbol} {node_type}: {count} nodes")
    lines.append("")
    
    # Show connections
    lines.append("🔗 CONNECTIONS:")
    edges_shown = 0
    max_edges = 20
    
    for node_id in nodes_to_show:
        neighbors = list(graph.neighbors(node_id))
        if neighbors:
            # Get node symbol
            node_type = node_types[node_id]
            symbol = {"drug": "💊", "disease": "🦠", "protein": "🧬", "gene": "🧬"}.get(node_type, "⚫")
            
            lines.append(f"\n{symbol} {node_labels[node_id]} ({node_type}):")
            
            for neighbor in neighbors[:5]:  # Limit neighbors shown
                if neighbor in nodes_to_show:
                    neighbor_type = node_types[neighbor]
                    neighbor_symbol = {"drug": "💊", "disease": "🦠", "protein": "🧬", "gene": "🧬"}.get(neighbor_type, "⚫")
                    
                    # Get edge relation if available
                    edge_data = graph.get_edge_data(node_id, neighbor)
                    relation = edge_data.get('relation', 'relates') if edge_data else 'relates'
                    
                    lines.append(f"    ├─ [{relation}] ──→ {neighbor_symbol} {node_labels[neighbor]}")
                    edges_shown += 1
                    
                    if edges_shown >= max_edges:
                        break
            
            if len(neighbors) > 5:
                lines.append(f"    └─ ... and {len(neighbors) - 5} more connections")
        
        if edges_shown >= max_edges:
            break
    
    if graph.number_of_edges() > edges_shown:
        lines.append(f"\n... and {graph.number_of_edges() - edges_shown} more edges")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"📈 STATS: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    lines.append("=" * 80)
    
    return "\n".join(lines)


@asset(group_name="embeddings", compute_kind="database")
def random_graph_sample(
    context: AssetExecutionContext,
    primekg_edges_loaded: Dict,  # Ensure data is loaded first
) -> Dict[str, Any]:
    """Get random sample of nodes and edges for quick analysis - uses efficient queries."""
    from neo4j import GraphDatabase
    
    context.log.info("Sampling random nodes and edges from Memgraph...")
    
    # Get Memgraph connection details
    memgraph_uri = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
    memgraph_user = os.getenv("MEMGRAPH_USER", "")
    memgraph_password = os.getenv("MEMGRAPH_PASSWORD", "")
    
    # Handle authentication
    auth = None
    if memgraph_user or memgraph_password:
        auth = (memgraph_user, memgraph_password)
    
    driver = GraphDatabase.driver(memgraph_uri, auth=auth)
    
    try:
        with driver.session() as session:
            # Single query to sample 3-hop paths and collect nodes from the same result set
            context.log.info("Sampling 3-hop paths and extracting nodes from same result set...")
            unified_sample_query = """
            MATCH (a:Node)-[r1:RELATES]->(intermediate1:Node)-[r2:RELATES]->(intermediate2:Node)-[r3:RELATES]->(b:Node)
            WHERE (
                (a.node_type = 'drug' AND b.node_type = 'disease') OR 
                (a.node_type = 'disease' AND b.node_type = 'drug')
            )
              AND a.node_id <> b.node_id 
              AND a.node_id <> intermediate1.node_id 
              AND a.node_id <> intermediate2.node_id
              AND intermediate1.node_id <> b.node_id 
              AND intermediate1.node_id <> intermediate2.node_id
              AND intermediate2.node_id <> b.node_id
            WITH a, r1, intermediate1, r2, intermediate2, r3, b
            LIMIT 10
            WITH a, r1, intermediate1, r2, intermediate2, r3, b
            ORDER BY a.node_name
            LIMIT 5
            RETURN a.node_id as source_id,
                   a.node_name as source_name,
                   a.node_type as source_type,
                   a.node_source as source_source,
                   r1.relation as relation1,
                   r1.display_relation as display_relation1,
                   intermediate1.node_id as intermediate1_id,
                   intermediate1.node_name as intermediate1_name,
                   intermediate1.node_type as intermediate1_type,
                   intermediate1.node_source as intermediate1_source,
                   r2.relation as relation2,
                   r2.display_relation as display_relation2,
                   intermediate2.node_id as intermediate2_id,
                   intermediate2.node_name as intermediate2_name,
                   intermediate2.node_type as intermediate2_type,
                   intermediate2.node_source as intermediate2_source,
                   r3.relation as relation3,
                   r3.display_relation as display_relation3,
                   b.node_id as target_id,
                   b.node_name as target_name,
                   b.node_type as target_type,
                   b.node_source as target_source
            """
            
            result = session.run(unified_sample_query)
            edges_data = []
            edges_set = set()  # Use set to avoid duplicate edges
            nodes_dict = {}  # Use dict to avoid duplicates
            
            for record in result:
                # For 3-hop paths, we create three edges: a->intermediate1->intermediate2->b
                # First edge: source to intermediate1
                edge1_key = (record['source_id'], record['intermediate1_id'], record['relation1'])
                if edge1_key not in edges_set:
                    edges_set.add(edge1_key)
                    edges_data.append({
                        'source_id': record['source_id'],
                        'source_name': record['source_name'],
                        'source_type': record['source_type'],
                        'relation': record['relation1'],
                        'display_relation': record['display_relation1'],
                        'target_id': record['intermediate1_id'],
                        'target_name': record['intermediate1_name'],
                        'target_type': record['intermediate1_type']
                    })
                
                # Second edge: intermediate1 to intermediate2
                edge2_key = (record['intermediate1_id'], record['intermediate2_id'], record['relation2'])
                if edge2_key not in edges_set:
                    edges_set.add(edge2_key)
                    edges_data.append({
                        'source_id': record['intermediate1_id'],
                        'source_name': record['intermediate1_name'],
                        'source_type': record['intermediate1_type'],
                        'relation': record['relation2'],
                        'display_relation': record['display_relation2'],
                        'target_id': record['intermediate2_id'],
                        'target_name': record['intermediate2_name'],
                        'target_type': record['intermediate2_type']
                    })
                
                # Third edge: intermediate2 to target
                edge3_key = (record['intermediate2_id'], record['target_id'], record['relation3'])
                if edge3_key not in edges_set:
                    edges_set.add(edge3_key)
                    edges_data.append({
                        'source_id': record['intermediate2_id'],
                        'source_name': record['intermediate2_name'],
                        'source_type': record['intermediate2_type'],
                        'relation': record['relation3'],
                        'display_relation': record['display_relation3'],
                        'target_id': record['target_id'],
                        'target_name': record['target_name'],
                        'target_type': record['target_type']
                    })
                
                # Collect unique nodes from source, both intermediates, and target
                nodes_dict[record['source_id']] = {
                    'node_id': record['source_id'],
                    'node_name': record['source_name'],
                    'node_type': record['source_type'],
                    'node_source': record['source_source']
                }
                nodes_dict[record['intermediate1_id']] = {
                    'node_id': record['intermediate1_id'],
                    'node_name': record['intermediate1_name'],
                    'node_type': record['intermediate1_type'],
                    'node_source': record['intermediate1_source']
                }
                nodes_dict[record['intermediate2_id']] = {
                    'node_id': record['intermediate2_id'],
                    'node_name': record['intermediate2_name'],
                    'node_type': record['intermediate2_type'],
                    'node_source': record['intermediate2_source']
                }
                nodes_dict[record['target_id']] = {
                    'node_id': record['target_id'],
                    'node_name': record['target_name'],
                    'node_type': record['target_type'],
                    'node_source': record['target_source']
                }
            
            # Convert nodes dict to list
            nodes_data = list(nodes_dict.values())
            
        # Create DataFrames
        nodes_df = pd.DataFrame(nodes_data)
        edges_df = pd.DataFrame(edges_data)
        
        context.log.info(f"Sampled {len(nodes_df)} random nodes and {len(edges_df)} random edges")
        
        # Log some sample data for verification
        if len(nodes_df) > 0:
            context.log.info(f"Sample node types: {nodes_df['node_type'].value_counts().head().to_dict()}")
        if len(edges_df) > 0:
            context.log.info(f"Sample edge relations: {edges_df['relation'].value_counts().head().to_dict()}")
        
        # Create a temporary NetworkX graph for visualization
        temp_graph = nx.Graph()
        for _, node in nodes_df.iterrows():
            node_id = str(node["node_id"])
            temp_graph.add_node(node_id, **{
                "node_name": node["node_name"],
                "node_type": node["node_type"],
                "node_source": node["node_source"]
            })
        
        for _, edge in edges_df.iterrows():
            source_id = str(edge["source_id"])
            target_id = str(edge["target_id"])
            if source_id in temp_graph.nodes and target_id in temp_graph.nodes:
                temp_graph.add_edge(source_id, target_id, **{
                    "relation": edge["relation"],
                    "display_relation": edge["display_relation"]
                })
        
        # Create ASCII graph visualization
        ascii_graph = create_ascii_graph_visualization(temp_graph, nodes_df)
        context.log.info("Random Sample Graph ASCII Visualization:")
        context.log.info("\n" + ascii_graph)
        
        return {
            "random_nodes": nodes_df,
            "random_edges": edges_df,
            "sample_size": {"nodes": len(nodes_df), "edges": len(edges_df)}
        }
        
    finally:
        driver.close()


@asset(group_name="embeddings", compute_kind="transform")
def knowledge_graph(
    context: AssetExecutionContext,
    random_graph_sample: Dict,  # Use the random sample instead of full graph
) -> nx.Graph:
    """Build knowledge graph from random sample data for efficient processing."""
    context.log.info("Building knowledge graph from random sample...")
    
    # Extract sample data
    nodes_df = random_graph_sample["random_nodes"]
    edges_df = random_graph_sample["random_edges"]
    sample_size = random_graph_sample["sample_size"]
    
    context.log.info(f"Building graph from {sample_size['nodes']} nodes and {sample_size['edges']} edges")
    
    # Create NetworkX graph
    graph = nx.Graph()
    
    # Add nodes to graph
    for _, node in nodes_df.iterrows():
        node_id = str(node["node_id"])
        graph.add_node(node_id, **{
            "node_name": node["node_name"],
            "node_type": node["node_type"],
            "node_source": node["node_source"]
        })
    
    # Add edges to graph
    for _, edge in edges_df.iterrows():
        source_id = str(edge["source_id"])
        target_id = str(edge["target_id"])
        
        # Only add edge if both nodes exist in our sample
        if source_id in graph.nodes and target_id in graph.nodes:
            graph.add_edge(source_id, target_id, **{
                "relation": edge["relation"],
                "display_relation": edge["display_relation"]
            })
    

    # Get node type distribution
    node_types = {}
    for node_id in graph.nodes:
        node_type = graph.nodes[node_id].get("node_type", "unknown")
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    # Get relation distribution
    relations = {}
    for edge in graph.edges(data=True):
        relation = edge[2].get("relation", "unknown")
        relations[relation] = relations.get(relation, 0) + 1
    
    context.add_output_metadata({
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "sample_based": True,
        "node_types": node_types,
        "edge_relations": dict(list(relations.items())[:10]),  # Top 10 relations
        "connected_components": nx.number_connected_components(graph),
        "avg_clustering": float(round(nx.average_clustering(graph), 4)) if graph.number_of_edges() > 0 else 0.0
    })
    
    return graph


@asset(group_name="embeddings", compute_kind="ml")
def node2vec_embeddings(
    context: AssetExecutionContext,
    knowledge_graph: nx.Graph,
) -> Dict[str, np.ndarray]:
    """Train Node2Vec embeddings on the knowledge graph."""
    context.log.info("Training Node2Vec embeddings...")
    
    # Node2Vec hyperparameters
    embedding_params = {
        "dimensions": 512,
        "walk_length": 30,
        "num_walks": 10,
        "window_size": 10,
        "workers": 4,
        "p": 1.0,
        "q": 1.0,
        "seed": 42,
    }
    
    embeddings = train_node2vec_embeddings(
        graph=knowledge_graph,
        **embedding_params
    )
    
    # Convert all keys to strings for Dagster type compliance
    string_keyed_embeddings = {str(k): v for k, v in embeddings.items()}
    
    context.add_output_metadata({
        "num_embeddings": len(string_keyed_embeddings),
        "embedding_dimension": int(string_keyed_embeddings[list(string_keyed_embeddings.keys())[0]].shape[0]),
        "hyperparameters": embedding_params,
    })
    
    return string_keyed_embeddings


@asset(group_name="embeddings", compute_kind="transform")
def flattened_embeddings(
    context: AssetExecutionContext,
    node2vec_embeddings: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Flatten embeddings for use in ML models."""
    context.log.info("Flattening embeddings for ML models...")
    
    # Create DataFrame directly from embeddings dictionary
    data = []
    for node_id, embedding in node2vec_embeddings.items():
        data.append({
            'node_id': node_id,
            'embedding': embedding,
        })
    
    embedding_df = pd.DataFrame(data)
    
    flattened_df = flatten_embeddings_for_model(
        df=embedding_df,
        embedding_column="embedding",
        prefix="emb_",
    )
    
    # Save flattened embeddings for inspection
    output_path = "data/06_models/embeddings/flattened_embeddings.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save a sample for inspection (full file would be too large)
    sample_df = flattened_df.head(1000)
    sample_df.to_csv(output_path, index=False)
    
    context.add_output_metadata({
        "num_rows": len(flattened_df),
        "num_columns": len(flattened_df.columns),
        "embedding_dimensions": len([col for col in flattened_df.columns if col.startswith("emb_")]),
        "sample_saved_to": output_path,
        "memory_usage_mb": float(round(flattened_df.memory_usage(deep=True).sum() / (1024 * 1024), 2)),
    })
    
    return flattened_df


@asset(group_name="embeddings", compute_kind="visualization")
def embedding_visualizations(
    context: AssetExecutionContext,
    node2vec_embeddings: Dict[str, np.ndarray],
    random_graph_sample: Dict,
) -> Dict[str, Any]:
    """Create visualizations of the Node2Vec embeddings."""
    context.log.info("Creating embedding visualizations...")
    
    try:
        # Import visualization libraries
        from sklearn.decomposition import PCA
        import plotly.express as px
    except ImportError as e:
        context.log.warning(f"Visualization libraries not available: {e}")
        context.log.info("Install with: pip install scikit-learn plotly")
        return {"status": "skipped", "reason": "missing_dependencies"}
    
    # Prepare data
    nodes_df = random_graph_sample["random_nodes"]
    
    # Get embeddings and metadata
    embedding_data = []
    for node_id, embedding in node2vec_embeddings.items():
        # Find node info
        node_info = nodes_df[nodes_df['node_id'] == int(node_id)]
        if not node_info.empty:
            embedding_data.append({
                'node_id': node_id,
                'node_name': node_info.iloc[0]['node_name'],
                'node_type': node_info.iloc[0]['node_type'],
                'embedding': embedding
            })
    
    if not embedding_data:
        context.log.warning("No embedding data found for visualization")
        return {"status": "no_data"}
    
    # Convert to arrays
    embeddings_matrix = np.array([item['embedding'] for item in embedding_data])
    node_types = [item['node_type'] for item in embedding_data]
    node_names = [item['node_name'] for item in embedding_data]
    
    context.log.info(f"Visualizing {len(embeddings_matrix)} embeddings of dimension {embeddings_matrix.shape[1]}")
    
    # Create visualization directory
    viz_dir = Path("data/06_models/embeddings/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. PCA Analysis
    context.log.info("Computing PCA...")
    pca = PCA(n_components=min(3, embeddings_matrix.shape[1]))
    pca_embeddings = pca.fit_transform(embeddings_matrix)
    
    # Create PCA DataFrame
    pca_df = pd.DataFrame({
        'node_id': [item['node_id'] for item in embedding_data],
        'node_name': node_names,
        'node_type': node_types,
        'PC1': pca_embeddings[:, 0],
        'PC2': pca_embeddings[:, 1],
        'PC3': pca_embeddings[:, 2] if pca_embeddings.shape[1] > 2 else 0
    })
    
    # 2. Create interactive Plotly visualization
    context.log.info("Creating interactive plots...")
    
    # PCA 2D scatter plot
    fig_2d = px.scatter(
        pca_df, 
        x='PC1', 
        y='PC2',
        color='node_type',
        hover_data=['node_name', 'node_id'],
        title='Node2Vec Embeddings - PCA 2D Projection',
        width=800,
        height=600
    )
    
    # Save interactive plot
    plot_2d_path = viz_dir / "embeddings_pca_2d.html"
    fig_2d.write_html(str(plot_2d_path))
    
    # PCA 3D scatter plot
    fig_3d = px.scatter_3d(
        pca_df,
        x='PC1',
        y='PC2', 
        z='PC3',
        color='node_type',
        hover_data=['node_name', 'node_id'],
        title='Node2Vec Embeddings - PCA 3D Projection',
        width=800,
        height=600
    )
    
    # Save 3D plot
    plot_3d_path = viz_dir / "embeddings_pca_3d.html"
    fig_3d.write_html(str(plot_3d_path))
    
    # 3. Embedding statistics by node type
    context.log.info("Computing embedding statistics by node type...")
    
    type_stats = {}
    for node_type in set(node_types):
        type_mask = np.array(node_types) == node_type
        type_embeddings = embeddings_matrix[type_mask]
        
        if len(type_embeddings) > 0:
            type_stats[node_type] = {
                'count': len(type_embeddings),
                'mean_norm': float(np.mean([np.linalg.norm(emb) for emb in type_embeddings])),
                'std_norm': float(np.std([np.linalg.norm(emb) for emb in type_embeddings])),
                'mean_values': float(np.mean(type_embeddings)),
                'std_values': float(np.std(type_embeddings))
            }
    
    # 4. Create ASCII embedding summary
    ascii_summary = create_embedding_ascii_summary(pca_df, type_stats, pca)
    context.log.info("Embedding Visualization Summary:")
    context.log.info("\n" + ascii_summary)
    
    # 5. Save summary data
    summary_data = {
        'pca_explained_variance': [float(x) for x in pca.explained_variance_ratio_.tolist()],
        'node_type_stats': type_stats,
        'total_nodes': len(embedding_data),
        'embedding_dimension': int(embeddings_matrix.shape[1]),
        'visualization_files': {
            'pca_2d': str(plot_2d_path),
            'pca_3d': str(plot_3d_path)
        }
    }
    
    context.add_output_metadata({
        "visualizations_created": 2,
        "pca_explained_variance_2d": f"{float(sum(pca.explained_variance_ratio_[:2])):.3f}",
        "pca_explained_variance_3d": f"{float(sum(pca.explained_variance_ratio_[:3])):.3f}",
        "output_directory": str(viz_dir),
        "node_types_visualized": len(type_stats),
        "interactive_plots": ["embeddings_pca_2d.html", "embeddings_pca_3d.html"],
        "📊 2D PCA Plot": f"file://{plot_2d_path.absolute()}",
        "🎯 3D PCA Plot": f"file://{plot_3d_path.absolute()}",
        "visualization_urls": {
            "2d_plot": f"file://{plot_2d_path.absolute()}",
            "3d_plot": f"file://{plot_3d_path.absolute()}"
        }
    })
    
    return summary_data


def create_embedding_ascii_summary(pca_df: pd.DataFrame, type_stats: Dict, pca) -> str:
    """Create ASCII summary of embedding analysis."""
    lines = []
    lines.append("=" * 80)
    lines.append("🎯 NODE2VEC EMBEDDINGS VISUALIZATION SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    
    # PCA Analysis
    lines.append("📊 PCA ANALYSIS:")
    explained_var = pca.explained_variance_ratio_
    lines.append(f"   PC1 explains: {explained_var[0]:.1%} of variance")
    lines.append(f"   PC2 explains: {explained_var[1]:.1%} of variance")
    if len(explained_var) > 2:
        lines.append(f"   PC3 explains: {explained_var[2]:.1%} of variance")
    lines.append(f"   Total (2D): {sum(explained_var[:2]):.1%} of variance captured")
    lines.append("")
    
    # Node type distribution in embedding space
    lines.append("🏷️  NODE TYPES IN EMBEDDING SPACE:")
    for node_type, stats in type_stats.items():
        symbol = {"drug": "💊", "disease": "🦠", "protein": "🧬", "gene": "🧬"}.get(node_type, "⚫")
        lines.append(f"   {symbol} {node_type}: {stats['count']} nodes")
        lines.append(f"      ├─ Avg embedding norm: {stats['mean_norm']:.3f} ± {stats['std_norm']:.3f}")
        lines.append(f"      └─ Avg value: {stats['mean_values']:.3f} ± {stats['std_values']:.3f}")
    lines.append("")
    
    # Embedding space insights
    lines.append("🔍 EMBEDDING SPACE INSIGHTS:")
    
    # Find node types that are close/far in PC1-PC2 space
    type_means = {}
    for node_type in type_stats.keys():
        type_mask = pca_df['node_type'] == node_type
        if type_mask.sum() > 0:
            type_means[node_type] = {
                'pc1': pca_df[type_mask]['PC1'].mean(),
                'pc2': pca_df[type_mask]['PC2'].mean()
            }
    
    # Calculate distances between node types
    if len(type_means) > 1:
        distances = []
        type_list = list(type_means.keys())
        for i in range(len(type_list)):
            for j in range(i+1, len(type_list)):
                t1, t2 = type_list[i], type_list[j]
                dist = np.sqrt((type_means[t1]['pc1'] - type_means[t2]['pc1'])**2 + 
                              (type_means[t1]['pc2'] - type_means[t2]['pc2'])**2)
                distances.append((t1, t2, dist))
        
        # Sort by distance
        distances.sort(key=lambda x: x[2])
        
        if distances:
            closest = distances[0]
            farthest = distances[-1]
            lines.append(f"   📍 Closest node types: {closest[0]} ↔ {closest[1]} (distance: {closest[2]:.2f})")
            lines.append(f"   📍 Farthest node types: {farthest[0]} ↔ {farthest[1]} (distance: {farthest[2]:.2f})")
    
    lines.append("")
    lines.append("📁 INTERACTIVE VISUALIZATIONS SAVED:")
    lines.append("   ├─ embeddings_pca_2d.html (2D scatter plot)")
    lines.append("   └─ embeddings_pca_3d.html (3D scatter plot)")
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


@asset(group_name="embeddings", compute_kind="memgraph", deps=[node2vec_embeddings])
def memgraph_embedding_visualizations(
    context: AssetExecutionContext,
) -> Dict[str, Any]:
    """Create visualizations using embeddings fetched directly from Memgraph."""
    import numpy as np
    
    context.log.info("Creating embedding visualizations using Memgraph embeddings...")
    
    try:
        # Import visualization libraries
        from sklearn.decomposition import PCA
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError as e:
        context.log.warning(f"Visualization libraries not available: {e}")
        context.log.info("Install with: pip install scikit-learn plotly")
        return {"status": "skipped", "reason": "missing_dependencies"}
    
    # Get all nodes from Memgraph
    context.log.info("Fetching all nodes from Memgraph...")
    
    # Query all nodes from Memgraph using direct connection (since MCP functions aren't directly importable)
    try:
        from neo4j import GraphDatabase
        import os
        
        # Get Memgraph connection details
        memgraph_uri = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
        memgraph_user = os.getenv("MEMGRAPH_USER", "")
        memgraph_password = os.getenv("MEMGRAPH_PASSWORD", "")
        
        # Handle authentication
        auth = None
        if memgraph_user or memgraph_password:
            auth = (memgraph_user, memgraph_password)
        
        driver = GraphDatabase.driver(memgraph_uri, auth=auth)
        
        with driver.session() as session:
            # Query all nodes from Memgraph with their embeddings
            all_nodes_query = """
            MATCH (n:Node)
            WHERE n.embedding IS NOT NULL
            RETURN n.node_id as node_id,
                   n.node_name as node_name, 
                   n.node_type as node_type,
                   n.node_source as node_source,
                   n.embedding as embedding
            LIMIT 1000
            """
            
            result = session.run(all_nodes_query)
            embedding_data = []
            node_ids = set()
            
            for record in result:
                embedding = np.array(record['embedding'])
                node_id = str(record['node_id'])
                embedding_data.append({
                    'node_id': node_id,
                    'node_name': record['node_name'],
                    'node_type': record['node_type'],
                    'embedding': embedding
                })
                node_ids.add(record['node_id'])
            
            # Query edges between the nodes that have embeddings
            edges_query = """
            MATCH (a:Node)-[r:RELATES]->(b:Node)
            WHERE a.node_id IN $node_ids AND b.node_id IN $node_ids
            RETURN a.node_id as source_id,
                   b.node_id as target_id,
                   r.relation as relation,
                   r.display_relation as display_relation
            """
            
            result = session.run(edges_query, node_ids=list(node_ids))
            edges_data = []
            
            for record in result:
                edges_data.append({
                    'source_id': str(record['source_id']),
                    'target_id': str(record['target_id']),
                    'relation': record['relation'],
                    'display_relation': record.get('display_relation', record['relation'])
                })
        
        driver.close()
        
        if not embedding_data:
            raise ValueError("No embeddings found in Memgraph. Ensure embeddings are stored in node properties.")
        
        context.log.info(f"Retrieved {len(embedding_data)} nodes with embeddings from Memgraph")
        
        # Log embedding statistics
        embedding_dims = [len(item['embedding']) for item in embedding_data]
        context.log.info(f"Embedding dimensions: min={min(embedding_dims)}, max={max(embedding_dims)}, avg={np.mean(embedding_dims):.1f}")
        
        # Log node types
        node_types = [item['node_type'] for item in embedding_data]
        type_counts = {}
        for node_type in node_types:
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        context.log.info(f"Node types with embeddings: {type_counts}")
        
    except Exception as e:
        context.log.error(f"Failed to fetch nodes and embeddings from Memgraph: {e}")
        raise
    
    # Convert to arrays for analysis
    embeddings_matrix = np.array([item['embedding'] for item in embedding_data])
    node_types = [item['node_type'] for item in embedding_data]
    node_names = [item['node_name'] for item in embedding_data]
    
    context.log.info(f"Analyzing {len(embeddings_matrix)} Memgraph embeddings of dimension {embeddings_matrix.shape[1]}")
    
    # Create visualization directory
    viz_dir = Path("data/06_models/embeddings/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. PCA Analysis
    context.log.info("Computing PCA on Memgraph embeddings...")
    pca = PCA(n_components=min(3, embeddings_matrix.shape[1]))
    pca_embeddings = pca.fit_transform(embeddings_matrix)
    
    # Create PCA DataFrame
    pca_df = pd.DataFrame({
        'node_id': [item['node_id'] for item in embedding_data],
        'node_name': node_names,
        'node_type': node_types,
        'PC1': pca_embeddings[:, 0],
        'PC2': pca_embeddings[:, 1],
        'PC3': pca_embeddings[:, 2] if pca_embeddings.shape[1] > 2 else 0
    })
    
    # 2. Create interactive Plotly visualizations
    context.log.info("Creating interactive plots from Memgraph embeddings...")
    
    # PCA 2D scatter plot
    fig_2d = px.scatter(
        pca_df, 
        x='PC1', 
        y='PC2',
        color='node_type',
        hover_data=['node_name', 'node_id'],
        title='Memgraph Node Embeddings - PCA 2D Projection',
        width=800,
        height=600
    )
    
    # Save interactive plot
    plot_2d_path = viz_dir / "memgraph_embeddings_pca_2d.html"
    fig_2d.write_html(str(plot_2d_path))
    
    # PCA 3D scatter plot
    fig_3d = px.scatter_3d(
        pca_df,
        x='PC1',
        y='PC2', 
        z='PC3',
        color='node_type',
        hover_data=['node_name', 'node_id'],
        title='Memgraph Node Embeddings - PCA 3D Projection',
        width=800,
        height=600
    )
    
    # Save 3D plot
    plot_3d_path = viz_dir / "memgraph_embeddings_pca_3d.html"
    fig_3d.write_html(str(plot_3d_path))
    
    # 3. Create graph visualization with legends
    context.log.info("Creating graph visualization with legends...")
    
    # Build a NetworkX graph from the embedding data
    G = nx.Graph()
    
    # Add nodes with their types and embeddings
    for item in embedding_data:
        G.add_node(item['node_id'], 
                  node_name=item['node_name'],
                  node_type=item['node_type'],
                  embedding=item['embedding'])
    
    # Add edges from Memgraph data
    for edge in edges_data:
        source = edge['source_id']
        target = edge['target_id']
        if source in G.nodes() and target in G.nodes():
            G.add_edge(source, target, 
                      relation=edge['relation'],
                      display_relation=edge['display_relation'])
    
    # Create interactive network plot using plotly
    # Get node positions using spring layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Prepare node traces for different types
    node_traces = {}
    node_type_colors = {
        'drug': '#FF6B6B',      # Red
        'disease': '#4ECDC4',   # Teal  
        'protein': '#45B7D1',   # Blue
        'gene': '#96CEB4',      # Green
        'other': '#FECA57'      # Yellow
    }
    
    for node_type in set(node_types):
        node_traces[node_type] = {
            'x': [],
            'y': [],
            'text': [],
            'ids': []
        }
    
    # Fill node traces
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        node_type = node_data['node_type']
        
        if node_type not in node_traces:
            node_type = 'other'
            
        x, y = pos[node_id]
        node_traces[node_type]['x'].append(x)
        node_traces[node_type]['y'].append(y)
        node_traces[node_type]['text'].append(f"{node_data['node_name']}<br>Type: {node_type}<br>ID: {node_id}")
        node_traces[node_type]['ids'].append(node_id)
    
    # Create edge traces with relationship information
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Get relationship information
        edge_data = edge[2]
        relation = edge_data.get('display_relation', edge_data.get('relation', 'connected'))
        source_name = G.nodes[edge[0]]['node_name']
        target_name = G.nodes[edge[1]]['node_name']
        edge_info.extend([f"{source_name} → {relation} → {target_name}", 
                         f"{source_name} → {relation} → {target_name}", 
                         None])
    
    # Create the figure
    
    fig_graph = go.Figure()
    
    # Add edges with hover information
    fig_graph.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        mode='lines',
        text=edge_info,
        hovertemplate='%{text}<extra></extra>',
        name='Relationships'
    ))
    
    # Add node traces for each type
    symbols = {'drug': 'circle', 'disease': 'square', 'protein': 'diamond', 'gene': 'triangle-up', 'other': 'star'}
    
    for node_type, trace_data in node_traces.items():
        if trace_data['x']:  # Only add if there are nodes of this type
            symbol = symbols.get(node_type, 'circle')
            color = node_type_colors.get(node_type, '#FECA57')
            emoji = {"drug": "💊", "disease": "🦠", "protein": "🧬", "gene": "🧬"}.get(node_type, "⚫")
            
            fig_graph.add_trace(go.Scatter(
                x=trace_data['x'], 
                y=trace_data['y'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=color,
                    symbol=symbol,
                    line=dict(width=2, color='white')
                ),
                text=trace_data['text'],
                hoverinfo='text',
                name=f'{emoji} {node_type.title()}'
            ))
    
    # Update layout
    fig_graph.update_layout(
        title={
            'text': 'Memgraph Knowledge Graph - Actual Node Relationships',
            'font': {'size': 16}
        },
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="Nodes connected by actual relationships from Memgraph knowledge graph",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color="#666", size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=900,
        height=700
    )
    
    # Save graph plot
    plot_graph_path = viz_dir / "memgraph_embedding_graph.html"
    fig_graph.write_html(str(plot_graph_path))
    
    context.log.info(f"Created graph visualization with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # 4. Embedding statistics by node type
    context.log.info("Computing Memgraph embedding statistics by node type...")
    
    type_stats = {}
    for node_type in set(node_types):
        type_mask = np.array(node_types) == node_type
        type_embeddings = embeddings_matrix[type_mask]
        
        if len(type_embeddings) > 0:
            type_stats[node_type] = {
                'count': len(type_embeddings),
                'mean_norm': float(np.mean([np.linalg.norm(emb) for emb in type_embeddings])),
                'std_norm': float(np.std([np.linalg.norm(emb) for emb in type_embeddings])),
                'mean_values': float(np.mean(type_embeddings)),
                'std_values': float(np.std(type_embeddings))
            }
    
    # 4. Create ASCII embedding summary
    ascii_summary = create_memgraph_embedding_ascii_summary(pca_df, type_stats, pca)
    context.log.info("Memgraph Embedding Visualization Summary:")
    context.log.info("\n" + ascii_summary)
    
    # 5. Save summary data
    summary_data = {
        'source': 'memgraph',
        'pca_explained_variance': [float(x) for x in pca.explained_variance_ratio_.tolist()],
        'node_type_stats': type_stats,
        'total_nodes_processed': len(embedding_data),
        'embedding_dimension': int(embeddings_matrix.shape[1]),
        'graph_stats': {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges()
        },
        'visualization_files': {
            'pca_2d': str(plot_2d_path),
            'pca_3d': str(plot_3d_path),
            'graph': str(plot_graph_path)
        }
    }
    
    context.add_output_metadata({
        "embedding_source": "memgraph",
        "visualizations_created": 3,
        "pca_explained_variance_2d": f"{float(sum(pca.explained_variance_ratio_[:2])):.3f}",
        "pca_explained_variance_3d": f"{float(sum(pca.explained_variance_ratio_[:3])):.3f}",
        "output_directory": str(viz_dir),
        "node_types_visualized": len(type_stats),
        "embeddings_retrieved": len(embedding_data),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "interactive_plots": ["memgraph_embeddings_pca_2d.html", "memgraph_embeddings_pca_3d.html", "memgraph_embedding_graph.html"],
        "📊 2D PCA Plot": f"file://{plot_2d_path.absolute()}",
        "🎯 3D PCA Plot": f"file://{plot_3d_path.absolute()}",
        "🕸️ Graph Plot": f"file://{plot_graph_path.absolute()}",
        "visualization_urls": {
            "2d_plot": f"file://{plot_2d_path.absolute()}",
            "3d_plot": f"file://{plot_3d_path.absolute()}",
            "graph_plot": f"file://{plot_graph_path.absolute()}"
        }
    })
    
    return summary_data


def create_memgraph_embedding_ascii_summary(pca_df: pd.DataFrame, type_stats: Dict, pca) -> str:
    """Create ASCII summary of Memgraph embedding analysis."""
    lines = []
    lines.append("=" * 80)
    lines.append("🎯 MEMGRAPH EMBEDDINGS VISUALIZATION SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    
    # Data source info
    lines.append("📊 DATA SOURCE:")
    lines.append("   ├─ Source: Memgraph Database (All Nodes with Embeddings)")
    lines.append(f"   └─ Successfully retrieved: {len(pca_df)} embeddings")
    lines.append("")
    
    # PCA Analysis
    lines.append("📈 PCA ANALYSIS:")
    explained_var = pca.explained_variance_ratio_
    lines.append(f"   PC1 explains: {explained_var[0]:.1%} of variance")
    lines.append(f"   PC2 explains: {explained_var[1]:.1%} of variance")
    if len(explained_var) > 2:
        lines.append(f"   PC3 explains: {explained_var[2]:.1%} of variance")
    lines.append(f"   Total (2D): {sum(explained_var[:2]):.1%} of variance captured")
    lines.append("")
    
    # Node type distribution in embedding space
    lines.append("🏷️  NODE TYPES IN MEMGRAPH EMBEDDING SPACE:")
    for node_type, stats in type_stats.items():
        symbol = {"drug": "💊", "disease": "🦠", "protein": "🧬", "gene": "🧬"}.get(node_type, "⚫")
        lines.append(f"   {symbol} {node_type}: {stats['count']} nodes")
        lines.append(f"      ├─ Avg embedding norm: {stats['mean_norm']:.3f} ± {stats['std_norm']:.3f}")
        lines.append(f"      └─ Avg value: {stats['mean_values']:.3f} ± {stats['std_values']:.3f}")
    lines.append("")
    
    # Embedding space insights
    lines.append("🔍 MEMGRAPH EMBEDDING SPACE INSIGHTS:")
    
    # Find node types that are close/far in PC1-PC2 space
    type_means = {}
    for node_type in type_stats.keys():
        type_mask = pca_df['node_type'] == node_type
        if type_mask.sum() > 0:
            type_means[node_type] = {
                'pc1': pca_df[type_mask]['PC1'].mean(),
                'pc2': pca_df[type_mask]['PC2'].mean()
            }
    
    # Calculate distances between node types
    if len(type_means) > 1:
        distances = []
        type_list = list(type_means.keys())
        for i in range(len(type_list)):
            for j in range(i+1, len(type_list)):
                t1, t2 = type_list[i], type_list[j]
                dist = np.sqrt((type_means[t1]['pc1'] - type_means[t2]['pc1'])**2 + 
                              (type_means[t1]['pc2'] - type_means[t2]['pc2'])**2)
                distances.append((t1, t2, dist))
        
        # Sort by distance
        distances.sort(key=lambda x: x[2])
        
        if distances:
            closest = distances[0]
            farthest = distances[-1]
            lines.append(f"   📍 Closest node types: {closest[0]} ↔ {closest[1]} (distance: {closest[2]:.2f})")
            lines.append(f"   📍 Farthest node types: {farthest[0]} ↔ {farthest[1]} (distance: {farthest[2]:.2f})")
    
    lines.append("")
    lines.append("📁 INTERACTIVE VISUALIZATIONS SAVED:")
    lines.append("   ├─ memgraph_embeddings_pca_2d.html (2D scatter plot)")
    lines.append("   ├─ memgraph_embeddings_pca_3d.html (3D scatter plot)")
    lines.append("   └─ memgraph_embedding_graph.html (Network graph with legends)")
    lines.append("")
    lines.append("🔄 COMPARISON WITH NODE2VEC:")
    lines.append("   ├─ This asset fetches REAL embeddings from Memgraph n.embedding property")
    lines.append("   ├─ Only processes nodes that have embeddings stored")
    lines.append("   ├─ Previous asset (embedding_visualizations) uses Node2Vec output from sample")
    lines.append("   └─ Both run after node2vec_embeddings for comparison")
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)