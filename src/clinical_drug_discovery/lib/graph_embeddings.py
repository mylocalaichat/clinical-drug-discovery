"""
Graph embeddings using Node2Vec for knowledge graph node embeddings.

This module implements the Node2Vec algorithm to generate topological embeddings
for nodes in the knowledge graph, which are then used as features for the
supervised learning model.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from node2vec import Node2Vec
from tqdm import tqdm


def load_graph_from_neo4j(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    database: str = "primekg",
    exclude_edge_types: Optional[List[str]] = None,
    limit_nodes: Optional[int] = None,
) -> nx.Graph:
    """
    Load knowledge graph from Neo4j into NetworkX.

    Args:
        neo4j_uri: Neo4j connection URI
        neo4j_user: Username
        neo4j_password: Password
        database: Database name
        exclude_edge_types: Edge types to exclude (e.g., ['INDICATION'] to prevent leakage)
        limit_nodes: Limit number of nodes to load for testing (None for all nodes)

    Returns:
        NetworkX graph
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    # Build edge type filter
    edge_filter = ""
    if exclude_edge_types:
        edge_filter = f"AND NOT type(r) IN {exclude_edge_types}"

    # Build limit clause for testing with smaller graphs
    if limit_nodes:
        # First get a sample of nodes, then their edges
        query = f"""
        MATCH (n:PrimeKGNode)
        WITH n LIMIT {limit_nodes}
        MATCH (n)-[r]->(m:PrimeKGNode)
        WHERE r IS NOT NULL
        {edge_filter}
        RETURN n.node_id as source, m.node_id as target, type(r) as edge_type
        """
    else:
        # Original query for full graph
        query = f"""
        MATCH (n:PrimeKGNode)-[r]->(m:PrimeKGNode)
        WHERE r IS NOT NULL
        {edge_filter}
        RETURN n.node_id as source, m.node_id as target, type(r) as edge_type
        """

    limit_info = f" (limited to {limit_nodes} nodes)" if limit_nodes else ""
    print(f"Loading graph from Neo4j{limit_info}...")
    try:
        with driver.session(database=database) as session:
            result = session.run(query)
            edges = [(record["source"], record["target"], {"edge_type": record["edge_type"]})
                     for record in result]

        G = nx.Graph()
        G.add_edges_from(edges)

        limit_msg = f" (limited from {limit_nodes} source nodes)" if limit_nodes else ""
        print(f"Loaded graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges{limit_msg}")
        return G

    finally:
        driver.close()


def train_node2vec_embeddings(
    graph: nx.Graph,
    dimensions: int = 512,
    walk_length: int = 30,
    num_walks: int = 10,
    window_size: int = 10,
    workers: int = 4,
    p: float = 1.0,
    q: float = 1.0,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Train Node2Vec embeddings on the graph.

    Args:
        graph: NetworkX graph
        dimensions: Embedding dimensionality (default 512)
        walk_length: Length of each random walk (default 30)
        num_walks: Number of walks per node (default 10)
        window_size: Context window size for skip-gram (default 10)
        workers: Number of parallel workers
        p: Return parameter (controls likelihood of revisiting nodes)
        q: In-out parameter (controls BFS vs DFS exploration)
        seed: Random seed

    Returns:
        Dictionary mapping node_id to embedding vector
    """
    print(f"\nTraining Node2Vec embeddings...")
    print(f"Parameters:")
    print(f"  - Dimensions: {dimensions}")
    print(f"  - Walk length: {walk_length}")
    print(f"  - Num walks: {num_walks}")
    print(f"  - Window size: {window_size}")
    print(f"  - p: {p}, q: {q}")

    # Initialize Node2Vec
    node2vec = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        seed=seed,
    )

    # Train the model
    print("Generating random walks...")
    model = node2vec.fit(
        window=window_size,
        min_count=1,
        batch_words=4,
        workers=workers,
    )

    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = {}
    nodes_with_embeddings = 0
    nodes_without_embeddings = 0
    
    for node in tqdm(graph.nodes(), desc="Extracting embeddings"):
        try:
            embeddings[node] = model.wv[node]
            nodes_with_embeddings += 1
        except KeyError:
            # Node not in vocabulary (likely isolated or very few connections)
            nodes_without_embeddings += 1
            continue

    print(f"✓ Generated {len(embeddings):,} node embeddings of dimension {dimensions}")
    if nodes_without_embeddings > 0:
        print(f"⚠️  {nodes_without_embeddings:,} nodes skipped (not in vocabulary)")
        print(f"📊 Coverage: {nodes_with_embeddings}/{nodes_with_embeddings + nodes_without_embeddings} nodes ({100 * nodes_with_embeddings / (nodes_with_embeddings + nodes_without_embeddings):.1f}%)")

    return embeddings


def save_embeddings(
    embeddings: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    """
    Save embeddings to disk.

    Args:
        embeddings: Dictionary of node_id -> embedding
        output_path: Path to save the embeddings
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"Saved embeddings to: {output_file}")


def load_embeddings(input_path: str) -> Dict[str, np.ndarray]:
    """
    Load embeddings from disk.

    Args:
        input_path: Path to the embeddings file

    Returns:
        Dictionary of node_id -> embedding
    """
    with open(input_path, 'rb') as f:
        embeddings = pickle.load(f)

    print(f"Loaded {len(embeddings):,} embeddings from {input_path}")
    return embeddings


def get_embedding_for_node(
    node_id: str,
    embeddings: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    """
    Get embedding for a specific node.

    Args:
        node_id: Node identifier
        embeddings: Dictionary of embeddings

    Returns:
        Embedding vector or None if not found
    """
    return embeddings.get(node_id)


def create_embedding_dataframe(
    embeddings: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Convert embeddings dictionary to DataFrame.

    Args:
        embeddings: Dictionary of node_id -> embedding

    Returns:
        DataFrame with node_id and embedding columns
    """
    data = []
    for node_id, embedding in embeddings.items():
        data.append({
            'node_id': node_id,
            'embedding': embedding,
        })

    df = pd.DataFrame(data)
    print(f"Created embedding DataFrame: {len(df):,} rows")
    return df


def flatten_embeddings_for_model(
    df: pd.DataFrame,
    embedding_column: str = 'embedding',
    prefix: str = 'emb_',
) -> pd.DataFrame:
    """
    Flatten embedding arrays into individual columns for ML models.

    This transforms a DataFrame with an embedding column containing arrays
    into a DataFrame with separate columns for each dimension.

    Example:
        Input:  | node_id | embedding          |
                | A       | [0.1, 0.2, 0.3]    |

        Output: | node_id | emb_0 | emb_1 | emb_2 |
                | A       | 0.1   | 0.2   | 0.3   |

    Args:
        df: DataFrame with embedding column
        embedding_column: Name of the column containing embeddings
        prefix: Prefix for the new columns

    Returns:
        DataFrame with flattened embeddings
    """
    # Extract embeddings
    embeddings_list = df[embedding_column].tolist()

    # Get dimensions
    embedding_dim = len(embeddings_list[0])

    # Create column names
    column_names = [f"{prefix}{i}" for i in range(embedding_dim)]

    # Create DataFrame from embeddings
    embeddings_df = pd.DataFrame(embeddings_list, columns=column_names, index=df.index)

    # Concatenate with original DataFrame (excluding embedding column)
    result = pd.concat([df.drop(columns=[embedding_column]), embeddings_df], axis=1)

    print(f"Flattened embeddings: {embedding_dim} dimensions -> {len(column_names)} columns")
    return result
