# Edge Prediction Guide for Drug Repurposing

## Overview

There are multiple ways to predict new drug-disease edges using your existing graph:

1. **Pattern-Based Inference** (Cypher queries) - Simple, interpretable
2. **Path-Based Scoring** - Score potential edges based on connecting paths
3. **Graph Embeddings** - ML-based approach using node2vec or similar
4. **Graph Neural Networks** - Advanced ML with PyTorch Geometric
5. **Ensemble Methods** - Combine multiple approaches

## Method 1: Pattern-Based Inference (Cypher)

### Approach 1.1: Similarity-Based Inference

Find drugs that might treat diseases based on drug/disease similarities:

```cypher
// Find candidate drug-disease pairs
// Drug A treats Disease B, Drug A similar to Drug C
// Therefore Drug C might treat Disease B

MATCH (drugA)-[treats:RELATES]->(disease),
      (drugA)-[sim:RELATES]->(drugC)
WHERE drugA.node_type = 'drug'
AND disease.node_type = 'disease'
AND drugC.node_type = 'drug'
AND treats.relation = 'drug_treats_disease'
AND sim.relation = 'drug_similarity'
AND sim.similarity_score > 0.6

// Check that drugC doesn't already treat disease
AND NOT EXISTS {
    MATCH (drugC)-[r:RELATES]->(disease)
    WHERE r.relation = 'drug_treats_disease'
}

RETURN DISTINCT
    drugC.node_name as CandidateDrug,
    disease.node_name as TargetDisease,
    drugA.node_name as SimilarDrug,
    sim.similarity_score as DrugSimilarity,
    treats.evidence as KnownEvidence
ORDER BY sim.similarity_score DESC
```

### Approach 1.2: Disease Similarity Inference

```cypher
// Drug treats Disease A, Disease A similar to Disease B
// Therefore Drug might treat Disease B

MATCH (drug)-[treats:RELATES]->(diseaseA),
      (diseaseA)-[sim:RELATES]->(diseaseB)
WHERE drug.node_type = 'drug'
AND diseaseA.node_type = 'disease'
AND diseaseB.node_type = 'disease'
AND treats.relation = 'drug_treats_disease'
AND sim.relation = 'disease_similarity'
AND sim.similarity_score > 0.7

AND NOT EXISTS {
    MATCH (drug)-[r:RELATES]->(diseaseB)
    WHERE r.relation = 'drug_treats_disease'
}

RETURN DISTINCT
    drug.node_name as CandidateDrug,
    diseaseB.node_name as TargetDisease,
    diseaseA.node_name as KnownDisease,
    sim.similarity_score as DiseaseSimilarity,
    treats.efficacy as KnownEfficacy
ORDER BY sim.similarity_score DESC
```

### Approach 1.3: Combined Similarity (Triangle Pattern)

```cypher
// The golden inference pattern:
// Drug A treats Disease X, Drug A similar to Drug B
// Disease X similar to Disease Y
// Therefore Drug B might treat Disease Y

MATCH (drugA)-[treats:RELATES]->(diseaseX),
      (drugA)-[drugSim:RELATES]->(drugB),
      (diseaseX)-[diseaseSim:RELATES]->(diseaseY)
WHERE drugA.node_type = 'drug'
AND drugB.node_type = 'drug'
AND diseaseX.node_type = 'disease'
AND diseaseY.node_type = 'disease'
AND treats.relation = 'drug_treats_disease'
AND drugSim.relation = 'drug_similarity'
AND diseaseSim.relation = 'disease_similarity'
AND drugSim.similarity_score > 0.6
AND diseaseSim.similarity_score > 0.6

AND NOT EXISTS {
    MATCH (drugB)-[r:RELATES]->(diseaseY)
    WHERE r.relation = 'drug_treats_disease'
}

RETURN DISTINCT
    drugB.node_name as CandidateDrug,
    diseaseY.node_name as TargetDisease,
    drugA.node_name as SimilarDrug,
    diseaseX.node_name as SimilarDisease,
    drugSim.similarity_score as DrugSim,
    diseaseSim.similarity_score as DiseaseSim,
    (drugSim.similarity_score + diseaseSim.similarity_score) / 2.0 as AvgScore
ORDER BY AvgScore DESC
```

## Method 2: Path-Based Scoring

Score potential edges based on all connecting paths:

```cypher
// Find all paths from drug to disease through intermediate nodes
// Score based on path count and path quality

MATCH path = (drug)-[*2..4]-(disease)
WHERE drug.node_type = 'drug'
AND disease.node_type = 'disease'
AND drug.is_example = true
AND disease.is_example = true

// Ensure no direct treatment exists
AND NOT EXISTS {
    MATCH (drug)-[r:RELATES]->(disease)
    WHERE r.relation = 'drug_treats_disease'
}

WITH drug, disease,
     count(path) as path_count,
     avg(length(path)) as avg_path_length,
     collect(path) as paths

// Calculate a simple score
WITH drug, disease, path_count, avg_path_length,
     path_count * (1.0 / avg_path_length) as path_score

RETURN drug.node_name as CandidateDrug,
       disease.node_name as TargetDisease,
       path_count as NumPaths,
       round(avg_path_length * 100) / 100 as AvgPathLength,
       round(path_score * 100) / 100 as Score
ORDER BY Score DESC
LIMIT 10
```

## Method 3: Shared Neighbor Analysis

```cypher
// Find drug-disease pairs with many shared protein/pathway connections

MATCH (drug)-[:RELATES]->(intermediate)<-[:RELATES]-(disease)
WHERE drug.node_type = 'drug'
AND disease.node_type = 'disease'
AND intermediate.node_type IN ['gene/protein', 'pathway']

AND NOT EXISTS {
    MATCH (drug)-[r:RELATES]->(disease)
    WHERE r.relation = 'drug_treats_disease'
}

WITH drug, disease,
     count(DISTINCT intermediate) as shared_count,
     collect(DISTINCT intermediate.node_name) as shared_entities

RETURN drug.node_name as CandidateDrug,
       disease.node_name as TargetDisease,
       shared_count as SharedConnections,
       shared_entities as SharedEntities
ORDER BY shared_count DESC
LIMIT 10
```

## Method 4: Graph Embeddings (Python)

Use graph embeddings to learn node representations, then predict edges:

```python
from neo4j import GraphDatabase
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from node2vec import Node2Vec
import networkx as nx

def load_graph_from_memgraph():
    """Load graph from Memgraph into NetworkX"""
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=None)
    session = driver.session()

    # Get all nodes
    result = session.run("""
        MATCH (n {is_example: true})
        RETURN n.node_id as id, n.node_name as name, n.node_type as type
    """)

    G = nx.Graph()
    for record in result:
        G.add_node(record['id'], name=record['name'], type=record['type'])

    # Get all edges
    result = session.run("""
        MATCH (a {is_example: true})-[r {is_example: true}]-(b {is_example: true})
        RETURN a.node_id as source, b.node_id as target, r.relation as relation
    """)

    for record in result:
        G.add_edge(record['source'], record['target'], relation=record['relation'])

    session.close()
    driver.close()

    return G

def train_node2vec_embeddings(G, dimensions=64):
    """Train node2vec embeddings"""
    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=10,
        num_walks=80,
        workers=4
    )

    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    return model

def create_edge_features(model, node1_id, node2_id):
    """Create edge features from node embeddings"""
    emb1 = model.wv[node1_id]
    emb2 = model.wv[node2_id]

    # Use various operations on embeddings
    features = np.concatenate([
        emb1,
        emb2,
        emb1 * emb2,  # Element-wise product
        np.abs(emb1 - emb2)  # Absolute difference
    ])

    return features

def prepare_training_data(G, model):
    """Prepare positive and negative samples"""
    X_positive = []
    X_negative = []

    # Positive samples: existing drug-disease edges
    for edge in G.edges():
        node1, node2 = edge
        type1 = G.nodes[node1].get('type')
        type2 = G.nodes[node2].get('type')

        if (type1 == 'drug' and type2 == 'disease') or \
           (type1 == 'disease' and type2 == 'drug'):
            features = create_edge_features(model, node1, node2)
            X_positive.append(features)

    # Negative samples: non-existing drug-disease pairs
    drugs = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'drug']
    diseases = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'disease']

    for drug in drugs:
        for disease in diseases:
            if not G.has_edge(drug, disease):
                features = create_edge_features(model, drug, disease)
                X_negative.append(features)

    # Combine and create labels
    X = np.array(X_positive + X_negative)
    y = np.array([1] * len(X_positive) + [0] * len(X_negative))

    return X, y

def predict_new_edges(G, model, clf, top_k=10):
    """Predict top K most likely new drug-disease edges"""
    predictions = []

    drugs = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'drug']
    diseases = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'disease']

    for drug in drugs:
        for disease in diseases:
            if not G.has_edge(drug, disease):
                features = create_edge_features(model, drug, disease)
                score = clf.predict_proba([features])[0][1]

                predictions.append({
                    'drug': G.nodes[drug]['name'],
                    'disease': G.nodes[disease]['name'],
                    'score': score
                })

    # Sort by score
    predictions.sort(key=lambda x: x['score'], reverse=True)

    return predictions[:top_k]

# Main workflow
def main():
    print("Loading graph from Memgraph...")
    G = load_graph_from_memgraph()

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("Training node2vec embeddings...")
    model = train_node2vec_embeddings(G)

    print("Preparing training data...")
    X, y = prepare_training_data(G, model)

    print(f"Training data: {len(X)} samples")

    print("Training classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.3f}")

    print("\nTop 10 predicted drug-disease edges:")
    predictions = predict_new_edges(G, model, clf, top_k=10)

    for i, pred in enumerate(predictions, 1):
        print(f"{i}. {pred['drug']} -> {pred['disease']} (score: {pred['score']:.3f})")

if __name__ == "__main__":
    main()
```

## Method 5: Feature Engineering Approach

Create explicit features for drug-disease pairs:

```python
from neo4j import GraphDatabase
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

def extract_pair_features(session, drug_id, disease_id):
    """Extract features for a drug-disease pair"""

    features = {}

    # Feature 1: Count of shared proteins
    result = session.run("""
        MATCH (drug {node_id: $drug_id})-[:RELATES]->(protein)<-[:RELATES]-(disease {node_id: $disease_id})
        WHERE protein.node_type = 'gene/protein'
        RETURN count(DISTINCT protein) as shared_proteins
    """, drug_id=drug_id, disease_id=disease_id)
    features['shared_proteins'] = result.single()['shared_proteins']

    # Feature 2: Count of shared pathways
    result = session.run("""
        MATCH (drug {node_id: $drug_id})-[:RELATES*2]-(pathway)-[:RELATES*2]-(disease {node_id: $disease_id})
        WHERE pathway.node_type = 'pathway'
        RETURN count(DISTINCT pathway) as shared_pathways
    """, drug_id=drug_id, disease_id=disease_id)
    features['shared_pathways'] = result.single()['shared_pathways']

    # Feature 3: Shortest path length
    result = session.run("""
        MATCH path = shortestPath((drug {node_id: $drug_id})-[*]-(disease {node_id: $disease_id}))
        RETURN length(path) as path_length
    """, drug_id=drug_id, disease_id=disease_id)
    rec = result.single()
    features['shortest_path'] = rec['path_length'] if rec else 999

    # Feature 4: Drug similarity to known treatments
    result = session.run("""
        MATCH (drug {node_id: $drug_id})-[sim:RELATES]->(other_drug)-[treats:RELATES]->(disease {node_id: $disease_id})
        WHERE sim.relation = 'drug_similarity'
        AND treats.relation = 'drug_treats_disease'
        RETURN max(sim.similarity_score) as max_drug_sim
    """, drug_id=drug_id, disease_id=disease_id)
    rec = result.single()
    features['max_drug_similarity'] = rec['max_drug_sim'] if rec['max_drug_sim'] else 0

    # Feature 5: Disease similarity to treated diseases
    result = session.run("""
        MATCH (drug {node_id: $drug_id})-[treats:RELATES]->(other_disease)-[sim:RELATES]->(disease {node_id: $disease_id})
        WHERE treats.relation = 'drug_treats_disease'
        AND sim.relation = 'disease_similarity'
        RETURN max(sim.similarity_score) as max_disease_sim
    """, drug_id=drug_id, disease_id=disease_id)
    rec = result.single()
    features['max_disease_similarity'] = rec['max_disease_sim'] if rec['max_disease_sim'] else 0

    return features

def build_training_dataset():
    """Build a dataset of drug-disease pairs with features"""
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=None)
    session = driver.session()

    # Get all drug-disease pairs
    result = session.run("""
        MATCH (drug {is_example: true}), (disease {is_example: true})
        WHERE drug.node_type = 'drug' AND disease.node_type = 'disease'
        RETURN drug.node_id as drug_id, disease.node_id as disease_id
    """)

    all_pairs = list(result)

    # Get positive samples (known treatments)
    result = session.run("""
        MATCH (drug {is_example: true})-[r:RELATES]->(disease {is_example: true})
        WHERE r.relation = 'drug_treats_disease'
        RETURN drug.node_id as drug_id, disease.node_id as disease_id
    """)

    positive_pairs = {(r['drug_id'], r['disease_id']) for r in result}

    # Extract features for all pairs
    data = []
    for pair in all_pairs:
        drug_id = pair['drug_id']
        disease_id = pair['disease_id']

        features = extract_pair_features(session, drug_id, disease_id)
        features['drug_id'] = drug_id
        features['disease_id'] = disease_id
        features['label'] = 1 if (drug_id, disease_id) in positive_pairs else 0

        data.append(features)

    session.close()
    driver.close()

    return pd.DataFrame(data)

def train_and_predict():
    """Train model and predict new edges"""
    print("Building dataset...")
    df = build_training_dataset()

    print(f"Dataset: {len(df)} pairs, {df['label'].sum()} positive")

    # Separate features and labels
    feature_cols = ['shared_proteins', 'shared_pathways', 'shortest_path',
                    'max_drug_similarity', 'max_disease_similarity']
    X = df[feature_cols]
    y = df['label']

    # Train model
    print("Training model...")
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    clf.fit(X, y)

    # Predict on all pairs
    df['score'] = clf.predict_proba(X)[:, 1]

    # Get top predictions (excluding known treatments)
    new_predictions = df[df['label'] == 0].sort_values('score', ascending=False)

    print("\nTop 10 predicted drug-disease edges:")
    for idx, row in new_predictions.head(10).iterrows():
        print(f"{row['drug_id']} -> {row['disease_id']} (score: {row['score']:.3f})")
        print(f"  Features: {row[feature_cols].to_dict()}")

    return new_predictions

if __name__ == "__main__":
    predictions = train_and_predict()
```

## Quick Start: Simple Pattern Queries

For immediate results, run these queries in Memgraph Lab:

```cypher
// 1. Find candidates using drug similarity
MATCH (drugA)-[treats:RELATES]->(disease),
      (drugA)-[sim:RELATES {relation: 'drug_similarity'}]->(drugB)
WHERE treats.relation = 'drug_treats_disease'
AND NOT EXISTS {(drugB)-[:RELATES {relation: 'drug_treats_disease'}]->(disease)}
RETURN drugB.node_name as Candidate, disease.node_name as Disease,
       sim.similarity_score as Score
ORDER BY Score DESC

// 2. Find candidates using disease similarity
MATCH (drug)-[treats:RELATES]->(diseaseA),
      (diseaseA)-[sim:RELATES {relation: 'disease_similarity'}]->(diseaseB)
WHERE treats.relation = 'drug_treats_disease'
AND NOT EXISTS {(drug)-[:RELATES {relation: 'drug_treats_disease'}]->(diseaseB)}
RETURN drug.node_name as Candidate, diseaseB.node_name as Disease,
       sim.similarity_score as Score
ORDER BY Score DESC
```

## Recommendations

**Start with:**
1. Pattern-based inference (Method 1) - Easy to understand and implement
2. Add path-based scoring (Method 2) - More comprehensive

**Then progress to:**
3. Feature engineering (Method 5) - Good balance of simplicity and power
4. Graph embeddings (Method 4) - State-of-the-art performance

**Choose based on:**
- **Interpretability**: Methods 1-2 (pattern-based)
- **Performance**: Methods 4-5 (ML-based)
- **Data size**: Small graphs → patterns, Large graphs → embeddings
