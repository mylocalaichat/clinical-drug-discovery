# Embedding-Based Drug Discovery with Cypher

## Overview

Instead of predicting edges, you can use **embedding similarity** to directly discover off-label drug candidates:

1. Generate embeddings for all nodes (drugs, diseases, proteins, pathways)
2. Store embeddings as node properties in Memgraph
3. Use **vector similarity search** in Cypher to find candidates
4. Query: "Which drugs have embeddings similar to this disease's embedding?"

## Why This Works Better

### Traditional Edge Prediction
- ❌ Requires features for each drug-disease pair (quadratic complexity)
- ❌ Limited to local patterns
- ❌ Needs training data (known treatments)

### Embedding-Based Discovery
- ✅ Pre-compute embeddings once (linear complexity)
- ✅ Captures global graph structure
- ✅ Works with zero-shot discovery
- ✅ Fast similarity search with vector indexing

## Approach 1: Embedding Similarity Queries

### Step 1: Generate and Store Embeddings

```python
"""
Generate embeddings and store them in Memgraph
"""
from neo4j import GraphDatabase
import numpy as np
from node2vec import Node2Vec
import networkx as nx

def load_graph_from_memgraph():
    """Load graph into NetworkX"""
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=None)
    session = driver.session()

    # Get all nodes
    result = session.run("MATCH (n) RETURN n.node_id as id, n.node_name as name, n.node_type as type")

    G = nx.Graph()
    for record in result:
        G.add_node(record['id'], name=record['name'], type=record['type'])

    # Get all edges
    result = session.run("MATCH (a)-[r]-(b) RETURN a.node_id as source, b.node_id as target")

    for record in result:
        G.add_edge(record['source'], record['target'])

    session.close()
    driver.close()

    return G

def train_embeddings(G, dimensions=128):
    """Train node2vec embeddings"""
    print("Training node2vec embeddings...")

    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=10,
        num_walks=80,
        workers=4,
        p=1,  # Return parameter
        q=1   # In-out parameter
    )

    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    return model

def store_embeddings_in_memgraph(model, G):
    """Store embeddings as node properties"""
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=None)
    session = driver.session()

    print("Storing embeddings in Memgraph...")

    for node_id in G.nodes():
        embedding = model.wv[node_id].tolist()

        # Store embedding as a list property
        session.run("""
            MATCH (n {node_id: $node_id})
            SET n.embedding = $embedding
        """, node_id=node_id, embedding=embedding)

    print(f"Stored embeddings for {len(G.nodes())} nodes")

    session.close()
    driver.close()

# Main workflow
def main():
    G = load_graph_from_memgraph()
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    model = train_embeddings(G, dimensions=128)
    store_embeddings_in_memgraph(model, G)

    print("✓ Embeddings stored! Now you can use Cypher queries.")

if __name__ == "__main__":
    main()
```

### Step 2: Query with Cypher - Cosine Similarity

```cypher
// Find drugs with embeddings similar to a specific disease

// First, get the target disease embedding
MATCH (target_disease:Node {node_name: 'Metabolic Syndrome'})
WHERE target_disease.node_type = 'disease'

// Find all drugs
MATCH (drug:Node)
WHERE drug.node_type = 'drug'

// Calculate cosine similarity between embeddings
WITH target_disease, drug,
     gds.similarity.cosine(target_disease.embedding, drug.embedding) as similarity

// Exclude drugs that already treat this disease
WHERE NOT EXISTS {
    MATCH (drug)-[r:RELATES]->(target_disease)
    WHERE r.relation = 'drug_treats_disease'
}

RETURN drug.node_name as CandidateDrug,
       target_disease.node_name as TargetDisease,
       similarity as EmbeddingSimilarity
ORDER BY similarity DESC
LIMIT 10
```

## Approach 2: Disease Context Embedding

Instead of comparing drug to disease directly, compare drug to the "disease context":

```cypher
// Find drugs similar to the neighborhood of a disease

// Step 1: Get disease and its immediate context (proteins, pathways)
MATCH (disease:Node {node_name: 'Type 2 Diabetes'})-[:RELATES]-(context)
WHERE disease.node_type = 'disease'
AND context.node_type IN ['gene/protein', 'pathway']

// Calculate average context embedding
WITH disease,
     context,
     avg([node IN collect(context) | node.embedding]) as context_embedding

// Step 2: Find drugs similar to this context
MATCH (drug:Node)
WHERE drug.node_type = 'drug'

WITH drug, disease, context_embedding,
     gds.similarity.cosine(drug.embedding, context_embedding) as similarity

WHERE NOT EXISTS {
    MATCH (drug)-[r:RELATES]->(disease)
    WHERE r.relation = 'drug_treats_disease'
}

RETURN drug.node_name as CandidateDrug,
       disease.node_name as TargetDisease,
       similarity as ContextSimilarity
ORDER BY similarity DESC
LIMIT 10
```

## Approach 3: Embedding Arithmetic

Use embedding arithmetic like: `drug_embedding ≈ disease_embedding - symptom_embedding + treatment_embedding`

```cypher
// Drug discovery using embedding arithmetic
// Find: Drug that is to Disease B, what Known Drug is to Disease A

MATCH (diseaseA:Node {node_name: 'Type 2 Diabetes'})<-[treats:RELATES]-(knownDrug:Node)
WHERE diseaseA.node_type = 'disease'
AND knownDrug.node_type = 'drug'
AND treats.relation = 'drug_treats_disease'

MATCH (diseaseB:Node {node_name: 'Metabolic Syndrome'})
WHERE diseaseB.node_type = 'disease'

// Calculate target embedding: diseaseB + (knownDrug - diseaseA)
// This represents "what drug treats diseaseB like knownDrug treats diseaseA"
WITH diseaseB, knownDrug, diseaseA,
     [i IN range(0, size(diseaseB.embedding)-1) |
         diseaseB.embedding[i] + knownDrug.embedding[i] - diseaseA.embedding[i]
     ] as target_embedding

// Find drugs closest to this target
MATCH (candidateDrug:Node)
WHERE candidateDrug.node_type = 'drug'
AND candidateDrug.node_id <> knownDrug.node_id

WITH candidateDrug, diseaseB, knownDrug, diseaseA, target_embedding,
     gds.similarity.cosine(candidateDrug.embedding, target_embedding) as similarity

RETURN candidateDrug.node_name as CandidateDrug,
       diseaseB.node_name as TargetDisease,
       knownDrug.node_name as AnalogousTo,
       diseaseA.node_name as OriginalDisease,
       similarity as AnalogySimilarity
ORDER BY similarity DESC
LIMIT 10
```

## Approach 4: Multi-Hop Embedding Queries

Find drugs that target proteins similar to disease-associated proteins:

```cypher
// Find drugs that target proteins with embeddings similar to disease-associated proteins

MATCH (disease:Node {node_name: 'Cardiovascular Disease'})-[:RELATES]->(diseaseProtein:Node)
WHERE disease.node_type = 'disease'
AND diseaseProtein.node_type = 'gene/protein'

// For each disease protein, find similar proteins
MATCH (targetProtein:Node)
WHERE targetProtein.node_type = 'gene/protein'
AND targetProtein.node_id <> diseaseProtein.node_id

WITH disease, diseaseProtein, targetProtein,
     gds.similarity.cosine(diseaseProtein.embedding, targetProtein.embedding) as proteinSimilarity
WHERE proteinSimilarity > 0.7

// Find drugs that target these similar proteins
MATCH (drug:Node)-[targets:RELATES]->(targetProtein)
WHERE drug.node_type = 'drug'
AND targets.relation = 'drug_targets_protein'

// Check drug doesn't already treat the disease
WHERE NOT EXISTS {
    MATCH (drug)-[r:RELATES]->(disease)
    WHERE r.relation = 'drug_treats_disease'
}

RETURN drug.node_name as CandidateDrug,
       disease.node_name as TargetDisease,
       targetProtein.node_name as TargetProtein,
       diseaseProtein.node_name as DiseaseProtein,
       proteinSimilarity as ProteinSimilarity
ORDER BY proteinSimilarity DESC
LIMIT 10
```

## Approach 5: Embedding-Based Clustering

Find disease clusters and drugs that treat similar diseases:

```cypher
// Find drugs that treat diseases in the same embedding cluster

MATCH (targetDisease:Node {node_name: 'Metabolic Syndrome'})
WHERE targetDisease.node_type = 'disease'

// Find similar diseases
MATCH (similarDisease:Node)
WHERE similarDisease.node_type = 'disease'
AND similarDisease.node_id <> targetDisease.node_id

WITH targetDisease, similarDisease,
     gds.similarity.cosine(targetDisease.embedding, similarDisease.embedding) as diseaseSimilarity
WHERE diseaseSimilarity > 0.8

// Find drugs that treat these similar diseases
MATCH (drug:Node)-[treats:RELATES]->(similarDisease)
WHERE drug.node_type = 'drug'
AND treats.relation = 'drug_treats_disease'

// Check drug doesn't treat target disease yet
WHERE NOT EXISTS {
    MATCH (drug)-[r:RELATES]->(targetDisease)
    WHERE r.relation = 'drug_treats_disease'
}

RETURN drug.node_name as CandidateDrug,
       targetDisease.node_name as TargetDisease,
       collect(DISTINCT similarDisease.node_name) as TreatsDiseasesIn,
       avg(diseaseSimilarity) as AvgDiseaseSimilarity
ORDER BY AvgDiseaseSimilarity DESC
LIMIT 10
```

## Complete Python Implementation

```python
"""
Embedding-based drug discovery with direct Cypher queries
"""
from neo4j import GraphDatabase
import numpy as np


class EmbeddingDrugDiscovery:
    def __init__(self, uri='bolt://localhost:7687'):
        self.driver = GraphDatabase.driver(uri, auth=None)

    def close(self):
        self.driver.close()

    def cosine_similarity(self, emb1, emb2):
        """Calculate cosine similarity"""
        dot = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(a * a for a in emb1) ** 0.5
        norm2 = sum(b * b for b in emb2) ** 0.5
        return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0

    def find_similar_drugs(self, disease_name, top_k=10):
        """Find drugs with embeddings similar to a disease"""
        with self.driver.session() as session:
            # Get disease embedding
            result = session.run("""
                MATCH (d:Node {node_name: $disease_name})
                WHERE d.node_type = 'disease'
                RETURN d.embedding as embedding
            """, disease_name=disease_name)

            disease_emb = result.single()['embedding']

            # Get all drugs
            result = session.run("""
                MATCH (drug:Node)
                WHERE drug.node_type = 'drug'
                AND EXISTS(drug.embedding)
                RETURN drug.node_id as id,
                       drug.node_name as name,
                       drug.embedding as embedding
            """)

            candidates = []
            for record in result:
                drug_emb = record['embedding']
                similarity = self.cosine_similarity(disease_emb, drug_emb)

                # Check if already treats disease
                exists = session.run("""
                    MATCH (drug {node_id: $drug_id})-[r:RELATES]->(disease {node_name: $disease_name})
                    WHERE r.relation = 'drug_treats_disease'
                    RETURN count(r) > 0 as exists
                """, drug_id=record['id'], disease_name=disease_name).single()['exists']

                if not exists:
                    candidates.append({
                        'drug': record['name'],
                        'similarity': similarity
                    })

            # Sort and return top k
            candidates.sort(key=lambda x: x['similarity'], reverse=True)
            return candidates[:top_k]

    def find_by_analogy(self, disease_a, disease_b, known_drug, top_k=10):
        """
        Find drug that is to disease_b what known_drug is to disease_a
        Analogy: disease_a:known_drug :: disease_b:?
        """
        with self.driver.session() as session:
            # Get embeddings
            result = session.run("""
                MATCH (da:Node {node_name: $disease_a}),
                      (db:Node {node_name: $disease_b}),
                      (kd:Node {node_name: $known_drug})
                RETURN da.embedding as da_emb,
                       db.embedding as db_emb,
                       kd.embedding as kd_emb
            """, disease_a=disease_a, disease_b=disease_b, known_drug=known_drug)

            record = result.single()
            da_emb = record['da_emb']
            db_emb = record['db_emb']
            kd_emb = record['kd_emb']

            # Calculate target: db + (kd - da)
            target_emb = [
                db_emb[i] + kd_emb[i] - da_emb[i]
                for i in range(len(db_emb))
            ]

            # Find closest drugs
            result = session.run("""
                MATCH (drug:Node)
                WHERE drug.node_type = 'drug'
                AND EXISTS(drug.embedding)
                AND drug.node_name <> $known_drug
                RETURN drug.node_name as name,
                       drug.embedding as embedding
            """, known_drug=known_drug)

            candidates = []
            for record in result:
                drug_emb = record['embedding']
                similarity = self.cosine_similarity(target_emb, drug_emb)
                candidates.append({
                    'drug': record['name'],
                    'similarity': similarity,
                    'analogy': f"{disease_a}:{known_drug} :: {disease_b}:{record['name']}"
                })

            candidates.sort(key=lambda x: x['similarity'], reverse=True)
            return candidates[:top_k]

    def find_by_protein_similarity(self, disease_name, protein_sim_threshold=0.7, top_k=10):
        """
        Find drugs targeting proteins similar to disease-associated proteins
        """
        with self.driver.session() as session:
            # Get disease-associated proteins
            result = session.run("""
                MATCH (disease:Node {node_name: $disease_name})-[:RELATES]->(protein:Node)
                WHERE disease.node_type = 'disease'
                AND protein.node_type = 'gene/protein'
                AND EXISTS(protein.embedding)
                RETURN protein.node_id as id,
                       protein.node_name as name,
                       protein.embedding as embedding
            """, disease_name=disease_name)

            disease_proteins = list(result)

            # For each disease protein, find similar proteins and their drugs
            candidates = {}

            for dp in disease_proteins:
                # Find similar proteins
                result = session.run("""
                    MATCH (tp:Node)
                    WHERE tp.node_type = 'gene/protein'
                    AND EXISTS(tp.embedding)
                    AND tp.node_id <> $dp_id
                    RETURN tp.node_id as id,
                           tp.node_name as name,
                           tp.embedding as embedding
                """, dp_id=dp['id'])

                target_proteins = []
                for record in result:
                    similarity = self.cosine_similarity(dp['embedding'], record['embedding'])
                    if similarity > protein_sim_threshold:
                        target_proteins.append({
                            'id': record['id'],
                            'name': record['name'],
                            'similarity': similarity
                        })

                # Find drugs targeting these proteins
                for tp in target_proteins:
                    result = session.run("""
                        MATCH (drug:Node)-[r:RELATES]->(tp {node_id: $tp_id})
                        WHERE drug.node_type = 'drug'
                        AND r.relation = 'drug_targets_protein'
                        RETURN drug.node_name as name
                    """, tp_id=tp['id'])

                    for record in result:
                        drug_name = record['name']
                        if drug_name not in candidates:
                            candidates[drug_name] = {
                                'drug': drug_name,
                                'max_protein_similarity': tp['similarity'],
                                'evidence': []
                            }

                        candidates[drug_name]['evidence'].append({
                            'disease_protein': dp['name'],
                            'target_protein': tp['name'],
                            'similarity': tp['similarity']
                        })

                        candidates[drug_name]['max_protein_similarity'] = max(
                            candidates[drug_name]['max_protein_similarity'],
                            tp['similarity']
                        )

            # Sort and return top k
            result = sorted(candidates.values(),
                          key=lambda x: x['max_protein_similarity'],
                          reverse=True)
            return result[:top_k]


# Usage example
def main():
    discovery = EmbeddingDrugDiscovery()

    print("Method 1: Direct embedding similarity")
    print("=" * 80)
    candidates = discovery.find_similar_drugs('Metabolic Syndrome', top_k=5)
    for i, c in enumerate(candidates, 1):
        print(f"{i}. {c['drug']} (similarity: {c['similarity']:.3f})")

    print("\nMethod 2: Analogy-based discovery")
    print("=" * 80)
    candidates = discovery.find_by_analogy(
        'Type 2 Diabetes',
        'Metabolic Syndrome',
        'Metformin',
        top_k=5
    )
    for i, c in enumerate(candidates, 1):
        print(f"{i}. {c['drug']} (similarity: {c['similarity']:.3f})")
        print(f"   Analogy: {c['analogy']}")

    print("\nMethod 3: Protein similarity-based")
    print("=" * 80)
    candidates = discovery.find_by_protein_similarity('Cardiovascular Disease', top_k=5)
    for i, c in enumerate(candidates, 1):
        print(f"{i}. {c['drug']} (max similarity: {c['max_protein_similarity']:.3f})")
        print(f"   Evidence: {len(c['evidence'])} protein connections")

    discovery.close()


if __name__ == "__main__":
    main()
```

## Comparison: Edge Prediction vs Embedding Queries

| Aspect | Edge Prediction | Embedding Queries |
|--------|----------------|-------------------|
| **Speed** | Slow (needs to compute features for all pairs) | Fast (pre-computed embeddings) |
| **Scalability** | O(n²) for n drugs | O(n) with vector index |
| **Interpretability** | High (explicit patterns) | Lower (learned representations) |
| **Novelty** | Limited to local patterns | Can find novel connections |
| **Zero-shot** | Needs training data | Works immediately |
| **Query flexibility** | Limited | Very flexible (analogies, arithmetic) |

## Best Practice: Combine Both Approaches

```cypher
// Hybrid approach: Use embeddings to rank, patterns to explain

// Step 1: Get embedding-based candidates
MATCH (disease:Node {node_name: 'Metabolic Syndrome'})
WHERE disease.node_type = 'disease'

MATCH (drug:Node)
WHERE drug.node_type = 'drug'

WITH drug, disease,
     gds.similarity.cosine(drug.embedding, disease.embedding) as emb_similarity
WHERE emb_similarity > 0.6

// Step 2: Check for supporting patterns
OPTIONAL MATCH (drug)-[drugSim:RELATES {relation: 'drug_similarity'}]->(otherDrug)
              -[treats:RELATES {relation: 'drug_treats_disease'}]->(simDisease)
              -[diseaseSim:RELATES {relation: 'disease_similarity'}]->(disease)

// Step 3: Combine scores
WITH drug, disease,
     emb_similarity,
     count(DISTINCT otherDrug) as pattern_support,
     avg(drugSim.similarity_score) as avg_drug_sim,
     avg(diseaseSim.similarity_score) as avg_disease_sim

// Final score: 60% embedding, 40% patterns
WITH drug, disease,
     (0.6 * emb_similarity +
      0.4 * coalesce(avg_drug_sim, 0) * coalesce(avg_disease_sim, 0)) as final_score,
     emb_similarity,
     pattern_support

RETURN drug.node_name as CandidateDrug,
       disease.node_name as TargetDisease,
       round(final_score * 1000) / 1000 as FinalScore,
       round(emb_similarity * 1000) / 1000 as EmbeddingScore,
       pattern_support as PatternSupport
ORDER BY final_score DESC
LIMIT 20
```

## Key Advantages

1. **No training needed** - Embeddings work zero-shot
2. **Fast queries** - Single Cypher query, no Python loop
3. **Flexible** - Can do analogies, arithmetic, clustering
4. **Scalable** - Works on large graphs with vector indexing
5. **Interactive** - Query results instantly in Memgraph Lab

## Next Steps

1. Generate embeddings for your full graph (129K nodes)
2. Store embeddings as node properties
3. Create vector index for fast similarity search
4. Query with Cypher for instant drug discovery!
