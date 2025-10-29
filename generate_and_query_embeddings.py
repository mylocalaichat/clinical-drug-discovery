"""
Generate embeddings and use them for drug discovery via Cypher queries

This script:
1. Generates node2vec embeddings for all nodes in the graph
2. Stores embeddings as node properties in Memgraph
3. Uses Cypher queries for embedding-based drug discovery
"""

from neo4j import GraphDatabase
import numpy as np
from node2vec import Node2Vec
import networkx as nx


class EmbeddingDrugDiscovery:
    def __init__(self, uri='bolt://localhost:7687'):
        self.driver = GraphDatabase.driver(uri, auth=None)
        self.session = self.driver.session()

    def close(self):
        self.session.close()
        self.driver.close()

    def load_graph_from_memgraph(self):
        """Load graph into NetworkX"""
        print("Loading graph from Memgraph...")

        # Get all example nodes
        result = self.session.run("""
            MATCH (n {is_example: true})
            RETURN n.node_id as id, n.node_name as name, n.node_type as type
        """)

        G = nx.Graph()
        nodes = list(result)
        for record in nodes:
            G.add_node(record['id'], name=record['name'], type=record['type'])

        print(f"  Loaded {len(nodes)} nodes")

        # Get all edges
        result = self.session.run("""
            MATCH (a {is_example: true})-[r {is_example: true}]-(b {is_example: true})
            RETURN DISTINCT a.node_id as source, b.node_id as target
        """)

        edges = list(result)
        for record in edges:
            G.add_edge(record['source'], record['target'])

        print(f"  Loaded {len(edges)} edges")

        return G

    def train_embeddings(self, G, dimensions=64):
        """Train node2vec embeddings"""
        print(f"\nTraining node2vec embeddings (dim={dimensions})...")

        node2vec = Node2Vec(
            G,
            dimensions=dimensions,
            walk_length=10,
            num_walks=80,
            workers=1,
            p=1,
            q=1
        )

        model = node2vec.fit(window=10, min_count=1, batch_words=4, epochs=5)
        print("  ✓ Embeddings trained")

        return model

    def store_embeddings(self, model, G):
        """Store embeddings as node properties"""
        print("\nStoring embeddings in Memgraph...")

        count = 0
        for node_id in G.nodes():
            embedding = model.wv[node_id].tolist()

            self.session.run("""
                MATCH (n {node_id: $node_id})
                SET n.embedding = $embedding
            """, node_id=node_id, embedding=embedding)
            count += 1

        print(f"  ✓ Stored embeddings for {count} nodes")

    def cosine_similarity_cypher(self, emb1, emb2):
        """Calculate cosine similarity in Python (Memgraph doesn't have built-in)"""
        dot = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(a * a for a in emb1) ** 0.5
        norm2 = sum(b * b for b in emb2) ** 0.5
        return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0

    def discover_by_embedding_similarity(self, disease_name):
        """Method 1: Find drugs with embeddings similar to a disease"""
        print(f"\n{'='*80}")
        print(f"METHOD 1: Direct Embedding Similarity")
        print(f"Target Disease: {disease_name}")
        print(f"{'='*80}")

        # Get disease embedding
        result = self.session.run("""
            MATCH (d:Node {node_name: $disease_name})
            WHERE d.node_type = 'disease' AND d.is_example = true
            RETURN d.embedding as embedding
        """, disease_name=disease_name)

        record = result.single()
        if not record or not record['embedding']:
            print(f"  ✗ Disease '{disease_name}' not found or has no embedding")
            return []

        disease_emb = record['embedding']

        # Get all drugs and calculate similarity
        result = self.session.run("""
            MATCH (drug:Node)
            WHERE drug.node_type = 'drug'
            AND drug.is_example = true
            AND EXISTS(drug.embedding)
            RETURN drug.node_id as id,
                   drug.node_name as name,
                   drug.embedding as embedding
        """)

        candidates = []
        for record in result:
            drug_emb = record['embedding']
            similarity = self.cosine_similarity_cypher(disease_emb, drug_emb)

            # Check if already treats disease
            exists = self.session.run("""
                MATCH (drug {node_id: $drug_id})-[r:RELATES]->(disease {node_name: $disease_name})
                WHERE r.relation = 'drug_treats_disease'
                RETURN count(r) > 0 as exists
            """, drug_id=record['id'], disease_name=disease_name).single()['exists']

            if not exists:
                candidates.append({
                    'drug': record['name'],
                    'similarity': similarity
                })

        candidates.sort(key=lambda x: x['similarity'], reverse=True)

        print(f"\nTop candidates (not already treating {disease_name}):")
        for i, c in enumerate(candidates[:5], 1):
            print(f"  {i}. {c['drug']:20} similarity: {c['similarity']:.4f}")

        return candidates

    def discover_by_analogy(self, disease_a, disease_b, known_drug):
        """Method 2: Analogy-based discovery"""
        print(f"\n{'='*80}")
        print(f"METHOD 2: Analogy-Based Discovery")
        print(f"Analogy: {disease_a}:{known_drug} :: {disease_b}:?")
        print(f"{'='*80}")

        # Get embeddings
        result = self.session.run("""
            MATCH (da:Node {node_name: $disease_a}),
                  (db:Node {node_name: $disease_b}),
                  (kd:Node {node_name: $known_drug})
            WHERE da.is_example = true
            AND db.is_example = true
            AND kd.is_example = true
            RETURN da.embedding as da_emb,
                   db.embedding as db_emb,
                   kd.embedding as kd_emb
        """, disease_a=disease_a, disease_b=disease_b, known_drug=known_drug)

        record = result.single()
        if not record:
            print("  ✗ One or more entities not found")
            return []

        da_emb = record['da_emb']
        db_emb = record['db_emb']
        kd_emb = record['kd_emb']

        # Calculate target: db + (kd - da)
        target_emb = [
            db_emb[i] + kd_emb[i] - da_emb[i]
            for i in range(len(db_emb))
        ]

        # Find closest drugs
        result = self.session.run("""
            MATCH (drug:Node)
            WHERE drug.node_type = 'drug'
            AND drug.is_example = true
            AND EXISTS(drug.embedding)
            AND drug.node_name <> $known_drug
            RETURN drug.node_name as name,
                   drug.embedding as embedding
        """, known_drug=known_drug)

        candidates = []
        for record in result:
            drug_emb = record['embedding']
            similarity = self.cosine_similarity_cypher(target_emb, drug_emb)
            candidates.append({
                'drug': record['name'],
                'similarity': similarity
            })

        candidates.sort(key=lambda x: x['similarity'], reverse=True)

        print(f"\nTop candidates by analogy:")
        for i, c in enumerate(candidates[:5], 1):
            print(f"  {i}. {c['drug']:20} similarity: {c['similarity']:.4f}")

        return candidates

    def discover_by_disease_context(self, disease_name):
        """Method 3: Find drugs similar to disease's protein/pathway context"""
        print(f"\n{'='*80}")
        print(f"METHOD 3: Disease Context Similarity")
        print(f"Target Disease: {disease_name}")
        print(f"{'='*80}")

        # Get disease and its context embeddings
        result = self.session.run("""
            MATCH (disease:Node {node_name: $disease_name})-[:RELATES]-(context)
            WHERE disease.node_type = 'disease'
            AND disease.is_example = true
            AND context.node_type IN ['gene/protein', 'pathway']
            AND EXISTS(context.embedding)
            RETURN collect(context.embedding) as context_embeddings,
                   collect(context.node_name) as context_names
        """, disease_name=disease_name)

        record = result.single()
        if not record or not record['context_embeddings']:
            print(f"  ✗ No context found for {disease_name}")
            return []

        context_embeddings = record['context_embeddings']
        context_names = record['context_names']

        print(f"  Disease context: {', '.join(context_names)}")

        # Calculate average context embedding
        avg_context_emb = [
            sum(emb[i] for emb in context_embeddings) / len(context_embeddings)
            for i in range(len(context_embeddings[0]))
        ]

        # Find drugs similar to context
        result = self.session.run("""
            MATCH (drug:Node)
            WHERE drug.node_type = 'drug'
            AND drug.is_example = true
            AND EXISTS(drug.embedding)
            RETURN drug.node_id as id,
                   drug.node_name as name,
                   drug.embedding as embedding
        """)

        candidates = []
        for record in result:
            drug_emb = record['embedding']
            similarity = self.cosine_similarity_cypher(avg_context_emb, drug_emb)

            # Check if already treats disease
            exists = self.session.run("""
                MATCH (drug {node_id: $drug_id})-[r:RELATES]->(disease {node_name: $disease_name})
                WHERE r.relation = 'drug_treats_disease'
                RETURN count(r) > 0 as exists
            """, drug_id=record['id'], disease_name=disease_name).single()['exists']

            if not exists:
                candidates.append({
                    'drug': record['name'],
                    'similarity': similarity
                })

        candidates.sort(key=lambda x: x['similarity'], reverse=True)

        print(f"\nTop candidates by context similarity:")
        for i, c in enumerate(candidates[:5], 1):
            print(f"  {i}. {c['drug']:20} similarity: {c['similarity']:.4f}")

        return candidates


def main():
    print("="*80)
    print("EMBEDDING-BASED DRUG DISCOVERY")
    print("="*80)

    discovery = EmbeddingDrugDiscovery()

    # Step 1: Load graph
    G = discovery.load_graph_from_memgraph()

    if G.number_of_nodes() == 0:
        print("\n⚠️  No example data found!")
        print("Please run: python3 load_example_progressive.py")
        discovery.close()
        return

    # Step 2: Train embeddings
    model = discovery.train_embeddings(G, dimensions=64)

    # Step 3: Store embeddings
    discovery.store_embeddings(model, G)

    print("\n" + "="*80)
    print("RUNNING DISCOVERY QUERIES")
    print("="*80)

    # Method 1: Direct embedding similarity
    discovery.discover_by_embedding_similarity('Metabolic Syndrome')

    # Method 2: Analogy-based
    discovery.discover_by_analogy(
        'Type 2 Diabetes',
        'Metabolic Syndrome',
        'Metformin'
    )

    # Method 3: Context similarity
    discovery.discover_by_disease_context('Cardiovascular Disease')

    print("\n" + "="*80)
    print("DISCOVERY COMPLETE")
    print("="*80)
    print("\nEmbeddings are now stored in Memgraph!")
    print("You can query them directly in Cypher:")
    print("""
Example query:
    MATCH (disease:Node {node_name: 'Metabolic Syndrome'})
    MATCH (drug:Node)
    WHERE drug.node_type = 'drug'
    AND EXISTS(drug.embedding)
    RETURN drug.node_name, drug.embedding[0..5] as embedding_preview
    """)

    discovery.close()


if __name__ == "__main__":
    main()
