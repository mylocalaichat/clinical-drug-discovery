"""
Progressive Loading Demo for Drug Repurposing Example

This script demonstrates the power of similarity relationships by:
1. First loading the base graph (drugs, diseases, proteins, pathways and their interactions)
2. Then adding similarity edges to show how they enable drug repurposing inference
"""

from neo4j import GraphDatabase
import re


def execute_cypher_file(session, filename, description):
    """Execute a cypher file statement by statement"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")

    with open(filename, 'r') as f:
        content = f.read()

    # Extract CREATE statements using regex
    create_pattern = r'CREATE\s+\([^)]+:[^)]+\{[^}]+\}\);'
    match_pattern = r'MATCH.*?(?:CREATE|MERGE).*?(?:\{[^}]+\}\))?(?:->\([^)]*\))?;'

    node_creates = re.findall(create_pattern, content, re.DOTALL)
    rel_creates = re.findall(match_pattern, content, re.DOTALL)

    print(f"Found {len(node_creates)} node CREATE statements")
    print(f"Found {len(rel_creates)} relationship CREATE statements")

    # Execute node creates first
    for i, statement in enumerate(node_creates, 1):
        statement = statement.strip().rstrip(';')
        try:
            session.run(statement)
            print(f"✓ Node {i}/{len(node_creates)} created")
        except Exception as e:
            print(f"✗ Node {i} failed: {e}")

    # Then execute relationship creates
    for i, statement in enumerate(rel_creates, 1):
        statement = statement.strip().rstrip(';')
        try:
            session.run(statement)
            print(f"✓ Relationship {i}/{len(rel_creates)} created")
        except Exception as e:
            print(f"✗ Relationship {i} failed: {e}")

    return len(node_creates), len(rel_creates)


def count_example_data(session):
    """Count example nodes and relationships"""
    result = session.run("MATCH (n {is_example: true}) RETURN count(n) as count")
    node_count = result.single()['count']

    result = session.run("MATCH ()-[r {is_example: true}]->() RETURN count(r) as count")
    rel_count = result.single()['count']

    return node_count, rel_count


def test_inference_query(session):
    """Test if the drug repurposing inference query works"""
    print("\n" + "="*80)
    print("TESTING DRUG REPURPOSING INFERENCE")
    print("="*80)

    query = """
        MATCH (drugA {is_example: true})-[treats:RELATES {is_example: true}]->(diseaseB {is_example: true}),
              (drugE {is_example: true})-[:RELATES {is_example: true}]->(diseaseC {is_example: true}),
              (drugA)-[drugSim:RELATES {is_example: true}]->(drugE),
              (diseaseB)-[diseaseSim1:RELATES {is_example: true}]->(diseaseD {is_example: true}),
              (diseaseC)-[diseaseSim2:RELATES {is_example: true}]->(diseaseD)
        WHERE treats.relation = 'drug_treats_disease'
        AND drugSim.relation = 'drug_similarity'
        AND diseaseSim1.relation = 'disease_similarity'
        AND diseaseSim2.relation = 'disease_similarity'
        RETURN drugA.node_name as CandidateDrug,
               diseaseD.node_name as TargetDisease,
               drugSim.similarity_score as DrugSimilarity,
               diseaseSim1.similarity_score as DiseaseSim1,
               diseaseSim2.similarity_score as DiseaseSim2
    """

    result = session.run(query)
    records = list(result)

    if records:
        print("\n✓ INFERENCE SUCCESSFUL!")
        for record in records:
            print(f"\nCandidate Drug: {record['CandidateDrug']}")
            print(f"Target Disease: {record['TargetDisease']}")
            print(f"Drug Similarity: {record['DrugSimilarity']}")
            print(f"Disease Similarity 1: {record['DiseaseSim1']}")
            print(f"Disease Similarity 2: {record['DiseaseSim2']}")
    else:
        print("\n✗ NO INFERENCE FOUND (similarity edges may be missing)")

    return len(records) > 0


def main():
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=None)
    session = driver.session()

    print("\n" + "="*80)
    print("PROGRESSIVE DRUG REPURPOSING EXAMPLE LOADING")
    print("="*80)
    print("\nThis demo shows how similarity edges enable drug repurposing inference")

    # Step 0: Clean up any existing example data
    print("\n" + "="*80)
    print("STEP 0: Cleaning up existing example data")
    print("="*80)
    session.run("MATCH (n {is_example: true}) DETACH DELETE n")
    print("✓ Cleanup complete")

    # Step 1: Load base graph
    nodes1, rels1 = execute_cypher_file(
        session,
        'example_drug_repurposing_base.cypher',
        'STEP 1: Loading Base Graph (drugs, diseases, proteins, pathways)'
    )

    node_count, rel_count = count_example_data(session)
    print(f"\nAfter Step 1: {node_count} nodes, {rel_count} relationships")

    # Test inference (should fail without similarities)
    print("\n" + "="*80)
    print("Testing inference WITHOUT similarity edges...")
    print("="*80)
    inference_works = test_inference_query(session)
    if not inference_works:
        print("✓ Expected: Inference doesn't work without similarity edges")

    # Step 2: Add similarity edges
    nodes2, rels2 = execute_cypher_file(
        session,
        'example_drug_repurposing_similarities.cypher',
        'STEP 2: Adding Similarity Edges (drug-drug, disease-disease similarities)'
    )

    node_count, rel_count = count_example_data(session)
    print(f"\nAfter Step 2: {node_count} nodes, {rel_count} relationships")

    # Test inference (should work with similarities)
    print("\n" + "="*80)
    print("Testing inference WITH similarity edges...")
    print("="*80)
    inference_works = test_inference_query(session)
    if inference_works:
        print("\n✓ Success! Similarity edges enable drug repurposing inference!")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Base graph loaded: {nodes1} nodes, {rels1-nodes1} relationships")
    print(f"Similarity edges added: {rels2} relationships")
    print(f"Total: {node_count} nodes, {rel_count} relationships")
    print("\nKey Insight:")
    print("The similarity edges (purple dashed lines in visualization) are what enable")
    print("the graph to infer new drug-disease relationships through pattern matching!")
    print("\nVisualze in Memgraph Lab with:")
    print("  MATCH (n {is_example: true})-[r {is_example: true}]-(m {is_example: true})")
    print("  RETURN n, r, m")
    print("="*80)

    session.close()
    driver.close()


if __name__ == "__main__":
    main()
