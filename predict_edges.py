"""
Edge Prediction for Drug Repurposing

This script predicts new drug-disease relationships using pattern-based inference
and path scoring methods on your existing Memgraph data.
"""

from neo4j import GraphDatabase
from collections import defaultdict


def pattern_based_inference(session):
    """Method 1: Pattern-based inference using similarities"""
    print("\n" + "="*80)
    print("METHOD 1: PATTERN-BASED INFERENCE")
    print("="*80)

    predictions = []

    # Pattern 1: Drug similarity
    print("\nPattern 1: Drug Similarity Inference")
    print("-" * 80)
    query1 = """
        MATCH (drugA)-[treats:RELATES]->(disease),
              (drugA)-[sim:RELATES]->(drugB)
        WHERE drugA.node_type = 'drug'
        AND disease.node_type = 'disease'
        AND drugB.node_type = 'drug'
        AND treats.relation = 'drug_treats_disease'
        AND sim.relation = 'drug_similarity'
        AND sim.similarity_score > 0.5

        AND NOT EXISTS {
            MATCH (drugB)-[r:RELATES]->(disease)
            WHERE r.relation = 'drug_treats_disease'
        }

        RETURN DISTINCT
            drugB.node_name as candidate_drug,
            disease.node_name as target_disease,
            drugA.node_name as similar_drug,
            sim.similarity_score as drug_similarity,
            'drug_similarity' as method
        ORDER BY sim.similarity_score DESC
    """

    result = session.run(query1)
    for record in result:
        predictions.append({
            'candidate_drug': record['candidate_drug'],
            'target_disease': record['target_disease'],
            'method': 'drug_similarity',
            'score': record['drug_similarity'],
            'evidence': f"Similar to {record['similar_drug']}"
        })
        print(f"  {record['candidate_drug']} -> {record['target_disease']}")
        print(f"    Similar to {record['similar_drug']} (score: {record['drug_similarity']:.3f})")

    # Pattern 2: Disease similarity
    print("\nPattern 2: Disease Similarity Inference")
    print("-" * 80)
    query2 = """
        MATCH (drug)-[treats:RELATES]->(diseaseA),
              (diseaseA)-[sim:RELATES]->(diseaseB)
        WHERE drug.node_type = 'drug'
        AND diseaseA.node_type = 'disease'
        AND diseaseB.node_type = 'disease'
        AND treats.relation = 'drug_treats_disease'
        AND sim.relation = 'disease_similarity'
        AND sim.similarity_score > 0.6

        AND NOT EXISTS {
            MATCH (drug)-[r:RELATES]->(diseaseB)
            WHERE r.relation = 'drug_treats_disease'
        }

        RETURN DISTINCT
            drug.node_name as candidate_drug,
            diseaseB.node_name as target_disease,
            diseaseA.node_name as similar_disease,
            sim.similarity_score as disease_similarity,
            'disease_similarity' as method
        ORDER BY sim.similarity_score DESC
    """

    result = session.run(query2)
    for record in result:
        predictions.append({
            'candidate_drug': record['candidate_drug'],
            'target_disease': record['target_disease'],
            'method': 'disease_similarity',
            'score': record['disease_similarity'],
            'evidence': f"Treats similar disease: {record['similar_disease']}"
        })
        print(f"  {record['candidate_drug']} -> {record['target_disease']}")
        print(f"    Treats {record['similar_disease']} (score: {record['disease_similarity']:.3f})")

    # Pattern 3: Combined (Triangle)
    print("\nPattern 3: Combined Similarity Inference (Triangle)")
    print("-" * 80)
    query3 = """
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
        AND drugSim.similarity_score > 0.5
        AND diseaseSim.similarity_score > 0.6

        AND NOT EXISTS {
            MATCH (drugB)-[r:RELATES]->(diseaseY)
            WHERE r.relation = 'drug_treats_disease'
        }

        RETURN DISTINCT
            drugB.node_name as candidate_drug,
            diseaseY.node_name as target_disease,
            drugA.node_name as similar_drug,
            diseaseX.node_name as similar_disease,
            drugSim.similarity_score as drug_sim,
            diseaseSim.similarity_score as disease_sim,
            (drugSim.similarity_score + diseaseSim.similarity_score) / 2.0 as avg_score
        ORDER BY avg_score DESC
    """

    result = session.run(query3)
    for record in result:
        avg_score = (record['drug_sim'] + record['disease_sim']) / 2.0
        predictions.append({
            'candidate_drug': record['candidate_drug'],
            'target_disease': record['target_disease'],
            'method': 'combined_similarity',
            'score': avg_score,
            'evidence': f"Similar drug: {record['similar_drug']}, Similar disease: {record['similar_disease']}"
        })
        print(f"  {record['candidate_drug']} -> {record['target_disease']}")
        print(f"    Similar drug: {record['similar_drug']} (sim: {record['drug_sim']:.3f})")
        print(f"    Similar disease: {record['similar_disease']} (sim: {record['disease_sim']:.3f})")
        print(f"    Average score: {avg_score:.3f}")

    return predictions


def path_based_scoring(session):
    """Method 2: Score based on connecting paths"""
    print("\n" + "="*80)
    print("METHOD 2: PATH-BASED SCORING")
    print("="*80)

    query = """
        MATCH path = (drug)-[*2..4]-(disease)
        WHERE drug.node_type = 'drug'
        AND disease.node_type = 'disease'

        AND NOT EXISTS {
            MATCH (drug)-[r:RELATES]->(disease)
            WHERE r.relation = 'drug_treats_disease'
        }

        WITH drug, disease,
             count(path) as path_count,
             avg(length(path)) as avg_path_length

        WITH drug, disease, path_count, avg_path_length,
             path_count * (1.0 / avg_path_length) as path_score

        WHERE path_count > 0

        RETURN drug.node_name as candidate_drug,
               disease.node_name as target_disease,
               path_count as num_paths,
               round(avg_path_length * 100) / 100 as avg_path_length,
               round(path_score * 100) / 100 as score
        ORDER BY score DESC
        LIMIT 10
    """

    predictions = []
    result = session.run(query)

    for record in result:
        predictions.append({
            'candidate_drug': record['candidate_drug'],
            'target_disease': record['target_disease'],
            'method': 'path_based',
            'score': record['score'],
            'evidence': f"{record['num_paths']} paths, avg length {record['avg_path_length']}"
        })
        print(f"  {record['candidate_drug']} -> {record['target_disease']}")
        print(f"    Paths: {record['num_paths']}, Avg length: {record['avg_path_length']}, Score: {record['score']:.3f}")

    return predictions


def shared_neighbor_analysis(session):
    """Method 3: Analyze shared proteins/pathways"""
    print("\n" + "="*80)
    print("METHOD 3: SHARED NEIGHBOR ANALYSIS")
    print("="*80)

    query = """
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

        RETURN drug.node_name as candidate_drug,
               disease.node_name as target_disease,
               shared_count as shared_connections,
               shared_entities as shared_entities
        ORDER BY shared_count DESC
        LIMIT 10
    """

    predictions = []
    result = session.run(query)

    for record in result:
        predictions.append({
            'candidate_drug': record['candidate_drug'],
            'target_disease': record['target_disease'],
            'method': 'shared_neighbors',
            'score': record['shared_connections'],
            'evidence': f"Shared: {', '.join(record['shared_entities'][:3])}"
        })
        print(f"  {record['candidate_drug']} -> {record['target_disease']}")
        print(f"    Shared connections: {record['shared_connections']}")
        print(f"    Entities: {', '.join(record['shared_entities'][:5])}")

    return predictions


def aggregate_predictions(all_predictions):
    """Aggregate predictions from multiple methods"""
    print("\n" + "="*80)
    print("AGGREGATED PREDICTIONS (BY DRUG-DISEASE PAIR)")
    print("="*80)

    # Group by drug-disease pair
    pairs = defaultdict(list)
    for pred in all_predictions:
        key = (pred['candidate_drug'], pred['target_disease'])
        pairs[key].append(pred)

    # Calculate aggregate scores
    results = []
    for (drug, disease), preds in pairs.items():
        methods = [p['method'] for p in preds]
        scores = [p['score'] for p in preds]
        evidences = [p['evidence'] for p in preds]

        results.append({
            'drug': drug,
            'disease': disease,
            'num_methods': len(methods),
            'avg_score': sum(scores) / len(scores),
            'methods': methods,
            'evidences': evidences
        })

    # Sort by number of methods (consensus) and average score
    results.sort(key=lambda x: (x['num_methods'], x['avg_score']), reverse=True)

    print(f"\nFound {len(results)} unique drug-disease predictions")
    print("\nTop Predictions (sorted by consensus):\n")

    for i, result in enumerate(results[:10], 1):
        print(f"{i}. {result['drug']} -> {result['disease']}")
        print(f"   Supported by {result['num_methods']} method(s)")
        print(f"   Average score: {result['avg_score']:.3f}")
        print(f"   Methods: {', '.join(result['methods'])}")
        for evidence in result['evidences']:
            print(f"     - {evidence}")
        print()

    return results


def main():
    print("="*80)
    print("DRUG REPURPOSING EDGE PREDICTION")
    print("="*80)
    print("\nConnecting to Memgraph...")

    driver = GraphDatabase.driver('bolt://localhost:7687', auth=None)
    session = driver.session()

    # Check if we have example data
    result = session.run("MATCH (n {is_example: true}) RETURN count(n) as count")
    count = result.single()['count']

    if count == 0:
        print("\n⚠️  No example data found!")
        print("Please run: python3 load_example_progressive.py")
        session.close()
        driver.close()
        return

    print(f"Found {count} example nodes in graph\n")

    # Run all prediction methods
    all_predictions = []

    # Method 1: Pattern-based
    predictions1 = pattern_based_inference(session)
    all_predictions.extend(predictions1)

    # Method 2: Path-based
    predictions2 = path_based_scoring(session)
    all_predictions.extend(predictions2)

    # Method 3: Shared neighbors
    predictions3 = shared_neighbor_analysis(session)
    all_predictions.extend(predictions3)

    # Aggregate results
    if all_predictions:
        aggregate_predictions(all_predictions)
    else:
        print("\n⚠️  No predictions found!")
        print("This could mean:")
        print("1. All possible drug-disease pairs already exist")
        print("2. Similarity edges are missing (run example_drug_repurposing_similarities.cypher)")
        print("3. Thresholds are too strict")

    print("\n" + "="*80)
    print("PREDICTION COMPLETE")
    print("="*80)

    session.close()
    driver.close()


if __name__ == "__main__":
    main()
