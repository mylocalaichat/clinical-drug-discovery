"""
Drug Repurposing Example Queries
Demonstrates how graph patterns can be used to infer new drug-disease relationships
"""

from neo4j import GraphDatabase


def run_query(session, title, query, description=None):
    """Execute a query and print formatted results"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    if description:
        print(f"{description}\n")
    print(f"Query:\n{query}\n")

    result = session.run(query)
    records = list(result)

    if not records:
        print("No results found.")
        return

    for record in records:
        print(record.data())

    print(f"\n{len(records)} result(s) found")
    return records


def main():
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=None)
    session = driver.session()

    print("\n" + "="*80)
    print("DRUG REPURPOSING EXAMPLE - Graph Analysis")
    print("="*80)

    # Query 1: Show the example graph structure
    run_query(
        session,
        "1. Overview of Example Graph",
        """
        MATCH (n)
        WHERE n.example_set = 'drug_repurposing'
        RETURN n.node_type as node_type, collect(n.node_name) as nodes
        ORDER BY node_type
        """,
        "See all nodes in the example graph grouped by type"
    )

    # Query 2: Known drug-disease treatments
    run_query(
        session,
        "2. Known Drug-Disease Treatments",
        """
        MATCH (drug)-[r]->(disease)
        WHERE drug.example_set = 'drug_repurposing'
        AND disease.example_set = 'drug_repurposing'
        AND r.relation = 'drug_treats_disease'
        RETURN drug.node_name as Drug,
               disease.node_name as Disease,
               r.evidence as Evidence,
               r.efficacy as Efficacy
        """,
        "Show which drugs are known to treat which diseases"
    )

    # Query 3: Drug A's mechanism of action
    run_query(
        session,
        "3. Drug A (Metformin) - Mechanism of Action",
        """
        MATCH path = (drug)-[*1..3]->(disease)
        WHERE drug.node_name = 'Metformin'
        AND disease.node_name = 'Type 2 Diabetes'
        AND drug.example_set = 'drug_repurposing'
        WITH path, [n in nodes(path) | n.node_name] as node_names,
             [r in relationships(path) | r.display_relation] as rels
        RETURN node_names, rels
        LIMIT 5
        """,
        "How does Metformin treat Type 2 Diabetes?"
    )

    # Query 4: Drug E's mechanism of action
    run_query(
        session,
        "4. Drug E (Aspirin) - Mechanism of Action",
        """
        MATCH path = (drug)-[*1..3]->(disease)
        WHERE drug.node_name = 'Aspirin'
        AND disease.node_name = 'Cardiovascular Disease'
        AND drug.example_set = 'drug_repurposing'
        WITH path, [n in nodes(path) | n.node_name] as node_names,
             [r in relationships(path) | r.display_relation] as rels
        RETURN node_names, rels
        LIMIT 5
        """,
        "How does Aspirin treat Cardiovascular Disease?"
    )

    # Query 5: Disease similarities
    run_query(
        session,
        "5. Disease Similarities",
        """
        MATCH (d1)-[r]->(d2)
        WHERE d1.example_set = 'drug_repurposing'
        AND d2.example_set = 'drug_repurposing'
        AND r.relation = 'disease_similarity'
        RETURN d1.node_name as Disease1,
               d2.node_name as Disease2,
               r.similarity_score as Similarity,
               r.shared_features as SharedFeatures
        """,
        "Which diseases are similar to each other?"
    )

    # Query 6: Drug similarities
    run_query(
        session,
        "6. Drug Similarities",
        """
        MATCH (drug1)-[r]->(drug2)
        WHERE drug1.example_set = 'drug_repurposing'
        AND drug2.example_set = 'drug_repurposing'
        AND r.relation = 'drug_similarity'
        RETURN drug1.node_name as Drug1,
               drug2.node_name as Drug2,
               r.similarity_score as Similarity,
               r.shared_features as SharedFeatures
        """,
        "Which drugs are similar to each other?"
    )

    # Query 7: Shared mechanisms between diseases
    run_query(
        session,
        "7. Shared Biological Mechanisms",
        """
        MATCH (d1)<-[:RELATES]-(protein)-[:RELATES]->(d2)
        WHERE d1.example_set = 'drug_repurposing'
        AND d2.example_set = 'drug_repurposing'
        AND d1.node_type = 'disease'
        AND d2.node_type = 'disease'
        AND protein.node_type = 'gene/protein'
        AND d1.node_name < d2.node_name
        RETURN d1.node_name as Disease1,
               protein.node_name as SharedProtein,
               d2.node_name as Disease2
        """,
        "What proteins/mechanisms are shared between diseases?"
    )

    # Query 8: THE BIG ONE - Drug Repurposing Inference
    run_query(
        session,
        "8. DRUG REPURPOSING INFERENCE",
        """
        // Find potential new drug-disease treatments based on:
        // 1. Drug similarity (Drug A similar to Drug E)
        // 2. Disease similarity (Disease B/C similar to Disease D)
        // 3. Known treatment (Drug A treats B, Drug E treats C)
        // INFERENCE: Drug A might treat Disease D

        MATCH (drugA)-[treats:RELATES]->(diseaseB),
              (drugE)-[:RELATES]->(diseaseC),
              (drugA)-[drugSim:RELATES]->(drugE),
              (diseaseB)-[diseaseSim1:RELATES]->(diseaseD),
              (diseaseC)-[diseaseSim2:RELATES]->(diseaseD)
        WHERE drugA.example_set = 'drug_repurposing'
        AND treats.relation = 'drug_treats_disease'
        AND drugSim.relation = 'drug_similarity'
        AND diseaseSim1.relation = 'disease_similarity'
        AND diseaseSim2.relation = 'disease_similarity'
        AND diseaseD.node_name = 'Metabolic Syndrome'

        // Check that drugA doesn't already treat diseaseD (new discovery)
        AND NOT EXISTS {
            MATCH (drugA)-[r:RELATES]->(diseaseD)
            WHERE r.relation = 'drug_treats_disease'
        }

        RETURN DISTINCT
            drugA.node_name as CandidateDrug,
            diseaseD.node_name as TargetDisease,
            drugA.mechanism as DrugMechanism,
            diseaseB.node_name as KnownTreatedDisease1,
            diseaseC.node_name as KnownTreatedDisease2,
            drugE.node_name as SimilarDrug,
            drugSim.similarity_score as DrugSimilarity,
            diseaseSim1.similarity_score as DiseaseSimilarity1,
            diseaseSim2.similarity_score as DiseaseSimilarity2,
            'Drug similarity + Disease similarity + Known treatment' as InferenceReason
        """,
        """This is the key query! It finds that:
        - Metformin treats Type 2 Diabetes
        - Aspirin treats Cardiovascular Disease
        - Metformin and Aspirin are similar (anti-inflammatory effects)
        - Type 2 Diabetes and Metabolic Syndrome are similar
        - Cardiovascular Disease and Metabolic Syndrome are similar
        THEREFORE: Metformin might treat Metabolic Syndrome!"""
    )

    # Query 9: Pathway overlap analysis
    run_query(
        session,
        "9. Shared Pathway Analysis",
        """
        MATCH (drug)-[:RELATES]->(protein)-[:RELATES]->(pathway)-[:RELATES]->(disease)
        WHERE drug.example_set = 'drug_repurposing'
        AND disease.example_set = 'drug_repurposing'
        RETURN drug.node_name as Drug,
               protein.node_name as Protein,
               pathway.node_name as Pathway,
               disease.node_name as Disease
        ORDER BY Drug, Pathway
        """,
        "Which pathways connect drugs to diseases?"
    )

    # Query 10: Full subgraph visualization data
    run_query(
        session,
        "10. Complete Example Subgraph",
        """
        MATCH (n)
        WHERE n.example_set = 'drug_repurposing'
        WITH collect(n) as nodes
        MATCH (a)-[r]->(b)
        WHERE a.example_set = 'drug_repurposing'
        AND b.example_set = 'drug_repurposing'
        RETURN
            size(nodes) as total_nodes,
            count(r) as total_relationships,
            [n in nodes | {name: n.node_name, type: n.node_type}] as node_list,
            collect({
                from: a.node_name,
                to: b.node_name,
                relation: r.display_relation,
                type: r.relation
            }) as edge_list
        """,
        "Get complete graph structure for visualization"
    )

    session.close()
    driver.close()

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print("\nKey Insight:")
    print("Query #8 demonstrates drug repurposing inference:")
    print("Because Metformin and Aspirin share anti-inflammatory properties,")
    print("and Metabolic Syndrome is similar to both Type 2 Diabetes and")
    print("Cardiovascular Disease, we can infer that Metformin (which treats")
    print("Type 2 Diabetes) might also be effective for Metabolic Syndrome!")
    print("="*80)


if __name__ == "__main__":
    main()
