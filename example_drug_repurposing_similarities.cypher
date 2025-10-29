// Drug Repurposing Example - SIMILARITY EDGES ONLY
// This script adds ONLY the similarity relationships that enable drug repurposing inference
//
// IMPORTANT: Run example_drug_repurposing_base.cypher FIRST to create the base graph
// Then run this script to add the similarity edges
//
// The similarity edges are key to inference:
// - Drug similarities allow us to infer that similar drugs may treat similar diseases
// - Disease similarities allow us to infer that drugs treating one disease may treat similar diseases
//
// These 3 edges enable the inference: Metformin (treats Diabetes, similar to Aspirin which treats CVD)
// might treat Metabolic Syndrome (similar to both Diabetes and CVD)

// ===== DISEASE SIMILARITY RELATIONSHIPS =====
MATCH (diseaseB:Node {node_id: 'EXAMPLE_DISEASE_B'})
MATCH (diseaseD:Node {node_id: 'EXAMPLE_DISEASE_D'})
CREATE (diseaseB)-[:RELATES {
    relation: 'disease_similarity',
    display_relation: 'similar_to',
    example_set: 'drug_repurposing',
    is_example: true,
    similarity_score: 0.75,
    shared_features: 'metabolic_dysfunction',
    similarity_method: 'gene_expression_profile',
    comorbidity_rate: 0.42,
    clinical_overlap_score: 0.68,
    shared_pathways: 3,
    shared_genes: 127
}]->(diseaseD);

MATCH (diseaseC:Node {node_id: 'EXAMPLE_DISEASE_C'})
MATCH (diseaseD:Node {node_id: 'EXAMPLE_DISEASE_D'})
CREATE (diseaseC)-[:RELATES {
    relation: 'disease_similarity',
    display_relation: 'similar_to',
    example_set: 'drug_repurposing',
    is_example: true,
    similarity_score: 0.70,
    shared_features: 'cardiovascular_risk',
    similarity_method: 'phenotype_similarity',
    comorbidity_rate: 0.38,
    clinical_overlap_score: 0.61,
    shared_pathways: 2,
    shared_genes: 93
}]->(diseaseD);

// ===== DRUG SIMILARITY RELATIONSHIPS =====
MATCH (drugA:Node {node_id: 'EXAMPLE_DRUG_A'})
MATCH (drugE:Node {node_id: 'EXAMPLE_DRUG_E'})
CREATE (drugA)-[:RELATES {
    relation: 'drug_similarity',
    display_relation: 'similar_to',
    example_set: 'drug_repurposing',
    is_example: true,
    similarity_score: 0.65,
    shared_features: 'anti-inflammatory_effects',
    structural_similarity: 0.42,
    pharmacological_similarity: 0.73,
    mechanism_overlap_score: 0.58,
    side_effect_similarity: 0.55,
    target_overlap: 0.31
}]->(drugE);
