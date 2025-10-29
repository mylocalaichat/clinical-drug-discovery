// =====================================================================
// SIMPLE OFF-LABEL DRUG PREDICTION
// =====================================================================
// A straightforward query to find candidate off-label drugs for a disease
//
// HOW IT WORKS:
// 1. Finds drugs NOT currently approved for the disease
// 2. Looks for connections through proteins, pathways, and similarities
// 3. Ranks candidates by the strength of evidence
//
// USAGE: Change the disease_name below or use parameter
// =====================================================================

// Option 1: Set disease by name
MATCH (disease:Node {node_type: 'disease', node_name: 'Metabolic Syndrome'})

// Option 2: Or set disease by ID (comment out Option 1 and uncomment below)
// MATCH (disease:Node {node_type: 'disease', node_id: 'EXAMPLE_DISEASE_D'})

// Find candidate drugs through multiple paths
MATCH (drug:Node {node_type: 'drug'})
WHERE NOT EXISTS {
    (drug)-[:RELATES {relation: 'drug_treats_disease'}]->(disease)
}

// Path 1: Drug targets protein associated with disease
OPTIONAL MATCH (drug)-[dt:RELATES {relation: 'drug_targets_protein'}]->(protein:Node)
    -[pd:RELATES {relation: 'protein_disease_association'}]->(disease)

// Path 2: Drug modulates pathway dysregulated in disease
OPTIONAL MATCH (drug)-[:RELATES {relation: 'drug_targets_protein'}]->(:Node)
    -[:RELATES {relation: 'protein_participates_pathway'}]->(pathway:Node)
    -[pwd:RELATES {relation: 'pathway_disease_association'}]->(disease)

// Path 3: Drug treats similar disease
OPTIONAL MATCH (drug)-[:RELATES {relation: 'drug_treats_disease'}]->(similar_disease:Node)
    -[ds:RELATES {relation: 'disease_similarity'}]->(disease)

// Path 4: Drug is similar to known treatment
OPTIONAL MATCH (drug)-[drug_sim:RELATES {relation: 'drug_similarity'}]->(:Node)
    -[:RELATES {relation: 'drug_treats_disease'}]->(disease)

WITH drug, disease,
    // Count evidence from each path
    COUNT(DISTINCT protein) AS direct_protein_targets,
    COUNT(DISTINCT pathway) AS relevant_pathways,
    COUNT(DISTINCT similar_disease) AS similar_diseases_treated,
    COUNT(DISTINCT drug_sim) AS similar_to_known_drugs,

    // Get top evidence scores
    MAX(COALESCE(pd.association_strength, 0)) AS best_protein_association,
    MAX(COALESCE(pwd.clinical_relevance_score, 0)) AS best_pathway_score,
    MAX(COALESCE(ds.similarity_score, 0)) AS best_disease_similarity,
    MAX(COALESCE(drug_sim.similarity_score, 0)) AS best_drug_similarity,

    // Collect evidence details
    COLLECT(DISTINCT protein.node_name)[0..3] AS sample_protein_targets,
    COLLECT(DISTINCT pathway.node_name)[0..3] AS sample_pathways,
    COLLECT(DISTINCT similar_disease.node_name)[0..3] AS sample_similar_diseases

// Calculate total evidence score
WITH drug, disease,
    direct_protein_targets, relevant_pathways,
    similar_diseases_treated, similar_to_known_drugs,
    best_protein_association, best_pathway_score,
    best_disease_similarity, best_drug_similarity,
    sample_protein_targets, sample_pathways, sample_similar_diseases,

    // Weighted scoring
    (direct_protein_targets * 0.3 * best_protein_association) +
    (relevant_pathways * 0.25 * best_pathway_score) +
    (similar_diseases_treated * 0.25 * best_disease_similarity) +
    (similar_to_known_drugs * 0.2 * best_drug_similarity) AS evidence_score

WHERE evidence_score > 0

RETURN
    drug.node_name AS drug_name,
    drug.drug_class AS drug_class,
    drug.mechanism AS mechanism_of_action,
    disease.node_name AS target_disease,

    // Overall score and confidence
    ROUND(evidence_score * 100) / 100 AS prediction_score,
    CASE
        WHEN evidence_score >= 0.3 THEN 'HIGH'
        WHEN evidence_score >= 0.15 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS confidence_level,

    // Evidence counts
    direct_protein_targets AS protein_targets_count,
    relevant_pathways AS pathway_connections,
    similar_diseases_treated AS similar_diseases_count,
    similar_to_known_drugs AS drug_similarity_count,

    // Evidence samples
    sample_protein_targets AS example_protein_targets,
    sample_pathways AS example_pathways,
    sample_similar_diseases AS example_similar_diseases,

    // Drug properties for evaluation
    drug.administration_route AS route,
    drug.cost_category AS cost,
    drug.side_effects AS known_side_effects,
    drug.fda_approval_year AS fda_approved_since,

    // Rationale
    CASE
        WHEN direct_protein_targets > 0 AND relevant_pathways > 0
            THEN 'Drug targets disease-associated proteins and relevant pathways'
        WHEN direct_protein_targets > 0
            THEN 'Drug targets proteins directly associated with disease'
        WHEN similar_diseases_treated > 0
            THEN 'Drug treats diseases similar to target disease'
        WHEN relevant_pathways > 0
            THEN 'Drug modulates pathways dysregulated in disease'
        ELSE 'Drug has similarity to known treatments'
    END AS prediction_rationale

ORDER BY prediction_score DESC, confidence_level DESC
LIMIT 10;


// =====================================================================
// QUICK SUMMARY OF RESULTS
// =====================================================================
// For a quick overview, uncomment and run this query after the above:
/*
MATCH (disease:Node {node_type: 'disease', node_name: 'Metabolic Syndrome'})
MATCH (drug:Node {node_type: 'drug'})
WHERE NOT EXISTS {
    (drug)-[:RELATES {relation: 'drug_treats_disease'}]->(disease)
}
AND (
    EXISTS {(drug)-[:RELATES {relation: 'drug_targets_protein'}]->()-[:RELATES]->(disease)} OR
    EXISTS {(drug)-[:RELATES]->()-[:RELATES {relation: 'disease_similarity'}]->(disease)}
)
RETURN
    COUNT(drug) AS total_candidate_drugs,
    disease.node_name AS target_disease,
    'Run full query above for detailed predictions' AS next_step;
*/
