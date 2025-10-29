// =====================================================================
// TARGETED OFF-LABEL DRUG PREDICTION
// =====================================================================
// This query allows you to search for off-label drugs with specific
// characteristics (mechanism, drug class, specific pathways, etc.)
//
// Use cases:
// - Find anti-inflammatory drugs for metabolic diseases
// - Find drugs that modulate specific pathways
// - Find drugs with specific mechanisms for repurposing
// =====================================================================

// =============================================
// SCENARIO 1: Find drugs with specific mechanism for a disease
// =============================================
// Example: Find COX inhibitors or AMPK activators for Metabolic Syndrome
MATCH (disease:Node {node_type: 'disease', node_name: 'Metabolic Syndrome'})
MATCH (drug:Node {node_type: 'drug'})
WHERE drug.mechanism IN ['COX inhibitor', 'AMPK activator', 'anti-inflammatory']
AND NOT EXISTS {
    (drug)-[:RELATES {relation: 'drug_treats_disease'}]->(disease)
}

// Find connections to disease
OPTIONAL MATCH path1 = (drug)-[:RELATES*1..3]->(disease)
WHERE ALL(r IN relationships(path1) WHERE r.relation IN [
    'drug_targets_protein', 'protein_disease_association',
    'protein_participates_pathway', 'pathway_disease_association'
])

WITH drug, disease, COLLECT(path1) AS paths
WHERE SIZE(paths) > 0

// Extract evidence from paths
UNWIND paths AS path
WITH drug, disease,
    [n IN nodes(path) WHERE n.node_type = 'gene/protein' | n.node_name] AS proteins,
    [n IN nodes(path) WHERE n.node_type = 'pathway' | n.node_name] AS pathways

WITH drug, disease,
    COLLECT(DISTINCT proteins) AS all_proteins,
    COLLECT(DISTINCT pathways) AS all_pathways

RETURN
    drug.node_name AS drug_name,
    drug.mechanism AS mechanism,
    drug.drug_class AS drug_class,
    disease.node_name AS target_disease,
    SIZE([p IN all_proteins WHERE p <> [] | p]) AS protein_connections,
    SIZE([p IN all_pathways WHERE p <> [] | p]) AS pathway_connections,
    [p IN all_proteins WHERE p <> [] | p[0]][0..3] AS sample_proteins,
    [p IN all_pathways WHERE p <> [] | p[0]][0..3] AS sample_pathways,
    drug.side_effects AS side_effects,
    drug.cost_category AS cost,
    'Mechanism-based prediction' AS prediction_type
ORDER BY protein_connections + pathway_connections DESC
LIMIT 10;


// =============================================
// SCENARIO 2: Find drugs that modulate specific pathway
// =============================================
/*
MATCH (disease:Node {node_type: 'disease', node_name: 'Cardiovascular Disease'})
MATCH (pathway:Node {node_type: 'pathway', node_name: 'Inflammation Pathway'})
MATCH (drug:Node {node_type: 'drug'})

WHERE NOT EXISTS {
    (drug)-[:RELATES {relation: 'drug_treats_disease'}]->(disease)
}

// Drug must modulate the pathway
AND EXISTS {
    (drug)-[:RELATES {relation: 'drug_targets_protein'}]->(:Node)
    -[:RELATES {relation: 'protein_participates_pathway'}]->(pathway)
}

// Pathway must be relevant to disease
MATCH (pathway)-[pwd:RELATES {relation: 'pathway_disease_association'}]->(disease)

// Get drug effects on pathway
MATCH (drug)-[dt:RELATES {relation: 'drug_targets_protein'}]->(target:Node)
    -[pp:RELATES {relation: 'protein_participates_pathway'}]->(pathway)

WITH drug, disease, pathway, pwd,
    COLLECT(DISTINCT {
        target: target.node_name,
        drug_effect: dt.effect,
        pathway_role: pp.role
    }) AS pathway_interactions

RETURN
    drug.node_name AS candidate_drug,
    drug.mechanism AS mechanism,
    disease.node_name AS target_disease,
    pathway.node_name AS modulated_pathway,
    pwd.pathway_activity_change AS pathway_status_in_disease,
    pwd.clinical_relevance_score AS pathway_relevance,
    SIZE(pathway_interactions) AS number_of_targets_in_pathway,
    pathway_interactions[0..3] AS sample_targets,
    drug.administration_route AS route,
    drug.cost_category AS cost,
    CASE
        WHEN pwd.pathway_activity_change = 'increased' AND
             ANY(x IN pathway_interactions WHERE x.drug_effect = 'inhibition')
            THEN 'Drug may normalize overactive pathway'
        WHEN pwd.pathway_activity_change = 'decreased' AND
             ANY(x IN pathway_interactions WHERE x.drug_effect = 'activation')
            THEN 'Drug may normalize underactive pathway'
        ELSE 'Drug modulates disease-relevant pathway'
    END AS mechanistic_rationale

ORDER BY pathway_relevance DESC, number_of_targets_in_pathway DESC
LIMIT 10;
*/


// =============================================
// SCENARIO 3: Find drugs from specific class for disease
// =============================================
/*
MATCH (disease:Node {node_type: 'disease', node_name: 'Type 2 Diabetes'})
MATCH (drug:Node {node_type: 'drug'})
WHERE drug.drug_class IN ['NSAID', 'Biguanide', 'Statin']
AND NOT EXISTS {
    (drug)-[:RELATES {relation: 'drug_treats_disease'}]->(disease)
}

// Find any path to disease
MATCH path = (drug)-[:RELATES*1..4]->(disease)
WHERE ALL(r IN relationships(path) WHERE r.relation IN [
    'drug_targets_protein', 'protein_disease_association',
    'protein_participates_pathway', 'pathway_disease_association',
    'protein_regulates_protein', 'drug_similarity', 'disease_similarity',
    'drug_treats_disease'
])

WITH drug, disease, path,
    LENGTH(path) AS path_length,
    [r IN relationships(path) | r.relation] AS path_relations,
    [n IN nodes(path)[1..-1] | n.node_name] AS intermediate_nodes

RETURN
    drug.node_name AS drug_name,
    drug.drug_class AS drug_class,
    disease.node_name AS target_disease,
    path_length AS connection_distance,
    path_relations AS evidence_chain,
    intermediate_nodes AS connecting_entities,
    drug.side_effects AS side_effects,
    CASE path_length
        WHEN 1 THEN 'DIRECT'
        WHEN 2 THEN 'STRONG'
        WHEN 3 THEN 'MODERATE'
        ELSE 'INDIRECT'
    END AS evidence_strength

ORDER BY path_length ASC, drug.drug_class ASC
LIMIT 15;
*/


// =============================================
// SCENARIO 4: Multi-disease repurposing
// Find drugs that could treat multiple related diseases
// =============================================
/*
MATCH (disease1:Node {node_id: 'EXAMPLE_DISEASE_B'})  // Type 2 Diabetes
MATCH (disease2:Node {node_id: 'EXAMPLE_DISEASE_D'})  // Metabolic Syndrome
MATCH (drug:Node {node_type: 'drug'})

WHERE NOT EXISTS {(drug)-[:RELATES {relation: 'drug_treats_disease'}]->(disease1)}
AND NOT EXISTS {(drug)-[:RELATES {relation: 'drug_treats_disease'}]->(disease2)}

// Find connections to both diseases
OPTIONAL MATCH (drug)-[:RELATES {relation: 'drug_targets_protein'}]->(p1:Node)
    -[:RELATES {relation: 'protein_disease_association'}]->(disease1)
OPTIONAL MATCH (drug)-[:RELATES {relation: 'drug_targets_protein'}]->(p2:Node)
    -[:RELATES {relation: 'protein_disease_association'}]->(disease2)

WITH drug, disease1, disease2,
    COUNT(DISTINCT p1) AS targets_for_disease1,
    COUNT(DISTINCT p2) AS targets_for_disease2,
    COLLECT(DISTINCT p1.node_name) AS proteins_d1,
    COLLECT(DISTINCT p2.node_name) AS proteins_d2

WHERE targets_for_disease1 > 0 AND targets_for_disease2 > 0

RETURN
    drug.node_name AS candidate_drug,
    drug.mechanism AS mechanism,
    disease1.node_name AS disease_1,
    disease2.node_name AS disease_2,
    targets_for_disease1 AS disease1_protein_targets,
    targets_for_disease2 AS disease2_protein_targets,
    proteins_d1[0..2] AS sample_disease1_targets,
    proteins_d2[0..2] AS sample_disease2_targets,
    (targets_for_disease1 + targets_for_disease2) AS total_evidence_score,
    'Multi-disease repurposing candidate' AS opportunity_type

ORDER BY total_evidence_score DESC
LIMIT 10;
*/


// =============================================
// SCENARIO 5: Safety-filtered predictions
// Find low-risk repurposing candidates
// =============================================
/*
MATCH (disease:Node {node_type: 'disease', node_name: 'Metabolic Syndrome'})
MATCH (drug:Node {node_type: 'drug'})

WHERE NOT EXISTS {
    (drug)-[:RELATES {relation: 'drug_treats_disease'}]->(disease)
}
// Safety filters
AND drug.cost_category IN ['low', 'moderate']
AND drug.pregnancy_category IN ['A', 'B', 'C']  // Exclude highest risk
AND drug.administration_route = 'oral'
AND drug.fda_approval_year < 2000  // Long track record

// Must have evidence
AND EXISTS {
    (drug)-[:RELATES*1..3]->(disease)
}

MATCH (drug)-[:RELATES {relation: 'drug_targets_protein'}]->(protein:Node)
    -[:RELATES {relation: 'protein_disease_association'}]->(disease)

WITH drug, disease,
    COUNT(DISTINCT protein) AS evidence_count,
    COLLECT(DISTINCT protein.node_name)[0..3] AS sample_targets

RETURN
    drug.node_name AS safe_candidate,
    drug.drug_class AS drug_class,
    disease.node_name AS target_disease,
    evidence_count AS evidence_strength,
    sample_targets AS protein_targets,
    drug.fda_approval_year AS approved_since,
    drug.cost_category AS cost,
    drug.side_effects AS side_effects,
    drug.daily_dose_mg AS typical_dose_mg,
    'Low-risk repurposing candidate' AS candidate_type

ORDER BY evidence_count DESC
LIMIT 10;
*/
