// =====================================================================
// COMPREHENSIVE OFF-LABEL DRUG PREDICTION QUERY
// =====================================================================
// This query predicts potential off-label drugs for a given disease
// using multiple graph-based inference strategies
//
// USAGE:
// Set the disease_id parameter to the disease you want to find treatments for
// Example: disease_id = 'EXAMPLE_DISEASE_D' (Metabolic Syndrome)
//
// SCORING:
// Each prediction path contributes to a total score:
// - Direct pathway modulation: 0.4
// - Shared protein targets: 0.35
// - Similar disease treatment: 0.3
// - Drug similarity to known treatments: 0.25
// =====================================================================

// Set your target disease here
:param disease_id => 'EXAMPLE_DISEASE_D';

// =====================================================================
// STRATEGY 1: Pathway-Based Discovery
// Find drugs that modulate pathways dysregulated in the disease
// =====================================================================
MATCH (disease:Node {node_id: $disease_id})
MATCH (pathway:Node {node_type: 'pathway'})-[pd:RELATES {relation: 'pathway_disease_association'}]->(disease)
MATCH (protein:Node {node_type: 'gene/protein'})-[pp:RELATES {relation: 'protein_participates_pathway'}]->(pathway)
MATCH (drug:Node {node_type: 'drug'})-[dt:RELATES {relation: 'drug_targets_protein'}]->(protein)
WHERE NOT EXISTS {
    (drug)-[:RELATES {relation: 'drug_treats_disease'}]->(disease)
}
WITH drug, disease,
     COUNT(DISTINCT pathway) AS pathway_count,
     AVG(pd.clinical_relevance_score) AS pathway_relevance,
     AVG(dt.confidence) AS target_confidence,
     COLLECT(DISTINCT {
         pathway: pathway.node_name,
         protein: protein.node_name,
         effect: dt.effect,
         pathway_activity_change: pd.pathway_activity_change
     }) AS pathway_evidence
RETURN
    'pathway_modulation' AS strategy,
    drug.node_name AS drug_name,
    drug.node_id AS drug_id,
    disease.node_name AS disease_name,
    (pathway_count * pathway_relevance * target_confidence * 0.4) AS strategy_score,
    pathway_count,
    pathway_relevance,
    pathway_evidence
ORDER BY strategy_score DESC
LIMIT 10

UNION

// =====================================================================
// STRATEGY 2: Shared Protein Target Discovery
// Find drugs targeting proteins associated with the disease
// =====================================================================
MATCH (disease:Node {node_id: $disease_id})
MATCH (protein:Node {node_type: 'gene/protein'})-[pd:RELATES {relation: 'protein_disease_association'}]->(disease)
MATCH (drug:Node {node_type: 'drug'})-[dt:RELATES {relation: 'drug_targets_protein'}]->(protein)
WHERE NOT EXISTS {
    (drug)-[:RELATES {relation: 'drug_treats_disease'}]->(disease)
}
WITH drug, disease,
     COUNT(DISTINCT protein) AS protein_count,
     AVG(pd.association_strength) AS avg_association_strength,
     AVG(dt.confidence) AS avg_target_confidence,
     COLLECT(DISTINCT {
         protein: protein.node_name,
         drug_effect: dt.effect,
         protein_association: pd.association_type,
         expression_change: pd.expression_change,
         binding_affinity_nm: dt.binding_affinity_nm
     }) AS protein_evidence
RETURN
    'shared_protein_targets' AS strategy,
    drug.node_name AS drug_name,
    drug.node_id AS drug_id,
    disease.node_name AS disease_name,
    (protein_count * avg_association_strength * avg_target_confidence * 0.35) AS strategy_score,
    protein_count,
    avg_association_strength,
    protein_evidence
ORDER BY strategy_score DESC
LIMIT 10

UNION

// =====================================================================
// STRATEGY 3: Similar Disease Treatment
// Find drugs treating diseases similar to the target disease
// =====================================================================
MATCH (disease:Node {node_id: $disease_id})
MATCH (disease)<-[ds:RELATES {relation: 'disease_similarity'}]-(similar_disease:Node {node_type: 'disease'})
MATCH (drug:Node {node_type: 'drug'})-[td:RELATES {relation: 'drug_treats_disease'}]->(similar_disease)
WHERE NOT EXISTS {
    (drug)-[:RELATES {relation: 'drug_treats_disease'}]->(disease)
}
WITH drug, disease,
     COUNT(DISTINCT similar_disease) AS similar_disease_count,
     AVG(ds.similarity_score) AS avg_similarity,
     AVG(ds.clinical_overlap_score) AS avg_clinical_overlap,
     COLLECT(DISTINCT {
         similar_disease: similar_disease.node_name,
         similarity_score: ds.similarity_score,
         shared_features: ds.shared_features,
         comorbidity_rate: ds.comorbidity_rate,
         shared_pathways: ds.shared_pathways
     }) AS disease_similarity_evidence
RETURN
    'similar_disease_treatment' AS strategy,
    drug.node_name AS drug_name,
    drug.node_id AS drug_id,
    disease.node_name AS disease_name,
    (similar_disease_count * avg_similarity * avg_clinical_overlap * 0.3) AS strategy_score,
    similar_disease_count,
    avg_similarity,
    disease_similarity_evidence
ORDER BY strategy_score DESC
LIMIT 10

UNION

// =====================================================================
// STRATEGY 4: Drug Similarity to Known Treatments
// Find drugs similar to those already treating the disease
// =====================================================================
MATCH (disease:Node {node_id: $disease_id})
MATCH (known_drug:Node {node_type: 'drug'})-[:RELATES {relation: 'drug_treats_disease'}]->(disease)
MATCH (known_drug)<-[drug_sim:RELATES {relation: 'drug_similarity'}]-(candidate_drug:Node {node_type: 'drug'})
WHERE NOT EXISTS {
    (candidate_drug)-[:RELATES {relation: 'drug_treats_disease'}]->(disease)
}
WITH candidate_drug, disease,
     COUNT(DISTINCT known_drug) AS known_drug_count,
     AVG(drug_sim.similarity_score) AS avg_drug_similarity,
     AVG(drug_sim.pharmacological_similarity) AS avg_pharm_similarity,
     COLLECT(DISTINCT {
         known_drug: known_drug.node_name,
         similarity_score: drug_sim.similarity_score,
         shared_features: drug_sim.shared_features,
         pharmacological_similarity: drug_sim.pharmacological_similarity,
         mechanism_overlap_score: drug_sim.mechanism_overlap_score
     }) AS drug_similarity_evidence
RETURN
    'drug_similarity' AS strategy,
    candidate_drug.node_name AS drug_name,
    candidate_drug.node_id AS drug_id,
    disease.node_name AS disease_name,
    (known_drug_count * avg_drug_similarity * avg_pharm_similarity * 0.25) AS strategy_score,
    known_drug_count,
    avg_drug_similarity,
    drug_similarity_evidence
ORDER BY strategy_score DESC
LIMIT 10

UNION

// =====================================================================
// STRATEGY 5: Multi-Hop Protein Regulation
// Find drugs that modulate proteins which regulate disease-associated proteins
// =====================================================================
MATCH (disease:Node {node_id: $disease_id})
MATCH (disease_protein:Node {node_type: 'gene/protein'})-[pd:RELATES {relation: 'protein_disease_association'}]->(disease)
MATCH (regulator:Node {node_type: 'gene/protein'})-[pr:RELATES {relation: 'protein_regulates_protein'}]->(disease_protein)
MATCH (drug:Node {node_type: 'drug'})-[dt:RELATES {relation: 'drug_targets_protein'}]->(regulator)
WHERE NOT EXISTS {
    (drug)-[:RELATES {relation: 'drug_treats_disease'}]->(disease)
}
AND (
    (pr.effect = 'upregulation' AND pd.expression_change = 'upregulated' AND dt.effect = 'inhibition') OR
    (pr.effect = 'downregulation' AND pd.expression_change = 'upregulated' AND dt.effect = 'activation') OR
    (pr.effect = 'upregulation' AND pd.expression_change = 'downregulated' AND dt.effect = 'activation') OR
    (pr.effect = 'downregulation' AND pd.expression_change = 'downregulated' AND dt.effect = 'inhibition')
)
WITH drug, disease,
     COUNT(DISTINCT disease_protein) AS regulated_protein_count,
     AVG(pd.association_strength) AS avg_disease_association,
     AVG(pr.interaction_strength) AS avg_regulation_strength,
     COLLECT(DISTINCT {
         disease_protein: disease_protein.node_name,
         regulator_protein: regulator.node_name,
         drug_effect: dt.effect,
         regulatory_effect: pr.effect,
         mechanism: pr.mechanism
     }) AS regulation_evidence
RETURN
    'protein_regulation' AS strategy,
    drug.node_name AS drug_name,
    drug.node_id AS drug_id,
    disease.node_name AS disease_name,
    (regulated_protein_count * avg_disease_association * avg_regulation_strength * 0.3) AS strategy_score,
    regulated_protein_count,
    avg_disease_association,
    regulation_evidence
ORDER BY strategy_score DESC
LIMIT 10;
