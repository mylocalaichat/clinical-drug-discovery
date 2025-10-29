// =====================================================================
// AGGREGATED OFF-LABEL DRUG PREDICTION - COMBINED SCORE
// =====================================================================
// This query combines all prediction strategies to provide a unified ranking
// of candidate off-label drugs for a given disease
//
// USAGE:
// :param disease_id => 'EXAMPLE_DISEASE_D'
//
// The query returns:
// - Combined confidence score from all strategies
// - Individual strategy contributions
// - Comprehensive evidence summary
// - Drug safety and pharmacological properties
// =====================================================================

:param disease_id => 'EXAMPLE_DISEASE_D';

// Strategy 1: Pathway-based
MATCH (disease:Node {node_id: $disease_id})
OPTIONAL MATCH (pathway:Node {node_type: 'pathway'})-[pd:RELATES {relation: 'pathway_disease_association'}]->(disease)
OPTIONAL MATCH (protein:Node {node_type: 'gene/protein'})-[:RELATES {relation: 'protein_participates_pathway'}]->(pathway)
OPTIONAL MATCH (drug:Node {node_type: 'drug'})-[dt:RELATES {relation: 'drug_targets_protein'}]->(protein)
WHERE drug IS NOT NULL
AND NOT EXISTS {
    (drug)-[:RELATES {relation: 'drug_treats_disease'}]->(disease)
}
WITH drug, disease,
     COUNT(DISTINCT pathway) AS pathway_count,
     AVG(COALESCE(pd.clinical_relevance_score, 0)) AS pathway_score
WITH drug, disease,
     (pathway_count * pathway_score * 0.4) AS pathway_strategy_score

// Strategy 2: Shared protein targets
OPTIONAL MATCH (disease)<-[pd:RELATES {relation: 'protein_disease_association'}]-(protein:Node {node_type: 'gene/protein'})
OPTIONAL MATCH (drug)-[dt:RELATES {relation: 'drug_targets_protein'}]->(protein)
WHERE drug IS NOT NULL
WITH drug, disease, pathway_strategy_score,
     COUNT(DISTINCT protein) AS protein_count,
     AVG(COALESCE(pd.association_strength, 0)) AS protein_score
WITH drug, disease, pathway_strategy_score,
     (protein_count * protein_score * 0.35) AS protein_strategy_score

// Strategy 3: Similar disease treatment
OPTIONAL MATCH (disease)<-[ds:RELATES {relation: 'disease_similarity'}]-(similar_disease:Node {node_type: 'disease'})
OPTIONAL MATCH (drug)-[:RELATES {relation: 'drug_treats_disease'}]->(similar_disease)
WHERE drug IS NOT NULL
WITH drug, disease, pathway_strategy_score, protein_strategy_score,
     COUNT(DISTINCT similar_disease) AS similar_disease_count,
     AVG(COALESCE(ds.similarity_score, 0)) AS disease_sim_score
WITH drug, disease, pathway_strategy_score, protein_strategy_score,
     (similar_disease_count * disease_sim_score * 0.3) AS disease_sim_strategy_score

// Strategy 4: Drug similarity
OPTIONAL MATCH (disease)<-[:RELATES {relation: 'drug_treats_disease'}]-(known_drug:Node {node_type: 'drug'})
OPTIONAL MATCH (known_drug)<-[drug_sim:RELATES {relation: 'drug_similarity'}]-(drug)
WHERE drug IS NOT NULL
WITH drug, disease, pathway_strategy_score, protein_strategy_score, disease_sim_strategy_score,
     COUNT(DISTINCT known_drug) AS similar_known_drugs,
     AVG(COALESCE(drug_sim.similarity_score, 0)) AS drug_sim_score
WITH drug, disease, pathway_strategy_score, protein_strategy_score, disease_sim_strategy_score,
     (similar_known_drugs * drug_sim_score * 0.25) AS drug_sim_strategy_score

// Strategy 5: Protein regulation
OPTIONAL MATCH (disease)<-[pd:RELATES {relation: 'protein_disease_association'}]-(disease_protein:Node {node_type: 'gene/protein'})
OPTIONAL MATCH (regulator:Node {node_type: 'gene/protein'})-[pr:RELATES {relation: 'protein_regulates_protein'}]->(disease_protein)
OPTIONAL MATCH (drug)-[dt:RELATES {relation: 'drug_targets_protein'}]->(regulator)
WHERE drug IS NOT NULL
AND (
    (pr.effect = 'upregulation' AND dt.effect = 'inhibition') OR
    (pr.effect = 'downregulation' AND dt.effect = 'activation')
)
WITH drug, disease, pathway_strategy_score, protein_strategy_score,
     disease_sim_strategy_score, drug_sim_strategy_score,
     COUNT(DISTINCT disease_protein) AS regulated_proteins,
     AVG(COALESCE(pr.interaction_strength, 0)) AS regulation_score
WITH drug, disease, pathway_strategy_score, protein_strategy_score,
     disease_sim_strategy_score, drug_sim_strategy_score,
     (regulated_proteins * regulation_score * 0.3) AS regulation_strategy_score

// Calculate total score
WITH drug, disease,
     COALESCE(pathway_strategy_score, 0) AS pathway_score,
     COALESCE(protein_strategy_score, 0) AS protein_score,
     COALESCE(disease_sim_strategy_score, 0) AS disease_sim_score,
     COALESCE(drug_sim_strategy_score, 0) AS drug_similarity_score,
     COALESCE(regulation_strategy_score, 0) AS regulation_score
WITH drug, disease,
     pathway_score, protein_score, disease_sim_score, drug_similarity_score, regulation_score,
     (pathway_score + protein_score + disease_sim_score + drug_similarity_score + regulation_score) AS total_score

WHERE total_score > 0

// Gather detailed evidence
OPTIONAL MATCH (drug)-[:RELATES {relation: 'drug_targets_protein'}]->(target:Node {node_type: 'gene/protein'})
OPTIONAL MATCH (target)-[:RELATES {relation: 'protein_disease_association'}]->(disease)
WITH drug, disease, pathway_score, protein_score, disease_sim_score,
     drug_similarity_score, regulation_score, total_score,
     COLLECT(DISTINCT target.node_name) AS targeted_disease_proteins

OPTIONAL MATCH (drug)-[:RELATES {relation: 'drug_targets_protein'}]->(p:Node)-[:RELATES {relation: 'protein_participates_pathway'}]->(path:Node)
OPTIONAL MATCH (path)-[:RELATES {relation: 'pathway_disease_association'}]->(disease)
WITH drug, disease, pathway_score, protein_score, disease_sim_score,
     drug_similarity_score, regulation_score, total_score, targeted_disease_proteins,
     COLLECT(DISTINCT path.node_name) AS relevant_pathways

OPTIONAL MATCH (disease)<-[:RELATES {relation: 'disease_similarity'}]-(sim_disease:Node)
OPTIONAL MATCH (drug)-[:RELATES {relation: 'drug_treats_disease'}]->(sim_disease)
WITH drug, disease, pathway_score, protein_score, disease_sim_score,
     drug_similarity_score, regulation_score, total_score,
     targeted_disease_proteins, relevant_pathways,
     COLLECT(DISTINCT sim_disease.node_name) AS similar_diseases_treated

RETURN
    drug.node_name AS candidate_drug,
    drug.node_id AS drug_id,
    disease.node_name AS target_disease,
    ROUND(total_score * 100) / 100 AS total_confidence_score,
    ROUND(pathway_score * 100) / 100 AS pathway_contribution,
    ROUND(protein_score * 100) / 100 AS protein_target_contribution,
    ROUND(disease_sim_score * 100) / 100 AS disease_similarity_contribution,
    ROUND(drug_similarity_score * 100) / 100 AS drug_similarity_contribution,
    ROUND(regulation_score * 100) / 100 AS regulation_contribution,
    SIZE(targeted_disease_proteins) AS num_disease_associated_targets,
    SIZE(relevant_pathways) AS num_relevant_pathways,
    SIZE(similar_diseases_treated) AS num_similar_diseases_treated,
    targeted_disease_proteins[0..3] AS top_targets,
    relevant_pathways[0..3] AS top_pathways,
    similar_diseases_treated AS similar_diseases,
    drug.mechanism AS drug_mechanism,
    drug.drug_class AS drug_class,
    drug.administration_route AS route,
    drug.cost_category AS cost,
    drug.side_effects AS known_side_effects,
    CASE
        WHEN total_score >= 0.25 THEN 'HIGH'
        WHEN total_score >= 0.15 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS prediction_confidence,
    CASE
        WHEN SIZE(targeted_disease_proteins) > 0 AND SIZE(relevant_pathways) > 0 THEN 'Strong mechanistic evidence'
        WHEN SIZE(similar_diseases_treated) > 0 THEN 'Similar disease evidence'
        WHEN drug_similarity_score > 0 THEN 'Drug similarity evidence'
        ELSE 'Indirect evidence'
    END AS evidence_type

ORDER BY total_confidence_score DESC
LIMIT 20;
