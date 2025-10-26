// ENHANCED OFF-LABEL DRUG DISCOVERY QUERY WITH CLINICAL EVIDENCE
// Works for ANY disease using graph topology + clinical evidence scores
// Finds drugs through diseases that share proteins AND considers clinical evidence
//
// Usage: Set disease_id parameter to any disease node_id
// Example: WITH '15564' as disease_id  // Castleman's disease

WITH $disease_id as disease_id  // PARAMETER - Replace with target disease ID

// Step 1: Find target disease and its associated proteins
MATCH (target_disease:PrimeKGNode {node_id: disease_id})
MATCH (target_disease)-[:DISEASE_PROTEIN]-(shared_protein:PrimeKGNode)
WHERE shared_protein.node_type = 'gene/protein'

// Step 2: Find similar diseases (diseases that share proteins with target)
MATCH (similar_disease:PrimeKGNode)-[:DISEASE_PROTEIN]-(shared_protein)
WHERE similar_disease.node_type = 'disease'
  AND similar_disease.node_id <> target_disease.node_id

// Step 3: Find drugs indicated for these similar diseases
MATCH (candidate_drug:PrimeKGNode)-[:INDICATION]-(similar_disease)
WHERE candidate_drug.node_type = 'drug'
  // Exclude drugs already indicated for target disease
  AND NOT EXISTS {
    MATCH (candidate_drug)-[:INDICATION]-(target_disease)
  }

// Step 4: Score and aggregate results
WITH candidate_drug,
     count(DISTINCT similar_disease) as similar_disease_count,
     count(DISTINCT shared_protein) as shared_protein_count,
     collect(DISTINCT similar_disease.node_name)[0..5] as similar_diseases,
     collect(DISTINCT shared_protein.node_name) as shared_proteins,
     // Base relevance score: shared proteins (10pts each) + similar diseases (1pt each)
     (count(DISTINCT shared_protein) * 10) + count(DISTINCT similar_disease) as base_score

// Step 5: Add clinical evidence if available
OPTIONAL MATCH (candidate_drug)-[clinical:CLINICAL_EVIDENCE]->(target_disease)

// Step 6: Get drug targets for additional context
OPTIONAL MATCH (candidate_drug)-[r:DRUG_PROTEIN]-(drug_target:PrimeKGNode)
WHERE drug_target.node_type = 'gene/protein'

WITH candidate_drug,
     base_score,
     shared_protein_count,
     similar_disease_count,
     shared_proteins,
     similar_diseases,
     collect(DISTINCT drug_target.node_name)[0..10] as drug_targets,
     collect(DISTINCT r.display_relation) as interaction_types,
     COALESCE(clinical.score, 0.0) as clinical_score,
     COALESCE(clinical.confidence, 0.0) as clinical_confidence,
     COALESCE(clinical.evidence_strength, 'none') as evidence_strength,
     // Enhanced relevance score: base score + clinical evidence boost/penalty
     // Positive clinical scores get significant boost, negative scores get penalty
     base_score + (COALESCE(clinical.score, 0.0) * 50) as enhanced_score

// Step 7: Return results ordered by enhanced relevance
RETURN candidate_drug.node_id as drug_id,
       candidate_drug.node_name as drug_name,
       candidate_drug.description as description,
       substring(coalesce(candidate_drug.mechanism_of_action, 'N/A'), 0, 300) as mechanism,
       enhanced_score,
       base_score,
       clinical_score,
       clinical_confidence,
       evidence_strength,
       shared_protein_count,
       shared_proteins,
       similar_disease_count,
       similar_diseases,
       drug_targets,
       interaction_types,
       CASE 
         WHEN clinical_score > 0 THEN 'Graph + Positive Clinical' 
         WHEN clinical_score < 0 THEN 'Graph + Negative Clinical'
         ELSE 'Graph Only' 
       END as evidence_type
ORDER BY enhanced_score DESC, shared_protein_count DESC
LIMIT 20

// INTERPRETATION:
// - enhanced_score = base relevance + clinical evidence boost/penalty
// - clinical_score = proportional score (-1 to +1) from clinical text analysis
// - evidence_strength = 'strong', 'moderate', 'weak', or 'none'
// - Positive clinical_score boosts ranking, negative penalizes it
// - shared_protein_count = number of proteins shared between target and similar diseases
// - similar_disease_count = number of similar diseases this drug treats
// - evidence_type = shows if clinical evidence influenced the ranking
