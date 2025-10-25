// GENERIC OFF-LABEL DRUG DISCOVERY QUERY
// Works for ANY disease using only graph topology
// Finds drugs through diseases that share proteins
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
     collect(DISTINCT shared_protein.node_name) as shared_proteins

// Step 5: Get drug targets for additional context
OPTIONAL MATCH (candidate_drug)-[r:DRUG_PROTEIN]-(drug_target:PrimeKGNode)
WHERE drug_target.node_type = 'gene/protein'

WITH candidate_drug,
     shared_protein_count,
     similar_disease_count,
     shared_proteins,
     similar_diseases,
     collect(DISTINCT drug_target.node_name)[0..10] as drug_targets,
     collect(DISTINCT r.display_relation) as interaction_types,
     // Relevance score: shared proteins (10pts each) + similar diseases (1pt each)
     (shared_protein_count * 10) + similar_disease_count as relevance_score

// Step 6: Return results ordered by relevance
RETURN candidate_drug.node_id as drug_id,
       candidate_drug.node_name as drug_name,
       candidate_drug.description as description,
       substring(coalesce(candidate_drug.mechanism_of_action, 'N/A'), 0, 300) as mechanism,
       relevance_score,
       shared_protein_count,
       shared_proteins,
       similar_disease_count,
       similar_diseases,
       drug_targets,
       interaction_types
ORDER BY relevance_score DESC, shared_protein_count DESC
LIMIT 20

// INTERPRETATION:
// - Higher relevance_score = more similar to diseases sharing proteins with target
// - shared_protein_count = number of proteins shared between target and similar diseases
// - similar_disease_count = number of similar diseases this drug treats
// - shared_proteins = which proteins create the connection
// - similar_diseases = example diseases that are similar to target
// - drug_targets = what proteins this drug actually targets
