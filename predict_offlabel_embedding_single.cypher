// =====================================================================
// SINGLE EMBEDDING-BASED OFF-LABEL DRUG PREDICTION QUERY
// =====================================================================
// This query uses node embeddings to predict off-label drugs for a disease
// using multiple strategies combined into a single comprehensive query
//
// PREREQUISITES:
// 1. Run: python3 generate_and_query_embeddings.py
//    (This generates and stores embeddings as node.embedding property)
//
// USAGE:
// Set the disease name parameter below
// =====================================================================

// NOTE: Memgraph doesn't have built-in cosine similarity, so we calculate it using Cypher
// Helper: Cosine similarity = dot(A,B) / (||A|| * ||B||)

// =========================
// SET TARGET DISEASE HERE
// =========================
MATCH (target_disease:Node {node_name: 'Metabolic Syndrome'})
WHERE target_disease.node_type = 'disease'
  AND EXISTS(target_disease.embedding)

// Get disease embedding and calculate its magnitude
WITH target_disease,
     target_disease.embedding AS disease_emb,
     sqrt(reduce(s = 0.0, i IN range(0, size(target_disease.embedding)-1) |
          s + target_disease.embedding[i] * target_disease.embedding[i])) AS disease_norm

// =========================
// STRATEGY 1: Direct Drug-Disease Embedding Similarity
// =========================
MATCH (drug:Node)
WHERE drug.node_type = 'drug'
  AND EXISTS(drug.embedding)
  AND NOT EXISTS {
      (drug)-[:RELATES {relation: 'drug_treats_disease'}]->(target_disease)
  }

// Calculate cosine similarity: dot product / (norm1 * norm2)
WITH target_disease, disease_emb, disease_norm, drug,
     // Dot product
     reduce(s = 0.0, i IN range(0, size(disease_emb)-1) |
            s + disease_emb[i] * drug.embedding[i]) AS dot_product,
     // Drug embedding magnitude
     sqrt(reduce(s = 0.0, i IN range(0, size(drug.embedding)-1) |
          s + drug.embedding[i] * drug.embedding[i])) AS drug_norm

WITH target_disease, drug,
     CASE WHEN disease_norm > 0 AND drug_norm > 0
          THEN dot_product / (disease_norm * drug_norm)
          ELSE 0.0
     END AS direct_similarity

// =========================
// STRATEGY 2: Context-Based Similarity
// Find drugs similar to disease's protein/pathway context
// =========================
OPTIONAL MATCH (target_disease)-[:RELATES]-(context:Node)
WHERE context.node_type IN ['gene/protein', 'pathway']
  AND EXISTS(context.embedding)

// Calculate average context embedding
WITH target_disease, drug, direct_similarity,
     CASE WHEN count(context) > 0
          THEN [i IN range(0, size(target_disease.embedding)-1) |
                reduce(s = 0.0, c IN collect(context.embedding) |
                       s + c[i]) / toFloat(count(context))]
          ELSE NULL
     END AS avg_context_emb,
     count(context) AS context_count

// Calculate similarity to context
WITH target_disease, drug, direct_similarity, context_count,
     CASE WHEN avg_context_emb IS NOT NULL
          THEN (
              // Dot product
              reduce(s = 0.0, i IN range(0, size(avg_context_emb)-1) |
                     s + avg_context_emb[i] * drug.embedding[i])
              /
              // Norms
              (sqrt(reduce(s = 0.0, i IN range(0, size(avg_context_emb)-1) |
                          s + avg_context_emb[i] * avg_context_emb[i]))
               *
               sqrt(reduce(s = 0.0, i IN range(0, size(drug.embedding)-1) |
                          s + drug.embedding[i] * drug.embedding[i])))
          )
          ELSE 0.0
     END AS context_similarity

// =========================
// STRATEGY 3: Similar Disease Treatment
// Find embeddings of diseases similar to target, check if drug treats them
// =========================
OPTIONAL MATCH (similar_disease:Node)
WHERE similar_disease.node_type = 'disease'
  AND similar_disease.node_id <> target_disease.node_id
  AND EXISTS(similar_disease.embedding)

// Calculate disease-disease similarity
WITH target_disease, drug, direct_similarity, context_similarity, context_count,
     similar_disease,
     CASE WHEN similar_disease IS NOT NULL
          THEN (
              reduce(s = 0.0, i IN range(0, size(target_disease.embedding)-1) |
                     s + target_disease.embedding[i] * similar_disease.embedding[i])
              /
              (sqrt(reduce(s = 0.0, i IN range(0, size(target_disease.embedding)-1) |
                          s + target_disease.embedding[i] * target_disease.embedding[i]))
               *
               sqrt(reduce(s = 0.0, i IN range(0, size(similar_disease.embedding)-1) |
                          s + similar_disease.embedding[i] * similar_disease.embedding[i])))
          )
          ELSE 0.0
     END AS disease_disease_similarity

// Check if drug treats similar disease
OPTIONAL MATCH (drug)-[treats:RELATES]->(similar_disease)
WHERE treats.relation = 'drug_treats_disease'
  AND disease_disease_similarity > 0.6

WITH target_disease, drug, direct_similarity, context_similarity, context_count,
     max(disease_disease_similarity) AS max_similar_disease_score,
     count(DISTINCT similar_disease) AS similar_diseases_treated

// =========================
// STRATEGY 4: Protein Target Similarity
// Find if drug targets proteins with embeddings similar to disease-associated proteins
// =========================
OPTIONAL MATCH (target_disease)-[:RELATES {relation: 'protein_disease_association'}]->(disease_protein:Node)
WHERE disease_protein.node_type = 'gene/protein'
  AND EXISTS(disease_protein.embedding)

OPTIONAL MATCH (drug)-[:RELATES {relation: 'drug_targets_protein'}]->(drug_target:Node)
WHERE drug_target.node_type = 'gene/protein'
  AND EXISTS(drug_target.embedding)

// Calculate protein-protein similarity
WITH target_disease, drug, direct_similarity, context_similarity, context_count,
     max_similar_disease_score, similar_diseases_treated,
     disease_protein, drug_target,
     CASE WHEN disease_protein IS NOT NULL AND drug_target IS NOT NULL
          THEN (
              reduce(s = 0.0, i IN range(0, size(disease_protein.embedding)-1) |
                     s + disease_protein.embedding[i] * drug_target.embedding[i])
              /
              (sqrt(reduce(s = 0.0, i IN range(0, size(disease_protein.embedding)-1) |
                          s + disease_protein.embedding[i] * disease_protein.embedding[i]))
               *
               sqrt(reduce(s = 0.0, i IN range(0, size(drug_target.embedding)-1) |
                          s + drug_target.embedding[i] * drug_target.embedding[i])))
          )
          ELSE 0.0
     END AS protein_similarity

WITH target_disease, drug, direct_similarity, context_similarity, context_count,
     max_similar_disease_score, similar_diseases_treated,
     max(protein_similarity) AS max_protein_similarity,
     count(DISTINCT CASE WHEN protein_similarity > 0.7 THEN drug_target END) AS high_similarity_proteins

// =========================
// AGGREGATE SCORES
// =========================
WITH target_disease, drug,
     // Individual strategy scores
     direct_similarity,
     context_similarity,
     COALESCE(max_similar_disease_score, 0.0) AS similar_disease_score,
     COALESCE(max_protein_similarity, 0.0) AS protein_target_score,

     // Evidence counts
     context_count,
     similar_diseases_treated,
     high_similarity_proteins,

     // Combined score (weighted average)
     (0.35 * direct_similarity +
      0.25 * context_similarity +
      0.25 * COALESCE(max_similar_disease_score, 0.0) +
      0.15 * COALESCE(max_protein_similarity, 0.0)) AS combined_score

WHERE combined_score > 0.1  // Minimum threshold

// =========================
// GATHER SUPPORTING EVIDENCE
// =========================
OPTIONAL MATCH (drug)-[:RELATES {relation: 'drug_targets_protein'}]->(target:Node)
              -[:RELATES {relation: 'protein_disease_association'}]->(target_disease)
WHERE target.node_type = 'gene/protein'

WITH target_disease, drug,
     direct_similarity, context_similarity, similar_disease_score, protein_target_score,
     context_count, similar_diseases_treated, high_similarity_proteins,
     combined_score,
     collect(DISTINCT target.node_name) AS direct_protein_targets

OPTIONAL MATCH (drug)-[:RELATES {relation: 'drug_targets_protein'}]->(p:Node)
              -[:RELATES {relation: 'protein_participates_pathway'}]->(pathway:Node)
              -[:RELATES {relation: 'pathway_disease_association'}]->(target_disease)

WITH target_disease, drug,
     direct_similarity, context_similarity, similar_disease_score, protein_target_score,
     context_count, similar_diseases_treated, high_similarity_proteins,
     combined_score, direct_protein_targets,
     collect(DISTINCT pathway.node_name) AS relevant_pathways

// =========================
// RETURN RESULTS
// =========================
RETURN
    // Drug information
    drug.node_name AS candidate_drug,
    drug.drug_class AS drug_class,
    drug.mechanism AS mechanism,
    target_disease.node_name AS target_disease,

    // Overall score
    round(combined_score * 1000) / 1000 AS overall_confidence_score,
    CASE
        WHEN combined_score >= 0.35 THEN 'HIGH'
        WHEN combined_score >= 0.20 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS prediction_confidence,

    // Individual strategy scores
    round(direct_similarity * 1000) / 1000 AS embedding_similarity,
    round(context_similarity * 1000) / 1000 AS context_similarity,
    round(similar_disease_score * 1000) / 1000 AS similar_disease_score,
    round(protein_target_score * 1000) / 1000 AS protein_target_score,

    // Evidence counts
    context_count AS disease_context_size,
    similar_diseases_treated AS treats_similar_diseases,
    high_similarity_proteins AS similar_protein_targets,
    SIZE(direct_protein_targets) AS direct_protein_connections,
    SIZE(relevant_pathways) AS pathway_connections,

    // Evidence details
    CASE WHEN SIZE(direct_protein_targets) > 0
         THEN direct_protein_targets[0..3]
         ELSE []
    END AS sample_protein_targets,
    CASE WHEN SIZE(relevant_pathways) > 0
         THEN relevant_pathways[0..3]
         ELSE []
    END AS sample_pathways,

    // Drug properties
    drug.administration_route AS route,
    drug.cost_category AS cost,
    drug.side_effects AS side_effects,

    // Primary evidence explanation
    CASE
        WHEN direct_similarity > 0.4 THEN 'Strong direct embedding similarity to disease'
        WHEN context_similarity > 0.4 THEN 'High similarity to disease molecular context'
        WHEN similar_disease_score > 0.5 THEN 'Treats diseases with similar embeddings'
        WHEN protein_target_score > 0.6 THEN 'Targets proteins similar to disease-associated proteins'
        WHEN SIZE(direct_protein_targets) > 0 THEN 'Has direct protein connections to disease'
        ELSE 'Moderate evidence across multiple signals'
    END AS primary_rationale

ORDER BY overall_confidence_score DESC, prediction_confidence DESC
LIMIT 20;
