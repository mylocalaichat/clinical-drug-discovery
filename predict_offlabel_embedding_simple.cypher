// =====================================================================
// SIMPLE EMBEDDING-BASED OFF-LABEL DRUG PREDICTION
// =====================================================================
// Single query using direct embedding similarity
//
// PREREQUISITES: Run python3 generate_and_query_embeddings.py first
//
// USAGE: Change the disease name below
// =====================================================================

// Get target disease with its embedding
MATCH (disease:Node {node_name: 'Metabolic Syndrome'})
WHERE disease.node_type = 'disease'
  AND EXISTS(disease.embedding)

// Get all candidate drugs with embeddings (not currently treating this disease)
MATCH (drug:Node)
WHERE drug.node_type = 'drug'
  AND EXISTS(drug.embedding)
  AND NOT EXISTS {
      (drug)-[:RELATES {relation: 'drug_treats_disease'}]->(disease)
  }

// Calculate cosine similarity between drug and disease embeddings
// Formula: cosine_sim = dot(A,B) / (||A|| * ||B||)
WITH disease, drug,
     // Dot product: sum of element-wise multiplication
     reduce(dot = 0.0, i IN range(0, size(disease.embedding)-1) |
            dot + disease.embedding[i] * drug.embedding[i]) AS dot_product,

     // Magnitude of disease embedding: sqrt(sum of squares)
     sqrt(reduce(sum_sq = 0.0, i IN range(0, size(disease.embedding)-1) |
                 sum_sq + disease.embedding[i] * disease.embedding[i])) AS disease_magnitude,

     // Magnitude of drug embedding
     sqrt(reduce(sum_sq = 0.0, i IN range(0, size(drug.embedding)-1) |
                 sum_sq + drug.embedding[i] * drug.embedding[i])) AS drug_magnitude

// Calculate cosine similarity
WITH disease, drug,
     CASE WHEN disease_magnitude > 0 AND drug_magnitude > 0
          THEN dot_product / (disease_magnitude * drug_magnitude)
          ELSE 0.0
     END AS embedding_similarity

WHERE embedding_similarity > 0.3  // Minimum similarity threshold

// Get supporting evidence (graph structure)
OPTIONAL MATCH (drug)-[:RELATES {relation: 'drug_targets_protein'}]->(protein:Node)
              -[:RELATES {relation: 'protein_disease_association'}]->(disease)

WITH disease, drug, embedding_similarity,
     collect(DISTINCT protein.node_name) AS shared_proteins

OPTIONAL MATCH (drug)-[:RELATES {relation: 'drug_targets_protein'}]->(:Node)
              -[:RELATES {relation: 'protein_participates_pathway'}]->(pathway:Node)
              -[:RELATES {relation: 'pathway_disease_association'}]->(disease)

WITH disease, drug, embedding_similarity, shared_proteins,
     collect(DISTINCT pathway.node_name) AS shared_pathways

// Return results
RETURN
    drug.node_name AS candidate_drug,
    disease.node_name AS target_disease,
    round(embedding_similarity * 1000) / 1000 AS similarity_score,

    CASE
        WHEN embedding_similarity >= 0.5 THEN 'HIGH'
        WHEN embedding_similarity >= 0.35 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS confidence,

    SIZE(shared_proteins) AS protein_connections,
    SIZE(shared_pathways) AS pathway_connections,

    shared_proteins[0..3] AS sample_proteins,
    shared_pathways[0..3] AS sample_pathways,

    drug.mechanism AS mechanism,
    drug.drug_class AS drug_class,
    drug.cost_category AS cost,

    CASE
        WHEN SIZE(shared_proteins) > 0
             THEN 'High embedding similarity + direct protein connections'
        WHEN SIZE(shared_pathways) > 0
             THEN 'High embedding similarity + shared pathway involvement'
        ELSE 'High embedding similarity in graph structure'
    END AS rationale

ORDER BY similarity_score DESC
LIMIT 15;


// =====================================================================
// ALTERNATIVE: Context-Based Prediction
// Find drugs similar to the disease's molecular context (proteins/pathways)
// =====================================================================
/*
// Uncomment this section to use context-based approach instead:

MATCH (disease:Node {node_name: 'Cardiovascular Disease'})
WHERE disease.node_type = 'disease'

// Get disease context: proteins and pathways associated with disease
MATCH (disease)-[:RELATES]-(context:Node)
WHERE context.node_type IN ['gene/protein', 'pathway']
  AND EXISTS(context.embedding)

// Calculate average context embedding
WITH disease,
     [i IN range(0, 63) |  // Assuming 64-dimensional embeddings (0-63)
      reduce(sum = 0.0, node IN collect(context) |
             sum + node.embedding[i]) / toFloat(count(context))
     ] AS avg_context_embedding,
     collect(context.node_name) AS context_entities

// Find drugs with embeddings similar to this context
MATCH (drug:Node)
WHERE drug.node_type = 'drug'
  AND EXISTS(drug.embedding)
  AND NOT EXISTS {(drug)-[:RELATES {relation: 'drug_treats_disease'}]->(disease)}

// Calculate similarity to context
WITH disease, drug, avg_context_embedding, context_entities,
     reduce(dot = 0.0, i IN range(0, size(avg_context_embedding)-1) |
            dot + avg_context_embedding[i] * drug.embedding[i]) AS dot_product,
     sqrt(reduce(sum_sq = 0.0, i IN range(0, size(avg_context_embedding)-1) |
                 sum_sq + avg_context_embedding[i] * avg_context_embedding[i])) AS context_magnitude,
     sqrt(reduce(sum_sq = 0.0, i IN range(0, size(drug.embedding)-1) |
                 sum_sq + drug.embedding[i] * drug.embedding[i])) AS drug_magnitude

WITH disease, drug, context_entities,
     dot_product / (context_magnitude * drug_magnitude) AS context_similarity

WHERE context_similarity > 0.35

RETURN
    drug.node_name AS candidate_drug,
    disease.node_name AS target_disease,
    round(context_similarity * 1000) / 1000 AS context_similarity_score,
    SIZE(context_entities) AS disease_context_size,
    context_entities[0..5] AS disease_context_sample,
    drug.mechanism AS mechanism,
    'Similar to disease molecular context' AS rationale

ORDER BY context_similarity_score DESC
LIMIT 15;
*/


// =====================================================================
// ALTERNATIVE: Analogy-Based Prediction
// Find: "What drug is to Disease B, as Known Drug is to Disease A?"
// Example: Metformin:Diabetes :: ?:Metabolic Syndrome
// =====================================================================
/*
// Uncomment to use analogy approach:

// Known relationship
MATCH (disease_a:Node {node_name: 'Type 2 Diabetes'})
MATCH (known_drug:Node {node_name: 'Metformin'})
MATCH (disease_b:Node {node_name: 'Metabolic Syndrome'})

WHERE disease_a.node_type = 'disease'
  AND disease_b.node_type = 'disease'
  AND known_drug.node_type = 'drug'
  AND EXISTS(disease_a.embedding)
  AND EXISTS(disease_b.embedding)
  AND EXISTS(known_drug.embedding)

// Calculate target embedding: disease_b + (known_drug - disease_a)
WITH disease_a, disease_b, known_drug,
     [i IN range(0, size(disease_b.embedding)-1) |
      disease_b.embedding[i] + known_drug.embedding[i] - disease_a.embedding[i]
     ] AS target_embedding

// Find drugs closest to this target
MATCH (candidate:Node)
WHERE candidate.node_type = 'drug'
  AND EXISTS(candidate.embedding)
  AND candidate.node_id <> known_drug.node_id

WITH disease_a, disease_b, known_drug, candidate, target_embedding,
     reduce(dot = 0.0, i IN range(0, size(target_embedding)-1) |
            dot + target_embedding[i] * candidate.embedding[i]) AS dot_product,
     sqrt(reduce(sum_sq = 0.0, i IN range(0, size(target_embedding)-1) |
                 sum_sq + target_embedding[i] * target_embedding[i])) AS target_magnitude,
     sqrt(reduce(sum_sq = 0.0, i IN range(0, size(candidate.embedding)-1) |
                 sum_sq + candidate.embedding[i] * candidate.embedding[i])) AS candidate_magnitude

WITH disease_a, disease_b, known_drug, candidate,
     dot_product / (target_magnitude * candidate_magnitude) AS analogy_score

WHERE analogy_score > 0.4

RETURN
    candidate.node_name AS candidate_drug,
    disease_b.node_name AS target_disease,
    round(analogy_score * 1000) / 1000 AS analogy_similarity,
    known_drug.node_name + ':' + disease_a.node_name + ' :: ' +
    candidate.node_name + ':' + disease_b.node_name AS analogy,
    candidate.mechanism AS mechanism,
    'Embedding-based analogy reasoning' AS rationale

ORDER BY analogy_score DESC
LIMIT 15;
*/
