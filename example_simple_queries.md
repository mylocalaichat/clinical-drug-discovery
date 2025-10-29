# Simple Query Examples Using is_example Attribute

All example nodes and edges have `is_example: true` for easy filtering.

## Basic Node Queries

### 1. Get all example nodes
```cypher
MATCH (n)
WHERE n.is_example = true
RETURN n
```

Or using inline filter:
```cypher
MATCH (n {is_example: true})
RETURN n
```

### 2. Count example nodes by type
```cypher
MATCH (n {is_example: true})
RETURN n.node_type as type, count(*) as count
ORDER BY count DESC
```

### 3. Get only example drugs
```cypher
MATCH (n {is_example: true})
WHERE n.node_type = 'drug'
RETURN n.node_name, n.mechanism, n.drug_class
```

### 4. Get example drugs with specific properties
```cypher
MATCH (n {is_example: true})
WHERE n.node_type = 'drug'
AND n.cost_category = 'low'
AND n.fda_approval_year < 2000
RETURN n.node_name, n.fda_approval_year, n.daily_dose_mg
```

### 5. Get diseases with high prevalence
```cypher
MATCH (d {is_example: true})
WHERE d.node_type = 'disease'
AND d.prevalence_per_100k > 5000
RETURN d.node_name, d.prevalence_per_100k, d.severity
ORDER BY d.prevalence_per_100k DESC
```

### 6. Get therapeutic target proteins
```cypher
MATCH (p {is_example: true})
WHERE p.node_type = 'gene/protein'
AND p.is_therapeutic_target = true
RETURN p.node_name, p.gene_symbol, p.druggability_score
ORDER BY p.druggability_score DESC
```

## Relationship Queries

### 7. Get all example relationships
```cypher
MATCH (a)-[r {is_example: true}]->(b)
RETURN a.node_name, type(r), r.display_relation, b.node_name
LIMIT 10
```

### 8. Find high-confidence drug-target interactions
```cypher
MATCH (drug)-[r {is_example: true}]->(protein)
WHERE drug.node_type = 'drug'
AND protein.node_type = 'gene/protein'
AND r.confidence > 0.9
RETURN drug.node_name, r.confidence, r.effect, protein.node_name
```

### 9. Find disease similarities above threshold
```cypher
MATCH (d1)-[r {is_example: true}]->(d2)
WHERE r.relation = 'disease_similarity'
AND r.similarity_score > 0.7
RETURN d1.node_name, d2.node_name, r.similarity_score, r.shared_features
```

### 10. Find pathways with high therapeutic potential
```cypher
MATCH (pathway)-[r {is_example: true}]->(disease)
WHERE pathway.node_type = 'pathway'
AND r.therapeutic_potential = 'high'
RETURN pathway.node_name, disease.node_name,
       r.dysregulation_magnitude, r.clinical_relevance_score
```

## Path Queries

### 11. Find all paths from drugs to diseases (example only)
```cypher
MATCH path = (drug {is_example: true})-[*1..3]->(disease {is_example: true})
WHERE drug.node_type = 'drug'
AND disease.node_type = 'disease'
RETURN drug.node_name,
       length(path) as path_length,
       disease.node_name
LIMIT 10
```

### 12. Drug repurposing inference (example subgraph only)
```cypher
MATCH (drugA {is_example: true})-[treats:RELATES {is_example: true}]->(diseaseB {is_example: true}),
      (drugE {is_example: true})-[:RELATES {is_example: true}]->(diseaseC {is_example: true}),
      (drugA)-[drugSim:RELATES {is_example: true}]->(drugE),
      (diseaseB)-[diseaseSim1:RELATES {is_example: true}]->(diseaseD {is_example: true}),
      (diseaseC)-[diseaseSim2:RELATES {is_example: true}]->(diseaseD)
WHERE treats.relation = 'drug_treats_disease'
AND drugSim.relation = 'drug_similarity'
AND diseaseSim1.relation = 'disease_similarity'
AND diseaseSim2.relation = 'disease_similarity'
RETURN drugA.node_name as CandidateDrug,
       diseaseD.node_name as TargetDisease,
       drugSim.similarity_score as DrugSimilarity,
       diseaseSim1.similarity_score as DiseaseSim1,
       diseaseSim2.similarity_score as DiseaseSim2
```

## Aggregation Queries

### 13. Average similarity scores
```cypher
MATCH ()-[r {is_example: true}]->()
WHERE r.similarity_score IS NOT NULL
RETURN r.relation as relationship_type,
       avg(r.similarity_score) as avg_similarity,
       count(*) as count
```

### 14. Protein expression in diseases
```cypher
MATCH (protein {is_example: true})-[r {is_example: true}]->(disease {is_example: true})
WHERE protein.node_type = 'gene/protein'
AND disease.node_type = 'disease'
AND r.fold_change_in_disease IS NOT NULL
RETURN protein.node_name,
       count(disease) as num_diseases,
       avg(r.fold_change_in_disease) as avg_fold_change
ORDER BY num_diseases DESC
```

### 15. Count relationships by type
```cypher
MATCH ()-[r {is_example: true}]->()
RETURN r.relation as relation_type,
       r.display_relation as display_name,
       count(*) as count
ORDER BY count DESC
```

## Deletion Queries

### 16. Delete all example data
```cypher
MATCH (n {is_example: true})
DETACH DELETE n
```

### 17. Delete only example relationships (keep nodes)
```cypher
MATCH ()-[r {is_example: true}]->()
DELETE r
```

### 18. Delete specific example node type
```cypher
MATCH (n {is_example: true})
WHERE n.node_type = 'pathway'
DETACH DELETE n
```

## Property Filtering Examples

### 19. Filter drugs by multiple properties
```cypher
MATCH (drug {is_example: true})
WHERE drug.node_type = 'drug'
AND drug.bioavailability > 0.5
AND drug.half_life_hours > 1
AND drug.molecular_weight < 200
RETURN drug.node_name, drug.bioavailability,
       drug.half_life_hours, drug.molecular_weight
```

### 20. Filter relationships by clinical significance
```cypher
MATCH (drug {is_example: true})-[r {is_example: true}]->(protein {is_example: true})
WHERE r.clinical_significance = 'high'
AND r.binding_affinity_nm < 10
RETURN drug.node_name, protein.node_name,
       r.binding_affinity_nm, r.ki_value
ORDER BY r.binding_affinity_nm
```

## Tips

- Use `{is_example: true}` as an inline filter for cleaner queries
- Use `WHERE n.is_example = true` when you need additional WHERE conditions
- Always include `is_example: true` in both node and relationship filters to ensure you're only querying the example data
- The example data is completely isolated from your real data by the `is_example` attribute
