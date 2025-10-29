# Drug Repurposing Example Graph

## Overview

This example demonstrates how graph-based inference can identify potential new drug-disease treatments through similarity patterns.

## Graph Structure

**Total Nodes: 10**
- 2 Drugs: Metformin, Aspirin
- 3 Diseases: Type 2 Diabetes, Cardiovascular Disease, Metabolic Syndrome
- 3 Proteins: AMPK, COX2, IL6
- 2 Pathways: Glucose Metabolism Pathway, Inflammation Pathway

**Total Relationships: 18**

## The Inference Pattern

### Known Facts

1. **Drug A (Metformin)** treats **Disease B (Type 2 Diabetes)**
   - Mechanism: Metformin → AMPK → Glucose Metabolism → Type 2 Diabetes
   - Evidence: FDA approved, high efficacy

2. **Drug E (Aspirin)** treats **Disease C (Cardiovascular Disease)**
   - Mechanism: Aspirin → COX2 → Inflammation Pathway → Cardiovascular Disease
   - Evidence: FDA approved, moderate efficacy

3. **Drug Similarity**: Metformin is similar to Aspirin
   - Similarity Score: 0.65
   - Shared Feature: Anti-inflammatory effects

4. **Disease Similarities**:
   - Type 2 Diabetes similar to Metabolic Syndrome (score: 0.75, shared: metabolic dysfunction)
   - Cardiovascular Disease similar to Metabolic Syndrome (score: 0.70, shared: cardiovascular risk)

5. **Shared Mechanism**: All three diseases involve IL6 (inflammatory cytokine)

### The Inference

**Because:**
- Metformin and Aspirin share anti-inflammatory properties (drug similarity)
- Type 2 Diabetes and Metabolic Syndrome are highly similar (disease similarity)
- Cardiovascular Disease and Metabolic Syndrome are similar (disease similarity)
- All three diseases share IL6 as a common mechanism

**We can infer:**
**Metformin might be effective for treating Metabolic Syndrome**

This is a new hypothesis that doesn't exist in the original data but is discovered through graph pattern matching!

## Key Properties for Filtering

All example nodes have:
- `example_set: 'drug_repurposing'` - to easily filter/delete this example

Drugs have:
- `mechanism` - how the drug works
- `drug_class` - drug category

Diseases have:
- `disease_category` - type of disease
- `severity` - disease severity level

Proteins have:
- `function` - biological function

Pathways have:
- `pathway_type` - category of pathway

Relationships have:
- `relation` - machine-readable relationship type
- `display_relation` - human-readable relationship name
- `example_set: 'drug_repurposing'` - for filtering
- Type-specific properties (e.g., `effect`, `confidence`, `similarity_score`)

## How to Use This Example

### Load the Example
```bash
python3 << 'EOF'
from neo4j import GraphDatabase
import re

driver = GraphDatabase.driver('bolt://localhost:7687', auth=None)
session = driver.session()

# Delete existing example
session.run("MATCH (n) WHERE n.example_set = 'drug_repurposing' DETACH DELETE n")

# Read and parse the cypher file
with open('example_drug_repurposing.cypher', 'r') as f:
    content = f.read()

create_pattern = r'CREATE\s+\([^)]+:[^)]+\{[^}]+\}\);'
match_pattern = r'MATCH.*?(?:CREATE|MERGE).*?(?:\{[^}]+\}\))?(?:->\([^)]*\))?;'

node_creates = re.findall(create_pattern, content, re.DOTALL)
rel_creates = re.findall(match_pattern, content, re.DOTALL)

# Execute creates
for statement in node_creates:
    session.run(statement.strip().rstrip(';'))

for statement in rel_creates:
    session.run(statement.strip().rstrip(';'))

session.close()
driver.close()
EOF
```

### Run the Queries
```bash
python3 example_queries.py
```

### Delete the Example
```cypher
MATCH (n) WHERE n.example_set = 'drug_repurposing' DETACH DELETE n
```

## Query #8: The Drug Repurposing Inference Query

This is the key query that demonstrates the inference pattern:

```cypher
// Find potential new drug-disease treatments based on:
// 1. Drug similarity (Drug A similar to Drug E)
// 2. Disease similarity (Disease B/C similar to Disease D)
// 3. Known treatment (Drug A treats B, Drug E treats C)
// INFERENCE: Drug A might treat Disease D

MATCH (drugA)-[treats:RELATES]->(diseaseB),
      (drugE)-[:RELATES]->(diseaseC),
      (drugA)-[drugSim:RELATES]->(drugE),
      (diseaseB)-[diseaseSim1:RELATES]->(diseaseD),
      (diseaseC)-[diseaseSim2:RELATES]->(diseaseD)
WHERE drugA.example_set = 'drug_repurposing'
AND treats.relation = 'drug_treats_disease'
AND drugSim.relation = 'drug_similarity'
AND diseaseSim1.relation = 'disease_similarity'
AND diseaseSim2.relation = 'disease_similarity'
AND diseaseD.node_name = 'Metabolic Syndrome'

// Check that drugA doesn't already treat diseaseD
AND NOT EXISTS {
    MATCH (drugA)-[r:RELATES]->(diseaseD)
    WHERE r.relation = 'drug_treats_disease'
}

RETURN DISTINCT
    drugA.node_name as CandidateDrug,
    diseaseD.node_name as TargetDisease,
    drugA.mechanism as DrugMechanism,
    diseaseB.node_name as KnownTreatedDisease1,
    diseaseC.node_name as KnownTreatedDisease2,
    drugE.node_name as SimilarDrug,
    drugSim.similarity_score as DrugSimilarity,
    diseaseSim1.similarity_score as DiseaseSimilarity1,
    diseaseSim2.similarity_score as DiseaseSimilarity2
```

### Result

```
{
  'CandidateDrug': 'Metformin',
  'TargetDisease': 'Metabolic Syndrome',
  'DrugMechanism': 'AMPK activator',
  'KnownTreatedDisease1': 'Type 2 Diabetes',
  'KnownTreatedDisease2': 'Cardiovascular Disease',
  'SimilarDrug': 'Aspirin',
  'DrugSimilarity': 0.65,
  'DiseaseSimilarity1': 0.75,
  'DiseaseSimilarity2': 0.7
}
```

## Extending This Example

You can modify this example to:

1. **Add more drugs** - Create additional drugs with different mechanisms
2. **Add more diseases** - Include related diseases with different similarity scores
3. **Add more intermediate nodes** - Include more proteins, pathways, biological processes
4. **Adjust similarity scores** - Experiment with different thresholds
5. **Add more relationship types** - Include negative regulation, contraindications, etc.

## Real-World Application

In your actual application, you would:

1. **Calculate drug similarities** using:
   - Chemical structure similarity
   - Target protein overlap
   - Side effect profiles
   - Gene expression signatures

2. **Calculate disease similarities** using:
   - Shared genetic markers
   - Common symptoms
   - Overlapping pathways
   - Co-occurrence patterns

3. **Weight the evidence** by:
   - Combining multiple similarity metrics
   - Considering path lengths and strengths
   - Incorporating clinical trial data
   - Using machine learning to score candidates

4. **Validate predictions** through:
   - Literature review
   - Clinical trials
   - Electronic health records analysis
