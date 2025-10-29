# Drug Repurposing Example - Usage Guide

## Overview

The drug repurposing example has been split into **two scripts** to demonstrate the power of similarity relationships:

1. **`example_drug_repurposing_base.cypher`** - Base graph (10 nodes, 15 relationships)
2. **`example_drug_repurposing_similarities.cypher`** - Similarity edges only (3 relationships)

This separation allows you to:
- First visualize the basic drug-disease-protein network
- Then add similarity edges to see how they enable drug repurposing inference

## Files

### Core Scripts
- `example_drug_repurposing_base.cypher` - Creates nodes and base relationships
- `example_drug_repurposing_similarities.cypher` - Adds only similarity relationships
- `example_drug_repurposing.cypher` - Original complete script (all-in-one)

### Helper Scripts
- `load_example_progressive.py` - Automated progressive loading demo
- `example_queries.py` - 10 example analytical queries
- `example_simple_queries.md` - 20 simple query examples

### Styling
- `memgraph_style.cypherl` - Custom Memgraph visualization style
  - Green = Drugs, Red = Diseases, Blue = Proteins, Orange = Pathways

## Quick Start

### Option 1: Progressive Loading (Recommended)

Run the automated demo:
```bash
python3 load_example_progressive.py
```

This will:
1. Clean up existing example data
2. Load base graph (drugs, diseases, proteins, pathways)
3. Test inference (will fail - no similarities yet)
4. Add similarity edges
5. Test inference again (will succeed!)

### Option 2: Manual Loading

**Step 1: Load Base Graph**
```bash
python3 << 'EOF'
from neo4j import GraphDatabase
import re

driver = GraphDatabase.driver('bolt://localhost:7687', auth=None)
session = driver.session()

# Clean up
session.run("MATCH (n {is_example: true}) DETACH DELETE n")

# Load base graph
with open('example_drug_repurposing_base.cypher', 'r') as f:
    content = f.read()

# Execute statements
for statement in content.split(';'):
    statement = statement.strip()
    if statement and not statement.startswith('//'):
        session.run(statement)

print("Base graph loaded!")
session.close()
driver.close()
EOF
```

**Visualize in Memgraph Lab:**
```cypher
MATCH (n {is_example: true})-[r {is_example: true}]-(m {is_example: true})
RETURN n, r, m
```

At this point you'll see:
- 🟢 2 Drugs (Metformin, Aspirin)
- 🔴 3 Diseases (Type 2 Diabetes, Cardiovascular Disease, Metabolic Syndrome)
- 🔵 3 Proteins (AMPK, COX2, IL6)
- 🟠 2 Pathways (Inflammation, Glucose Metabolism)
- Various colored edges showing interactions

**Step 2: Add Similarity Edges**
```bash
python3 << 'EOF'
from neo4j import GraphDatabase

driver = GraphDatabase.driver('bolt://localhost:7687', auth=None)
session = driver.session()

with open('example_drug_repurposing_similarities.cypher', 'r') as f:
    content = f.read()

for statement in content.split(';'):
    statement = statement.strip()
    if statement and not statement.startswith('//'):
        session.run(statement)

print("Similarity edges added!")
session.close()
driver.close()
EOF
```

**Refresh visualization** - you'll now see 3 purple dashed lines (similarities)

### Option 3: Load Complete Graph (All-in-One)

If you want everything at once:
```bash
python3 << 'EOF'
from neo4j import GraphDatabase
import re

driver = GraphDatabase.driver('bolt://localhost:7687', auth=None)
session = driver.session()

session.run("MATCH (n {is_example: true}) DETACH DELETE n")

with open('example_drug_repurposing.cypher', 'r') as f:
    content = f.read()

create_pattern = r'CREATE\s+\([^)]+:[^)]+\{[^}]+\}\);'
match_pattern = r'MATCH.*?(?:CREATE|MERGE).*?(?:\{[^}]+\}\))?(?:->\([^)]*\))?;'

node_creates = re.findall(create_pattern, content, re.DOTALL)
rel_creates = re.findall(match_pattern, content, re.DOTALL)

for statement in node_creates + rel_creates:
    session.run(statement.strip().rstrip(';'))

print("Complete graph loaded!")
session.close()
driver.close()
EOF
```

## Understanding the Graph

### Base Graph (Step 1)
**Nodes:**
- Drugs: Metformin, Aspirin
- Diseases: Type 2 Diabetes, Cardiovascular Disease, Metabolic Syndrome
- Proteins: AMPK, COX2, IL6
- Pathways: Inflammation, Glucose Metabolism

**Relationships:**
- Drug → targets → Protein
- Protein → regulates → Pathway
- Protein → regulates → Protein
- Protein → involved_in → Disease
- Pathway → dysregulated_in → Disease
- Drug → treats → Disease (2 known treatments)

### Similarity Edges (Step 2)
**3 Critical Edges:**
1. Metformin ↔ Aspirin (drug similarity: 0.65)
2. Type 2 Diabetes ↔ Metabolic Syndrome (disease similarity: 0.75)
3. Cardiovascular Disease ↔ Metabolic Syndrome (disease similarity: 0.70)

## The Inference

### Without Similarity Edges (Step 1 only):
❌ Cannot infer new drug-disease relationships

### With Similarity Edges (Step 1 + 2):
✅ Can infer: **Metformin might treat Metabolic Syndrome**

**Reasoning:**
- Metformin treats Type 2 Diabetes (known)
- Aspirin treats Cardiovascular Disease (known)
- Metformin is similar to Aspirin (drug similarity)
- Type 2 Diabetes is similar to Metabolic Syndrome (disease similarity)
- Cardiovascular Disease is similar to Metabolic Syndrome (disease similarity)
- **Therefore**: Metformin might treat Metabolic Syndrome!

### Test the Inference Query

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

**Result:**
```
CandidateDrug: Metformin
TargetDisease: Metabolic Syndrome
DrugSimilarity: 0.65
DiseaseSim1: 0.75
DiseaseSim2: 0.7
```

## Query Examples

### Simple Queries
```cypher
-- Get all example nodes
MATCH (n {is_example: true})
RETURN n

-- Get only drugs
MATCH (n {is_example: true})
WHERE n.node_type = 'drug'
RETURN n.node_name, n.mechanism

-- Get similarity relationships
MATCH (a)-[r {is_example: true}]->(b)
WHERE r.relation IN ['drug_similarity', 'disease_similarity']
RETURN a.node_name, r.similarity_score, b.node_name

-- Delete all example data
MATCH (n {is_example: true})
DETACH DELETE n
```

For more examples, see:
- `example_simple_queries.md` - 20 simple query patterns
- `example_queries.py` - 10 analytical queries with Python

## Visualization Tips

1. **Apply the custom style** in Memgraph Lab:
   - Copy contents of `memgraph_style.cypherl`
   - Paste in Style Editor
   - Nodes will be colored by type, similarity edges will be purple

2. **Filter by relationship type:**
   ```cypher
   -- Show only treatment relationships
   MATCH (n {is_example: true})-[r {is_example: true}]->(m {is_example: true})
   WHERE r.relation = 'drug_treats_disease'
   RETURN n, r, m

   -- Show only similarities
   MATCH (n {is_example: true})-[r {is_example: true}]->(m {is_example: true})
   WHERE r.relation IN ['drug_similarity', 'disease_similarity']
   RETURN n, r, m
   ```

3. **Show paths from drugs to diseases:**
   ```cypher
   MATCH path = (drug {is_example: true})-[*1..3]->(disease {is_example: true})
   WHERE drug.node_type = 'drug' AND disease.node_type = 'disease'
   RETURN path
   LIMIT 10
   ```

## Real-World Application

This example demonstrates the core concept of drug repurposing through knowledge graphs. In a real application:

1. **Drug similarities** would be calculated from:
   - Chemical structure
   - Target proteins
   - Side effect profiles
   - Gene expression signatures

2. **Disease similarities** would be from:
   - Shared genetic markers
   - Common symptoms
   - Overlapping molecular pathways
   - Co-occurrence patterns

3. **Inference** would be weighted by:
   - Similarity scores
   - Path lengths and strengths
   - Clinical evidence
   - Machine learning models

## Cleanup

```cypher
-- Delete all example data
MATCH (n {is_example: true})
DETACH DELETE n
```

Or in Python:
```python
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=None)
session = driver.session()
session.run("MATCH (n {is_example: true}) DETACH DELETE n")
session.close()
driver.close()
```
