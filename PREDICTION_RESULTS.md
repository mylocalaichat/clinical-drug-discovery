# Edge Prediction Results

## Summary

The edge prediction script successfully identified **4 new drug-disease candidates** using your existing graph:

### Top Predictions

#### 🥇 #1: Aspirin → Metabolic Syndrome
**Confidence: HIGH (3 methods agree)**
- **Disease Similarity**: Aspirin treats Cardiovascular Disease, which is similar to Metabolic Syndrome (0.70)
- **Combined Pattern**: Aspirin is similar to Metformin (0.65), which treats Type 2 Diabetes (similar to Metabolic Syndrome: 0.75)
- **Path Analysis**: 14 connecting paths with average length 3.5

**Real-world validation**: ✅ Aspirin is actually studied for metabolic syndrome! Low-dose aspirin has been investigated for improving insulin sensitivity and reducing cardiovascular risk in metabolic syndrome patients.

---

#### 🥈 #2: Metformin → Metabolic Syndrome
**Confidence: MEDIUM (2 methods agree)**
- **Disease Similarity**: Metformin treats Type 2 Diabetes, which is similar to Metabolic Syndrome (0.75)
- **Path Analysis**: 14 connecting paths with average length 3.5

**Real-world validation**: ✅ Metformin is widely used off-label for metabolic syndrome! It's a first-line treatment for prediabetes and metabolic syndrome.

---

#### 🥉 #3: Aspirin → Type 2 Diabetes
**Confidence: MEDIUM (2 methods agree)**
- **Drug Similarity**: Aspirin is similar to Metformin (0.65), which treats Type 2 Diabetes
- **Path Analysis**: 12 connecting paths with average length 3.58

**Real-world validation**: ⚠️ Aspirin doesn't directly treat diabetes, but it IS commonly prescribed to diabetic patients to reduce cardiovascular complications.

---

#### #4: Metformin → Cardiovascular Disease
**Confidence: LOW (1 method)**
- **Path Analysis**: 12 connecting paths

**Real-world validation**: ✅ Metformin has shown cardiovascular benefits in diabetic patients and is being studied for heart failure!

---

## How It Works

### Method 1: Pattern-Based Inference
Uses graph patterns to find relationships:

```
Drug A --treats--> Disease X
   |                   |
similar            similar
   |                   |
Drug B              Disease Y

INFERENCE: Drug B might treat Disease Y
```

### Method 2: Path-Based Scoring
Counts and scores all paths between drug-disease pairs:
- More paths = stronger connection
- Shorter paths = stronger connection

### Method 3: Shared Neighbor Analysis
Looks for drugs and diseases that share many intermediate nodes (proteins, pathways)

## Using Predictions

### Step 1: Review Predictions in Memgraph

Visualize the predicted edge:
```cypher
// Show Aspirin and Metabolic Syndrome connection
MATCH path = (aspirin {node_name: 'Aspirin'})-[*1..3]-(ms {node_name: 'Metabolic Syndrome'})
WHERE aspirin.is_example = true
AND ms.is_example = true
RETURN path
LIMIT 10
```

### Step 2: Validate in Literature

For each prediction:
1. Search PubMed for: `"[Drug]" AND "[Disease]"`
2. Check clinical trial databases
3. Review systematic reviews

### Step 3: Score and Prioritize

Combine multiple factors:
```python
final_score = (
    0.4 * pattern_score +      # Graph evidence
    0.3 * path_score +          # Connectivity
    0.2 * literature_score +    # Published evidence
    0.1 * safety_score          # Known safety profile
)
```

## Extending to Full Dataset

### For Your Full Graph (129,375 nodes)

1. **Add similarity edges**:
   ```python
   # Calculate drug similarities
   - Chemical structure (ECFP fingerprints)
   - Target protein overlap
   - Side effect similarity

   # Calculate disease similarities
   - Gene expression profiles
   - Shared genes/proteins
   - Symptom overlap
   ```

2. **Run predictions**:
   ```bash
   python3 predict_edges.py
   ```

3. **Filter by confidence**:
   - Require 2+ methods agree
   - Minimum similarity threshold (0.6+)
   - Maximum path length (≤4)

4. **Validate top 100**:
   - Literature review
   - Clinical trial search
   - Expert consultation

## Advanced: Add Your Own Features

Extend the prediction script to include:

### Clinical Features
```python
# In predict_edges.py, add:
- Disease prevalence
- Drug safety profile
- Patient demographics
- Cost-effectiveness
```

### Biological Features
```python
# Add to feature extraction:
- Gene expression correlation
- Protein-protein interaction strength
- Pathway enrichment scores
- Tissue specificity overlap
```

### Machine Learning
```python
# Train a classifier
from sklearn.ensemble import RandomForestClassifier

features = [
    'drug_similarity',
    'disease_similarity',
    'shared_proteins',
    'shared_pathways',
    'shortest_path_length',
    'num_connecting_paths'
]

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict_proba(X_test)
```

## Next Steps

1. ✅ **You've done**: Loaded example graph with similarities
2. ✅ **You've done**: Run basic edge prediction
3. 🔄 **Next**: Apply to full graph (129K nodes)
4. 🔄 **Next**: Calculate similarities for all drugs/diseases
5. 🔄 **Next**: Train ML model with features
6. 🔄 **Next**: Validate top predictions

## Files Reference

- `predict_edges.py` - Main prediction script (run this)
- `edge_prediction_guide.md` - Detailed methods documentation
- `example_queries.py` - Query examples
- `load_example_progressive.py` - Load example data

## Quick Commands

```bash
# Load example data
python3 load_example_progressive.py

# Run predictions
python3 predict_edges.py

# Query predictions in Memgraph
# (paste in Memgraph Lab)
MATCH (drug {is_example: true}), (disease {is_example: true})
WHERE drug.node_type = 'drug'
AND disease.node_type = 'disease'
AND NOT EXISTS {
    (drug)-[:RELATES {relation: 'drug_treats_disease'}]->(disease)
}
RETURN drug.node_name, disease.node_name
```
