# Embedding-Based Off-Label Drug Prediction Queries

## Quick Answer: Which Query Should I Use?

### ✅ **RECOMMENDED: `predict_offlabel_embedding_simple.cypher`**

**Best for most use cases.** This is a single, fast query that:
- Uses direct embedding similarity between drugs and diseases
- Calculates cosine similarity in pure Cypher
- Includes supporting evidence from graph structure
- Easy to modify (just change the disease name)
- Fast execution

**Use this if:** You want a straightforward, production-ready query

---

## All Available Queries

### 1. **predict_offlabel_embedding_simple.cypher** ⭐ RECOMMENDED
- **Type:** Single query, direct similarity
- **Speed:** Fast (< 1 second)
- **Complexity:** Simple
- **Strategies:** 1 main + 2 optional variations
- **Best for:** Production use, quick predictions

```cypher
// Just change the disease name:
MATCH (disease:Node {node_name: 'Metabolic Syndrome'})
```

**Strategies included:**
1. **Direct similarity** (default): Drug embedding ≈ Disease embedding
2. **Context-based** (optional): Drug embedding ≈ Average of disease's protein/pathway embeddings
3. **Analogy-based** (optional): Metformin:Diabetes :: ?:Metabolic Syndrome

---

### 2. **predict_offlabel_embedding_single.cypher**
- **Type:** Single query, comprehensive
- **Speed:** Medium (2-5 seconds)
- **Complexity:** Advanced
- **Strategies:** 4 combined strategies
- **Best for:** Comprehensive analysis, research

**Combines:**
1. Direct drug-disease embedding similarity
2. Context-based similarity (disease's molecular neighborhood)
3. Similar disease treatment (embedding-based disease clustering)
4. Protein target similarity (embedding-based protein matching)

**Returns:** Weighted combined score from all strategies

---

### 3. **Previous Non-Embedding Queries** (for comparison)
- `predict_offlabel_simple.cypher` - Graph pattern-based (no embeddings)
- `predict_offlabel_drugs.cypher` - Multi-strategy graph patterns
- `predict_offlabel_drugs_aggregated.cypher` - Aggregated graph patterns

---

## Prerequisites

**Before running any embedding-based query:**

```bash
# Generate and store embeddings in Memgraph
python3 generate_and_query_embeddings.py
```

This will:
1. Load your graph from Memgraph
2. Train node2vec embeddings (64 dimensions)
3. Store embeddings as `node.embedding` property

**Check if embeddings exist:**
```cypher
MATCH (n:Node)
WHERE EXISTS(n.embedding)
RETURN count(n) as nodes_with_embeddings, n.node_type
```

---

## Usage Examples

### Example 1: Basic Usage (Simple Query)

```cypher
// File: predict_offlabel_embedding_simple.cypher
// Just edit line 11:

MATCH (disease:Node {node_name: 'Metabolic Syndrome'})
WHERE disease.node_type = 'disease'
  AND EXISTS(disease.embedding)
// ... rest of query
```

**Change to your disease:**
```cypher
MATCH (disease:Node {node_name: 'Cardiovascular Disease'})
```

---

### Example 2: Comprehensive Analysis

```cypher
// File: predict_offlabel_embedding_single.cypher
// Edit line 22:

MATCH (target_disease:Node {node_name: 'Type 2 Diabetes'})
WHERE target_disease.node_type = 'disease'
  AND EXISTS(target_disease.embedding)
```

**Returns:**
- Overall confidence score (0-1)
- Individual strategy scores
- Evidence counts
- Sample protein targets and pathways
- Prediction rationale

---

## Comparison: Embedding vs Graph-Based

| Feature | Embedding Queries | Graph Pattern Queries |
|---------|------------------|---------------------|
| **Speed** | ⚡ Fast (< 1 sec) | 🐢 Slower (2-10 sec) |
| **Novelty** | 🎯 Finds novel connections | 📊 Limited to known patterns |
| **Explainability** | 🤔 Moderate | ✅ High (shows exact paths) |
| **Scalability** | 📈 Excellent (129K nodes) | 📉 Degrades with size |
| **Prerequisites** | Needs embeddings | Works immediately |
| **Global patterns** | ✅ Captures global structure | ❌ Local patterns only |

---

## Performance Tips

### For Large Graphs (> 100K nodes)

```cypher
// Add similarity threshold to filter early
WHERE embedding_similarity > 0.3  // Adjust threshold

// Limit intermediate results
LIMIT 20  // At the end
```

### Adjust Embedding Dimensions

In `generate_and_query_embeddings.py`:
```python
model = discovery.train_embeddings(G, dimensions=128)  # Increase for better accuracy
```

- **64 dims**: Fast, good for < 10K nodes
- **128 dims**: Balanced, good for 10K-100K nodes
- **256 dims**: Slow, best for > 100K nodes

---

## Understanding the Results

### Similarity Score Interpretation

| Score Range | Meaning | Action |
|-------------|---------|--------|
| 0.6 - 1.0 | Very High | Strong candidate, prioritize |
| 0.4 - 0.6 | High | Good candidate, investigate |
| 0.3 - 0.4 | Medium | Possible candidate, needs validation |
| < 0.3 | Low | Weak signal, likely noise |

### Confidence Levels

- **HIGH**: Multiple strong signals, direct evidence
- **MEDIUM**: Single strong signal or multiple moderate signals
- **LOW**: Weak signals, speculative

---

## Troubleshooting

### No results returned?

**Check 1:** Embeddings exist
```cypher
MATCH (n:Node) WHERE EXISTS(n.embedding) RETURN count(n);
```

**Check 2:** Lower the threshold
```cypher
WHERE embedding_similarity > 0.2  // Was 0.3
```

**Check 3:** Verify disease name
```cypher
MATCH (d:Node) WHERE d.node_type = 'disease' RETURN d.node_name;
```

### Slow query performance?

**Solution 1:** Add early filtering
```cypher
MATCH (drug:Node)
WHERE drug.node_type = 'drug'
  AND EXISTS(drug.embedding)
  AND drug.fda_approval_year < 2020  // Add filters
```

**Solution 2:** Use indexed properties
```cypher
CREATE INDEX ON :Node(node_type);
CREATE INDEX ON :Node(node_name);
```

---

## Next Steps

1. ✅ Run `python3 generate_and_query_embeddings.py` (one time)
2. ✅ Use `predict_offlabel_embedding_simple.cypher` for quick predictions
3. ✅ Validate top candidates with literature search
4. ✅ Use `predict_offlabel_embedding_single.cypher` for detailed analysis

---

## Summary

**For most users:** Start with `predict_offlabel_embedding_simple.cypher`

**Advantages:**
- Single query (easy to run)
- Fast performance
- Uses embeddings (captures global patterns)
- Returns ranked candidates with evidence
- Easy to customize

**You can always switch to the comprehensive query later for deeper analysis!**
