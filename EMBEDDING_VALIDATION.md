# Embedding Validation Guide

This guide explains how to validate and compare GraphSAGE vs. HGT embeddings for off-label drug discovery.

## Overview

You have two embedding approaches:
1. **GraphSAGE** (baseline): Standard GNN with link prediction
2. **HGT** (optimized): Heterogeneous Graph Transformer with contrastive learning

The key question: **Do drugs/diseases with similar mechanisms cluster together in the embedding space?**

## Validation Scripts

### 1. Drug Mechanism Clustering (`validate_drug_clustering.py`)

**Purpose**: Check if drugs with the same mechanism of action have similar embeddings.

**Example drug classes tested:**
- Beta Blockers (Atenolol, Metoprolol, Propranolol)
- Statins (Atorvastatin, Simvastatin, Lovastatin)
- ACE Inhibitors (Lisinopril, Enalapril, Ramipril)
- SSRIs (Fluoxetine, Sertraline, Citalopram)
- NSAIDs (Ibuprofen, Naproxen, Diclofenac)

**How to run:**
```bash
uv run python validate_drug_clustering.py
```

**What to look for:**
- **Separation Score**: Positive value = good (within-class > between-class similarity)
- **t-SNE Plot**: Drugs from same class should cluster together
- **Example output**: Finding similar drugs to Atorvastatin should return other statins

**Output files:**
- `data/06_models/embeddings/validation/graphsage_drug_mechanism_clustering.html`
- `data/06_models/embeddings/validation/hgt_drug_mechanism_clustering.html`

---

### 2. Disease Mechanism Clustering (`validate_disease_clustering.py`)

**Purpose**: Check if diseases sharing genes/pathways have similar embeddings.

**This is the KEY validation for off-label drug discovery!**

**Method:**
1. Compute ground truth similarity using Jaccard index on shared genes/pathways
2. Compute embedding similarity using cosine similarity
3. Measure correlation between ground truth and embeddings

**How to run:**
```bash
uv run python validate_disease_clustering.py
```

**What to look for:**
- **Spearman Correlation**: Higher = better (0.3+ is good, 0.5+ is excellent)
- **Binned Analysis**: Embedding similarity should increase with mechanism similarity
- **Scatter Plot**: Should show positive correlation trend

**Output files:**
- `data/06_models/embeddings/validation/graphsage_disease_similarity_correlation.html`
- `data/06_models/embeddings/validation/hgt_disease_similarity_correlation.html`

**Interpretation:**
```
Jaccard > 0.2 + High Cosine Similarity = Good
→ Diseases share mechanisms AND have similar embeddings

Jaccard > 0.2 + Low Cosine Similarity = Bad
→ Diseases share mechanisms BUT embeddings don't capture it
```

---

## Expected Results

### GraphSAGE (Baseline)
- Treats all edge types the same
- Likely lower correlation between mechanism and embedding similarity
- May cluster drugs/diseases by degree (highly connected nodes)

### HGT with Contrastive Learning (Optimized)
- Explicitly optimizes for disease similarity via shared genes
- Should show higher correlation
- Better for off-label discovery

### Example Comparison

**GraphSAGE:**
```
Spearman correlation: 0.25
Within-class similarity: 0.45
Between-class similarity: 0.38
Separation score: 0.07
```

**HGT:**
```
Spearman correlation: 0.42  ← Better!
Within-class similarity: 0.58  ← Higher
Between-class similarity: 0.35  ← Lower
Separation score: 0.23  ← Much better!
```

---

## Use Cases

### Off-Label Drug Discovery Workflow

1. **Drug treats Disease A** via mechanism M
2. **Find Disease B** with similar embedding to Disease A
3. **Check if they share mechanism** (genes/pathways)
4. **Propose Drug for Disease B** (off-label use)

**Example:**
```python
# In the validation script, find diseases similar to a known indication
disease_A = "Type 2 Diabetes"
similar_diseases = find_similar_diseases(disease_A, embeddings_df, top_k=20)

# Check if they share mechanisms
for disease in similar_diseases:
    shared_genes = get_shared_genes(disease_A, disease, kg_df)
    if len(shared_genes) > threshold:
        print(f"{disease} shares {len(shared_genes)} genes with {disease_A}")
        print(f"  → Candidate for drug repurposing!")
```

---

## Visualizations

### 1. t-SNE Drug Clustering
**File**: `*_drug_mechanism_clustering.html`

**What you'll see:**
- Each point = a drug
- Colors = mechanism class
- Good embeddings: tight clusters by color

**Interactive features:**
- Hover over points to see drug names
- Zoom in/out
- Inspect cluster quality

### 2. Disease Similarity Correlation
**File**: `*_disease_similarity_correlation.html`

**What you'll see:**
- X-axis: Mechanism similarity (Jaccard)
- Y-axis: Embedding similarity (Cosine)
- Good embeddings: positive slope, tight trend

**Interpretation:**
- Points above trend line: embeddings over-estimate similarity
- Points below trend line: embeddings under-estimate similarity
- Outliers: interesting cases to investigate

---

## Metrics Explained

### Separation Score
```
Separation = Within-Class Similarity - Between-Class Similarity
```
- Positive = good (same class more similar than different classes)
- Higher = better clustering

### Spearman Correlation
```
Correlation(Mechanism Similarity, Embedding Similarity)
```
- Range: -1 to 1
- 0.3+ = good
- 0.5+ = excellent
- Measures if embeddings preserve mechanism relationships

### Cohesion Score (per class)
```
Cohesion = Within-Class Similarity - Between-Class Similarity
```
- Per drug/disease class
- Identifies which classes cluster well vs. poorly

---

## Next Steps

### If GraphSAGE is better:
- HGT may be overfitting to contrastive loss
- Try adjusting `contrastive_weight` parameter (currently 0.5)
- Try different `similarity_threshold` (currently 0.1)

### If HGT is better:
- Use HGT embeddings for downstream tasks!
- Retrain XGBoost models with HGT embeddings
- Compare off-label prediction performance

### If both are similar:
- Contrastive learning may not help for your data
- Consider other approaches (metapath2vec, knowledge graph embeddings)
- Check if enough diseases share pathways (Jaccard > 0.1)

---

## Troubleshooting

### "No drugs from mechanism classes found"
- Drug names may not match between embeddings and validation list
- Check drug names in embeddings CSV: `node_name` column
- Update `DRUG_CLASSES` dictionary with actual names

### "Too few diseases"
- Your embeddings may have filtered out diseases
- Check `include_node_types` in embedding generation
- Make sure 'disease' is included

### Correlation is very low (<0.1)
- Embeddings may not capture mechanisms
- Check if diseases actually share genes in knowledge graph
- Try more training epochs
- Adjust contrastive learning weight

---

## Questions to Answer

1. **Do drugs with same mechanism cluster together?**
   → Run `validate_drug_clustering.py`

2. **Do diseases sharing pathways have similar embeddings?**
   → Run `validate_disease_clustering.py`

3. **Which model is better for off-label discovery?**
   → Compare Spearman correlations

4. **Which drug/disease classes cluster well?**
   → Check per-class cohesion scores

5. **Can I find mechanistically similar diseases?**
   → Use HGT embeddings + cosine similarity
