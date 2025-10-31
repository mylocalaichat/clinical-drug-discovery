# Complete Drug Discovery Workflow

## End-to-End Pipeline: From Graph to Predictions

This document explains the **complete workflow** from raw knowledge graph to ranked drug-disease predictions.

---

## Overview

```
Knowledge Graph → Node2Vec Embeddings → XGBoost Training → Predictions → Discovery
```

**Time to results:** ~15 minutes on example data

---

## Phase 1: Setup (One-Time)

### Step 1.1: Load Data into Memgraph

```bash
# Load example drug repurposing data
python load_example_progressive.py
```

**What this does:**
- Creates nodes: Drugs (2), Diseases (3), Proteins (3), Pathways (2)
- Creates edges: Drug-protein, protein-pathway, pathway-disease, etc.
- Adds similarity edges: Drug-drug, disease-disease

**Result:**
```
Graph loaded:
  - 10 nodes
  - 20+ relationships
  - Ready for embedding
```

### Step 1.2: Generate Embeddings

```bash
# Generate Node2Vec embeddings for all nodes
python generate_and_query_embeddings.py
```

**What this does:**
1. Loads graph from Memgraph into NetworkX
2. Runs Node2Vec random walks (10 walks × 30 steps per node)
3. Trains Skip-gram neural network (512 dimensions)
4. Stores embeddings back in Memgraph as `node.embedding` property

**Result:**
```
Embeddings generated:
  - 10 nodes embedded
  - Each has 64-dimensional vector (example uses 64, can be 512)
  - Stored in Memgraph: node.embedding = [0.81, 0.34, ..., 0.12]
```

**Time:** ~2 minutes

---

## Phase 2: Training (Every Time You Want New Predictions)

### Step 2.1: Train XGBoost Model

```bash
# Full pipeline with cross-validation
python train_xgboost_drug_discovery.py

# OR quick version for testing
python train_xgboost_simple.py
```

**What this does:**

#### Part A: Load Data
- Fetches embeddings from Memgraph
- Fetches known drug-disease relationships (ground truth)

#### Part B: Create Training Pairs
```python
# Positive samples (label = 1): Known treatments
(Metformin, Diabetes) → 1  # Known to treat

# Negative samples (label = 0): Random non-treatments
(Aspirin, Castleman) → 0   # Doesn't treat

# Unknown samples (label = 2): Synthetic unknowns
(RandomDrug, RandomDisease) → 2  # Uncertain
```

#### Part C: Feature Engineering
```python
# For each pair:
drug_embedding = [0.81, 0.34, ..., 0.12]  # 64 floats
disease_embedding = [0.78, 0.29, ..., 0.15]  # 64 floats

# Concatenate:
features = [0.81, 0.34, ..., 0.12, 0.78, 0.29, ..., 0.15]  # 128 floats
```

#### Part D: Train XGBoost
```python
# Train gradient-boosted trees
model = XGBClassifier(n_estimators=100, max_depth=6)
model.fit(X_train, y_train)

# Cross-validation ensures generalization
cv_scores = cross_val_score(model, X, y, cv=5)
```

#### Part E: Predict All Pairs
```python
# Generate ALL drug × disease combinations
all_pairs = drugs × diseases = 2 × 3 = 6 pairs (in example)

# For each unknown pair:
X = concatenate(drug_emb, disease_emb)
prob_treats = model.predict_proba(X)[1]  # Probability of treatment

# Sort by probability
ranked_predictions = sorted(all_pairs, key='prob_treats', reverse=True)
```

**Time:** ~5 minutes

**Output Files:**
- `drug_discovery_predictions.csv` (ranked predictions)
- Console output with top 20 candidates

---

## Phase 3: Analysis & Validation

### Step 3.1: Review Top Predictions

```bash
# View top 20
head -20 drug_discovery_predictions.csv

# Filter high confidence
awk -F, '$5 == "HIGH"' drug_discovery_predictions.csv
```

**Example output:**
```csv
rank,drug_name,disease_name,prob_treats,confidence
1,Rapamycin,Castleman Disease,0.9524,HIGH
2,Sirolimus,Sarcoidosis,0.9312,HIGH
3,Metformin,Metabolic Syndrome,0.8945,HIGH
```

### Step 3.2: Literature Validation

For each top candidate:

1. **PubMed Search:**
   ```
   "Rapamycin" AND "Castleman Disease"
   ```

2. **Clinical Trials:**
   - Search ClinicalTrials.gov
   - Check for ongoing or completed trials

3. **Mechanism Analysis:**
   - Does the drug target relevant pathways?
   - Are there shared molecular mechanisms?

### Step 3.3: Prioritization Matrix

| Drug | Disease | Prob | Literature | Mechanism | Safety | Priority |
|------|---------|------|-----------|-----------|--------|----------|
| Rapamycin | Castleman | 0.95 | Yes (5 papers) | mTOR pathway | Known | **HIGH** |
| Sirolimus | Sarcoidosis | 0.93 | No | Immune modulation | Known | MEDIUM |
| Metformin | MetabolicSyn | 0.89 | Yes (2 papers) | AMPK activation | Known | **HIGH** |

---

## Complete Code Flow

### Detailed Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ KNOWLEDGE GRAPH (Memgraph)                                 │
│                                                             │
│ Nodes: [Drug, Disease, Protein, Pathway]                   │
│ Edges: [targets, participates, dysregulated, treats]       │
└─────────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ RANDOM WALKS (Node2Vec)                                    │
│                                                             │
│ Walk 1: [Drug_A, Protein_X, Pathway_Y, Disease_B]         │
│ Walk 2: [Disease_B, Pathway_Y, Protein_Z, Drug_C]         │
│ ... 10 walks × 10 nodes = 100 walks                       │
└─────────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ TRAINING PAIRS (Skip-gram)                                 │
│                                                             │
│ (Drug_A, Protein_X), (Drug_A, Pathway_Y),                 │
│ (Pathway_Y, Disease_B), ...                                │
│ Millions of (center, context) pairs                        │
└─────────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ NEURAL NETWORK TRAINING                                    │
│                                                             │
│ Input: One-hot [100,000 dimensions]                        │
│ Hidden: Embedding layer [64 dimensions]  ← LEARNED!        │
│ Output: Softmax [100,000 dimensions]                       │
│ Loss: Cross-entropy                                        │
└─────────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ EMBEDDINGS (Stored in Memgraph)                           │
│                                                             │
│ Drug_A.embedding = [0.81, 0.34, ..., 0.12] (64 floats)    │
│ Disease_B.embedding = [0.78, 0.29, ..., 0.15] (64 floats) │
│ ... all nodes have embeddings                              │
└─────────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ TRAINING DATASET (Supervised Learning)                     │
│                                                             │
│ X = [drug_emb + disease_emb] → [128 features]             │
│ y = {0: doesn't treat, 1: treats, 2: unknown}             │
│                                                             │
│ Positive: (Drug_A, Disease_X) → 1                         │
│ Negative: (Drug_A, Disease_Y) → 0                         │
└─────────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ XGBOOST TRAINING                                           │
│                                                             │
│ Train gradient-boosted decision trees                      │
│ Learn: "Which embedding patterns → treatment?"             │
│ Trees: 100 trees, depth 6                                  │
│ Validation: 5-fold cross-validation                        │
└─────────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ PREDICTION (All Drug-Disease Pairs)                        │
│                                                             │
│ For each unknown pair:                                     │
│   X = [drug_emb + disease_emb]                            │
│   probs = model.predict_proba(X)                          │
│   prob_treats = probs[1]                                   │
└─────────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ RANKED RESULTS (CSV Output)                                │
│                                                             │
│ Rank 1: Rapamycin → Castleman (0.95)                      │
│ Rank 2: Sirolimus → Sarcoidosis (0.93)                    │
│ Rank 3: Metformin → Metabolic Syndrome (0.89)             │
│ ... all unknown pairs ranked                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Files Created

### Input Files
- `example_drug_repurposing_base.cypher` - Base graph structure
- `example_drug_repurposing_similarities.cypher` - Similarity edges

### Scripts
- `generate_and_query_embeddings.py` - Generate Node2Vec embeddings
- `train_xgboost_drug_discovery.py` - Full training pipeline
- `train_xgboost_simple.py` - Simplified version for testing

### Output Files
- `drug_discovery_predictions.csv` - Main results (ranked predictions)
- `predictions_simple.csv` - Results from simple version

### Documentation
- `XGBOOST_PIPELINE_GUIDE.md` - Detailed guide
- `COMPLETE_WORKFLOW.md` - This file
- `EMBEDDING_QUERY_GUIDE.md` - Embedding query reference

---

## Key Parameters to Tune

### Node2Vec Parameters

```python
# In generate_and_query_embeddings.py
node2vec = Node2Vec(
    dimensions=64,      # Embedding size (64, 128, 512)
    walk_length=30,     # Steps per walk (20-80)
    num_walks=10,       # Walks per node (10-100)
    window=10,          # Context window (5-10)
    p=1,                # Return parameter (0.5-2)
    q=1                 # In-out parameter (0.5-2)
)
```

**Effect:**
- Higher `dimensions` → More expressive but slower
- More `num_walks` → Better coverage but slower
- Larger `window` → Captures longer-range relationships

### XGBoost Parameters

```python
# In train_xgboost_drug_discovery.py
model = XGBClassifier(
    n_estimators=100,   # Number of trees (50-500)
    max_depth=6,        # Tree depth (3-10)
    learning_rate=0.1,  # Step size (0.01-0.3)
    subsample=0.8,      # Sample fraction (0.5-1.0)
)
```

**Effect:**
- More `n_estimators` → Better performance but slower
- Higher `max_depth` → More complex patterns but risk overfitting
- Lower `learning_rate` → More careful learning (need more trees)

### Sampling Ratios

```python
training_samples = create_training_data(
    negative_ratio=1.0,  # Negatives:Positives (0.5-2.0)
    unknown_ratio=0.5    # Unknowns:Positives (0.0-1.0)
)
```

**Effect:**
- Higher `negative_ratio` → Stricter model (fewer false positives)
- Higher `unknown_ratio` → More robust to uncertainty

---

## Performance Benchmarks

### Example Data (10 nodes)

| Phase | Time | Output |
|-------|------|--------|
| Load data | 10 sec | 10 nodes, 20 edges |
| Generate embeddings | 2 min | 10 × 64D vectors |
| Train XGBoost | 5 sec | Model trained |
| Predict all pairs | 1 sec | 6 predictions |
| **Total** | **~3 min** | Ranked results |

### Production Scale (100K nodes)

| Phase | Time | Output |
|-------|------|--------|
| Load data | 1 min | 100K nodes, 500K edges |
| Generate embeddings | 30 min | 100K × 512D vectors |
| Train XGBoost | 10 min | Model trained |
| Predict all pairs | 30 min | 1M+ predictions |
| **Total** | **~70 min** | Ranked results |

---

## Success Criteria

### Technical Metrics

| Metric | Target | Excellent |
|--------|--------|-----------|
| Cross-validation accuracy | > 0.75 | > 0.85 |
| ROC-AUC | > 0.80 | > 0.90 |
| Precision (class 1) | > 0.70 | > 0.85 |
| Recall (class 1) | > 0.70 | > 0.85 |

### Validation Metrics

| Criterion | Good | Excellent |
|-----------|------|-----------|
| Top 10 have literature support | ≥ 5 | ≥ 8 |
| Top 10 have mechanistic rationale | ≥ 7 | ≥ 9 |
| Novel discoveries (not in literature) | ≥ 2 | ≥ 4 |

---

## Troubleshooting

### Issue: Embeddings not found

**Error:** `KeyError: 'embedding'`

**Solution:**
```bash
# Re-run embedding generation
python generate_and_query_embeddings.py
```

### Issue: Low accuracy (< 0.7)

**Cause:** Poor embeddings or insufficient training data

**Solution:**
1. Increase Node2Vec parameters:
   ```python
   dimensions=128, num_walks=20
   ```
2. Add more training samples:
   ```python
   negative_ratio=2.0
   ```

### Issue: All predictions similar

**Cause:** Model not learning meaningful patterns

**Solution:**
1. Increase model complexity:
   ```python
   n_estimators=200, max_depth=8
   ```
2. Check embeddings are meaningful:
   ```python
   # Similar drugs should have similar embeddings
   cosine_similarity(drug1_emb, drug2_emb) > 0.7
   ```

---

## Summary

**Complete workflow in 3 commands:**

```bash
# 1. Generate embeddings (one-time)
python generate_and_query_embeddings.py

# 2. Train model and predict
python train_xgboost_drug_discovery.py

# 3. View results
head -20 drug_discovery_predictions.csv
```

**Result:** Ranked list of drug-disease predictions ready for validation!

**Next:** Investigate top candidates with literature search and experimental validation.
