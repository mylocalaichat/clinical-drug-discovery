# XGBoost Drug Discovery Pipeline - Complete Guide

## Overview

This pipeline uses **Node2Vec embeddings + XGBoost** to predict which drugs can treat which diseases.

```
Node2Vec → Embeddings → Feature Engineering → XGBoost → Predictions → Ranked Results
```

---

## Quick Start

### 1. Prerequisites

```bash
# Ensure Memgraph is running
# Ensure embeddings are generated

python generate_and_query_embeddings.py  # Generate embeddings first
```

### 2. Run the Pipeline

```bash
python train_xgboost_drug_discovery.py
```

### 3. View Results

```bash
# Output file: drug_discovery_predictions.csv
head -20 drug_discovery_predictions.csv
```

---

## Pipeline Steps Explained

### Step 1: Load Embeddings from Memgraph

**What happens:**
- Queries Memgraph for all nodes with `embedding` property
- Separates drugs and diseases
- Each node has a 64-dimensional embedding vector

**Output:**
```python
embeddings = {
    'DRUG:Rapamycin': {
        'name': 'Rapamycin',
        'type': 'drug',
        'embedding': [0.81, 0.34, -0.23, ..., 0.12]  # 64 floats
    },
    'MONDO:0005301': {
        'name': 'Castleman Disease',
        'type': 'disease',
        'embedding': [0.78, 0.29, -0.19, ..., 0.15]  # 64 floats
    }
}
```

---

### Step 2: Load Known Drug-Disease Relationships

**What happens:**
- Queries Memgraph for known treatments (`drug_treats_disease` relationships)
- These are the **ground truth** labels

**Output:**
```python
known_pairs = [
    {'drug_id': 'DRUG:Metformin', 'disease_id': 'MONDO:0005148', 'label': 1},  # Treats
    {'drug_id': 'DRUG:Aspirin', 'disease_id': 'MONDO:0005252', 'label': 1},   # Treats
    # ... more known treatments
]
```

---

### Step 3: Create Training Dataset

**What happens:**
- **Positive samples (label=1):** Known treatments
- **Negative samples (label=0):** Random drug-disease pairs that DON'T have a treatment relationship
- **Unknown samples (label=2):** Synthetic samples for robustness

**Why negatives?** XGBoost needs examples of "doesn't treat" to learn the pattern!

**Output:**
```python
training_samples = [
    # Positive (treats)
    {'drug': 'Metformin', 'disease': 'Diabetes', 'label': 1},
    {'drug': 'Aspirin', 'disease': 'Heart Disease', 'label': 1},

    # Negative (doesn't treat)
    {'drug': 'Aspirin', 'disease': 'Castleman', 'label': 0},
    {'drug': 'Metformin', 'disease': 'Sarcoidosis', 'label': 0},

    # Unknown (synthetic)
    {'drug': 'RandomDrug', 'disease': 'RandomDisease', 'label': 2},
]
```

---

### Step 4: Flatten Embeddings into Feature Vectors

**What happens:**
- Concatenate drug embedding + disease embedding
- Creates 128-dimensional feature vector (64 + 64)

**Example:**
```python
# Input:
drug_emb = [0.81, 0.34, -0.23, ..., 0.12]      # 64 dims
disease_emb = [0.78, 0.29, -0.19, ..., 0.15]   # 64 dims

# Output:
feature_vector = [
    0.81, 0.34, -0.23, ..., 0.12,  # Drug features (0-63)
    0.78, 0.29, -0.19, ..., 0.15   # Disease features (64-127)
]  # 128 total features
```

**Feature Matrix:**
```
+-------+-------+-----+--------+--------+--------+-----+--------+-------+
|feat_0 |feat_1 | ... |feat_63 |feat_64 |feat_65 | ... |feat_127| label |
+-------+-------+-----+--------+--------+--------+-----+--------+-------+
| 0.81  | 0.34  | ... | 0.12   | 0.78   | 0.29   | ... | 0.15   |   1   | ← Rapamycin + Castleman (treats)
| -0.23 | 0.45  | ... | 0.67   | 0.78   | 0.29   | ... | 0.15   |   0   | ← Aspirin + Castleman (doesn't treat)
+-------+-------+-----+--------+--------+--------+-----+--------+-------+
```

---

### Step 5: Train/Test Split

**What happens:**
- Split data 80/20 (training/testing)
- Stratified to maintain class balance

**Output:**
```
Training set: 80% of data
  - Class 0 (doesn't treat): 40%
  - Class 1 (treats): 40%
  - Class 2 (unknown): 20%

Test set: 20% of data (held out for evaluation)
```

---

### Step 6: Train XGBoost with Cross-Validation

**What happens:**
- 5-fold cross-validation to ensure model generalizes
- Train gradient-boosted trees
- Each tree learns patterns in the 128-dimensional feature space

**Model learns:**
```python
# Example decision tree pattern:
IF feat_0 > 0.7 AND feat_64 > 0.7:  # Both high on dimension 0
    IF feat_47 > 0.6 AND feat_111 > 0.5:  # Both high on dimension 47
        PREDICT: TREATS (class 1) with probability 0.9
    ELSE:
        PREDICT: DOESN'T TREAT (class 0)
```

**Output:**
```
Cross-validation accuracy: 0.87 (+/- 0.03)
Model trained on full training set
```

---

### Step 7: Evaluate on Test Set

**What happens:**
- Test model on held-out data (never seen during training)
- Calculate metrics: accuracy, precision, recall, ROC-AUC

**Output:**
```
Classification Report:
              precision    recall  f1-score   support

  Doesn't Treat    0.85      0.88      0.86       100
         TREATS    0.89      0.91      0.90       100
        Unknown    0.78      0.72      0.75        50

ROC-AUC (Class 1 'TREATS'): 0.92
```

---

### Step 8: Score ALL Drug-Disease Pairs

**What happens:**
- Generate ALL possible drug × disease combinations
- Remove known treatments (already validated)
- Predict treatment probability for each pair

**Example:**
```python
# Generate pairs:
all_pairs = [
    (Rapamycin, Castleman),     # Unknown pair
    (Rapamycin, Sarcoidosis),   # Unknown pair
    (Aspirin, Castleman),       # Unknown pair
    # ... thousands more ...
]

# For each pair:
drug_emb = get_embedding(drug)
disease_emb = get_embedding(disease)
X = concatenate(drug_emb, disease_emb)  # 128 features
probs = model.predict_proba(X)  # [P(not_treat), P(treat), P(unknown)]
```

**Output:**
```
Scored 1,950 unknown drug-disease pairs
Each pair has:
  - prob_not_treat (P(class 0))
  - prob_treats (P(class 1))  ← KEY METRIC
  - prob_unknown (P(class 2))
```

---

### Step 9: Rank and Output Results

**What happens:**
- Sort all predictions by `prob_treats` (descending)
- Assign confidence levels
- Export to CSV

**Output CSV:**
```csv
rank,drug_name,disease_name,prob_treats,confidence,prob_not_treat,prob_unknown
1,Rapamycin,Castleman Disease,0.9524,HIGH,0.0321,0.0155
2,Sirolimus,Sarcoidosis,0.9312,HIGH,0.0456,0.0232
3,Metformin,Metabolic Syndrome,0.8945,HIGH,0.0789,0.0266
4,Tacrolimus,Castleman Disease,0.8723,HIGH,0.0912,0.0365
...
```

**Top 20 Candidates:**
```
TOP 20 DRUG REPURPOSING CANDIDATES
================================================================================
rank  drug_name    disease_name           prob_treats  confidence
1     Rapamycin    Castleman Disease      0.9524       HIGH
2     Sirolimus    Sarcoidosis            0.9312       HIGH
3     Metformin    Metabolic Syndrome     0.8945       HIGH
...
```

---

## Understanding the Results

### Probability Interpretation

| prob_treats | Confidence | Interpretation |
|-------------|-----------|----------------|
| 0.90 - 1.00 | HIGH      | Very strong candidate - prioritize for investigation |
| 0.70 - 0.89 | HIGH      | Strong candidate - good evidence |
| 0.40 - 0.69 | MEDIUM    | Moderate candidate - requires validation |
| 0.00 - 0.39 | LOW       | Weak candidate - likely false positive |

### What Drives High Scores?

**High probability (e.g., 0.95) means:**
1. Drug and disease embeddings are **similar** (close in 128D space)
2. This pattern **matches known treatments** XGBoost learned
3. Multiple dimensions align (not just one)

**Example: Rapamycin + Castleman (0.95)**
```python
# Why high score?
rapamycin_emb[0] = 0.81  → castleman_emb[0] = 0.78  ✓ Similar
rapamycin_emb[47] = 0.72 → castleman_emb[47] = 0.68 ✓ Similar
rapamycin_emb[123] = 0.65 → castleman_emb[123] = 0.61 ✓ Similar

# XGBoost learned: "When drug and disease are similar on dimensions 0, 47, 123,
#                   they usually have a treatment relationship"
```

---

## Configuration Options

### Negative Sampling Ratio

```python
training_samples = self.create_training_data(
    drugs, diseases, known_pairs, embeddings,
    negative_ratio=1.0,  # 1.0 = equal negatives to positives
    unknown_ratio=0.5    # 0.5 = half as many unknowns
)
```

**Adjust these to:**
- `negative_ratio=2.0` → More negatives (stricter model)
- `unknown_ratio=0.0` → No unknown samples (binary classification)

### XGBoost Hyperparameters

```python
model = XGBClassifier(
    n_estimators=100,      # Number of trees (increase for better performance)
    max_depth=6,           # Tree depth (increase for more complex patterns)
    learning_rate=0.1,     # Step size (decrease for slower, more careful learning)
    subsample=0.8,         # Fraction of samples per tree (for regularization)
    colsample_bytree=0.8,  # Fraction of features per tree
)
```

**For better performance:**
- Increase `n_estimators` to 200-500
- Tune `max_depth` (deeper = more complex, but risk overfitting)
- Lower `learning_rate` to 0.01-0.05 (more trees needed)

---

## Validation Workflow

### 1. Check Top Predictions

```python
import pandas as pd

results = pd.read_csv('drug_discovery_predictions.csv')

# Top 10 candidates
print(results.head(10))

# High confidence only
high_conf = results[results['confidence'] == 'HIGH']
print(f"High confidence predictions: {len(high_conf)}")
```

### 2. Literature Search

For each top candidate:
1. Search PubMed: `"drug_name" AND "disease_name"`
2. Check clinical trials: ClinicalTrials.gov
3. Review mechanism: Does it make biological sense?

### 3. Validate with Domain Experts

Share top 20 with:
- Clinicians (Is this plausible?)
- Pharmacologists (Any contraindications?)
- Researchers (Worth investigating?)

---

## Common Issues & Solutions

### Issue 1: Low Accuracy

**Symptoms:** Cross-validation accuracy < 0.7

**Solutions:**
1. Check embeddings: Are they meaningful?
   ```python
   # Test: Similar drugs should have similar embeddings
   cosine_similarity(drug1_emb, drug2_emb)
   ```
2. Increase training data: More negative samples
3. Tune hyperparameters: More trees, deeper trees

### Issue 2: All Predictions Are Similar

**Symptoms:** All `prob_treats` between 0.4-0.6

**Solutions:**
1. Increase model complexity: `max_depth=8`, `n_estimators=200`
2. Better negative sampling: Ensure true negatives are included
3. Feature engineering: Add more features beyond embeddings

### Issue 3: Known Treatments Score Low

**Symptoms:** Known drugs score poorly on known diseases

**Solutions:**
1. **Critical:** Check data leakage - are known pairs in test set?
2. Embeddings may not capture treatment relationship well
3. Re-train Node2Vec with different parameters

---

## Advanced: Feature Importance

```python
import matplotlib.pyplot as plt

# Get feature importance
importance = model.feature_importances_

# Plot top 20 features
plt.figure(figsize=(10, 6))
plt.barh(range(20), sorted(importance, reverse=True)[:20])
plt.xlabel('Feature Importance')
plt.ylabel('Feature Index')
plt.title('Top 20 Most Important Features')
plt.savefig('feature_importance.png')
```

**Interpretation:**
- High importance = This embedding dimension is critical for predictions
- Can help identify which graph patterns matter most

---

## Output Files

### 1. drug_discovery_predictions.csv

**Columns:**
- `rank`: Ranking by treatment probability (1 = best)
- `drug_name`: Drug name
- `disease_name`: Disease name
- `prob_treats`: **KEY METRIC** - Probability drug treats disease
- `confidence`: HIGH/MEDIUM/LOW
- `prob_not_treat`: Probability it doesn't treat
- `prob_unknown`: Probability unknown/uncertain
- `drug_id`: Drug node ID
- `disease_id`: Disease node ID

**Use this for:**
- Identifying top candidates
- Filtering by confidence level
- Further analysis in Excel/Python

---

## Next Steps After Results

### 1. Immediate Actions

```bash
# View top 20
head -20 drug_discovery_predictions.csv

# Filter high confidence
awk -F, '$5 == "HIGH"' drug_discovery_predictions.csv > high_confidence.csv
```

### 2. Investigation Priority

**Tier 1 (Immediate):**
- prob_treats > 0.85
- Known safety profile
- Existing preclinical data

**Tier 2 (Short-term):**
- prob_treats 0.70-0.85
- Mechanistic rationale
- Similar drug precedent

**Tier 3 (Long-term):**
- prob_treats 0.50-0.70
- Novel mechanism
- Exploratory research

### 3. Experimental Validation

**In vitro:**
1. Cell line assays
2. Dose-response curves
3. Mechanism studies

**In vivo:**
1. Animal models
2. Pharmacokinetics
3. Efficacy studies

**Clinical:**
1. Phase I safety
2. Phase II efficacy
3. Phase III confirmation

---

## Summary

The pipeline takes:
- **Input:** Node2Vec embeddings (512D per node)
- **Process:** Flatten to 128D features → Train XGBoost → Predict all pairs
- **Output:** Ranked list of drug-disease treatment predictions

**Key insight:** Embeddings compress graph structure into vectors. XGBoost learns which embedding patterns indicate treatment relationships. Apply to all pairs → discover new treatments!

**Success metric:** High-confidence predictions validated by literature or experiments.
