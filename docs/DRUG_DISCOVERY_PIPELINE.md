# Drug-Disease Link Prediction Pipeline

## Overview

This implementation recreates a drug discovery pipeline using **Dagster** for orchestration and **MLflow** for experiment tracking. Every data transformation is a separate Dagster asset, making the pipeline highly inspectable and reproducible.

## How the Pipeline Works

The pipeline uses supervised machine learning for link prediction to find new drug-disease relationships. Unlike traditional graph queries, the approach:

1. **Generates ALL possible drug-disease combinations** (Cartesian product)
2. **Removes known training pairs** to find "missing edges"
3. **Trains ML models** on known positives/negatives + synthetic unknowns
4. **Scores all missing edges** and ranks by treatment probability

### Key Insight

> **Missing edges are found by set subtraction (all possible pairs - known pairs), not by graph queries.**

## Architecture

### 1. Embeddings (Node2Vec)

**What it does:** Converts graph structure into 512-dimensional vectors

```
Knowledge Graph
    ↓ (Random walks + Skip-gram)
512-dim embeddings for each node
```

**Key Parameters:**
- `embedding_dim`: 512 (chosen hyperparameter)
- `walk_length`: 30 steps per walk
- `num_walks`: 10 walks per node
- `window_size`: 10 (skip-gram context)

**Important:** Excludes INDICATION edges to prevent data leakage!

### 2. Training Data Generation

**Labels:**
- `0`: Does NOT treat (negative samples)
- `1`: Does treat (known positives)
- `2`: Unknown (synthetic negatives for uncertainty)

**Negative Sampling:**
- Generate 2 negatives per positive
- Half labeled 0, half labeled 2
- Random sampling from all possible pairs NOT in known set

### 3. Model Training

**Ensemble of Gradient Boosted Trees:**

1. **XGBoost**
   - Tree method: hist
   - Max depth: 6
   - Learning rate: 0.1
   - N estimators: 100

2. **LightGBM**
   - GBDT boosting
   - Max depth: 6
   - Learning rate: 0.1
   - N estimators: 100

3. **Random Forest**
   - N estimators: 100
   - Max depth: 10

**Feature Engineering:**
```
Drug embedding (512 dims) + Disease embedding (512 dims) = 1024 features
```

### 4. Prediction Pipeline

```
All drugs × All diseases (Cartesian product)
    ↓
Remove known training pairs
    ↓
Attach embeddings
    ↓
Score with ensemble models
    ↓
Rank by P(treat)
    ↓
Top candidates for drug repurposing
```

## Dagster Assets

### Data Loading Assets

1. **`link_prediction_drugs`** - Load all drugs from Neo4j
2. **`link_prediction_diseases`** - Load all diseases from Neo4j
3. **`link_prediction_known_pairs`** - Load known drug-disease treatments

### Embedding Asset

4. **`link_prediction_node2vec_embeddings`** - Train Node2Vec on knowledge graph
   - Excludes INDICATION edges
   - Generates 512-dim embeddings
   - Logs to MLflow

### Training Assets

5. **`link_prediction_training_data`** - Create training data with negative sampling
6. **`link_prediction_training_data_with_embeddings`** - Attach embeddings to training pairs
7. **`link_prediction_ensemble_models`** - Train XGBoost, LightGBM, RandomForest
   - Logs metrics to MLflow
   - Saves models to disk

### Prediction Assets

8. **`link_prediction_all_drug_disease_pairs`** - Generate Cartesian product
9. **`link_prediction_unknown_pairs`** - Remove known pairs to find missing edges
10. **`link_prediction_predictions`** - Score and rank all unknown pairs
    - Logs top predictions to MLflow
    - Saves to CSV for inspection

## Output Inspection

Every asset outputs data to disk for easy inspection:

```
data/
├── 06_models/link_prediction/
│   ├── drugs.csv                    # All drugs
│   ├── diseases.csv                 # All diseases
│   ├── known_pairs.csv              # Known treatments
│   ├── node2vec_embeddings.pkl      # Trained embeddings
│   ├── training_data.csv            # Training data with labels
│   ├── xgboost_model.pkl            # Trained XGBoost
│   ├── lightgbm_model.pkl           # Trained LightGBM
│   ├── random_forest_model.pkl      # Trained RF
│   └── all_pairs_sample.csv         # Sample of Cartesian product
└── 07_model_output/
    └── link_prediction_predictions.csv       # Final ranked predictions
```

## MLflow Tracking

All experiments are logged to MLflow:

```bash
# View MLflow UI
mlflow ui
```

**Logged Experiments:**
1. **node2vec_embeddings** - Graph stats, embedding params
2. **ensemble_training** - Model metrics, validation scores
3. **predictions** - Prediction statistics, top candidates

**Metrics Tracked:**
- Number of nodes/edges in graph
- Embedding dimensions
- Training/validation accuracy
- F1 scores
- Mean/max treat scores
- Top 100 mean score

## Usage

### 1. Install Dependencies

```bash
# Update dependencies
uv sync
```

### 2. Run the Pipeline

**Option A: Run all assets in sequence**

```bash
# Run complete link prediction pipeline
dagster dev
# Navigate to: http://localhost:3000
# Materialize: link_prediction_predictions (will run all upstream assets)
```

**Option B: Run specific assets**

```bash
# Just generate embeddings
dagster asset materialize -m dagster_definitions -a link_prediction_node2vec_embeddings

# Just train models
dagster asset materialize -m dagster_definitions -a link_prediction_ensemble_models

# Just make predictions
dagster asset materialize -m dagster_definitions -a link_prediction_predictions
```

### 3. View Results

**In Dagster UI:**
- View metadata for each asset
- See previews of DataFrames
- Check execution logs

**In MLflow UI:**
- Compare experiment runs
- View model metrics
- Download artifacts

**In Filesystem:**
- Inspect CSV files
- Load pickled models
- Analyze predictions

## Example Output

### Top Predictions

```
rank | drug_name      | disease_name        | treat_score | quantile_rank
-----|----------------|---------------------|-------------|---------------
1    | Rapamycin      | Castleman Disease   | 0.950       | 0.0000005
2    | Sirolimus      | Sarcoidosis         | 0.932       | 0.0000010
3    | Metformin      | Type 2 Diabetes     | 0.915       | 0.0000015
```

### Classification Report

```
              precision    recall  f1-score   support
           0       0.85      0.82      0.83      5000
           1       0.88      0.91      0.89      5000
           2       0.80      0.79      0.79      5000

    accuracy                           0.84     15000
   macro avg       0.84      0.84      0.84     15000
```

## Technical Details

### Node2Vec Algorithm

1. **Generate Random Walks**
   - For each node, generate 10 walks of length 30
   - Total: 100,000 nodes × 10 walks = 1M walks

2. **Create Skip-Gram Training Pairs**
   - Sliding window of size 10
   - Each center node paired with all context nodes
   - ~200M training pairs

3. **Train Neural Network**
   ```
   INPUT LAYER (100k nodes)
        ↓
   HIDDEN LAYER (512 dims)  ← This is the embedding!
        ↓
   OUTPUT LAYER (100k nodes)
   ```

4. **Extract Embeddings**
   - Weight matrix of hidden layer = embeddings
   - Each row = 512-dim vector for one node

### Negative Sampling Strategy

```python
# For each known positive (drug_A, disease_B):
# Generate 2 negatives:

1. Random disease: (drug_A, disease_random) with label=0
2. Random drug: (drug_random, disease_B) with label=2
```

### Ensemble Prediction

```python
# For each unknown pair:
probs_xgb = xgb_model.predict_proba(X)
probs_lgb = lgb_model.predict_proba(X)
probs_rf = rf_model.predict_proba(X)

# Average across models
final_probs = np.mean([probs_xgb, probs_lgb, probs_rf], axis=0)

# Extract treat score
treat_score = final_probs[:, 1]  # P(label=1)
```

## Key Features of This Implementation

| Aspect | Description |
|--------|-------------|
| Orchestration | Dagster assets |
| Graph backend | NetworkX + node2vec |
| Embedding | Python node2vec library |
| Experiment tracking | MLflow |
| Inspectability | Dagster metadata + MLflow |

## Performance Considerations

**Computational Complexity:**
- Node2Vec training: O(N × W × L) where N=nodes, W=walks/node, L=walk length
- Cartesian product: O(D × S) where D=drugs, S=diseases
- Prediction: O(P × F) where P=pairs, F=features

**Optimizations:**
- Batch processing for predictions (10k pairs/batch)
- Parallel workers for Node2Vec (4 workers)
- Ensemble averaging instead of stacking

**Approximate Runtime:**
- Embeddings: 10-30 minutes (depends on graph size)
- Training: 5-10 minutes
- Predictions: 15-30 minutes (depends on number of pairs)
- **Total: ~1 hour** for full pipeline

## References

1. Node2Vec: [Grover & Leskovec, 2016](https://arxiv.org/abs/1607.00653)
2. Skip-gram: [Mikolov et al., 2013](https://arxiv.org/abs/1301.3781)
3. Link prediction in biomedical knowledge graphs

## Troubleshooting

### Common Issues

1. **Out of Memory during Cartesian Product**
   - Reduce batch size in predictions
   - Process diseases in chunks

2. **Missing Embeddings**
   - Ensure all drugs/diseases in training data exist in graph
   - Check Node2Vec trained on correct graph

3. **Low Model Performance**
   - Increase n_negatives_per_positive
   - Tune hyperparameters (max_depth, learning_rate)
   - Add more features (graph centrality, degree, etc.)

## Future Enhancements

- [ ] Hyperparameter tuning with Optuna
- [ ] Multi-GPU training for XGBoost
- [ ] Add graph attention networks (GAT) embeddings
- [ ] Include additional features (drug structures, disease ontology)
- [ ] A/B testing different ensemble strategies
- [ ] Online learning for continuous updates
