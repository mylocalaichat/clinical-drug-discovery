# XGBoost Drug Discovery - Dagster Integration

## Overview

The XGBoost drug discovery pipeline is now fully integrated into your Dagster workflow as a dedicated asset group called `xgboost_drug_discovery`.

## Asset Group Structure

All 9 assets are organized in the `xgboost_drug_discovery` group:

1. **xgboost_known_drug_disease_pairs** - Loads known treatment relationships from Memgraph
2. **xgboost_node_embeddings** - Loads all node embeddings from Memgraph
3. **xgboost_training_data** - Creates training dataset with positive/negative/unknown samples
4. **xgboost_feature_vectors** - Flattens embeddings into feature vectors (drug_emb + disease_emb)
5. **xgboost_trained_model** - Trains XGBoost classifier with 5-fold CV, saves model to disk
6. **xgboost_model_evaluation** - Evaluates model on test set, computes ROC-AUC and PR-AUC
7. **xgboost_all_drug_disease_pairs** - Generates all unknown drug-disease pairs to score
8. **xgboost_predictions** - Predicts treatment probabilities for all pairs
9. **xgboost_ranked_results** - Ranks by probability, adds confidence levels, saves to CSV

## Running the Pipeline

### Option 1: Using Dagster UI

```bash
# Start Dagster dev server
dagster dev

# Open http://localhost:3000
# Navigate to Asset Groups → xgboost_drug_discovery
# Click "Materialize all" to run the entire pipeline
```

**Note**: When you materialize any XGBoost asset, Dagster will automatically run the embeddings pipeline first (if not already materialized), ensuring the complete dependency chain executes in the correct order.

### Option 2: Using CLI

```bash
# Materialize all XGBoost assets
dagster asset materialize --select "xgboost_*"

# Or materialize just the final results (will trigger dependencies)
dagster asset materialize -m src.dagster_definitions xgboost_ranked_results
```

### Option 3: Using Python API

```python
from dagster import materialize
from src.dagster_definitions.assets.xgboost_drug_discovery import (
    xgboost_ranked_results
)

# Materialize the final asset (automatically runs all dependencies)
result = materialize([xgboost_ranked_results])
```

## Output Files

The pipeline generates several output files:

- **Model**: `data/06_models/xgboost/drug_disease_model.pkl`
- **Predictions**: `data/07_reporting/xgboost/xgboost_ranked_predictions.csv`

### CSV Columns

The final results CSV contains:

| Column | Description |
|--------|-------------|
| rank | Ranking by treatment probability (1 = best) |
| drug_name | Drug name |
| disease_name | Disease name |
| prob_treats | **KEY METRIC** - Probability drug treats disease |
| confidence | HIGH (>0.7), MEDIUM (0.4-0.7), LOW (<0.4) |
| prob_not_treat | Probability it doesn't treat |
| prob_unknown | Probability unknown/uncertain |
| drug_id | Drug node ID |
| disease_id | Disease node ID |

## Dependencies

The XGBoost pipeline depends on the embeddings pipeline:

**Dependency Chain:**
```
Data Loading (ALL must complete):
  - primekg_edges_loaded
  - drug_features_loaded
  - disease_features_loaded
    ↓
Embeddings Pipeline:
  random_graph_sample → knowledge_graph → node2vec_embeddings
    → flattened_embeddings
      ↓
XGBoost Pipeline:
  xgboost_node_embeddings → xgboost_training_data → ...
```

The pipeline requires:

1. **Embeddings pipeline completed** - `flattened_embeddings` asset must run first
2. **Memgraph running** with data loaded
3. **Known drug-disease relationships** in the graph
4. **OpenMP runtime** installed (`brew install libomp` on Mac)
5. **Python packages**: xgboost, scikit-learn, neo4j, pandas, numpy

## Prerequisites

Before running the pipeline:

```bash
# 1. Ensure Memgraph is running
# 2. Load data into Memgraph
python load_example_progressive.py

# 3. Generate embeddings (if not already done)
python generate_and_query_embeddings.py

# 4. Verify embeddings exist
# Run this Cypher query in Memgraph:
# MATCH (n) WHERE EXISTS(n.embedding) RETURN count(n);
```

## Configuration

You can customize the pipeline by modifying parameters in `src/dagster_definitions/assets/xgboost_drug_discovery.py`:

### Training Data Configuration

```python
# In xgboost_training_data asset:
negative_ratio = 1.0  # Change to 2.0 for more strict model
unknown_ratio = 0.5   # Change to 0.0 for binary classification only
```

### XGBoost Hyperparameters

```python
# In xgboost_trained_model asset:
model = XGBClassifier(
    n_estimators=100,      # Increase for better performance (200-500)
    max_depth=6,           # Increase for more complex patterns (8-10)
    learning_rate=0.1,     # Decrease for slower learning (0.01-0.05)
    subsample=0.8,         # Fraction of samples per tree
    colsample_bytree=0.8,  # Fraction of features per tree
)
```

## Monitoring

Each asset logs rich metadata that you can view in the Dagster UI:

- **Row counts** for dataframes
- **Distribution statistics** for predictions
- **Model metrics** (ROC-AUC, PR-AUC)
- **Cross-validation scores**
- **Top predictions** preview

## Troubleshooting

### Issue: Assets fail with "embedding not found"

**Solution**: Ensure embeddings are generated first:
```bash
python generate_and_query_embeddings.py
```

### Issue: XGBoost import error (libomp.dylib)

**Solution**: Install OpenMP:
```bash
brew install libomp
```

### Issue: Low prediction accuracy

**Solution**:
1. Increase Node2Vec embedding dimensions (edit `generate_and_query_embeddings.py`)
2. Increase XGBoost trees: `n_estimators=200`
3. Add more training data

## Integration Complete

The XGBoost pipeline is now fully integrated into your Dagster workflow. You can:

- View all assets in the Dagster UI
- Run the pipeline on-demand or on a schedule
- Monitor asset lineage and dependencies
- Track model performance over time
- Integrate with existing pipelines (e.g., run after embeddings are generated)

## Next Steps

1. **Test the pipeline**: Materialize `xgboost_ranked_results` in Dagster UI
2. **Review predictions**: Check `data/07_reporting/xgboost/xgboost_ranked_predictions.csv`
3. **Validate top candidates**: Use literature search and domain expert review
4. **Create a schedule** (optional): Add to `src/dagster_definitions/schedules.py` to run automatically
