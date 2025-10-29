# MLflow Tracking in Clinical Drug Discovery Pipeline

## Overview

Every major stage of the pipeline logs to MLflow for full observability. All runs are tracked in the `clinical-drug-discovery` experiment.

## Tracked Pipeline Stages

### 1. Data Loading (`data_loading` group)

**Asset:** `disease_features_loaded`
**Run Name:** `data_loading`

**Parameters Logged:**
- `database` - Neo4j database name
- `data_source` - "PrimeKG"

**Metrics Logged:**
- `nodes_loaded` - Total nodes in Neo4j
- `disease_features_processed` - Number of disease features loaded

**Artifacts:** None (data stored in Neo4j)

---

### 2. Clinical Extraction (`clinical_extraction` group)

**Asset:** `clinical_drug_disease_pairs`
**Run Name:** `clinical_extraction`

**Parameters Logged:**
- `ner_model` - "en_ner_bc5cdr_md"
- `min_frequency` - Minimum co-occurrence frequency
- `max_note_length` - Maximum clinical note length
- `num_input_notes` - Number of input clinical notes

**Metrics Logged:**
- `num_extracted_pairs` - Total drug-disease pairs extracted
- `num_unique_drugs` - Unique drugs found
- `num_unique_diseases` - Unique diseases found
- `extraction_*` - Various extraction statistics

**Artifacts:**
- `data/03_primary/clinical_drug_disease_pairs.csv` - Extracted pairs

---

### 3. Graph Enrichment (`graph_enrichment` group)

**Asset:** `clinical_enrichment_stats`
**Run Name:** `graph_enrichment`

**Parameters Logged:**
- `min_score` - Minimum clinical evidence score (0.1)
- `num_input_pairs` - Number of input clinical pairs

**Metrics Logged:**
- `clinical_relationships_added` - CLINICAL_EVIDENCE relationships added to graph
- `relationships_with_positive_score` - Positive clinical associations
- `relationships_with_negative_score` - Negative clinical associations

**Artifacts:** None (relationships stored in Neo4j)

---

### 4. Drug Discovery (`drug_discovery` group)

**Asset:** `drug_discovery_results`
**Run Name:** `discovery_{disease_name}` (one run per disease)

**Parameters Logged:**
- `disease_id` - Target disease ID
- `disease_name` - Target disease name

**Metrics Logged:**
- `clinical_relationships_added` - From enrichment
- `total_clinical_evidence` - From validation
- `base_candidates_found` - Candidates from topology-only query
- `enhanced_candidates_found` - Candidates from enhanced query
- `candidates_with_clinical_evidence` - Candidates with clinical support
- `max_score_improvement` - Best score improvement
- `avg_score_improvement` - Average score improvement

**Artifacts:**
- `base_results.csv` - Topology-only results
- `enhanced_results.csv` - Enhanced results with clinical evidence
- `comparison.csv` - Side-by-side comparison

---

## Viewing MLflow Results

### Start MLflow Server

```bash
make mlflow
```

Navigate to: **http://localhost:5000**

### Browse Experiments

1. Click on experiment: **clinical-drug-discovery**
2. View all runs sorted by time
3. Filter by run name:
   - `data_loading`
   - `clinical_extraction`
   - `graph_enrichment`
   - `discovery_*` (per disease)

### Compare Runs

1. Select multiple runs (checkbox)
2. Click "Compare" button
3. View metrics side-by-side
4. Download comparison CSV

### Download Artifacts

1. Click on a run
2. Scroll to "Artifacts" section
3. Click file name to download

---

## Example MLflow Workflow

### Scenario: Compare Clinical Extraction Across Runs

**Terminal 1:**
```bash
make mlflow
```

**Terminal 2:**
```bash
make dagster
# Materialize: clinical_drug_disease_pairs
```

**In MLflow UI (http://localhost:5000):**

1. **View Latest Extraction Run:**
   - Experiment: `clinical-drug-discovery`
   - Filter runs by: `clinical_extraction`
   - Click latest run

2. **Check Extraction Quality:**
   ```
   Parameters:
     - ner_model: en_ner_bc5cdr_md
     - num_input_notes: 5000

   Metrics:
     - num_extracted_pairs: 342
     - num_unique_drugs: 87
     - num_unique_diseases: 156
   ```

3. **Download Extracted Pairs:**
   - Artifacts → `clinical_drug_disease_pairs.csv`
   - Click to download

4. **Compare with Previous Run:**
   - Select 2 runs
   - Click "Compare"
   - View metric differences

---

## Programmatic Access

### Query MLflow via Python

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Get experiment
experiment = mlflow.get_experiment_by_name("clinical-drug-discovery")

# Get all runs
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.mlflow.runName = 'clinical_extraction'",
    order_by=["start_time DESC"],
)

# Access metrics
for idx, run in runs.iterrows():
    print(f"Run ID: {run['run_id']}")
    print(f"Pairs extracted: {run['metrics.num_extracted_pairs']}")
    print(f"Unique drugs: {run['metrics.num_unique_drugs']}")
```

### Download Artifacts Programmatically

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get latest run
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1,
)

run_id = runs.iloc[0]['run_id']

# Download artifact
local_path = client.download_artifacts(
    run_id=run_id,
    path="clinical_drug_disease_pairs.csv",
    dst_path="./downloads"
)

print(f"Downloaded to: {local_path}")
```

---

## Metrics Reference

### Data Loading Metrics
| Metric | Description | Type |
|--------|-------------|------|
| nodes_loaded | Total nodes in graph | int |
| disease_features_processed | Disease features added | int |

### Clinical Extraction Metrics
| Metric | Description | Type |
|--------|-------------|------|
| num_extracted_pairs | Total pairs extracted | int |
| num_unique_drugs | Unique drugs found | int |
| num_unique_diseases | Unique diseases found | int |

### Graph Enrichment Metrics
| Metric | Description | Type |
|--------|-------------|------|
| clinical_relationships_added | CLINICAL_EVIDENCE edges added | int |
| relationships_with_positive_score | Positive associations | int |
| relationships_with_negative_score | Negative associations | int |

### Drug Discovery Metrics
| Metric | Description | Type |
|--------|-------------|------|
| base_candidates_found | Topology-only candidates | int |
| enhanced_candidates_found | Enhanced candidates | int |
| candidates_with_clinical_evidence | With clinical support | int |
| max_score_improvement | Best improvement | float |
| avg_score_improvement | Average improvement | float |

---

## Troubleshooting

### "No experiment found"
- Ensure MLflow server is running: `make mlflow`
- Check experiment exists: Should auto-create on first run

### "Artifact not found"
- Some assets don't produce artifacts (e.g., graph enrichment)
- Check asset documentation for artifact list

### "Cannot connect to tracking server"
- Default URL: http://localhost:5000
- Check server is running
- Update tracking URI if using different port

---

## Best Practices

### 1. Run MLflow Before Pipeline
Always start MLflow server before running Dagster:
```bash
# Terminal 1
make mlflow

# Terminal 2
make dagster
```

### 2. Tag Important Runs
Add custom tags in code:
```python
mlflow.set_tag("experiment_type", "production")
mlflow.set_tag("data_version", "v2.0")
```

### 3. Compare Across Parameters
Use MLflow UI to compare runs with different:
- `min_frequency` values
- `min_score` thresholds
- NER models

### 4. Archive Important Runs
In MLflow UI:
1. Select run
2. Click "..." menu
3. "Archive" → prevents deletion

---

## Next Steps

### Enable More Tracking

Want to track intermediate steps? Add MLflow to any asset:

```python
@asset(group_name="my_group", compute_kind="processing")
def my_asset(context: AssetExecutionContext, input_data: pd.DataFrame) -> pd.DataFrame:
    import mlflow

    mlflow.set_experiment("clinical-drug-discovery")

    with mlflow.start_run(run_name="my_processing_step"):
        # Log parameters
        mlflow.log_params({
            "param1": "value1",
        })

        # Do processing
        result = process(input_data)

        # Log metrics
        mlflow.log_metrics({
            "rows_processed": len(result),
        })

        # Save and log artifact
        result.to_csv("output.csv", index=False)
        mlflow.log_artifact("output.csv")

    return result
```

### Custom Metrics

Track custom metrics:
```python
mlflow.log_metric("accuracy", 0.95)
mlflow.log_metric("f1_score", 0.92)
mlflow.log_metric("runtime_seconds", 45.2)
```

### Hyperparameter Tracking

For model training:
```python
params = {
    "learning_rate": 0.01,
    "max_depth": 6,
    "n_estimators": 100,
}
mlflow.log_params(params)
```
