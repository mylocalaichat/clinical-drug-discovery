# Quick Start Guide

## Prerequisites

- Python 3.10+
- Neo4j database running
- `.env` file configured with Neo4j credentials

## Installation

```bash
make install
```

## Available Commands

Run `make help` to see all available commands:

```bash
make help
```

Output:
```
Usage: make [target]

Available targets:
  clean           Clean up generated files
  dagster         Start Dagster web server
  help            Show this help message
  install         Install dependencies
  mlflow          Start MLflow server on port 5000
  run             Alias for dagster command
  test            Run basic validation tests
```

## Running the Pipeline

### Option 1: View and Run Assets in Dagster UI

1. **Start Dagster**:
   ```bash
   make dagster
   ```
   Navigate to: http://localhost:3000

2. **Materialize Assets**:
   - Click on "Assets" in the left sidebar
   - Select an asset (e.g., `drug_discovery_results`)
   - Click "Materialize selected"
   - Watch the execution in real-time

### Option 2: View MLflow Experiments

1. **Start MLflow** (in a separate terminal):
   ```bash
   make mlflow
   ```
   Navigate to: http://localhost:5000

2. **View Experiments**:
   - All pipeline runs are logged to MLflow
   - Browse experiments: "clinical-drug-discovery"
   - View metrics, parameters, and artifacts
   - Compare runs side-by-side

## Pipeline Overview

The pipeline consists of 4 main stages:

### 1. Data Loading (`data_loading` group)
- `download_data` - Download PrimeKG dataset
- `neo4j_database_ready` - Setup Neo4j database
- `primekg_nodes_loaded` - Load nodes to Neo4j
- `primekg_edges_loaded` - Load edges to Neo4j
- `drug_features_loaded` - Load drug features
- `disease_features_loaded` - Load disease features

### 2. Clinical Extraction (`clinical_extraction` group)
- `mtsamples_raw` - Load clinical notes
- `clinical_drug_disease_pairs` - Extract drug-disease relationships from text
- `clinical_extraction_stats` - Statistics on extraction

### 3. Graph Enrichment (`graph_enrichment` group)
- `clinical_enrichment_stats` - Add clinical evidence to graph
- `clinical_validation_stats` - Validate enrichment

### 4. Drug Discovery (`drug_discovery` group)
- `drug_discovery_results` - Find drug repurposing candidates
  - Compares base (topology-only) vs enhanced (with clinical evidence) approaches
  - Logs all results to MLflow

## Inspecting Data

All intermediate data is saved to disk for inspection:

```
data/
├── 01_raw/                  # Raw downloaded data
├── 02_intermediate/         # Processed clinical data
├── 03_primary/             # Enriched graph data
└── 07_model_output/        # Final results
    └── drug_discovery_results.csv  # Ranked drug candidates
```

## MLflow Tracking

Every asset execution is tracked in MLflow:

**Logged Information:**
- Parameters (disease ID, query type, etc.)
- Metrics (number of candidates, score improvements, etc.)
- Artifacts (CSV files with results)
- Execution time

**View in MLflow UI:**
1. Start MLflow: `make mlflow`
2. Navigate to: http://localhost:5000
3. Click on experiment: "clinical-drug-discovery"
4. View runs, compare metrics, download artifacts

## Example Workflow

**Terminal 1 - MLflow Server:**
```bash
make mlflow
```

**Terminal 2 - Dagster:**
```bash
make dagster
```

**In Dagster UI (http://localhost:3000):**
1. Go to Assets
2. Select `drug_discovery_results`
3. Click "Materialize selected"
4. Wait for completion

**In MLflow UI (http://localhost:5000):**
1. Click "clinical-drug-discovery" experiment
2. View latest run
3. Check metrics:
   - `base_candidates_found`
   - `enhanced_candidates_found`
   - `max_score_improvement`
4. Download artifacts (CSV files)

## Output Example

After running `drug_discovery_results`, check the output:

```bash
cat data/07_model_output/drug_discovery_results.csv
```

Example output:
```
drug_name,score_enhanced,base_score,clinical_score,evidence_type,disease_name
Rapamycin,125.3,45.0,1.6,Graph + Positive Clinical,Castleman disease
Sirolimus,118.7,42.0,1.5,Graph + Positive Clinical,Castleman disease
```

## Troubleshooting

### "Neo4j connection refused"
- Ensure Neo4j is running
- Check `.env` file has correct credentials

### "MLflow server already running"
- Stop existing MLflow: `pkill -f "mlflow server"`
- Or use a different port: `uv run mlflow server --port 5001`

### "Dagster already running"
- Stop existing Dagster: `pkill -f "dagster dev"`

## Next Steps

### Enable Link Prediction Assets (Advanced)

The link prediction assets are disabled by default due to OpenMP dependency requirements.

**To enable:**

1. Install OpenMP:
   ```bash
   brew install libomp
   ```

2. Enable in `src/dagster_definitions/assets/__init__.py`:
   - Uncomment the link prediction imports
   - Add asset names to `__all__` list

3. Restart Dagster:
   ```bash
   make dagster
   ```

See `docs/DRUG_DISCOVERY_PIPELINE.md` for full documentation.

## Development

### Run Tests
```bash
make test
```

### Clean Cache
```bash
make clean
```

### View Logs
Dagster logs are displayed in the terminal and in the UI under "Runs".

MLflow logs are in `mlruns/` directory.
