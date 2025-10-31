# Clinical-Enriched Drug Discovery

A Dagster pipeline that enriches PrimeKG knowledge graph with clinical evidence from medical notes to improve drug repurposing predictions.

## Overview

This project demonstrates how combining **graph topology** with **clinical evidence** can improve drug discovery:

1. **Data Loading**: Download PrimeKG knowledge graph and load into Neo4j
2. **Clinical Extraction**: Extract drug-disease co-occurrences from clinical notes (MTSamples) using NER
3. **Graph Enrichment**: Add clinical evidence relationships to Neo4j
4. **Drug Discovery**: Query for off-label drug candidates with and without clinical evidence
5. **Comparison**: Track results in MLflow to show improvement

### Expected Outcome

For a disease (e.g., Castleman's disease):
- **Before**: Top drug has score 450 (graph topology only)
- **After**: Top drug has score 550 (450 + 100 clinical boost)
- **Explanation**: Shows which proteins + clinical observations led to score

## Prerequisites

1. **PostgreSQL** (v12+)
   - For Dagster state storage
   - Assumes PostgreSQL is already installed and running on your machine
   - Install: `brew install postgresql` (Mac) or your system's package manager

2. **Neo4j Desktop or Server** (v5.x)
   - Download: https://neo4j.com/download/
   - Create a new database named `clinical-drug-discovery`
   - Note your username/password

3. **Python 3.10+** and **uv** package manager
   - Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`

4. **Disk Space**
   - ~500MB for PrimeKG dataset (automatically downloaded)
   - ~1GB for Neo4j database
   - ~50MB for PostgreSQL

## Installation

### 1. Clone and Setup

```bash
cd clinical-drug-discovery
uv sync
```

### 2. Setup PostgreSQL for Dagster

```bash
# Ensure PostgreSQL is running
# (Assumes PostgreSQL is already installed and started on your machine)

# Create Dagster database and schema
psql -U postgres -f setup_postgres.sql
```

This creates:
- Database: `clinical_drug_discovery_db`
- User: `dagster_user` / Password: `dagster_password_123`
- Schema: `dagster`

### 3. Configure Environment Variables

The `.env` file is already created with default values. Update if needed:

```bash
# Edit .env file
nano .env
```

Key variables:
- `DAGSTER_POSTGRES_*`: PostgreSQL connection details
- `NEO4J_*`: Neo4j connection details
- `MLFLOW_TRACKING_URI`: MLflow storage location

### 4. Install scispaCy NER Model

```bash
./install_models.sh
```

This installs the `en_ner_bc5cdr_md` biomedical NER model for clinical text extraction.

## Usage

### Start Dagster UI

```bash
uv run dagster dev -m dagster_definitions
```

This starts:
- **Dagster UI** at http://localhost:3000
- Webserver and daemon processes
- Connection to PostgreSQL for state storage

### Run the Full Pipeline

**Option 1: Via UI (Recommended)**

1. Open http://localhost:3000
2. Click "Assets" in the left sidebar
3. Click "Materialize all" button
4. Watch real-time execution with logs

**Option 2: Via CLI**

```bash
# Materialize all assets
uv run dagster asset materialize --select "*" -m dagster_definitions

# Or specific asset groups
uv run dagster asset materialize --select "tag:group_name=data_loading" -m dagster_definitions
```

### View Results

#### Dagster UI (Pipeline Execution)

```bash
uv run dagster dev -m dagster_definitions
```

Open http://localhost:3000 to view:
- **Real-time execution** with progress bars
- **Asset lineage** graph showing dependencies
- **Run history** stored in PostgreSQL
- **Logs** for each asset materialization
- **Metadata** and statistics

#### MLflow UI (Experiment Tracking)

```bash
uv run mlflow ui
```

Open http://localhost:5000 to view:
- Experiment runs for each disease
- Metrics (candidates found, clinical evidence added)
- Artifacts (result CSVs, comparison tables)
- Parameter tracking

#### Result Files

Results are saved to:
- `data/07_model_output/drug_discovery_results.csv` - Final comparison table
- `mlruns/` - MLflow tracking data

## Project Structure

```
clinical-drug-discovery/
├── src/
│   ├── clinical_drug_discovery/
│   │   ├── lib/                    # Business logic modules
│   │   │   ├── data_loading.py     # PrimeKG data loading functions
│   │   │   ├── clinical_extraction.py  # NER and text processing
│   │   │   ├── graph_enrichment.py # Neo4j graph enhancement
│   │   │   └── drug_discovery.py   # Drug discovery queries
│   │   └── queries/                # Cypher query files
│   │       └── base_discovery.cypher
│   └── dagster_definitions/        # Dagster orchestration
│       ├── __init__.py             # Definitions + Postgres config
│       └── assets/                 # Dagster assets
│           ├── data_loading.py
│           ├── clinical_extraction.py
│           ├── graph_enrichment.py
│           └── drug_discovery.py
├── .env                            # Environment variables
├── setup_postgres.sql              # PostgreSQL setup script
├── install_models.sh               # scispaCy model installer
└── README.md
```

## Asset Groups

The pipeline is organized into 4 asset groups:

### 1. **data_loading** (6 assets)
- `download_data` - Download PrimeKG from Harvard Dataverse
- `neo4j_database_ready` - Setup Neo4j database
- `primekg_nodes_loaded` - Load nodes into Neo4j
- `primekg_edges_loaded` - Load edges into Neo4j
- `drug_features_loaded` - Load drug metadata
- `disease_features_loaded` - Load disease metadata

### 2. **clinical_extraction** (3 assets)
- `mtsamples_raw` - Download MTSamples clinical notes
- `clinical_drug_disease_pairs` - Extract using NER
- `clinical_extraction_stats` - Compute statistics

### 3. **graph_enrichment** (2 assets)
- `clinical_enrichment_stats` - Add CLINICAL_EVIDENCE to Neo4j
- `clinical_validation_stats` - Validate relationships

### 4. **drug_discovery** (1 asset)
- `drug_discovery_results` - Query Neo4j + log to MLflow

## How Dagster Works

### State Management

All execution state is stored in **PostgreSQL**:
- **Run history**: Every pipeline execution
- **Event logs**: Step-by-step execution events
- **Asset materializations**: When each asset was last computed
- **Schedules & sensors**: Trigger configuration

Query state:
```sql
-- Connect to Postgres
psql -U dagster_user -d clinical_drug_discovery_db

-- View recent runs
SELECT * FROM dagster.runs
ORDER BY create_timestamp DESC
LIMIT 10;
```

### Asset Dependencies

Dagster automatically determines execution order based on asset dependencies:

```
download_data
    ↓
neo4j_database_ready
    ↓
primekg_nodes_loaded → primekg_edges_loaded
    ↓                       ↓
drug_features_loaded    disease_features_loaded
    ↓                       ↓
    ↓ ← mtsamples_raw → clinical_drug_disease_pairs
    ↓                       ↓
    ↓ ← clinical_enrichment_stats
    ↓                       ↓
    ↓ ← clinical_validation_stats
    ↓                       ↓
    → drug_discovery_results
```

### Re-running Assets

**Selective re-materialization:**
```bash
# Re-run just clinical extraction
uv run dagster asset materialize --select "tag:group_name=clinical_extraction" -m dagster_definitions

# Re-run from a specific asset onwards
uv run dagster asset materialize --select "clinical_drug_disease_pairs+" -m dagster_definitions
```

## Configuration

### Postgres Configuration

Edit `.env`:
```bash
DAGSTER_POSTGRES_USER=dagster_user
DAGSTER_POSTGRES_PASSWORD=dagster_password_123
DAGSTER_POSTGRES_HOST=localhost
DAGSTER_POSTGRES_PORT=5432
DAGSTER_POSTGRES_DB=clinical_drug_discovery_db
DAGSTER_POSTGRES_SCHEMA=dagster
```

### Neo4j Configuration

Edit `.env`:
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
NEO4J_DATABASE=clinical-drug-discovery
```

### Test Diseases

To add more diseases, edit `src/dagster_definitions/assets/drug_discovery.py`:

```python
TEST_DISEASES = [
    {"disease_id": "15564", "name": "Castleman disease"},
    {"disease_id": "8170", "name": "Ovarian cancer"},
    # Add more here
]
```

## Troubleshooting

### PrimeKG Download Issues (HTTP 403 Forbidden)

If you get a 403 Forbidden error when downloading PrimeKG data:

```bash
# The error indicates Harvard Dataverse access restrictions
# Manual download steps:

1. Visit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM
2. Register for a free Harvard Dataverse account
3. Request access to the PrimeKG dataset
4. Download these files manually to data/01_raw/primekg/:
   - nodes.csv
   - edges.csv
   - drug_features.csv
   - disease_features.csv
```

### PostgreSQL Connection Issues

```bash
# Check if Postgres is running
brew services list | grep postgresql

# Test connection
psql -U dagster_user -d clinical_drug_discovery_db -c "SELECT 1"
```

### Dagster Won't Start

```bash
# Clear Dagster cache
rm -rf $DAGSTER_HOME

# Rebuild Python package
uv sync

# Start with verbose logging
DAGSTER_LOG_LEVEL=DEBUG uv run dagster dev -m dagster_definitions
```

### Neo4j Connection Issues

```bash
# Verify Neo4j is running
neo4j status

# Test connection
cypher-shell -u neo4j -p your_password -d clinical-drug-discovery "RETURN 1"
```

### scispaCy Model Not Found

```bash
# Install model manually
uv run -- python -m pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

# Verify
uv run -- python -c "import spacy; spacy.load('en_ner_bc5cdr_md'); print('OK')"
```

## Development

### Adding New Assets

Create a new asset in `src/dagster_definitions/assets/`:

```python
from dagster import asset, AssetExecutionContext

@asset(group_name="my_group", compute_kind="python")
def my_new_asset(context: AssetExecutionContext, dependency_asset) -> dict:
    """Your asset logic here."""
    context.log.info("Running my asset")
    return {"status": "success"}
```

Then add it to `assets/__init__.py` exports.

### Scheduling

Add a schedule in `src/dagster_definitions/__init__.py`:

```python
from dagster import define_asset_job, ScheduleDefinition

# Define job
daily_job = define_asset_job("daily_pipeline", selection="*")

# Define schedule
daily_schedule = ScheduleDefinition(
    job=daily_job,
    cron_schedule="0 0 * * *",  # Daily at midnight
)

defs = Definitions(
    assets=all_assets,
    schedules=[daily_schedule],
    # ... storage config
)
```

## References

- **Dagster Docs**: https://docs.dagster.io/
- **PrimeKG**: Chandak et al., "Building a knowledge graph to enable precision medicine" (Scientific Data, 2023)
- **MTSamples**: Public clinical notes dataset
- **scispaCy**: Biomedical NER models for spaCy

## License

This project is for research and educational purposes only.
