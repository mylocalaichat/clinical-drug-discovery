# GitHub Copilot Instructions

## Project Overview

This is a **clinical drug discovery pipeline** that combines PrimeKG knowledge graphs with clinical evidence from medical notes. The system uses **Dagster for orchestration**, **Memgraph/Neo4j for graph storage**, and **MLflow for experiment tracking**.

## Architecture Patterns

### Dagster Asset-Based Pipeline
- All data transformations are **Dagster assets** in `src/dagster_definitions/assets/`
- Assets are grouped by function: `data_loading`, `clinical_extraction`, `graph_enrichment`, `drug_discovery`, `embeddings`
- **Sequential execution only** (`max_concurrent=1`) to prevent resource exhaustion from ML training
- Use `@asset(group_name="...", compute_kind="...")` decorator pattern

### Data Layer Architecture
```
data/
├── 01_raw/primekg/          # PrimeKG dataset (nodes.csv, edges.csv)
├── 06_models/embeddings/    # Node2Vec/GNN embeddings
├── 06_models/xgboost/       # ML models
└── 07_model_output/         # Final results
```

### Graph Database Pattern
- **Memgraph** is the primary graph database (compatible with Neo4j drivers)
- Connection via `GraphDatabase.driver()` with environment variables
- Empty credentials for local Memgraph: `auth=None if not (user or password)`
- Database parameter ignored (single database model)

### MLflow Integration
- **Every asset logs to MLflow** for experiment tracking
- Use `mlflow.start_run()` context managers in asset functions
- Artifacts saved to `mlruns/` directory
- Access via `http://localhost:5000`

## Essential Workflows

### Development Commands
```bash
make install          # uv sync dependencies
make dagster         # Start Dagster UI (localhost:3000)
make mlflow          # Start MLflow UI (localhost:5000)
make test           # Validate imports and connections
```

### Asset Materialization Patterns
- **Manual execution only** - no schedules/sensors active
- Materialize via Dagster UI or CLI: `uv run dagster asset materialize -m dagster_definitions`
- Assets auto-run upstream dependencies
- Use `+` suffix for downstream selection: `asset_name+`

### GNN/Embedding Workflow
- CSV-based loading (no graph database dependency)
- Node type filtering: excludes `cellular_component` and `exposure` by default
- **MPS compatibility**: `PYTORCH_ENABLE_MPS_FALLBACK=1` for Apple Silicon
- Memory-efficient with chunked processing

## Code Conventions

### Asset Definition Pattern
```python
@asset(group_name="group_name", compute_kind="technology")
def asset_name(
    context: AssetExecutionContext,
    dependency_asset: ReturnType,
) -> ReturnType:
    """Docstring explaining the asset."""
    context.log.info("Asset execution started")
    
    # MLflow tracking
    with mlflow.start_run():
        # Asset logic
        result = process_data()
        mlflow.log_metric("key", value)
    
    # Dagster metadata
    context.add_output_metadata({
        "key": MetadataValue.type(value),
    })
    
    return result
```

### Graph Connection Pattern
```python
def get_database_session(driver, database: str = None):
    """Database parameter ignored for Memgraph."""
    return driver.session()

# Connection setup
auth = None if not (user or password) else (user, password)
driver = GraphDatabase.driver(uri, auth=auth)
```

### Environment Configuration
- `.env` file with `MEMGRAPH_*`, `DAGSTER_POSTGRES_*`, `MLFLOW_*` variables
- PostgreSQL for Dagster state storage (not data processing)
- scispaCy model: `en_ner_bc5cdr_md` for biomedical NER

## Integration Points

### Drug Discovery Methods
1. **Base Discovery**: Graph topology queries via Cypher
2. **Clinical Enhancement**: Add clinical evidence relationships  
3. **Embedding Enhancement**: Use Node2Vec similarity for drug repurposing

### ML Pipeline Integration
- **Node2Vec embeddings** → cosine similarity → drug similarity matrix
- **XGBoost training** on drug-disease pairs with embeddings
- **Link prediction** for missing drug-disease relationships

### Data Flow Dependencies
```
download_data → neo4j_setup → nodes_loaded → edges_loaded
                                     ↓
clinical_extraction → graph_enrichment → drug_discovery
                                     ↓
embeddings → similarity_matrix → enhanced_discovery
```

## Testing & Debugging

### Validation Patterns
- Test diseases: Castleman disease (ID: 15564), Ovarian cancer (ID: 8170)
- Check imports with: `python -c "from dagster_definitions import defs"`
- Asset metadata provides execution statistics and previews

### Common Issues
- **OpenMP dependency**: Link prediction assets disabled by default
- **Memory management**: Sequential execution prevents OOM in GNN training
- **MPS compatibility**: Automatic fallback for unsupported operations on Apple Silicon

## Key Files to Understand

- `src/dagster_definitions/__init__.py` - Pipeline configuration
- `src/clinical_drug_discovery/lib/` - Core business logic
- `src/dagster_definitions/assets/` - Dagster asset definitions
- `pyproject.toml` - Dependencies and versions
- `Makefile` - Development commands