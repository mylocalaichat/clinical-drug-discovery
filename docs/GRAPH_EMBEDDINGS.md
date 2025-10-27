# Graph Embeddings Pipeline

The graph embeddings functionality has been split into multiple Dagster assets that can be run independently. This allows for better modularity, easier debugging, and selective execution.

## Assets Overview

### 1. `knowledge_graph`
- **Purpose**: Load the knowledge graph from Neo4j into NetworkX format
- **Output**: NetworkX Graph object
- **Dependencies**: `clinical_validation_stats` (ensures clinical evidence is in graph)
- **Configuration**: Uses environment variables for Neo4j connection
- **Excludes**: INDICATION and CONTRAINDICATION edges to prevent data leakage

### 2. `node2vec_embeddings`
- **Purpose**: Train Node2Vec embeddings on the knowledge graph
- **Output**: Dictionary mapping node_id → embedding vector
- **Dependencies**: `knowledge_graph`
- **Parameters**: 512 dimensions, 30 walk length, 10 walks per node
- **Duration**: ~10-20 minutes depending on graph size

### 3. `saved_embeddings`
- **Purpose**: Persist embeddings to disk for reuse
- **Output**: File path to saved embeddings
- **Dependencies**: `node2vec_embeddings`
- **Location**: `data/06_models/embeddings/node2vec_embeddings.pkl`

### 4. `embedding_dataframe`
- **Purpose**: Convert embeddings dictionary to pandas DataFrame
- **Output**: DataFrame with node_id and embedding columns
- **Dependencies**: `node2vec_embeddings`
- **Use case**: Data analysis and exploration

### 5. `flattened_embeddings`
- **Purpose**: Flatten embedding vectors into individual columns for ML models
- **Output**: DataFrame with node_id and emb_0, emb_1, ..., emb_511 columns
- **Dependencies**: `embedding_dataframe`
- **Use case**: Feature engineering for supervised learning

### 6. `embedding_validation_stats`
- **Purpose**: Compute quality metrics and statistics for embeddings
- **Output**: Dictionary of validation statistics
- **Dependencies**: `node2vec_embeddings`, `flattened_embeddings`
- **Metrics**: Mean/std norms, value ranges, zero embeddings count

### 7. `drug_similarity_matrix` ⭐
- **Purpose**: Compute drug-drug similarity using embeddings
- **Output**: Similarity matrix between all drugs
- **Dependencies**: `flattened_embeddings`
- **Use case**: Drug repurposing and finding similar compounds

### 8. `embedding_enhanced_drug_discovery` ⭐
- **Purpose**: Enhanced drug discovery using embedding similarity
- **Output**: Drug candidates enhanced with similarity scores
- **Dependencies**: `drug_similarity_matrix`, `clinical_validation_stats`
- **Use case**: Find drugs similar to known treatments

## Usage Examples

### Run Individual Assets

```bash
# Load graph only
dagster asset materialize -m dagster_definitions -a knowledge_graph

# Train embeddings (requires knowledge_graph)
dagster asset materialize -m dagster_definitions -a node2vec_embeddings

# Save embeddings to disk
dagster asset materialize -m dagster_definitions -a saved_embeddings

# Convert to DataFrame format
dagster asset materialize -m dagster_definitions -a embedding_dataframe

# Flatten for ML models
dagster asset materialize -m dagster_definitions -a flattened_embeddings

# Validate embeddings quality
dagster asset materialize -m dagster_definitions -a embedding_validation_stats
```

### Run Full Pipeline

```bash
# Run all embedding assets in dependency order
dagster asset materialize -m dagster_definitions -a embedding_validation_stats
# This will automatically run all upstream dependencies:
# knowledge_graph → node2vec_embeddings → saved_embeddings, embedding_dataframe → flattened_embeddings → embedding_validation_stats
```

### Run Partial Pipeline

```bash
# Just load graph and train embeddings
dagster asset materialize -m dagster_definitions -a node2vec_embeddings

# Just create DataFrames (assumes embeddings exist)
dagster asset materialize -m dagster_definitions -a flattened_embeddings
```

## Asset Dependencies

```
                                    Data Loading Assets
                                            ↓
primekg_download → neo4j_database_ready → nodes/edges_loaded → drug/disease_features_loaded
                                                                         ↓
                                                           mtsamples_raw → clinical_drug_disease_pairs 
                                                                         ↓
                                                                clinical_enrichment_stats 
                                                                         ↓
                                                                clinical_validation_stats
                                                                         ↓
                                                                knowledge_graph
                                                                         ↓
                                                                node2vec_embeddings
                                                                         ↓                     ↓
                                                                saved_embeddings    embedding_dataframe
                                                                                             ↓
                                                                                       flattened_embeddings
                                                                                             ↓                    ↓
                                                                                    embedding_validation_stats   drug_similarity_matrix
                                                                                                                  ↓
                                                                                                          embedding_enhanced_drug_discovery
```

## Configuration

Assets use environment variables for configuration:

```bash
# Neo4j connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j
NEO4J_DATABASE=primekg
```

## Output Locations

```
data/06_models/embeddings/
├── node2vec_embeddings.pkl      # Saved embeddings dictionary
├── flattened_embeddings.csv     # Sample of flattened embeddings (first 1000 rows)
```

## Monitoring

Each asset provides rich metadata including:
- Performance metrics (timing, memory usage)
- Data quality metrics (shapes, statistics)
- Configuration parameters used
- File sizes and locations

View this metadata in the Dagster UI at `http://localhost:3000` after running assets.

## Benefits of Asset Splitting

1. **Selective Execution**: Run only what you need
2. **Better Caching**: Reuse expensive computations (embeddings)
3. **Easier Debugging**: Isolate issues to specific steps
4. **Parallel Development**: Work on different parts independently
5. **Resource Management**: Control memory usage by running assets separately
6. **Experiment Tracking**: Better observability for each step
7. **Logical Ordering**: Embeddings are extracted AFTER clinical evidence is added to the graph

## Why This Execution Order Matters

The pipeline now follows a logical data flow:

1. **📥 Data Loading First**: PrimeKG data must be loaded into Neo4j before any processing
2. **🔍 Clinical Extraction Second**: Name matching requires the database to be populated with drug/disease entities
3. **🔗 Graph Enrichment Third**: Clinical evidence is added to the enriched graph
4. **📊 Embeddings Last**: Embeddings are extracted from the fully enriched graph

This ensures:

- ✅ **Database is ready** before clinical extraction tries to match entity names
- ✅ **Clinical evidence is added** before embedding extraction
- ✅ **Embeddings capture the full enriched graph** structure including clinical relationships
- ✅ **Better representation quality** since embeddings include both topology and clinical evidence
- ✅ **No failed lookups** during name matching process

## Performance Notes

- `knowledge_graph`: Fast (~1-2 minutes)
- `node2vec_embeddings`: Slow (~10-20 minutes) - most expensive step
- `saved_embeddings`: Fast (~30 seconds)
- `embedding_dataframe`: Fast (~1 minute)
- `flattened_embeddings`: Medium (~2-5 minutes) - memory intensive
- `embedding_validation_stats`: Fast (~30 seconds)

The most expensive operation is training Node2Vec embeddings. Once trained, you can reuse the `saved_embeddings` asset and skip retraining.