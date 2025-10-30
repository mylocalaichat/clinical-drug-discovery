# Disabled Assets Summary

## Overview

This document tracks which Dagster asset groups have been disabled and why.

## Disabled Asset Groups

### 1. Clinical Extraction (3 assets)
**Status**: Disabled
**Reason**: User request
**Location**: Moved to `src/dagster_definitions/assets/_disabled/clinical_extraction.py`

**Assets**:
- `mtsamples_raw`
- `clinical_drug_disease_pairs`
- `clinical_extraction_stats`

### 2. Graph Enrichment (2 assets)
**Status**: Disabled
**Reason**: Depends on clinical_extraction assets
**Location**: Moved to `src/dagster_definitions/assets/_disabled/graph_enrichment.py`

**Assets**:
- `clinical_pairs_loaded`
- `clinical_validation_stats`

### 3. Drug Discovery (3 assets)
**Status**: Disabled
**Reason**: Depends on clinical_extraction assets
**Location**: Moved to `src/dagster_definitions/assets/_disabled/`

**Files moved**:
- `drug_discovery.py`
- `embedding_drug_discovery.py`

**Assets**:
- `drug_discovery_results`
- `drug_similarity_matrix`
- `embedding_enhanced_drug_discovery`

## Disabled Schedules & Jobs

### Schedules
- `daily_clinical_extraction` - Disabled in `src/dagster_definitions/schedules.py`

### Jobs
- `clinical_extraction_job` - Disabled in `src/dagster_definitions/schedules.py`

### Sensors
- `new_clinical_data_sensor` - Disabled in `src/dagster_definitions/sensors.py`

## Active Asset Groups

### 1. Data Loading (6 assets)
- `primekg_download_status`
- `memgraph_database_ready`
- `primekg_nodes_loaded`
- `primekg_edges_loaded`
- `drug_features_loaded`
- `disease_features_loaded`

### 2. Embeddings (6 assets)
- `random_graph_sample`
- `knowledge_graph`
- `node2vec_embeddings`
- `flattened_embeddings`
- `embedding_visualizations`
- `memgraph_embedding_visualizations`

### 3. XGBoost Drug Discovery (9 assets)
- `xgboost_known_drug_disease_pairs`
- `xgboost_node_embeddings`
- `xgboost_training_data`
- `xgboost_feature_vectors`
- `xgboost_trained_model`
- `xgboost_model_evaluation`
- `xgboost_all_drug_disease_pairs`
- `xgboost_predictions`
- `xgboost_ranked_results`

## Summary

**Total disabled**: 8 assets (3 clinical_extraction + 2 graph_enrichment + 3 drug_discovery)
**Total active**: 21 assets (6 data_loading + 6 embeddings + 9 xgboost_drug_discovery)

## Re-enabling Assets

To re-enable any of these asset groups:

1. Move the Python files from `src/dagster_definitions/assets/_disabled/` back to `src/dagster_definitions/assets/`
2. Uncomment the imports in `src/dagster_definitions/assets/__init__.py`
3. Uncomment the exports in the `__all__` list
4. Uncomment any related schedules/jobs in `src/dagster_definitions/schedules.py`
5. Uncomment any related sensors in `src/dagster_definitions/sensors.py`
6. Update `src/dagster_definitions/__init__.py` to include them in the Definitions

## Changes Made

### Files Modified
- `src/dagster_definitions/assets/__init__.py` - Commented out imports and exports
- `src/dagster_definitions/__init__.py` - Removed from schedules/sensors/jobs lists
- `src/dagster_definitions/schedules.py` - Disabled clinical_extraction_job
- `src/dagster_definitions/sensors.py` - Disabled new_clinical_data_sensor
- `src/dagster_definitions/assets/xgboost_drug_discovery.py` - Fixed `positive_df` → `positive_samples` bug

### Files Moved
- `clinical_extraction.py` → `_disabled/clinical_extraction.py`
- `graph_enrichment.py` → `_disabled/graph_enrichment.py`
- `drug_discovery.py` → `_disabled/drug_discovery.py`
- `embedding_drug_discovery.py` → `_disabled/embedding_drug_discovery.py`

## Notes

The `_disabled/` folder is ignored by `load_assets_from_package_module()` because module names starting with underscore are typically excluded from package auto-discovery.
