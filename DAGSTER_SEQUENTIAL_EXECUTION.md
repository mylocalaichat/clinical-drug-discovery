# Dagster Configuration

## Manual Execution Only (No Automation)

Configured Dagster for **manual asset materialization only**:
- ❌ No scheduled jobs
- ❌ No sensors
- ❌ No automation
- ✅ Sequential execution (1 asset at a time)
- ✅ Manual control from UI

## What Was Changed

### 1. `dagster.yaml` - Run Launcher Configuration
Added multiprocess executor config with `max_concurrent: 1`:

```yaml
run_launcher:
  module: dagster.core.launcher
  class: DefaultRunLauncher
  config:
    run_config:
      execution:
        config:
          timeout: 3600
          multiprocess:
            max_concurrent: 1  # <-- Sequential execution
```

### 2. `src/dagster_definitions/__init__.py` - Removed All Automation
**Removed:**
- All schedules (weekly_data_refresh, monthly_pipeline_run)
- All sensors (primekg_update_sensor)
- All jobs (weekly_data_refresh_job, complete_pipeline_job)

**Kept:**
- Assets (can be materialized manually)
- Sequential executor configuration

```python
defs = Definitions(
    assets=all_assets,
    executor=multiprocess_executor.configured({"max_concurrent": 1}),
)
```

This ensures:
- No automatic execution
- Manual control only
- Sequential execution when you materialize assets

## Why This Matters

### Problem Without Sequential Execution:
- **Embedding training** (GNN, HGT) is resource-intensive:
  - High memory usage (loading full knowledge graph)
  - GPU/CPU intensive (training neural networks)
  - Long-running (100 epochs)

- If `gnn_embeddings` and `hgt_embeddings` run in parallel:
  - 2x memory consumption → OOM errors
  - GPU contention → slower training
  - System becomes unresponsive

### Solution:
- Assets now execute **one at a time**
- Example execution order:
  1. `download_data` ✓
  2. `gnn_embeddings` ✓ (waits for #1)
  3. `hgt_embeddings` ✓ (waits for #2)
  4. `flattened_embeddings` ✓ (waits for #3)
  5. `hgt_flattened_embeddings` ✓ (waits for #4)

## Trade-offs

### Pros:
- ✓ No resource exhaustion
- ✓ Predictable memory usage
- ✓ Stable execution on laptops/limited hardware
- ✓ GPU not oversubscribed

### Cons:
- ✗ Longer total pipeline runtime
- ✗ Can't parallelize independent assets (e.g., `download_data` + `clinical_extraction`)

## How to Use Dagster (Manual Execution)

### Starting Dagster:
```bash
dagster dev
```

Navigate to: http://localhost:3000

### Materializing Assets:
1. Go to **Assets** tab
2. Select the assets you want to materialize (e.g., `hgt_embeddings`)
3. Click **Materialize**
4. Assets will execute sequentially

### Available Assets:

**Data Loading:**
- `download_data` - Download PrimeKG from Harvard Dataverse

**Embeddings (choose one or both):**
- `gnn_embeddings` - GraphSAGE baseline
- `hgt_embeddings` - HGT with contrastive learning

**Flattened Embeddings:**
- `flattened_embeddings` - From GNN
- `hgt_flattened_embeddings` - From HGT

**Visualizations:**
- `embedding_visualizations` - GNN visualizations
- `hgt_embedding_visualizations` - HGT visualizations

### Typical Workflow:
1. Materialize `download_data` first
2. Then materialize `hgt_embeddings` (or `gnn_embeddings`)
3. Then materialize `hgt_flattened_embeddings`
4. Optionally materialize visualizations

## When You Might Want Parallelism

If you have a powerful server with lots of RAM/GPUs, you can:
1. Edit `dagster.yaml` and increase `max_concurrent`
2. Edit `__init__.py` and adjust executor config

## Verification

To verify sequential execution:

### In Dagster UI:
1. Start `dagster dev`
2. Navigate to http://localhost:3000
3. Materialize multiple assets
4. Check the run timeline → assets execute one after another

### In Logs:
```
[2024-01-15 10:00:00] Starting gnn_embeddings
[2024-01-15 10:15:00] Completed gnn_embeddings
[2024-01-15 10:15:01] Starting hgt_embeddings  <-- Waits for previous
[2024-01-15 10:30:00] Completed hgt_embeddings
```

## Current Configuration Summary

```
Mode: Manual execution only
Automation: None (no schedules, sensors, or jobs)
Concurrent runs: 1 (only 1 run at a time)
Concurrent assets within run: 1 (sequential execution)
Executor: multiprocess with max_concurrent=1
Timeout: 2 hours per operation
```

This ensures:
- ✅ Full manual control
- ✅ Minimal resource usage
- ✅ Predictable execution
- ✅ No surprise automatic runs
