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

- If multiple embedding assets run in parallel:
  - 2x memory consumption → OOM errors
  - GPU contention → slower training
  - System becomes unresponsive

### Solution:
- Assets now execute **one at a time**
- Example execution order:
  1. `download_data` ✓
  2. `gnn_embeddings` ✓ (waits for #1)
  3. `flattened_embeddings` ✓ (waits for #2)

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
2. Select the assets you want to materialize (e.g., `gnn_embeddings`)
3. Click **Materialize**
4. Assets will execute sequentially

### Available Assets:

**Data Loading:**
- `download_data` - Download PrimeKG from Harvard Dataverse

**Embeddings:**
- `gnn_embeddings` - GraphSAGE baseline

**Flattened Embeddings:**
- `flattened_embeddings` - From GNN

**Visualizations:**
- `embedding_visualizations` - GNN visualizations

### Typical Workflow:
1. Materialize `download_data` first
2. Then materialize `gnn_embeddings`
3. Then materialize `flattened_embeddings`
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
[2024-01-15 10:00:00] Starting download_data
[2024-01-15 10:05:00] Completed download_data
[2024-01-15 10:05:01] Starting gnn_embeddings  <-- Waits for previous
[2024-01-15 10:20:00] Completed gnn_embeddings
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
