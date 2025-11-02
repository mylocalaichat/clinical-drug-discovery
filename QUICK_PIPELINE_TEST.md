# Quick Pipeline Test - 10 Epochs

> **⚠️ OBSOLETE:** This document was for testing HGT embeddings which have been removed from the project. The content below is no longer applicable.

## Strategy

✅ **Smart approach:** Get end-to-end pipeline working first, optimize later.

### Phase 1: Quick Test (Now)
- **10 epochs** (~5 minutes)
- Verify entire pipeline works
- Check all components integrate
- Identify any blockers

### Phase 2: Production (Later)
- **100+ epochs** (~7-8 hours)
- High-quality embeddings
- Full convergence
- Production-ready

---

## Configuration Updated

**Changed:** `num_epochs: 100 → 10`

**File:** `src/dagster_definitions/assets/embeddings.py` line 172

```python
"num_epochs": 10,  # Quick test - increase to 100+ for production
```

---

## What to Expect (10 Epochs)

### Training Time
```
10 epochs × ~30s/epoch = ~5 minutes total ✅
```

### Loss Values (Quick Training)
```
Epoch 1:  link: ~1.68  contr: ~0.00   total: ~0.84
Epoch 5:  link: ~1.40  contr: ~0.35   total: ~0.87
Epoch 10: link: ~1.35  contr: ~0.33   total: ~0.84
```

**Note:** Loss won't converge fully with only 10 epochs, but that's OK for testing!

### Embedding Quality (10 Epochs)
```
⚠️  Not fully trained but usable:
- Mean similarity: ~0.6-0.7 (vs 0.4-0.5 fully trained)
- PCA components: ~8-12 (vs 15-20+ fully trained)
- Still much better than collapsed (0.926 similarity, 4 components)
```

---

## Pipeline Components to Test

With 10-epoch embeddings, you can test:

### 1. ✅ Embedding Generation
- HGT training completes
- CSV saved correctly
- Contrastive loss working

### 2. ✅ Flattening
- `hgt_flattened_embeddings` asset
- 512 dimensions per node
- Correct format for ML

### 3. ✅ XGBoost Training
- Loads embeddings
- Trains drug-disease predictor
- Generates predictions

### 4. ✅ Evaluation
- Metrics calculated
- Validation working
- Results interpretable

### 5. ✅ Prediction Pipeline
- Off-label predictions
- Drug repurposing candidates
- Output format correct

---

## How to Run

### 1. Restart Training with 10 Epochs

```bash
# Stop current training (if running)
# In Dagster UI: Cancel job

# Delete partial embeddings
rm ./data/06_models/embeddings/hgt_embeddings.csv

# Restart Dagster
dagster dev

# Materialize hgt_embeddings
# Will now run for 10 epochs (~5 minutes)
```

### 2. After HGT Completes (~5 min)

**Continue through pipeline:**

```bash
# In Dagster UI, materialize in order:
1. hgt_embeddings ✅ (just completed)
2. hgt_flattened_embeddings (~1 min)
3. [Whatever your next assets are]
```

### 3. Quick Validation

```bash
# Validate embeddings (optional)
python validate_hgt_embeddings.py ./data/06_models/embeddings/hgt_embeddings.csv
```

**Expected results (10 epochs):**
- ⚠️  Mean similarity: ~0.6-0.7 (higher than ideal, but OK)
- ⚠️  PCA components: ~8-12 (lower than ideal, but OK)
- ✅ Shows variance (not collapsed)
- ✅ Contrastive loss working

---

## After Pipeline Test Succeeds

### Then Run Full Training

Once you've verified the entire pipeline works:

```python
# Edit embeddings.py line 172:
"num_epochs": 100,  # Or even 200 for best quality
```

**Run overnight:**
```bash
dagster dev
# Materialize hgt_embeddings
# Let run for 7-8 hours
```

**Result:** High-quality production embeddings

---

## Troubleshooting (10 Epochs)

### If Pipeline Breaks Downstream

**Examples:**
- XGBoost fails to load embeddings
- Format issues
- Missing columns
- Integration problems

**Action:**
- Fix the issue
- Can quickly re-run with 10 epochs to test fix
- Don't waste time on 100-epoch reruns until pipeline is solid

### If Embeddings Look Terrible

**Symptoms:**
- Mean similarity > 0.9 (still collapsed)
- PCA components < 5
- Contrastive loss still 0.0000

**Action:**
- Check contrastive loss in logs
- Verify fix was applied
- Debug before running 100 epochs

---

## Timeline

### Quick Test (Now)
```
HGT training (10 epochs):       ~5 min
Flattening:                     ~1 min
XGBoost training:               ~5-10 min (depends on your config)
Evaluation:                     ~1 min
Total:                          ~15-20 min ✅

Result: Know if pipeline works end-to-end
```

### Production Run (Later)
```
HGT training (100 epochs):      ~7-8 hours
Flattening:                     ~1 min
XGBoost training:               ~5-10 min
Evaluation:                     ~1 min
Total:                          ~7-8 hours

Result: Production-quality predictions
```

---

## Success Criteria (10 Epochs)

**Pipeline test succeeds if:**

✅ HGT training completes (10 epochs, ~5 min)
✅ Contrastive loss > 0 (e.g., 0.30-0.35)
✅ Embeddings generated (hgt_embeddings.csv exists)
✅ Flattening works (hgt_flattened_embeddings.csv exists)
✅ XGBoost trains successfully
✅ Predictions generated
✅ No errors in pipeline

**Don't worry about:**
- ⚠️  Loss not fully converged (expected with 10 epochs)
- ⚠️  Embeddings not perfect (expected)
- ⚠️  Predictions not optimal (expected)

**Goal:** Verify the plumbing works, not the quality.

---

## Production Configuration (For Later)

When ready for production run:

```python
# embeddings.py - Production settings
embedding_params = {
    "num_epochs": 100,        # Or 200 for best quality
    "learning_rate": 0.001,
    # ... rest same
}
```

**Then:**
```bash
# Run overnight or over weekend
dagster dev
# Materialize hgt_embeddings (7-8 hours)
# Then materialize rest of pipeline
```

---

## Summary

| Aspect | Quick Test (10 epochs) | Production (100 epochs) |
|--------|------------------------|-------------------------|
| **Time** | ~5 min | ~7-8 hours |
| **Purpose** | Verify pipeline | Production quality |
| **Quality** | Usable but not optimal | High quality |
| **When** | **Now** | After pipeline validated |
| **Goal** | Find issues fast | Best embeddings |

---

## Current Status

✅ **Configuration updated:** 10 epochs
✅ **Contrastive loss:** Fixed (will work in new run)
✅ **CPU training:** Enabled (reliable)
✅ **Ready to start:** Restart training now

---

**Restart training with 10 epochs to test the pipeline end-to-end. Takes only ~5 minutes!**
