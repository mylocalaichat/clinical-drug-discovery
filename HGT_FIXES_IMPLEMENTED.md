# HGT Embedding Fixes - Implementation Summary

## Changes Implemented

### ✅ Fix 1: Increased Training Epochs (CRITICAL)
**File**: `src/dagster_definitions/assets/embeddings.py:158`

```python
# BEFORE:
"num_epochs": 10,

# AFTER:
"num_epochs": 100,  # Increased from 10 to 100 for proper convergence
```

**Impact**: Model will train 10x longer, allowing embeddings to properly separate and converge.

---

### ✅ Fix 2: Force CPU Device (CRITICAL)
**File**: `src/dagster_definitions/assets/embeddings.py:160`

```python
# BEFORE:
"device": None,  # Auto-detect (cuda/mps/cpu)

# AFTER:
"device": "cpu",  # Use CPU for more memory (allows full 512 dims)
```

**Impact**:
- Prevents MPS auto-scaling that reduced dimensions from 512 → 160
- CPU has more available memory (16-32GB vs 10GB MPS limit)
- Ensures full 512-dimensional embeddings are trained

---

### ✅ Fix 3: Increased Edge Sampling (HIGH)
**File**: `src/dagster_definitions/assets/embeddings.py:163`

```python
# BEFORE:
# (not specified, default was 1000)

# AFTER:
"edge_sample_size": 5000,  # Increased from 1000 to see more edges per epoch
```

**Impact**: Model sees 5x more edges per epoch, improving signal quality.

---

### ✅ Fix 4: Improved Auto-Scaling Logic (MEDIUM)
**File**: `src/clinical_drug_discovery/lib/gnn_hgt.py:574-586`

```python
# BEFORE:
new_embedding_dim = max(new_embedding_dim, 64)   # Minimum 64
new_hidden_dim = max(new_hidden_dim, 32)         # Minimum 32

# AFTER:
new_embedding_dim = max(new_embedding_dim, 384)  # Minimum 384 (6x increase)
new_hidden_dim = max(new_hidden_dim, 192)        # Minimum 192 (6x increase)
```

**Impact**:
- If auto-scaling still triggers (e.g., future runs on MPS), minimum dimensions are much higher
- Prevents severe embedding collapse even with auto-scaling
- Added warning message about dimension preservation

---

### ✅ Fix 5: Parameter Propagation
**Files**:
- `src/clinical_drug_discovery/lib/gnn_hgt.py:835` (function signature)
- `src/clinical_drug_discovery/lib/gnn_hgt.py:896` (function call)
- `src/clinical_drug_discovery/lib/gnn_hgt.py:485` (default value)

**Changes**:
- Added `edge_sample_size` parameter to `generate_hgt_embeddings()`
- Added `edge_sample_size` parameter to `train_hgt_embeddings()`
- Updated default from 1000 to 5000
- Added documentation

---

## Files Modified

1. **src/dagster_definitions/assets/embeddings.py**
   - Lines 158, 160, 163: Updated hyperparameters

2. **src/clinical_drug_discovery/lib/gnn_hgt.py**
   - Lines 485: Updated `train_hgt_embeddings()` signature
   - Lines 574-586: Improved auto-scaling logic
   - Lines 835: Updated `generate_hgt_embeddings()` signature
   - Lines 854: Added parameter documentation
   - Lines 896: Pass `edge_sample_size` to training

3. **New Files Created**:
   - `validate_hgt_embeddings.py`: Comprehensive validation script
   - `test_hgt_fixes.py`: Quick test script
   - `HGT_EMBEDDING_COLLAPSE_REPORT.md`: Root cause analysis
   - `HGT_FIXES_IMPLEMENTED.md`: This file

---

## How to Run

### Option 1: Quick Test (Recommended First)
Test with a small subset to validate fixes work:

```bash
python test_hgt_fixes.py
```

**Expected runtime**: ~5-10 minutes
**What it tests**:
- Full 512 dimensions preserved
- Training runs without errors
- Quick validation of embeddings

---

### Option 2: Full Training with Dagster

```bash
# Start Dagster UI
dagster dev

# In UI: Materialize the "hgt_embeddings" asset
# This will trigger full training with 100 epochs
```

**Expected runtime**: ~2-4 hours (100 epochs on CPU)
**Output**: `data/06_models/embeddings/hgt_embeddings.csv`

---

### Option 3: Direct Python Script

```python
from clinical_drug_discovery.lib.gnn_hgt import generate_hgt_embeddings

stats = generate_hgt_embeddings(
    edges_csv="data/01_raw/kg.csv",
    output_csv="data/06_models/embeddings/hgt_embeddings.csv",
    embedding_dim=512,
    hidden_dim=256,
    num_layers=2,
    num_heads=8,
    num_epochs=100,
    learning_rate=0.001,
    device="cpu",
    edge_sample_size=5000,
    contrastive_weight=0.5,
    similarity_threshold=0.1,
    include_node_types=['drug', 'disease', 'gene/protein', 'pathway', 'biological_process']
)
```

---

## Validation

After training, validate the embeddings:

```bash
python validate_hgt_embeddings.py ./data/06_models/embeddings/hgt_embeddings.csv
```

### Success Criteria

**Before fixes**:
- ❌ Mean cosine similarity: 0.926 (too high!)
- ❌ 90% variance in 4 PCA components (collapsed!)
- ❌ Disease within-type similarity: 0.997 (nearly identical!)

**After fixes (expected)**:
- ✅ Mean cosine similarity: < 0.5
- ✅ 90% variance requires 15-20+ PCA components
- ✅ Disease within-type similarity: 0.5-0.8 (similar but distinct)
- ✅ Clear visual separation in PCA plot

---

## Expected Improvements

### 1. Embedding Dimensions
```
BEFORE: 160 dimensions (auto-scaled)
AFTER:  512 dimensions (full size)
IMPROVEMENT: 3.2x capacity increase
```

### 2. Training Duration
```
BEFORE: 10 epochs
AFTER:  100 epochs
IMPROVEMENT: 10x more training iterations
```

### 3. Edge Coverage
```
BEFORE: 1,000 edges/type/epoch (~0.5% of graph)
AFTER:  5,000 edges/type/epoch (~2.5% of graph)
IMPROVEMENT: 5x more edges seen per epoch
```

### 4. Model Capacity
```
BEFORE: 160 dims ÷ 66,903 nodes = 0.0024 dims per node
AFTER:  512 dims ÷ 66,903 nodes = 0.0077 dims per node
IMPROVEMENT: 3.2x representation capacity
```

---

## Monitoring During Training

Watch for these indicators of success:

### Loss Metrics (in progress bar)
```python
# Example output:
Training HGT: 45%|████▌     | 45/100 [15:23<17:32, 19.13s/epoch]
  link: 0.4521  contr: 0.3245  total: 0.3883
```

**Good signs**:
- Total loss decreasing steadily
- Loss stabilizing after ~50 epochs
- Both link and contrastive losses decreasing

**Bad signs**:
- Loss not decreasing after 20 epochs
- Loss exploding (>10.0)
- NaN values

---

## Troubleshooting

### Issue: Out of Memory (OOM) on CPU
**Solution**: Reduce node count
```python
"limit_nodes": 30000,  # Reduce from 66k
```

### Issue: Training too slow
**Solution**:
1. Reduce epochs for testing: `"num_epochs": 50`
2. Use smaller edge sample: `"edge_sample_size": 3000`
3. Consider GPU if available

### Issue: Embeddings still collapsing
**Symptoms**:
- Mean cosine similarity still > 0.9
- PCA collapse still present

**Solutions**:
1. Increase epochs to 200
2. Lower learning rate: `"learning_rate": 0.0005`
3. Add learning rate schedule
4. Check that device="cpu" (not auto-scaled to MPS)

---

## Performance Comparison

### Training Time Estimates

| Configuration | Epochs | Device | Estimated Time |
|--------------|--------|--------|----------------|
| Quick test | 20 | CPU | 5-10 min |
| Original (broken) | 10 | MPS | ~10 min |
| **Fixed (current)** | **100** | **CPU** | **2-4 hours** |
| Full training | 200 | CPU | 4-8 hours |

---

## Next Steps After Successful Training

1. **Validate embeddings**: Run `validate_hgt_embeddings.py`

2. **Check plots**: Review `validation_plots/pca_visualization.png`

3. **Use embeddings**: If validation passes, proceed with:
   - XGBoost training
   - Drug repurposing predictions
   - Clustering analysis

4. **Production deployment**:
   - Document final hyperparameters
   - Set up monitoring
   - Consider checkpointing for long runs

---

## Rollback Plan

If fixes cause issues, revert with:

```bash
git checkout HEAD~1 src/dagster_definitions/assets/embeddings.py
git checkout HEAD~1 src/clinical_drug_discovery/lib/gnn_hgt.py
```

Or manually change:
- `num_epochs`: 100 → 10
- `device`: "cpu" → None
- Remove `edge_sample_size` parameter

---

## Additional Optimizations (Future Work)

Not implemented but would help further:

1. **Learning rate schedule**: Cosine annealing
2. **Early stopping**: Stop if loss plateaus
3. **Node features**: Use actual drug/disease properties
4. **Batch contrastive loss**: More efficient computation
5. **Gradient clipping**: Prevent exploding gradients
6. **Mixed precision training**: Faster computation

---

## Summary

✅ **All critical fixes implemented**
✅ **Test script created**
✅ **Validation pipeline ready**
✅ **Documentation complete**

**Ready to train!** Start with `python test_hgt_fixes.py` for quick validation.
