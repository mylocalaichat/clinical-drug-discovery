# HGT Embedding Collapse - Root Cause Analysis & Fixes

## Executive Summary

**Status**: 🚨 CRITICAL - HGT embeddings have collapsed and are NOT learning meaningful representations.

**Evidence**:
- All disease nodes have **identical embeddings** (cosine similarity = 1.0)
- Many drug and gene nodes share identical embeddings
- 90% of variance captured by only 4 PCA components (should be 20-30+)
- Within-type similarity: diseases (0.997), pathways (1.000), biological_process (0.998)

## Validation Results

### Disease Embeddings (Sample of 100)
```
All diseases have IDENTICAL statistics:
  Mean: -0.00387
  Std:   0.03947

Cosine Similarity (first 20 diseases):
  Mean: 0.926
  Max:  1.000
  308 out of 380 pairs have perfect similarity (>0.999)
```

### PCA Analysis
```
PC1: 49.57% variance
PC2: 21.67% variance
PC3: 16.08% variance
PC4:  5.11% variance
---
90% variance in 4 components (CRITICAL - should be 20-30+)
```

### Node Type Separation
```
Within-type vs Across-type similarity:
  biological_process: 0.998 vs -0.016 (too high)
  disease:            0.997 vs  0.068 (too high - nearly identical!)
  pathway:            1.000 vs  0.011 (IDENTICAL embeddings!)
  gene/protein:       0.953 vs  0.038 (very high)
  drug:               0.707 vs -0.040 (moderate, but still concerning)
```

## Root Causes Identified

### 1. **Insufficient Embedding Dimensions** (CRITICAL)
**Current**: 160 dimensions
**Configured**: 512 dimensions
**What happened**: Auto-scaling reduced dimensions by 68% due to memory constraints (lines 564-592 in gnn_hgt.py)

```python
# From gnn_hgt.py:564-592
if device == 'mps' and total_memory_gb > 10:
    # Auto-adjust dimensions to fit in memory
    scale_factor = target_memory_gb / total_memory_gb
    new_embedding_dim = int(embedding_dim * scale_factor)  # 512 -> 160
```

**Impact**:
- Model doesn't have enough capacity to represent 66,903 diverse nodes
- With only 160 dimensions for drugs, diseases, genes, pathways, etc., the model collapses to a small subspace

### 2. **Too Few Training Epochs** (HIGH)
**Current**: 10 epochs
**Recommended**: 100-500 epochs for heterogeneous graphs

**Impact**:
- Model hasn't converged
- Loss likely still decreasing
- Embeddings haven't had time to separate

### 3. **Aggressive Edge Sampling** (MEDIUM)
**Current**: 1,000 edges per edge type per epoch
**Total edges**: 200k+ edges

```python
# From gnn_hgt.py:653
num_edges = min(edge_index.size(1), edge_sample_size)  # edge_sample_size=1000
```

**Impact**:
- Model only sees ~5% of edges per epoch
- Important relationships may never be learned
- Training signal is noisy and incomplete

### 4. **Weak Input Features** (MEDIUM)
**Current**: All nodes initialized with `torch.ones((num_nodes, 1))`

```python
# From gnn_hgt.py:218
data[node_type].x = torch.ones((num_nodes, 1), dtype=torch.float)
```

**Impact**:
- No initial signal for model to work with
- All nodes start identical - model must learn everything from scratch
- No node-specific features (e.g., drug properties, disease categories)

### 5. **Potential Learning Rate Issues** (LOW-MEDIUM)
**Current**: 0.001 (from embeddings.py:159)

**Considerations**:
- May be too low for 10 epochs (slow convergence)
- Or too high causing instability
- No learning rate schedule/warmup

### 6. **Contrastive Loss Configuration** (LOW)
**Current**:
- `contrastive_weight`: 0.5
- `similarity_threshold`: 0.1

**Potential issues**:
- May be pulling all similar diseases together too aggressively
- Threshold of 0.1 may be too low (too many positive pairs)

## Recommended Fixes (Priority Order)

### Priority 1: Increase Embedding Dimensions ✅
**Action**: Remove auto-scaling or increase target memory

**Option A - Use CPU for larger embeddings**:
```python
embedding_params = {
    "embedding_dim": 512,  # Full size
    "hidden_dim": 256,
    "device": "cpu",  # More memory available
    "num_epochs": 100,
}
```

**Option B - Reduce graph size**:
```python
embedding_params = {
    "embedding_dim": 512,
    "hidden_dim": 256,
    "limit_nodes": 30000,  # Reduce from 66k nodes
    "include_node_types": ['drug', 'disease', 'gene/protein', 'pathway']  # Only 4 types
}
```

**Option C - Adjust auto-scaling logic**:
```python
# In gnn_hgt.py:564-592
if device == 'mps' and total_memory_gb > 10:
    # More conservative scaling - minimum 384 dims instead of auto
    new_embedding_dim = max(384, int(embedding_dim * scale_factor))
    new_hidden_dim = max(192, int(hidden_dim * scale_factor))
```

### Priority 2: Increase Training Epochs ✅
**Action**: Train for 100-200 epochs instead of 10

```python
embedding_params = {
    "num_epochs": 100,  # 10x increase
}
```

**Expected impact**: Embeddings will have time to separate and converge

### Priority 3: Reduce Edge Sampling ✅
**Action**: Increase edge_sample_size or remove sampling entirely

```python
# In train_hgt_embeddings call:
train_hgt_embeddings(
    edge_sample_size=10000,  # 10x increase (from 1000)
    # Or remove sampling for edge types with < 10k edges
)
```

### Priority 4: Add Better Input Features (OPTIONAL)
**Action**: Initialize nodes with more informative features

**For diseases**:
- One-hot encode disease categories
- Use disease-gene association counts

**For drugs**:
- One-hot encode drug classes
- Use drug-target counts

**For genes**:
- Random initialization (Xavier/He)
- Or pre-trained gene embeddings

### Priority 5: Learning Rate Schedule (OPTIONAL)
**Action**: Add cosine annealing or step decay

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# In training loop:
optimizer.step()
scheduler.step()
```

## Recommended Action Plan

### Quick Fix (Test within 1 hour)
1. Set `device="cpu"` to allow full 512 dimensions
2. Increase `num_epochs` to 50
3. Increase `edge_sample_size` to 5000
4. Re-run training and validation

### Full Fix (Production quality)
1. Implement all Priority 1-3 fixes
2. Add early stopping based on validation loss
3. Add tensorboard logging to monitor:
   - Training loss
   - Embedding variance
   - Within-type vs across-type similarity
4. Consider adding node features (Priority 4)
5. Train for 200+ epochs with learning rate schedule

## Testing & Validation

After implementing fixes, validate using:

```bash
# Run training
dagster dev
# OR
python -c "from clinical_drug_discovery.lib.gnn_hgt import generate_hgt_embeddings; ..."

# Validate embeddings
python validate_hgt_embeddings.py ./data/06_models/embeddings/hgt_embeddings.csv
```

### Success Criteria
- [ ] Mean cosine similarity < 0.5 (currently 0.926)
- [ ] 90% variance requires 15+ PCA components (currently 4)
- [ ] Within-type similarity < 0.8 for diseases (currently 0.997)
- [ ] Visual separation in PCA plot
- [ ] Different diseases have different embedding statistics

## Files to Modify

1. **src/dagster_definitions/assets/embeddings.py:151-166**
   - Increase `num_epochs` from 10 to 100
   - Change `device` to "cpu" OR add `limit_nodes`
   - Increase `edge_sample_size` parameter

2. **src/clinical_drug_discovery/lib/gnn_hgt.py:485**
   - Add `edge_sample_size` parameter with default 10000
   - Update function signature

3. **src/clinical_drug_discovery/lib/gnn_hgt.py:564-592**
   - Adjust auto-scaling to preserve minimum 384 dimensions
   - Or disable auto-scaling when using CPU

## Expected Outcomes

**After Quick Fix**:
- Embeddings should show more variance
- PCA should require 10+ components for 90% variance
- Disease similarity should drop below 0.8

**After Full Fix**:
- Embeddings will be high-quality and ready for downstream tasks
- Good separation between different node types
- Similar diseases cluster together, but with clear differences
- Ready for XGBoost training and drug repurposing predictions

## References

- Validation script: `validate_hgt_embeddings.py`
- HGT implementation: `src/clinical_drug_discovery/lib/gnn_hgt.py`
- Asset definition: `src/dagster_definitions/assets/embeddings.py`
- Plots: `validation_plots/embedding_distributions.png`, `validation_plots/pca_visualization.png`
