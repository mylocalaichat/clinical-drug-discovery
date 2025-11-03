# Embedding Visualization Guide

This guide explains how to extract and visualize the learned embeddings from your trained R-GCN model.

## What Are Embeddings?

After training, the model learns a **128-dimensional vector representation** (embedding) for each node in the graph. These embeddings capture:
- Semantic relationships between nodes
- Structural patterns in the graph
- What the model learned about each drug, disease, protein, etc.

## Step 1: Extract Embeddings

After training the model, extract the embeddings:

```bash
dagster asset materialize --select offlabel_model_embeddings
```

This will save:
- **CSV files**: `data/06_models/offlabel/embeddings_<node_type>.csv`
  - Format: `node_id, node_name, dim_0, dim_1, ..., dim_127`
- **NPZ file**: `data/06_models/offlabel/embeddings_all.npz`
  - Numpy arrays for all node types

## Step 2: Visualize Embeddings

### Basic Visualization

Visualize drug embeddings using t-SNE:

```bash
python scripts/visualize_embeddings.py --node-type drug --method tsne
```

Visualize disease embeddings using UMAP:

```bash
python scripts/visualize_embeddings.py --node-type disease --method umap
```

### Highlight Specific Nodes

Highlight specific drugs to see where they cluster:

```bash
python scripts/visualize_embeddings.py \
  --node-type drug \
  --highlight "Topiramate,Tamoxifen,Paclitaxel,Docetaxel" \
  --method tsne
```

### Find Similar Nodes

Find drugs most similar to Topiramate in embedding space:

```bash
python scripts/visualize_embeddings.py \
  --node-type drug \
  --similar-to "Topiramate"
```

This uses **cosine similarity** in the 128-dimensional embedding space to find the top 10 most similar drugs.

## Understanding the Visualizations

### What to Look For:

1. **Clustering**: Drugs/diseases with similar therapeutic uses should cluster together
   - Example: All chemotherapy drugs should be near each other

2. **Distance**: Distance in the 2D plot reflects similarity in the model's learned representation
   - Closer = more similar in the 128-D space

3. **Outliers**: Nodes far from others may be:
   - Unique/rare in the graph
   - Poorly connected
   - Learned as distinct by the model

### Expected Patterns (After Correct Training):

- **Cancer drugs** should cluster together
- **Cardiovascular drugs** should form their own cluster
- **Diseases** with similar symptoms/pathways should be near each other
- **Drugs with similar indications** should be close

### Red Flags (Indicating Model Issues):

- Random/uniform distribution (model didn't learn structure)
- Drugs for same disease are far apart
- Known similar drugs are in opposite corners

## Available Methods

### t-SNE (default)
- Good for: Finding local clusters
- Preserves: Local structure
- Pros: Reveals clusters well
- Cons: Distorts global distances, non-deterministic

```bash
python scripts/visualize_embeddings.py --node-type drug --method tsne
```

### UMAP
- Good for: Both local and global structure
- Preserves: Better global structure than t-SNE
- Pros: Faster, more consistent
- Cons: Requires `umap-learn` package

```bash
pip install umap-learn
python scripts/visualize_embeddings.py --node-type drug --method umap
```

### PCA
- Good for: Quick overview, linear structure
- Preserves: Variance
- Pros: Fast, deterministic, interpretable
- Cons: May miss non-linear patterns

```bash
python scripts/visualize_embeddings.py --node-type drug --method pca
```

## Advanced Usage

### Batch Visualization

Create visualizations for all node types:

```bash
for node_type in drug disease protein; do
  python scripts/visualize_embeddings.py \
    --node-type $node_type \
    --method tsne \
    --output data/07_model_output/offlabel/visualizations/${node_type}_tsne.png
done
```

### Programmatic Access

Load embeddings in Python:

```python
import pandas as pd
import numpy as np

# Option 1: Load from CSV (includes node IDs and names)
drug_embeddings_df = pd.read_csv('data/06_models/offlabel/embeddings_drug.csv')
node_ids = drug_embeddings_df['node_id'].values
node_names = drug_embeddings_df['node_name'].values
embeddings = drug_embeddings_df.iloc[:, 2:].values  # 128-D vectors

# Option 2: Load from NPZ (faster, numpy arrays only)
data = np.load('data/06_models/offlabel/embeddings_all.npz')
drug_embeddings = data['drug_embeddings']
disease_embeddings = data['disease_embeddings']
```

### Clustering Analysis

Use embeddings for clustering:

```python
from sklearn.cluster import KMeans
import pandas as pd

# Load embeddings
df = pd.read_csv('data/06_models/offlabel/embeddings_drug.csv')
embeddings = df.iloc[:, 2:].values

# Cluster drugs
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Add cluster labels
df['cluster'] = clusters
df.groupby('cluster')['node_name'].apply(list)
```

### Similarity Search

Find drugs similar to a query drug:

```python
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

df = pd.read_csv('data/06_models/offlabel/embeddings_drug.csv')
embeddings = df.iloc[:, 2:].values

# Find index of query drug
query_drug = "Topiramate"
query_idx = df[df['node_name'] == query_drug].index[0]

# Compute similarities
similarities = cosine_similarity([embeddings[query_idx]], embeddings)[0]

# Get top 10 most similar
top_indices = similarities.argsort()[::-1][1:11]  # Exclude self
for i, idx in enumerate(top_indices, 1):
    print(f"{i}. {df.iloc[idx]['node_name']} (similarity: {similarities[idx]:.4f})")
```

## Troubleshooting

### "Embeddings file not found"
Run the extraction step first:
```bash
dagster asset materialize --select offlabel_model_embeddings
```

### UMAP not installed
Install it:
```bash
pip install umap-learn
```

### Visualization looks random
This may indicate:
1. Model hasn't been trained yet
2. Training failed or converged poorly
3. Not enough training data or epochs
4. Graph structure is too sparse

Check training metrics (AUC-ROC should be > 0.7 for good embeddings)

## Output Files

After running the scripts, you'll have:

```
data/06_models/offlabel/
├── embeddings_drug.csv              # Drug embeddings with IDs/names
├── embeddings_disease.csv           # Disease embeddings
├── embeddings_protein.csv           # Protein embeddings
├── embeddings_biological_process.csv
├── embeddings_pathway.csv
├── embeddings_molecular_function.csv
├── embeddings_effect_phenotype.csv
└── embeddings_all.npz               # All embeddings (numpy format)

data/07_model_output/offlabel/visualizations/
├── drug_embeddings_tsne.png
├── drug_embeddings_umap.png
├── disease_embeddings_tsne.png
└── ...
```

## Next Steps

After visualizing embeddings, you can:

1. **Validate** the model learned meaningful patterns
2. **Debug** if clusters don't make sense
3. **Discover** new drug-disease relationships by finding similar nodes
4. **Analyze** which drugs cluster together (may share mechanisms)
5. **Export** embeddings for downstream tasks (transfer learning)
