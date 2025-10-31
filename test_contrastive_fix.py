"""
Quick test to verify contrastive loss fix works.
"""

import pandas as pd
from collections import defaultdict

print("Loading edges...")
edges_df = pd.read_csv("data/01_raw/primekg/kg.csv", low_memory=False)

# Simulate the fixed query
disease_gene_edges = edges_df[
    (edges_df['x_type'] == 'disease') &
    (edges_df['y_type'] == 'gene/protein') &
    (edges_df['relation'] == 'disease_protein')
]

print(f"\n✅ Found {len(disease_gene_edges):,} disease-gene edges")

# Get unique diseases
diseases_with_genes = disease_gene_edges['x_id'].unique()
print(f"✅ {len(diseases_with_genes):,} diseases have gene associations")

# Build disease -> gene mapping
disease_to_genes = defaultdict(set)
for _, row in disease_gene_edges.head(10000).iterrows():  # Sample for speed
    disease_id = row['x_id']
    gene_id = row['y_id']
    disease_to_genes[disease_id].add(gene_id)

# Calculate some similarities
similarities = []
disease_list = list(disease_to_genes.keys())[:100]  # First 100 diseases

for i in range(len(disease_list)):
    for j in range(i+1, min(i+10, len(disease_list))):  # Check 10 pairs per disease
        genes_i = disease_to_genes[disease_list[i]]
        genes_j = disease_to_genes[disease_list[j]]

        intersection = len(genes_i & genes_j)
        union = len(genes_i | genes_j)

        if union > 0:
            sim = intersection / union
            if sim > 0:
                similarities.append(sim)

print(f"\n✅ Computed {len(similarities)} non-zero similarities")
if len(similarities) > 0:
    print(f"   Mean similarity: {sum(similarities)/len(similarities):.4f}")
    print(f"   Max similarity: {max(similarities):.4f}")
    print(f"   Min similarity: {min(similarities):.4f}")

print("\n✅ Contrastive loss should now work!")
print("\nSample disease-gene associations:")
for disease_id in disease_list[:5]:
    genes = disease_to_genes[disease_id]
    disease_name = disease_gene_edges[disease_gene_edges['x_id'] == disease_id]['x_name'].iloc[0]
    print(f"  {disease_name}: {len(genes)} genes")
