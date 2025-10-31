"""
Diagnose why disease-gene associations aren't being found.
"""

import pandas as pd

print("Loading edges CSV...")
edges_df = pd.read_csv("data/01_raw/primekg/kg.csv", low_memory=False)

print(f"\nTotal edges: {len(edges_df):,}")

# Check what node types exist
print("\nUnique x_type values:")
print(edges_df['x_type'].value_counts().head(10))

print("\nUnique y_type values:")
print(edges_df['y_type'].value_counts().head(10))

# Check for disease nodes
disease_edges = edges_df[edges_df['x_type'] == 'disease']
print(f"\nEdges with x_type='disease': {len(disease_edges):,}")

# Check for gene/protein nodes
gene_edges = edges_df[edges_df['y_type'] == 'gene/protein']
print(f"Edges with y_type='gene/protein': {len(gene_edges):,}")

# Check for disease -> gene/protein edges
disease_gene_edges = edges_df[
    (edges_df['x_type'] == 'disease') &
    (edges_df['y_type'] == 'gene/protein')
]
print(f"\nEdges disease → gene/protein: {len(disease_gene_edges):,}")

if len(disease_gene_edges) > 0:
    print("\nRelations for disease → gene/protein:")
    print(disease_gene_edges['relation'].value_counts())

    print("\nSample edges:")
    print(disease_gene_edges.head(10)[['x_name', 'relation', 'y_name']])
else:
    print("\n❌ NO disease → gene/protein edges found!")

    # Check if edges go the other direction
    gene_disease_edges = edges_df[
        (edges_df['x_type'] == 'gene/protein') &
        (edges_df['y_type'] == 'disease')
    ]
    print(f"\nEdges gene/protein → disease: {len(gene_disease_edges):,}")

    if len(gene_disease_edges) > 0:
        print("\nRelations for gene/protein → disease:")
        print(gene_disease_edges['relation'].value_counts())

        print("\nSample edges:")
        print(gene_disease_edges.head(10)[['x_name', 'relation', 'y_name']])

# Check all relation types
print("\n\nAll unique relation types in dataset:")
print(edges_df['relation'].value_counts())
