"""
Off-Label Drug Discovery: Graph Loading and Pruning Module

This module implements:
- Step 1: Selective edge type loading from CSV files or Memgraph
- Step 2: Graph pruning (drug-drug and protein-protein edges)
- Step 3: Data preparation for link prediction task
"""

import logging
from typing import Dict, Tuple, Optional, Set
from pathlib import Path
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import torch
from torch_geometric.data import HeteroData
from sklearn.model_selection import train_test_split
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Edge type configuration
# We INCLUDE all edge types initially, including the target edges we want to predict.
# Later, during data preparation, we'll:
#   1. Extract indication/off-label/contraindication as TRAINING LABELS
#   2. Build the graph structure using ONLY the non-target edges (to avoid data leakage)
INCLUSION_EDGE_TYPES = {
    "bioprocess_bioprocess",
    "bioprocess_protein",
    "contraindication",        # Included for extraction as negative labels
    "disease_disease",
    "disease_phenotype_positive",
    "disease_protein",
    "drug_drug",
    "drug_effect",
    "drug_protein",
    "indication",              # Included for extraction as positive labels
    "molfunc_protein",
    "off-label use",           # Included for extraction as positive labels
    "pathway_pathway",
    "pathway_protein",
    "phenotype_protein",
    "protein_protein",
}

# Edge types to use for GRAPH STRUCTURE (excludes target prediction edges)
# These are the edges the GNN will use for message passing
GRAPH_STRUCTURE_EDGE_TYPES = {
    "bioprocess_bioprocess",
    "bioprocess_protein",
    # "contraindication",      # EXCLUDED from graph - used only as labels
    "disease_disease",
    "disease_phenotype_positive",
    "disease_protein",
    "drug_drug",
    "drug_effect",
    "drug_protein",
    # "indication",            # EXCLUDED from graph - used only as labels
    "molfunc_protein",
    # "off-label use",         # EXCLUDED from graph - used only as labels
    "pathway_pathway",
    "pathway_protein",
    "phenotype_protein",
    "protein_protein",
}

REJECTION_EDGE_TYPES = {
    "anatomy_protein_present",
    "anatomy_protein_absent",
    "cellcomp_protein",
    "cellcomp_cellcomp",
    "phenotype_phenotype",
    "molfunc_molfunc",
    "anatomy_anatomy",
    "exposure_bioprocess",
    "exposure_cellcomp",
    "exposure_disease",
    "exposure_exposure",
    "exposure_molfunc",
    "exposure_protein",
    "disease_phenotype_negative",
}


class MemgraphGraphLoader:
    """Loads graph data from Memgraph with selective edge type filtering."""

    def __init__(self, uri: str, user: str = "", password: str = "", database: str = "memgraph"):
        """Initialize connection to Memgraph."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        logger.info(f"Connected to Memgraph at {uri}")

    def close(self):
        """Close the database connection."""
        self.driver.close()

    def load_edges_with_filter(self) -> pd.DataFrame:
        """
        Load edges from Memgraph, filtering by inclusion list.

        Returns:
            DataFrame with columns: source_id, source_type, relation, target_id, target_type
        """
        logger.info("Loading edges from Memgraph with inclusion filter...")

        # Build Cypher query to filter by relation type
        relation_filter = ", ".join([f"'{rel}'" for rel in INCLUSION_EDGE_TYPES])

        query = f"""
        MATCH (n)-[r:RELATES]->(m)
        WHERE r.relation IN [{relation_filter}]
        RETURN
            n.node_id AS source_id,
            n.node_type AS source_type,
            r.relation AS relation,
            m.node_id AS target_id,
            m.node_type AS target_type
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            records = list(result)

        if not records:
            logger.warning("No edges found in Memgraph with the specified edge types!")
            logger.warning("Make sure PrimeKG data is loaded in Memgraph.")
            # Return empty DataFrame with expected columns
            df = pd.DataFrame(columns=['source_id', 'source_type', 'relation', 'target_id', 'target_type'])
        else:
            df = pd.DataFrame([dict(record) for record in records])
            logger.info(f"Loaded {len(df):,} edges with {len(df['relation'].unique())} relation types")

        return df

    def get_node_metadata(self) -> pd.DataFrame:
        """
        Load all node metadata from Memgraph.

        Returns:
            DataFrame with columns: node_id, node_type, node_name
        """
        logger.info("Loading node metadata from Memgraph...")

        query = """
        MATCH (n)
        RETURN
            n.node_id AS node_id,
            n.node_type AS node_type,
            n.node_name AS node_name
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            records = list(result)

        if not records:
            logger.warning("No nodes found in Memgraph!")
            logger.warning("Make sure PrimeKG data is loaded in Memgraph.")
            # Return empty DataFrame with expected columns
            df = pd.DataFrame(columns=['node_id', 'node_type', 'node_name'])
        else:
            df = pd.DataFrame([dict(record) for record in records])
            logger.info(f"Loaded metadata for {len(df):,} nodes of {len(df['node_type'].unique())} types")

        return df

    def compute_drug_similarity(self, drug_id: str, neighbor_id: str) -> Tuple[int, int]:
        """
        Compute similarity between two drugs based on shared indications and proteins.

        Returns:
            (shared_indications_count, shared_proteins_count)
        """
        query = """
        MATCH (d1 {node_id: $drug1}), (d2 {node_id: $drug2})

        // Count shared indications
        OPTIONAL MATCH (d1)-[r1:RELATES]->(disease)<-[r2:RELATES]-(d2)
        WHERE r1.relation = 'indication' AND r2.relation = 'indication'
        WITH d1, d2, count(DISTINCT disease) AS shared_indications

        // Count shared proteins
        OPTIONAL MATCH (d1)-[r3:RELATES]->(protein)<-[r4:RELATES]-(d2)
        WHERE r3.relation = 'drug_protein' AND r4.relation = 'drug_protein'
        WITH shared_indications, count(DISTINCT protein) AS shared_proteins

        RETURN shared_indications, shared_proteins
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, drug1=drug_id, drug2=neighbor_id)
            record = result.single()

            if record:
                return record["shared_indications"], record["shared_proteins"]
            return 0, 0

    def compute_protein_similarity(self, protein_id: str, neighbor_id: str) -> Tuple[int, int, int]:
        """
        Compute similarity between two proteins based on shared biological processes,
        molecular functions, and diseases.

        Returns:
            (shared_bioprocesses, shared_molfuncs, shared_diseases)
        """
        query = """
        MATCH (p1 {node_id: $protein1}), (p2 {node_id: $protein2})

        // Count shared biological processes
        OPTIONAL MATCH (p1)<-[r1:RELATES]-(bp)-[r2:RELATES]->(p2)
        WHERE r1.relation = 'bioprocess_protein' AND r2.relation = 'bioprocess_protein'
        WITH p1, p2, count(DISTINCT bp) AS shared_bp

        // Count shared molecular functions
        OPTIONAL MATCH (p1)<-[r3:RELATES]-(mf)-[r4:RELATES]->(p2)
        WHERE r3.relation = 'molfunc_protein' AND r4.relation = 'molfunc_protein'
        WITH p1, p2, shared_bp, count(DISTINCT mf) AS shared_mf

        // Count shared diseases
        OPTIONAL MATCH (p1)<-[r5:RELATES]-(disease)-[r6:RELATES]->(p2)
        WHERE r5.relation = 'disease_protein' AND r6.relation = 'disease_protein'
        WITH shared_bp, shared_mf, count(DISTINCT disease) AS shared_diseases

        RETURN shared_bp, shared_mf, shared_diseases
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, protein1=protein_id, protein2=neighbor_id)
            record = result.single()

            if record:
                return record["shared_bp"], record["shared_mf"], record["shared_diseases"]
            return 0, 0, 0


class CSVGraphLoader:
    """Loads graph data from CSV files with selective edge type filtering."""

    def __init__(self, data_dir: str = "data/01_raw/primekg", edges_df: Optional[pd.DataFrame] = None):
        """
        Initialize CSV-based graph loader.

        Args:
            data_dir: Directory containing PrimeKG CSV files (kg.csv)
            edges_df: Pre-loaded edges DataFrame (optional, if provided will skip CSV loading)
        """
        self.data_dir = Path(data_dir) if data_dir else None

        if edges_df is not None:
            # Use pre-loaded DataFrame
            logger.info("Initializing CSVGraphLoader with pre-loaded edges DataFrame")
            self.all_edges_df = edges_df
            logger.info(f"Using {len(self.all_edges_df):,} pre-loaded edges")
        else:
            # Load from CSV file
            if self.data_dir is None:
                raise ValueError("Either data_dir or edges_df must be provided")

            self.edges_file = self.data_dir / "kg.csv"
            if not self.edges_file.exists():
                raise FileNotFoundError(f"Edge file not found: {self.edges_file}")

            logger.info(f"Initializing CSVGraphLoader from {self.data_dir}")

            # Load all edges into memory for fast similarity computations
            logger.info("Loading edges from CSV...")
            self.all_edges_df = pd.read_csv(self.edges_file)
            logger.info(f"Loaded {len(self.all_edges_df):,} total edges")

        # Precompute relationship lookups for similarity calculations
        self._precompute_relationships()

    def close(self):
        """Close method for API compatibility with MemgraphGraphLoader."""
        pass

    def _precompute_relationships(self):
        """Precompute relationship lookups for fast similarity calculations using vectorized batch operations."""
        logger.info("Precomputing relationship lookups for similarity calculations...")

        # Detect column naming scheme (original CSV uses x_id/y_id, filtered uses source_id/target_id)
        if 'x_id' in self.all_edges_df.columns:
            source_col, target_col = 'x_id', 'y_id'
        else:
            source_col, target_col = 'source_id', 'target_id'

        # Drug -> Indications (batch groupby operation)
        indication_edges = self.all_edges_df[self.all_edges_df['relation'] == 'indication']
        self.drug_indications = defaultdict(set,
            indication_edges.groupby(source_col)[target_col].apply(set).to_dict()
        )

        # Drug -> Proteins (batch groupby operation)
        drug_protein_edges = self.all_edges_df[self.all_edges_df['relation'] == 'drug_protein']
        self.drug_proteins = defaultdict(set,
            drug_protein_edges.groupby(source_col)[target_col].apply(set).to_dict()
        )

        # Protein -> Biological Processes (batch groupby operation)
        bp_edges = self.all_edges_df[self.all_edges_df['relation'] == 'bioprocess_protein']
        self.protein_bioprocesses = defaultdict(set,
            bp_edges.groupby(target_col)[source_col].apply(set).to_dict()
        )

        # Protein -> Molecular Functions (batch groupby operation)
        mf_edges = self.all_edges_df[self.all_edges_df['relation'] == 'molfunc_protein']
        self.protein_molfuncs = defaultdict(set,
            mf_edges.groupby(target_col)[source_col].apply(set).to_dict()
        )

        # Protein -> Diseases (batch groupby operation)
        disease_protein_edges = self.all_edges_df[self.all_edges_df['relation'] == 'disease_protein']
        self.protein_diseases = defaultdict(set,
            disease_protein_edges.groupby(target_col)[source_col].apply(set).to_dict()
        )

        logger.info("Relationship lookups precomputed successfully")

    def load_edges_with_filter(self) -> pd.DataFrame:
        """
        Load edges from CSV, filtering by inclusion list.

        Returns:
            DataFrame with columns: source_id, source_type, relation, target_id, target_type
        """
        logger.info("Loading edges with inclusion filter...")

        # Filter by relation type
        filtered_df = self.all_edges_df[self.all_edges_df['relation'].isin(INCLUSION_EDGE_TYPES)].copy()

        # Rename columns to match expected format (including name columns if they exist)
        rename_dict = {
            'x_id': 'source_id',
            'x_type': 'source_type',
            'y_id': 'target_id',
            'y_type': 'target_type'
        }
        if 'x_name' in filtered_df.columns:
            rename_dict['x_name'] = 'source_name'
        if 'y_name' in filtered_df.columns:
            rename_dict['y_name'] = 'target_name'

        filtered_df = filtered_df.rename(columns=rename_dict)

        if len(filtered_df) == 0:
            logger.warning("No edges found with the specified edge types!")
            logger.warning("Make sure PrimeKG data is downloaded.")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['source_id', 'source_type', 'relation', 'target_id', 'target_type'])

        logger.info(f"Loaded {len(filtered_df):,} edges with {len(filtered_df['relation'].unique())} relation types")
        return filtered_df

    def get_node_metadata(self) -> pd.DataFrame:
        """
        Extract all node metadata from edges.

        Returns:
            DataFrame with columns: node_id, node_type, node_name
        """
        logger.info("Extracting node metadata from edges...")

        # Detect column naming scheme (original CSV uses x_id/y_id, filtered uses source_id/target_id)
        if 'x_id' in self.all_edges_df.columns:
            source_id_col, source_type_col, source_name_col = 'x_id', 'x_type', 'x_name'
            target_id_col, target_type_col, target_name_col = 'y_id', 'y_type', 'y_name'
        else:
            source_id_col, source_type_col, source_name_col = 'source_id', 'source_type', 'source_id'
            target_id_col, target_type_col, target_name_col = 'target_id', 'target_type', 'target_id'

            # Check if name columns exist (they may not in filtered DataFrames)
            if 'source_name' in self.all_edges_df.columns:
                source_name_col = 'source_name'
            if 'target_name' in self.all_edges_df.columns:
                target_name_col = 'target_name'

        # Extract unique source nodes
        source_cols = [source_id_col, source_type_col]
        if source_name_col in self.all_edges_df.columns:
            source_cols.append(source_name_col)

        source_nodes = self.all_edges_df[source_cols].copy()
        source_nodes.columns = ['node_id', 'node_type'] if len(source_cols) == 2 else ['node_id', 'node_type', 'node_name']

        # Extract unique target nodes
        target_cols = [target_id_col, target_type_col]
        if target_name_col in self.all_edges_df.columns:
            target_cols.append(target_name_col)

        target_nodes = self.all_edges_df[target_cols].copy()
        target_nodes.columns = ['node_id', 'node_type'] if len(target_cols) == 2 else ['node_id', 'node_type', 'node_name']

        # Combine and deduplicate
        all_nodes = pd.concat([source_nodes, target_nodes], ignore_index=True)
        nodes_df = all_nodes.drop_duplicates(subset=['node_id']).reset_index(drop=True)

        # Add node_name column if it doesn't exist (use node_id as fallback)
        if 'node_name' not in nodes_df.columns:
            nodes_df['node_name'] = nodes_df['node_id']

        logger.info(f"Extracted metadata for {len(nodes_df):,} nodes of {len(nodes_df['node_type'].unique())} types")
        return nodes_df

    def compute_drug_similarity(self, drug_id: str, neighbor_id: str) -> Tuple[int, int]:
        """
        Compute similarity between two drugs based on shared indications and proteins.

        Returns:
            (shared_indications_count, shared_proteins_count)
        """
        drug1_indications = self.drug_indications.get(drug_id, set())
        drug2_indications = self.drug_indications.get(neighbor_id, set())
        shared_indications = len(drug1_indications & drug2_indications)

        drug1_proteins = self.drug_proteins.get(drug_id, set())
        drug2_proteins = self.drug_proteins.get(neighbor_id, set())
        shared_proteins = len(drug1_proteins & drug2_proteins)

        return shared_indications, shared_proteins

    def compute_protein_similarity(self, protein_id: str, neighbor_id: str) -> Tuple[int, int, int]:
        """
        Compute similarity between two proteins based on shared biological processes,
        molecular functions, and diseases.

        Returns:
            (shared_bioprocesses, shared_molfuncs, shared_diseases)
        """
        protein1_bp = self.protein_bioprocesses.get(protein_id, set())
        protein2_bp = self.protein_bioprocesses.get(neighbor_id, set())
        shared_bp = len(protein1_bp & protein2_bp)

        protein1_mf = self.protein_molfuncs.get(protein_id, set())
        protein2_mf = self.protein_molfuncs.get(neighbor_id, set())
        shared_mf = len(protein1_mf & protein2_mf)

        protein1_diseases = self.protein_diseases.get(protein_id, set())
        protein2_diseases = self.protein_diseases.get(neighbor_id, set())
        shared_diseases = len(protein1_diseases & protein2_diseases)

        return shared_bp, shared_mf, shared_diseases


class GraphPruner:
    """Prunes drug-drug and protein-protein edges based on similarity scoring."""

    def __init__(self, loader):
        """
        Initialize GraphPruner with a graph loader.

        Args:
            loader: Graph loader instance (MemgraphGraphLoader or CSVGraphLoader)
        """
        self.loader = loader

    def prune_drug_drug_edges(self, edges_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        """
        Prune drug-drug edges, keeping only top K most similar neighbors per drug.

        Similarity score = (5 × shared_indications) + (3 × shared_proteins)

        Args:
            edges_df: DataFrame with all edges
            top_k: Number of top neighbors to keep per drug

        Returns:
            DataFrame with pruned edges
        """
        logger.info(f"Pruning drug-drug edges (keeping top {top_k} per drug)...")

        # Extract drug-drug edges
        drug_drug_mask = edges_df['relation'] == 'drug_drug'
        drug_drug_edges = edges_df[drug_drug_mask].copy()
        other_edges = edges_df[~drug_drug_mask].copy()

        initial_count = len(drug_drug_edges)
        logger.info(f"Initial drug-drug edges: {initial_count:,}")

        if initial_count == 0:
            return edges_df

        # Vectorized similarity computation
        logger.info("Computing similarities (optimized vectorized)...")

        # Get drug indications and proteins as sets for each edge
        drug_indications = self.loader.drug_indications
        drug_proteins = self.loader.drug_proteins

        # Compute similarity scores using list comprehension (much faster than apply)
        similarities = [
            (5 * len(drug_indications.get(src, set()) & drug_indications.get(tgt, set()))) +
            (3 * len(drug_proteins.get(src, set()) & drug_proteins.get(tgt, set())))
            for src, tgt in zip(drug_drug_edges['source_id'], drug_drug_edges['target_id'])
        ]
        drug_drug_edges['similarity_score'] = similarities

        logger.info("Selecting top K neighbors per drug...")
        # Keep top K neighbors per drug using groupby
        pruned_drug_drug = (drug_drug_edges
                           .sort_values('similarity_score', ascending=False)
                           .groupby('source_id', group_keys=False)
                           .head(top_k)
                           .reset_index(drop=True))

        # Combine pruned edges with other edge types
        final_edges = pd.concat([other_edges, pruned_drug_drug], ignore_index=True)

        logger.info(f"Pruned drug-drug edges: {initial_count:,} → {len(pruned_drug_drug):,} "
                   f"({len(pruned_drug_drug)/initial_count*100:.1f}% retained)")

        return final_edges

    def prune_protein_protein_edges(self, edges_df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
        """
        Prune protein-protein edges, keeping only top K most similar neighbors per protein.

        Similarity score = (5 × shared_bioprocesses) + (3 × shared_molfuncs) + (4 × shared_diseases)

        Args:
            edges_df: DataFrame with all edges
            top_k: Number of top neighbors to keep per protein

        Returns:
            DataFrame with pruned edges
        """
        logger.info(f"Pruning protein-protein edges (keeping top {top_k} per protein)...")

        # Extract protein-protein edges
        prot_prot_mask = edges_df['relation'] == 'protein_protein'
        prot_prot_edges = edges_df[prot_prot_mask].copy()
        other_edges = edges_df[~prot_prot_mask].copy()

        initial_count = len(prot_prot_edges)
        logger.info(f"Initial protein-protein edges: {initial_count:,}")

        if initial_count == 0:
            return edges_df

        # Vectorized similarity computation
        logger.info("Computing similarities (optimized vectorized)...")

        # Get protein relationships as sets
        protein_bioprocesses = self.loader.protein_bioprocesses
        protein_molfuncs = self.loader.protein_molfuncs
        protein_diseases = self.loader.protein_diseases

        # Compute similarity scores using list comprehension (much faster than apply)
        similarities = [
            (5 * len(protein_bioprocesses.get(src, set()) & protein_bioprocesses.get(tgt, set()))) +
            (3 * len(protein_molfuncs.get(src, set()) & protein_molfuncs.get(tgt, set()))) +
            (4 * len(protein_diseases.get(src, set()) & protein_diseases.get(tgt, set())))
            for src, tgt in zip(prot_prot_edges['source_id'], prot_prot_edges['target_id'])
        ]
        prot_prot_edges['similarity_score'] = similarities

        logger.info("Selecting top K neighbors per protein...")
        # Keep top K neighbors per protein using groupby
        pruned_prot_prot = (prot_prot_edges
                           .sort_values('similarity_score', ascending=False)
                           .groupby('source_id', group_keys=False)
                           .head(top_k)
                           .reset_index(drop=True))

        # Combine pruned edges with other edge types
        final_edges = pd.concat([other_edges, pruned_prot_prot], ignore_index=True)

        logger.info(f"Pruned protein-protein edges: {initial_count:,} → {len(pruned_prot_prot):,} "
                   f"({len(pruned_prot_prot)/initial_count*100:.1f}% retained)")

        return final_edges


class OffLabelDataPreparator:
    """Prepares training/test data for off-label drug discovery link prediction."""

    def __init__(self, edges_df: pd.DataFrame):
        self.edges_df = edges_df

    def prepare_link_prediction_data(
        self,
        test_size: float = 0.2,
        num_contraindication_samples: int = 4800,
        num_random_negatives: int = 1600,
        random_seed: int = 42
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare train/test splits for off-label drug discovery.

        Positive examples: indication AND off-label use edges (therapeutic relationships)
        Negative examples: contraindications + random drug-disease pairs (automatically balanced to match positives)

        Args:
            test_size: Fraction of therapeutic edges to use for testing
            num_contraindication_samples: MINIMUM number of contraindications to sample (auto-increased to balance positives)
            num_random_negatives: MINIMUM number of random negatives to sample (auto-increased to balance positives)
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary with keys: 'train_edges', 'test_edges', 'train_positives',
                                  'train_negatives', 'test_positives', 'test_negatives'

        Note:
            The function will automatically calculate the needed number of negatives to balance
            the positive examples. If the provided num_contraindication_samples or num_random_negatives
            are lower than needed, they will be increased to 75% and 25% of total positives respectively.
        """
        logger.info("Preparing link prediction data...")

        # Extract therapeutic edges (positives): both indication AND off-label use
        therapeutic_edges = self.edges_df[
            self.edges_df['relation'].isin(['indication', 'off-label use'])
        ].copy()

        indication_count = len(self.edges_df[self.edges_df['relation'] == 'indication'])
        offlabel_count = len(self.edges_df[self.edges_df['relation'] == 'off-label use'])

        logger.info(f"Found {len(therapeutic_edges):,} therapeutic edges:")
        logger.info(f"  - {indication_count:,} indication edges")
        logger.info(f"  - {offlabel_count:,} off-label use edges")

        # Split therapeutic edges into train/test
        train_positives, test_positives = train_test_split(
            therapeutic_edges,
            test_size=test_size,
            random_state=random_seed
        )

        logger.info(f"Split: {len(train_positives):,} train, {len(test_positives):,} test")

        # Create training graph: exclude ALL target edges (indication, off-label, contraindication)
        # AND exclude test positive edges to prevent leakage
        # This ensures the GNN learns from STRUCTURE, not from seeing the answer
        target_edge_types = {'indication', 'off-label use', 'contraindication'}

        train_graph_edges = self.edges_df[
            ~self.edges_df['relation'].isin(target_edge_types)
        ].copy()

        logger.info(f"Training graph edges (structure only, no target edges): {len(train_graph_edges):,}")
        logger.info(f"  Excluded {len(self.edges_df) - len(train_graph_edges):,} target edges "
                   f"(indication/off-label/contraindication)")

        # Calculate how many negatives we need to balance the positives
        num_positives = len(therapeutic_edges)
        logger.info(f"Total positive examples (therapeutic edges): {num_positives:,}")

        # Balance negatives to match positives (75% contraindications, 25% random)
        # This ensures we have enough negatives to balance the dataset
        target_contraindications = int(num_positives * 0.75)
        target_random = int(num_positives * 0.25)

        # Sample contraindications as negatives (from ORIGINAL edges, not training graph)
        # Note: We get them from self.edges_df because we excluded them from train_graph_edges
        # to prevent data leakage in the GNN structure
        contraindications = self.edges_df[self.edges_df['relation'] == 'contraindication'].copy()
        logger.info(f"Total contraindications available: {len(contraindications):,}")

        # Use the larger of: default parameter or calculated target
        actual_contraindications = max(num_contraindication_samples, target_contraindications)
        actual_contraindications = min(actual_contraindications, len(contraindications))

        sampled_contraindications = contraindications.sample(
            n=actual_contraindications,
            random_state=random_seed
        )
        logger.info(f"Sampled {len(sampled_contraindications):,} contraindications as negatives")

        # Generate random negatives (drug-disease pairs with no edges)
        # Use train_graph_edges to ensure all sampled drugs/diseases will be in node_mapping
        actual_random = max(num_random_negatives, target_random)

        random_negatives = self._generate_random_negatives(
            num_samples=actual_random,
            random_seed=random_seed,
            edges_df=train_graph_edges
        )
        logger.info(f"Generated {len(random_negatives):,} random negative pairs")

        # Combine negatives
        all_negatives = pd.concat([sampled_contraindications, random_negatives], ignore_index=True)
        logger.info(f"Total negative examples: {len(all_negatives):,} (target was ~{num_positives:,} to balance positives)")

        # Split negatives into train/test (same ratio)
        train_negatives, test_negatives = train_test_split(
            all_negatives,
            test_size=test_size,
            random_state=random_seed
        )

        # Prepare final datasets
        result = {
            'train_edges': train_graph_edges,  # Graph for training (excludes test positives)
            'test_edges': self.edges_df,  # Full graph for testing
            'train_positives': train_positives,
            'train_negatives': train_negatives,
            'test_positives': test_positives,
            'test_negatives': test_negatives,
        }

        logger.info("Final dataset sizes:")
        logger.info(f"  Train: {len(train_positives):,} pos + {len(train_negatives):,} neg = "
                   f"{len(train_positives) + len(train_negatives):,} total")
        logger.info(f"  Test: {len(test_positives):,} pos + {len(test_negatives):,} neg = "
                   f"{len(test_positives) + len(test_negatives):,} total")

        return result

    def _generate_random_negatives(self, num_samples: int, random_seed: int, edges_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate random drug-disease pairs with no existing edges.

        Args:
            num_samples: Number of random negative pairs to generate
            random_seed: Random seed for reproducibility
            edges_df: Optional DataFrame to sample from. If None, uses self.edges_df
        """
        np.random.seed(random_seed)

        # Use provided edges_df or fall back to self.edges_df
        source_edges = edges_df if edges_df is not None else self.edges_df

        # Get all drugs and diseases
        all_drugs = source_edges[source_edges['source_type'] == 'drug']['source_id'].unique()
        all_diseases = source_edges[source_edges['target_type'] == 'disease']['target_id'].unique()

        # Get existing drug-disease pairs (indication, off-label, contraindication)
        existing_pairs = set()
        for rel in ['indication', 'off-label use', 'contraindication']:
            edges = source_edges[source_edges['relation'] == rel]
            pairs = set(zip(edges['source_id'], edges['target_id']))
            existing_pairs.update(pairs)

        # Sample random pairs that don't exist
        random_pairs = []
        attempts = 0
        max_attempts = num_samples * 100

        while len(random_pairs) < num_samples and attempts < max_attempts:
            drug = np.random.choice(all_drugs)
            disease = np.random.choice(all_diseases)

            if (drug, disease) not in existing_pairs:
                random_pairs.append({
                    'source_id': drug,
                    'source_type': 'drug',
                    'relation': 'random_negative',
                    'target_id': disease,
                    'target_type': 'disease'
                })
                existing_pairs.add((drug, disease))

            attempts += 1

        return pd.DataFrame(random_pairs)


def create_heterogeneous_graph(
    edges_df: pd.DataFrame,
    node_metadata_df: pd.DataFrame
) -> Tuple[HeteroData, Dict[str, Dict[str, int]]]:
    """
    Convert edge DataFrame to PyTorch Geometric HeteroData object.

    Args:
        edges_df: DataFrame with edges
        node_metadata_df: DataFrame with node metadata

    Returns:
        (HeteroData object, node_mapping dictionary)
    """
    logger.info("Creating heterogeneous graph...")

    # Create node mappings (node_id -> integer index per type)
    node_types = node_metadata_df['node_type'].unique()
    node_mapping = {}

    for node_type in node_types:
        type_nodes = node_metadata_df[node_metadata_df['node_type'] == node_type]
        node_mapping[node_type] = {
            node_id: idx for idx, node_id in enumerate(type_nodes['node_id'])
        }

    # Create HeteroData object
    data = HeteroData()

    # Add node counts
    for node_type, mapping in node_mapping.items():
        data[node_type].num_nodes = len(mapping)
        logger.info(f"  {node_type}: {len(mapping):,} nodes")

    # Add edges by type
    edge_types = edges_df['relation'].unique()
    edge_counts = {}

    for edge_type in edge_types:
        edges = edges_df[edges_df['relation'] == edge_type]

        if len(edges) == 0:
            continue

        # Get source and target types
        source_type = edges['source_type'].iloc[0]
        target_type = edges['target_type'].iloc[0]

        # Map node IDs to indices using vectorized operations
        # Create series with node mappings for batch lookup
        source_mapping = pd.Series(node_mapping[source_type])
        target_mapping = pd.Series(node_mapping[target_type])

        # Batch map all source and target IDs
        edges_mapped = edges.copy()
        edges_mapped['source_idx'] = edges_mapped['source_id'].map(source_mapping)
        edges_mapped['target_idx'] = edges_mapped['target_id'].map(target_mapping)

        # Filter out edges where mapping failed (NaN values)
        valid_edges = edges_mapped.dropna(subset=['source_idx', 'target_idx'])

        if len(valid_edges) == 0:
            continue

        # Extract indices as lists
        source_indices = valid_edges['source_idx'].astype(int).tolist()
        target_indices = valid_edges['target_idx'].astype(int).tolist()

        # Create edge index tensor
        edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)

        # Add to HeteroData
        data[source_type, edge_type, target_type].edge_index = edge_index

        # Also add reverse edges
        data[target_type, f"{edge_type}_reverse", source_type].edge_index = torch.tensor(
            [target_indices, source_indices], dtype=torch.long
        )

        edge_counts[edge_type] = len(source_indices)
        logger.info(f"  {source_type} -[{edge_type}]-> {target_type}: {len(source_indices):,} edges")

    logger.info(f"Created heterogeneous graph with {len(node_types)} node types and "
               f"{len(edge_types)} edge types")

    return data, node_mapping


if __name__ == "__main__":
    # Example usage
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Initialize loader
    loader = MemgraphGraphLoader(
        uri=os.getenv("MEMGRAPH_URI", "bolt://localhost:7687"),
        user=os.getenv("MEMGRAPH_USER", ""),
        password=os.getenv("MEMGRAPH_PASSWORD", ""),
        database=os.getenv("MEMGRAPH_DATABASE", "memgraph")
    )

    try:
        # Step 1: Load edges with filtering
        edges_df = loader.load_edges_with_filter()

        # Load node metadata
        node_metadata_df = loader.get_node_metadata()

        # Step 2: Prune graph
        pruner = GraphPruner(loader)
        pruned_edges_df = pruner.prune_drug_drug_edges(edges_df, top_k=10)
        pruned_edges_df = pruner.prune_protein_protein_edges(pruned_edges_df, top_k=20)

        # Step 3: Prepare training data
        preparator = OffLabelDataPreparator(pruned_edges_df)
        datasets = preparator.prepare_link_prediction_data()

        # Step 4: Create heterogeneous graph
        train_graph, node_mapping = create_heterogeneous_graph(
            datasets['train_edges'],
            node_metadata_df
        )

        logger.info("Data loading and preparation complete!")

    finally:
        loader.close()
