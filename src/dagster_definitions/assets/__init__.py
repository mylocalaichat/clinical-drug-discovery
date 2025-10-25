"""
Dagster assets for Clinical-Enriched Drug Discovery.
"""

from .clinical_extraction import (
    clinical_drug_disease_pairs,
    clinical_extraction_stats,
    mtsamples_raw,
)
from .data_loading import (
    disease_features_loaded,
    drug_features_loaded,
    neo4j_database_ready,
    primekg_download_status,
    primekg_edges_loaded,
    primekg_nodes_loaded,
)
from .drug_discovery import drug_discovery_results
from .graph_enrichment import clinical_enrichment_stats, clinical_validation_stats

__all__ = [
    # Data Loading
    "primekg_download_status",
    "neo4j_database_ready",
    "primekg_nodes_loaded",
    "primekg_edges_loaded",
    "drug_features_loaded",
    "disease_features_loaded",
    # Clinical Extraction
    "mtsamples_raw",
    "clinical_drug_disease_pairs",
    "clinical_extraction_stats",
    # Graph Enrichment
    "clinical_enrichment_stats",
    "clinical_validation_stats",
    # Drug Discovery
    "drug_discovery_results",
]
