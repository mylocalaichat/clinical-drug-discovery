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
    memgraph_database_ready,
    primekg_download_status,
    primekg_edges_loaded,
    primekg_nodes_loaded,
)
from .drug_discovery import drug_discovery_results
from .embedding_drug_discovery import (
    drug_similarity_matrix,
    embedding_enhanced_drug_discovery,
)
from .embeddings import (
    embedding_dataframe,
    embedding_validation_stats,
    flattened_embeddings,
    knowledge_graph,
    node2vec_embeddings,
    saved_embeddings,
)
from .graph_enrichment import clinical_pairs_loaded, clinical_validation_stats

# TODO: Enable link prediction assets after installing OpenMP (brew install libomp)
# from .link_prediction import (
#     link_prediction_all_drug_disease_pairs,
#     link_prediction_diseases,
#     link_prediction_drugs,
#     link_prediction_ensemble_models,
#     link_prediction_known_pairs,
#     link_prediction_node2vec_embeddings,
#     link_prediction_predictions,
#     link_prediction_training_data,
#     link_prediction_training_data_with_embeddings,
#     link_prediction_unknown_pairs,
# )

__all__ = [
    # Data Loading
    "primekg_download_status",
    "memgraph_database_ready",
    "primekg_nodes_loaded",
    "primekg_edges_loaded",
    "drug_features_loaded",
    "disease_features_loaded",
    # Clinical Extraction
    "mtsamples_raw",
    "clinical_drug_disease_pairs",
    "clinical_extraction_stats",
    # Graph Enrichment
    "clinical_pairs_loaded",
    "clinical_validation_stats",
    # Drug Discovery
    "drug_discovery_results",
    "drug_similarity_matrix",
    "embedding_enhanced_drug_discovery",
    # Graph Embeddings
    "knowledge_graph",
    "node2vec_embeddings",
    "saved_embeddings",
    "embedding_dataframe",
    "flattened_embeddings",
    "embedding_validation_stats",
    # Link Prediction (disabled until OpenMP installed)
    # "link_prediction_drugs",
    # "link_prediction_diseases",
    # "link_prediction_known_pairs",
    # "link_prediction_node2vec_embeddings",
    # "link_prediction_training_data",
    # "link_prediction_training_data_with_embeddings",
    # "link_prediction_ensemble_models",
    # "link_prediction_all_drug_disease_pairs",
    # "link_prediction_unknown_pairs",
    # "link_prediction_predictions",
]
