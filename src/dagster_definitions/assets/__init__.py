"""
Dagster assets for Clinical-Enriched Drug Discovery.
"""

# DISABLED: clinical_extraction asset group
# from .clinical_extraction import (
#     clinical_drug_disease_pairs,
#     clinical_extraction_stats,
#     mtsamples_raw,
# )
from .data_loading import (
    disease_features_loaded,
    drug_features_loaded,
    memgraph_database_ready,
    primekg_download_status,
    primekg_edges_loaded,
    primekg_nodes_loaded,
)
# DISABLED: drug_discovery assets (depend on clinical_extraction)
# from .drug_discovery import drug_discovery_results
# from .embedding_drug_discovery import (
#     drug_similarity_matrix,
#     embedding_enhanced_drug_discovery,
# )
from .embeddings import (
    embedding_visualizations,
    flattened_embeddings,
    gnn_embeddings,
)
# DISABLED: graph_enrichment asset group (depends on clinical_extraction)
# from .graph_enrichment import clinical_pairs_loaded, clinical_validation_stats
from .xgboost_drug_discovery import (
    xgboost_all_drug_disease_pairs,
    xgboost_feature_vectors,
    xgboost_known_drug_disease_pairs,
    xgboost_model_evaluation,
    xgboost_node_embeddings,
    xgboost_predictions,
    xgboost_ranked_results,
    xgboost_trained_model,
    xgboost_training_data,
)

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
    # Clinical Extraction (DISABLED)
    # "mtsamples_raw",
    # "clinical_drug_disease_pairs",
    # "clinical_extraction_stats",
    # Graph Enrichment (DISABLED - depends on clinical_extraction)
    # "clinical_pairs_loaded",
    # "clinical_validation_stats",
    # Drug Discovery (DISABLED - depends on clinical_extraction)
    # "drug_discovery_results",
    # "drug_similarity_matrix",
    # "embedding_enhanced_drug_discovery",
    # GNN Embeddings (replaces Node2Vec)
    "gnn_embeddings",
    "flattened_embeddings",
    "embedding_visualizations",
    # XGBoost Drug Discovery
    "xgboost_known_drug_disease_pairs",
    "xgboost_node_embeddings",
    "xgboost_training_data",
    "xgboost_feature_vectors",
    "xgboost_trained_model",
    "xgboost_model_evaluation",
    "xgboost_all_drug_disease_pairs",
    "xgboost_predictions",
    "xgboost_ranked_results",
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
