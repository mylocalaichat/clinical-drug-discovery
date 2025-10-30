"""
Dagster assets for XGBoost-based drug discovery pipeline.

This module implements the complete drug-disease prediction pipeline:
1. Load known drug-disease relationships
2. Create training data (positive/negative samples)
3. Feature engineering (flatten embeddings)
4. Train XGBoost classifier
5. Evaluate model performance
6. Predict all drug-disease pairs
7. Rank and output results
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from dagster import AssetExecutionContext, asset
from neo4j import GraphDatabase
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score, auc
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBClassifier
import random


# =====================================================================
# ASSET 1: Load Known Drug-Disease Relationships
# =====================================================================

@asset(group_name="xgboost_drug_discovery", compute_kind="database")
def xgboost_known_drug_disease_pairs(
    context: AssetExecutionContext,
) -> pd.DataFrame:
    """Load known drug-disease treatment relationships from Memgraph."""
    context.log.info("Loading known drug-disease relationships...")

    # Get Memgraph connection
    memgraph_uri = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
    memgraph_user = os.getenv("MEMGRAPH_USER", "")
    memgraph_password = os.getenv("MEMGRAPH_PASSWORD", "")

    auth = None
    if memgraph_user or memgraph_password:
        auth = (memgraph_user, memgraph_password)

    driver = GraphDatabase.driver(memgraph_uri, auth=auth)

    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (drug:Node)-[r:RELATES {relation: 'drug_treats_disease'}]->(disease:Node)
                WHERE drug.is_example = true AND disease.is_example = true
                RETURN drug.node_id as drug_id,
                       drug.node_name as drug_name,
                       disease.node_id as disease_id,
                       disease.node_name as disease_name
            """)

            known_pairs = []
            for record in result:
                known_pairs.append({
                    'drug_id': record['drug_id'],
                    'drug_name': record['drug_name'],
                    'disease_id': record['disease_id'],
                    'disease_name': record['disease_name'],
                    'label': 1  # Known treatment
                })
    finally:
        driver.close()

    df = pd.DataFrame(known_pairs)

    context.add_output_metadata({
        "num_known_relationships": len(df),
        "unique_drugs": df['drug_id'].nunique(),
        "unique_diseases": df['disease_id'].nunique(),
        "sample_pairs": df[['drug_name', 'disease_name']].head(5).to_dict('records')
    })

    context.log.info(f"Loaded {len(df)} known treatment relationships")

    return df


# =====================================================================
# ASSET 2: Load Embeddings from Memgraph
# =====================================================================

@asset(group_name="xgboost_drug_discovery", compute_kind="database")
def xgboost_node_embeddings(
    context: AssetExecutionContext,
    flattened_embeddings: pd.DataFrame,  # Depends on embeddings being ready
) -> Dict[str, Dict[str, Any]]:
    """Load all node embeddings from Memgraph (after embeddings pipeline completes)."""
    context.log.info("Loading node embeddings from Memgraph...")
    context.log.info(f"Embeddings pipeline provided {len(flattened_embeddings)} flattened embeddings")

    # Get Memgraph connection
    memgraph_uri = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
    memgraph_user = os.getenv("MEMGRAPH_USER", "")
    memgraph_password = os.getenv("MEMGRAPH_PASSWORD", "")

    auth = None
    if memgraph_user or memgraph_password:
        auth = (memgraph_user, memgraph_password)

    driver = GraphDatabase.driver(memgraph_uri, auth=auth)

    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE EXISTS(n.embedding) AND n.is_example = true
                RETURN n.node_id as id,
                       n.node_name as name,
                       n.node_type as type,
                       n.embedding as embedding
            """)

            embeddings = {}
            for record in result:
                embeddings[record['id']] = {
                    'name': record['name'],
                    'type': record['type'],
                    'embedding': np.array(record['embedding'])
                }
    finally:
        driver.close()

    # Separate by type
    drugs = {k: v for k, v in embeddings.items() if v['type'] == 'drug'}
    diseases = {k: v for k, v in embeddings.items() if v['type'] == 'disease'}

    embedding_dim = len(list(embeddings.values())[0]['embedding']) if embeddings else 0

    context.add_output_metadata({
        "total_nodes_with_embeddings": len(embeddings),
        "num_drugs": len(drugs),
        "num_diseases": len(diseases),
        "embedding_dimensions": embedding_dim
    })

    context.log.info(f"Loaded embeddings for {len(embeddings)} nodes ({len(drugs)} drugs, {len(diseases)} diseases)")

    return embeddings


# =====================================================================
# ASSET 3: Create Training Dataset
# =====================================================================

@asset(group_name="xgboost_drug_discovery", compute_kind="transform")
def xgboost_training_data(
    context: AssetExecutionContext,
    xgboost_known_drug_disease_pairs: pd.DataFrame,
    xgboost_node_embeddings: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Create training dataset with positive, negative, and unknown samples."""
    context.log.info("Creating training dataset...")

    embeddings = xgboost_node_embeddings
    known_df = xgboost_known_drug_disease_pairs

    # Get drug and disease IDs
    drugs = {k: v for k, v in embeddings.items() if v['type'] == 'drug'}
    diseases = {k: v for k, v in embeddings.items() if v['type'] == 'disease'}

    drug_ids = list(drugs.keys())
    disease_ids = list(diseases.keys())

    # Create set of known pairs
    known_set = {(row['drug_id'], row['disease_id']) for _, row in known_df.iterrows()}

    # Positive samples
    positive_samples = known_df.copy()
    positive_samples['type'] = 'positive'
    positive_samples['label'] = 1

    # Negative samples (random non-treatments)
    negative_ratio = 1.0  # Same number as positives
    n_negatives = int(len(positive_samples) * negative_ratio)

    negative_samples = []
    attempts = 0
    max_attempts = n_negatives * 10

    while len(negative_samples) < n_negatives and attempts < max_attempts:
        drug_id = random.choice(drug_ids)
        disease_id = random.choice(disease_ids)

        if (drug_id, disease_id) not in known_set:
            negative_samples.append({
                'drug_id': drug_id,
                'disease_id': disease_id,
                'drug_name': embeddings[drug_id]['name'],
                'disease_name': embeddings[disease_id]['name'],
                'type': 'negative',
                'label': 0
            })
        attempts += 1

    negative_df = pd.DataFrame(negative_samples)

    # Unknown samples (synthetic)
    unknown_ratio = 0.5
    n_unknowns = int(len(positive_samples) * unknown_ratio)

    unknown_samples = []
    attempts = 0
    max_attempts = n_unknowns * 10

    while len(unknown_samples) < n_unknowns and attempts < max_attempts:
        drug_id = random.choice(drug_ids)
        disease_id = random.choice(disease_ids)

        if (drug_id, disease_id) not in known_set:
            unknown_samples.append({
                'drug_id': drug_id,
                'disease_id': disease_id,
                'drug_name': embeddings[drug_id]['name'],
                'disease_name': embeddings[disease_id]['name'],
                'type': 'unknown',
                'label': 2
            })
        attempts += 1

    unknown_df = pd.DataFrame(unknown_samples)

    # Combine all samples
    training_data = pd.concat([positive_df, negative_df, unknown_df], ignore_index=True)
    training_data = training_data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    context.add_output_metadata({
        "total_samples": len(training_data),
        "positive_samples": len(positive_df),
        "negative_samples": len(negative_df),
        "unknown_samples": len(unknown_df),
        "positive_ratio": f"{len(positive_df)/len(training_data):.2%}",
        "negative_ratio": f"{len(negative_df)/len(training_data):.2%}",
        "unknown_ratio": f"{len(unknown_df)/len(training_data):.2%}"
    })

    context.log.info(f"Created {len(training_data)} training samples")
    context.log.info(f"  Positive: {len(positive_df)} ({len(positive_df)/len(training_data)*100:.1f}%)")
    context.log.info(f"  Negative: {len(negative_df)} ({len(negative_df)/len(training_data)*100:.1f}%)")
    context.log.info(f"  Unknown: {len(unknown_df)} ({len(unknown_df)/len(training_data)*100:.1f}%)")

    return training_data


# =====================================================================
# ASSET 4: Feature Engineering (Flatten Embeddings)
# =====================================================================

@asset(group_name="xgboost_drug_discovery", compute_kind="transform")
def xgboost_feature_vectors(
    context: AssetExecutionContext,
    xgboost_training_data: pd.DataFrame,
    xgboost_node_embeddings: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Create feature vectors by concatenating drug and disease embeddings."""
    context.log.info("Creating feature vectors...")

    embeddings = xgboost_node_embeddings
    training_data = xgboost_training_data

    X = []
    y = []
    valid_samples = []

    for _, row in training_data.iterrows():
        drug_id = row['drug_id']
        disease_id = row['disease_id']

        if drug_id in embeddings and disease_id in embeddings:
            # Get embeddings
            drug_emb = embeddings[drug_id]['embedding']
            disease_emb = embeddings[disease_id]['embedding']

            # Concatenate
            feature_vector = np.concatenate([drug_emb, disease_emb])

            X.append(feature_vector)
            y.append(row['label'])
            valid_samples.append(row.to_dict())

    X = np.array(X)
    y = np.array(y)
    samples_df = pd.DataFrame(valid_samples)

    embedding_dim = len(embeddings[list(embeddings.keys())[0]]['embedding'])

    context.add_output_metadata({
        "feature_matrix_shape": str(X.shape),
        "num_samples": X.shape[0],
        "num_features": X.shape[1],
        "drug_embedding_dim": embedding_dim,
        "disease_embedding_dim": embedding_dim,
        "label_distribution": {
            "class_0": int(sum(y == 0)),
            "class_1": int(sum(y == 1)),
            "class_2": int(sum(y == 2))
        }
    })

    context.log.info(f"Created feature matrix: {X.shape}")
    context.log.info(f"  Features: {X.shape[1]} ({embedding_dim} drug + {embedding_dim} disease)")

    return {
        'X': X,
        'y': y,
        'samples': samples_df
    }


# =====================================================================
# ASSET 5: Train XGBoost Model
# =====================================================================

@asset(group_name="xgboost_drug_discovery", compute_kind="ml")
def xgboost_trained_model(
    context: AssetExecutionContext,
    xgboost_feature_vectors: Dict[str, Any],
) -> Dict[str, Any]:
    """Train XGBoost classifier with cross-validation."""
    context.log.info("Training XGBoost model...")

    X = xgboost_feature_vectors['X']
    y = xgboost_feature_vectors['y']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )

    # Cross-validation
    context.log.info("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1
    )

    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())

    context.log.info(f"Cross-validation accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")

    # Train on full training set
    context.log.info("Training on full training set...")
    model.fit(X_train, y_train)

    # Save model to disk
    model_dir = Path("data/06_models/xgboost")
    model_dir.mkdir(parents=True, exist_ok=True)

    import pickle
    model_path = model_dir / "drug_disease_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    context.add_output_metadata({
        "model_type": "XGBClassifier",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "cv_accuracy_mean": f"{cv_mean:.4f}",
        "cv_accuracy_std": f"{cv_std:.4f}",
        "training_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
        "model_saved_to": str(model_path)
    })

    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'cv_scores': cv_scores,
        'model_path': str(model_path)
    }


# =====================================================================
# ASSET 6: Evaluate Model
# =====================================================================

@asset(group_name="xgboost_drug_discovery", compute_kind="ml")
def xgboost_model_evaluation(
    context: AssetExecutionContext,
    xgboost_trained_model: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate model performance on test set."""
    context.log.info("Evaluating model on test set...")

    model = xgboost_trained_model['model']
    X_test = xgboost_trained_model['X_test']
    y_test = xgboost_trained_model['y_test']

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Classification report
    report = classification_report(
        y_test, y_pred,
        target_names=['Does NOT Treat (0)', 'TREATS (1)', 'Unknown (2)'],
        output_dict=True,
        zero_division=0
    )

    # ROC-AUC for binary classification (class 1 vs rest)
    y_test_binary = (y_test == 1).astype(int)
    y_pred_binary = y_pred_proba[:, 1]

    roc_auc = float(roc_auc_score(y_test_binary, y_pred_binary))

    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_test_binary, y_pred_binary)
    pr_auc = float(auc(recall, precision))

    # Log results
    context.log.info("\nClassification Report:")
    for class_name, metrics in report.items():
        if isinstance(metrics, dict):
            context.log.info(f"  {class_name}:")
            context.log.info(f"    Precision: {metrics['precision']:.3f}")
            context.log.info(f"    Recall: {metrics['recall']:.3f}")
            context.log.info(f"    F1-score: {metrics['f1-score']:.3f}")

    context.log.info(f"\nROC-AUC (TREATS vs rest): {roc_auc:.4f}")
    context.log.info(f"Precision-Recall AUC: {pr_auc:.4f}")

    context.add_output_metadata({
        "test_accuracy": f"{report['accuracy']:.4f}",
        "class_1_precision": f"{report['TREATS (1)']['precision']:.4f}",
        "class_1_recall": f"{report['TREATS (1)']['recall']:.4f}",
        "class_1_f1": f"{report['TREATS (1)']['f1-score']:.4f}",
        "roc_auc": f"{roc_auc:.4f}",
        "pr_auc": f"{pr_auc:.4f}"
    })

    return {
        'classification_report': report,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


# =====================================================================
# ASSET 7: Generate All Drug-Disease Pairs
# =====================================================================

@asset(group_name="xgboost_drug_discovery", compute_kind="transform")
def xgboost_all_drug_disease_pairs(
    context: AssetExecutionContext,
    xgboost_node_embeddings: Dict[str, Dict[str, Any]],
    xgboost_known_drug_disease_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Generate all possible drug-disease pairs (excluding known treatments)."""
    context.log.info("Generating all drug-disease pairs...")

    embeddings = xgboost_node_embeddings
    known_df = xgboost_known_drug_disease_pairs

    # Get drugs and diseases
    drugs = {k: v for k, v in embeddings.items() if v['type'] == 'drug'}
    diseases = {k: v for k, v in embeddings.items() if v['type'] == 'disease'}

    # Create set of known pairs
    known_set = {(row['drug_id'], row['disease_id']) for _, row in known_df.iterrows()}

    # Generate all pairs
    all_pairs = []
    for drug_id, drug_data in drugs.items():
        for disease_id, disease_data in diseases.items():
            if (drug_id, disease_id) not in known_set:
                all_pairs.append({
                    'drug_id': drug_id,
                    'disease_id': disease_id,
                    'drug_name': drug_data['name'],
                    'disease_name': disease_data['name']
                })

    pairs_df = pd.DataFrame(all_pairs)

    context.add_output_metadata({
        "total_possible_pairs": len(drugs) * len(diseases),
        "known_pairs_excluded": len(known_df),
        "unknown_pairs_to_score": len(pairs_df),
        "unique_drugs": len(drugs),
        "unique_diseases": len(diseases)
    })

    context.log.info(f"Generated {len(pairs_df):,} unknown drug-disease pairs to score")

    return pairs_df


# =====================================================================
# ASSET 8: Predict All Pairs
# =====================================================================

@asset(group_name="xgboost_drug_discovery", compute_kind="ml")
def xgboost_predictions(
    context: AssetExecutionContext,
    xgboost_all_drug_disease_pairs: pd.DataFrame,
    xgboost_trained_model: Dict[str, Any],
    xgboost_node_embeddings: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Predict treatment probabilities for all drug-disease pairs."""
    context.log.info("Predicting treatment probabilities...")

    model = xgboost_trained_model['model']
    embeddings = xgboost_node_embeddings
    pairs_df = xgboost_all_drug_disease_pairs.copy()

    # Create feature vectors
    X_all = []
    for _, row in pairs_df.iterrows():
        drug_emb = embeddings[row['drug_id']]['embedding']
        disease_emb = embeddings[row['disease_id']]['embedding']
        feature_vector = np.concatenate([drug_emb, disease_emb])
        X_all.append(feature_vector)

    X_all = np.array(X_all)

    context.log.info(f"Predicting for {len(X_all):,} pairs...")

    # Predict
    predictions = model.predict_proba(X_all)

    # Add predictions to DataFrame
    pairs_df['prob_not_treat'] = predictions[:, 0]
    pairs_df['prob_treats'] = predictions[:, 1]  # KEY SCORE
    pairs_df['prob_unknown'] = predictions[:, 2]

    context.add_output_metadata({
        "pairs_scored": len(pairs_df),
        "avg_prob_treats": f"{pairs_df['prob_treats'].mean():.4f}",
        "max_prob_treats": f"{pairs_df['prob_treats'].max():.4f}",
        "min_prob_treats": f"{pairs_df['prob_treats'].min():.4f}"
    })

    context.log.info(f"Predictions complete for {len(pairs_df):,} pairs")

    return pairs_df


# =====================================================================
# ASSET 9: Ranked Results
# =====================================================================

@asset(group_name="xgboost_drug_discovery", compute_kind="transform")
def xgboost_ranked_results(
    context: AssetExecutionContext,
    xgboost_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Rank predictions by treatment probability and add confidence levels."""
    context.log.info("Ranking predictions...")

    # Sort by prob_treats (descending)
    df_sorted = xgboost_predictions.sort_values('prob_treats', ascending=False).reset_index(drop=True)
    df_sorted['rank'] = df_sorted.index + 1

    # Add confidence level
    def confidence_level(score):
        if score >= 0.7:
            return 'HIGH'
        elif score >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'

    df_sorted['confidence'] = df_sorted['prob_treats'].apply(confidence_level)

    # Reorder columns
    df_sorted = df_sorted[[
        'rank', 'drug_name', 'disease_name', 'prob_treats', 'confidence',
        'prob_not_treat', 'prob_unknown', 'drug_id', 'disease_id'
    ]]

    # Save to CSV
    output_dir = Path("data/07_reporting/xgboost")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "drug_discovery_predictions.csv"
    df_sorted.to_csv(output_file, index=False)

    # Statistics
    confidence_dist = df_sorted['confidence'].value_counts().to_dict()

    context.add_output_metadata({
        "total_predictions": len(df_sorted),
        "confidence_distribution": confidence_dist,
        "top_prediction": {
            "drug": df_sorted.iloc[0]['drug_name'],
            "disease": df_sorted.iloc[0]['disease_name'],
            "prob_treats": f"{df_sorted.iloc[0]['prob_treats']:.4f}"
        },
        "output_file": str(output_file),
        "prob_treats_stats": {
            "mean": f"{df_sorted['prob_treats'].mean():.4f}",
            "median": f"{df_sorted['prob_treats'].median():.4f}",
            "std": f"{df_sorted['prob_treats'].std():.4f}"
        }
    })

    # Log top 10
    context.log.info("\nTop 10 Drug Repurposing Candidates:")
    for i, row in df_sorted.head(10).iterrows():
        context.log.info(
            f"  {row['rank']}. {row['drug_name']} → {row['disease_name']} "
            f"(prob: {row['prob_treats']:.4f}, conf: {row['confidence']})"
        )

    context.log.info(f"\nResults saved to: {output_file}")

    return df_sorted
