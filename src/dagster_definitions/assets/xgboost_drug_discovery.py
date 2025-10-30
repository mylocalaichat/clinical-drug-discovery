"""
Dagster assets for XGBoost-based drug discovery pipeline.

This module implements the complete drug-disease prediction pipeline:
1. Load known drug-disease relationships (positive: indications, negative: contraindications)
2. Create training data using indication and contraindication relationships
3. Feature engineering (flatten embeddings)
4. Train XGBoost classifier
5. Evaluate model performance
6. Predict all drug-disease pairs
7. Rank and output results
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from dagster import AssetExecutionContext, asset
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score, auc
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBClassifier


# NOTE: `xgboost_known_drug_disease_pairs` removed — its functionality is consolidated
# into `xgboost_train_test_split` which reads edges and joins embeddings in one asset.



# NOTE: `xgboost_training_data` removed — its functionality is consolidated
# into `xgboost_train_test_split` which reads edges and joins embeddings in one asset.


# =====================================================================
# ASSET 4: Feature Engineering (Flatten Embeddings)
# =====================================================================
# =====================================================================

# NOTE: `xgboost_feature_vectors` removed — its functionality is consolidated
# into `xgboost_train_test_split` which reads edges and joins embeddings in one asset.


# =====================================================================
# ASSET 5: Train/Test Split
# =====================================================================

@asset(group_name="xgboost_drug_discovery", compute_kind="transform")
def xgboost_train_test_split(
    context: AssetExecutionContext,
    flattened_embeddings: pd.DataFrame,
) -> Dict[str, Any]:
    """Read edges CSV, join embeddings, build feature vectors, and split into train/test.

    Outputs CSVs for train and test datasets and returns arrays and sample DataFrames.
    """
    context.log.info("Creating training and test datasets from edges CSV + flattened embeddings...")

    edges_file = "data/01_raw/primekg/edges.csv"
    try:
        edges_df = pd.read_csv(edges_file)
        context.log.info(f"Loaded {len(edges_df):,} total edges from {edges_file}")
    except FileNotFoundError:
        context.log.error(f"Edges file not found: {edges_file}")
        return {}
    except Exception as e:
        context.log.error(f"Error reading edges CSV: {e}")
        return {}

    # Filter for indication and contraindication relationships between drugs and diseases
    relation_df = edges_df[
        ((edges_df['relation'] == 'indication') | (edges_df['relation'] == 'contraindication')) &
        (edges_df['x_type'] == 'drug') &
        (edges_df['y_type'] == 'disease')
    ].copy()

    context.log.info(f"Filtered to {len(relation_df):,} drug-disease edges (indication/contraindication)")

    # Build known pairs list
    known_pairs = []
    for _, row in relation_df.iterrows():
        label = 1 if row['relation'] == 'indication' else 0
        known_pairs.append({
            'drug_id': str(row['x_id']),
            'drug_name': str(row.get('x_name', '')),
            'disease_id': str(row['y_id']),
            'disease_name': str(row.get('y_name', '')),
            'label': label,
            'relationship_type': row['relation']
        })

    if not known_pairs:
        context.log.warning("No known drug-disease pairs found in edges CSV")
        return {}

    known_df = pd.DataFrame(known_pairs)

    # Build embeddings lookup
    embedding_cols = [c for c in flattened_embeddings.columns if c.startswith('emb_')]
    if not embedding_cols:
        raise ValueError("No embedding columns found in flattened_embeddings DataFrame")

    embeddings_dict = {}
    for _, row in flattened_embeddings.iterrows():
        embeddings_dict[str(row['node_id'])] = row[embedding_cols].values.astype(np.float32)

    context.log.info(f"Created embeddings lookup for {len(embeddings_dict)} nodes")

    # Build samples with concatenated embeddings
    X_list = []
    y_list = []
    samples = []

    for _, row in known_df.iterrows():
        d_id = str(row['drug_id'])
        dis_id = str(row['disease_id'])
        if d_id in embeddings_dict and dis_id in embeddings_dict:
            drug_emb = embeddings_dict[d_id]
            dis_emb = embeddings_dict[dis_id]
            feat = np.concatenate([drug_emb, dis_emb])
            X_list.append(feat)
            y_list.append(int(row['label']))
            samples.append({
                'drug_id': d_id,
                'disease_id': dis_id,
                'drug_name': row['drug_name'],
                'disease_name': row['disease_name'],
                'label': int(row['label']),
                'relationship_type': row['relationship_type']
            })

    if len(X_list) == 0:
        context.log.warning("No samples with embeddings found. Aborting.")
        return {}

    X = np.vstack(X_list)
    y = np.array(y_list)
    samples_df = pd.DataFrame(samples)

    # Train/test split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, samples_df.index.values, test_size=0.2, random_state=42, stratify=y
    )

    samples_train = samples_df.loc[idx_train].reset_index(drop=True)
    samples_test = samples_df.loc[idx_test].reset_index(drop=True)

    # Attach features as lists for CSV output
    samples_train = samples_train.copy()
    samples_train['features'] = [list(x) for x in X_train]
    samples_train['label'] = y_train

    samples_test = samples_test.copy()
    samples_test['features'] = [list(x) for x in X_test]
    samples_test['label'] = y_test

    # Save CSVs
    output_dir = Path("data/07_model_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_file = output_dir / "xgboost_train_set.csv"
    test_file = output_dir / "xgboost_test_set.csv"
    samples_train.to_csv(train_file, index=False)
    samples_test.to_csv(test_file, index=False)

    # Add metadata including file URIs
    context.add_output_metadata({
        'train_file_uri': str(train_file.resolve()),
        'test_file_uri': str(test_file.resolve()),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_label_distribution': {
            'indications_1': int(sum(y_train == 1)),
            'contraindications_0': int(sum(y_train == 0))
        },
        'test_label_distribution': {
            'indications_1': int(sum(y_test == 1)),
            'contraindications_0': int(sum(y_test == 0))
        }
    })

    context.log.info(f"Saved train/test CSVs: {train_file}, {test_file}")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_file': str(train_file.resolve()),
        'test_file': str(test_file.resolve()),
        'samples_train': samples_train,
        'samples_test': samples_test,
    }


# =====================================================================
# ASSET 6: Train XGBoost Model
# =====================================================================

@asset(group_name="xgboost_drug_discovery", compute_kind="ml")
def xgboost_trained_model(
    context: AssetExecutionContext,
    xgboost_train_test_split: Dict[str, Any],
) -> Dict[str, Any]:
    """Train XGBoost classifier with cross-validation."""
    context.log.info("Training XGBoost model...")

    X_train = xgboost_train_test_split['X_train']
    y_train = xgboost_train_test_split['y_train']
    X_test = xgboost_train_test_split['X_test']
    y_test = xgboost_train_test_split['y_test']

    # Initialize model for binary classification (0=contraindication, 1=indication)
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
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

    # Debug logging for input data
    context.log.info(f"X_test shape: {X_test.shape}")
    context.log.info(f"y_test shape before prediction: {y_test.shape}")
    context.log.info(f"y_test type: {type(y_test)}")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Debug logging for predictions
    context.log.info(f"y_pred shape after prediction: {y_pred.shape}")
    context.log.info(f"y_pred_proba shape: {y_pred_proba.shape}")

    # Ensure y_test and y_pred are 1D arrays (fix for multilabel-indicator error)
    y_test = np.asarray(y_test).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Debug logging
    context.log.info(f"y_test shape after flatten: {y_test.shape}, dtype: {y_test.dtype}")
    context.log.info(f"y_pred shape after flatten: {y_pred.shape}, dtype: {y_pred.dtype}")
    context.log.info(f"y_test unique values: {np.unique(y_test)}")
    context.log.info(f"y_pred unique values: {np.unique(y_pred)}")

    # Additional safety check - truncate to minimum length if sizes don't match
    min_len = min(len(y_test), len(y_pred))
    if len(y_test) != len(y_pred):
        context.log.warning(f"Size mismatch: y_test={len(y_test)}, y_pred={len(y_pred)}. Truncating to {min_len}")
        y_test = y_test[:min_len]
        y_pred = y_pred[:min_len]
        y_pred_proba_class1 = y_pred_proba[:min_len, 1]
    else:
        y_pred_proba_class1 = y_pred_proba[:, 1]

    # Classification report for binary classification
    report = classification_report(
        y_test, y_pred,
        target_names=['Contraindication (0)', 'Indication (1)'],
        output_dict=True,
        zero_division=0
    )

    # ROC-AUC for binary classification
    roc_auc = float(roc_auc_score(y_test, y_pred_proba_class1))

    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_class1)
    pr_auc = float(auc(recall, precision))

    # Log results
    context.log.info("\nClassification Report:")
    for class_name, metrics in report.items():
        if isinstance(metrics, dict):
            context.log.info(f"  {class_name}:")
            context.log.info(f"    Precision: {metrics['precision']:.3f}")
            context.log.info(f"    Recall: {metrics['recall']:.3f}")
            context.log.info(f"    F1-score: {metrics['f1-score']:.3f}")

    context.log.info(f"\nROC-AUC: {roc_auc:.4f}")
    context.log.info(f"Precision-Recall AUC: {pr_auc:.4f}")

    context.add_output_metadata({
        "test_accuracy": f"{report['accuracy']:.4f}",
        "indication_precision": f"{report['Indication (1)']['precision']:.4f}",
        "indication_recall": f"{report['Indication (1)']['recall']:.4f}",
        "indication_f1": f"{report['Indication (1)']['f1-score']:.4f}",
        "contraindication_precision": f"{report['Contraindication (0)']['precision']:.4f}",
        "contraindication_recall": f"{report['Contraindication (0)']['recall']:.4f}",
        "contraindication_f1": f"{report['Contraindication (0)']['f1-score']:.4f}",
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
# ASSET 8: Predict All Pairs
# =====================================================================

@asset(group_name="xgboost_drug_discovery", compute_kind="ml")
def xgboost_predictions(
    context: AssetExecutionContext,
    xgboost_trained_model: Dict[str, Any],
    flattened_embeddings: pd.DataFrame,
) -> pd.DataFrame:
    """Predict treatment probabilities for all drug-disease pairs.

    NOTE: This asset is currently disabled because xgboost_all_drug_disease_pairs was removed.
    """
    raise NotImplementedError("xgboost_all_drug_disease_pairs asset has been removed")


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
