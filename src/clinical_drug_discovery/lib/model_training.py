"""
Model training module for supervised drug-disease link prediction.

This module implements:
1. Data loading from Neo4j (drugs, diseases, known pairs)
2. Negative sampling for training data generation
3. Feature engineering with embeddings
4. Ensemble model training (XGBoost, LightGBM, RandomForest)
"""

import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

# Try to import LightGBM, but make it optional (can fail on some systems)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: LightGBM not available: {e}")
    print("LightGBM model training will be skipped.")
    LIGHTGBM_AVAILABLE = False
    lgb = None


def load_drugs_from_neo4j(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    database: str = "primekg",
) -> pd.DataFrame:
    """
    Load all drugs from Neo4j.

    Returns:
        DataFrame with columns: node_id, node_name
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    query = """
    MATCH (n:PrimeKGNode)
    WHERE n.node_type = 'drug'
    RETURN n.node_id as node_id, n.node_name as node_name
    """

    try:
        with driver.session(database=database) as session:
            result = session.run(query)
            drugs_df = pd.DataFrame([dict(record) for record in result])

        print(f"Loaded {len(drugs_df):,} drugs from Neo4j")
        return drugs_df

    finally:
        driver.close()


def load_diseases_from_neo4j(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    database: str = "primekg",
) -> pd.DataFrame:
    """
    Load all diseases from Neo4j.

    Returns:
        DataFrame with columns: node_id, node_name
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    query = """
    MATCH (n:PrimeKGNode)
    WHERE n.node_type = 'disease'
    RETURN n.node_id as node_id, n.node_name as node_name
    """

    try:
        with driver.session(database=database) as session:
            result = session.run(query)
            diseases_df = pd.DataFrame([dict(record) for record in result])

        print(f"Loaded {len(diseases_df):,} diseases from Neo4j")
        return diseases_df

    finally:
        driver.close()


def load_known_drug_disease_pairs(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    database: str = "primekg",
) -> pd.DataFrame:
    """
    Load known drug-disease treatment relationships from Neo4j.

    Returns:
        DataFrame with columns: drug_id, disease_id, label (1 for known treatment)
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    query = """
    MATCH (drug:PrimeKGNode)-[:INDICATION]->(disease:PrimeKGNode)
    WHERE drug.node_type = 'drug' AND disease.node_type = 'disease'
    RETURN drug.node_id as drug_id,
           disease.node_id as disease_id,
           1 as label
    """

    try:
        with driver.session(database=database) as session:
            result = session.run(query)
            pairs_df = pd.DataFrame([dict(record) for record in result])

        print(f"Loaded {len(pairs_df):,} known drug-disease pairs")
        return pairs_df

    finally:
        driver.close()


def generate_negative_samples(
    drugs: List[str],
    diseases: List[str],
    known_pairs: pd.DataFrame,
    n_negatives: int,
    label: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate negative samples by randomly sampling drug-disease pairs
    that are NOT in the known pairs set.

    Args:
        drugs: List of drug IDs
        diseases: List of disease IDs
        known_pairs: DataFrame with known (drug_id, disease_id) pairs
        n_negatives: Number of negative samples to generate
        label: Label for negative samples (0 = not treat, 2 = unknown)
        seed: Random seed

    Returns:
        DataFrame with negative samples
    """
    random.seed(seed)
    np.random.seed(seed)

    # Create set of known pairs for fast lookup
    known_set = set(zip(known_pairs['drug_id'], known_pairs['disease_id']))

    print(f"Generating {n_negatives:,} negative samples...")

    negative_samples = []
    attempts = 0
    max_attempts = n_negatives * 10  # Prevent infinite loop

    with tqdm(total=n_negatives, desc="Generating negatives") as pbar:
        while len(negative_samples) < n_negatives and attempts < max_attempts:
            drug = random.choice(drugs)
            disease = random.choice(diseases)
            attempts += 1

            if (drug, disease) not in known_set:
                negative_samples.append({
                    'drug_id': drug,
                    'disease_id': disease,
                    'label': label,
                })
                pbar.update(1)

    if len(negative_samples) < n_negatives:
        print(f"Warning: Only generated {len(negative_samples):,} negatives (requested {n_negatives:,})")

    return pd.DataFrame(negative_samples)


def create_training_data_with_negative_sampling(
    drugs_df: pd.DataFrame,
    diseases_df: pd.DataFrame,
    known_pairs_df: pd.DataFrame,
    n_negatives_per_positive: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create training dataset with positive and negative samples.

    Following the link prediction approach:
    - Label 0: Does NOT treat (negative samples)
    - Label 1: Does treat (known positives)
    - Label 2: Unknown (synthetic negative samples for additional uncertainty)

    Args:
        drugs_df: DataFrame with drug information
        diseases_df: DataFrame with disease information
        known_pairs_df: DataFrame with known drug-disease pairs
        n_negatives_per_positive: Number of negative samples per positive (default 2)
        seed: Random seed

    Returns:
        DataFrame with training data
    """
    print("\n=== Creating Training Data with Negative Sampling ===")

    # Prepare data
    drugs_list = drugs_df['node_id'].tolist()
    diseases_list = diseases_df['node_id'].tolist()

    n_positives = len(known_pairs_df)
    n_negatives = n_positives * n_negatives_per_positive

    print(f"Known positives (label=1): {n_positives:,}")
    print(f"Generating negatives (label=0): {n_negatives:,}")

    # Split negatives: half as label=0, half as label=2 (unknown)
    n_label_0 = n_negatives // 2
    n_label_2 = n_negatives - n_label_0

    # Generate label=0 negatives (not treat)
    negatives_0 = generate_negative_samples(
        drugs=drugs_list,
        diseases=diseases_list,
        known_pairs=known_pairs_df,
        n_negatives=n_label_0,
        label=0,
        seed=seed,
    )

    # Generate label=2 negatives (unknown)
    negatives_2 = generate_negative_samples(
        drugs=drugs_list,
        diseases=diseases_list,
        known_pairs=known_pairs_df,
        n_negatives=n_label_2,
        label=2,
        seed=seed + 1,  # Different seed
    )

    # Combine all data
    training_data = pd.concat([
        known_pairs_df,  # label=1
        negatives_0,     # label=0
        negatives_2,     # label=2
    ], ignore_index=True)

    # Shuffle
    training_data = training_data.sample(frac=1, random_state=seed).reset_index(drop=True)

    print("\nFinal training data:")
    print(f"  Label 0 (not treat): {len(training_data[training_data['label'] == 0]):,}")
    print(f"  Label 1 (treat): {len(training_data[training_data['label'] == 1]):,}")
    print(f"  Label 2 (unknown): {len(training_data[training_data['label'] == 2]):,}")
    print(f"  Total: {len(training_data):,}")

    return training_data


def attach_embeddings_to_pairs(
    pairs_df: pd.DataFrame,
    embeddings: Dict[str, np.ndarray],
    drug_id_col: str = 'drug_id',
    disease_id_col: str = 'disease_id',
) -> pd.DataFrame:
    """
    Attach drug and disease embeddings to pairs DataFrame.

    Args:
        pairs_df: DataFrame with drug_id and disease_id columns
        embeddings: Dictionary mapping node_id to embedding vector
        drug_id_col: Name of drug ID column
        disease_id_col: Name of disease ID column

    Returns:
        DataFrame with added drug_embedding and disease_embedding columns
    """
    print("Attaching embeddings to pairs...")

    # Create copies to avoid modifying original
    result = pairs_df.copy()

    # Attach drug embeddings
    result['drug_embedding'] = result[drug_id_col].apply(
        lambda x: embeddings.get(x)
    )

    # Attach disease embeddings
    result['disease_embedding'] = result[disease_id_col].apply(
        lambda x: embeddings.get(x)
    )

    # Check for missing embeddings
    missing_drugs = result['drug_embedding'].isna().sum()
    missing_diseases = result['disease_embedding'].isna().sum()

    if missing_drugs > 0:
        print(f"Warning: {missing_drugs} pairs have missing drug embeddings")
    if missing_diseases > 0:
        print(f"Warning: {missing_diseases} pairs have missing disease embeddings")

    # Drop rows with missing embeddings
    before_len = len(result)
    result = result.dropna(subset=['drug_embedding', 'disease_embedding'])
    after_len = len(result)

    if before_len != after_len:
        print(f"Dropped {before_len - after_len} pairs with missing embeddings")

    print(f"Final pairs with embeddings: {len(result):,}")

    return result


def prepare_features_for_training(
    training_data: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix X and label vector y from training data.

    Concatenates drug and disease embeddings to create 1024-dimensional
    feature vectors (512 from drug + 512 from disease).

    Args:
        training_data: DataFrame with drug_embedding, disease_embedding, and label

    Returns:
        Tuple of (X, y) where X is features and y is labels
    """
    print("Preparing features for training...")

    # Extract embeddings
    drug_embeddings = np.stack(training_data['drug_embedding'].values)
    disease_embeddings = np.stack(training_data['disease_embedding'].values)

    # Concatenate: [512 drug dims] + [512 disease dims] = 1024 features
    X = np.concatenate([drug_embeddings, disease_embeddings], axis=1)

    # Extract labels
    y = training_data['label'].values

    print(f"Feature matrix X shape: {X.shape}")
    print(f"Label vector y shape: {y.shape}")
    print("Label distribution:")
    for label in sorted(np.unique(y)):
        count = (y == label).sum()
        pct = count / len(y) * 100
        print(f"  Label {label}: {count:,} ({pct:.1f}%)")

    return X, y


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    **kwargs
) -> XGBClassifier:
    """
    Train XGBoost classifier for drug-disease prediction.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        **kwargs: Additional XGBoost parameters

    Returns:
        Trained XGBClassifier
    """
    print("\n=== Training XGBoost Model ===")

    # Default parameters (can be overridden)
    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'multi:softprob',
        'num_class': 3,
        'random_state': 42,
        'n_jobs': -1,
    }
    params.update(kwargs)

    print(f"Parameters: {params}")

    model = XGBClassifier(**params)

    # Prepare eval set if validation data provided
    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]

    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False,
    )

    print("✓ XGBoost training complete")

    return model


def train_lightgbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    **kwargs
):
    """
    Train LightGBM classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        **kwargs: Additional LightGBM parameters

    Returns:
        Trained LGBMClassifier or None if LightGBM not available
    """
    if not LIGHTGBM_AVAILABLE:
        print("\n=== Skipping LightGBM Model (not available) ===")
        return None

    print("\n=== Training LightGBM Model ===")

    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'multiclass',
        'num_class': 3,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }
    params.update(kwargs)

    print(f"Parameters: {params}")

    model = lgb.LGBMClassifier(**params)

    # Prepare eval set
    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]

    model.fit(
        X_train, y_train,
        eval_set=eval_set,
    )

    print("✓ LightGBM training complete")

    return model


def train_random_forest_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    **kwargs
) -> RandomForestClassifier:
    """
    Train Random Forest classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        **kwargs: Additional RandomForest parameters

    Returns:
        Trained RandomForestClassifier
    """
    print("\n=== Training Random Forest Model ===")

    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'n_jobs': -1,
    }
    params.update(kwargs)

    print(f"Parameters: {params}")

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    print("✓ Random Forest training complete")

    return model


def save_model(model, output_path: str) -> None:
    """Save trained model to disk."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(model, f)

    print(f"Saved model to: {output_file}")


def load_model(input_path: str):
    """Load trained model from disk."""
    with open(input_path, 'rb') as f:
        model = pickle.load(f)

    print(f"Loaded model from {input_path}")
    return model
