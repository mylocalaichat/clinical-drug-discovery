"""
Simplified Drug-Disease Prediction with XGBoost

Quick version for testing and experimentation.

Usage:
    python train_xgboost_simple.py
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import random


def load_data(uri='bolt://localhost:7687'):
    """Load embeddings and known relationships from Memgraph"""
    driver = GraphDatabase.driver(uri, auth=None)
    session = driver.session()

    print("Loading embeddings...")
    # Get embeddings
    result = session.run("""
        MATCH (n)
        WHERE EXISTS(n.embedding) AND n.is_example = true
        RETURN n.node_id as id, n.node_name as name,
               n.node_type as type, n.embedding as embedding
    """)

    embeddings = {}
    for r in result:
        embeddings[r['id']] = {
            'name': r['name'],
            'type': r['type'],
            'embedding': np.array(r['embedding'])
        }

    drugs = {k: v for k, v in embeddings.items() if v['type'] == 'drug'}
    diseases = {k: v for k, v in embeddings.items() if v['type'] == 'disease'}

    print(f"  Drugs: {len(drugs)}, Diseases: {len(diseases)}")

    # Get known treatments
    print("Loading known treatments...")
    result = session.run("""
        MATCH (drug:Node)-[r:RELATES {relation: 'drug_treats_disease'}]->(disease:Node)
        WHERE drug.is_example = true AND disease.is_example = true
        RETURN drug.node_id as drug_id, disease.node_id as disease_id
    """)

    known = [(r['drug_id'], r['disease_id']) for r in result]
    print(f"  Known treatments: {len(known)}")

    session.close()
    driver.close()

    return embeddings, drugs, diseases, known


def create_dataset(embeddings, drugs, diseases, known):
    """Create training dataset"""
    print("\nCreating dataset...")

    X, y, pairs = [], [], []
    known_set = set(known)

    drug_ids = list(drugs.keys())
    disease_ids = list(diseases.keys())

    # Positive samples
    for drug_id, disease_id in known:
        if drug_id in embeddings and disease_id in embeddings:
            drug_emb = embeddings[drug_id]['embedding']
            disease_emb = embeddings[disease_id]['embedding']
            X.append(np.concatenate([drug_emb, disease_emb]))
            y.append(1)
            pairs.append((drug_id, disease_id))

    n_positives = len(X)
    print(f"  Positives: {n_positives}")

    # Negative samples
    n_negatives = n_positives
    count = 0
    while count < n_negatives:
        drug_id = random.choice(drug_ids)
        disease_id = random.choice(disease_ids)

        if (drug_id, disease_id) not in known_set:
            drug_emb = embeddings[drug_id]['embedding']
            disease_emb = embeddings[disease_id]['embedding']
            X.append(np.concatenate([drug_emb, disease_emb]))
            y.append(0)
            pairs.append((drug_id, disease_id))
            count += 1

    print(f"  Negatives: {count}")
    print(f"  Total samples: {len(X)}")
    print(f"  Features per sample: {len(X[0])}")

    return np.array(X), np.array(y), pairs


def train_model(X, y):
    """Train XGBoost classifier"""
    print("\nTraining model...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Train
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\nTest Set Performance:")
    print(classification_report(y_test, y_pred,
                               target_names=['Doesn\'t Treat', 'Treats']))

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")

    return model


def predict_all_pairs(model, embeddings, drugs, diseases, known):
    """Predict all unknown drug-disease pairs"""
    print("\nPredicting all drug-disease pairs...")

    known_set = set(known)
    predictions = []

    drug_ids = list(drugs.keys())
    disease_ids = list(diseases.keys())

    for drug_id in drug_ids:
        for disease_id in disease_ids:
            if (drug_id, disease_id) not in known_set:
                drug_emb = embeddings[drug_id]['embedding']
                disease_emb = embeddings[disease_id]['embedding']
                X = np.concatenate([drug_emb, disease_emb]).reshape(1, -1)

                prob = model.predict_proba(X)[0, 1]

                predictions.append({
                    'drug': embeddings[drug_id]['name'],
                    'disease': embeddings[disease_id]['name'],
                    'prob_treats': prob
                })

    # Sort by probability
    predictions.sort(key=lambda x: x['prob_treats'], reverse=True)

    print(f"  Total predictions: {len(predictions)}")

    return predictions


def save_results(predictions, filename='predictions_simple.csv'):
    """Save predictions to CSV"""
    df = pd.DataFrame(predictions)
    df['rank'] = range(1, len(df) + 1)
    df = df[['rank', 'drug', 'disease', 'prob_treats']]

    df.to_csv(filename, index=False)
    print(f"\nSaved results to: {filename}")

    print("\nTop 20 Predictions:")
    print(df.head(20).to_string(index=False))

    return df


def main():
    """Main pipeline"""
    print("="*80)
    print("SIMPLIFIED DRUG-DISEASE PREDICTION")
    print("="*80)

    # Load data
    embeddings, drugs, diseases, known = load_data()

    # Create dataset
    X, y, pairs = create_dataset(embeddings, drugs, diseases, known)

    # Train model
    model = train_model(X, y)

    # Predict all pairs
    predictions = predict_all_pairs(model, embeddings, drugs, diseases, known)

    # Save results
    results = save_results(predictions)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
