"""
LightGCN Collaborative Filtering Model (CF Only Baseline)

This script implements the LightGCN graph neural network for generating
collaborative filtering scores based on user-query interaction patterns.

Uses the 181 filtered participants (same as IC baselines) as per paper
Section 5.2.1: participants with ≥5 always/never sharing decisions.

Inputs:
    - ../data/user_study.json - User-item interaction data (task3)
    - ../data/processed_dataset.json - Filtered participant IDs (181 participants)
    - ../queries.json - Query (item) information

Outputs:
    - Trained LightGCN model
    - CF scores saved to: cf_scores.csv
    - Model metrics (ROC-AUC, precision, recall)
"""

import os
import json
import re
import pandas as pd
import numpy as np
import tensorflow as tf

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import merge_rating_true_pred, auc
from recommenders.models.deeprec.deeprec_utils import prepare_hparams
from recommenders.utils.constants import SEED as DEFAULT_SEED
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import unified evaluation utilities
from evaluation_utils import calculate_metrics, save_metrics, save_predictions, save_results_csv, print_metrics

# Configuration
TOP_K = 5
EPOCHS = 120
BATCH_SIZE = 256
SEED = DEFAULT_SEED

# LightGCN configuration
LIGHTGCN_CONFIG = {
    # Model parameters
    'model_type': 'lightgcn',
    'embed_size': 64,
    'n_layers': 20,

    # Training parameters
    'batch_size': BATCH_SIZE,
    'decay': 0.0001,  # L2 regularization
    'epochs': EPOCHS,
    'learning_rate': 0.0001,
    'eval_epoch': 10,
    'top_k': TOP_K,

    # Model saving and metrics
    'save_model': False,
    'save_epoch': 100,
    'metrics': ['recall', 'ndcg', 'precision', 'map'],
    'MODEL_DIR': './lightgcn_model',
    'SUMMARIES_DIR': './lightgcn_summaries'
}


def load_interaction_data():
    """
    Load and prepare user-query interaction data.

    Uses only the 181 filtered participants from processed_dataset.json to ensure
    consistency with IC and IC+CF baselines (per paper Section 5.2.1).

    Returns:
        DataFrame: Interaction data with columns [userID, itemID, rating]
    """
    print("Loading user study data...")
    with open('../data/user_study.json', 'r') as f:
        users = json.load(f)

    # Load processed_dataset to get the 181 filtered participant IDs
    with open('../data/processed_dataset.json', 'r') as f:
        processed_dataset = json.load(f)

    filtered_participant_ids = set(processed_dataset.keys())
    print(f"Using {len(filtered_participant_ids)} filtered participants (same as IC baselines)")

    with open('../queries.json', 'r') as f:
        queries = json.load(f)

    # Build interaction matrix from Task 3 (permission preferences)
    # Only include the 181 filtered participants
    # Include BOTH positive (always share) and negative (never share) interactions
    interactions = []
    for user in users:
        user_id = user['uuid']

        # Skip participants not in filtered set
        if user_id not in filtered_participant_ids:
            continue

        for response in user.get('task3', []):
            query_id = response['id']
            permissions = response.get('answer', {})

            # Process each data type permission separately
            for permission_key, permission_value in permissions.items():
                # permission_key format: "Receiver, Data Type (additional info)"
                # e.g., "Tesla,Your car rental details"

                # Only include "always share" or "never share" responses
                if permission_value == "Yes, always share":
                    rating = 1
                elif permission_value == "No, never share":
                    rating = 0
                else:
                    # Skip "ask me every time" and other responses
                    continue

                # Create item as query:::receiver:::datatype (matching reference format)
                # Extract receiver and datatype from permission_key
                parts = permission_key.split(',', 1)
                if len(parts) == 2:
                    receiver = parts[0].strip()
                    data_type = parts[1].strip()
                    # Remove parenthetical info from data type
                    data_type = re.sub(r'\(.*?\)', '', data_type).strip()
                else:
                    # Fallback if format is different
                    receiver = "Unknown"
                    data_type = permission_key.strip()

                # Create composite itemID (will be mapped to integer later)
                item_key = f"{query_id}:::{receiver}:::{data_type}"

                interactions.append({
                    'userID': user_id,
                    'itemID': item_key,
                    'rating': rating
                })

    df = pd.DataFrame(interactions)
    print(f"Loaded {len(df)} interactions from {len(df['userID'].unique())} users and {len(queries)} queries")
    print(f"Rating distribution: {df['rating'].value_counts().to_dict()}")
    return df


def train_lightgcn(train_data, test_data):
    """
    Train LightGCN model.

    Args:
        train_data: Training interaction data
        test_data: Test interaction data

    Returns:
        LightGCN: Trained model
    """
    print("\nPreparing model...")

    # Prepare hyperparameters
    hparams = prepare_hparams(
        None,
        **LIGHTGCN_CONFIG
    )

    # Prepare data model
    data = ImplicitCF(train=train_data, test=test_data, seed=SEED)

    # Create and train model
    print("Training LightGCN...")
    model = LightGCN(hparams, data, seed=SEED)
    model.fit()

    return model, data


def find_optimal_threshold(y_true, y_pred, method='fpr_fnr'):
    """
    Find the optimal threshold using different methods.

    Args:
        y_true: True labels
        y_pred: Predicted scores
        method: 'fpr_fnr' (FPR=FNR), 'f1' (maximize F1), or 'accuracy' (maximize accuracy)

    Returns:
        float: Optimal threshold
    """
    thresholds = np.linspace(y_pred.min(), y_pred.max(), 1000)

    if method == 'fpr_fnr':
        # Find threshold where FPR = FNR (balanced error rate)
        best_diff = float('inf')
        best_threshold = 0

        for threshold in thresholds:
            binary_predictions = (y_pred >= threshold).astype(int)

            # Calculate FPR and FNR
            TP = FP = TN = FN = 0
            for true_label, pred_label in zip(y_true, binary_predictions):
                if true_label == 1 and pred_label == 1:
                    TP += 1
                elif true_label == 1 and pred_label == 0:
                    FN += 1
                elif true_label == 0 and pred_label == 0:
                    TN += 1
                elif true_label == 0 and pred_label == 1:
                    FP += 1

            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
            FNR = FN / (FN + TP) if (FN + TP) > 0 else 0

            # Find where FPR ≈ FNR
            diff = abs(FPR - FNR)
            if diff < best_diff:
                best_diff = diff
                best_threshold = threshold

        return best_threshold

    elif method == 'f1':
        # Maximize F1 score
        best_f1 = 0
        best_threshold = 0

        for threshold in thresholds:
            binary_predictions = (y_pred >= threshold).astype(int)
            f1 = f1_score(y_true, binary_predictions, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold

    elif method == 'accuracy':
        # Maximize accuracy
        best_accuracy = 0
        best_threshold = 0

        for threshold in thresholds:
            binary_predictions = (y_pred >= threshold).astype(int)
            acc = accuracy_score(y_true, binary_predictions)
            if acc > best_accuracy:
                best_accuracy = acc
                best_threshold = threshold

        return best_threshold

    else:
        raise ValueError(f"Unknown method: {method}")


def evaluate_model(model, data, test_data, threshold=None):
    """
    Evaluate model performance using binary classification metrics.

    Args:
        model: Trained LightGCN model
        data: ImplicitCF data object
        test_data: Test DataFrame
        threshold: Threshold for binary classification

    Returns:
        dict: Evaluation metrics (Accuracy, Precision, Recall, F1, FPR, FNR, AUC)
    """
    print("\nEvaluating model...")

    # Get predictions for test set users
    user_ids = test_data['userID'].unique()
    user_ids_mapped = [data.user2id[uid] for uid in user_ids]

    # Get scores for all items for these users
    all_scores = model.score(user_ids_mapped, remove_seen=False)

    # Create results DataFrame matching test data structure
    results = test_data.copy()
    predictions = []

    for index, row in test_data.iterrows():
        user_id = row['userID']
        item_id = row['itemID']

        # Get mapped IDs
        user_idx = data.user2id[user_id]
        item_idx = data.item2id[item_id]

        # Get prediction score
        user_position = user_ids_mapped.index(user_idx)
        pred_score = float(all_scores[user_position][item_idx])
        predictions.append(pred_score)

    results['prediction'] = predictions

    # Merge true and predicted ratings
    y_true, y_pred = merge_rating_true_pred(test_data, results, col_prediction='prediction')

    # Calculate AUC
    roc_auc = auc(test_data, results, col_prediction='prediction')

    # Find optimal threshold if not provided
    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_pred, method='fpr_fnr')
        print(f"  Optimal threshold (FPR=FNR): {threshold:.4f}")

    # Binarize predictions based on threshold
    binary_predictions = (y_pred >= threshold).astype(int)

    # Calculate binary classification metrics
    accuracy = accuracy_score(y_true, binary_predictions)
    precision = precision_score(y_true, binary_predictions)
    recall = recall_score(y_true, binary_predictions)
    f1 = f1_score(y_true, binary_predictions)

    # Calculate confusion matrix components
    TP = FP = TN = FN = 0
    for true_label, pred_label in zip(y_true, binary_predictions):
        if true_label == 1 and pred_label == 1:
            TP += 1
        elif true_label == 1 and pred_label == 0:
            FN += 1
        elif true_label == 0 and pred_label == 0:
            TN += 1
        elif true_label == 0 and pred_label == 1:
            FP += 1

    # Calculate rates
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) != 0 else 0

    metrics = {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': FPR,
        'fnr': FNR,
        'auc': roc_auc
    }

    return metrics, results


def generate_cf_scores(model, data):
    """
    Generate collaborative filtering scores for all user-query pairs.

    Args:
        model: Trained LightGCN model
        data: Data model

    Returns:
        DataFrame: CF scores with columns [user_id, query_id, score]
    """
    print("\nGenerating CF scores...")

    # Get all unique users and items
    users = data.train.userID.unique()
    items = data.train.itemID.unique()

    # Generate scores for all user-item pairs
    all_scores = []
    for user in users:
        # model.score() returns scores for all items for given user
        # Returns array of shape (n_items,)
        user_scores = model.score([user], remove_seen=False)

        # user_scores is 2D array: [n_users, n_items]
        # We only passed one user, so take first row
        scores_for_user = user_scores[0]

        for item_idx, item in enumerate(items):
            all_scores.append({
                'user_id': user,
                'query_id': item,
                'score': float(scores_for_user[item_idx])
            })

    scores_df = pd.DataFrame(all_scores)
    print(f"Generated {len(scores_df)} CF scores")

    return scores_df


def main():
    """Main execution function."""

    print("="*60)
    print("LightGCN Collaborative Filtering - CF Only Baseline")
    print("="*60)

    # Load data
    interactions = load_interaction_data()

    # Split into train/test
    print("\nSplitting data (80/20)...")
    train, test = python_stratified_split(interactions, ratio=0.8, seed=SEED)

    # LightGCN uses BPR loss which requires only positive examples for training
    # Filter out rating=0 from training data, but keep them in test data for evaluation
    train_positive_only = train[train['rating'] > 0].copy()

    print(f"Train: {len(train)} interactions ({len(train_positive_only)} positive for model training)")
    print(f"Test: {len(test)} interactions (positive: {len(test[test['rating'] > 0])}, negative: {len(test[test['rating'] == 0])})")

    # Train model (using only positive examples)
    model, data = train_lightgcn(train_positive_only, test)

    # Evaluate on test set
    raw_metrics, results = evaluate_model(model, data, test)

    # Convert to unified metrics format
    y_true = results['rating'].values
    y_pred = (results['prediction'] >= raw_metrics['threshold']).astype(int).values

    unified_metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        method_name="CF Only",
        threshold=raw_metrics['threshold']
    )
    unified_metrics['auc'] = raw_metrics['auc']

    # Print metrics
    print_metrics(unified_metrics)

    # Create predictions in unified format
    predictions = {
        "cf_only": {
            "method": "CF Only",
            "test_predictions": results.to_dict('records'),
            "metrics": unified_metrics
        }
    }

    # Save outputs to results/ folder
    os.makedirs("../results", exist_ok=True)
    save_predictions(predictions, "../results/cf_only_predictions.json")
    save_metrics(unified_metrics, "../results/cf_only_metrics.json")

    # Generate CF scores for all user-query pairs (used by IC+CF hybrid)
    cf_scores = generate_cf_scores(model, data)

    # Save CF scores (needed by IC+CF script)
    cf_scores.to_csv("../results/cf_scores.csv", index=False)
    print(f"✓ CF scores saved to ../results/cf_scores.csv (for IC+CF hybrid)")

    print("\n" + "="*60)
    print("Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
