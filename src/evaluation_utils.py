"""
Shared evaluation utilities for all permission prediction methods.

Provides unified metrics calculation and output formatting across:
- CF-only (Collaborative Filtering)
- IC-only (In-Context Learning)
- IC+CF (Hybrid approach)
"""

import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def calculate_metrics(y_true, y_pred, method_name, threshold=None):
    """
    Calculate standard evaluation metrics.

    Args:
        y_true: List of true labels (0 or 1)
        y_pred: List of predicted labels (0 or 1)
        method_name: Name of the method (e.g., "IC Only", "IC+CF", "CF Only")
        threshold: Optional threshold used for binary classification

    Returns:
        dict: Dictionary containing all metrics
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Calculate FPR and FNR from confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    metrics = {
        "method": method_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "n_predictions": len(y_true)
    }

    if threshold is not None:
        metrics["threshold"] = float(threshold)

    return metrics


def save_metrics(metrics, output_path):
    """Save metrics to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {output_path}")


def save_predictions(predictions, output_path):
    """Save predictions to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"✓ Predictions saved to {output_path}")


def save_results_csv(results_df, output_path):
    """Save detailed results to CSV file."""
    results_df.to_csv(output_path, index=False)
    print(f"✓ Detailed results saved to {output_path}")


def print_metrics(metrics):
    """Print metrics in a formatted table."""
    print(f"\n{'='*60}")
    print(f"{metrics['method']} - Evaluation Metrics")
    print(f"{'='*60}")
    print(f"Predictions:  {metrics['n_predictions']:,}")
    if 'n_participants' in metrics:
        print(f"Participants: {metrics['n_participants']}")
    if 'threshold' in metrics:
        print(f"Threshold:    {metrics['threshold']:.4f}")
    print(f"\nPerformance:")
    print(f"  Accuracy:   {metrics['accuracy']*100:6.1f}%")
    print(f"  Precision:  {metrics['precision']*100:6.1f}%")
    print(f"  Recall:     {metrics['recall']*100:6.1f}%")
    print(f"  F1 Score:   {metrics['f1']*100:6.1f}%")
    print(f"  FPR:        {metrics['fpr']*100:6.1f}%")
    print(f"  FNR:        {metrics['fnr']*100:6.1f}%")
    print(f"{'='*60}\n")


def evaluate_predictions_from_file(predictions_file, method_name):
    """
    Evaluate predictions from JSON file (IC-only or IC+CF format).

    Args:
        predictions_file: Path to predictions JSON file
        method_name: Name of the method

    Returns:
        dict: Metrics dictionary
    """
    with open(predictions_file, 'r') as f:
        results = json.load(f)

    all_true = []
    all_pred = []
    n_participants = 0

    for participant_id, data in results.items():
        if data.get('skipped') or 'error' in data:
            continue

        n_participants += 1
        predictions = data.get('predictions', [])
        ground_truth = data.get('ground_truth', [])

        # Match predictions with ground truth by query ID
        for gt_item in ground_truth:
            query_id = gt_item['id']
            gt_permissions = gt_item.get('answer', {})

            # Find corresponding prediction
            pred_item = next((p for p in predictions if p.get('id') == query_id), None)
            pred_permissions = pred_item.get('permission', {}) if pred_item else {}

            # For each data type in ground truth
            for data_type, gt_label in gt_permissions.items():
                # If no prediction for this data type, default to "No, never share" (0)
                if data_type in pred_permissions:
                    pred_label = pred_permissions[data_type].get('label', '')
                    pred_binary = 1 if "Yes" in pred_label else 0
                else:
                    pred_binary = 0  # Default: No, never share

                # Convert ground truth to binary
                gt_binary = 1 if "Yes" in gt_label else 0

                all_true.append(gt_binary)
                all_pred.append(pred_binary)

    if len(all_true) == 0:
        raise ValueError(f"No valid predictions found in {predictions_file}")

    metrics = calculate_metrics(all_true, all_pred, method_name, threshold=0.5)
    metrics['n_participants'] = n_participants

    return metrics
