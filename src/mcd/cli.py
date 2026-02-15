import argparse
import logging
import os
import json
import pandas as pd
import numpy as np
import random
import hashlib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from .io import load_labeled_tickets_csv, resolve_dataset_path
from .modeling.classifier import MahalanobisDriftDetector
from .config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Mahalanobis Concept Drift Detector CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Fit command
    fit_parser = subparsers.add_parser('fit', help='Fit the model')
    fit_parser.add_argument('--data', required=True, help='Path to CSV data file')
    fit_parser.add_argument('--label-column', default='queue', help='Column name for labels')
    fit_parser.add_argument('--text-columns', nargs='+', default=['subject', 'body'], help='Columns for text')
    fit_parser.add_argument('--model-file', required=True, help='Path to save model')
    fit_parser.add_argument('--threshold-quantile', type=float, default=0.99, help='Threshold quantile')
    fit_parser.add_argument('--min-cluster-size', type=int, default=10, help='Minimum cluster size')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict on text')
    predict_parser.add_argument('--model-file', required=True, help='Path to model file')
    predict_parser.add_argument('--text', required=True, help='Text to predict')

    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate model on dataset')
    eval_parser.add_argument('--data', required=True, help='Path to CSV or ZIP data file')
    eval_parser.add_argument('--label-column', default='queue', help='Column name for labels')
    eval_parser.add_argument('--text-columns', nargs='+', default=['subject', 'body'], help='Columns for text')
    eval_parser.add_argument('--train-cluster-frac', type=float, default=0.8, help='Fraction of clusters for training')
    eval_parser.add_argument('--threshold-quantile', type=float, default=0.99, help='Threshold quantile')
    eval_parser.add_argument('--min-cluster-size', type=int, default=10, help='Minimum cluster size')
    eval_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    eval_parser.add_argument('--out-dir', help='Output directory')
    eval_parser.add_argument('--auto-demo', action='store_true', help='Use demo defaults')

    args = parser.parse_args()

    # Handle auto-demo defaults
    if hasattr(args, 'auto_demo') and args.auto_demo:
        args.label_column = 'queue'
        args.text_columns = ['subject', 'body']
        args.threshold_quantile = 0.99
        args.min_cluster_size = 10
        args.train_cluster_frac = 0.8
        args.seed = 42
        if not args.out_dir:
            args.out_dir = 'reports/demo_run'

    # Resolve dataset path
    if hasattr(args, 'data'):
        args.data, args.selected_csv = resolve_dataset_path(args.data)

    if args.command == 'fit':
        # Load data
        texts, labels, label_mapping, _ = load_labeled_tickets_csv(args.data, args.label_column)
        
        # Fit model
        model = MahalanobisDriftDetector(threshold_quantile=args.threshold_quantile, min_cluster_size=args.min_cluster_size)
        model.fit(texts, labels)
        
        # Save model
        model.save(args.model_file)
        logger.info(f"Model saved to {args.model_file}")

    elif args.command == 'predict':
        # Load model
        model = MahalanobisDriftDetector.load(args.model_file)
        
        # Predict
        predicted_label, distance, threshold, is_drift = model.predict(args.text)
        print(f"Predicted label: {predicted_label}")
        print(f"Distance: {distance:.4f}")
        print(f"Threshold: {threshold:.4f}")
        print(f"Drift detected: {is_drift}")

    elif args.command == 'eval':
        run_eval(args)

    else:
        parser.print_help()

def run_eval(args):
    # Load data
    logger.info("Loading data...")
    texts, labels, label_to_index, index_to_label = load_labeled_tickets_csv(args.data, args.label_column)
    df = pd.read_csv(args.data)  # for additional columns like language
    df = df.dropna(subset=['subject', 'body', args.label_column])  # ensure same rows

    # Get unique clusters and filter by size
    cluster_sizes = {}
    for label in set(labels):
        size = labels.count(label)
        if size >= args.min_cluster_size:
            cluster_sizes[label] = size
        else:
            logger.warning(f"Skipping cluster {label} with size {size} < {args.min_cluster_size}")

    unique_clusters = list(cluster_sizes.keys())
    if len(unique_clusters) < 2:
        raise ValueError(f"Not enough clusters (>= {args.min_cluster_size}) for evaluation: {len(unique_clusters)}")

    # Shuffle clusters
    random.seed(args.seed)
    random.shuffle(unique_clusters)

    # Split clusters
    n_in = int(args.train_cluster_frac * len(unique_clusters))
    in_clusters = unique_clusters[:n_in]
    ood_clusters = unique_clusters[n_in:]

    logger.info(f"IN clusters: {len(in_clusters)}, OOD clusters: {len(ood_clusters)}")

    # Create indices
    in_indices = [i for i, label in enumerate(labels) if label in in_clusters]
    ood_indices = [i for i, label in enumerate(labels) if label in ood_clusters]

    # Stratified split IN into train/test (80/20 per cluster)
    in_labels = [labels[i] for i in in_indices]
    train_indices, test_in_indices = train_test_split(
        in_indices, test_size=0.2, stratify=in_labels, random_state=args.seed
    )

    # Train data
    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]

    # Fit model
    logger.info("Fitting model...")
    model = MahalanobisDriftDetector(threshold_quantile=args.threshold_quantile, min_cluster_size=args.min_cluster_size)
    model.fit(train_texts, train_labels)

    # Test data: test_in + test_ood
    test_indices = test_in_indices + ood_indices
    test_texts = [texts[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    test_true_drift = [0] * len(test_in_indices) + [1] * len(ood_indices)
    test_split = ['in'] * len(test_in_indices) + ['ood'] * len(ood_indices)

    # Predict
    logger.info("Evaluating...")
    batch_predictions = model.predict_batch(test_texts)
    predictions = []
    for pred_label, distance, threshold, is_drift in batch_predictions:
        score = distance - threshold
        predictions.append((pred_label, distance, threshold, is_drift, score))

    # Unpack predictions
    pred_labels, distances, thresholds, pred_drifts, scores = zip(*predictions)

    # Metrics for drift detection
    y_true_drift = test_true_drift
    y_pred_drift = [int(d) for d in pred_drifts]
    y_score = scores

    precision = precision_score(y_true_drift, y_pred_drift, zero_division=0)
    recall = recall_score(y_true_drift, y_pred_drift, zero_division=0)
    f1 = f1_score(y_true_drift, y_pred_drift, zero_division=0)
    accuracy_drift = accuracy_score(y_true_drift, y_pred_drift)
    try:
        roc_auc = roc_auc_score(y_true_drift, y_score)
    except ValueError as e:
        logger.warning(f"ROC-AUC failed: {e}")
        roc_auc = None

    cm = confusion_matrix(y_true_drift, y_pred_drift)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Metrics for classification (only on test_in)
    test_in_true_labels = [test_labels[i] for i in range(len(test_in_indices))]
    test_in_pred_labels = [pred_labels[i] for i in range(len(test_in_indices))]
    accuracy_class = accuracy_score(test_in_true_labels, test_in_pred_labels)

    # Prepare outputs
    os.makedirs(args.out_dir, exist_ok=True)

    # metrics.json
    metrics = {
        'config': {
            'data': args.data,
            'label_column': args.label_column,
            'text_columns': args.text_columns,
            'train_cluster_frac': args.train_cluster_frac,
            'threshold_quantile': args.threshold_quantile,
            'min_cluster_size': args.min_cluster_size,
            'seed': args.seed
        },
        'data_stats': {
            'total_samples': len(texts),
            'total_clusters': len(unique_clusters),
            'in_clusters': len(in_clusters),
            'ood_clusters': len(ood_clusters),
            'train_samples': len(train_indices),
            'test_in_samples': len(test_in_indices),
            'test_ood_samples': len(ood_indices),
            'cluster_sizes': {
                'min': min(cluster_sizes.values()),
                'max': max(cluster_sizes.values()),
                'mean': np.mean(list(cluster_sizes.values())),
                'median': np.median(list(cluster_sizes.values()))
            }
        },
        'drift_metrics': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy_drift,
            'roc_auc': roc_auc,
            'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}
        },
        'classification_metrics': {
            'accuracy': accuracy_class
        }
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # predictions.csv
    pred_df = pd.DataFrame({
        'truncated_text': [text[:100] + '...' if len(text) > 100 else text for text in test_texts],
        'text_len': [len(text) for text in test_texts],
        'text_hash': [hashlib.md5(text.encode('utf-8')).hexdigest() for text in test_texts],
        'true_label': test_labels,
        'predicted_label': pred_labels,
        'distance': distances,
        'threshold': thresholds,
        'score': scores,
        'drift_pred': y_pred_drift,
        'drift_true': y_true_drift,
        'split': test_split,
        'language': [df.iloc[i]['language'] if 'language' in df.columns else '' for i in test_indices]
    })
    pred_df.to_csv(os.path.join(args.out_dir, 'predictions.csv'), index=False)

    # splits.json
    splits = {
        'in_clusters': in_clusters,
        'ood_clusters': ood_clusters
    }
    with open(os.path.join(args.out_dir, 'splits.json'), 'w') as f:
        json.dump(splits, f, indent=2)

    # README.md
    readme = f"""# Evaluation Run

Selected CSV: {getattr(args, 'selected_csv', 'N/A')}

Number of clusters: {len(unique_clusters)}

IN clusters: {len(in_clusters)}

OOD clusters: {len(ood_clusters)}

Drift Metrics:
- Precision: {precision:.4f}
- Recall: {recall:.4f}
- F1: {f1:.4f}
- ROC-AUC: {roc_auc if roc_auc is not None else 'N/A'}

Classification Accuracy on IN: {accuracy_class:.4f}
"""
    with open(os.path.join(args.out_dir, 'README.md'), 'w') as f:
        f.write(readme)

    logger.info(f"Evaluation results saved to {args.out_dir}")

if __name__ == '__main__':
    main() 
