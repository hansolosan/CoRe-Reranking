#!/usr/bin/env python3
"""
Train head weights using Binary Cross-Entropy loss with L1 regularization.
This implements the logistic regression optimization proposed in the report.
"""

import json
import argparse
import sys
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def get_unique_filepath(base_path):
    """Return a unique filepath by adding numeric suffix if file exists."""
    path = Path(base_path)
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    counter = 1
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def load_features(llm_name, num_samples):
    """Load extracted attention features."""
    feature_file = Path(__file__).parent.parent / 'head_data' / llm_name / f'attention_features_n{num_samples}.npz'

    if not feature_file.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feature_file}\n"
            f"Run extract_head_features.py first:\n"
            f"  python scripts/extract_head_features.py --llm {llm_name} --max_samples {num_samples}"
        )

    data = np.load(feature_file)
    return data['features'], data['labels']


def get_head_info(llm_name):
    """Get number of layers and heads for a model."""
    model_configs = {
        'mistral': (32, 32),   # 32 layers, 32 heads
        'llama': (32, 32),     # 32 layers, 32 heads
        'phi': (40, 40),       # 40 layers, 40 heads
        'granite': (40, 32),   # 40 layers, 32 heads
    }
    return model_configs.get(llm_name, (32, 32))


def train_logistic_regression(X_train, y_train, X_val, y_val, lambda_l1=0.01, max_iter=1000):
    """
    Train L1-regularized logistic regression.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        lambda_l1: L1 regularization strength (C = 1/lambda_l1)
        max_iter: Maximum iterations for solver

    Returns:
        model: Trained model
        metrics: Dictionary of evaluation metrics
    """
    # sklearn uses C = 1/lambda
    C = 1.0 / lambda_l1 if lambda_l1 > 0 else 1e6

    model = LogisticRegression(
        penalty='l1',
        C=C,
        solver='saga',
        max_iter=max_iter,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance (1 pos vs 49 neg)
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'auc_roc': roc_auc_score(y_val, y_prob),
        'num_nonzero_weights': np.sum(np.abs(model.coef_[0]) > 1e-6),
        'sparsity': 1.0 - np.sum(np.abs(model.coef_[0]) > 1e-6) / len(model.coef_[0])
    }

    return model, metrics


def analyze_weights(model, num_layers, num_heads, top_k=20):
    """Analyze learned head weights."""
    weights = model.coef_[0]

    # Create head index mapping
    head_weights = []
    for layer in range(num_layers):
        for head in range(num_heads):
            idx = layer * num_heads + head
            if idx < len(weights):
                head_weights.append({
                    'layer': layer,
                    'head': head,
                    'weight': weights[idx],
                    'abs_weight': abs(weights[idx])
                })

    # Sort by absolute weight
    head_weights.sort(key=lambda x: x['abs_weight'], reverse=True)

    return head_weights[:top_k], head_weights


def compare_with_core_heads(learned_heads, llm_name, temp=0.001, prune=0.0):
    """Compare learned heads with original CoRe heads."""
    core_file = Path(__file__).parent.parent / 'head_data' / llm_name / f'core_temp{temp}_prune{prune}.json'

    if not core_file.exists():
        print(f"CoRe head file not found: {core_file}")
        return

    with open(core_file, 'r') as f:
        core_scores = json.load(f)

    # Get top 8 CoRe heads
    core_heads = []
    for head_key, scores in core_scores.items():
        layer, head = map(int, head_key.split('-'))
        avg_score = np.mean(scores) if isinstance(scores, list) else scores
        core_heads.append({'layer': layer, 'head': head, 'score': avg_score})

    core_heads.sort(key=lambda x: x['score'], reverse=True)
    core_top8 = set((h['layer'], h['head']) for h in core_heads[:8])

    # Get top 8 learned heads
    learned_top8 = set((h['layer'], h['head']) for h in learned_heads[:8])

    overlap = len(core_top8 & learned_top8)

    print(f"\n{'='*60}")
    print("Comparison with CoRe Heads")
    print('='*60)
    print(f"Top 8 CoRe heads: {sorted(core_top8)}")
    print(f"Top 8 Learned heads: {sorted(learned_top8)}")
    print(f"Overlap: {overlap}/8 heads")

    return core_top8, learned_top8


def convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    return obj


def save_results(output_dir, lambda_l1, num_samples, metrics, top_heads, all_heads, command):
    """Save results for a single lambda value."""
    base_file = output_dir / f'bce_weights_lambda{lambda_l1}_n{num_samples}.json'
    output_file = get_unique_filepath(base_file)

    results = {
        'command': command,
        'timestamp': datetime.now().isoformat(),
        'lambda_l1': float(lambda_l1),
        'num_samples': int(num_samples),
        'metrics': convert_to_native(metrics),
        'top_heads': [{'layer': int(h['layer']), 'head': int(h['head']), 'weight': float(h['weight'])}
                      for h in top_heads],
        'all_weights': {f"{h['layer']}-{h['head']}": float(h['weight']) for h in all_heads}
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    return output_file


def train_single_lambda(lambda_l1, X_train, y_train, X_val, y_val, num_layers, num_heads, max_iter=1000):
    """
    Train a single model for one lambda value. Designed for parallel execution.

    Returns:
        dict with lambda_l1, metrics, top_heads, all_heads
    """
    model, metrics = train_logistic_regression(
        X_train, y_train, X_val, y_val, lambda_l1, max_iter=max_iter
    )
    top_heads, all_heads = analyze_weights(model, num_layers, num_heads, top_k=20)

    return {
        'lambda_l1': lambda_l1,
        'model': model,
        'metrics': metrics,
        'top_heads': top_heads,
        'all_heads': all_heads
    }


def main():
    parser = argparse.ArgumentParser(description='Train head weights with BCE + L1')
    parser.add_argument('--llm', type=str, default='mistral',
                        choices=['mistral', 'llama', 'phi', 'granite'])
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples used for feature extraction')
    parser.add_argument('--lambda_l1', type=float, nargs='+',
                        default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                        help='L1 regularization strengths to try')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--temp', type=float, default=0.001,
                        help='Temperature used for CoRe head detection (for comparison)')
    parser.add_argument('--save_all', action='store_true',
                        help='Save results for all lambda values (default: only best)')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs (-1 for all CPUs, default: 1)')
    parser.add_argument('--max_iter', type=int, default=1000,
                        help='Maximum iterations for SAGA solver (default: 1000)')
    args = parser.parse_args()

    # Capture the command used to run this script
    command = ' '.join(sys.argv)

    print(f"Training head weights for {args.llm}")
    print(f"L1 regularization values: {args.lambda_l1}")

    # Load features
    print(f"\nLoading features...")
    try:
        X, y = load_features(args.llm, args.num_samples)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class balance: {y.sum()} positive, {len(y) - y.sum()} negative")

    # Get model config
    num_layers, num_heads = get_head_info(args.llm)
    print(f"Model config: {num_layers} layers, {num_heads} heads")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(y_train)}, Val size: {len(y_val)}")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train with different regularization strengths
    output_dir = Path(__file__).parent.parent / 'head_data' / args.llm
    output_dir.mkdir(parents=True, exist_ok=True)

    n_jobs = args.n_jobs if args.n_jobs != 0 else 1
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    print(f"\n{'='*60}")
    print(f"Training Results (n_jobs={n_jobs}, max_iter={args.max_iter})")
    print('='*60)

    if n_jobs > 1 and len(args.lambda_l1) > 1:
        # Parallel training
        print(f"Training {len(args.lambda_l1)} models in parallel...")
        all_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(train_single_lambda)(
                lambda_l1, X_train_scaled, y_train, X_val_scaled, y_val,
                num_layers, num_heads, args.max_iter
            )
            for lambda_l1 in args.lambda_l1
        )
        # Sort by lambda for consistent ordering
        all_results.sort(key=lambda x: x['lambda_l1'])

        # Print results table after parallel completion
        print(f"\n{'Lambda':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC-ROC':<10} {'Non-zero':<10} {'Sparsity':<10}")
        print('-'*90)
        for result in all_results:
            metrics = result['metrics']
            print(f"{result['lambda_l1']:<10.4f} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f} {metrics['auc_roc']:<10.4f} "
                  f"{metrics['num_nonzero_weights']:<10d} {metrics['sparsity']:<10.4f}")
    else:
        # Sequential training (original behavior)
        print(f"{'Lambda':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC-ROC':<10} {'Non-zero':<10} {'Sparsity':<10}")
        print('-'*90)
        all_results = []
        for lambda_l1 in args.lambda_l1:
            result = train_single_lambda(
                lambda_l1, X_train_scaled, y_train, X_val_scaled, y_val,
                num_layers, num_heads, args.max_iter
            )
            all_results.append(result)

            # Print immediately in sequential mode
            metrics = result['metrics']
            print(f"{lambda_l1:<10.4f} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f} {metrics['auc_roc']:<10.4f} "
                  f"{metrics['num_nonzero_weights']:<10d} {metrics['sparsity']:<10.4f}", flush=True)

    # Find best model and save results
    best_result = max(all_results, key=lambda x: x['metrics']['auc_roc'])
    best_lambda = best_result['lambda_l1']
    best_auc = best_result['metrics']['auc_roc']

    # Save results
    for result in all_results:
        if args.save_all or result['lambda_l1'] == best_lambda:
            output_file = save_results(output_dir, result['lambda_l1'], args.num_samples,
                                       result['metrics'], result['top_heads'],
                                       result['all_heads'], command)
            if args.save_all:
                print(f"  Saved lambda={result['lambda_l1']} to {output_file}")

    print(f"\nBest model: lambda={best_lambda}, AUC-ROC={best_auc:.4f}")

    # Get best model results from stored data
    best_result = next(r for r in all_results if r['lambda_l1'] == best_lambda)
    top_heads = best_result['top_heads']
    all_heads = best_result['all_heads']

    # Analyze best model weights
    print(f"\n{'='*60}")
    print(f"Top 20 Heads by Learned Weight (lambda={best_lambda})")
    print('='*60)

    print(f"{'Rank':<6} {'Layer':<8} {'Head':<8} {'Weight':<12}")
    print('-'*40)
    for i, h in enumerate(top_heads):
        print(f"{i+1:<6} {h['layer']:<8} {h['head']:<8} {h['weight']:<12.6f}")

    # Compare with CoRe heads
    compare_with_core_heads(top_heads, args.llm, temp=args.temp)

    if not args.save_all:
        print(f"\nSaved best model to {output_dir}/")
    else:
        print(f"\nAll {len(args.lambda_l1)} models saved to {output_dir}/")


if __name__ == '__main__':
    main()
