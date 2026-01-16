#!/usr/bin/env python3
"""
Train head weights using Binary Cross-Entropy loss with L1 regularization.
This implements the logistic regression optimization proposed in the report.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


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


def train_logistic_regression(X_train, y_train, X_val, y_val, lambda_l1=0.01):
    """
    Train L1-regularized logistic regression.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        lambda_l1: L1 regularization strength (C = 1/lambda_l1)

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
        max_iter=1000,
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
    args = parser.parse_args()

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
    print(f"\n{'='*60}")
    print("Training Results")
    print('='*60)
    print(f"{'Lambda':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC-ROC':<10} {'Non-zero':<10} {'Sparsity':<10}")
    print('-'*90)

    best_model = None
    best_auc = 0
    best_lambda = None

    for lambda_l1 in args.lambda_l1:
        model, metrics = train_logistic_regression(
            X_train_scaled, y_train, X_val_scaled, y_val, lambda_l1
        )

        print(f"{lambda_l1:<10.4f} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f} {metrics['auc_roc']:<10.4f} "
              f"{metrics['num_nonzero_weights']:<10d} {metrics['sparsity']:<10.4f}")

        if metrics['auc_roc'] > best_auc:
            best_auc = metrics['auc_roc']
            best_model = model
            best_lambda = lambda_l1

    print(f"\nBest model: lambda={best_lambda}, AUC-ROC={best_auc:.4f}")

    # Analyze best model weights
    print(f"\n{'='*60}")
    print(f"Top 20 Heads by Learned Weight (lambda={best_lambda})")
    print('='*60)

    top_heads, all_heads = analyze_weights(best_model, num_layers, num_heads, top_k=20)

    print(f"{'Rank':<6} {'Layer':<8} {'Head':<8} {'Weight':<12}")
    print('-'*40)
    for i, h in enumerate(top_heads):
        print(f"{i+1:<6} {h['layer']:<8} {h['head']:<8} {h['weight']:<12.6f}")

    # Compare with CoRe heads
    compare_with_core_heads(top_heads, args.llm, temp=args.temp)

    # Save results
    output_dir = Path(__file__).parent.parent / 'head_data' / args.llm
    output_file = output_dir / f'bce_weights_lambda{best_lambda}_n{args.num_samples}.json'

    results = {
        'lambda_l1': best_lambda,
        'num_samples': args.num_samples,
        'metrics': {
            'auc_roc': best_auc,
        },
        'top_heads': [{'layer': h['layer'], 'head': h['head'], 'weight': float(h['weight'])}
                      for h in top_heads],
        'all_weights': {f"{h['layer']}-{h['head']}": float(h['weight']) for h in all_heads}
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_file}")


if __name__ == '__main__':
    main()
