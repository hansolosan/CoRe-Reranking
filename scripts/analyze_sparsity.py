#!/usr/bin/env python3
"""
Analyze sparsity and performance trade-offs across different L1 regularization strengths.
Helps find the optimal lambda that achieves good sparsity while maintaining performance.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_features(llm_name, num_samples):
    """Load extracted attention features."""
    feature_file = Path(__file__).parent.parent / 'head_data' / llm_name / f'attention_features_n{num_samples}.npz'
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    data = np.load(feature_file)
    return data['features'], data['labels']


def find_bce_weight_files(llm_name, num_samples):
    """Find all BCE weight files for different lambda values."""
    head_dir = Path(__file__).parent.parent / 'head_data' / llm_name
    files = list(head_dir.glob(f'bce_weights_lambda*_n{num_samples}.json'))

    results = []
    for f in files:
        # Extract lambda from filename
        name = f.stem
        lambda_str = name.split('lambda')[1].split('_')[0]
        try:
            lambda_val = float(lambda_str)
            results.append((lambda_val, f))
        except ValueError:
            continue

    return sorted(results, key=lambda x: x[0])


def analyze_weights(weight_file):
    """Analyze weight distribution from a BCE weights file."""
    with open(weight_file, 'r') as f:
        data = json.load(f)

    weights = np.array(list(data['all_weights'].values()))

    nonzero = np.sum(np.abs(weights) > 1e-6)
    sparsity = 1.0 - nonzero / len(weights)

    # Get top heads
    head_weights = []
    for key, w in data['all_weights'].items():
        layer, head = map(int, key.split('-'))
        head_weights.append((layer, head, w, abs(w)))
    head_weights.sort(key=lambda x: x[3], reverse=True)

    return {
        'lambda': data['lambda_l1'],
        'nonzero': nonzero,
        'total': len(weights),
        'sparsity': sparsity,
        'auc_roc': data['metrics'].get('auc_roc', None),
        'top_heads': head_weights[:20],
        'all_weights': weights
    }


def evaluate_with_weights(X, y, all_weights, num_heads=32):
    """Evaluate AUC-ROC using learned weights."""
    weights = np.zeros(X.shape[1])
    for key, w in all_weights.items():
        layer, head = map(int, key.split('-'))
        idx = layer * num_heads + head
        if idx < len(weights):
            weights[idx] = w
    scores = (X * weights).sum(axis=1)
    return roc_auc_score(y, scores)


def main():
    parser = argparse.ArgumentParser(description='Analyze sparsity across lambda values')
    parser.add_argument('--llm', type=str, default='mistral',
                        choices=['mistral', 'llama', 'phi', 'granite'])
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    args = parser.parse_args()

    num_heads_per_layer = {'mistral': 32, 'llama': 32, 'phi': 40, 'granite': 32}
    num_heads = num_heads_per_layer[args.llm]

    print(f"Analyzing sparsity for {args.llm}")
    print("=" * 80)

    # Find all weight files
    weight_files = find_bce_weight_files(args.llm, args.num_samples)

    if not weight_files:
        print(f"No BCE weight files found for {args.llm} with n={args.num_samples}")
        print("Run train_head_weights_bce.py first with various --lambda_l1 values")
        return

    print(f"Found {len(weight_files)} weight files\n")

    # Load features for evaluation
    try:
        X, y = load_features(args.llm, args.num_samples)
    except FileNotFoundError:
        X, y = None, None
        print("Warning: Feature file not found, skipping AUC evaluation\n")

    # Analyze each
    results = []
    print(f"{'Lambda':<12} {'Non-zero':<12} {'Sparsity':<12} {'AUC-ROC':<12} {'Top 5 Heads'}")
    print("-" * 80)

    for lambda_val, weight_file in weight_files:
        analysis = analyze_weights(weight_file)

        # Recompute AUC if we have features
        if X is not None:
            with open(weight_file, 'r') as f:
                data = json.load(f)
            auc = evaluate_with_weights(X, y, data['all_weights'], num_heads)
        else:
            auc = analysis['auc_roc']

        top5 = [(l, h) for l, h, w, aw in analysis['top_heads'][:5]]

        print(f"{lambda_val:<12.4f} {analysis['nonzero']:<12} {analysis['sparsity']:<12.4f} "
              f"{auc:<12.6f} {top5}")

        results.append({
            'lambda': lambda_val,
            'nonzero': analysis['nonzero'],
            'sparsity': analysis['sparsity'],
            'auc': auc,
            'top_heads': analysis['top_heads']
        })

    # Find optimal lambda (highest AUC with reasonable sparsity)
    print("\n" + "=" * 80)
    print("Recommendations:")
    print("=" * 80)

    # Sort by sparsity
    sparse_results = [r for r in results if r['sparsity'] > 0.9]
    if sparse_results:
        best_sparse = max(sparse_results, key=lambda x: x['auc'])
        print(f"Best sparse model (>90% sparsity): lambda={best_sparse['lambda']}, "
              f"AUC={best_sparse['auc']:.4f}, heads={best_sparse['nonzero']}")

    # Best overall
    best_overall = max(results, key=lambda x: x['auc'])
    print(f"Best overall AUC: lambda={best_overall['lambda']}, "
          f"AUC={best_overall['auc']:.4f}, heads={best_overall['nonzero']}")

    # Generate plot if requested
    if args.plot and len(results) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        lambdas = [r['lambda'] for r in results]
        aucs = [r['auc'] for r in results]
        nonzeros = [r['nonzero'] for r in results]

        ax1.semilogx(lambdas, aucs, 'b-o', linewidth=2, markersize=8)
        ax1.set_xlabel('Lambda (L1 regularization)')
        ax1.set_ylabel('AUC-ROC')
        ax1.set_title('Performance vs Regularization')
        ax1.grid(True, alpha=0.3)

        ax2.semilogx(lambdas, nonzeros, 'r-o', linewidth=2, markersize=8)
        ax2.set_xlabel('Lambda (L1 regularization)')
        ax2.set_ylabel('Number of Non-zero Heads')
        ax2.set_title('Sparsity vs Regularization')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = Path(__file__).parent.parent / 'head_data' / args.llm / f'sparsity_analysis_n{args.num_samples}.png'
        plt.savefig(plot_file, dpi=150)
        print(f"\nPlot saved to {plot_file}")


if __name__ == '__main__':
    main()
