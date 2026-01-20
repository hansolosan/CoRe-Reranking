#!/usr/bin/env python3
"""
Compare different head selection methods:
- CoRe greedy selection with equal weights
- CoRe heads with BCE-learned weights
- Full BCE model with all heads
- BCE model top-k heads by absolute weight
"""

import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score


def load_features(llm_name, num_samples):
    """Load extracted attention features."""
    feature_file = Path(__file__).parent.parent / 'head_data' / llm_name / f'attention_features_n{num_samples}.npz'
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    data = np.load(feature_file)
    return data['features'], data['labels']


def load_core_heads(llm_name, temp=0.001, prune=0.0, top_k=8):
    """Load top-k CoRe heads."""
    core_file = Path(__file__).parent.parent / 'head_data' / llm_name / f'core_temp{temp}_prune{prune}.json'
    if not core_file.exists():
        raise FileNotFoundError(f"CoRe file not found: {core_file}")

    with open(core_file, 'r') as f:
        core_scores = json.load(f)

    heads = []
    for key, scores in core_scores.items():
        layer, head = map(int, key.split('-'))
        avg_score = np.mean(scores) if isinstance(scores, list) else scores
        heads.append((layer, head, avg_score))

    heads.sort(key=lambda x: x[2], reverse=True)
    return [(l, h) for l, h, s in heads[:top_k]]


def load_bce_weights(llm_name, lambda_l1, num_samples):
    """Load BCE-learned weights."""
    bce_file = Path(__file__).parent.parent / 'head_data' / llm_name / f'bce_weights_lambda{lambda_l1}_n{num_samples}.json'
    if not bce_file.exists():
        raise FileNotFoundError(f"BCE weights file not found: {bce_file}")

    with open(bce_file, 'r') as f:
        return json.load(f)


def get_head_index(layer, head, num_heads=32):
    """Convert (layer, head) to feature index."""
    return layer * num_heads + head


def evaluate_equal_weights(X, y, heads, num_heads=32):
    """Evaluate using equal weights on selected heads."""
    indices = [get_head_index(l, h, num_heads) for l, h in heads]
    scores = X[:, indices].sum(axis=1)
    return roc_auc_score(y, scores)


def evaluate_learned_weights(X, y, all_weights, num_heads=32):
    """Evaluate using all learned weights."""
    num_layers = X.shape[1] // num_heads
    weights = np.zeros(X.shape[1])
    for key, w in all_weights.items():
        layer, head = map(int, key.split('-'))
        idx = get_head_index(layer, head, num_heads)
        if idx < len(weights):
            weights[idx] = w
    scores = (X * weights).sum(axis=1)
    return roc_auc_score(y, scores)


def evaluate_topk_learned(X, y, all_weights, top_k, num_heads=32):
    """Evaluate using top-k heads by absolute learned weight."""
    # Sort heads by absolute weight
    head_weights = []
    for key, w in all_weights.items():
        layer, head = map(int, key.split('-'))
        head_weights.append((layer, head, w, abs(w)))
    head_weights.sort(key=lambda x: x[3], reverse=True)

    # Get top-k
    top_heads = head_weights[:top_k]
    indices = [get_head_index(l, h, num_heads) for l, h, w, aw in top_heads]
    weights = np.array([w for l, h, w, aw in top_heads])

    scores = (X[:, indices] * weights).sum(axis=1)
    return roc_auc_score(y, scores), [(l, h, w) for l, h, w, aw in top_heads]


def main():
    parser = argparse.ArgumentParser(description='Compare head selection methods')
    parser.add_argument('--llm', type=str, default='mistral',
                        choices=['mistral', 'llama', 'phi', 'granite'])
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--lambda_l1', type=float, default=0.1)
    parser.add_argument('--top_k', type=int, nargs='+', default=[8, 16, 32, 64],
                        help='Number of top heads to evaluate')
    parser.add_argument('--temp', type=float, default=0.001)
    args = parser.parse_args()

    # Model configs
    num_heads_per_layer = {
        'mistral': 32, 'llama': 32, 'phi': 40, 'granite': 32
    }
    num_heads = num_heads_per_layer[args.llm]

    print(f"Comparing head selection methods for {args.llm}")
    print(f"=" * 70)

    # Load data
    print(f"\nLoading features (n={args.num_samples})...")
    X, y = load_features(args.llm, args.num_samples)
    print(f"Features shape: {X.shape}, Labels: {y.sum()} pos / {len(y) - y.sum()} neg")

    # Load CoRe heads
    print(f"\nLoading CoRe heads...")
    core_heads_8 = load_core_heads(args.llm, args.temp, top_k=8)
    print(f"Top 8 CoRe heads: {core_heads_8}")

    # Load BCE weights
    print(f"\nLoading BCE weights (lambda={args.lambda_l1})...")
    try:
        bce_data = load_bce_weights(args.llm, args.lambda_l1, args.num_samples)
        all_weights = bce_data['all_weights']
        nonzero = sum(1 for w in all_weights.values() if abs(w) > 1e-6)
        print(f"Non-zero weights: {nonzero}/{len(all_weights)}")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        all_weights = None

    # Evaluate methods
    print(f"\n{'Method':<45} {'AUC-ROC':<10} {'Heads':<10}")
    print("-" * 70)

    # CoRe with equal weights
    for k in args.top_k:
        core_heads = load_core_heads(args.llm, args.temp, top_k=k)
        auc = evaluate_equal_weights(X, y, core_heads, num_heads)
        print(f"CoRe Top-{k} (equal weights){'':<20} {auc:<10.6f} {k:<10}")

    if all_weights:
        print("-" * 70)

        # Full BCE model
        auc_full = evaluate_learned_weights(X, y, all_weights, num_heads)
        print(f"BCE Full (lambda={args.lambda_l1}){'':<20} {auc_full:<10.6f} {nonzero:<10}")

        # BCE top-k by absolute weight
        for k in args.top_k:
            auc_topk, top_heads = evaluate_topk_learned(X, y, all_weights, k, num_heads)
            print(f"BCE Top-{k} (learned weights){'':<18} {auc_topk:<10.6f} {k:<10}")

        # CoRe heads with BCE weights
        print("-" * 70)
        core_indices = [get_head_index(l, h, num_heads) for l, h in core_heads_8]
        core_bce_weights = np.array([all_weights.get(f"{l}-{h}", 0) for l, h in core_heads_8])
        scores = (X[:, core_indices] * core_bce_weights).sum(axis=1)
        auc_core_bce = roc_auc_score(y, scores)
        print(f"CoRe Top-8 (BCE weights){'':<22} {auc_core_bce:<10.6f} {8:<10}")

    # Head overlap analysis
    if all_weights:
        print(f"\n{'='*70}")
        print("Head Overlap Analysis (Top 8)")
        print("=" * 70)

        bce_top8 = []
        for key, w in sorted(all_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:8]:
            layer, head = map(int, key.split('-'))
            bce_top8.append((layer, head))

        core_set = set(core_heads_8)
        bce_set = set(bce_top8)
        overlap = core_set & bce_set

        print(f"CoRe Top 8: {sorted(core_heads_8)}")
        print(f"BCE Top 8:  {sorted(bce_top8)}")
        print(f"Overlap:    {sorted(overlap)} ({len(overlap)}/8 heads)")


if __name__ == '__main__':
    main()
