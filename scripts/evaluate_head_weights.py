#!/usr/bin/env python3
"""
Evaluate head weights on ranking metrics (NDCG@k, Precision@k, MAP@k).

This script loads attention features and head weights, computes document scores,
and evaluates ranking performance on the head detection data.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_features(feature_file=None, llm_name=None, num_samples=None):
    """Load extracted attention features.

    Args:
        feature_file: Path to feature file (if provided, llm_name and num_samples are ignored)
        llm_name: LLM name for default path construction
        num_samples: Number of samples for default path construction

    Returns:
        features, labels arrays
    """
    if feature_file is not None:
        path = Path(feature_file)
    else:
        path = Path(__file__).parent.parent / 'head_data' / llm_name / f'attention_features_n{num_samples}.npz'

    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    data = np.load(path, allow_pickle=True)

    # Handle different feature file formats
    features = data['features']
    labels = data['labels']

    # Get docs_per_query if available
    docs_per_query = None
    if 'docs_per_query' in data:
        docs_per_query = data['docs_per_query']

    return features, labels, docs_per_query


def load_head_weights(weight_file, num_heads_per_layer=32, num_layers=32):
    """
    Load head weights from a file (BCE weights JSON or CoRe scores JSON).

    Returns:
        weights: numpy array of shape (num_layers * num_heads_per_layer,)
        metadata: dict with file info
    """
    with open(weight_file, 'r') as f:
        data = json.load(f)

    total_heads = num_layers * num_heads_per_layer
    weights = np.zeros(total_heads)

    # Check if it's a BCE weights file or CoRe scores file
    if 'all_weights' in data:
        # BCE weights format: {"all_weights": {"layer-head": weight, ...}}
        for key, w in data['all_weights'].items():
            layer, head = map(int, key.split('-'))
            idx = layer * num_heads_per_layer + head
            if idx < total_heads:
                weights[idx] = w
        metadata = {
            'type': 'bce',
            'lambda_l1': data.get('lambda_l1'),
            'num_samples': data.get('num_samples'),
            'nonzero': sum(1 for w in weights if abs(w) > 1e-6)
        }
    else:
        # CoRe scores format: {"layer-head": [scores...] or score, ...}
        for key, scores in data.items():
            layer, head = map(int, key.split('-'))
            idx = layer * num_heads_per_layer + head
            if idx < total_heads:
                # CoRe uses average score as weight (equal weighting of top-k)
                weights[idx] = np.mean(scores) if isinstance(scores, list) else scores
        metadata = {
            'type': 'core',
            'nonzero': sum(1 for w in weights if abs(w) > 1e-6)
        }

    return weights, metadata


def get_top_k_heads(weights, k):
    """Get indices of top-k heads by absolute weight."""
    abs_weights = np.abs(weights)
    top_indices = np.argsort(abs_weights)[-k:][::-1]
    return top_indices


def compute_scores(features, weights, top_k=None):
    """
    Compute document scores using head weights.

    Args:
        features: (num_docs, num_heads) attention features
        weights: (num_heads,) head weights
        top_k: if set, only use top-k heads by absolute weight

    Returns:
        scores: (num_docs,) document scores
    """
    if top_k is not None and top_k < len(weights):
        # Zero out all but top-k heads
        top_indices = get_top_k_heads(weights, top_k)
        mask = np.zeros_like(weights)
        mask[top_indices] = 1
        weights = weights * mask

    return features @ weights


def dcg_at_k(relevances, k):
    """Compute DCG@k."""
    relevances = np.asarray(relevances)[:k]
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return np.sum(relevances / discounts)


def ndcg_at_k(relevances, k):
    """Compute NDCG@k."""
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def precision_at_k(relevances, k):
    """Compute Precision@k."""
    relevances = np.asarray(relevances)[:k]
    if relevances.size == 0:
        return 0.0
    return np.sum(relevances > 0) / k


def average_precision(relevances):
    """Compute Average Precision."""
    relevances = np.asarray(relevances)
    if np.sum(relevances) == 0:
        return 0.0

    precisions = []
    num_relevant = 0
    for i, rel in enumerate(relevances):
        if rel > 0:
            num_relevant += 1
            precisions.append(num_relevant / (i + 1))

    if len(precisions) == 0:
        return 0.0
    return np.mean(precisions)


def reciprocal_rank(relevances):
    """Compute Reciprocal Rank (position of first relevant doc)."""
    relevances = np.asarray(relevances)
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_ranking(features, labels, weights, docs_per_query=50, top_k_heads=None, ks=[1, 5, 10]):
    """
    Evaluate ranking metrics.

    Args:
        features: (num_docs, num_heads) attention features
        labels: (num_docs,) binary relevance labels
        weights: (num_heads,) head weights
        docs_per_query: number of documents per query (int) or array of docs per query
        top_k_heads: if set, only use top-k heads
        ks: list of k values for @k metrics

    Returns:
        metrics: dict of metric name -> value
    """
    # Compute scores
    scores = compute_scores(features, weights, top_k=top_k_heads)

    # Compute metrics per query
    all_ndcg = {k: [] for k in ks}
    all_precision = {k: [] for k in ks}
    all_ap = []
    all_rr = []

    # Handle variable or fixed docs_per_query
    if isinstance(docs_per_query, (list, np.ndarray)):
        # Variable docs per query
        num_queries = len(docs_per_query)
        doc_offset = 0
        for q in range(num_queries):
            n_docs = docs_per_query[q]
            q_scores = scores[doc_offset:doc_offset + n_docs]
            q_labels = labels[doc_offset:doc_offset + n_docs]
            doc_offset += n_docs

            # Rank by score (descending)
            ranking = np.argsort(-q_scores)
            ranked_labels = q_labels[ranking]

            # Compute metrics
            for k in ks:
                all_ndcg[k].append(ndcg_at_k(ranked_labels, k))
                all_precision[k].append(precision_at_k(ranked_labels, k))

            all_ap.append(average_precision(ranked_labels))
            all_rr.append(reciprocal_rank(ranked_labels))
    else:
        # Fixed docs per query
        num_queries = len(labels) // docs_per_query
        for q in range(num_queries):
            start = q * docs_per_query
            end = start + docs_per_query

            q_scores = scores[start:end]
            q_labels = labels[start:end]

            # Rank by score (descending)
            ranking = np.argsort(-q_scores)
            ranked_labels = q_labels[ranking]

            # Compute metrics
            for k in ks:
                all_ndcg[k].append(ndcg_at_k(ranked_labels, k))
                all_precision[k].append(precision_at_k(ranked_labels, k))

            all_ap.append(average_precision(ranked_labels))
            all_rr.append(reciprocal_rank(ranked_labels))

    # Aggregate
    metrics = {}
    for k in ks:
        metrics[f'NDCG@{k}'] = np.mean(all_ndcg[k])
        metrics[f'P@{k}'] = np.mean(all_precision[k])
    metrics['MAP'] = np.mean(all_ap)
    metrics['MRR'] = np.mean(all_rr)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate head weights on ranking metrics')
    parser.add_argument('--llm', type=str, default='mistral',
                        choices=['mistral', 'llama', 'phi', 'granite'])
    parser.add_argument('--weight_file', type=str, required=True,
                        help='Path to head weights file (BCE JSON or CoRe JSON)')
    parser.add_argument('--feature_file', '-f', type=str, default=None,
                        help='Path to feature file (.npz). Default: head_data/{llm}/attention_features_n{num_samples}.npz')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples in feature file (used for default path)')
    parser.add_argument('--top_k_heads', type=int, nargs='+', default=None,
                        help='Evaluate with only top-k heads (can specify multiple)')
    parser.add_argument('--docs_per_query', type=int, default=None,
                        help='Number of documents per query (auto-detected from npz if available)')
    parser.add_argument('--ks', type=int, nargs='+', default=[1, 5, 10],
                        help='k values for @k metrics')
    parser.add_argument('--compare_equal', action='store_true',
                        help='Also compare with equal weights on same heads')
    args = parser.parse_args()

    # Model configs
    model_configs = {
        'mistral': (32, 32),
        'llama': (32, 32),
        'phi': (40, 40),
        'granite': (40, 32),
    }
    num_layers, num_heads = model_configs[args.llm]

    print(f"Evaluating head weights for {args.llm}")
    print(f"Weight file: {args.weight_file}")
    print("=" * 70)

    # Load features
    if args.feature_file:
        print(f"\nLoading features from {args.feature_file}...")
    else:
        print(f"\nLoading features (n={args.num_samples})...")

    X, y, docs_per_query_arr = load_features(
        feature_file=args.feature_file,
        llm_name=args.llm,
        num_samples=args.num_samples
    )

    # Determine docs_per_query
    if docs_per_query_arr is not None:
        # Variable docs per query - use the array
        docs_per_query = docs_per_query_arr
        num_queries = len(docs_per_query)
        print(f"Features shape: {X.shape}")
        print(f"Number of queries: {num_queries}")
        print(f"Docs per query: variable (min={min(docs_per_query)}, max={max(docs_per_query)}, avg={np.mean(docs_per_query):.1f})")
    elif args.docs_per_query is not None:
        docs_per_query = args.docs_per_query
        num_queries = len(y) // docs_per_query
        print(f"Features shape: {X.shape}")
        print(f"Number of queries: {num_queries}")
        print(f"Docs per query: {docs_per_query}")
    else:
        # Default to 50
        docs_per_query = 50
        num_queries = len(y) // docs_per_query
        print(f"Features shape: {X.shape}")
        print(f"Number of queries: {num_queries}")
        print(f"Docs per query: {docs_per_query} (default)")

    print(f"Positive docs: {(y == 1).sum()} ({100*(y == 1).mean():.1f}%)")

    # Load weights
    print(f"\nLoading weights...")
    weights, metadata = load_head_weights(args.weight_file, num_heads, num_layers)
    print(f"Weight type: {metadata['type']}")
    print(f"Non-zero heads: {metadata['nonzero']}/{len(weights)}")
    if metadata.get('lambda_l1'):
        print(f"Lambda L1: {metadata['lambda_l1']}")

    # Determine top_k_heads to evaluate
    if args.top_k_heads is None:
        top_k_list = [None]  # Use all heads with non-zero weights
    else:
        top_k_list = args.top_k_heads

    # Evaluate
    print(f"\n{'='*70}")
    print("Ranking Metrics")
    print("=" * 70)

    results = []
    for top_k in top_k_list:
        label = f"top-{top_k}" if top_k else "all"

        metrics = evaluate_ranking(
            X, y, weights,
            docs_per_query=docs_per_query,
            top_k_heads=top_k,
            ks=args.ks
        )

        results.append({
            'config': label,
            'weights': 'learned',
            'metrics': metrics
        })

        # Compare with equal weights if requested
        if args.compare_equal and top_k is not None:
            # Create equal weights for top-k heads
            equal_weights = np.zeros_like(weights)
            top_indices = get_top_k_heads(weights, top_k)
            equal_weights[top_indices] = 1.0 / top_k

            eq_metrics = evaluate_ranking(
                X, y, equal_weights,
                docs_per_query=docs_per_query,
                top_k_heads=None,  # Already masked
                ks=args.ks
            )

            results.append({
                'config': label,
                'weights': 'equal',
                'metrics': eq_metrics
            })

    # Print results table
    metric_names = [f'NDCG@{k}' for k in args.ks] + [f'P@{k}' for k in args.ks] + ['MAP', 'MRR']

    # Header
    header = f"{'Config':<12} {'Weights':<10}"
    for name in metric_names:
        header += f" {name:<10}"
    print(header)
    print("-" * len(header))

    # Results
    for r in results:
        row = f"{r['config']:<12} {r['weights']:<10}"
        for name in metric_names:
            row += f" {r['metrics'][name]:<10.4f}"
        print(row)

    # Save results
    output_file = Path(args.weight_file).with_suffix('.metrics.json')
    output_data = {
        'weight_file': str(args.weight_file),
        'llm': args.llm,
        'num_samples': args.num_samples,
        'num_queries': num_queries,
        'results': results
    }
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved metrics to {output_file}")


if __name__ == '__main__':
    main()
