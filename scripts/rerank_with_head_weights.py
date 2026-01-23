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

from utils import log_command, load_features, get_head_info


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


def match_at_k(relevances, k):
    """Compute Match@k (1 if any relevant doc in top-k, 0 otherwise)."""
    relevances = np.asarray(relevances)[:k]
    if relevances.size == 0:
        return 0.0
    return 1.0 if np.any(relevances > 0) else 0.0


def evaluate_ranking(features, labels, weights, docs_per_query=50, top_k_heads=None,
                     ks=[1, 5, 10], metric_types=None, use_baseline=False):
    """
    Evaluate ranking metrics.

    Args:
        features: (num_docs, num_heads) attention features
        labels: (num_docs,) binary relevance labels
        weights: (num_heads,) head weights (ignored if use_baseline=True)
        docs_per_query: number of documents per query (int) or array of docs per query
        top_k_heads: if set, only use top-k heads
        ks: list of k values for @k metrics
        metric_types: list of metric types to compute ('ndcg', 'p', 'm', 'map', 'mrr')
                     If None, computes all metrics.
        use_baseline: if True, use original document order as baseline (no reranking)

    Returns:
        metrics: dict of metric name -> value
    """
    if metric_types is None:
        metric_types = ['ndcg', 'p', 'm', 'map', 'mrr']

    # Compute scores
    if use_baseline:
        # For baseline, use descending scores to preserve original order
        scores = np.arange(len(features), 0, -1, dtype=float)
    else:
        scores = compute_scores(features, weights, top_k=top_k_heads)

    # Initialize collectors for requested metrics
    compute_ndcg = 'ndcg' in metric_types
    compute_precision = 'p' in metric_types
    compute_match = 'm' in metric_types
    compute_map = 'map' in metric_types
    compute_mrr = 'mrr' in metric_types

    all_ndcg = {k: [] for k in ks} if compute_ndcg else {}
    all_precision = {k: [] for k in ks} if compute_precision else {}
    all_match = {k: [] for k in ks} if compute_match else {}
    all_ap = [] if compute_map else None
    all_rr = [] if compute_mrr else None

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

            # Compute requested metrics
            for k in ks:
                if compute_ndcg:
                    all_ndcg[k].append(ndcg_at_k(ranked_labels, k))
                if compute_precision:
                    all_precision[k].append(precision_at_k(ranked_labels, k))
                if compute_match:
                    all_match[k].append(match_at_k(ranked_labels, k))

            if compute_map:
                all_ap.append(average_precision(ranked_labels))
            if compute_mrr:
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

            # Compute requested metrics
            for k in ks:
                if compute_ndcg:
                    all_ndcg[k].append(ndcg_at_k(ranked_labels, k))
                if compute_precision:
                    all_precision[k].append(precision_at_k(ranked_labels, k))
                if compute_match:
                    all_match[k].append(match_at_k(ranked_labels, k))

            if compute_map:
                all_ap.append(average_precision(ranked_labels))
            if compute_mrr:
                all_rr.append(reciprocal_rank(ranked_labels))

    # Aggregate
    metrics = {}
    for k in ks:
        if compute_ndcg:
            metrics[f'NDCG@{k}'] = np.mean(all_ndcg[k])
        if compute_precision:
            metrics[f'P@{k}'] = np.mean(all_precision[k])
        if compute_match:
            metrics[f'M@{k}'] = np.mean(all_match[k])
    if compute_map:
        metrics['MAP'] = np.mean(all_ap)
    if compute_mrr:
        metrics['MRR'] = np.mean(all_rr)

    return metrics


def evaluate_single_feature_file(feature_file, llm_name, num_samples, weights, metadata,
                                  top_k_list, ks, metric_types, docs_per_query_override,
                                  compare_equal, include_baseline=True, verbose=True):
    """
    Evaluate a single feature file and return results.

    Args:
        include_baseline: if True, compute and include baseline retriever performance

    Returns:
        dict with feature_file info and results list
    """
    # Load features
    X, y, docs_per_query_arr = load_features(
        feature_file=feature_file,
        llm_name=llm_name,
        num_samples=num_samples
    )

    # Determine docs_per_query
    if docs_per_query_arr is not None:
        docs_per_query = docs_per_query_arr
        num_queries = len(docs_per_query)
        docs_info = f"variable (min={min(docs_per_query)}, max={max(docs_per_query)}, avg={np.mean(docs_per_query):.1f})"
    elif docs_per_query_override is not None:
        docs_per_query = docs_per_query_override
        num_queries = len(y) // docs_per_query
        docs_info = str(docs_per_query)
    else:
        docs_per_query = 50
        num_queries = len(y) // docs_per_query
        docs_info = f"{docs_per_query} (default)"

    if verbose:
        file_label = Path(feature_file).name if feature_file else f"n{num_samples}"
        print(f"\n{file_label}: {X.shape[0]} docs, {num_queries} queries, "
              f"{(y == 1).sum()} positive ({100*(y == 1).mean():.1f}%)")

    # Evaluate baseline (original retriever ranking) first if requested
    results = []
    if include_baseline:
        baseline_metrics = evaluate_ranking(
            X, y, weights=None,
            docs_per_query=docs_per_query,
            top_k_heads=None,
            ks=ks,
            metric_types=metric_types,
            use_baseline=True
        )

        results.append({
            'config': 'baseline',
            'weights': 'retriever',
            'metrics': baseline_metrics
        })

    # Evaluate for each top_k setting
    for top_k in top_k_list:
        label = f"top-{top_k}" if top_k else "all"

        metrics = evaluate_ranking(
            X, y, weights,
            docs_per_query=docs_per_query,
            top_k_heads=top_k,
            ks=ks,
            metric_types=metric_types
        )

        results.append({
            'config': label,
            'weights': metadata['type'],
            'metrics': metrics
        })

        # Compare with equal weights if requested
        if compare_equal and top_k is not None:
            equal_weights = np.zeros_like(weights)
            top_indices = get_top_k_heads(weights, top_k)
            equal_weights[top_indices] = 1.0 / top_k

            eq_metrics = evaluate_ranking(
                X, y, equal_weights,
                docs_per_query=docs_per_query,
                top_k_heads=None,
                ks=ks,
                metric_types=metric_types
            )

            results.append({
                'config': label,
                'weights': 'equal',
                'metrics': eq_metrics
            })

    return {
        'feature_file': str(feature_file) if feature_file else None,
        'num_docs': X.shape[0],
        'num_queries': num_queries,
        'docs_per_query': docs_info,
        'positive_docs': int((y == 1).sum()),
        'results': results
    }


def main():
    # Log command execution
    log_command()

    parser = argparse.ArgumentParser(description='Evaluate head weights on ranking metrics')
    parser.add_argument('--llm', type=str, default='mistral',
                        choices=['mistral', 'llama', 'phi', 'granite'])
    parser.add_argument('--weight_file', type=str, required=True,
                        help='Path to head weights file (BCE JSON or CoRe JSON)')
    parser.add_argument('--feature_file', '-f', type=str, nargs='+', default=None,
                        help='Path to feature file(s) (.npz). Can specify multiple files. '
                             'Default: head_data/{llm}/attention_features_n{num_samples}.npz')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples in feature file (used for default path)')
    parser.add_argument('--top_k_heads', type=int, nargs='+', default=None,
                        help='Evaluate with only top-k heads (can specify multiple)')
    parser.add_argument('--docs_per_query', type=int, default=None,
                        help='Number of documents per query (auto-detected from npz if available)')
    parser.add_argument('--ks', type=int, nargs='+', default=[1, 5, 10],
                        help='k values for @k metrics')
    parser.add_argument('--metrics', type=str, nargs='+',
                        default=['ndcg', 'p', 'm', 'map', 'mrr'],
                        help='Metrics to compute: ndcg, p (precision), m (match), map, mrr (default: all)')
    parser.add_argument('--compare_equal', action='store_true',
                        help='Also compare with equal weights on same heads')
    parser.add_argument('--no_baseline', action='store_true',
                        help='Skip computing baseline retriever performance')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file for metrics JSON (optional, no save if not specified)')
    args = parser.parse_args()

    # Normalize metric names to lowercase
    args.metrics = [m.lower() for m in args.metrics]

    # Get model config
    num_layers, num_heads = get_head_info(args.llm)

    print(f"Evaluating head weights for {args.llm}")
    print(f"Weight file: {args.weight_file}")
    print("=" * 70)

    # Load weights
    print(f"\nLoading weights...")
    weights, metadata = load_head_weights(args.weight_file, num_heads, num_layers)
    print(f"Weight type: {metadata['type']}")
    print(f"Non-zero heads: {metadata['nonzero']}/{len(weights)}")
    if metadata.get('lambda_l1'):
        print(f"Lambda L1: {metadata['lambda_l1']}")

    # Determine top_k_heads to evaluate
    if args.top_k_heads is None:
        top_k_list = [None]
    else:
        top_k_list = args.top_k_heads

    # Determine feature files to evaluate
    if args.feature_file is None:
        feature_files = [None]  # Will use default path
    else:
        feature_files = args.feature_file

    # Evaluate each feature file
    print(f"\nEvaluating on {len(feature_files)} feature file(s)...")
    all_file_results = []

    for feature_file in feature_files:
        file_results = evaluate_single_feature_file(
            feature_file=feature_file,
            llm_name=args.llm,
            num_samples=args.num_samples,
            weights=weights,
            metadata=metadata,
            top_k_list=top_k_list,
            ks=args.ks,
            metric_types=args.metrics,
            docs_per_query_override=args.docs_per_query,
            compare_equal=args.compare_equal,
            include_baseline=not args.no_baseline,
            verbose=True
        )
        all_file_results.append(file_results)

    # Print combined results table
    print(f"\n{'='*70}")
    print("Ranking Metrics")
    print("=" * 70)

    # Build metric names based on selected metrics
    metric_names = []
    if 'ndcg' in args.metrics:
        metric_names += [f'NDCG@{k}' for k in args.ks]
    if 'p' in args.metrics:
        metric_names += [f'P@{k}' for k in args.ks]
    if 'm' in args.metrics:
        metric_names += [f'M@{k}' for k in args.ks]
    if 'map' in args.metrics:
        metric_names.append('MAP')
    if 'mrr' in args.metrics:
        metric_names.append('MRR')

    # Collect all rows for finding max values, tracking file index
    all_rows = []
    file_row_ranges = []  # (start_idx, end_idx) for each file
    row_idx = 0

    for file_results in all_file_results:
        if file_results['feature_file']:
            file_label = Path(file_results['feature_file']).stem
            if len(file_label) > 28:
                file_label = file_label[:25] + "..."
        else:
            file_label = f"default (n={args.num_samples})"

        start_idx = row_idx
        for i, r in enumerate(file_results['results']):
            label = file_label if i == 0 else ""
            all_rows.append({
                'file_label': label,
                'config': r['config'],
                'weights': r['weights'],
                'metrics': r['metrics']
            })
            row_idx += 1
        file_row_ranges.append((start_idx, row_idx))

    # Find global max value for each metric (only highlight if multiple rows)
    global_max = {}
    if len(all_rows) > 1:
        for name in metric_names:
            global_max[name] = max(row['metrics'][name] for row in all_rows)

    # Find per-file max values (only if multiple files)
    file_max = [{} for _ in range(len(file_row_ranges))]
    if len(file_row_ranges) > 1:
        for file_idx, (start, end) in enumerate(file_row_ranges):
            file_rows = all_rows[start:end]
            if len(file_rows) > 1:
                for name in metric_names:
                    file_max[file_idx][name] = max(row['metrics'][name] for row in file_rows)

    # ANSI color codes
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    # Header
    header = f"{'File':<30} {'Config':<12} {'Weights':<10}"
    for name in metric_names:
        header += f" {name:<8}"
    print(header)
    print("-" * len(header))

    # Print rows with highlighting
    for row_idx, row in enumerate(all_rows):
        # Determine which file this row belongs to
        file_idx = next(i for i, (start, end) in enumerate(file_row_ranges) if start <= row_idx < end)

        line = f"{row['file_label']:<30} {row['config']:<12} {row['weights']:<10}"
        for name in metric_names:
            value = row['metrics'][name]
            formatted = f"{value:<8.4f}"
            # Highlight global max in green+bold
            if global_max and value == global_max[name]:
                formatted = f"{GREEN}{BOLD}{value:<8.4f}{RESET}"
            # Highlight per-file max in blue+bold (if not global max and multiple files)
            elif file_max[file_idx].get(name) is not None and value == file_max[file_idx][name]:
                formatted = f"{BLUE}{BOLD}{value:<8.4f}{RESET}"
            line += f" {formatted}"
        print(line)

    # Save results if output file specified
    if args.output:
        output_file = Path(args.output)
        output_data = {
            'weight_file': str(args.weight_file),
            'llm': args.llm,
            'num_samples': args.num_samples,
            'feature_files': [f['feature_file'] for f in all_file_results],
            'file_results': all_file_results
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved metrics to {output_file}")


if __name__ == '__main__':
    main()
