#!/usr/bin/env python3
"""
Compare two attention feature files to measure the effect of quantization.

Treats the first file as ground truth (e.g., full precision) and the second
as the system output (e.g., quantized). Computes ranking correlation metrics
to measure how much quantization affects document rankings.

Assumes both files have the same queries in the same order.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict


def load_features(npz_file):
    """Load features from npz file."""
    data = np.load(npz_file, allow_pickle=True)
    result = {
        'features': data['features'],
        'labels': data['labels'],
    }
    # Optional fields
    if 'docs_per_query' in data:
        result['docs_per_query'] = data['docs_per_query']
    if 'query_ids' in data:
        result['query_ids'] = data['query_ids']
    if 'doc_ids' in data:
        result['doc_ids'] = data['doc_ids']
    return result


def compute_document_scores(features, weights=None):
    """
    Compute document scores from features.

    If weights is None, uses sum of all head features (equal weighting).
    """
    if weights is None:
        return features.sum(axis=1)
    return features @ weights


def dcg_at_k(relevances, k):
    """Compute DCG@k."""
    relevances = np.asarray(relevances)[:k]
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return np.sum(relevances / discounts)


def ndcg_at_k(ranked_relevances, ideal_relevances, k):
    """Compute NDCG@k given ranked and ideal relevances."""
    dcg = dcg_at_k(ranked_relevances, k)
    idcg = dcg_at_k(sorted(ideal_relevances, reverse=True), k)
    if idcg == 0:
        return 1.0  # Perfect score if no relevant docs
    return dcg / idcg


def precision_at_k(relevances, k):
    """Compute Precision@k."""
    relevances = np.asarray(relevances)[:k]
    if relevances.size == 0:
        return 0.0
    return np.sum(relevances > 0) / k


def reciprocal_rank(relevances):
    """Compute Reciprocal Rank."""
    relevances = np.asarray(relevances)
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def rank_biased_overlap(list1, list2, p=0.9):
    """
    Compute Rank-Biased Overlap (RBO) between two ranked lists.

    Args:
        list1, list2: Ranked lists (item IDs or indices)
        p: Persistence parameter (0 < p < 1), higher = more weight to deeper ranks

    Returns:
        RBO score in [0, 1]
    """
    if len(list1) == 0 or len(list2) == 0:
        return 0.0

    k = min(len(list1), len(list2))

    # Compute overlap at each depth
    set1 = set()
    set2 = set()
    agreement = 0.0

    for d in range(1, k + 1):
        set1.add(list1[d-1])
        set2.add(list2[d-1])
        overlap = len(set1 & set2)
        agreement += (p ** (d-1)) * (overlap / d)

    return (1 - p) * agreement


def compare_rankings(scores_ref, scores_sys, doc_indices, ks=[1, 5, 10, 20, 100]):
    """
    Compare rankings from reference and system scores.

    Uses reference ranking as ground truth relevance.
    """
    # Get rankings
    rank_ref = np.argsort(-scores_ref)
    rank_sys = np.argsort(-scores_sys)

    # Create relevance scores based on reference ranking
    # Top doc gets highest relevance, decreasing
    n_docs = len(scores_ref)
    ref_relevances = np.zeros(n_docs)
    for i, doc_idx in enumerate(rank_ref):
        ref_relevances[doc_idx] = n_docs - i  # Higher rank = higher relevance

    # Get relevances in system ranking order
    sys_ranked_relevances = ref_relevances[rank_sys]

    metrics = {}

    # NDCG@k - how well does system preserve reference ranking
    for k in ks:
        if k <= n_docs:
            metrics[f'NDCG@{k}'] = ndcg_at_k(sys_ranked_relevances, ref_relevances, k)

    # Precision@k - what fraction of system's top-k are in reference's top-k
    for k in ks:
        if k <= n_docs:
            ref_topk = set(rank_ref[:k])
            sys_topk = set(rank_sys[:k])
            metrics[f'P@{k}'] = len(ref_topk & sys_topk) / k

    # Rank correlation
    if n_docs > 1:
        tau, _ = stats.kendalltau(scores_ref, scores_sys)
        rho, _ = stats.spearmanr(scores_ref, scores_sys)
        metrics['Kendall_tau'] = tau if not np.isnan(tau) else 0.0
        metrics['Spearman_rho'] = rho if not np.isnan(rho) else 0.0

    # RBO
    metrics['RBO_0.9'] = rank_biased_overlap(list(rank_ref), list(rank_sys), p=0.9)
    metrics['RBO_0.95'] = rank_biased_overlap(list(rank_ref), list(rank_sys), p=0.95)

    # Top-1 agreement
    metrics['Top1_match'] = 1.0 if rank_ref[0] == rank_sys[0] else 0.0

    return metrics


def compare_features_directly(feat_ref, feat_sys):
    """Compare features directly (element-wise)."""
    # Flatten if needed
    feat_ref = feat_ref.flatten()
    feat_sys = feat_sys.flatten()

    metrics = {}

    # Mean Squared Error
    metrics['MSE'] = np.mean((feat_ref - feat_sys) ** 2)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])

    # Mean Absolute Error
    metrics['MAE'] = np.mean(np.abs(feat_ref - feat_sys))

    # Correlation
    if np.std(feat_ref) > 0 and np.std(feat_sys) > 0:
        corr = np.corrcoef(feat_ref, feat_sys)[0, 1]
        metrics['Pearson_r'] = corr if not np.isnan(corr) else 0.0
    else:
        metrics['Pearson_r'] = 1.0 if np.allclose(feat_ref, feat_sys) else 0.0

    # Cosine similarity
    norm_ref = np.linalg.norm(feat_ref)
    norm_sys = np.linalg.norm(feat_sys)
    if norm_ref > 0 and norm_sys > 0:
        metrics['Cosine_sim'] = np.dot(feat_ref, feat_sys) / (norm_ref * norm_sys)
    else:
        metrics['Cosine_sim'] = 1.0 if np.allclose(feat_ref, feat_sys) else 0.0

    # Relative error
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err = np.abs(feat_ref - feat_sys) / (np.abs(feat_ref) + 1e-10)
        metrics['Mean_rel_error'] = np.mean(rel_err)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Compare two feature files to measure quantization effects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare full precision vs 4-bit quantized
  python scripts/compare_features.py \\
      --reference head_data/mistral/attention_features_nq_n100.npz \\
      --system head_data/mistral/attention_features_nq_n100_4bit.npz

  # Use specific head weights for scoring
  python scripts/compare_features.py \\
      --reference features_fp16.npz --system features_4bit.npz \\
      --weights head_data/mistral/bce_weights_lambda100.0_n1000.json
        """
    )
    parser.add_argument('--reference', '-r', type=str, required=True,
                        help='Reference feature file (e.g., full precision)')
    parser.add_argument('--system', '-s', type=str, required=True,
                        help='System feature file (e.g., quantized)')
    parser.add_argument('--weights', '-w', type=str, default=None,
                        help='Head weights file (JSON) for scoring. Default: equal weights')
    parser.add_argument('--docs_per_query', type=int, default=None,
                        help='Documents per query (auto-detected if in npz)')
    parser.add_argument('--ks', type=int, nargs='+', default=[1, 5, 10, 20],
                        help='k values for @k metrics (default: 1 5 10 20)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON file for results')
    args = parser.parse_args()

    print("=" * 70)
    print("Feature Comparison: Reference vs System")
    print("=" * 70)
    print(f"Reference: {args.reference}")
    print(f"System:    {args.system}")

    # Load features
    print("\nLoading features...", flush=True)
    ref_data = load_features(args.reference)
    sys_data = load_features(args.system)

    feat_ref = ref_data['features']
    feat_sys = sys_data['features']

    print(f"Reference shape: {feat_ref.shape}")
    print(f"System shape:    {feat_sys.shape}")

    if feat_ref.shape != feat_sys.shape:
        print("ERROR: Feature shapes don't match!")
        return

    # Determine docs per query
    if 'docs_per_query' in ref_data:
        docs_per_query = ref_data['docs_per_query']
        n_queries = len(docs_per_query)
    elif args.docs_per_query is not None:
        n_docs = feat_ref.shape[0]
        n_queries = n_docs // args.docs_per_query
        docs_per_query = [args.docs_per_query] * n_queries
    else:
        print("ERROR: docs_per_query not found in npz and not provided via --docs_per_query")
        return

    print(f"Queries: {n_queries}")

    # Load weights if provided
    weights = None
    if args.weights:
        print(f"\nLoading weights from {args.weights}...")
        with open(args.weights, 'r') as f:
            weight_data = json.load(f)

        n_features = feat_ref.shape[1]
        weights = np.zeros(n_features)

        if 'all_weights' in weight_data:
            # Assume 32 heads per layer (adjust if needed)
            num_heads = 32
            for key, w in weight_data['all_weights'].items():
                layer, head = map(int, key.split('-'))
                idx = layer * num_heads + head
                if idx < n_features:
                    weights[idx] = w

        n_nonzero = np.sum(np.abs(weights) > 1e-6)
        print(f"Loaded weights: {n_nonzero} non-zero out of {n_features}")

    # Compute document scores
    scores_ref = compute_document_scores(feat_ref, weights)
    scores_sys = compute_document_scores(feat_sys, weights)

    # Compare per query
    print("\n" + "=" * 70)
    print("Per-Query Ranking Comparison")
    print("=" * 70)

    all_metrics = defaultdict(list)
    doc_offset = 0

    for q_idx in range(n_queries):
        n_docs = docs_per_query[q_idx]
        q_scores_ref = scores_ref[doc_offset:doc_offset + n_docs]
        q_scores_sys = scores_sys[doc_offset:doc_offset + n_docs]
        doc_indices = list(range(doc_offset, doc_offset + n_docs))

        q_metrics = compare_rankings(q_scores_ref, q_scores_sys, doc_indices, args.ks)

        for k, v in q_metrics.items():
            all_metrics[k].append(v)

        doc_offset += n_docs

    # Aggregate metrics
    print(f"\n{'Metric':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 70)

    results = {'ranking_metrics': {}}
    for metric_name in sorted(all_metrics.keys()):
        values = all_metrics[metric_name]
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        print(f"{metric_name:<20} {mean_val:<12.4f} {std_val:<12.4f} {min_val:<12.4f} {max_val:<12.4f}")

        results['ranking_metrics'][metric_name] = {
            'mean': float(mean_val),
            'std': float(std_val),
            'min': float(min_val),
            'max': float(max_val)
        }

    # Feature-level comparison
    print("\n" + "=" * 70)
    print("Feature-Level Comparison (All Features)")
    print("=" * 70)

    feat_metrics = compare_features_directly(feat_ref, feat_sys)

    print(f"\n{'Metric':<20} {'Value':<12}")
    print("-" * 35)
    for metric_name, value in feat_metrics.items():
        print(f"{metric_name:<20} {value:<12.6f}")

    results['feature_metrics'] = {k: float(v) for k, v in feat_metrics.items()}

    # Per-head comparison
    print("\n" + "=" * 70)
    print("Per-Head Feature Comparison")
    print("=" * 70)

    n_heads = feat_ref.shape[1]
    head_correlations = []
    for h in range(n_heads):
        if np.std(feat_ref[:, h]) > 0 and np.std(feat_sys[:, h]) > 0:
            corr = np.corrcoef(feat_ref[:, h], feat_sys[:, h])[0, 1]
            if not np.isnan(corr):
                head_correlations.append(corr)

    if head_correlations:
        print(f"Head correlation: mean={np.mean(head_correlations):.4f}, "
              f"std={np.std(head_correlations):.4f}, "
              f"min={np.min(head_correlations):.4f}, "
              f"max={np.max(head_correlations):.4f}")

        # Find worst heads
        head_corrs = []
        for h in range(n_heads):
            if np.std(feat_ref[:, h]) > 0 and np.std(feat_sys[:, h]) > 0:
                corr = np.corrcoef(feat_ref[:, h], feat_sys[:, h])[0, 1]
                if not np.isnan(corr):
                    head_corrs.append((h, corr))

        head_corrs.sort(key=lambda x: x[1])
        print(f"\nWorst 5 heads (lowest correlation):")
        for h, corr in head_corrs[:5]:
            layer = h // 32
            head = h % 32
            print(f"  Layer {layer}, Head {head}: r={corr:.4f}")

        results['head_correlation'] = {
            'mean': float(np.mean(head_correlations)),
            'std': float(np.std(head_correlations)),
            'min': float(np.min(head_correlations)),
            'max': float(np.max(head_correlations))
        }

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    ndcg10 = results['ranking_metrics'].get('NDCG@10', {}).get('mean', 0)
    tau = results['ranking_metrics'].get('Kendall_tau', {}).get('mean', 0)
    top1 = results['ranking_metrics'].get('Top1_match', {}).get('mean', 0)
    pearson = results['feature_metrics'].get('Pearson_r', 0)

    print(f"Ranking preservation (NDCG@10): {ndcg10:.4f}")
    print(f"Rank correlation (Kendall tau): {tau:.4f}")
    print(f"Top-1 agreement: {top1:.1%}")
    print(f"Feature correlation (Pearson r): {pearson:.4f}")

    # Save results
    if args.output:
        results['config'] = {
            'reference': args.reference,
            'system': args.system,
            'weights': args.weights,
            'n_queries': n_queries,
            'n_documents': int(feat_ref.shape[0]),
            'n_features': int(feat_ref.shape[1])
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.output}")


if __name__ == '__main__':
    main()
