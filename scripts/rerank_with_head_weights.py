#!/usr/bin/env python3
"""
Evaluate head weights on ranking metrics (NDCG@k, Precision@k, MAP@k).

This script loads attention features and head weights, computes document scores,
and evaluates ranking performance on the head detection data.

Evaluation modes:
- Baseline: Original retriever ranking (no reranking)
- Oracle: Upper bound performance (gold document moved to rank 1 if present)
- Reranking: Using learned head weights with various top-k configurations

Note: Oracle results are displayed but excluded from color highlighting to avoid
skewing comparisons between actual reranking methods.

IMPORTANT: For proper BEIR evaluation, use --qrels to provide the external qrels file.
The labels in the .npz file only cover documents in the retrieved set, but NDCG
computation requires the full qrels to properly compute the ideal DCG.
"""

import json
import argparse
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

from utils import log_command, load_features, get_head_info


def load_qrels_file(qrels_path):
    """
    Load qrels from a file (TSV or TREC format).

    Supports:
    - TSV format: query-id, corpus-id, score (with header)
    - TREC format: query-id, iteration, corpus-id, score (no header)

    Returns:
        qrels: dict of {query_id: {doc_id: relevance}}
    """
    qrels = {}
    qrels_path = Path(qrels_path)

    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels file not found: {qrels_path}")

    with open(qrels_path, 'r') as f:
        first_line = f.readline().strip()
        f.seek(0)  # Reset to beginning

        # Check if first line is a header (TSV format)
        if first_line.startswith('query-id') or first_line.startswith('query_id'):
            # TSV format with header
            import csv
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                # Handle different column names
                q_id = str(row.get('query-id', row.get('query_id', '')))
                d_id = str(row.get('corpus-id', row.get('corpus_id', row.get('doc-id', row.get('doc_id', '')))))
                score = int(row.get('score', row.get('relevance', 0)))

                if q_id and d_id:
                    if q_id not in qrels:
                        qrels[q_id] = {}
                    if score > 0:  # Only store positive relevance
                        qrels[q_id][d_id] = score
        else:
            # TREC format (no header)
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) >= 3:
                    if len(parts) == 3:
                        # Format: query-id, corpus-id, score
                        q_id, d_id, score = parts
                    else:
                        # Format: query-id, iteration, corpus-id, score
                        q_id, _, d_id, score = parts[:4]

                    q_id = str(q_id)
                    d_id = str(d_id)
                    try:
                        score = int(score)
                    except ValueError:
                        score = int(float(score))

                    if q_id not in qrels:
                        qrels[q_id] = {}
                    if score > 0:  # Only store positive relevance
                        qrels[q_id][d_id] = score

    return qrels

# Try to import BEIR evaluator (optional)
try:
    from beir.retrieval.evaluation import EvaluateRetrieval
    BEIR_AVAILABLE = True
except ImportError:
    BEIR_AVAILABLE = False


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


def compute_ranked_results(features, weights, query_ids, doc_ids, docs_per_query, top_k_heads=None):
    """
    Compute ranked document lists with scores.

    Args:
        features: (num_docs, num_heads) attention features
        weights: (num_heads,) head weights
        query_ids: (num_docs,) query IDs for each document
        doc_ids: (num_docs,) document IDs
        docs_per_query: number of documents per query (int) or array
        top_k_heads: if set, only use top-k heads

    Returns:
        results: dict of {query_id: {doc_id: score, ...}} sorted by score descending
    """
    scores = compute_scores(features, weights, top_k=top_k_heads)
    results = {}

    if isinstance(docs_per_query, (list, np.ndarray)):
        doc_offset = 0
        for q_idx in range(len(docs_per_query)):
            n_docs = docs_per_query[q_idx]
            q_scores = scores[doc_offset:doc_offset + n_docs]
            q_doc_ids = doc_ids[doc_offset:doc_offset + n_docs]
            q_id = str(query_ids[doc_offset])
            doc_offset += n_docs

            # Sort by score descending
            sorted_indices = np.argsort(-q_scores)
            results[q_id] = {
                str(q_doc_ids[i]): float(q_scores[i])
                for i in sorted_indices
            }
    else:
        num_queries = len(features) // docs_per_query
        for q in range(num_queries):
            start = q * docs_per_query
            end = start + docs_per_query

            q_scores = scores[start:end]
            q_doc_ids = doc_ids[start:end]
            q_id = str(query_ids[start])

            # Sort by score descending
            sorted_indices = np.argsort(-q_scores)
            results[q_id] = {
                str(q_doc_ids[i]): float(q_scores[i])
                for i in sorted_indices
            }

    return results


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


def reciprocal_rank_fusion(rankings, k=60):
    """
    Combine multiple rankings using Reciprocal Rank Fusion (RRF).

    RRF score for document d = sum over rankings of 1/(k + rank(d))

    Args:
        rankings: list of lists, each inner list is doc_ids in rank order
        k: RRF constant (default 60, as in the original paper)

    Returns:
        fused_ranking: list of doc_ids sorted by RRF score (descending)
        fused_scores: dict of doc_id -> RRF score
    """
    scores = defaultdict(float)

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)

    # Sort by score descending
    sorted_docs = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)

    return sorted_docs, dict(scores)


def compute_fused_scores(features, weights, docs_per_query, top_k_heads=None, rrf_k=60):
    """
    Compute fused scores combining baseline (original order) and reranked scores using RRF.

    Args:
        features: (num_docs, num_heads) attention features
        weights: (num_heads,) head weights
        docs_per_query: number of documents per query (int) or array
        top_k_heads: if set, only use top-k heads for reranking
        rrf_k: RRF constant (default 60)

    Returns:
        fused_scores: (num_docs,) array of RRF-fused scores
    """
    # Compute reranked scores
    rerank_scores = compute_scores(features, weights, top_k=top_k_heads)

    # Baseline scores (preserve original order - higher index = lower rank)
    baseline_scores = np.arange(len(features), 0, -1, dtype=float)

    fused_scores = np.zeros(len(features))

    if isinstance(docs_per_query, (list, np.ndarray)):
        doc_offset = 0
        for q_idx in range(len(docs_per_query)):
            n_docs = docs_per_query[q_idx]

            # Get scores for this query
            q_baseline = baseline_scores[doc_offset:doc_offset + n_docs]
            q_rerank = rerank_scores[doc_offset:doc_offset + n_docs]

            # Get rankings (indices sorted by score descending)
            baseline_ranking = np.argsort(-q_baseline)
            rerank_ranking = np.argsort(-q_rerank)

            # Compute RRF scores
            q_fused = np.zeros(n_docs)
            for rank, idx in enumerate(baseline_ranking, start=1):
                q_fused[idx] += 1.0 / (rrf_k + rank)
            for rank, idx in enumerate(rerank_ranking, start=1):
                q_fused[idx] += 1.0 / (rrf_k + rank)

            fused_scores[doc_offset:doc_offset + n_docs] = q_fused
            doc_offset += n_docs
    else:
        num_queries = len(features) // docs_per_query
        for q in range(num_queries):
            start = q * docs_per_query
            end = start + docs_per_query

            q_baseline = baseline_scores[start:end]
            q_rerank = rerank_scores[start:end]

            baseline_ranking = np.argsort(-q_baseline)
            rerank_ranking = np.argsort(-q_rerank)

            q_fused = np.zeros(docs_per_query)
            for rank, idx in enumerate(baseline_ranking, start=1):
                q_fused[idx] += 1.0 / (rrf_k + rank)
            for rank, idx in enumerate(rerank_ranking, start=1):
                q_fused[idx] += 1.0 / (rrf_k + rank)

            fused_scores[start:end] = q_fused

    return fused_scores


def evaluate_ranking_beir(features, labels, weights, query_ids, doc_ids, docs_per_query,
                          top_k_heads=None, ks=[1, 5, 10], metric_types=None, use_baseline=False,
                          use_oracle=False, use_fusion=False, rrf_k=60, external_qrels=None):
    """
    Evaluate ranking metrics using BEIR's EvaluateRetrieval.

    Args:
        features: (num_docs, num_heads) attention features
        labels: (num_docs,) binary relevance labels (used for oracle if external_qrels not provided)
        weights: (num_heads,) head weights (ignored if use_baseline=True or use_oracle=True)
        query_ids: (num_docs,) query IDs for each document
        doc_ids: (num_docs,) document IDs
        docs_per_query: number of documents per query (int) or array of docs per query
        top_k_heads: if set, only use top-k heads
        ks: list of k values for @k metrics
        metric_types: list of metric types to compute (ignored - BEIR computes all)
        use_baseline: if True, use original document order as baseline (no reranking)
        use_oracle: if True, move gold document to first position (upper bound)
        use_fusion: if True, use RRF fusion of baseline and reranked scores
        rrf_k: RRF constant (default 60)
        external_qrels: dict of {query_id: {doc_id: relevance}} from external qrels file.
                       If provided, uses this for evaluation instead of labels from .npz.
                       IMPORTANT: For proper NDCG, this should contain ALL relevant docs,
                       not just those in the retrieved set.

    Returns:
        metrics: dict of metric name -> value
    """
    if not BEIR_AVAILABLE:
        raise ImportError("BEIR is not installed. Install with: pip install beir")

    # Compute scores
    if use_baseline:
        # For baseline, use descending scores to preserve original order
        scores = np.arange(len(features), 0, -1, dtype=float)
    elif use_oracle:
        # For oracle, assign highest score to gold document if it exists
        scores = np.zeros(len(features))
        # Will be handled in ranking phase below
    elif use_fusion:
        # RRF fusion of baseline and reranked scores
        scores = compute_fused_scores(features, weights, docs_per_query, top_k_heads, rrf_k)
    else:
        scores = compute_scores(features, weights, top_k=top_k_heads)

    # Convert to BEIR format
    # results: {query_id: {doc_id: score}}
    results = {}

    # Handle variable or fixed docs_per_query
    if isinstance(docs_per_query, (list, np.ndarray)):
        # Variable docs per query
        doc_offset = 0
        for q_idx in range(len(docs_per_query)):
            n_docs = docs_per_query[q_idx]
            q_scores = scores[doc_offset:doc_offset + n_docs].copy()
            q_labels = labels[doc_offset:doc_offset + n_docs]
            q_doc_ids = doc_ids[doc_offset:doc_offset + n_docs]
            q_id = str(query_ids[doc_offset])

            # Oracle: if gold doc exists in retrieved set, give it highest score
            if use_oracle:
                # Check external qrels first, then fall back to labels
                if external_qrels is not None and q_id in external_qrels:
                    gold_indices = [i for i, d_id in enumerate(q_doc_ids)
                                    if str(d_id) in external_qrels[q_id] and external_qrels[q_id][str(d_id)] > 0]
                else:
                    gold_indices = np.where(q_labels > 0)[0].tolist()

                if len(gold_indices) > 0:
                    # Move first gold doc to top by giving it max score + 1
                    max_score = q_scores.max() if len(q_scores) > 0 else 0
                    q_scores[gold_indices[0]] = max_score + 1.0

            doc_offset += n_docs

            # Add to results
            results[q_id] = {}
            for doc_id, score in zip(q_doc_ids, q_scores):
                results[q_id][str(doc_id)] = float(score)
    else:
        # Fixed docs per query
        num_queries = len(labels) // docs_per_query
        for q in range(num_queries):
            start = q * docs_per_query
            end = start + docs_per_query

            q_scores = scores[start:end].copy()
            q_labels = labels[start:end]
            q_doc_ids = doc_ids[start:end]
            q_id = str(query_ids[start])

            # Oracle: if gold doc exists in retrieved set, give it highest score
            if use_oracle:
                # Check external qrels first, then fall back to labels
                if external_qrels is not None and q_id in external_qrels:
                    gold_indices = [i for i, d_id in enumerate(q_doc_ids)
                                    if str(d_id) in external_qrels[q_id] and external_qrels[q_id][str(d_id)] > 0]
                else:
                    gold_indices = np.where(q_labels > 0)[0].tolist()

                if len(gold_indices) > 0:
                    # Move first gold doc to top by giving it max score + 1
                    max_score = q_scores.max() if len(q_scores) > 0 else 0
                    q_scores[gold_indices[0]] = max_score + 1.0

            # Add to results
            results[q_id] = {}
            for doc_id, score in zip(q_doc_ids, q_scores):
                results[q_id][str(doc_id)] = float(score)

    # Determine qrels to use
    if external_qrels is not None:
        # Use external qrels - filter to only queries we have results for
        qrels = {q_id: external_qrels[q_id] for q_id in results.keys() if q_id in external_qrels}

        # Warn if some queries don't have qrels
        missing_qrels = set(results.keys()) - set(qrels.keys())
        if missing_qrels:
            # Show examples to help debug ID format mismatches
            sample_results = list(results.keys())[:3]
            sample_qrels = list(external_qrels.keys())[:3]
            print(f"Warning: {len(missing_qrels)}/{len(results)} queries have no qrels")
            print(f"  Sample query IDs from .npz: {sample_results}")
            print(f"  Sample query IDs from qrels: {sample_qrels}")
            if len(missing_qrels) == len(results):
                print(f"  ERROR: No queries matched! Check that query ID formats match between .npz and qrels file.")

        # Also check doc ID coverage - how many relevant docs from qrels are in our results?
        total_relevant_in_qrels = 0
        total_relevant_found = 0
        sample_missing_docs = []
        sample_found_docs = []
        for q_id in qrels:
            if q_id in results:
                result_doc_ids = set(results[q_id].keys())
                for d_id, rel in qrels[q_id].items():
                    if rel > 0:
                        total_relevant_in_qrels += 1
                        if d_id in result_doc_ids:
                            total_relevant_found += 1
                            if len(sample_found_docs) < 3:
                                sample_found_docs.append(d_id)
                        elif len(sample_missing_docs) < 3:
                            sample_missing_docs.append(d_id)

        if total_relevant_in_qrels > 0:
            coverage = 100 * total_relevant_found / total_relevant_in_qrels
            print(f"  Relevant doc coverage: {total_relevant_found}/{total_relevant_in_qrels} ({coverage:.1f}%)")
            if total_relevant_found == 0 and sample_missing_docs:
                # Show sample doc IDs to help debug format mismatch
                sample_result_docs = []
                for q_id in list(results.keys())[:1]:
                    sample_result_docs = list(results[q_id].keys())[:3]
                print(f"  Sample doc IDs from .npz: {sample_result_docs}")
                print(f"  Sample doc IDs from qrels: {sample_missing_docs}")
                print(f"  ERROR: No docs matched! Check that doc ID formats match between .npz and qrels file.")
    else:
        # Build qrels from labels (only covers docs in retrieved set - may underestimate NDCG)
        qrels = {}
        if isinstance(docs_per_query, (list, np.ndarray)):
            doc_offset = 0
            for q_idx in range(len(docs_per_query)):
                n_docs = docs_per_query[q_idx]
                q_labels = labels[doc_offset:doc_offset + n_docs]
                q_doc_ids = doc_ids[doc_offset:doc_offset + n_docs]
                q_id = str(query_ids[doc_offset])
                doc_offset += n_docs

                qrels[q_id] = {}
                for doc_id, label in zip(q_doc_ids, q_labels):
                    if label > 0:  # Only positive labels
                        qrels[q_id][str(doc_id)] = int(label)
        else:
            num_queries = len(labels) // docs_per_query
            for q in range(num_queries):
                start = q * docs_per_query
                end = start + docs_per_query
                q_labels = labels[start:end]
                q_doc_ids = doc_ids[start:end]
                q_id = str(query_ids[start])

                qrels[q_id] = {}
                for doc_id, label in zip(q_doc_ids, q_labels):
                    if label > 0:  # Only positive labels
                        qrels[q_id][str(doc_id)] = int(label)

    # Use BEIR evaluator
    evaluator = EvaluateRetrieval()

    # Compute metrics at specified k values
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, ks)

    # Format metrics to match custom evaluator output
    metrics = {}
    for k in ks:
        metrics[f'NDCG@{k}'] = ndcg.get(f'NDCG@{k}', 0.0)
        metrics[f'P@{k}'] = precision.get(f'P@{k}', 0.0)
        # BEIR doesn't have Match@k, use Recall@k as proxy
        metrics[f'M@{k}'] = recall.get(f'Recall@{k}', 0.0)

    # Add MAP and MRR if available
    # BEIR computes MAP and MRR at highest k value
    map_key = f'MAP@{max(ks) if ks else 100}'
    if map_key in _map:
        metrics['MAP'] = _map[map_key]
    elif 'MAP@100' in _map:
        metrics['MAP'] = _map['MAP@100']

    mrr_key = f'MRR@{max(ks) if ks else 10}'
    if mrr_key in _map:
        metrics['MRR'] = _map[mrr_key]
    elif 'MRR@10' in _map:
        metrics['MRR'] = _map['MRR@10']

    return metrics


def evaluate_ranking(features, labels, weights, docs_per_query=50, top_k_heads=None,
                     ks=[1, 5, 10], metric_types=None, use_baseline=False, use_oracle=False,
                     use_fusion=False, rrf_k=60):
    """
    Evaluate ranking metrics.

    Args:
        features: (num_docs, num_heads) attention features
        labels: (num_docs,) binary relevance labels
        weights: (num_heads,) head weights (ignored if use_baseline=True or use_oracle=True)
        docs_per_query: number of documents per query (int) or array of docs per query
        top_k_heads: if set, only use top-k heads
        ks: list of k values for @k metrics
        metric_types: list of metric types to compute ('ndcg', 'p', 'm', 'map', 'mrr')
                     If None, computes all metrics.
        use_baseline: if True, use original document order as baseline (no reranking)
        use_oracle: if True, move gold document to first position (upper bound)
        use_fusion: if True, use RRF fusion of baseline and reranked scores
        rrf_k: RRF constant (default 60)

    Returns:
        metrics: dict of metric name -> value
    """
    if metric_types is None:
        metric_types = ['ndcg', 'p', 'm', 'map', 'mrr']

    # Compute scores
    if use_baseline:
        # For baseline, use descending scores to preserve original order
        scores = np.arange(len(features), 0, -1, dtype=float)
    elif use_oracle:
        # For oracle, will handle per-query below
        scores = np.zeros(len(features))
    elif use_fusion:
        # RRF fusion of baseline and reranked scores
        scores = compute_fused_scores(features, weights, docs_per_query, top_k_heads, rrf_k)
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
            q_scores = scores[doc_offset:doc_offset + n_docs].copy()
            q_labels = labels[doc_offset:doc_offset + n_docs]
            doc_offset += n_docs

            # Oracle: if gold doc exists, give it highest score
            if use_oracle:
                gold_indices = np.where(q_labels > 0)[0]
                if len(gold_indices) > 0:
                    # Move first gold doc to top by giving it max score + 1
                    max_score = q_scores.max() if len(q_scores) > 0 else 0
                    q_scores[gold_indices[0]] = max_score + 1.0

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

            q_scores = scores[start:end].copy()
            q_labels = labels[start:end]

            # Oracle: if gold doc exists, give it highest score
            if use_oracle:
                gold_indices = np.where(q_labels > 0)[0]
                if len(gold_indices) > 0:
                    # Move first gold doc to top by giving it max score + 1
                    max_score = q_scores.max() if len(q_scores) > 0 else 0
                    q_scores[gold_indices[0]] = max_score + 1.0

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
                                  compare_equal, include_baseline=True, include_oracle=True,
                                  include_fusion=False, rrf_k=60,
                                  evaluator='custom', external_qrels=None, verbose=True,
                                  return_ranked_results=False):
    """
    Evaluate a single feature file and return results.

    Args:
        include_baseline: if True, compute and include baseline retriever performance
        include_oracle: if True, compute and include oracle (upper bound) performance
        include_fusion: if True, compute RRF fusion of baseline and reranked scores
        rrf_k: RRF constant for fusion (default 60)
        evaluator: 'custom' or 'beir' - which evaluation method to use
        external_qrels: dict of {query_id: {doc_id: relevance}} from external qrels file.
                       If provided with evaluator='beir', uses this for proper NDCG computation.
        return_ranked_results: if True, also return ranked document lists with scores

    Returns:
        dict with feature_file info, results list, and optionally ranked_results
    """
    # Load features
    X, y, docs_per_query_arr, query_ids, doc_ids = load_features(
        feature_file=feature_file,
        llm_name=llm_name,
        num_samples=num_samples,
        return_ids=True
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

    # Choose evaluation function based on evaluator type
    if evaluator == 'beir':
        if not BEIR_AVAILABLE:
            print("Warning: BEIR not available, falling back to custom evaluator", flush=True)
            evaluator = 'custom'
        else:
            eval_func = lambda **kwargs: evaluate_ranking_beir(
                features=kwargs['features'],
                labels=kwargs['labels'],
                weights=kwargs['weights'],
                query_ids=query_ids,
                doc_ids=doc_ids,
                docs_per_query=kwargs['docs_per_query'],
                top_k_heads=kwargs.get('top_k_heads'),
                ks=kwargs['ks'],
                metric_types=kwargs.get('metric_types'),
                use_baseline=kwargs.get('use_baseline', False),
                use_oracle=kwargs.get('use_oracle', False),
                use_fusion=kwargs.get('use_fusion', False),
                rrf_k=kwargs.get('rrf_k', 60),
                external_qrels=external_qrels
            )

    if evaluator == 'custom':
        eval_func = lambda **kwargs: evaluate_ranking(
            features=kwargs['features'],
            labels=kwargs['labels'],
            weights=kwargs['weights'],
            docs_per_query=kwargs['docs_per_query'],
            top_k_heads=kwargs.get('top_k_heads'),
            ks=kwargs['ks'],
            metric_types=kwargs.get('metric_types'),
            use_baseline=kwargs.get('use_baseline', False),
            use_oracle=kwargs.get('use_oracle', False),
            use_fusion=kwargs.get('use_fusion', False),
            rrf_k=kwargs.get('rrf_k', 60)
        )

    # Evaluate baseline (original retriever ranking) first if requested
    results = []
    if include_baseline:
        baseline_metrics = eval_func(
            features=X,
            labels=y,
            weights=None,
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

    # Evaluate oracle (upper bound: gold doc at rank 1) if requested
    if include_oracle:
        oracle_metrics = eval_func(
            features=X,
            labels=y,
            weights=None,
            docs_per_query=docs_per_query,
            top_k_heads=None,
            ks=ks,
            metric_types=metric_types,
            use_oracle=True
        )

        results.append({
            'config': 'oracle',
            'weights': 'gold@1',
            'metrics': oracle_metrics
        })

    # Evaluate for each top_k setting
    for top_k in top_k_list:
        label = f"top-{top_k}" if top_k else "all"

        metrics = eval_func(
            features=X,
            labels=y,
            weights=weights,
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

            eq_metrics = eval_func(
                features=X,
                labels=y,
                weights=equal_weights,
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

        # Evaluate fusion (RRF of baseline + reranked) if requested
        if include_fusion:
            fusion_metrics = eval_func(
                features=X,
                labels=y,
                weights=weights,
                docs_per_query=docs_per_query,
                top_k_heads=top_k,
                ks=ks,
                metric_types=metric_types,
                use_fusion=True,
                rrf_k=rrf_k
            )

            results.append({
                'config': f"{label}+rrf",
                'weights': metadata['type'],
                'metrics': fusion_metrics
            })

    # Compute ranked results if requested
    ranked_results = None
    if return_ranked_results:
        # Use first top_k setting (or all heads if None)
        top_k = top_k_list[0] if top_k_list else None
        ranked_results = compute_ranked_results(
            features=X,
            weights=weights,
            query_ids=query_ids,
            doc_ids=doc_ids,
            docs_per_query=docs_per_query,
            top_k_heads=top_k
        )

    return {
        'feature_file': str(feature_file) if feature_file else None,
        'num_docs': X.shape[0],
        'num_queries': num_queries,
        'docs_per_query': docs_info,
        'positive_docs': int((y == 1).sum()),
        'results': results,
        'ranked_results': ranked_results
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
    parser.add_argument('--no_oracle', action='store_true',
                        help='Skip computing oracle (upper bound) performance')
    parser.add_argument('--fusion', action='store_true',
                        help='Include RRF fusion of baseline and reranked results')
    parser.add_argument('--rrf_k', type=int, default=60,
                        help='RRF constant k (default: 60, as in the original RRF paper)')
    parser.add_argument('--evaluator', type=str, default='beir', choices=['custom', 'beir'],
                        help='Evaluation method: beir (default, uses BEIR library) or custom (built-in metrics). '
                             'Note: BEIR requires "pip install beir". Use --qrels to provide external qrels for proper NDCG.')
    parser.add_argument('--qrels', type=str, default=None,
                        help='Path to external qrels file (TSV or TREC format). IMPORTANT: For proper NDCG '
                             'computation with BEIR evaluator, provide the full qrels file which may contain '
                             'relevant documents not in the retrieved set. Without this, NDCG is computed only '
                             'against docs in the .npz file which may underestimate the true score.')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file for metrics JSON (optional, no save if not specified)')
    parser.add_argument('--save_ranked', action='store_true',
                        help='Save ranked document lists with scores to reranked_results/<llm>/k<k>/')
    args = parser.parse_args()

    # Check if BEIR is available when requested
    if args.evaluator == 'beir' and not BEIR_AVAILABLE:
        print("Error: BEIR evaluation requested but beir package is not installed.")
        print("Install with: pip install beir")
        print("Falling back to custom evaluator.")
        args.evaluator = 'custom'

    # Normalize metric names to lowercase
    args.metrics = [m.lower() for m in args.metrics]

    # Get model config
    num_layers, num_heads = get_head_info(args.llm)

    print(f"Evaluating head weights for {args.llm}")
    print(f"Weight file: {args.weight_file}")
    print(f"Evaluator: {args.evaluator}")
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

    # Load external qrels if provided
    external_qrels = None
    if args.qrels:
        print(f"\nLoading external qrels from {args.qrels}...")
        external_qrels = load_qrels_file(args.qrels)
        total_relevant = sum(len(docs) for docs in external_qrels.values())
        print(f"Loaded qrels: {len(external_qrels)} queries, {total_relevant} total relevant docs")

        if args.evaluator == 'custom':
            print("Warning: External qrels are ignored with --evaluator custom. Use --evaluator beir (default) to use them.")

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
            include_oracle=not args.no_oracle,
            include_fusion=args.fusion,
            rrf_k=args.rrf_k,
            evaluator=args.evaluator,
            external_qrels=external_qrels,
            verbose=True,
            return_ranked_results=args.save_ranked
        )
        all_file_results.append(file_results)

        # Save ranked results if requested
        if args.save_ranked and file_results.get('ranked_results'):
            # Determine k from feature file name or docs_per_query
            if feature_file:
                # Try to extract k from filename like "attention_features_nq_k40.npz"
                import re
                k_match = re.search(r'_k(\d+)', str(feature_file))
                if k_match:
                    k_value = k_match.group(1)
                else:
                    # Fall back to docs_per_query
                    k_value = str(file_results.get('docs_per_query', 'unknown')).split()[0]
            else:
                k_value = 'unknown'

            # Create output directory
            output_dir = Path('reranked_results') / args.llm / f'k{k_value}'
            output_dir.mkdir(parents=True, exist_ok=True)

            # Determine output filename from input feature file
            if feature_file:
                input_stem = Path(feature_file).stem
            else:
                input_stem = f'features_n{args.num_samples}'

            output_file = output_dir / f'{input_stem}_reranked.json'

            with open(output_file, 'w') as f:
                json.dump(file_results['ranked_results'], f, indent=2)
            print(f"Saved ranked results to {output_file}")

    # Print combined results table
    print(f"\n{'='*70}")
    print("Ranking Metrics")
    print("=" * 70)

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

    # Build metric names based on selected metrics and what's actually available
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

    # Filter metric_names to only include those present in all rows
    if all_rows:
        available_metrics = set(all_rows[0]['metrics'].keys())
        for row in all_rows[1:]:
            available_metrics &= set(row['metrics'].keys())
        metric_names = [m for m in metric_names if m in available_metrics]

        if len(metric_names) == 0:
            print("Warning: No common metrics found across all results")
            return

    # Find global max value for each metric (only highlight if multiple rows)
    # Exclude oracle rows from coloring
    non_oracle_rows = [row for row in all_rows if row['config'] != 'oracle']
    global_max = {}
    if len(non_oracle_rows) > 1:
        for name in metric_names:
            global_max[name] = max(row['metrics'][name] for row in non_oracle_rows)

    # Find per-file max and second-max values (only if multiple files)
    # Exclude oracle rows from coloring
    file_max = [{} for _ in range(len(file_row_ranges))]
    file_second_max = [{} for _ in range(len(file_row_ranges))]
    if len(file_row_ranges) > 1:
        for file_idx, (start, end) in enumerate(file_row_ranges):
            file_rows = [row for row in all_rows[start:end] if row['config'] != 'oracle']
            if len(file_rows) > 1:
                for name in metric_names:
                    values = sorted([row['metrics'][name] for row in file_rows], reverse=True)
                    file_max[file_idx][name] = values[0]
                    # Get second max only if there are at least 2 rows
                    if len(values) >= 2:
                        file_second_max[file_idx][name] = values[1]

    # ANSI color codes
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
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
            # Skip coloring for oracle rows
            if row['config'] != 'oracle':
                # Highlight global max in green+bold
                if global_max and value == global_max[name]:
                    formatted = f"{GREEN}{BOLD}{value:<8.4f}{RESET}"
                # Highlight per-file max in cyan+bold (if not global max and multiple files)
                elif file_max[file_idx].get(name) is not None and value == file_max[file_idx][name]:
                    formatted = f"{CYAN}{BOLD}{value:<8.4f}{RESET}"
                # Highlight per-file second max in yellow+bold (if not max and multiple files)
                elif file_second_max[file_idx].get(name) is not None and value == file_second_max[file_idx][name]:
                    formatted = f"{YELLOW}{BOLD}{value:<8.4f}{RESET}"
            line += f" {formatted}"
        print(line)

    # Save results if output file specified
    if args.output:
        output_file = Path(args.output)
        output_data = {
            'weight_file': str(args.weight_file),
            'llm': args.llm,
            'num_samples': args.num_samples,
            'qrels_file': args.qrels,
            'evaluator': args.evaluator,
            'feature_files': [f['feature_file'] for f in all_file_results],
            'file_results': all_file_results
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved metrics to {output_file}")


if __name__ == '__main__':
    main()
