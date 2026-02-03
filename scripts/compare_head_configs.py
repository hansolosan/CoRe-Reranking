#!/usr/bin/env python3
"""
Compare two head configuration files using BEIR evaluation and statistical significance testing.

This script:
1. Takes two head weight/configuration files as input (CoRe or trained BCE/InfoNCE weights)
2. Evaluates both configurations on all BEIR datasets
3. Performs proper randomization tests for statistical significance using per-query scores

Statistical Significance Testing:
- Per-dataset: Permutation test on query-level metric scores within each dataset
- Overall BEIR: Permutation test on all query scores across datasets

Usage:
    python compare_head_configs.py \
        --config1 ../head_data/mistral/core_temp0.001_prune0.0.json \
        --config2 ../head_data/mistral/bce_weights_lambda0.001_n1000.json \
        --llm mistral \
        --feature_dir ../head_data/mistral \
        --k 10 \
        --num_heads 8 \
        --num_permutations 10000
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import log_command, load_features, get_head_info, parse_args_with_config

# Try to import BEIR evaluator (optional)
try:
    from beir.retrieval.evaluation import EvaluateRetrieval
    BEIR_AVAILABLE = True
except ImportError:
    BEIR_AVAILABLE = False

# BEIR dataset names
BEIR_MAIN_DATASETS = [
    'trec-covid', 'nfcorpus', 'dbpedia-entity', 'scifact', 'scidocs',
    'fiqa', 'nq', 'fever', 'climate-fever', 'hotpotqa',
    'webis-touche2020', 'msmarco', 'quora', 'arguana'
]

CQADUPSTACK_DOMAINS = [
    'android', 'english', 'gaming', 'gis', 'mathematica', 'physics',
    'programmers', 'stats', 'tex', 'unix', 'webmasters', 'wordpress'
]

ALL_BEIR_DATASETS = BEIR_MAIN_DATASETS + [f'cqadupstack-{d}' for d in CQADUPSTACK_DOMAINS]


# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    BOLD_GREEN = '\033[1;92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def color_value(value: float, color: str, width: int = 20) -> str:
    """Format a value with color, properly handling column width."""
    formatted = f"{value:.3f}"
    padded = f"{formatted:<{width}}"
    return f"{color}{padded}{Colors.RESET}"


def plain_value(value: float, width: int = 20) -> str:
    """Format a value without color, with proper column width."""
    return f"{value:<{width}.3f}"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare two head configurations with statistical significance testing'
    )
    parser.add_argument('--config1', type=str, required=True,
                        help='Path to first head configuration/weight file')
    parser.add_argument('--config2', type=str, required=True,
                        help='Path to second head configuration/weight file')
    parser.add_argument('--llm', type=str, default='mistral',
                        choices=['mistral', 'llama', 'phi', 'granite'],
                        help='LLM model to use')
    parser.add_argument('--feature_dir', type=str, required=True,
                        help='Directory containing feature .npz files')
    parser.add_argument('--k', type=int, default=10,
                        help='K value for feature files - matches *_k{K}.npz pattern')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of top heads to use for evaluation')
    parser.add_argument('--num_permutations', type=int, default=10000,
                        help='Number of random permutations for significance test')
    parser.add_argument('--metric', type=str, default='NDCG@10',
                        help='Metric to use for comparison (default: NDCG@10)')
    parser.add_argument('--beir_dir', type=str, default=None,
                        help='Path to BEIR data directory containing qrels files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for detailed results (JSON format)')
    parser.add_argument('--n_jobs', type=int, default=4,
                        help='Number of parallel workers for evaluation')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed progress information')
    return parser.parse_args()


def get_config_name(config_path: str) -> str:
    """Extract configuration name from file path."""
    return Path(config_path).stem


def load_head_weights(weight_file: str, num_heads_per_layer: int, num_layers: int) -> Tuple[np.ndarray, dict]:
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

    if 'all_weights' in data:
        # BCE weights format
        for key, w in data['all_weights'].items():
            layer, head = map(int, key.split('-'))
            idx = layer * num_heads_per_layer + head
            if idx < total_heads:
                weights[idx] = w
        metadata = {'type': 'bce', 'nonzero': sum(1 for w in weights if abs(w) > 1e-6)}
    else:
        # CoRe scores format
        for key, scores in data.items():
            layer, head = map(int, key.split('-'))
            idx = layer * num_heads_per_layer + head
            if idx < total_heads:
                weights[idx] = np.mean(scores) if isinstance(scores, list) else scores
        metadata = {'type': 'core', 'nonzero': sum(1 for w in weights if abs(w) > 1e-6)}

    return weights, metadata


def get_top_k_heads(weights: np.ndarray, k: int) -> np.ndarray:
    """Get indices of top-k heads by absolute weight."""
    abs_weights = np.abs(weights)
    return np.argsort(abs_weights)[-k:][::-1]


def compute_scores(features: np.ndarray, weights: np.ndarray, top_k: Optional[int] = None) -> np.ndarray:
    """Compute document scores using head weights."""
    if top_k is not None and top_k < len(weights):
        top_indices = get_top_k_heads(weights, top_k)
        mask = np.zeros_like(weights)
        mask[top_indices] = 1
        weights = weights * mask
    return features @ weights


def load_qrels_file(qrels_path: str) -> Dict[str, Dict[str, int]]:
    """Load qrels from a file (TSV or TREC format)."""
    qrels = {}
    qrels_path = Path(qrels_path)

    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels file not found: {qrels_path}")

    with open(qrels_path, 'r') as f:
        first_line = f.readline().strip()
        f.seek(0)

        if first_line.startswith('query-id') or first_line.startswith('query_id'):
            import csv
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                q_id = str(row.get('query-id', row.get('query_id', '')))
                d_id = str(row.get('corpus-id', row.get('corpus_id', row.get('doc-id', row.get('doc_id', '')))))
                score = int(row.get('score', row.get('relevance', 0)))
                if q_id and d_id:
                    if q_id not in qrels:
                        qrels[q_id] = {}
                    if score > 0:
                        qrels[q_id][d_id] = score
        else:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    if len(parts) == 3:
                        q_id, d_id, score = parts
                    else:
                        q_id, _, d_id, score = parts[:4]
                    q_id, d_id = str(q_id), str(d_id)
                    try:
                        score = int(score)
                    except ValueError:
                        score = int(float(score))
                    if q_id not in qrels:
                        qrels[q_id] = {}
                    if score > 0:
                        qrels[q_id][d_id] = score

    return qrels


def get_qrels_path(beir_dir: Path, dataset: str) -> Path:
    """Get the qrels file path for a BEIR dataset."""
    if dataset.startswith('cqadupstack-'):
        domain = dataset.split('-', 1)[1]
        nested_path = beir_dir / 'cqadupstack' / domain / 'qrels' / 'test.tsv'
        if nested_path.exists():
            return nested_path
        flat_path = beir_dir / dataset / 'qrels' / 'test.tsv'
        if flat_path.exists():
            return flat_path
        return nested_path
    return beir_dir / dataset / 'qrels' / 'test.tsv'


def find_feature_files(feature_dir: Path, k: int) -> Dict[str, Path]:
    """Find all .npz feature files for a given k value."""
    pattern = f"*_k{k}.npz"
    files = list(feature_dir.glob(pattern))

    dataset_files = {}
    for f in files:
        stem = f.stem
        if stem.startswith('attention_features_'):
            stem = stem[len('attention_features_'):]
        if stem.endswith(f'_k{k}'):
            dataset = stem[:-len(f'_k{k}')]
        else:
            continue
        dataset_files[dataset] = f

    return dataset_files


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Compute DCG@k."""
    relevances = np.asarray(relevances)[:k]
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return float(np.sum(relevances / discounts))


def ndcg_at_k(relevances: np.ndarray, k: int, ideal_relevances: Optional[np.ndarray] = None) -> float:
    """
    Compute NDCG@k.

    Args:
        relevances: relevance scores in ranked order
        k: cutoff
        ideal_relevances: all relevance scores for computing ideal DCG (may include docs not retrieved)
    """
    dcg = dcg_at_k(relevances, k)
    if ideal_relevances is None:
        ideal_relevances = relevances
    ideal_sorted = np.sort(ideal_relevances)[::-1]
    idcg = dcg_at_k(ideal_sorted, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_per_query(
    features: np.ndarray,
    weights: np.ndarray,
    query_ids: np.ndarray,
    doc_ids: np.ndarray,
    docs_per_query: np.ndarray,
    top_k_heads: int,
    metric: str,
    external_qrels: Optional[Dict[str, Dict[str, int]]] = None
) -> Dict[str, float]:
    """
    Evaluate ranking and return per-query metric scores.

    Args:
        features: (num_docs, num_heads) attention features
        weights: (num_heads,) head weights
        query_ids: (num_docs,) query IDs
        doc_ids: (num_docs,) document IDs
        docs_per_query: (num_queries,) docs per query
        top_k_heads: number of top heads to use
        metric: metric name (e.g., 'NDCG@10')
        external_qrels: optional external qrels for proper NDCG computation

    Returns:
        Dict mapping query_id to metric score
    """
    # Parse metric
    metric_upper = metric.upper()
    if metric_upper.startswith('NDCG@'):
        k = int(metric_upper.split('@')[1])
        metric_func = lambda rels, ideal_rels: ndcg_at_k(rels, k, ideal_rels)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Compute scores
    scores = compute_scores(features, weights, top_k=top_k_heads)

    # Compute per-query metrics
    per_query_metrics = {}
    doc_offset = 0

    for q_idx in range(len(docs_per_query)):
        n_docs = docs_per_query[q_idx]
        q_scores = scores[doc_offset:doc_offset + n_docs]
        q_doc_ids = doc_ids[doc_offset:doc_offset + n_docs]
        q_id = str(query_ids[doc_offset])
        doc_offset += n_docs

        # Sort by score descending
        ranking = np.argsort(-q_scores)
        ranked_doc_ids = q_doc_ids[ranking]

        # Get relevance labels
        if external_qrels is not None and q_id in external_qrels:
            q_qrels = external_qrels[q_id]
            # Relevances for ranked docs
            ranked_relevances = np.array([q_qrels.get(str(d_id), 0) for d_id in ranked_doc_ids])
            # All relevances for ideal DCG (including docs not retrieved)
            all_relevances = np.array(list(q_qrels.values()))
        else:
            # No external qrels - skip this query
            continue

        # Compute metric
        per_query_metrics[q_id] = metric_func(ranked_relevances, all_relevances)

    return per_query_metrics


def evaluate_dataset(
    dataset: str,
    feature_file: Path,
    weights1: np.ndarray,
    weights2: np.ndarray,
    llm: str,
    top_k_heads: int,
    metric: str,
    qrels_file: Optional[Path] = None
) -> Tuple[str, Dict[str, float], Dict[str, float]]:
    """
    Evaluate a single dataset with both weight configurations.

    Returns:
        (dataset_name, per_query_scores_config1, per_query_scores_config2)
    """
    # Load features
    X, y, docs_per_query, query_ids, doc_ids = load_features(
        feature_file=str(feature_file),
        llm_name=llm,
        return_ids=True
    )

    if docs_per_query is None:
        docs_per_query = np.array([50] * (len(y) // 50))

    # Load external qrels
    external_qrels = None
    if qrels_file is not None and qrels_file.exists():
        external_qrels = load_qrels_file(str(qrels_file))

    # Evaluate with config1
    scores1 = evaluate_per_query(
        X, weights1, query_ids, doc_ids, docs_per_query,
        top_k_heads, metric, external_qrels
    )

    # Evaluate with config2
    scores2 = evaluate_per_query(
        X, weights2, query_ids, doc_ids, docs_per_query,
        top_k_heads, metric, external_qrels
    )

    return dataset, scores1, scores2


def permutation_test(
    scores1: Dict[str, float],
    scores2: Dict[str, float],
    num_permutations: int = 10000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Perform permutation test on per-query scores.

    For each query, we have paired observations (score1, score2).
    Under H0, the labels are exchangeable.

    Args:
        scores1: {query_id: score} for config1
        scores2: {query_id: score} for config2
        num_permutations: number of permutations

    Returns:
        Dictionary with test results
    """
    # Get common queries
    common_queries = sorted(set(scores1.keys()) & set(scores2.keys()))
    n = len(common_queries)

    if n == 0:
        return {
            'observed_diff': 0.0,
            'mean_score1': 0.0,
            'mean_score2': 0.0,
            'p_value': 1.0,
            'num_queries': 0,
            'num_permutations': num_permutations,
            'significant_at_0.05': False,
            'significant_at_0.01': False,
        }

    s1 = np.array([scores1[q] for q in common_queries])
    s2 = np.array([scores2[q] for q in common_queries])

    # Observed test statistic: mean difference
    observed_diff = np.mean(s1 - s2)

    # Permutation test
    rng = np.random.RandomState(seed)
    count_extreme = 0

    for _ in range(num_permutations):
        # For each query, randomly swap config1/config2
        swap_mask = rng.randint(0, 2, n).astype(bool)
        perm_s1 = np.where(swap_mask, s2, s1)
        perm_s2 = np.where(swap_mask, s1, s2)
        perm_diff = np.mean(perm_s1 - perm_s2)

        if np.abs(perm_diff) >= np.abs(observed_diff):
            count_extreme += 1

    p_value = count_extreme / num_permutations

    return {
        'observed_diff': float(observed_diff),
        'mean_score1': float(np.mean(s1)),
        'mean_score2': float(np.mean(s2)),
        'p_value': float(p_value),
        'num_queries': n,
        'num_permutations': num_permutations,
        'significant_at_0.05': p_value < 0.05,
        'significant_at_0.01': p_value < 0.01,
    }


def run_comparison(args):
    """Run full comparison pipeline."""
    log_command()

    feature_dir = Path(args.feature_dir)
    config1_path = Path(args.config1)
    config2_path = Path(args.config2)
    beir_dir = Path(args.beir_dir) if args.beir_dir else None

    config1_name = get_config_name(str(config1_path))
    config2_name = get_config_name(str(config2_path))

    print("=" * 80)
    print("Comparing Head Configurations")
    print("=" * 80)
    print(f"Config 1: {config1_name}")
    print(f"Config 2: {config2_name}")
    print(f"LLM: {args.llm}")
    print(f"Feature dir: {feature_dir}")
    print(f"K value: {args.k}")
    print(f"Num heads: {args.num_heads}")
    print(f"Metric: {args.metric}")
    print(f"Permutations: {args.num_permutations}")
    print("=" * 80)

    # Get model config
    num_layers, num_heads_per_layer = get_head_info(args.llm)

    # Load weights
    print("\nLoading head weights...")
    weights1, meta1 = load_head_weights(str(config1_path), num_heads_per_layer, num_layers)
    weights2, meta2 = load_head_weights(str(config2_path), num_heads_per_layer, num_layers)
    print(f"  Config 1 ({meta1['type']}): {meta1['nonzero']} non-zero heads")
    print(f"  Config 2 ({meta2['type']}): {meta2['nonzero']} non-zero heads")

    # Find feature files
    print(f"\nFinding feature files for k={args.k}...")
    dataset_files = find_feature_files(feature_dir, args.k)
    print(f"Found {len(dataset_files)} datasets")

    # Filter to BEIR datasets only
    beir_datasets = {d: f for d, f in dataset_files.items() if d in ALL_BEIR_DATASETS}
    if len(beir_datasets) < len(BEIR_MAIN_DATASETS):
        missing = set(BEIR_MAIN_DATASETS) - set(beir_datasets.keys())
        print(f"Warning: Missing {len(missing)} main BEIR datasets: {sorted(missing)}")

    if not beir_datasets:
        print("Error: No BEIR datasets found")
        return

    # Evaluate all datasets
    print(f"\nEvaluating {len(beir_datasets)} datasets...")

    # Store per-query scores for each dataset
    all_scores1 = {}  # {dataset: {query_id: score}}
    all_scores2 = {}

    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        futures = {}
        for dataset, feature_file in beir_datasets.items():
            qrels_file = None
            if beir_dir is not None:
                qrels_file = get_qrels_path(beir_dir, dataset)
                if not qrels_file.exists():
                    if args.verbose:
                        print(f"  Warning: qrels not found for {dataset}")
                    qrels_file = None

            future = executor.submit(
                evaluate_dataset,
                dataset, feature_file, weights1, weights2,
                args.llm, args.num_heads, args.metric, qrels_file
            )
            futures[future] = dataset

        for future in as_completed(futures):
            dataset = futures[future]
            try:
                _, scores1, scores2 = future.result()
                all_scores1[dataset] = scores1
                all_scores2[dataset] = scores2
                if args.verbose:
                    n_queries = len(scores1)
                    avg1 = np.mean(list(scores1.values())) if scores1 else 0
                    avg2 = np.mean(list(scores2.values())) if scores2 else 0
                    print(f"  {dataset}: {n_queries} queries, avg1={avg1:.3f}, avg2={avg2:.3f}")
            except Exception as e:
                print(f"  Error evaluating {dataset}: {e}")

    # Compute per-dataset significance tests
    print("\nRunning per-dataset significance tests...")
    dataset_results = {}  # {dataset: {'mean1', 'mean2', 'p_value', ...}}

    for dataset in sorted(all_scores1.keys()):
        scores1 = all_scores1[dataset]
        scores2 = all_scores2.get(dataset, {})

        if not scores1 or not scores2:
            continue

        sig_test = permutation_test(scores1, scores2, args.num_permutations)
        dataset_results[dataset] = sig_test

    # Aggregate cqadupstack scores (pool all queries for significance test)
    cqa_scores1 = {}
    cqa_scores2 = {}
    cqa_domain_means1 = []  # For macro average display
    cqa_domain_means2 = []
    for dataset in all_scores1:
        if dataset.startswith('cqadupstack-'):
            for q_id, score in all_scores1[dataset].items():
                cqa_scores1[f"{dataset}:{q_id}"] = score
            for q_id, score in all_scores2.get(dataset, {}).items():
                cqa_scores2[f"{dataset}:{q_id}"] = score
            # Collect domain means for macro average
            if all_scores1[dataset]:
                cqa_domain_means1.append(np.mean(list(all_scores1[dataset].values())))
            if dataset in all_scores2 and all_scores2[dataset]:
                cqa_domain_means2.append(np.mean(list(all_scores2[dataset].values())))

    if cqa_scores1:
        cqa_sig_test = permutation_test(cqa_scores1, cqa_scores2, args.num_permutations)
        # Override mean scores with macro averages for display
        cqa_macro_avg1 = np.mean(cqa_domain_means1) if cqa_domain_means1 else 0.0
        cqa_macro_avg2 = np.mean(cqa_domain_means2) if cqa_domain_means2 else 0.0
        cqa_sig_test['mean_score1'] = cqa_macro_avg1
        cqa_sig_test['mean_score2'] = cqa_macro_avg2
        cqa_sig_test['observed_diff'] = cqa_macro_avg1 - cqa_macro_avg2
        dataset_results['cqadupstack'] = cqa_sig_test

    # Compute overall BEIR significance (all queries pooled for significance test)
    all_query_scores1 = {}
    all_query_scores2 = {}

    for dataset in BEIR_MAIN_DATASETS:
        if dataset in all_scores1:
            for q_id, score in all_scores1[dataset].items():
                all_query_scores1[f"{dataset}:{q_id}"] = score
            for q_id, score in all_scores2.get(dataset, {}).items():
                all_query_scores2[f"{dataset}:{q_id}"] = score

    # Add cqadupstack queries
    all_query_scores1.update(cqa_scores1)
    all_query_scores2.update(cqa_scores2)

    beir_sig_test = permutation_test(all_query_scores1, all_query_scores2, args.num_permutations)

    # Compute BEIR macro averages for display (average of per-dataset means)
    beir_dataset_means1 = []
    beir_dataset_means2 = []
    for dataset in BEIR_MAIN_DATASETS:
        if dataset in dataset_results:
            beir_dataset_means1.append(dataset_results[dataset]['mean_score1'])
            beir_dataset_means2.append(dataset_results[dataset]['mean_score2'])
    # Add cqadupstack as one dataset
    if 'cqadupstack' in dataset_results:
        beir_dataset_means1.append(dataset_results['cqadupstack']['mean_score1'])
        beir_dataset_means2.append(dataset_results['cqadupstack']['mean_score2'])

    beir_macro_avg1 = np.mean(beir_dataset_means1) if beir_dataset_means1 else 0.0
    beir_macro_avg2 = np.mean(beir_dataset_means2) if beir_dataset_means2 else 0.0
    # Override for display
    beir_sig_test['mean_score1'] = beir_macro_avg1
    beir_sig_test['mean_score2'] = beir_macro_avg2
    beir_sig_test['observed_diff'] = beir_macro_avg1 - beir_macro_avg2

    # Print results
    print("\n" + "=" * 100)
    print("Per-Dataset Results with Statistical Significance")
    print("=" * 100)
    print(f"{'Dataset':<30} {'Config1':<20} {'Config2':<20} {'Diff':<12} {'p-value':<12} {'Sig':<5}")
    print("-" * 100)
    print(f"Note: {Colors.BOLD_GREEN}Green values{Colors.RESET} indicate statistically significant improvements (p < 0.05)")

    # Main datasets
    main_datasets_in_results = [d for d in BEIR_MAIN_DATASETS if d in dataset_results]
    for dataset in sorted(main_datasets_in_results):
        result = dataset_results[dataset]
        score1 = result['mean_score1']
        score2 = result['mean_score2']
        diff = result['observed_diff']
        p_value = result['p_value']

        sig_marker = "**" if result['significant_at_0.01'] else "*" if result['significant_at_0.05'] else ""

        if result['significant_at_0.05']:
            if diff > 0:
                score1_str = color_value(score1, Colors.BOLD_GREEN)
                score2_str = plain_value(score2)
            else:
                score1_str = plain_value(score1)
                score2_str = color_value(score2, Colors.BOLD_GREEN)
        else:
            score1_str = plain_value(score1)
            score2_str = plain_value(score2)

        print(f"{dataset:<30} {score1_str} {score2_str} {diff:+12.3f} {p_value:<12.6f} {sig_marker:<5}")

    # CQADupstack average
    if 'cqadupstack' in dataset_results:
        print("-" * 100)
        result = dataset_results['cqadupstack']
        score1 = result['mean_score1']
        score2 = result['mean_score2']
        diff = result['observed_diff']
        p_value = result['p_value']

        sig_marker = "**" if result['significant_at_0.01'] else "*" if result['significant_at_0.05'] else ""

        if result['significant_at_0.05']:
            if diff > 0:
                score1_str = color_value(score1, Colors.BOLD_GREEN)
                score2_str = plain_value(score2)
            else:
                score1_str = plain_value(score1)
                score2_str = color_value(score2, Colors.BOLD_GREEN)
        else:
            score1_str = plain_value(score1)
            score2_str = plain_value(score2)

        print(f"{'cqadupstack-average':<30} {score1_str} {score2_str} {diff:+12.3f} {p_value:<12.6f} {sig_marker:<5}")

    # BEIR macro-average
    print("-" * 100)
    macro_avg1 = beir_sig_test['mean_score1']
    macro_avg2 = beir_sig_test['mean_score2']
    beir_diff = beir_sig_test['observed_diff']
    beir_p_value = beir_sig_test['p_value']

    sig_marker = "**" if beir_sig_test['significant_at_0.01'] else "*" if beir_sig_test['significant_at_0.05'] else ""

    if beir_sig_test['significant_at_0.05']:
        if beir_diff > 0:
            macro_avg1_str = color_value(macro_avg1, Colors.BOLD_GREEN)
            macro_avg2_str = plain_value(macro_avg2)
        else:
            macro_avg1_str = plain_value(macro_avg1)
            macro_avg2_str = color_value(macro_avg2, Colors.BOLD_GREEN)
    else:
        macro_avg1_str = plain_value(macro_avg1)
        macro_avg2_str = plain_value(macro_avg2)

    print(f"{'BEIR Macro Average':<30} {macro_avg1_str} {macro_avg2_str} "
          f"{beir_diff:+12.3f} {beir_p_value:<12.6f} {sig_marker:<5}")

    print("\n* p < 0.05 (significant)")
    print("** p < 0.01 (highly significant)")

    # Detailed significance test info
    print("\n" + "=" * 80)
    print("Overall BEIR Statistical Significance Test")
    print("=" * 80)
    print(f"Config 1 mean: {beir_sig_test['mean_score1']:.3f}")
    print(f"Config 2 mean: {beir_sig_test['mean_score2']:.3f}")
    print(f"Observed difference: {beir_sig_test['observed_diff']:+.3f}")
    print(f"Number of queries: {beir_sig_test['num_queries']}")
    print(f"Number of permutations: {beir_sig_test['num_permutations']}")
    print(f"P-value (two-tailed): {beir_sig_test['p_value']:.6f}")
    print(f"Significant at α=0.05: {'Yes' if beir_sig_test['significant_at_0.05'] else 'No'}")
    print(f"Significant at α=0.01: {'Yes' if beir_sig_test['significant_at_0.01'] else 'No'}")

    # Interpretation
    print("\n" + "=" * 80)
    print("Interpretation")
    print("=" * 80)

    if beir_sig_test['observed_diff'] > 0:
        better_config = config1_name
    else:
        better_config = config2_name

    print(f"Config '{better_config}' performs better by {abs(beir_sig_test['observed_diff']):.3f} on average.")

    if beir_sig_test['significant_at_0.01']:
        print("This difference is HIGHLY SIGNIFICANT (p < 0.01).")
    elif beir_sig_test['significant_at_0.05']:
        print("This difference is SIGNIFICANT (p < 0.05).")
    else:
        print("This difference is NOT statistically significant (p >= 0.05).")
        print("The observed difference could be due to random chance.")

    # Count significant wins
    sig_wins_1 = sum(1 for d, r in dataset_results.items() if r['significant_at_0.05'] and r['observed_diff'] > 0)
    sig_wins_2 = sum(1 for d, r in dataset_results.items() if r['significant_at_0.05'] and r['observed_diff'] < 0)

    print(f"\nPer-dataset significant wins (p < 0.05):")
    print(f"  Config 1 ({config1_name}): {sig_wins_1}")
    print(f"  Config 2 ({config2_name}): {sig_wins_2}")

    # Save results
    if args.output:
        output_data = {
            'config1': config1_name,
            'config2': config2_name,
            'llm': args.llm,
            'metric': args.metric,
            'k': args.k,
            'num_heads': args.num_heads,
            'num_permutations': args.num_permutations,
            'beir_overall': {
                'mean_score1': beir_sig_test['mean_score1'],
                'mean_score2': beir_sig_test['mean_score2'],
                'observed_diff': beir_sig_test['observed_diff'],
                'p_value': beir_sig_test['p_value'],
                'num_queries': beir_sig_test['num_queries'],
                'significant_at_0.05': beir_sig_test['significant_at_0.05'],
                'significant_at_0.01': beir_sig_test['significant_at_0.01'],
            },
            'per_dataset': {
                dataset: {
                    'mean_score1': r['mean_score1'],
                    'mean_score2': r['mean_score2'],
                    'observed_diff': r['observed_diff'],
                    'p_value': r['p_value'],
                    'num_queries': r['num_queries'],
                    'significant_at_0.05': r['significant_at_0.05'],
                    'significant_at_0.01': r['significant_at_0.01'],
                }
                for dataset, r in dataset_results.items()
            }
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    args = parse_args_with_config()
    run_comparison(args)
