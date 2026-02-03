#!/usr/bin/env python3
"""
Compare two head configuration files using BEIR evaluation and statistical significance testing.

This script:
1. Takes two head weight/configuration files as input (CoRe or trained BCE/InfoNCE weights)
2. Runs reranking with both configurations on BEIR datasets using evaluate_beir_aggregate.py logic
3. Evaluates both using NDCG metrics with proper cqadupstack aggregation
4. Performs randomization tests for statistical significance at:
   - Individual dataset level
   - Overall BEIR level (with cqadupstack datasets grouped)

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

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

# Add scripts directory to path to import evaluation functions
SCRIPT_DIR = Path(__file__).parent.parent / 'scripts'
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_beir_aggregate import (
    BEIR_MAIN_DATASETS,
    CQADUPSTACK_DOMAINS,
    find_feature_files,
    aggregate_cqadupstack,
    compute_beir_average,
    parse_results,
    get_qrels_path,
    run_reranking as run_single_reranking
)

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    BOLD_GREEN = '\033[1;92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare two head configurations with statistical significance testing'
    )
    parser.add_argument(
        '--config1',
        type=str,
        required=True,
        help='Path to first head configuration/weight file'
    )
    parser.add_argument(
        '--config2',
        type=str,
        required=True,
        help='Path to second head configuration/weight file'
    )
    parser.add_argument(
        '--llm',
        type=str,
        default='mistral',
        choices=['mistral', 'llama', 'phi', 'granite'],
        help='LLM model to use'
    )
    parser.add_argument(
        '--feature_dir',
        type=str,
        required=True,
        help='Directory containing feature .npz files (e.g., ../head_data/mistral)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='K value for feature files - matches *_k{K}.npz pattern (default: 10)'
    )
    parser.add_argument(
        '--num_heads',
        type=int,
        default=8,
        help='Number of top heads to use for evaluation (default: 8)'
    )
    parser.add_argument(
        '--num_permutations',
        type=int,
        default=10000,
        help='Number of random permutations for significance test (default: 10000)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='ndcg@10',
        help='Metric to use for comparison (default: ndcg@10)'
    )
    parser.add_argument(
        '--evaluator',
        type=str,
        default='beir',
        choices=['custom', 'beir'],
        help='Evaluator to use (default: beir)'
    )
    parser.add_argument(
        '--beir_dir',
        type=str,
        default=None,
        help='Path to BEIR data directory containing qrels files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for detailed results (JSON format)'
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=4,
        help='Number of parallel workers for evaluation (default: 4)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Print detailed progress information'
    )

    return parser.parse_args()


def get_config_name(config_path: str) -> str:
    """Extract configuration name from file path."""
    return os.path.splitext(os.path.basename(config_path))[0]


def run_evaluation_for_config(
    config_path: Path,
    llm: str,
    feature_dir: Path,
    k: int,
    num_heads: int,
    evaluator: str,
    beir_dir: Path,
    output_dir: Path,
    n_jobs: int,
    verbose: bool
) -> Dict[str, Dict[str, float]]:
    """
    Run BEIR evaluation for a single configuration.

    Returns:
        Dict mapping dataset -> metrics
    """
    # Find feature files
    dataset_files = find_feature_files(feature_dir, k)

    if not dataset_files:
        raise ValueError(f"No feature files found for k={k} in {feature_dir}")

    # Check for required datasets
    missing_main = set(BEIR_MAIN_DATASETS) - set(dataset_files.keys())
    if missing_main:
        raise ValueError(f"Missing required BEIR datasets: {sorted(missing_main)}")

    if verbose:
        print(f"Found {len(dataset_files)} datasets for k={k}")

    # Run evaluations in parallel
    results = {}
    failed = []

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {}
        for dataset, feature_file in dataset_files.items():
            # Get qrels file path if beir_dir is provided
            qrels_file = None
            if beir_dir is not None:
                qrels_file = get_qrels_path(beir_dir, dataset)
                if not qrels_file.exists():
                    if verbose:
                        print(f"Warning: qrels file not found for {dataset}: {qrels_file}")
                    qrels_file = None

            future = executor.submit(
                run_single_reranking,
                dataset,
                feature_file,
                llm,
                config_path,
                [num_heads],  # top_k_heads
                [1, 5, 10],  # ks
                evaluator,
                ['ndcg', 'p', 'm', 'map', 'mrr'],  # metrics
                output_dir,
                True,  # no_baseline
                True,  # no_oracle
                False,  # fusion
                60,  # rrf_k
                qrels_file
            )
            futures[future] = dataset

        for future in as_completed(futures):
            dataset = futures[future]
            try:
                dataset_name, success, result = future.result()
                if success:
                    if verbose:
                        print(f"✓ {dataset_name}")
                    results[dataset_name] = result
                else:
                    if verbose:
                        print(f"✗ {dataset_name}: {result}")
                    failed.append(dataset_name)
            except Exception as e:
                if verbose:
                    print(f"✗ {dataset}: {e}")
                failed.append(dataset)

    if failed:
        print(f"Warning: {len(failed)} datasets failed: {failed}")

    # Parse all results
    dataset_results = {}
    for dataset, output_file_path in results.items():
        try:
            parsed = parse_results(Path(output_file_path))
            # Extract metrics for the config we're testing
            # The key format is "top-{num_heads}_{weights}"
            for config_key, metrics in parsed.items():
                if config_key.startswith(f'top-{num_heads}'):
                    dataset_results[dataset] = metrics
                    break
        except Exception as e:
            if verbose:
                print(f"Error parsing {dataset}: {e}")

    return dataset_results


def randomization_test(
    scores1: Dict[str, float],
    scores2: Dict[str, float],
    num_permutations: int = 10000
) -> Dict[str, float]:
    """
    Perform randomization test (permutation test) for statistical significance.

    Args:
        scores1: Per-query scores for system 1 {query_id: score}
        scores2: Per-query scores for system 2 {query_id: score}
        num_permutations: Number of random permutations

    Returns:
        Dictionary with test results including p-value
    """
    # For dataset-level comparison, we just have single values
    if isinstance(scores1, (int, float)):
        scores1 = {'single': scores1}
    if isinstance(scores2, (int, float)):
        scores2 = {'single': scores2}

    # Ensure we have the same queries
    common_queries = sorted(set(scores1.keys()) & set(scores2.keys()))
    if len(common_queries) == 0:
        raise ValueError("No common queries between the two systems")

    # Get scores for common queries
    s1 = np.array([scores1[q] for q in common_queries])
    s2 = np.array([scores2[q] for q in common_queries])

    # Compute observed difference
    observed_diff = np.mean(s1 - s2)

    # Perform permutation test
    permuted_diffs = []
    rng = np.random.RandomState(42)

    for _ in range(num_permutations):
        # Random swap for each query
        swap_mask = rng.randint(0, 2, len(common_queries)).astype(bool)
        perm_s1 = np.where(swap_mask, s2, s1)
        perm_s2 = np.where(swap_mask, s1, s2)

        # Compute difference for this permutation
        perm_diff = np.mean(perm_s1 - perm_s2)
        permuted_diffs.append(perm_diff)

    permuted_diffs = np.array(permuted_diffs)

    # Calculate p-value (two-tailed)
    p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))

    return {
        'observed_diff': float(observed_diff),
        'mean_score1': float(np.mean(s1)),
        'mean_score2': float(np.mean(s2)),
        'p_value': float(p_value),
        'num_queries': len(common_queries),
        'num_permutations': num_permutations,
        'significant_at_0.05': p_value < 0.05,
        'significant_at_0.01': p_value < 0.01,
        'permuted_diffs_std': float(np.std(permuted_diffs))
    }


def macro_average_randomization_test(
    dataset_scores1: Dict[str, float],
    dataset_scores2: Dict[str, float],
    num_permutations: int = 10000
) -> Dict[str, float]:
    """
    Perform randomization test on macro-averaged scores across datasets.

    Args:
        dataset_scores1: Dict mapping dataset -> score for system 1
        dataset_scores2: Dict mapping dataset -> score for system 2
        num_permutations: Number of random permutations

    Returns:
        Dictionary with test results
    """
    datasets = sorted(set(dataset_scores1.keys()) & set(dataset_scores2.keys()))
    if len(datasets) == 0:
        raise ValueError("No common datasets between the two systems")

    # Get scores as arrays
    s1 = np.array([dataset_scores1[d] for d in datasets])
    s2 = np.array([dataset_scores2[d] for d in datasets])

    # Compute observed macro-average difference
    observed_macro_avg1 = np.mean(s1)
    observed_macro_avg2 = np.mean(s2)
    observed_diff = observed_macro_avg1 - observed_macro_avg2

    # Perform permutation test
    permuted_diffs = []
    rng = np.random.RandomState(42)

    for _ in range(num_permutations):
        # Random swap for each dataset
        swap_mask = rng.randint(0, 2, len(datasets)).astype(bool)
        perm_s1 = np.where(swap_mask, s2, s1)
        perm_s2 = np.where(swap_mask, s1, s2)

        # Compute macro-average difference for this permutation
        perm_diff = np.mean(perm_s1) - np.mean(perm_s2)
        permuted_diffs.append(perm_diff)

    permuted_diffs = np.array(permuted_diffs)

    # Calculate p-value (two-tailed)
    p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))

    return {
        'observed_diff': float(observed_diff),
        'macro_avg1': float(observed_macro_avg1),
        'macro_avg2': float(observed_macro_avg2),
        'p_value': float(p_value),
        'num_datasets': len(datasets),
        'num_permutations': num_permutations,
        'significant_at_0.05': p_value < 0.05,
        'significant_at_0.01': p_value < 0.01,
        'permuted_diffs_std': float(np.std(permuted_diffs))
    }


def run_comparison(args):
    """Run full comparison pipeline."""

    # Resolve paths
    feature_dir = Path(args.feature_dir)
    config1_path = Path(args.config1)
    config2_path = Path(args.config2)
    beir_dir = Path(args.beir_dir) if args.beir_dir else None

    # Get configuration names
    config1_name = get_config_name(str(config1_path))
    config2_name = get_config_name(str(config2_path))

    # Create temp output directories
    output_dir1 = Path(f'/tmp/compare_heads_{config1_name}')
    output_dir2 = Path(f'/tmp/compare_heads_{config2_name}')
    output_dir1.mkdir(parents=True, exist_ok=True)
    output_dir2.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print(f"Comparing Head Configurations")
    print("="*80)
    print(f"Config 1: {config1_name}")
    print(f"Config 2: {config2_name}")
    print(f"LLM: {args.llm}")
    print(f"Feature dir: {feature_dir}")
    print(f"K value: {args.k}")
    print(f"Num heads: {args.num_heads}")
    print(f"Metric: {args.metric}")
    print(f"Permutations: {args.num_permutations}")
    print("="*80)

    # Run evaluation for config 1
    print(f"\nEvaluating Config 1 ({config1_name})...")
    dataset_results1 = run_evaluation_for_config(
        config1_path, args.llm, feature_dir, args.k, args.num_heads,
        args.evaluator, beir_dir, output_dir1, args.n_jobs, args.verbose
    )

    # Run evaluation for config 2
    print(f"\nEvaluating Config 2 ({config2_name})...")
    dataset_results2 = run_evaluation_for_config(
        config2_path, args.llm, feature_dir, args.k, args.num_heads,
        args.evaluator, beir_dir, output_dir2, args.n_jobs, args.verbose
    )

    # Extract the specific metric we're comparing
    metric_key = args.metric.lower()

    # Build dataset scores for both configs
    dataset_scores1 = {}
    dataset_scores2 = {}

    for dataset in set(dataset_results1.keys()) | set(dataset_results2.keys()):
        if dataset in dataset_results1 and metric_key in dataset_results1[dataset]:
            dataset_scores1[dataset] = dataset_results1[dataset][metric_key]
        if dataset in dataset_results2 and metric_key in dataset_results2[dataset]:
            dataset_scores2[dataset] = dataset_results2[dataset][metric_key]

    # Group datasets
    cqadupstack_datasets = [f'cqadupstack-{d}' for d in CQADUPSTACK_DOMAINS]
    cqa_in_results1 = [d for d in cqadupstack_datasets if d in dataset_scores1]
    cqa_in_results2 = [d for d in cqadupstack_datasets if d in dataset_scores2]
    non_cqa_in_results = sorted(set(dataset_scores1.keys()) | set(dataset_scores2.keys()) - set(cqadupstack_datasets))

    # Print per-dataset results with individual significance tests
    print("\n" + "="*100)
    print("Per-Dataset Results with Statistical Significance")
    print("="*100)
    print(f"{'Dataset':<30} {'Config1':<20} {'Config2':<20} {'Diff':<12} {'p-value':<12} {'Sig':<5}")
    print("-"*100)
    print(f"Note: {Colors.BOLD_GREEN}Green values{Colors.RESET} indicate statistically significant improvements (p < 0.05)")

    per_dataset_significance = {}

    # Non-cqadupstack datasets
    for dataset in non_cqa_in_results:
        score1 = dataset_scores1.get(dataset, 0.0)
        score2 = dataset_scores2.get(dataset, 0.0)
        diff = score1 - score2

        # Run significance test for this dataset
        sig_test = randomization_test(
            {'single': score1},
            {'single': score2},
            num_permutations=args.num_permutations
        )
        per_dataset_significance[dataset] = sig_test

        sig_marker = "**" if sig_test['significant_at_0.01'] else "*" if sig_test['significant_at_0.05'] else ""

        # Color the better config green if significant
        if sig_test['significant_at_0.05']:
            if diff > 0:  # Config1 is better
                score1_str = f"{Colors.BOLD_GREEN}{score1:.4f}{Colors.RESET}"
                score2_str = f"{score2:.4f}"
            else:  # Config2 is better
                score1_str = f"{score1:.4f}"
                score2_str = f"{Colors.BOLD_GREEN}{score2:.4f}{Colors.RESET}"
        else:
            score1_str = f"{score1:.4f}"
            score2_str = f"{score2:.4f}"

        print(f"{dataset:<30} {score1_str:<20} {score2_str:<20} {diff:+12.4f} "
              f"{sig_test['p_value']:<12.6f} {sig_marker:<5}")

    # CQADupstack average (if applicable)
    if cqa_in_results1 or cqa_in_results2:
        print("-"*100)

        # Compute cqadupstack average
        cqa_avg1 = np.mean([dataset_scores1[d] for d in cqa_in_results1]) if cqa_in_results1 else 0.0
        cqa_avg2 = np.mean([dataset_scores2[d] for d in cqa_in_results2]) if cqa_in_results2 else 0.0
        cqa_diff = cqa_avg1 - cqa_avg2

        # Run significance test
        cqa_sig_test = randomization_test(
            {'single': cqa_avg1},
            {'single': cqa_avg2},
            num_permutations=args.num_permutations
        )
        per_dataset_significance['cqadupstack-average'] = cqa_sig_test

        sig_marker = "**" if cqa_sig_test['significant_at_0.01'] else "*" if cqa_sig_test['significant_at_0.05'] else ""

        # Color the better config green if significant
        if cqa_sig_test['significant_at_0.05']:
            if cqa_diff > 0:  # Config1 is better
                cqa_avg1_str = f"{Colors.BOLD_GREEN}{cqa_avg1:.4f}{Colors.RESET}"
                cqa_avg2_str = f"{cqa_avg2:.4f}"
            else:  # Config2 is better
                cqa_avg1_str = f"{cqa_avg1:.4f}"
                cqa_avg2_str = f"{Colors.BOLD_GREEN}{cqa_avg2:.4f}{Colors.RESET}"
        else:
            cqa_avg1_str = f"{cqa_avg1:.4f}"
            cqa_avg2_str = f"{cqa_avg2:.4f}"

        print(f"{'cqadupstack-average':<30} {cqa_avg1_str:<20} {cqa_avg2_str:<20} {cqa_diff:+12.4f} "
              f"{cqa_sig_test['p_value']:<12.6f} {sig_marker:<5}")

    # BEIR macro-average with significance test
    print("-"*100)

    # Build BEIR dataset scores (non-cqa + cqadupstack as single group)
    beir_scores1 = {d: dataset_scores1[d] for d in non_cqa_in_results if d in dataset_scores1}
    beir_scores2 = {d: dataset_scores2[d] for d in non_cqa_in_results if d in dataset_scores2}

    if cqa_in_results1 or cqa_in_results2:
        beir_scores1['cqadupstack'] = cqa_avg1
        beir_scores2['cqadupstack'] = cqa_avg2

    # Compute macro-average
    macro_avg1 = np.mean(list(beir_scores1.values())) if beir_scores1 else 0.0
    macro_avg2 = np.mean(list(beir_scores2.values())) if beir_scores2 else 0.0

    # Run macro-average significance test
    beir_sig_test = macro_average_randomization_test(
        beir_scores1,
        beir_scores2,
        num_permutations=args.num_permutations
    )

    sig_marker = "**" if beir_sig_test['significant_at_0.01'] else "*" if beir_sig_test['significant_at_0.05'] else ""

    # Color the better config green if significant
    beir_diff = macro_avg1 - macro_avg2
    if beir_sig_test['significant_at_0.05']:
        if beir_diff > 0:  # Config1 is better
            macro_avg1_str = f"{Colors.BOLD_GREEN}{macro_avg1:.4f}{Colors.RESET}"
            macro_avg2_str = f"{macro_avg2:.4f}"
        else:  # Config2 is better
            macro_avg1_str = f"{macro_avg1:.4f}"
            macro_avg2_str = f"{Colors.BOLD_GREEN}{macro_avg2:.4f}{Colors.RESET}"
    else:
        macro_avg1_str = f"{macro_avg1:.4f}"
        macro_avg2_str = f"{macro_avg2:.4f}"

    print(f"{'BEIR Macro Average':<30} {macro_avg1_str:<20} {macro_avg2_str:<20} "
          f"{beir_diff:+12.4f} {beir_sig_test['p_value']:<12.6f} {sig_marker:<5}")

    print("\n* p < 0.05 (significant)")
    print("** p < 0.01 (highly significant)")

    # Overall BEIR significance test details
    print("\n" + "="*80)
    print("Overall BEIR Statistical Significance Test")
    print("="*80)
    print(f"Config 1 macro average: {beir_sig_test['macro_avg1']:.4f}")
    print(f"Config 2 macro average: {beir_sig_test['macro_avg2']:.4f}")
    print(f"Observed difference: {beir_sig_test['observed_diff']:+.4f}")
    print(f"Number of dataset groups: {beir_sig_test['num_datasets']}")
    print(f"Number of permutations: {beir_sig_test['num_permutations']}")
    print(f"P-value (two-tailed): {beir_sig_test['p_value']:.6f}")
    print(f"Significant at α=0.05: {'Yes' if beir_sig_test['significant_at_0.05'] else 'No'}")
    print(f"Significant at α=0.01: {'Yes' if beir_sig_test['significant_at_0.01'] else 'No'}")

    # Interpretation
    print("\n" + "="*80)
    print("Interpretation")
    print("="*80)

    if beir_sig_test['observed_diff'] > 0:
        better_config = config1_name
        worse_config = config2_name
    else:
        better_config = config2_name
        worse_config = config1_name

    print(f"Config '{better_config}' performs better by {abs(beir_sig_test['observed_diff']):.4f} on BEIR macro average.")

    if beir_sig_test['significant_at_0.01']:
        print(f"This difference is HIGHLY SIGNIFICANT (p < 0.01).")
    elif beir_sig_test['significant_at_0.05']:
        print(f"This difference is SIGNIFICANT (p < 0.05).")
    else:
        print(f"This difference is NOT statistically significant (p ≥ 0.05).")
        print(f"The observed difference could be due to random chance.")

    # Count significant wins
    sig_wins_config1 = sum(
        1 for dataset, sig in per_dataset_significance.items()
        if dataset in dataset_scores1 and dataset in dataset_scores2
        and dataset_scores1[dataset] > dataset_scores2[dataset]
        and sig['significant_at_0.05']
    )
    sig_wins_config2 = sum(
        1 for dataset, sig in per_dataset_significance.items()
        if dataset in dataset_scores1 and dataset in dataset_scores2
        and dataset_scores2[dataset] > dataset_scores1[dataset]
        and sig['significant_at_0.05']
    )

    print(f"\nPer-dataset significant wins (p < 0.05):")
    print(f"  Config 1 ({config1_name}): {sig_wins_config1}")
    print(f"  Config 2 ({config2_name}): {sig_wins_config2}")

    # Save detailed results
    if args.output:
        output_data = {
            'config1': config1_name,
            'config2': config2_name,
            'llm': args.llm,
            'metric': args.metric,
            'k': args.k,
            'num_heads': args.num_heads,
            'num_permutations': args.num_permutations,
            'beir_macro_average': {
                'config1': float(macro_avg1),
                'config2': float(macro_avg2),
                'difference': float(beir_diff),
                'significance_test': beir_sig_test
            },
            'per_dataset_results': {
                dataset: {
                    'config1': dataset_scores1.get(dataset, 0.0),
                    'config2': dataset_scores2.get(dataset, 0.0),
                    'significance_test': per_dataset_significance.get(dataset, {})
                }
                for dataset in sorted(set(dataset_scores1.keys()) | set(dataset_scores2.keys()))
            }
        }

        # Add cqadupstack average if applicable
        if cqa_in_results1 or cqa_in_results2:
            output_data['cqadupstack_average'] = {
                'config1': float(cqa_avg1),
                'config2': float(cqa_avg2),
                'difference': float(cqa_diff),
                'significance_test': per_dataset_significance.get('cqadupstack-average', {})
            }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nDetailed results saved to: {args.output}")

    print("\n" + "="*80)


if __name__ == '__main__':
    args = parse_args()
    run_comparison(args)
