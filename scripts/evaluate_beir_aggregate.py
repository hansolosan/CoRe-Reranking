#!/usr/bin/env python3
"""
Evaluate BEIR aggregate score by running reranking on all BEIR datasets.

This script:
1. Finds all .npz feature files for a given k value (e.g., *_k10.npz)
2. Runs rerank_with_head_weights.py on each dataset in parallel (configurable workers)
3. Averages cqadupstack-* datasets into a single score
4. Computes the overall BEIR average across datasets (14 main + 1 cqadupstack)
5. Displays results with color-coded highlighting (green=best, yellow=second-best)

BEIR datasets (15 total after cqadupstack aggregation):
- 14 main: trec-covid, nfcorpus, dbpedia-entity, scifact, scidocs, fiqa, nq,
           fever, climate-fever, hotpotqa, webis-touche2020, msmarco, quora, arguana
- 1 aggregated: cqadupstack (average of up to 12 domains: android, english, gaming,
                gis, mathematica, physics, programmers, stats, tex, unix, webmasters,
                wordpress)

Features:
- Parallel execution: Run evaluations concurrently (default: 4 workers, configurable)
- Validation: Ensures all 14 main BEIR datasets are present
- Flexible cqadupstack: Warns but continues if some cqadupstack domains are missing
- Numerical sorting: Results sorted by top-k value (baseline, oracle, then top-1, top-2, etc.)
- Color highlighting: Best score (green) and second-best (yellow) per metric column
  (oracle results excluded from highlighting to avoid skewing comparisons)
- Comprehensive output: Individual JSON files per dataset + aggregate results

Output files:
- {output_dir}/{dataset}_metrics.json - Individual dataset results
- {output_dir}/beir_aggregate.json - Aggregate BEIR scores with breakdowns

Example usage:
    python evaluate_beir_aggregate.py \\
        --llm mistral \\
        --weight_file head_data/mistral/bce_weights_lambda0.0001_n5000.json \\
        --feature_dir head_data/mistral \\
        --k 10 \\
        --top_k_heads 1 2 4 8 16 32 \\
        --n_jobs 8 \\
        --output_dir results/beir_k10

Example output:
    ======================================================================
    BEIR Aggregate Results
    ======================================================================
    Config               Weights    NDCG@1   NDCG@5   NDCG@10  MAP      MRR
    --------------------------------------------------------------------------
    baseline             retriever  0.3245   0.4123   0.4567   0.3890   0.4234
    oracle               gold@1     0.9756   0.9812   0.9845   0.9801   0.9823
    top-1                bce        0.3312   0.4234   0.4678   0.3956   0.4345
    top-8                bce        0.3456   0.4389   0.4823   0.4123   0.4567  <- green
    top-16               bce        0.3512   0.4445   0.4891   0.4189   0.4623  <- yellow
    top-32               bce        0.3489   0.4421   0.4867   0.4156   0.4598
    ======================================================================

Notes:
- All 14 main BEIR datasets must be present (script exits with error if missing)
- Cqadupstack datasets are optional (warns if missing, uses available domains)
- If no cqadupstack datasets found, BEIR average computed across 14 main datasets only
- Results sorted: baseline first, oracle second, then top-k numerically (1, 2, 8, 16, not 1, 16, 2, 8)
- Each top_k_heads value produces a separate aggregate score
- Oracle shows upper bound performance (gold doc moved to rank 1), excluded from color highlighting
- Use --no_oracle to skip oracle evaluation
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple
from utils import log_command

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


def find_feature_files(feature_dir: Path, k: int) -> Dict[str, Path]:
    """
    Find all .npz feature files for a given k value.

    Returns:
        dict mapping dataset name to feature file path
    """
    pattern = f"*_k{k}.npz"
    files = list(feature_dir.glob(pattern))

    # Map files to dataset names
    dataset_files = {}
    for f in files:
        # Extract dataset name from filename
        # Format: attention_features_{dataset}_k{k}.npz
        stem = f.stem
        if stem.startswith('attention_features_'):
            stem = stem[len('attention_features_'):]

        # Remove _k{k} suffix
        if stem.endswith(f'_k{k}'):
            dataset = stem[:-len(f'_k{k}')]
        else:
            continue

        dataset_files[dataset] = f

    return dataset_files


def validate_datasets(dataset_files: Dict[str, Path]) -> Tuple[bool, List[str]]:
    """
    Validate that all BEIR datasets are present and no duplicates.

    Returns:
        (is_valid, missing_datasets)
    """
    found_datasets = set(dataset_files.keys())
    expected_datasets = set(ALL_BEIR_DATASETS)

    missing = expected_datasets - found_datasets
    extra = found_datasets - expected_datasets

    is_valid = len(missing) == 0

    issues = []
    if missing:
        issues.append(f"Missing datasets: {sorted(missing)}")
    if extra:
        issues.append(f"Extra datasets (not in BEIR): {sorted(extra)}")

    return is_valid, issues


def run_reranking(
    dataset: str,
    feature_file: Path,
    llm: str,
    weight_file: Path,
    top_k_heads: List[int],
    ks: List[int],
    evaluator: str,
    metrics: List[str],
    output_dir: Path,
    no_baseline: bool,
    no_oracle: bool
) -> Tuple[str, bool, str]:
    """
    Run rerank_with_head_weights.py for a single dataset.

    Returns:
        (dataset_name, success, output_json_path or error_message)
    """
    output_file = output_dir / f"{dataset}_metrics.json"

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "rerank_with_head_weights.py"),
        "--llm", llm,
        "--weight_file", str(weight_file),
        "--feature_file", str(feature_file),
        "--top_k_heads"] + [str(k) for k in top_k_heads] + [
        "--ks"] + [str(k) for k in ks] + [
        "--metrics"] + metrics + [
        "--evaluator", evaluator,
        "--output", str(output_file)
    ]

    if no_baseline:
        cmd.append("--no_baseline")

    if no_oracle:
        cmd.append("--no_oracle")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return (dataset, True, str(output_file))
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed with exit code {e.returncode}\nStderr: {e.stderr}"
        return (dataset, False, error_msg)


def parse_results(output_file: Path) -> Dict[str, Dict[str, float]]:
    """
    Parse reranking output JSON and extract metrics.

    Returns:
        dict mapping config (e.g., 'top-8') to metrics dict
    """
    with open(output_file) as f:
        data = json.load(f)

    # Extract metrics from file_results
    results = {}
    for file_result in data.get('file_results', []):
        for result in file_result.get('results', []):
            config = result['config']
            weights = result['weights']
            metrics = result['metrics']

            # Create key as config_weights (e.g., "top-8_bce")
            key = f"{config}_{weights}"
            results[key] = metrics

    return results


def aggregate_cqadupstack(dataset_results: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate cqadupstack-* datasets by averaging.

    Args:
        dataset_results: dict mapping dataset -> config -> metrics

    Returns:
        dict mapping config -> metrics for aggregated cqadupstack
        Empty dict if no cqadupstack datasets found
    """
    cqadupstack_datasets = [f'cqadupstack-{d}' for d in CQADUPSTACK_DOMAINS]

    # Check if any cqadupstack datasets are present
    found_cqa = [d for d in cqadupstack_datasets if d in dataset_results]
    if not found_cqa:
        return {}

    # Collect all configs
    all_configs = set()
    for dataset in cqadupstack_datasets:
        if dataset in dataset_results:
            all_configs.update(dataset_results[dataset].keys())

    # Average each config
    aggregated = {}
    for config in all_configs:
        # Collect metrics from all cqadupstack datasets for this config
        metric_values = defaultdict(list)

        for dataset in cqadupstack_datasets:
            if dataset in dataset_results and config in dataset_results[dataset]:
                metrics = dataset_results[dataset][config]
                for metric_name, value in metrics.items():
                    metric_values[metric_name].append(value)

        # Average (only if we have values)
        if metric_values:
            aggregated[config] = {
                metric_name: sum(values) / len(values)
                for metric_name, values in metric_values.items()
            }

    return aggregated


def compute_beir_average(dataset_results: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """
    Compute BEIR average across datasets (14 main + 1 cqadupstack if available).

    Args:
        dataset_results: dict mapping dataset -> config -> metrics

    Returns:
        dict mapping config -> metrics for BEIR average
    """
    # Aggregate cqadupstack
    cqadupstack_agg = aggregate_cqadupstack(dataset_results)

    # Collect datasets for averaging
    datasets_to_average = BEIR_MAIN_DATASETS.copy()
    final_results = {dataset: dataset_results[dataset] for dataset in BEIR_MAIN_DATASETS if dataset in dataset_results}

    # Add cqadupstack if available
    if cqadupstack_agg:
        datasets_to_average.append('cqadupstack')
        final_results['cqadupstack'] = cqadupstack_agg

    # Collect all configs
    all_configs = set()
    for dataset_metrics in final_results.values():
        all_configs.update(dataset_metrics.keys())

    # Average each config
    beir_avg = {}
    for config in all_configs:
        metric_values = defaultdict(list)

        for dataset in datasets_to_average:
            if dataset in final_results and config in final_results[dataset]:
                metrics = final_results[dataset][config]
                for metric_name, value in metrics.items():
                    metric_values[metric_name].append(value)

        # Average (only if we have values)
        if metric_values:
            beir_avg[config] = {
                metric_name: sum(values) / len(values)
                for metric_name, values in metric_values.items()
            }

    return beir_avg


def main():
    log_command()

    parser = argparse.ArgumentParser(
        description='Evaluate BEIR aggregate score across all datasets with parallel execution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script evaluates reranking performance on BEIR benchmark datasets:
- Runs rerank_with_head_weights.py on all BEIR datasets in parallel
- Aggregates 12 cqadupstack domains into a single score
- Computes BEIR average across 15 datasets (14 main + cqadupstack)
- Computes baseline (original retriever) and oracle (upper bound) performance
- Color-highlights best (green) and second-best (yellow) scores per metric
  (oracle results excluded from highlighting to avoid skewing comparisons)
- Sorts results: baseline, oracle, then top-k numerically (1, 2, 8, 16...)

Requirements:
- All 14 main BEIR datasets must have feature files (script exits if missing)
- Cqadupstack datasets are optional (warns if missing, uses available)

Output:
- Individual results: {output_dir}/{dataset}_metrics.json
- Aggregate results: {output_dir}/beir_aggregate.json

Example:
  python evaluate_beir_aggregate.py \\
      --llm mistral \\
      --weight_file head_data/mistral/bce_weights_lambda0.0001_n5000.json \\
      --feature_dir head_data/mistral \\
      --k 10 \\
      --top_k_heads 1 2 4 8 16 32 \\
      --n_jobs 8 \\
      --output_dir results/beir_k10
        """
    )

    parser.add_argument('--llm', type=str, required=True,
                        choices=['mistral', 'llama', 'phi', 'granite'],
                        help='LLM model name')
    parser.add_argument('--weight_file', type=str, required=True,
                        help='Path to head weights file (BCE or CoRe JSON)')
    parser.add_argument('--feature_dir', type=str, required=True,
                        help='Directory containing feature .npz files (e.g., head_data/mistral)')
    parser.add_argument('--k', type=int, required=True,
                        help='K value for feature files - matches *_k{K}.npz pattern (e.g., 10 for *_k10.npz)')
    parser.add_argument('--top_k_heads', type=int, nargs='+', required=True,
                        help='List of top-k head values to evaluate - each produces separate aggregate (e.g., 1 2 4 8 16 32)')
    parser.add_argument('--ks', type=int, nargs='+', default=[1, 5, 10],
                        help='K values for @k metrics - controls NDCG@K, P@K, etc. (default: 1 5 10)')
    parser.add_argument('--metrics', type=str, nargs='+',
                        default=['ndcg', 'p', 'm', 'map', 'mrr'],
                        help='Metrics to compute: ndcg, p (precision), m (match), map, mrr (default: all)')
    parser.add_argument('--evaluator', type=str, default='custom',
                        choices=['custom', 'beir'],
                        help='Evaluator to use: custom (built-in) or beir (requires beir package) (default: custom)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results - saves {dataset}_metrics.json and beir_aggregate.json')
    parser.add_argument('--no_baseline', action='store_true',
                        help='Skip baseline retriever evaluation (only evaluate reranked results)')
    parser.add_argument('--no_oracle', action='store_true',
                        help='Skip oracle (upper bound) evaluation')
    parser.add_argument('--n_jobs', type=int, default=4,
                        help='Number of parallel worker processes for dataset evaluation (default: 4)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print datasets and configuration without executing evaluations')

    args = parser.parse_args()

    # Resolve paths
    feature_dir = Path(args.feature_dir)
    weight_file = Path(args.weight_file)
    output_dir = Path(args.output_dir)

    if not feature_dir.exists():
        print(f"Error: Feature directory not found: {feature_dir}")
        sys.exit(1)

    if not weight_file.exists():
        print(f"Error: Weight file not found: {weight_file}")
        sys.exit(1)

    # Find feature files
    print(f"Finding feature files in {feature_dir} for k={args.k}...")
    dataset_files = find_feature_files(feature_dir, args.k)
    print(f"Found {len(dataset_files)} datasets")

    # Validate datasets
    is_valid, issues = validate_datasets(dataset_files)
    if not is_valid:
        print("\n⚠️  Dataset validation warnings:")
        for issue in issues:
            print(f"  - {issue}")
        print()

    # Check for required datasets
    missing_main = set(BEIR_MAIN_DATASETS) - set(dataset_files.keys())
    missing_cqa = [f'cqadupstack-{d}' for d in CQADUPSTACK_DOMAINS if f'cqadupstack-{d}' not in dataset_files]
    found_cqa = [f'cqadupstack-{d}' for d in CQADUPSTACK_DOMAINS if f'cqadupstack-{d}' in dataset_files]

    if missing_main:
        print(f"Error: Missing required main BEIR datasets: {sorted(missing_main)}")
        sys.exit(1)

    if missing_cqa:
        print(f"⚠️  Warning: Missing {len(missing_cqa)}/12 cqadupstack datasets: {sorted(missing_cqa)}")
        print(f"    Will aggregate using {len(found_cqa)} available cqadupstack datasets")

    print(f"✓ All {len(BEIR_MAIN_DATASETS)} main BEIR datasets found")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("\n[DRY RUN] Would evaluate the following datasets:")
        for dataset in sorted(dataset_files.keys()):
            print(f"  - {dataset}: {dataset_files[dataset].name}")
        print(f"\nOutput directory: {output_dir}")
        print(f"Parallel jobs: {args.n_jobs}")
        return

    # Run evaluations in parallel
    print(f"\nRunning evaluations on {len(dataset_files)} datasets with {args.n_jobs} parallel jobs...")

    results = {}
    failed = []

    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        futures = {
            executor.submit(
                run_reranking,
                dataset,
                feature_file,
                args.llm,
                weight_file,
                args.top_k_heads,
                args.ks,
                args.evaluator,
                args.metrics,
                output_dir,
                args.no_baseline,
                args.no_oracle
            ): dataset
            for dataset, feature_file in dataset_files.items()
        }

        for future in as_completed(futures):
            dataset = futures[future]
            try:
                dataset_name, success, result = future.result()
                if success:
                    print(f"✓ {dataset_name}")
                    results[dataset_name] = result
                else:
                    print(f"✗ {dataset_name}: {result}")
                    failed.append(dataset_name)
            except Exception as e:
                print(f"✗ {dataset}: {e}")
                failed.append(dataset)

    if failed:
        print(f"\n⚠️  {len(failed)} datasets failed:")
        for dataset in failed:
            print(f"  - {dataset}")

    # Parse all results
    print("\nParsing results...")
    dataset_results = {}
    for dataset, output_file in results.items():
        try:
            dataset_results[dataset] = parse_results(Path(output_file))
        except Exception as e:
            print(f"Error parsing {dataset}: {e}")

    # Compute BEIR average
    print("\nComputing BEIR aggregate scores...")
    beir_avg = compute_beir_average(dataset_results)

    # Save aggregated results
    output_file = output_dir / "beir_aggregate.json"

    # Also save per-dataset results for reference
    cqadupstack_agg = aggregate_cqadupstack(dataset_results)

    final_output = {
        'llm': args.llm,
        'weight_file': str(weight_file),
        'k': args.k,
        'top_k_heads': args.top_k_heads,
        'ks': args.ks,
        'metrics': args.metrics,
        'evaluator': args.evaluator,
        'beir_average': beir_avg,
        'cqadupstack_average': cqadupstack_agg,
        'per_dataset': {
            dataset: metrics
            for dataset, metrics in dataset_results.items()
            if dataset in BEIR_MAIN_DATASETS
        }
    }

    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2)

    print(f"\n✓ Saved aggregate results to {output_file}")

    # Print summary
    print("\n" + "="*70)
    print("BEIR Aggregate Results")
    print("="*70)

    # Get metric names from first config
    if beir_avg:
        first_config = list(beir_avg.keys())[0]
        metric_names = list(beir_avg[first_config].keys())

        # Find max and second max for each metric (exclude oracle from coloring)
        metric_max = {}
        metric_second_max = {}

        # Filter out oracle configs for max calculation
        non_oracle_configs = {k: v for k, v in beir_avg.items() if not k.split('_', 1)[0] == 'oracle'}

        if len(non_oracle_configs) > 1:
            for metric in metric_names:
                values = sorted([config_metrics.get(metric, 0.0) for config_metrics in non_oracle_configs.values()], reverse=True)
                metric_max[metric] = values[0]
                if len(values) >= 2:
                    metric_second_max[metric] = values[1]

        # ANSI color codes
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BOLD = '\033[1m'
        RESET = '\033[0m'

        # Print header
        header = f"{'Config':<20} {'Weights':<10}"
        for metric in metric_names:
            header += f" {metric:<8}"
        print(header)
        print("-" * len(header))

        # Sort configs numerically by top-k value
        def sort_key(config_key):
            """Sort baseline first, oracle second, then top-k numerically."""
            parts = config_key.split('_', 1)
            config = parts[0]

            if config == 'baseline':
                return (0, 0)  # baseline comes first
            elif config == 'oracle':
                return (0, 1)  # oracle comes second (after baseline)
            elif config.startswith('top-'):
                try:
                    k = int(config.split('-')[1])
                    return (1, k)  # sort top-k numerically
                except (ValueError, IndexError):
                    return (2, config)  # fallback to string sort
            else:
                return (2, config)  # other configs

        # Print results
        for config_key in sorted(beir_avg.keys(), key=sort_key):
            parts = config_key.split('_', 1)
            config = parts[0]
            weights = parts[1] if len(parts) > 1 else ''

            line = f"{config:<20} {weights:<10}"
            for metric in metric_names:
                value = beir_avg[config_key].get(metric, 0.0)
                formatted = f"{value:<8.4f}"

                # Skip coloring for oracle rows
                if config != 'oracle':
                    # Highlight max in green+bold
                    if metric in metric_max and value == metric_max[metric]:
                        formatted = f"{GREEN}{BOLD}{value:<8.4f}{RESET}"
                    # Highlight second max in yellow+bold
                    elif metric in metric_second_max and value == metric_second_max[metric]:
                        formatted = f"{YELLOW}{BOLD}{value:<8.4f}{RESET}"

                line += f" {formatted}"
            print(line)

    print("="*70)

    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
