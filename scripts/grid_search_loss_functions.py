#!/usr/bin/env python3
"""
Grid search over loss functions, lambda values, and temperatures for head weight training.

Trains models with all combinations and evaluates on BEIR to find the best configuration
per loss function.

Usage:
    python scripts/grid_search_loss_functions.py \
        --llm mistral \
        --feature_file head_data/mistral/attention_features_nq_k40.npz \
        --beir_feature_dir head_data/mistral \
        --beir_k 40 \
        --top_k_heads 8 \
        --output_dir results/grid_search
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import log_command, load_features, get_head_info

# Loss function configurations with reasonable hyperparameter ranges
LOSS_CONFIGS = {
    'bce': {
        'lambda_l1': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        'temp': [0.001, 0.01, 0.1, 1.0],
        'description': 'Binary Cross-Entropy with L1 (sklearn LogisticRegression)'
    },
    'bce_temp': {
        'lambda_l1': [1e-5, 1e-4, 1e-3, 1e-2, 0.1],
        'temp': [0.001, 0.01, 0.1, 1.0],
        'description': 'BCE with temperature-scaled softmax features (gradient descent)'
    },
    'infonce': {
        'lambda_l1': [1e-5, 1e-4, 1e-3, 1e-2, 0.1],
        'temp': [0.001, 0.01, 0.1, 1.0],
        'description': 'InfoNCE contrastive loss for listwise ranking'
    },
    'hinge': {
        'lambda_l1': [1e-5, 1e-4, 1e-3, 1e-2, 0.1],
        'temp': [0.001, 0.01, 0.1, 1.0],
        'description': 'Pairwise hinge loss with Elastic Net'
    },
    'approx_ndcg': {
        'lambda_l1': [1e-5, 1e-4, 1e-3, 1e-2],
        'temp': [0.001, 0.01, 0.1, 1.0],
        'description': 'Differentiable approximation of NDCG (slow, finite differences)'
    },
    'approx_ndcg_fast': {
        'lambda_l1': [1e-5, 1e-4, 1e-3, 1e-2],
        'temp': [0.001, 0.01, 0.1, 1.0],
        'description': 'Differentiable approximation of NDCG (fast, analytical gradients)'
    },
    'group_lasso': {
        'lambda_l1': [1e-4, 1e-3, 1e-2, 0.1, 1.0],
        'temp': [0.001, 0.01, 0.1, 1.0],
        'description': 'Group Lasso for layer-wise sparse selection'
    },
    'ranknet': {
        'lambda_l1': [1e-5, 1e-4, 1e-3, 1e-2, 0.1],
        'temp': [0.001, 0.01, 0.1, 1.0],
        'description': 'RankNet pairwise ranking with smooth loss'
    },
}


def train_single_model(llm, feature_file, loss, lambda_l1, temp, output_file, cv=5, max_iter=1000, input_file=None, verbose=False):
    """
    Train a single lambda configuration.

    Args:
        lambda_l1: Single lambda value to train
        output_file: Output file path for this lambda

    Returns:
        tuple: (lambda_l1, success, message)
    """
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "train_head_weights_bce.py"),
        "--llm", llm,
        "--feature_file", str(feature_file),
        "--loss", loss,
        "--lambda_l1", str(lambda_l1),
        "--temp", str(temp),
        "--cv", str(cv),
        "--n_jobs", "1",  # Single job since we're parallelizing at the grid search level
        "--max_iter", str(max_iter),
        "--output", str(output_file)
    ]

    if input_file:
        cmd.extend(["--input_file", str(input_file)])

    if verbose:
        print(f"\n  CMD: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=600
        )
        return lambda_l1, True, "OK"
    except subprocess.CalledProcessError as e:
        return lambda_l1, False, f"Training failed: {e.stderr[-500:]}"
    except subprocess.TimeoutExpired:
        return lambda_l1, False, "Training timeout"


def train_models_parallel(llm, feature_file, loss, lambda_l1_list, temp, weights_dir,
                          cv=5, n_jobs=4, max_iter=1000, input_file=None, verbose=False):
    """
    Train multiple lambda configurations in parallel using separate processes.

    Args:
        lambda_l1_list: List of lambda values to train
        weights_dir: Directory to save weight files
        n_jobs: Number of parallel processes

    Returns:
        dict: {lambda_l1: (success, message)} for each lambda
    """
    results = {}

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {}
        for lambda_l1 in lambda_l1_list:
            output_file = weights_dir / f"{loss}_lambda{lambda_l1}_temp{temp}.json"
            future = executor.submit(
                train_single_model,
                llm, feature_file, loss, lambda_l1, temp, output_file,
                cv, max_iter, input_file, verbose
            )
            futures[future] = lambda_l1

        for future in as_completed(futures):
            lambda_l1, success, message = future.result()
            results[lambda_l1] = (success, message)

    return results


def train_models_batched(llm, feature_file, loss, lambda_l1_list, temp, output_template,
                         cv=5, n_jobs=-1, max_iter=1000, input_file=None, verbose=False):
    """
    Train multiple lambda configurations in a single batched command.
    The training script handles parallelism internally via joblib.

    Args:
        lambda_l1_list: List of lambda values to train (will be run in parallel)
        output_template: Output template with {lambda} placeholder

    Returns:
        success: bool
        message: Success message or error
    """
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "train_head_weights_bce.py"),
        "--llm", llm,
        "--feature_file", str(feature_file),
        "--loss", loss,
        "--lambda_l1"] + [str(l) for l in lambda_l1_list] + [
        "--temp", str(temp),
        "--cv", str(cv),
        "--n_jobs", str(n_jobs),
        "--max_iter", str(max_iter),
        "--output", str(output_template)
    ]

    if input_file:
        cmd.extend(["--input_file", str(input_file)])

    if verbose:
        print(f"\n  CMD: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=600 * len(lambda_l1_list)  # Scale timeout with number of lambdas
        )
        return True, "OK"
    except subprocess.CalledProcessError as e:
        return False, f"Training failed: {e.stderr[-500:]}"
    except subprocess.TimeoutExpired:
        return False, "Training timeout"


def evaluate_beir_single(llm, weight_file, feature_dir, k, top_k_heads, output_dir,
                         beir_dir=None, verbose=False, config_id=None, lambda_l1=None):
    """
    Evaluate a trained model on BEIR.

    Args:
        config_id: Optional identifier for this configuration (for logging)
        lambda_l1: Optional lambda value (returned in result for identification)

    Returns:
        tuple: (lambda_l1, config_id, success, beir_result)
    """
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "evaluate_beir_aggregate.py"),
        "--llm", llm,
        "--weight_file", str(weight_file),
        "--feature_dir", str(feature_dir),
        "--k", str(k),
        "--top_k_heads", str(top_k_heads),
        "--output_dir", str(output_dir),
        "--n_jobs", "1",  # Single job since we parallelize at grid search level
        "--no_corpus_breakdown",
        "--metrics", "ndcg"  # Only compute NDCG for speed
    ]

    if beir_dir:
        cmd.extend(["--beir_dir", str(beir_dir)])

    if verbose:
        print(f"\n  CMD: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=1800  # 30 minute timeout for BEIR eval
        )

        # Parse the aggregate results
        aggregate_file = Path(output_dir) / f"k{k}" / "beir_aggregate.json"
        if aggregate_file.exists():
            with open(aggregate_file) as f:
                data = json.load(f)
            return lambda_l1, config_id, True, data.get('beir_average', {})
        else:
            return lambda_l1, config_id, False, "No aggregate file found"

    except subprocess.CalledProcessError as e:
        return lambda_l1, config_id, False, f"BEIR eval failed: {e.stderr[-500:]}"
    except subprocess.TimeoutExpired:
        return lambda_l1, config_id, False, "BEIR eval timeout (>30 min)"


def evaluate_beir_parallel(eval_configs, n_jobs=4):
    """
    Evaluate multiple configurations on BEIR in parallel.

    Args:
        eval_configs: List of dicts with keys: llm, weight_file, feature_dir, k,
                      top_k_heads, output_dir, beir_dir, verbose, config_id, lambda_l1
        n_jobs: Number of parallel processes

    Returns:
        dict: {config_id: (lambda_l1, success, beir_result)}
    """
    results = {}

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {}
        for cfg in eval_configs:
            future = executor.submit(
                evaluate_beir_single,
                cfg['llm'], cfg['weight_file'], cfg['feature_dir'], cfg['k'],
                cfg['top_k_heads'], cfg['output_dir'], cfg.get('beir_dir'),
                cfg.get('verbose', False), cfg['config_id'], cfg['lambda_l1']
            )
            futures[future] = cfg['config_id']

        for future in as_completed(futures):
            lambda_l1, config_id, success, beir_result = future.result()
            results[config_id] = (lambda_l1, success, beir_result)

    return results


def evaluate_beir(llm, weight_file, feature_dir, k, top_k_heads, output_dir, beir_dir=None, n_jobs=4, verbose=False):
    """
    Evaluate a trained model on BEIR (sequential mode).

    Returns:
        success: bool
        beir_avg: dict of metric -> value, or error message
    """
    _, _, success, result = evaluate_beir_single(
        llm, weight_file, feature_dir, k, top_k_heads, output_dir,
        beir_dir, verbose, config_id=None, lambda_l1=None
    )
    return success, result


def main():
    log_command()

    parser = argparse.ArgumentParser(
        description='Grid search over loss functions for head weight training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/grid_search_loss_functions.py \\
        --llm mistral \\
        --feature_file head_data/mistral/attention_features_nq_k40.npz \\
        --beir_feature_dir head_data/mistral \\
        --beir_k 40 \\
        --top_k_heads 8 \\
        --output_dir results/grid_search
        """
    )

    parser.add_argument('--llm', type=str, required=True,
                        choices=['mistral', 'llama', 'phi', 'granite'],
                        help='LLM model name')
    parser.add_argument('--feature_file', type=str, required=True,
                        help='Feature file for training (e.g., NQ features)')
    parser.add_argument('--input_file', type=str, default=None,
                        help='JSON file with query groupings for proper train/val split (e.g., nq_core.json)')
    parser.add_argument('--beir_feature_dir', type=str, required=True,
                        help='Directory containing BEIR feature files')
    parser.add_argument('--beir_k', type=int, required=True,
                        help='K value for BEIR feature files')
    parser.add_argument('--beir_dir', type=str, default=None,
                        help='Path to BEIR qrels directory (optional)')
    parser.add_argument('--top_k_heads', type=int, default=8,
                        help='Number of top heads to use for BEIR evaluation (default: 8)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--loss_functions', type=str, nargs='+',
                        default=list(LOSS_CONFIGS.keys()),
                        choices=list(LOSS_CONFIGS.keys()),
                        help='Loss functions to evaluate (default: all)')
    parser.add_argument('--cv', type=int, default=5,
                        help='Number of CV folds for training (default: 5)')
    parser.add_argument('--n_jobs', type=int, default=4,
                        help='Parallelization strategy: -1 = batched mode (single command, joblib parallelism), '
                             '>0 = run N separate training processes in parallel (default: 4)')
    parser.add_argument('--max_iter', type=int, default=1000,
                        help='Max iterations for training (default: 1000)')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training, only evaluate existing weights')
    parser.add_argument('--skip_beir', action='store_true',
                        help='Skip BEIR evaluation, only train models')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print commands being executed')

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = output_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    beir_results_dir = output_dir / "beir_results"
    beir_results_dir.mkdir(exist_ok=True)

    # Results storage
    all_results = []
    best_per_loss = {}

    # ANSI colors
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    print(f"\n{'='*80}")
    print(f"Grid Search: Loss Functions x Lambda x Temperature")
    print(f"{'='*80}")
    print(f"LLM: {args.llm}")
    print(f"Training features: {args.feature_file}")
    if args.input_file:
        print(f"Query groupings: {args.input_file}")
    print(f"BEIR features: {args.beir_feature_dir} (k={args.beir_k})")
    print(f"Top-k heads for eval: {args.top_k_heads}")
    print(f"Loss functions: {args.loss_functions}")
    print(f"CV folds: {args.cv}")
    if args.n_jobs == -1:
        print(f"Parallel strategy: batched (joblib, all CPUs)")
    else:
        print(f"Parallel strategy: {args.n_jobs} separate processes per (loss, temp)")
    print(f"{'='*80}\n")

    # Count total configurations
    total_configs = sum(
        len(LOSS_CONFIGS[loss]['lambda_l1']) * len(LOSS_CONFIGS[loss]['temp'])
        for loss in args.loss_functions
    )
    print(f"Total configurations to evaluate: {total_configs}\n")

    config_num = 0

    for loss in args.loss_functions:
        config = LOSS_CONFIGS[loss]
        print(f"\n{'-'*60}")
        print(f"Loss: {loss} - {config['description']}")
        print(f"{'-'*60}")

        loss_results = []
        lambda_list = config['lambda_l1']

        for temp in config['temp']:
            # Train all lambda values in parallel for this (loss, temp) combination
            temp_configs = len(lambda_list)
            config_num += temp_configs

            print(f"\n[{config_num - temp_configs + 1}-{config_num}/{total_configs}] {loss} temp={temp} (lambdas: {lambda_list})")

            # Train models
            training_failed_lambdas = set()
            if not args.skip_training:
                if args.n_jobs == -1:
                    # Batched mode: single command with multiple lambdas, internal parallelism via joblib
                    print(f"  Training {len(lambda_list)} lambda configs (batched, joblib)...", end=" ", flush=True)
                    output_template = str(weights_dir / f"{loss}_lambda{{lambda}}_temp{temp}.json")
                    success, result = train_models_batched(
                        llm=args.llm,
                        feature_file=args.feature_file,
                        loss=loss,
                        lambda_l1_list=lambda_list,
                        temp=temp,
                        output_template=output_template,
                        cv=args.cv,
                        n_jobs=-1,
                        max_iter=args.max_iter,
                        input_file=args.input_file,
                        verbose=args.verbose
                    )
                    if not success:
                        print(f"{RED}FAILED{RESET}: {result}")
                        training_failed_lambdas = set(lambda_list)
                    else:
                        print(f"{GREEN}OK{RESET}")
                else:
                    # Parallel process mode: separate processes for each lambda
                    print(f"  Training {len(lambda_list)} lambda configs ({args.n_jobs} parallel processes)...", end=" ", flush=True)
                    results = train_models_parallel(
                        llm=args.llm,
                        feature_file=args.feature_file,
                        loss=loss,
                        lambda_l1_list=lambda_list,
                        temp=temp,
                        weights_dir=weights_dir,
                        cv=args.cv,
                        n_jobs=args.n_jobs,
                        max_iter=args.max_iter,
                        input_file=args.input_file,
                        verbose=args.verbose
                    )
                    # Check results
                    succeeded = sum(1 for s, _ in results.values() if s)
                    failed = len(results) - succeeded
                    if failed > 0:
                        print(f"{YELLOW}{succeeded}/{len(results)} OK, {failed} failed{RESET}")
                        for lam, (s, msg) in results.items():
                            if not s:
                                print(f"    lambda={lam}: {RED}{msg}{RESET}")
                                training_failed_lambdas.add(lam)
                    else:
                        print(f"{GREEN}OK{RESET}")

            # Evaluate each lambda on BEIR
            if args.skip_beir:
                continue

            # Build list of configs to evaluate
            eval_configs = []
            for lambda_l1 in lambda_list:
                # Skip lambdas that failed training
                if lambda_l1 in training_failed_lambdas:
                    continue

                config_id = f"{loss}_lambda{lambda_l1}_temp{temp}"
                weight_file = weights_dir / f"{config_id}.json"

                # Check weight file exists
                if not weight_file.exists():
                    print(f"  {YELLOW}{config_id}: Weight file not found, skipping BEIR eval{RESET}")
                    continue

                eval_configs.append({
                    'llm': args.llm,
                    'weight_file': weight_file,
                    'feature_dir': args.beir_feature_dir,
                    'k': args.beir_k,
                    'top_k_heads': args.top_k_heads,
                    'output_dir': beir_results_dir / config_id,
                    'beir_dir': args.beir_dir,
                    'verbose': args.verbose,
                    'config_id': config_id,
                    'lambda_l1': lambda_l1,
                    'temp': temp,
                    'loss': loss
                })

            if not eval_configs:
                continue

            if args.n_jobs == -1:
                # Sequential mode (each eval may use all CPUs internally)
                for cfg in eval_configs:
                    print(f"  {cfg['config_id']}: Evaluating on BEIR...", end=" ", flush=True)
                    success, beir_result = evaluate_beir(
                        llm=cfg['llm'],
                        weight_file=cfg['weight_file'],
                        feature_dir=cfg['feature_dir'],
                        k=cfg['k'],
                        top_k_heads=cfg['top_k_heads'],
                        output_dir=cfg['output_dir'],
                        beir_dir=cfg['beir_dir'],
                        n_jobs=-1,
                        verbose=cfg['verbose']
                    )

                    if not success:
                        print(f"{RED}FAILED{RESET}: {beir_result}")
                        continue

                    # Extract NDCG@10 for the top-k config
                    top_k_key = f"top-{args.top_k_heads}_"
                    ndcg10 = None
                    for key, metrics in beir_result.items():
                        if key.startswith(top_k_key):
                            ndcg10 = metrics.get('NDCG@10', metrics.get('ndcg@10'))
                            break

                    if ndcg10 is None:
                        print(f"{YELLOW}No NDCG@10 found{RESET}")
                        continue

                    print(f"{GREEN}NDCG@10 = {ndcg10:.4f}{RESET}")

                    result_entry = {
                        'loss': cfg['loss'],
                        'lambda_l1': cfg['lambda_l1'],
                        'temp': cfg['temp'],
                        'ndcg@10': ndcg10,
                        'weight_file': str(cfg['weight_file']),
                        'all_metrics': beir_result
                    }
                    all_results.append(result_entry)
                    loss_results.append(result_entry)
            else:
                # Parallel mode
                print(f"  Evaluating {len(eval_configs)} configs on BEIR ({args.n_jobs} parallel)...", end=" ", flush=True)
                beir_results = evaluate_beir_parallel(eval_configs, n_jobs=args.n_jobs)

                succeeded = sum(1 for _, s, _ in beir_results.values() if s)
                failed = len(beir_results) - succeeded
                if failed > 0:
                    print(f"{YELLOW}{succeeded}/{len(beir_results)} OK, {failed} failed{RESET}")
                else:
                    print(f"{GREEN}OK{RESET}")

                # Process results
                for cfg in eval_configs:
                    config_id = cfg['config_id']
                    if config_id not in beir_results:
                        continue

                    lambda_l1, success, beir_result = beir_results[config_id]

                    if not success:
                        print(f"    {config_id}: {RED}{beir_result}{RESET}")
                        continue

                    # Extract NDCG@10 for the top-k config
                    top_k_key = f"top-{args.top_k_heads}_"
                    ndcg10 = None
                    for key, metrics in beir_result.items():
                        if key.startswith(top_k_key):
                            ndcg10 = metrics.get('NDCG@10', metrics.get('ndcg@10'))
                            break

                    if ndcg10 is None:
                        print(f"    {config_id}: {YELLOW}No NDCG@10 found{RESET}")
                        continue

                    print(f"    {config_id}: NDCG@10 = {GREEN}{ndcg10:.4f}{RESET}")

                    result_entry = {
                        'loss': cfg['loss'],
                        'lambda_l1': lambda_l1,
                        'temp': cfg['temp'],
                        'ndcg@10': ndcg10,
                        'weight_file': str(cfg['weight_file']),
                        'all_metrics': beir_result
                    }
                    all_results.append(result_entry)
                    loss_results.append(result_entry)

        # Find best for this loss function
        if loss_results:
            best = max(loss_results, key=lambda x: x['ndcg@10'])
            best_per_loss[loss] = best
            print(f"\n  {BOLD}Best for {loss}:{RESET} lambda={best['lambda_l1']}, temp={best['temp']}, NDCG@10={GREEN}{best['ndcg@10']:.4f}{RESET}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: Best BEIR NDCG@10 per Loss Function")
    print(f"{'='*80}")

    # Sort by NDCG@10
    sorted_losses = sorted(best_per_loss.items(), key=lambda x: x[1]['ndcg@10'], reverse=True)

    print(f"\n{'Loss':<15} {'Lambda':<12} {'Temp':<10} {'NDCG@10':<10}")
    print(f"{'-'*47}")

    for i, (loss, best) in enumerate(sorted_losses):
        ndcg = best['ndcg@10']
        if i == 0:
            # Best overall
            print(f"{GREEN}{BOLD}{loss:<15} {best['lambda_l1']:<12} {best['temp']:<10} {ndcg:<10.4f}{RESET}")
        elif i == 1:
            # Second best
            print(f"{YELLOW}{loss:<15} {best['lambda_l1']:<12} {best['temp']:<10} {ndcg:<10.4f}{RESET}")
        else:
            print(f"{loss:<15} {best['lambda_l1']:<12} {best['temp']:<10} {ndcg:<10.4f}")

    # Save full results
    results_file = output_dir / "grid_search_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'llm': args.llm,
                'feature_file': args.feature_file,
                'input_file': args.input_file,
                'beir_feature_dir': args.beir_feature_dir,
                'beir_k': args.beir_k,
                'top_k_heads': args.top_k_heads,
                'cv': args.cv
            },
            'best_per_loss': {k: {**v, 'all_metrics': None} for k, v in best_per_loss.items()},  # Exclude large metrics dict
            'all_results': all_results
        }, f, indent=2)

    print(f"\nFull results saved to: {results_file}")
    print(f"{'='*80}\n")

    # Return best overall
    if sorted_losses:
        best_loss, best_config = sorted_losses[0]
        print(f"{BOLD}Best overall:{RESET} {best_loss} with lambda={best_config['lambda_l1']}, temp={best_config['temp']}")
        print(f"NDCG@10 = {GREEN}{BOLD}{best_config['ndcg@10']:.4f}{RESET}")


if __name__ == '__main__':
    main()
