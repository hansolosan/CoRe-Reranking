#!/usr/bin/env python3
"""
Train head weights using Binary Cross-Entropy loss with L1 regularization.
This implements the logistic regression optimization proposed in the report.

The loss is normalized by the number of samples:
    Loss = (1/n) * sum(BCE_loss) + lambda * ||w||_1

This makes lambda comparable across different dataset sizes.

Data splitting is done at the BASE QUERY level to prevent data leakage:
- The NQ head detection data has 5 position variations per base query
- All variations of the same query are kept together in train or val
- Query groupings can be loaded from the original JSON file (--input_file)
- If no JSON file is provided, assumes every 5 consecutive samples are variations
"""

import json
import argparse
import sys
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from utils import log_command


def get_unique_filepath(base_path):
    """Return a unique filepath by adding numeric suffix if file exists."""
    path = Path(base_path)
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    counter = 1
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def load_features(feature_file=None, llm_name=None, num_samples=None):
    """
    Load extracted attention features.

    Args:
        feature_file: Path to feature file (.npz). If provided, llm_name and num_samples are ignored.
        llm_name: LLM name for default path construction
        num_samples: Number of samples for default path construction

    Returns:
        features, labels, docs_per_query arrays
    """
    if feature_file is not None:
        path = Path(feature_file)
    else:
        if llm_name is None:
            raise ValueError("Either feature_file or llm_name must be provided")
        path = Path(__file__).parent.parent / 'head_data' / llm_name / f'attention_features_n{num_samples}.npz'

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}\n"
            f"Run extract_head_features.py first to generate features."
        )

    data = np.load(path)
    features = data['features']
    labels = data['labels']

    # Get docs_per_query if available, otherwise assume 50
    if 'docs_per_query' in data:
        docs_per_query = data['docs_per_query']
    else:
        # Assume 50 docs per query (standard for NQ head detection data)
        n_queries = len(labels) // 50
        docs_per_query = np.full(n_queries, 50, dtype=np.int32)

    return features, labels, docs_per_query


def load_query_groups_from_json(json_file, num_samples=None):
    """
    Load query groupings from the original JSON file.

    Groups queries by their question text to identify which query samples
    are variations of the same base query.

    Args:
        json_file: Path to the JSON file (supports .gz and .bz2 compression)
        num_samples: Number of samples to consider (default: all)

    Returns:
        query_to_base: List mapping query index to base query index
        n_base_queries: Number of unique base queries
    """
    import gzip
    import bz2

    json_path = Path(json_file)

    # Open file with appropriate compression handling
    if json_path.suffix == '.gz':
        with gzip.open(json_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
    elif json_path.suffix == '.bz2':
        with bz2.open(json_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    if num_samples is not None:
        data = data[:num_samples]

    # Group queries by question text
    question_to_base_idx = {}
    query_to_base = []

    for q_idx, sample in enumerate(data):
        question = sample.get('question', sample.get('query', ''))

        if question not in question_to_base_idx:
            question_to_base_idx[question] = len(question_to_base_idx)

        query_to_base.append(question_to_base_idx[question])

    n_base_queries = len(question_to_base_idx)

    return query_to_base, n_base_queries


def split_by_base_query(X, y, docs_per_query, val_split=0.2, random_state=42,
                        query_to_base=None, positions_per_query=5):
    """
    Split data by base query, keeping all position variations together.

    The NQ head detection data has 5 position variations per base query
    (positive document shown at positions 0-4). We need to ensure all
    variations of a query end up in the same split.

    Args:
        X: Feature matrix (n_docs, n_features)
        y: Labels (n_docs,)
        docs_per_query: Number of docs per query sample
        val_split: Fraction for validation
        random_state: Random seed
        query_to_base: List mapping query index to base query index (from JSON file).
                      If None, assumes every `positions_per_query` consecutive queries
                      are variations of the same base query.
        positions_per_query: Number of position variations per base query (default: 5).
                            Only used if query_to_base is None.

    Returns:
        X_train, X_val, y_train, y_val, train_query_indices, val_query_indices, n_base_queries
    """
    n_queries = len(docs_per_query)

    # Determine query-to-base mapping
    if query_to_base is not None:
        # Use provided mapping from JSON file
        n_base_queries = max(query_to_base) + 1
        print(f"Using query groups from JSON: {n_base_queries} unique base queries")
    else:
        # Assume every `positions_per_query` consecutive queries are the same base query
        n_base_queries = n_queries // positions_per_query
        if n_queries % positions_per_query != 0:
            print(f"Warning: {n_queries} queries is not divisible by {positions_per_query}. "
                  f"Assuming {n_base_queries} base queries.")
        query_to_base = [q_idx // positions_per_query for q_idx in range(n_queries)]
        print(f"Assuming {positions_per_query} position variations per base query")

    # Create base query indices
    base_query_indices = np.arange(n_base_queries)

    # Shuffle and split base queries
    rng = np.random.RandomState(random_state)
    rng.shuffle(base_query_indices)

    n_val_base = int(n_base_queries * val_split)
    val_base_indices = set(base_query_indices[:n_val_base])

    # Map query samples to train/val based on their base query
    train_query_indices = []
    val_query_indices = []

    for q_idx in range(n_queries):
        base_idx = query_to_base[q_idx]
        if base_idx in val_base_indices:
            val_query_indices.append(q_idx)
        else:
            train_query_indices.append(q_idx)

    # Collect document indices for train/val
    train_doc_indices = []
    val_doc_indices = []

    doc_offset = 0
    for q_idx in range(n_queries):
        n_docs = docs_per_query[q_idx]
        doc_indices = list(range(doc_offset, doc_offset + n_docs))

        base_idx = query_to_base[q_idx]
        if base_idx in val_base_indices:
            val_doc_indices.extend(doc_indices)
        else:
            train_doc_indices.extend(doc_indices)

        doc_offset += n_docs

    # Create train/val splits
    train_doc_indices = np.array(train_doc_indices)
    val_doc_indices = np.array(val_doc_indices)

    X_train = X[train_doc_indices]
    X_val = X[val_doc_indices]
    y_train = y[train_doc_indices]
    y_val = y[val_doc_indices]

    return X_train, X_val, y_train, y_val, train_query_indices, val_query_indices, n_base_queries


def get_head_info(llm_name):
    """Get number of layers and heads for a model."""
    model_configs = {
        'mistral': (32, 32),   # 32 layers, 32 heads
        'llama': (32, 32),     # 32 layers, 32 heads
        'phi': (40, 40),       # 40 layers, 40 heads
        'granite': (40, 32),   # 40 layers, 32 heads
    }
    return model_configs.get(llm_name, (32, 32))


def train_logistic_regression(X_train, y_train, X_val, y_val, lambda_l1=0.01, max_iter=1000):
    """
    Train L1-regularized logistic regression.

    sklearn's objective is: ||w||_1 + C * sum(log_loss)
    We want normalized BCE: (1/n) * sum(log_loss) + lambda * ||w||_1

    These objectives are equivalent (same minimizer) when:
        lambda * ||w||_1 + (1/n) * sum = k * (||w||_1 + C * sum)

    Matching coefficients: k = lambda, and 1/n = k*C = lambda*C
    Therefore: C = 1 / (n * lambda)

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        lambda_l1: L1 regularization strength (normalized by num samples)
        max_iter: Maximum iterations for solver

    Returns:
        model: Trained model
        metrics: Dictionary of evaluation metrics
    """
    # Normalize by number of samples: C = 1 / (n * lambda)
    # This makes lambda independent of dataset size
    n_samples = len(y_train)
    C = 1.0 / (n_samples * lambda_l1) if lambda_l1 > 0 else 1e6

    model = LogisticRegression(
        penalty='l1',
        C=C,
        solver='saga',
        max_iter=max_iter,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance (1 pos vs 49 neg)
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'auc_roc': roc_auc_score(y_val, y_prob),
        'num_nonzero_weights': np.sum(np.abs(model.coef_[0]) > 1e-6),
        'sparsity': 1.0 - np.sum(np.abs(model.coef_[0]) > 1e-6) / len(model.coef_[0])
    }

    return model, metrics


def analyze_weights(model, num_layers, num_heads, top_k=20):
    """Analyze learned head weights."""
    weights = model.coef_[0]

    # Create head index mapping
    head_weights = []
    for layer in range(num_layers):
        for head in range(num_heads):
            idx = layer * num_heads + head
            if idx < len(weights):
                head_weights.append({
                    'layer': layer,
                    'head': head,
                    'weight': weights[idx],
                    'abs_weight': abs(weights[idx])
                })

    # Sort by absolute weight
    head_weights.sort(key=lambda x: x['abs_weight'], reverse=True)

    return head_weights[:top_k], head_weights


def compare_with_core_heads(learned_heads, llm_name, temp=0.001, prune=0.0):
    """Compare learned heads with original CoRe heads."""
    core_file = Path(__file__).parent.parent / 'head_data' / llm_name / f'core_temp{temp}_prune{prune}.json'

    if not core_file.exists():
        print(f"CoRe head file not found: {core_file}")
        return

    with open(core_file, 'r') as f:
        core_scores = json.load(f)

    # Get top 8 CoRe heads
    core_heads = []
    for head_key, scores in core_scores.items():
        layer, head = map(int, head_key.split('-'))
        avg_score = np.mean(scores) if isinstance(scores, list) else scores
        core_heads.append({'layer': layer, 'head': head, 'score': avg_score})

    core_heads.sort(key=lambda x: x['score'], reverse=True)
    core_top8 = set((h['layer'], h['head']) for h in core_heads[:8])

    # Get top 8 learned heads
    learned_top8 = set((h['layer'], h['head']) for h in learned_heads[:8])

    overlap = len(core_top8 & learned_top8)

    print(f"\n{'='*60}")
    print("Comparison with CoRe Heads")
    print('='*60)
    print(f"Top 8 CoRe heads: {sorted(core_top8)}")
    print(f"Top 8 Learned heads: {sorted(learned_top8)}")
    print(f"Overlap: {overlap}/8 heads")

    return core_top8, learned_top8


def convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    return obj


def save_results(output_dir, lambda_l1, num_samples, metrics, top_heads, all_heads, command,
                 n_train_docs=None, n_val_docs=None, n_train_base_queries=None, n_val_base_queries=None):
    """Save results for a single lambda value."""
    base_file = output_dir / f'bce_weights_lambda{lambda_l1}_n{num_samples}.json'
    output_file = get_unique_filepath(base_file)

    results = {
        'command': command,
        'timestamp': datetime.now().isoformat(),
        'lambda_l1': float(lambda_l1),
        'num_samples': int(num_samples),
        'metrics': convert_to_native(metrics),
        'split_info': {
            'split_type': 'base_query',
            'positions_per_query': 5,
            'n_train_docs': n_train_docs,
            'n_val_docs': n_val_docs,
            'n_train_base_queries': n_train_base_queries,
            'n_val_base_queries': n_val_base_queries,
        },
        'top_heads': [{'layer': int(h['layer']), 'head': int(h['head']), 'weight': float(h['weight'])}
                      for h in top_heads],
        'all_weights': {f"{h['layer']}-{h['head']}": float(h['weight']) for h in all_heads}
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    return output_file


def train_single_lambda(lambda_l1, X_train, y_train, X_val, y_val, num_layers, num_heads, max_iter=1000):
    """
    Train a single model for one lambda value. Designed for parallel execution.

    Returns:
        dict with lambda_l1, metrics, top_heads, all_heads
    """
    model, metrics = train_logistic_regression(
        X_train, y_train, X_val, y_val, lambda_l1, max_iter=max_iter
    )
    top_heads, all_heads = analyze_weights(model, num_layers, num_heads, top_k=20)

    return {
        'lambda_l1': lambda_l1,
        'model': model,
        'metrics': metrics,
        'top_heads': top_heads,
        'all_heads': all_heads
    }


def main():
    # Log command execution
    log_command()

    parser = argparse.ArgumentParser(description='Train head weights with BCE + L1')
    parser.add_argument('--llm', type=str, default='mistral',
                        choices=['mistral', 'llama', 'phi', 'granite'])
    parser.add_argument('--feature_file', '-f', type=str, default=None,
                        help='Path to feature file (.npz). If provided, --llm and --num_samples are used only for output paths.')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples used for feature extraction (ignored if --feature_file is provided)')
    parser.add_argument('--lambda_l1', type=float, nargs='+',
                        default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                        help='L1 regularization strengths to try (normalized by num samples)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--temp', type=float, default=0.001,
                        help='Temperature used for CoRe head detection (for comparison)')
    parser.add_argument('--save_best_only', action='store_true',
                        help='Only save the best model (default: save all lambda values)')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs (-1 for all CPUs, default: 1)')
    parser.add_argument('--max_iter', type=int, default=1000,
                        help='Maximum iterations for SAGA solver (default: 1000)')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Original JSON file to determine query groupings. '
                             'If not provided, assumes 5 position variations per query.')
    args = parser.parse_args()

    # Capture the command used to run this script
    command = ' '.join(sys.argv)

    print(f"Training head weights for {args.llm}")
    print(f"L1 regularization values: {args.lambda_l1}")

    # Load features
    if args.feature_file:
        print(f"\nLoading features from {args.feature_file}...")
    else:
        print(f"\nLoading features for {args.llm} (n={args.num_samples})...")
    try:
        X, y, docs_per_query = load_features(
            feature_file=args.feature_file,
            llm_name=args.llm,
            num_samples=args.num_samples
        )
    except FileNotFoundError as e:
        print(e)
        return

    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class balance: {y.sum()} positive, {len(y) - y.sum()} negative")

    # Get model config
    num_layers, num_heads = get_head_info(args.llm)
    print(f"Model config: {num_layers} layers, {num_heads} heads")

    # Load query groupings from JSON file if provided
    query_to_base = None
    n_queries = len(docs_per_query)

    if args.input_file is not None:
        input_path = Path(args.input_file)
        if input_path.exists():
            print(f"\nLoading query groups from {input_path}...")
            query_to_base, n_base_queries = load_query_groups_from_json(
                input_path, num_samples=args.num_samples
            )
            print(f"Found {n_base_queries} unique base queries from {n_queries} query samples")
        else:
            print(f"Warning: Input file not found: {input_path}")
            print("Falling back to assuming 5 position variations per query")
    else:
        # Try default location
        default_input = Path(__file__).parent.parent / 'head_data' / 'nq_core.json'
        if default_input.exists():
            print(f"\nLoading query groups from {default_input}...")
            query_to_base, n_base_queries = load_query_groups_from_json(
                default_input, num_samples=args.num_samples
            )
            print(f"Found {n_base_queries} unique base queries from {n_queries} query samples")
        else:
            print(f"\nNo input JSON file found, assuming 5 position variations per query")

    # Split data BY BASE QUERY (all variations stay together)
    X_train, X_val, y_train, y_val, train_q_idx, val_q_idx, n_base_queries = split_by_base_query(
        X, y, docs_per_query, val_split=args.val_split, random_state=42,
        query_to_base=query_to_base, positions_per_query=5
    )

    # Calculate number of base queries in each split
    if query_to_base is not None:
        train_base_set = set(query_to_base[q] for q in train_q_idx)
        val_base_set = set(query_to_base[q] for q in val_q_idx)
        n_train_base = len(train_base_set)
        n_val_base = len(val_base_set)
    else:
        n_train_base = len(train_q_idx) // 5
        n_val_base = len(val_q_idx) // 5

    print(f"\nSplit by base query (no overlap between train/val):")
    print(f"  Train: {len(y_train)} docs from {len(train_q_idx)} query samples ({n_train_base} base queries)")
    print(f"  Val:   {len(y_val)} docs from {len(val_q_idx)} query samples ({n_val_base} base queries)")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train with different regularization strengths
    output_dir = Path(__file__).parent.parent / 'head_data' / args.llm
    output_dir.mkdir(parents=True, exist_ok=True)

    n_jobs = args.n_jobs if args.n_jobs != 0 else 1
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    print(f"\n{'='*60}")
    print(f"Training Results (n_jobs={n_jobs}, max_iter={args.max_iter})")
    print('='*60)

    if n_jobs > 1 and len(args.lambda_l1) > 1:
        # Parallel training
        print(f"Training {len(args.lambda_l1)} models in parallel...")
        all_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(train_single_lambda)(
                lambda_l1, X_train_scaled, y_train, X_val_scaled, y_val,
                num_layers, num_heads, args.max_iter
            )
            for lambda_l1 in args.lambda_l1
        )
        # Sort by lambda for consistent ordering
        all_results.sort(key=lambda x: x['lambda_l1'])

        # Print results table after parallel completion
        print(f"\n{'Lambda':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC-ROC':<10} {'Non-zero':<10} {'Sparsity':<10}")
        print('-'*90)
        for result in all_results:
            metrics = result['metrics']
            print(f"{result['lambda_l1']:<10.4f} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f} {metrics['auc_roc']:<10.4f} "
                  f"{metrics['num_nonzero_weights']:<10d} {metrics['sparsity']:<10.4f}")
    else:
        # Sequential training (original behavior)
        print(f"{'Lambda':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC-ROC':<10} {'Non-zero':<10} {'Sparsity':<10}")
        print('-'*90)
        all_results = []
        for lambda_l1 in args.lambda_l1:
            result = train_single_lambda(
                lambda_l1, X_train_scaled, y_train, X_val_scaled, y_val,
                num_layers, num_heads, args.max_iter
            )
            all_results.append(result)

            # Print immediately in sequential mode
            metrics = result['metrics']
            print(f"{lambda_l1:<10.4f} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f} {metrics['auc_roc']:<10.4f} "
                  f"{metrics['num_nonzero_weights']:<10d} {metrics['sparsity']:<10.4f}", flush=True)

    # Find best model and save results
    best_result = max(all_results, key=lambda x: x['metrics']['auc_roc'])
    best_lambda = best_result['lambda_l1']
    best_auc = best_result['metrics']['auc_roc']

    # Save results
    for result in all_results:
        if not args.save_best_only or result['lambda_l1'] == best_lambda:
            output_file = save_results(
                output_dir, result['lambda_l1'], args.num_samples,
                result['metrics'], result['top_heads'],
                result['all_heads'], command,
                n_train_docs=len(y_train),
                n_val_docs=len(y_val),
                n_train_base_queries=n_train_base,
                n_val_base_queries=n_val_base
            )
            if not args.save_best_only:
                print(f"  Saved lambda={result['lambda_l1']} to {output_file}")

    print(f"\nBest model: lambda={best_lambda}, AUC-ROC={best_auc:.4f}")

    # Get best model results from stored data
    best_result = next(r for r in all_results if r['lambda_l1'] == best_lambda)
    top_heads = best_result['top_heads']
    all_heads = best_result['all_heads']

    # Analyze best model weights
    print(f"\n{'='*60}")
    print(f"Top 20 Heads by Learned Weight (lambda={best_lambda})")
    print('='*60)

    print(f"{'Rank':<6} {'Layer':<8} {'Head':<8} {'Weight':<12}")
    print('-'*40)
    for i, h in enumerate(top_heads):
        print(f"{i+1:<6} {h['layer']:<8} {h['head']:<8} {h['weight']:<12.6f}")

    # Compare with CoRe heads
    compare_with_core_heads(top_heads, args.llm, temp=args.temp)

    if args.save_best_only:
        print(f"\nSaved best model to {output_dir}/")
    else:
        print(f"\nAll {len(args.lambda_l1)} models saved to {output_dir}/")


if __name__ == '__main__':
    main()
