#!/usr/bin/env python3
"""
Train head weights using various loss functions with L1 regularization.

Supported loss functions:
- BCE: Binary Cross-Entropy (logistic regression)
- InfoNCE: Contrastive loss for listwise ranking

The loss is normalized by the number of samples:
    Loss = (1/n) * sum(loss) + lambda * ||w||_1

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
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from utils import log_command, load_features, get_head_info
from trainers import get_trainer, list_trainers


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
        X_train, X_val, y_train, y_val, docs_per_query_train, docs_per_query_val, train_query_indices, val_query_indices, n_base_queries
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

    # Collect document indices and docs_per_query for train/val
    train_doc_indices = []
    val_doc_indices = []
    train_docs_per_query = []
    val_docs_per_query = []

    doc_offset = 0
    for q_idx in range(n_queries):
        n_docs = docs_per_query[q_idx]
        doc_indices = list(range(doc_offset, doc_offset + n_docs))

        base_idx = query_to_base[q_idx]
        if base_idx in val_base_indices:
            val_doc_indices.extend(doc_indices)
            val_docs_per_query.append(n_docs)
        else:
            train_doc_indices.extend(doc_indices)
            train_docs_per_query.append(n_docs)

        doc_offset += n_docs

    # Create train/val splits
    train_doc_indices = np.array(train_doc_indices)
    val_doc_indices = np.array(val_doc_indices)
    train_docs_per_query = np.array(train_docs_per_query, dtype=np.int32)
    val_docs_per_query = np.array(val_docs_per_query, dtype=np.int32)

    X_train = X[train_doc_indices]
    X_val = X[val_doc_indices]
    y_train = y[train_doc_indices]
    y_val = y[val_doc_indices]

    return X_train, X_val, y_train, y_val, train_docs_per_query, val_docs_per_query, train_query_indices, val_query_indices, n_base_queries


def kfold_by_base_query(X, y, docs_per_query, n_folds=5, random_state=42,
                        query_to_base=None, positions_per_query=5):
    """
    Generate k-fold cross-validation splits at the base query level.

    Yields train/val data for each fold, ensuring all position variations
    of a query stay together in the same fold.

    Args:
        X: Feature matrix (n_docs, n_features)
        y: Labels (n_docs,)
        docs_per_query: Number of docs per query sample
        n_folds: Number of CV folds
        random_state: Random seed
        query_to_base: List mapping query index to base query index
        positions_per_query: Number of position variations per base query

    Yields:
        (X_train, X_val, y_train, y_val, docs_per_query_train, docs_per_query_val, fold_idx, n_base_queries) for each fold
    """
    n_queries = len(docs_per_query)

    # Determine query-to-base mapping
    if query_to_base is not None:
        n_base_queries = max(query_to_base) + 1
    else:
        n_base_queries = n_queries // positions_per_query
        query_to_base = [q_idx // positions_per_query for q_idx in range(n_queries)]

    # Create base query indices and shuffle
    base_query_indices = np.arange(n_base_queries)
    rng = np.random.RandomState(random_state)
    rng.shuffle(base_query_indices)

    # Compute fold sizes
    fold_size = n_base_queries // n_folds
    remainder = n_base_queries % n_folds

    # Build document offset map for fast lookup
    doc_offsets = np.zeros(n_queries + 1, dtype=np.int64)
    doc_offsets[1:] = np.cumsum(docs_per_query)

    # Generate folds
    start = 0
    for fold_idx in range(n_folds):
        # This fold gets one extra if fold_idx < remainder
        this_fold_size = fold_size + (1 if fold_idx < remainder else 0)
        end = start + this_fold_size

        val_base_indices = set(base_query_indices[start:end])

        # Collect document indices and docs_per_query for train/val
        train_doc_indices = []
        val_doc_indices = []
        train_docs_per_query = []
        val_docs_per_query = []

        for q_idx in range(n_queries):
            base_idx = query_to_base[q_idx]
            doc_start = doc_offsets[q_idx]
            doc_end = doc_offsets[q_idx + 1]
            doc_indices = list(range(doc_start, doc_end))

            if base_idx in val_base_indices:
                val_doc_indices.extend(doc_indices)
                val_docs_per_query.append(docs_per_query[q_idx])
            else:
                train_doc_indices.extend(doc_indices)
                train_docs_per_query.append(docs_per_query[q_idx])

        train_doc_indices = np.array(train_doc_indices)
        val_doc_indices = np.array(val_doc_indices)
        train_docs_per_query = np.array(train_docs_per_query, dtype=np.int32)
        val_docs_per_query = np.array(val_docs_per_query, dtype=np.int32)

        X_train = X[train_doc_indices]
        X_val = X[val_doc_indices]
        y_train = y[train_doc_indices]
        y_val = y[val_doc_indices]

        yield X_train, X_val, y_train, y_val, train_docs_per_query, val_docs_per_query, fold_idx, n_base_queries

        start = end


def train_single_lambda_cv(lambda_l1, X, y, docs_per_query, n_folds, num_layers, num_heads,
                           max_iter, query_to_base, positions_per_query, loss='bce', temperature=1.0):
    """
    Train a single lambda value with k-fold cross-validation.

    Returns:
        dict with lambda_l1, cv_metrics (mean/std), all_fold_metrics, and final model trained on all data
    """
    fold_metrics = []

    for X_train, X_val, y_train, y_val, docs_per_query_train, docs_per_query_val, fold_idx, n_base_queries in kfold_by_base_query(
        X, y, docs_per_query, n_folds=n_folds, random_state=42,
        query_to_base=query_to_base, positions_per_query=positions_per_query
    ):
        # Standardize features for this fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train model
        trainer, metrics = train_model(
            X_train_scaled, y_train, X_val_scaled, y_val,
            lambda_l1=lambda_l1, max_iter=max_iter, loss=loss,
            docs_per_query_train=docs_per_query_train,
            temperature=temperature
        )
        fold_metrics.append(metrics)

    # Aggregate metrics across folds
    cv_metrics = {}
    metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'num_nonzero_weights', 'sparsity']
    for key in metric_keys:
        values = [m[key] for m in fold_metrics]
        cv_metrics[f'{key}_mean'] = np.mean(values)
        cv_metrics[f'{key}_std'] = np.std(values)

    # Train final model on all data for saving weights
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # For final model, we use a dummy split (train on all, evaluate on all)
    final_trainer, _ = train_model(
        X_scaled, y, X_scaled, y, lambda_l1=lambda_l1, max_iter=max_iter, loss=loss,
        docs_per_query_train=docs_per_query,
        temperature=temperature
    )
    weights = final_trainer.get_weights()
    top_heads, all_heads = analyze_weights(weights, num_layers, num_heads, top_k=20)

    return {
        'lambda_l1': lambda_l1,
        'trainer': final_trainer,
        'cv_metrics': cv_metrics,
        'fold_metrics': fold_metrics,
        'top_heads': top_heads,
        'all_heads': all_heads
    }


def train_model(X_train, y_train, X_val, y_val, lambda_l1=0.01, max_iter=1000,
                loss='bce', docs_per_query_train=None, **trainer_kwargs):
    """
    Train a model using the specified loss function.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        lambda_l1: L1 regularization strength (normalized by num samples)
        max_iter: Maximum iterations for solver
        loss: Loss function name ('bce', 'infonce')
        docs_per_query_train: Docs per query (required for listwise losses)
        **trainer_kwargs: Additional arguments for the trainer

    Returns:
        trainer: Trained trainer object
        metrics: Dictionary of evaluation metrics
    """
    trainer = get_trainer(
        loss,
        lambda_l1=lambda_l1,
        max_iter=max_iter,
        **trainer_kwargs
    )

    trainer.fit(X_train, y_train, docs_per_query_train=docs_per_query_train)
    metrics = trainer.evaluate(X_val, y_val)

    return trainer, metrics


def analyze_weights(weights, num_layers, num_heads, top_k=20):
    """
    Analyze learned head weights.

    Args:
        weights: Weight array from trainer.get_weights()
        num_layers: Number of layers in model
        num_heads: Number of heads per layer
        top_k: Number of top heads to return

    Returns:
        top_heads: List of top-k heads by absolute weight
        all_heads: List of all heads with weights
    """

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
                 n_train_docs=None, n_val_docs=None, n_train_base_queries=None, n_val_base_queries=None,
                 loss='bce', llm_name=None, output_template=None, temperature=1.0):
    """Save results for a single lambda value.

    Args:
        output_dir: Directory to save results (used if output_template is None)
        lambda_l1: L1 regularization strength
        num_samples: Number of samples used for training
        metrics: Dictionary of evaluation metrics
        top_heads: List of top heads by weight
        all_heads: List of all heads with weights
        command: Command used to run the script
        n_train_docs: Number of training documents
        n_val_docs: Number of validation documents
        n_train_base_queries: Number of training base queries
        n_val_base_queries: Number of validation base queries
        loss: Loss function name
        llm_name: LLM name (for template substitution)
        output_template: Optional output file template with placeholders:
                        {lambda}, {n}, {loss}, {llm}
        temperature: Temperature value used for training (for bce_temp and infonce)
    """
    if output_template:
        # Substitute placeholders in template
        output_path = output_template.format(
            **{'lambda': lambda_l1, 'n': num_samples, 'loss': loss, 'llm': llm_name or 'unknown'}
        )
        output_file = Path(output_path)
        # Create parent directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file = get_unique_filepath(output_file)
    else:
        base_file = output_dir / f'{loss}_weights_lambda{lambda_l1}_n{num_samples}.json'
        output_file = get_unique_filepath(base_file)

    results = {
        'command': command,
        'timestamp': datetime.now().isoformat(),
        'loss': loss,
        'lambda_l1': float(lambda_l1),
        'temperature': float(temperature),
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


def train_single_lambda(lambda_l1, X_train, y_train, X_val, y_val, num_layers, num_heads,
                        max_iter=1000, loss='bce', docs_per_query_train=None, temperature=1.0):
    """
    Train a single model for one lambda value. Designed for parallel execution.

    Returns:
        dict with lambda_l1, metrics, top_heads, all_heads
    """
    trainer, metrics = train_model(
        X_train, y_train, X_val, y_val,
        lambda_l1=lambda_l1, max_iter=max_iter, loss=loss,
        docs_per_query_train=docs_per_query_train,
        temperature=temperature
    )
    weights = trainer.get_weights()
    top_heads, all_heads = analyze_weights(weights, num_layers, num_heads, top_k=20)

    return {
        'lambda_l1': lambda_l1,
        'trainer': trainer,
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
    parser.add_argument('--loss', type=str, default='bce',
                        choices=list_trainers(),
                        help=f'Loss function to use: {", ".join(list_trainers())} (default: bce)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (ignored if --cv is used)')
    parser.add_argument('--cv', type=int, default=None,
                        help='Number of cross-validation folds. If set, uses k-fold CV instead of single split.')
    parser.add_argument('--temp', type=float, default=0.001,
                        help='Temperature for scaling: (1) used in bce_temp trainer for logit scaling, '
                             '(2) used for CoRe head detection comparison. Lower values (e.g., 0.001) '
                             'make predictions sharper. Default matches CoRe: 0.001')
    parser.add_argument('--save_best_only', action='store_true',
                        help='Only save the best model (default: save all lambda values)')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs (-1 for all CPUs, default: 1)')
    parser.add_argument('--max_iter', type=int, default=1000,
                        help='Maximum iterations for SAGA solver (default: 1000)')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Original JSON file to determine query groupings. '
                             'If not provided, assumes 5 position variations per query.')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file template. Supports placeholders: {lambda}, {n}, {loss}, {llm}. '
                             'Example: "weights_{llm}_{loss}_lambda{lambda}.json". '
                             'Default: head_data/{llm}/{loss}_weights_lambda{lambda}_n{n}.json')
    args = parser.parse_args()

    # Capture the command used to run this script
    command = ' '.join(sys.argv)

    print(f"Training head weights for {args.llm}")
    print(f"Loss function: {args.loss}")
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
        print("Run extract_head_features.py first to generate features.")
        return

    # Handle missing docs_per_query (assume 50 docs per query, standard for NQ)
    if docs_per_query is None:
        n_queries = len(y) // 50
        docs_per_query = np.full(n_queries, 50, dtype=np.int32)
        print(f"docs_per_query not found in file, assuming 50 docs per query")

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

    # Train with different regularization strengths
    output_dir = Path(__file__).parent.parent / 'head_data' / args.llm
    output_dir.mkdir(parents=True, exist_ok=True)

    n_jobs = args.n_jobs if args.n_jobs != 0 else 1
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    # Determine n_base_queries for reporting
    if query_to_base is not None:
        n_base_queries = max(query_to_base) + 1
    else:
        n_base_queries = n_queries // 5

    # Cross-validation or single split
    if args.cv is not None and args.cv > 1:
        # K-fold cross-validation mode
        print(f"\n{'='*60}")
        print(f"Training with {args.cv}-fold Cross-Validation (n_jobs={n_jobs}, max_iter={args.max_iter})")
        print(f"Base queries: {n_base_queries} (split at base query level)")
        print('='*60)

        if n_jobs > 1 and len(args.lambda_l1) > 1:
            n_jobs = min(n_jobs, len(args.lambda_l1))
            print(f"Training {len(args.lambda_l1)} lambda values in parallel...")
            all_results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(train_single_lambda_cv)(
                    lambda_l1, X, y, docs_per_query, args.cv,
                    num_layers, num_heads, args.max_iter,
                    query_to_base, 5, args.loss, args.temp
                )
                for lambda_l1 in args.lambda_l1
            )
            all_results.sort(key=lambda x: x['lambda_l1'])
        else:
            all_results = []
            for lambda_l1 in args.lambda_l1:
                print(f"Training lambda={lambda_l1}...")
                result = train_single_lambda_cv(
                    lambda_l1, X, y, docs_per_query, args.cv,
                    num_layers, num_heads, args.max_iter,
                    query_to_base, 5, args.loss, args.temp
                )
                all_results.append(result)

        # Print CV results table (mean ± std)
        print(f"\n{'Lambda':<10} {'Accuracy':<14} {'Precision':<14} {'Recall':<14} {'F1':<14} {'AUC-ROC':<14} {'Non-zero':<14} {'Sparsity':<14}")
        print('-'*108)
        for result in all_results:
            m = result['cv_metrics']
            acc = f"{m['accuracy_mean']:.4f}±{m['accuracy_std']:.3f}"
            prec = f"{m['precision_mean']:.4f}±{m['precision_std']:.3f}"
            rec = f"{m['recall_mean']:.4f}±{m['recall_std']:.3f}"
            f1 = f"{m['f1_mean']:.4f}±{m['f1_std']:.3f}"
            auc = f"{m['auc_roc_mean']:.4f}±{m['auc_roc_std']:.3f}"
            nz = f"{m['num_nonzero_weights_mean']:.1f}±{m['num_nonzero_weights_std']:.1f}"
            sp = f"{m['sparsity_mean']:.4f}±{m['sparsity_std']:.3f}"
            print(f"{result['lambda_l1']:<10.4f} {acc:<14} {prec:<14} {rec:<14} {f1:<14} {auc:<14} {nz:<14} {sp:<14}")

        # Find best model by mean AUC-ROC
        best_result = max(all_results, key=lambda x: x['cv_metrics']['auc_roc_mean'])
        best_lambda = best_result['lambda_l1']
        best_auc = best_result['cv_metrics']['auc_roc_mean']
        best_auc_std = best_result['cv_metrics']['auc_roc_std']

        # Save results (final model trained on all data)
        for result in all_results:
            if not args.save_best_only or result['lambda_l1'] == best_lambda:
                # Use cv_metrics for saving, converting mean values to the expected format
                save_metrics = {k.replace('_mean', ''): v for k, v in result['cv_metrics'].items() if '_mean' in k}
                save_metrics['cv_folds'] = args.cv
                save_metrics['cv_std'] = {k.replace('_std', ''): v for k, v in result['cv_metrics'].items() if '_std' in k}

                output_file = save_results(
                    output_dir, result['lambda_l1'], args.num_samples,
                    save_metrics, result['top_heads'],
                    result['all_heads'], command,
                    n_train_docs=len(y),
                    n_val_docs=0,
                    n_train_base_queries=n_base_queries,
                    n_val_base_queries=0,
                    loss=args.loss,
                    llm_name=args.llm,
                    output_template=args.output,
                    temperature=args.temp
                )
                if not args.save_best_only:
                    print(f"  Saved lambda={result['lambda_l1']} to {output_file}")

        print(f"\nBest model: lambda={best_lambda}, AUC-ROC={best_auc:.4f}±{best_auc_std:.4f}")

    else:
        # Single train/val split mode (original behavior)
        X_train, X_val, y_train, y_val, docs_per_query_train, docs_per_query_val, train_q_idx, val_q_idx, n_base_queries = split_by_base_query(
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

        print(f"\n{'='*60}")
        print(f"Training Results (n_jobs={n_jobs}, max_iter={args.max_iter})")
        print('='*60)

        if n_jobs > 1 and len(args.lambda_l1) > 1:
            n_jobs = min(n_jobs, len(args.lambda_l1))
            # Parallel training
            print(f"Training {len(args.lambda_l1)} models in parallel...")
            all_results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(train_single_lambda)(
                    lambda_l1, X_train_scaled, y_train, X_val_scaled, y_val,
                    num_layers, num_heads, args.max_iter, args.loss, docs_per_query_train, args.temp
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
                    num_layers, num_heads, args.max_iter, args.loss, docs_per_query_train, args.temp
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
                    n_val_base_queries=n_val_base,
                    loss=args.loss,
                    llm_name=args.llm,
                    output_template=args.output,
                    temperature=args.temp
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
