#!/usr/bin/env python3
"""
Evaluate BEIR aggregate score by running reranking on all BEIR datasets.

Supports two modes:
1. Head weights mode: Uses attention head weights from .npz feature files
2. Embedding model mode: Uses bi-encoder or cross-encoder models on JSON input files

This script:
1. Finds all .npz feature files (head weights mode) or .json input files (embedding mode)
2. Runs rerank_with_head_weights.py on each dataset in parallel (configurable workers)
3. Averages cqadupstack-* datasets into a single score
4. Computes the overall BEIR average across datasets (14 main + 1 cqadupstack)
5. Displays results grouped by k value with color-coded highlighting

BEIR datasets (15 total after cqadupstack aggregation):
- 14 main: trec-covid, nfcorpus, dbpedia-entity, scifact, scidocs, fiqa, nq,
           fever, climate-fever, hotpotqa, webis-touche2020, msmarco, quora, arguana
- 1 aggregated: cqadupstack (average of up to 12 domains: android, english, gaming,
                gis, mathematica, physics, programmers, stats, tex, unix, webmasters,
                wordpress)

Features:
- Multiple k values: Evaluate across different numbers of retrieved documents
- Parallel execution: Run evaluations concurrently (default: 4 workers, configurable)
- Validation: Ensures all 14 main BEIR datasets are present for each k
- Flexible cqadupstack: Warns but continues if some cqadupstack domains are missing
- Numerical sorting: Results sorted by top-k value (baseline, oracle, then top-1, top-2, etc.)
- Color highlighting: Best score (green) and second-best (yellow) per metric column
  (oracle results excluded from highlighting to avoid skewing comparisons)
- Comprehensive output: Individual JSON files per dataset + aggregate results per k

Output files:
- {output_dir}/k{K}/{dataset}_metrics.json - Individual dataset results per k
- {output_dir}/k{K}/beir_aggregate.json - Aggregate BEIR scores per k
- {output_dir}/beir_aggregate_all.json - Combined results for all k values

Example usage:
    # Head weights mode - single k value
    python evaluate_beir_aggregate.py \\
        --llm mistral \\
        --weight_file head_data/mistral/bce_weights_lambda0.0001_n5000.json \\
        --feature_dir head_data/mistral \\
        --k 10 \\
        --top_k_heads 1 2 4 8 16 32 \\
        --n_jobs 8 \\
        --output_dir results/beir

    # Head weights mode - multiple k values
    python evaluate_beir_aggregate.py \\
        --llm mistral \\
        --weight_file head_data/mistral/bce_weights_lambda0.0001_n5000.json \\
        --feature_dir head_data/mistral \\
        --k 10 20 40 100 \\
        --top_k_heads 8 16 32 \\
        --n_jobs 8 \\
        --output_dir results/beir

    # Embedding model mode - cross-encoder
    python evaluate_beir_aggregate.py \\
        --reranker_model cross-encoder/ms-marco-MiniLM-L-6-v2 \\
        --input_dir retriever_output \\
        --beir_dir /path/to/beir \\
        --n_jobs 4 \\
        --output_dir results/beir_embedding

    # Embedding model mode - bi-encoder
    python evaluate_beir_aggregate.py \\
        --reranker_model sentence-transformers/all-MiniLM-L6-v2 \\
        --input_dir retriever_output \\
        --beir_dir /path/to/beir \\
        --output_dir results/beir_embedding

Example output:
    ======================================================================
    BEIR Aggregate Results (k=10 documents)
    ======================================================================
    Config               Weights    NDCG@1   NDCG@5   NDCG@10  MAP      MRR
    --------------------------------------------------------------------------
    baseline             retriever  0.3245   0.4123   0.4567   0.3890   0.4234
    oracle               gold@1     0.9756   0.9812   0.9845   0.9801   0.9823
    top-8                bce        0.3456   0.4389   0.4823   0.4123   0.4567  <- green
    top-16               bce        0.3512   0.4445   0.4891   0.4189   0.4623  <- yellow
    ======================================================================

    ======================================================================
    BEIR Aggregate Results (k=20 documents)
    ======================================================================
    ...

Notes:
- All 14 main BEIR datasets must be present for each k value (script exits if missing)
- Cqadupstack datasets are optional (warns if missing, uses available domains)
- Results sorted: baseline first, oracle second, then top-k numerically (1, 2, 8, 16...)
- Oracle shows upper bound performance (gold doc moved to rank 1), excluded from highlighting
- Use --no_oracle to skip oracle evaluation
- Use -v/--verbose for detailed progress information
"""

import argparse
import json
import subprocess
import sys
import os
import signal
import tempfile
import torch.multiprocessing as mp
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
from utils import log_command, parse_args_with_config, load_reranker

# Global list to track child processes for cleanup
_child_processes = []


def _cleanup_children(signum=None, frame=None):
    """Terminate all child processes on signal."""
    for p in _child_processes:
        if p.is_alive():
            p.terminate()
    for p in _child_processes:
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
    if signum is not None:
        sys.exit(1)

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


def get_qrels_path(beir_dir: Path, dataset: str) -> Path:
    """
    Get the qrels file path for a BEIR dataset.

    Args:
        beir_dir: Path to BEIR data directory
        dataset: Dataset name (e.g., 'nq', 'cqadupstack-android')

    Returns:
        Path to qrels file: {beir_dir}/{corpus}/qrels/test.tsv

    For cqadupstack, tries both path formats:
        - {beir_dir}/cqadupstack/android/qrels/test.tsv
        - {beir_dir}/cqadupstack-android/qrels/test.tsv
    """
    # Handle cqadupstack subdatasets
    if dataset.startswith('cqadupstack-'):
        domain = dataset.split('-', 1)[1]

        # Try nested format first: cqadupstack/android/qrels/test.tsv
        nested_path = beir_dir / 'cqadupstack' / domain / 'qrels' / 'test.tsv'
        if nested_path.exists():
            return nested_path

        # Try flat format: cqadupstack-android/qrels/test.tsv
        flat_path = beir_dir / dataset / 'qrels' / 'test.tsv'
        if flat_path.exists():
            return flat_path

        # Return nested path as default (will trigger warning if not found)
        return nested_path
    else:
        return beir_dir / dataset / 'qrels' / 'test.tsv'


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


def find_input_json_files(input_dir: Path) -> Dict[str, Path]:
    """
    Find all .json and .json.bz2 input files for embedding model reranking.

    Returns:
        dict mapping dataset name to input file path
    """
    # Find both .json and .json.bz2 files
    json_files = list(input_dir.glob("*.json"))
    bz2_files = list(input_dir.glob("*.json.bz2"))

    # Map files to dataset names
    dataset_files = {}

    for f in json_files:
        dataset = f.stem
        dataset_files[dataset] = f

    for f in bz2_files:
        # Remove both .json and .bz2 suffixes
        dataset = f.name[:-9]  # Remove '.json.bz2'
        # Only add if not already found as uncompressed
        if dataset not in dataset_files:
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


def gpu_worker_embedding(
    gpu_id: int,
    datasets: List[Tuple[str, Path, Optional[Path]]],  # (dataset_name, input_file, qrels_file)
    reranker_model: str,
    ks: List[int],
    model_batch_size: int,
    max_doc_tokens: int,
    output_file: str,
    output_dir: str = None
):
    """
    Worker function that processes multiple datasets on a specific GPU.

    Args:
        gpu_id: GPU device ID to use
        datasets: List of (dataset_name, input_file, qrels_file) tuples
        reranker_model: HuggingFace model name
        ks: K values for metrics
        model_batch_size: Batch size for inference
        max_doc_tokens: Max tokens per document
        output_file: Path to save results JSON (temp file for aggregation)
        output_dir: Directory to save individual dataset results (enables skip logic)
    """
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Limit parallelism
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    import torch
    import json
    import bz2
    import numpy as np
    from tqdm import tqdm

    # Print GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)  # 0 because we set CUDA_VISIBLE_DEVICES
        print(f"[GPU {gpu_id}] {gpu_name}", flush=True)
    else:
        print(f"[GPU {gpu_id}] WARNING: CUDA not available, using CPU", flush=True)

    # Import utilities
    from utils import load_reranker

    # Check which datasets already have results (skip if output_dir provided and files exist)
    datasets_to_process = []
    skipped_results = {}

    for dataset_name, input_file, qrels_file in datasets:
        if output_dir:
            reranked_file = os.path.join(output_dir, f"{dataset_name}.json.bz2")
            metrics_file = os.path.join(output_dir, f"{dataset_name}_metrics.json")
            # Check if both reranked data and metrics files exist
            if os.path.exists(reranked_file) and os.path.exists(metrics_file):
                try:
                    # Load metrics from the separate metrics file (not the reranked data list)
                    with open(metrics_file, 'r') as f:
                        metrics_data = json.load(f)
                    skipped_results[dataset_name] = {
                        'metrics': metrics_data.get('metrics', {}),
                        'num_queries': metrics_data.get('num_queries', 0),
                        'skipped': True
                    }
                    continue
                except Exception as e:
                    print(f"[GPU {gpu_id}] Warning: Failed to load metrics for {dataset_name}: {e}", flush=True)
        datasets_to_process.append((dataset_name, input_file, qrels_file))

    if skipped_results:
        print(f"[GPU {gpu_id}] Skipping {len(skipped_results)} datasets with existing results", flush=True)

    # If all datasets were skipped, save results and return early
    if not datasets_to_process:
        print(f"[GPU {gpu_id}] All datasets already processed", flush=True)
        with open(output_file, 'w') as f:
            json.dump(skipped_results, f, indent=2)
        return

    print(f"[GPU {gpu_id}] Starting worker with {len(datasets_to_process)} datasets", flush=True)

    # Load model using reranker class hierarchy
    print(f"[GPU {gpu_id}] Loading model: {reranker_model}", flush=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_reranker(reranker_model, device=device, verbose=True)
    print(f"[GPU {gpu_id}] Model loaded ({model.__class__.__name__})", flush=True)

    # Process each dataset - start with skipped results
    results = dict(skipped_results)

    # Statistics tracking
    import time
    start_time = time.time()
    total_queries = 0
    total_docs = 0
    total_tokens = 0  # Word count as proxy for tokens

    # Dataset-level progress bar (position 0)
    dataset_pbar = tqdm(datasets_to_process, desc=f"GPU {gpu_id} datasets", position=gpu_id*2, leave=True)

    for dataset_name, input_file, qrels_file in dataset_pbar:
        try:
            # Load input data
            input_str = str(input_file)
            if input_str.endswith('.bz2'):
                with bz2.open(input_str, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(input_str, 'r') as f:
                    data = json.load(f)

            # Load qrels if available
            qrels = None
            if qrels_file and qrels_file.exists():
                qrels = {}
                with open(qrels_file, 'r') as f:
                    first_line = f.readline().strip()
                    f.seek(0)
                    if first_line.startswith('query-id') or first_line.startswith('query_id'):
                        import csv
                        reader = csv.DictReader(f, delimiter='\t')
                        for row in reader:
                            q_id = str(row.get('query-id', row.get('query_id', '')))
                            d_id = str(row.get('corpus-id', row.get('corpus_id', row.get('doc-id', ''))))
                            score = int(row.get('score', row.get('relevance', 0)))
                            if q_id and d_id and score > 0:
                                if q_id not in qrels:
                                    qrels[q_id] = {}
                                qrels[q_id][d_id] = score
                    else:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 3:
                                if len(parts) == 3:
                                    q_id, d_id, score = parts
                                else:
                                    q_id, _, d_id, score = parts[:4]
                                score = int(float(score))
                                if score > 0:
                                    if q_id not in qrels:
                                        qrels[q_id] = {}
                                    qrels[q_id][d_id] = score

            # Process queries
            reranked_results = {}  # For metrics: {query_id: {doc_id: score}}
            reranked_data = []  # For output: same format as input

            # Query-level progress bar (position 1, shows dataset name)
            query_pbar = tqdm(data, desc=f"GPU {gpu_id} {dataset_name}", position=gpu_id*2+1, leave=False)

            for sample in query_pbar:
                query = sample.get('question', sample.get('query', ''))
                query_id = str(sample.get('idx', ''))
                paragraphs = sample.get('paragraphs', [])

                if not paragraphs or not query:
                    # Keep sample as-is if no paragraphs
                    reranked_data.append(sample)
                    continue

                # Extract documents for scoring
                docs = []
                query_tokens = len(query.split())
                for i, p in enumerate(paragraphs):
                    text = p.get('paragraph_text', '')
                    if max_doc_tokens:
                        text = ' '.join(text.split()[:max_doc_tokens])
                    docs.append(text)

                # Update statistics
                total_queries += 1
                total_docs += len(paragraphs)
                total_tokens += query_tokens + sum(len(doc.split()) for doc in docs)

                # Compute scores with OOM handling
                scores = None
                current_batch_size = model_batch_size
                pairs = [(query, doc) for doc in docs]

                for attempt in range(2):  # Try twice: original batch size, then batch_size=1
                    try:
                        scores = model.predict(pairs, batch_size=current_batch_size)
                        break  # Success
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower() or 'CUDA' in str(e):
                            torch.cuda.empty_cache()
                            if attempt == 0:
                                print(f"\n[GPU {gpu_id}] OOM at query {query_id} ({len(docs)} docs), retrying with batch_size=1", flush=True)
                                current_batch_size = 1
                            else:
                                print(f"\n[GPU {gpu_id}] OOM at query {query_id} even with batch_size=1, keeping original order", flush=True)
                                scores = None
                        else:
                            raise

                # Reorder paragraphs by score
                if scores is not None:
                    sorted_indices = np.argsort(-np.array(scores))
                    reranked_paragraphs = [paragraphs[i] for i in sorted_indices]
                    # Add scores to paragraphs
                    for rank, idx in enumerate(sorted_indices):
                        reranked_paragraphs[rank] = dict(reranked_paragraphs[rank])
                        reranked_paragraphs[rank]['rerank_score'] = float(scores[idx])
                    # Build metrics dict
                    reranked_results[query_id] = {
                        str(paragraphs[i].get('idx', f'doc_{i}')): float(scores[i])
                        for i in sorted_indices
                    }
                else:
                    # Keep original order
                    reranked_paragraphs = paragraphs
                    reranked_results[query_id] = {
                        str(p.get('idx', f'doc_{i}')): float(len(paragraphs) - i)
                        for i, p in enumerate(paragraphs)
                    }

                # Build reranked sample (same format as input)
                reranked_sample = dict(sample)
                reranked_sample['paragraphs'] = reranked_paragraphs
                reranked_data.append(reranked_sample)

            query_pbar.close()

            # Compute metrics if qrels available
            dataset_metrics = {}
            if qrels:
                try:
                    from beir.retrieval.evaluation import EvaluateRetrieval
                    evaluator = EvaluateRetrieval()
                    qrels_filtered = {q: qrels[q] for q in reranked_results if q in qrels}
                    if qrels_filtered:
                        ndcg, _map, recall, precision = evaluator.evaluate(qrels_filtered, reranked_results, ks)
                        dataset_metrics = {
                            **{f'NDCG@{k}': ndcg.get(f'NDCG@{k}', 0.0) for k in ks},
                            **{f'P@{k}': precision.get(f'P@{k}', 0.0) for k in ks},
                            **{f'Recall@{k}': recall.get(f'Recall@{k}', 0.0) for k in ks},
                        }
                        if _map:
                            dataset_metrics.update({k: v for k, v in _map.items()})
                except ImportError:
                    pass

            dataset_result = {
                'metrics': dataset_metrics,
                'num_queries': len(reranked_results)
            }
            results[dataset_name] = dataset_result

            # Save outputs to output_dir
            if output_dir:
                # Save reranked data as .json.bz2 (same format as input)
                reranked_file = os.path.join(output_dir, f"{dataset_name}.json.bz2")
                with bz2.open(reranked_file, 'wt', encoding='utf-8') as f:
                    json.dump(reranked_data, f)

                # Save metrics and info as separate .json file
                metrics_file = os.path.join(output_dir, f"{dataset_name}_metrics.json")
                metrics_output = {
                    'dataset': dataset_name,
                    'reranker_model': reranker_model,
                    'num_queries': len(reranked_results),
                    'metrics': dataset_metrics
                }
                with open(metrics_file, 'w') as f:
                    json.dump(metrics_output, f, indent=2)

        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing {dataset_name}: {e}", flush=True)
            results[dataset_name] = {'error': str(e)}

    # Close dataset progress bar
    dataset_pbar.close()

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print statistics
    elapsed_time = time.time() - start_time
    if elapsed_time > 0 and total_queries > 0:
        queries_per_sec = total_queries / elapsed_time
        docs_per_sec = total_docs / elapsed_time
        tokens_per_sec = total_tokens / elapsed_time
        print(f"\n[GPU {gpu_id}] Statistics:", flush=True)
        print(f"  Time: {elapsed_time:.1f}s", flush=True)
        print(f"  Queries: {total_queries} ({queries_per_sec:.1f}/s)", flush=True)
        print(f"  Documents: {total_docs} ({docs_per_sec:.1f}/s)", flush=True)
        print(f"  Tokens: {total_tokens} ({tokens_per_sec:.0f}/s)", flush=True)

    print(f"[GPU {gpu_id}] Done. Saved results to {output_file}", flush=True)

    # Clean up
    del model
    torch.cuda.empty_cache()


def run_reranking_embedding(
    dataset: str,
    input_file: Path,
    reranker_model: str,
    ks: List[int],
    evaluator: str,
    metrics: List[str],
    output_dir: Path,
    model_batch_size: int = 32,
    max_doc_tokens: int = 300,
    qrels_file: Path = None
) -> Tuple[str, bool, str]:
    """
    Run rerank_with_head_weights.py with embedding model for a single dataset.

    Returns:
        (dataset_name, success, output_json_path or error_message)
    """
    output_file = output_dir / f"{dataset}_metrics.json"

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "rerank_with_head_weights.py"),
        "--reranker_model", reranker_model,
        "--input_file", str(input_file),
        "--model_batch_size", str(model_batch_size),
        "--max_doc_tokens", str(max_doc_tokens),
        "--ks"] + [str(k) for k in ks] + [
        "--metrics"] + metrics + [
        "--evaluator", evaluator,
        "--output", str(output_file)
    ]

    if qrels_file is not None:
        cmd.extend(["--qrels", str(qrels_file)])

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
    no_oracle: bool,
    fusion: bool = False,
    rrf_k: int = 60,
    qrels_file: Path = None
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

    if qrels_file is not None:
        cmd.extend(["--qrels", str(qrels_file)])

    if no_baseline:
        cmd.append("--no_baseline")

    if no_oracle:
        cmd.append("--no_oracle")

    if fusion:
        cmd.extend(["--fusion", "--rrf_k", str(rrf_k)])

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

    # Extract metrics from file_results (head weights mode)
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


def parse_results_embedding(output_file: Path) -> Dict[str, Dict[str, float]]:
    """
    Parse embedding model reranking output JSON and extract metrics.

    Returns:
        dict mapping config to metrics dict
    """
    with open(output_file) as f:
        data = json.load(f)

    results = {}

    # Embedding model output format
    if 'metrics' in data:
        metrics_data = data['metrics']
        # Flatten metrics from nested structure
        flat_metrics = {}
        for metric_type, values in metrics_data.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    flat_metrics[k] = v
            else:
                flat_metrics[metric_type] = values

        # Use model name as config key
        model_name = data.get('reranker_model', 'embedding')
        # Shorten model name for display
        if '/' in model_name:
            model_name = model_name.split('/')[-1]

        results[f"rerank_{model_name}"] = flat_metrics

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


def print_corpus_breakdown(
    dataset_results: Dict[str, Dict[str, Dict[str, float]]],
    beir_avg: Dict[str, Dict[str, float]],
    display_metrics: List[str],
    k_val: int,
    no_baseline: bool = False,
    no_oracle: bool = False
):
    """
    Print per-corpus breakdown for baseline, oracle, and best performing config.

    Args:
        dataset_results: dict mapping dataset -> config -> metrics
        beir_avg: dict mapping config -> metrics (BEIR average)
        display_metrics: list of metric names to display (e.g., ['ndcg@10'])
        k_val: k value for display
        no_baseline: whether baseline was skipped
        no_oracle: whether oracle was skipped
    """
    # ANSI color codes
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    # Aggregate cqadupstack for display
    cqadupstack_agg = aggregate_cqadupstack(dataset_results)

    # Build display datasets: main + cqadupstack (aggregated)
    display_results = {d: dataset_results[d] for d in BEIR_MAIN_DATASETS if d in dataset_results}
    if cqadupstack_agg:
        display_results['cqadupstack'] = cqadupstack_agg

    # Order datasets: main datasets alphabetically, then cqadupstack at the end
    ordered_datasets = sorted([d for d in display_results.keys() if d != 'cqadupstack'])
    if 'cqadupstack' in display_results:
        ordered_datasets.append('cqadupstack')

    # Find configs to display
    configs_to_show = []

    # Find baseline config
    baseline_config = None
    for config_key in beir_avg.keys():
        if config_key.split('_', 1)[0] == 'baseline':
            baseline_config = config_key
            break

    # Find oracle config
    oracle_config = None
    for config_key in beir_avg.keys():
        if config_key.split('_', 1)[0] == 'oracle':
            oracle_config = config_key
            break

    # Find best non-baseline, non-oracle config by first display metric (typically ndcg@10)
    primary_metric = display_metrics[0] if display_metrics else 'ndcg@10'
    best_config = None
    best_value = -float('inf')

    for config_key, metrics in beir_avg.items():
        config_name = config_key.split('_', 1)[0]
        if config_name in ('baseline', 'oracle'):
            continue
        if primary_metric in metrics and metrics[primary_metric] > best_value:
            best_value = metrics[primary_metric]
            best_config = config_key

    # Build list of configs to show
    if not no_baseline and baseline_config:
        configs_to_show.append(('baseline', baseline_config))
    if not no_oracle and oracle_config:
        configs_to_show.append(('oracle', oracle_config))
    if best_config:
        # Extract display name (e.g., "top-8" from "top-8_bce")
        best_name = best_config.split('_', 1)[0]
        configs_to_show.append((f'best ({best_name})', best_config))

    if not configs_to_show:
        print("No configs available for corpus breakdown")
        return

    # Validate display metrics exist
    available_metrics = set()
    for config_key in beir_avg.keys():
        available_metrics.update(beir_avg[config_key].keys())

    valid_display_metrics = [m for m in display_metrics if m in available_metrics]
    if not valid_display_metrics:
        print(f"Warning: None of the requested metrics {display_metrics} found. Available: {sorted(available_metrics)}")
        return

    # Print header
    print(f"\n{'='*70}")
    print(f"Per-Corpus Breakdown (k={k_val} documents)")
    print(f"{'='*70}")

    # Calculate column widths
    dataset_col_width = max(len(d) for d in ordered_datasets) + 2
    metric_col_width = 10

    # Header row with config names
    header = f"{'Dataset':<{dataset_col_width}}"
    for display_name, _ in configs_to_show:
        for metric in valid_display_metrics:
            col_header = f"{display_name}" if len(valid_display_metrics) == 1 else f"{display_name}:{metric}"
            header += f" {col_header:>{metric_col_width}}"
    print(header)
    print("-" * len(header.replace('\033[', '').replace('m', '')))  # Approximate line width

    # Find max values per metric column for highlighting (exclude oracle)
    max_per_column = {}
    for display_name, config_key in configs_to_show:
        if display_name == 'oracle':
            continue
        for metric in valid_display_metrics:
            col_key = (display_name, metric)
            max_per_column[col_key] = -float('inf')
            for dataset in ordered_datasets:
                if dataset in display_results and config_key in display_results[dataset]:
                    val = display_results[dataset][config_key].get(metric, 0.0)
                    if val > max_per_column.get(col_key, -float('inf')):
                        max_per_column[col_key] = val

    # Print per-dataset rows
    for dataset in ordered_datasets:
        row = f"{dataset:<{dataset_col_width}}"

        # Find max value per metric for this row (excluding oracle)
        row_max = {}
        for metric in valid_display_metrics:
            max_val = -float('inf')
            for display_name, config_key in configs_to_show:
                if display_name == 'oracle':
                    continue
                if dataset in display_results and config_key in display_results[dataset]:
                    val = display_results[dataset][config_key].get(metric, 0.0)
                    if val > max_val:
                        max_val = val
            row_max[metric] = max_val

        for display_name, config_key in configs_to_show:
            for metric in valid_display_metrics:
                if dataset in display_results and config_key in display_results[dataset]:
                    val = display_results[dataset][config_key].get(metric, 0.0)
                    # Color the max value green (excluding oracle from coloring)
                    if display_name != 'oracle' and val == row_max[metric] and val > -float('inf'):
                        formatted = f"{GREEN}{val:>{metric_col_width}.3f}{RESET}"
                    else:
                        formatted = f"{val:>{metric_col_width}.3f}"
                else:
                    formatted = f"{'N/A':>{metric_col_width}}"

                row += f" {formatted}"

        print(row)

    # Print separator and BEIR average
    print("-" * len(header.replace('\033[', '').replace('m', '')))

    # Find max value per metric for BEIR average (excluding oracle)
    avg_max = {}
    for metric in valid_display_metrics:
        max_val = -float('inf')
        for display_name, config_key in configs_to_show:
            if display_name == 'oracle':
                continue
            if config_key in beir_avg:
                val = beir_avg[config_key].get(metric, 0.0)
                if val > max_val:
                    max_val = val
        avg_max[metric] = max_val

    avg_row = f"{'BEIR Average':<{dataset_col_width}}"
    for display_name, config_key in configs_to_show:
        for metric in valid_display_metrics:
            if config_key in beir_avg:
                val = beir_avg[config_key].get(metric, 0.0)
                # Color the max value green+bold (excluding oracle from coloring)
                if display_name != 'oracle' and val == avg_max[metric] and val > -float('inf'):
                    formatted = f"{GREEN}{BOLD}{val:>{metric_col_width}.3f}{RESET}"
                else:
                    formatted = f"{val:>{metric_col_width}.3f}"
            else:
                formatted = f"{'N/A':>{metric_col_width}}"
            avg_row += f" {formatted}"

    print(avg_row)


def main():
    log_command()

    parser = argparse.ArgumentParser(
        description='Evaluate BEIR aggregate score across all datasets with parallel execution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script evaluates reranking performance on BEIR benchmark datasets:
- Supports multiple k values (number of retrieved documents) in a single run
- Runs rerank_with_head_weights.py on all BEIR datasets in parallel
- Aggregates 12 cqadupstack domains into a single score
- Computes BEIR average across 15 datasets (14 main + cqadupstack)
- Computes baseline (original retriever) and oracle (upper bound) performance
- Color-highlights best (green) and second-best (yellow) scores per metric
- Results grouped by k value for easy comparison

Requirements:
- All 14 main BEIR datasets must have feature files for each k (script exits if missing)
- Cqadupstack datasets are optional (warns if missing, uses available)

Output:
- Per-k results: {output_dir}/k{K}/{dataset}_metrics.json
- Per-k aggregate: {output_dir}/k{K}/beir_aggregate.json
- Combined results: {output_dir}/beir_aggregate_all.json

Example (multiple k values):
  python evaluate_beir_aggregate.py \\
      --llm mistral \\
      --weight_file head_data/mistral/bce_weights_lambda0.0001_n5000.json \\
      --feature_dir head_data/mistral \\
      --k 10 20 40 100 \\
      --top_k_heads 8 16 32 \\
      --n_jobs 8 \\
      --output_dir results/beir
        """
    )

    parser.add_argument('--llm', type=str, default='mistral',
                        choices=['mistral', 'llama', 'phi', 'granite'],
                        help='LLM model name (required for head weights mode)')

    # Mutually exclusive: weight_file OR reranker_model (required=False for config file support)
    reranker_group = parser.add_mutually_exclusive_group(required=False)
    reranker_group.add_argument('--weight_file', type=str, default=None,
                                help='Path to head weights file (BCE or CoRe JSON)')
    reranker_group.add_argument('--reranker_model', type=str, default=None,
                                help='HuggingFace embedding model for reranking (bi-encoder or cross-encoder). '
                                     'Examples: BAAI/bge-reranker-base, sentence-transformers/all-MiniLM-L6-v2')

    # Head weights mode arguments
    parser.add_argument('--feature_dir', type=str, default=None,
                        help='Directory containing feature .npz files (required for head weights mode)')
    parser.add_argument('--k', type=int, nargs='+', default=None,
                        help='K value(s) for feature files - matches *_k{K}.npz pattern (required for head weights mode)')
    parser.add_argument('--top_k_heads', type=int, nargs='+', default=None,
                        help='List of top-k head values to evaluate (required for head weights mode)')

    # Embedding model mode arguments
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Directory containing input JSON files (required for embedding model mode)')
    parser.add_argument('--model_batch_size', type=int, default=32,
                        help='Batch size for embedding model inference (default: 32)')
    parser.add_argument('--max_doc_tokens', type=int, default=300,
                        help='Maximum tokens per document for embedding model (default: 300)')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU IDs for embedding model mode (e.g., "0,1,2,3"). '
                             'Each GPU runs one process with its own model instance. '
                             'Datasets are distributed across GPUs.')
    parser.add_argument('--ks', type=int, nargs='+', default=[1, 5, 10],
                        help='K values for @k metrics - controls NDCG@K, P@K, etc. (default: 1 5 10)')
    parser.add_argument('--metrics', type=str, nargs='+',
                        default=['ndcg', 'p', 'm', 'map', 'mrr'],
                        help='Metrics to compute: ndcg, p (precision), m (match), map, mrr (default: all)')
    parser.add_argument('--evaluator', type=str, default='beir',
                        choices=['custom', 'beir'],
                        help='Evaluator to use: beir (default, requires beir package) or custom (built-in)')
    parser.add_argument('--beir_dir', type=str, default=None,
                        help='Path to BEIR data directory containing qrels files (e.g., /path/to/beir). '
                             'Qrels are loaded from {beir_dir}/{corpus}/qrels/test.tsv. '
                             'Required when using --evaluator beir for proper NDCG computation.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results - saves {dataset}_metrics.json and beir_aggregate.json')
    parser.add_argument('--no_baseline', action='store_true',
                        help='Skip baseline retriever evaluation (only evaluate reranked results)')
    parser.add_argument('--no_oracle', action='store_true',
                        help='Skip oracle (upper bound) evaluation')
    parser.add_argument('--fusion', action='store_true',
                        help='Include RRF fusion of baseline and reranked results (combines retriever + attention head scores)')
    parser.add_argument('--rrf_k', type=int, default=60,
                        help='RRF constant k for fusion (default: 60)')
    parser.add_argument('--n_jobs', type=int, default=4,
                        help='Number of parallel worker processes for dataset evaluation (default: 4)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print datasets and configuration without executing evaluations')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed progress information (dataset status, parsing, etc.)')
    parser.add_argument('--display_metrics', type=str, nargs='+', default=['NDCG@10'],
                        help='Metrics to display in corpus breakdown (default: NDCG@10). '
                             'Examples: ndcg@1 ndcg@5 ndcg@10 map mrr')
    parser.add_argument('--no_corpus_breakdown', action='store_true',
                        help='Skip per-corpus breakdown table (only show aggregate)')

    args = parse_args_with_config(parser)

    # Validate required arguments
    if args.weight_file is None and args.reranker_model is None:
        parser.error("one of the arguments --weight_file --reranker_model is required")
    if args.weight_file is not None and args.reranker_model is not None:
        parser.error("argument --weight_file: not allowed with argument --reranker_model")
    if args.output_dir is None:
        parser.error("the following argument is required: --output_dir")

    # Determine mode: head weights vs embedding model
    embedding_mode = args.reranker_model is not None

    # Validate mode-specific arguments
    if embedding_mode:
        # Embedding model mode
        if args.input_dir is None:
            print("Error: --input_dir is required when using --reranker_model")
            sys.exit(1)

        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}")
            sys.exit(1)

        print(f"Mode: Embedding model reranking")
        print(f"Model: {args.reranker_model}")
        print(f"Input directory: {input_dir}")
    else:
        # Head weights mode
        if args.feature_dir is None:
            print("Error: --feature_dir is required when using --weight_file")
            sys.exit(1)
        if args.k is None:
            print("Error: --k is required when using --weight_file")
            sys.exit(1)
        if args.top_k_heads is None:
            print("Error: --top_k_heads is required when using --weight_file")
            sys.exit(1)

        feature_dir = Path(args.feature_dir)
        weight_file = Path(args.weight_file)

        if not feature_dir.exists():
            print(f"Error: Feature directory not found: {feature_dir}")
            sys.exit(1)

        if not weight_file.exists():
            print(f"Error: Weight file not found: {weight_file}")
            sys.exit(1)

        print(f"Mode: Head weights reranking")
        print(f"Weight file: {weight_file}")
        print(f"Feature directory: {feature_dir}")

    output_dir = Path(args.output_dir)

    # Resolve beir_dir if provided
    beir_dir = Path(args.beir_dir) if args.beir_dir else None
    if beir_dir is not None:
        if not beir_dir.exists():
            print(f"Error: BEIR directory not found: {beir_dir}")
            sys.exit(1)
        print(f"Using BEIR qrels from: {beir_dir}")
    elif args.evaluator == 'beir':
        print("Warning: --beir_dir not provided with --evaluator beir. NDCG will be computed using labels from .npz files only.")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.display_metrics is not None:
        args.display_metrics = [s.upper() for s in args.display_metrics]

    # Store results for all k values
    all_k_results = {}  # k -> beir_avg
    all_k_dataset_results = {}  # k -> dataset_results (for corpus breakdown)
    all_failed = []

    # ==========================================================================
    # EMBEDDING MODEL MODE
    # ==========================================================================
    if embedding_mode:
        # Find input JSON files
        if args.verbose:
            print(f"\nFinding input JSON files in {input_dir}...")
        dataset_files = find_input_json_files(input_dir)
        if args.verbose:
            print(f"Found {len(dataset_files)} datasets")

        # Validate datasets
        is_valid, issues = validate_datasets(dataset_files)
        if not is_valid and args.verbose:
            print("\n⚠️  Dataset validation warnings:")
            for issue in issues:
                print(f"  - {issue}")
            print()

        # Check for required datasets
        missing_main = set(BEIR_MAIN_DATASETS) - set(dataset_files.keys())
        missing_cqa = [f'cqadupstack-{d}' for d in CQADUPSTACK_DOMAINS if f'cqadupstack-{d}' not in dataset_files]
        found_cqa = [f'cqadupstack-{d}' for d in CQADUPSTACK_DOMAINS if f'cqadupstack-{d}' in dataset_files]

        if missing_main:
            print(f"Warning: Missing main BEIR datasets: {sorted(missing_main)}")
            print("Continuing with available datasets...")

        if missing_cqa and args.verbose:
            print(f"⚠️  Warning: Missing {len(missing_cqa)}/12 cqadupstack datasets")
            print(f"    Will aggregate using {len(found_cqa)} available cqadupstack datasets")

        if args.dry_run:
            print(f"\n[DRY RUN] Would evaluate the following datasets:")
            for dataset in sorted(dataset_files.keys()):
                print(f"  - {dataset}: {dataset_files[dataset].name}")
            print(f"\nOutput directory: {output_dir}")
            if args.gpus:
                print(f"GPUs: {args.gpus}")
            else:
                print(f"Parallel jobs: {args.n_jobs}")
            return

        # Create output directory
        emb_output_dir = output_dir / "embedding"
        emb_output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare dataset list with qrels
        datasets_with_qrels = []
        for dataset, input_file in dataset_files.items():
            qrels_file = None
            if beir_dir is not None:
                qrels_file = get_qrels_path(beir_dir, dataset)
                if not qrels_file.exists():
                    if args.verbose:
                        print(f"Warning: qrels file not found for {dataset}: {qrels_file}")
                    qrels_file = None
            datasets_with_qrels.append((dataset, input_file, qrels_file))

        # Check if using multi-GPU mode
        if args.gpus:
            # Multi-GPU mode: one process per GPU
            gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]
            num_gpus = len(gpu_ids)

            print(f"\nMulti-GPU mode: distributing {len(datasets_with_qrels)} datasets across {num_gpus} GPUs")

            # Distribute datasets across GPUs
            datasets_per_gpu = [[] for _ in range(num_gpus)]
            for i, dataset_info in enumerate(datasets_with_qrels):
                datasets_per_gpu[i % num_gpus].append(dataset_info)

            for i, gpu_id in enumerate(gpu_ids):
                print(f"  GPU {gpu_id}: {len(datasets_per_gpu[i])} datasets")

            # Create temp files for results
            temp_dir = tempfile.mkdtemp(prefix='beir_embedding_')
            temp_files = [os.path.join(temp_dir, f'gpu_{gpu_id}_results.json') for gpu_id in gpu_ids]

            # Set multiprocessing start method
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass  # Already set

            # Launch processes
            processes = []
            global _child_processes
            _child_processes = []

            # Set up signal handlers for cleanup
            original_sigint = signal.signal(signal.SIGINT, _cleanup_children)
            original_sigterm = signal.signal(signal.SIGTERM, _cleanup_children)

            try:
                for i, gpu_id in enumerate(gpu_ids):
                    if not datasets_per_gpu[i]:
                        continue

                    p = mp.Process(
                        target=gpu_worker_embedding,
                        args=(
                            gpu_id,
                            datasets_per_gpu[i],
                            args.reranker_model,
                            args.ks,
                            args.model_batch_size,
                            args.max_doc_tokens,
                            temp_files[i],
                            str(emb_output_dir)
                        )
                    )
                    processes.append((p, i, gpu_id))
                    _child_processes.append(p)

                print("\nStarting GPU workers...")
                for p, i, gpu_id in processes:
                    p.start()

                # Wait for completion
                for p, i, gpu_id in processes:
                    p.join()

            except KeyboardInterrupt:
                print("\nInterrupted! Terminating workers...")
                _cleanup_children()
                sys.exit(1)
            finally:
                # Restore original signal handlers
                signal.signal(signal.SIGINT, original_sigint)
                signal.signal(signal.SIGTERM, original_sigterm)
                _child_processes = []

            print("\nAll workers completed. Merging results...")

            # Merge results from all GPUs
            dataset_results = {}
            for i, (_, idx, gpu_id) in enumerate(processes):
                temp_file = temp_files[idx]
                if os.path.exists(temp_file):
                    with open(temp_file, 'r') as f:
                        gpu_results = json.load(f)

                    for dataset_name, result in gpu_results.items():
                        if 'error' in result:
                            all_failed.append(dataset_name)
                            if args.verbose:
                                print(f"  ✗ {dataset_name}: {result['error']}")
                        else:
                            metrics = result.get('metrics', {})
                            if metrics:
                                # Format as expected by aggregation functions
                                model_name = args.reranker_model.split('/')[-1]
                                dataset_results[dataset_name] = {
                                    f"rerank_{model_name}": metrics
                                }
                            if args.verbose:
                                print(f"  ✓ {dataset_name}: {result.get('num_queries', 0)} queries")

                    os.remove(temp_file)

            # Clean up temp dir
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass

        else:
            # Single-process mode using subprocess (original behavior)
            # Check which datasets already have results (.json.bz2 files)
            import bz2 as bz2_module
            datasets_to_run = []
            skipped_results = {}
            for dataset, input_file, qrels_file in datasets_with_qrels:
                reranked_file = emb_output_dir / f"{dataset}.json.bz2"
                if reranked_file.exists():
                    try:
                        with bz2_module.open(reranked_file, 'rt', encoding='utf-8') as f:
                            existing = json.load(f)
                        skipped_results[dataset] = str(reranked_file)
                    except Exception:
                        datasets_to_run.append((dataset, input_file, qrels_file))
                else:
                    datasets_to_run.append((dataset, input_file, qrels_file))

            if skipped_results:
                if args.verbose:
                    print(f"\nSkipping {len(skipped_results)} datasets with existing results")

            if args.verbose:
                print(f"\nRunning evaluations on {len(datasets_to_run)} datasets with {args.n_jobs} parallel jobs...")

            results = dict(skipped_results)  # Start with skipped results
            failed = []

            if datasets_to_run:
                with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
                    futures = {}
                    for dataset, input_file, qrels_file in datasets_to_run:
                        future = executor.submit(
                            run_reranking_embedding,
                            dataset,
                            input_file,
                            args.reranker_model,
                            args.ks,
                            args.evaluator,
                            args.metrics,
                            emb_output_dir,
                            args.model_batch_size,
                            args.max_doc_tokens,
                            qrels_file
                        )
                        futures[future] = dataset

                    for future in as_completed(futures):
                        dataset = futures[future]
                        try:
                            dataset_name, success, result = future.result()
                            if success:
                                if args.verbose:
                                    print(f"✓ {dataset_name}")
                                results[dataset_name] = result
                            else:
                                if args.verbose:
                                    print(f"✗ {dataset_name}: {result}")
                                failed.append(dataset_name)
                                all_failed.append(dataset_name)
                        except Exception as e:
                            if args.verbose:
                                print(f"✗ {dataset}: {e}")
                            failed.append(dataset)
                            all_failed.append(dataset)

            if failed and args.verbose:
                print(f"\n⚠️  {len(failed)} datasets failed:")
                for dataset in failed:
                    print(f"  - {dataset}")

            # Parse all results
            if args.verbose:
                print("\nParsing results...")
            dataset_results = {}
            for dataset, output_file_path in results.items():
                try:
                    dataset_results[dataset] = parse_results_embedding(Path(output_file_path))
                except Exception as e:
                    if args.verbose:
                        print(f"Error parsing {dataset}: {e}")

        # Compute BEIR average
        if args.verbose:
            print("\nComputing BEIR aggregate scores...")
        beir_avg = compute_beir_average(dataset_results)
        all_k_results['embedding'] = beir_avg
        all_k_dataset_results['embedding'] = dataset_results

        # Save aggregated results
        output_file = emb_output_dir / "beir_aggregate.json"
        cqadupstack_agg = aggregate_cqadupstack(dataset_results)

        output_data = {
            'reranker_model': args.reranker_model,
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
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Saved aggregate results to {output_file}")

        # Print results
        print("\n" + "="*70)
        print(f"BEIR Aggregate Results (Embedding Model)")
        print(f"Model: {args.reranker_model}")
        print("="*70)

        if beir_avg:
            first_config = list(beir_avg.keys())[0]
            metric_names = list(beir_avg[first_config].keys())

            # Print header
            header = f"{'Config':<30}"
            for metric in metric_names:
                header += f" {metric:<10}"
            print(header)
            print("-" * len(header))

            # Print results
            for config_key, metrics in beir_avg.items():
                line = f"{config_key:<30}"
                for metric in metric_names:
                    value = metrics.get(metric, 0.0)
                    line += f" {value:<10.4f}"
                print(line)

        print("="*70)

        if all_failed:
            sys.exit(1)
        return

    # ==========================================================================
    # HEAD WEIGHTS MODE
    # ==========================================================================
    # Process each k value
    k_values = sorted(args.k)

    for k_val in k_values:
        if args.verbose:
            print(f"\n{'='*70}")
            print(f"Processing k={k_val}")
            print(f"{'='*70}")

        # Find feature files for this k
        if args.verbose:
            print(f"Finding feature files in {feature_dir} for k={k_val}...")
        dataset_files = find_feature_files(feature_dir, k_val)
        if args.verbose:
            print(f"Found {len(dataset_files)} datasets")

        # Validate datasets
        is_valid, issues = validate_datasets(dataset_files)
        if not is_valid and args.verbose:
            print("\n⚠️  Dataset validation warnings:")
            for issue in issues:
                print(f"  - {issue}")
            print()

        # Check for required datasets
        missing_main = set(BEIR_MAIN_DATASETS) - set(dataset_files.keys())
        missing_cqa = [f'cqadupstack-{d}' for d in CQADUPSTACK_DOMAINS if f'cqadupstack-{d}' not in dataset_files]
        found_cqa = [f'cqadupstack-{d}' for d in CQADUPSTACK_DOMAINS if f'cqadupstack-{d}' in dataset_files]

        if missing_main:
            print(f"Error: Missing required main BEIR datasets for k={k_val}: {sorted(missing_main)}")
            sys.exit(1)

        if missing_cqa and args.verbose:
            print(f"⚠️  Warning: Missing {len(missing_cqa)}/12 cqadupstack datasets: {sorted(missing_cqa)}")
            print(f"    Will aggregate using {len(found_cqa)} available cqadupstack datasets")

        if args.verbose:
            print(f"✓ All {len(BEIR_MAIN_DATASETS)} main BEIR datasets found for k={k_val}")

        if args.dry_run:
            print(f"\n[DRY RUN] Would evaluate the following datasets for k={k_val}:")
            for dataset in sorted(dataset_files.keys()):
                print(f"  - {dataset}: {dataset_files[dataset].name}")
            continue

        # Create k-specific output directory
        k_output_dir = output_dir / f"k{k_val}"
        k_output_dir.mkdir(parents=True, exist_ok=True)

        # Run evaluations in parallel
        if args.verbose:
            print(f"\nRunning evaluations on {len(dataset_files)} datasets with {args.n_jobs} parallel jobs...")

        results = {}
        failed = []

        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            futures = {}
            for dataset, feature_file in dataset_files.items():
                # Get qrels file path if beir_dir is provided
                qrels_file = None
                if beir_dir is not None:
                    qrels_file = get_qrels_path(beir_dir, dataset)
                    if not qrels_file.exists():
                        print(f"Warning: qrels file not found for {dataset}: {qrels_file}")
                        qrels_file = None

                future = executor.submit(
                    run_reranking,
                    dataset,
                    feature_file,
                    args.llm,
                    weight_file,
                    args.top_k_heads,
                    args.ks,
                    args.evaluator,
                    args.metrics,
                    k_output_dir,
                    args.no_baseline,
                    args.no_oracle,
                    args.fusion,
                    args.rrf_k,
                    qrels_file
                )
                futures[future] = dataset

            for future in as_completed(futures):
                dataset = futures[future]
                try:
                    dataset_name, success, result = future.result()
                    if success:
                        if args.verbose:
                            print(f"✓ {dataset_name}")
                        results[dataset_name] = result
                    else:
                        if args.verbose:
                            print(f"✗ {dataset_name}: {result}")
                        failed.append(dataset_name)
                        all_failed.append(f"k={k_val}:{dataset_name}")
                except Exception as e:
                    if args.verbose:
                        print(f"✗ {dataset}: {e}")
                    failed.append(dataset)
                    all_failed.append(f"k={k_val}:{dataset}")

        if failed and args.verbose:
            print(f"\n⚠️  {len(failed)} datasets failed for k={k_val}:")
            for dataset in failed:
                print(f"  - {dataset}")

        # Parse all results
        if args.verbose:
            print("\nParsing results...")
        dataset_results = {}
        for dataset, output_file_path in results.items():
            try:
                dataset_results[dataset] = parse_results(Path(output_file_path))
            except Exception as e:
                if args.verbose:
                    print(f"Error parsing {dataset}: {e}")

        # Compute BEIR average
        if args.verbose:
            print("\nComputing BEIR aggregate scores...")
        beir_avg = compute_beir_average(dataset_results)
        all_k_results[k_val] = beir_avg
        all_k_dataset_results[k_val] = dataset_results

        # Save per-k aggregated results
        k_output_file = k_output_dir / "beir_aggregate.json"
        cqadupstack_agg = aggregate_cqadupstack(dataset_results)

        k_output_data = {
            'llm': args.llm,
            'weight_file': str(weight_file),
            'k': k_val,
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

        with open(k_output_file, 'w') as f:
            json.dump(k_output_data, f, indent=2)

        if args.verbose:
            print(f"✓ Saved aggregate results to {k_output_file}")

    if args.dry_run:
        print(f"\nOutput directory: {output_dir}")
        print(f"Parallel jobs: {args.n_jobs}")
        return

    # Save combined results for all k values
    combined_output_file = output_dir / "beir_aggregate_all.json"
    combined_output = {
        'llm': args.llm,
        'weight_file': str(weight_file),
        'k_values': k_values,
        'top_k_heads': args.top_k_heads,
        'ks': args.ks,
        'metrics': args.metrics,
        'evaluator': args.evaluator,
        'results_by_k': {str(k): beir_avg for k, beir_avg in all_k_results.items()}
    }

    with open(combined_output_file, 'w') as f:
        json.dump(combined_output, f, indent=2)

    if args.verbose:
        print(f"\n✓ Saved combined results to {combined_output_file}")

    # ANSI color codes
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

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

    # Print summary grouped by k
    for k_val in k_values:
        beir_avg = all_k_results.get(k_val, {})

        print("\n" + "="*70)
        print(f"BEIR Aggregate Results (k={k_val} documents)")
        print("="*70)

        if not beir_avg:
            print("No results available")
            continue

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

        # Print header
        header = f"{'Config':<20} {'Weights':<10}"
        for metric in metric_names:
            header += f" {metric:<8}"
        print(header)
        print("-" * len(header))

        # Print results
        for config_key in sorted(beir_avg.keys(), key=sort_key):
            parts = config_key.split('_', 1)
            config = parts[0]
            weights = parts[1] if len(parts) > 1 else ''

            line = f"{config:<20} {weights:<10}"
            for metric in metric_names:
                value = beir_avg[config_key].get(metric, 0.0)
                formatted = f"{value:<8.3f}"

                # Skip coloring for oracle rows
                if config != 'oracle':
                    # Highlight max in green+bold
                    if metric in metric_max and value == metric_max[metric]:
                        formatted = f"{GREEN}{BOLD}{value:<8.3f}{RESET}"
                    # Highlight second max in yellow+bold
                    elif metric in metric_second_max and value == metric_second_max[metric]:
                        formatted = f"{YELLOW}{BOLD}{value:<8.3f}{RESET}"

                line += f" {formatted}"
            print(line)

        # Print per-corpus breakdown if requested
        if not args.no_corpus_breakdown:
            dataset_results = all_k_dataset_results.get(k_val, {})
            if dataset_results:
                print_corpus_breakdown(
                    dataset_results,
                    beir_avg,
                    args.display_metrics,
                    k_val,
                    no_baseline=args.no_baseline,
                    no_oracle=args.no_oracle
                )

    print("="*70)

    if all_failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
