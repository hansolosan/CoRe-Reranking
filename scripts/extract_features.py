#!/usr/bin/env python3
"""
Unified feature extraction script with optional IDF/stopword filtering.

This script extracts attention features from LLM heads for document reranking.
When --filter_mode is 'none' (default), it behaves like extract_head_features.py.
When --filter_mode is 'stopwords', 'idf', or 'high_freq', it applies token filtering.

Supports multi-GPU processing with the --gpus flag.

Usage:
    # Standard extraction (no filtering)
    python extract_features.py --llm mistral --input_file data.json

    # With stopword filtering
    python extract_features.py --llm mistral --input_file data.json --filter_mode stopwords

    # With IDF weighting (requires pre-computed IDF file)
    python extract_features.py --llm mistral --input_file data.json --filter_mode idf

    # Multi-GPU extraction
    python extract_features.py --llm mistral --input_file data.json --gpus 0,1,2,3

    # Compute corpus IDF first
    python extract_features.py --llm mistral --compute_idf --corpus_dir /path/to/beir
"""

import gc
import os
import json
import argparse
import torch
import torch.multiprocessing as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
sys.path.insert(0, str(Path(__file__).parent))

import transformers
from extract_head_features import (
    HFFeatureExtractor, open_file, detect_input_format, LLM_NAMES,
    load_qrels, get_label_from_qrels
)
from extract_features_idf import (
    IDFFeatureExtractor, get_stopword_token_ids, compute_corpus_token_frequencies,
    get_high_frequency_tokens, compute_corpus_idf, load_idf_file
)
from utils import log_command, parse_args_with_config


def extract_with_batching(extractor, data, batch_size, max_doc_tokens, max_query_tokens,
                          input_format, qrels, relevance_threshold, max_docs, reverse_order=False,
                          desc="Processing"):
    """
    Extract features using batching.

    Args:
        extractor: Feature extractor instance
        data: List of samples
        batch_size: Number of queries per batch
        max_doc_tokens: Maximum tokens per document
        max_query_tokens: Maximum tokens per query
        input_format: Detected input format
        qrels: Optional qrels dict for labels
        relevance_threshold: Minimum qrels score for positive
        max_docs: Maximum documents per query
        reverse_order: If True, reverse document order (least relevant first)
        desc: Progress bar description

    Returns:
        Tuple of (features, labels, query_ids, doc_ids, docs_per_query)
    """
    all_features = []
    all_labels = []
    all_query_ids = []
    all_doc_ids = []
    docs_per_query = []

    # Prepare all queries and documents
    queries = []
    documents_list = []
    labels_list = []
    query_ids_list = []
    doc_ids_list = []

    for sample in data:
        query = sample.get('question', sample.get('query', ''))
        query_id = sample.get('idx', '')
        documents = sample.get('paragraphs', [])

        if max_docs is not None:
            documents = documents[:max_docs]

        if not documents:
            continue

        # Reverse document order if requested (least relevant first)
        if reverse_order:
            documents = documents[::-1]

        queries.append(query)
        documents_list.append(documents)
        query_ids_list.append(query_id)

        # Get document IDs (after potential reversal)
        doc_ids = [d.get('idx', f'doc_{i}') for i, d in enumerate(documents)]
        doc_ids_list.append(doc_ids)

        # Get labels based on input format
        if input_format == 'head_detection':
            # For head detection format, distinguish positives, hard negatives, and others:
            # 1 = is_positive=True (positive document)
            # 0 = is_negative=True (hard negative document)
            # -1 = neither (not used in CoRe scoring)
            def get_head_detection_label(d):
                if d.get('is_positive', False):
                    return 1
                elif d.get('is_negative', False):
                    return 0
                else:
                    return -1
            labels = np.array([get_head_detection_label(d) for d in documents], dtype=np.int32)
        elif qrels is not None:
            labels = np.array([
                get_label_from_qrels(query_id, doc_id, qrels, relevance_threshold)
                for doc_id in doc_ids
            ], dtype=np.int32)
        else:
            labels = np.full(len(documents), -1, dtype=np.int32)
        labels_list.append(labels)

    # Process in batches
    num_samples = len(queries)
    for i in tqdm(range(0, num_samples, batch_size), desc=desc):
        batch_queries = queries[i:i+batch_size]
        batch_docs = documents_list[i:i+batch_size]
        batch_labels = labels_list[i:i+batch_size]
        batch_query_ids = query_ids_list[i:i+batch_size]
        batch_doc_ids = doc_ids_list[i:i+batch_size]

        try:
            if batch_size > 1 and len(batch_queries) > 1:
                batch_features = extractor.extract_features_batch(
                    batch_queries, batch_docs,
                    max_doc_tokens=max_doc_tokens,
                    max_query_tokens=max_query_tokens
                )
            else:
                batch_features = [extractor.extract_features(
                    batch_queries[0], batch_docs[0],
                    max_doc_tokens=max_doc_tokens,
                    max_query_tokens=max_query_tokens
                )]

            for features, labels, q_id, d_ids in zip(batch_features, batch_labels, batch_query_ids, batch_doc_ids):
                all_features.append(features)
                all_labels.append(labels)
                all_query_ids.extend([q_id] * len(labels))
                all_doc_ids.extend(d_ids)
                docs_per_query.append(len(labels))

        except torch.cuda.OutOfMemoryError:
            print(f"Warning: OOM for batch {i}, falling back to sequential processing")
            torch.cuda.empty_cache()
            gc.collect()

            # Fall back to single sample processing
            for q, d, l, q_id, d_ids in zip(batch_queries, batch_docs, batch_labels, batch_query_ids, batch_doc_ids):
                try:
                    features = extractor.extract_features(
                        q, d,
                        max_doc_tokens=max_doc_tokens,
                        max_query_tokens=max_query_tokens
                    )
                    all_features.append(features)
                    all_labels.append(l)
                    all_query_ids.extend([q_id] * len(l))
                    all_doc_ids.extend(d_ids)
                    docs_per_query.append(len(l))
                except Exception as e2:
                    print(f"  Error processing single sample: {e2}")
                    continue

        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            # Fall back to single sample processing
            for q, d, l, q_id, d_ids in zip(batch_queries, batch_docs, batch_labels, batch_query_ids, batch_doc_ids):
                try:
                    features = extractor.extract_features(
                        q, d,
                        max_doc_tokens=max_doc_tokens,
                        max_query_tokens=max_query_tokens
                    )
                    all_features.append(features)
                    all_labels.append(l)
                    all_query_ids.extend([q_id] * len(l))
                    all_doc_ids.extend(d_ids)
                    docs_per_query.append(len(l))
                except Exception as e2:
                    print(f"  Error processing single sample: {e2}")
                    continue

        torch.cuda.empty_cache()
        gc.collect()

    if not all_features:
        return None, None, None, None, None

    return (np.vstack(all_features), np.concatenate(all_labels),
            np.array(all_query_ids, dtype=object), np.array(all_doc_ids, dtype=object),
            np.array(docs_per_query))


def gpu_worker(gpu_id, data_slice, args_dict, output_file, progress_queue=None):
    """
    Worker function that processes a slice of data on a specific GPU.

    Args:
        gpu_id: GPU device ID to use
        data_slice: List of samples to process
        args_dict: Dictionary of arguments
        output_file: Path to save results (temporary .npz file)
        progress_queue: Optional queue to report progress
    """
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    torch.cuda.set_device(0)  # After CUDA_VISIBLE_DEVICES, it becomes device 0

    # Import here to avoid issues with multiprocessing
    # Re-add paths for the subprocess
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
    sys.path.insert(0, str(Path(__file__).parent))

    import transformers
    from extract_head_features import (
        HFFeatureExtractor, detect_input_format, LLM_NAMES,
        load_qrels, get_label_from_qrels
    )
    from extract_features_idf import (
        IDFFeatureExtractor, get_stopword_token_ids, compute_corpus_token_frequencies,
        get_high_frequency_tokens, load_idf_file
    )

    print(f"[GPU {gpu_id}] Starting worker with {len(data_slice)} samples", flush=True)

    # Reconstruct args
    llm_name = LLM_NAMES[args_dict['llm']]
    calibrate = not args_dict['no_calibration']

    # Load tokenizer for filtering
    tokenizer = transformers.AutoTokenizer.from_pretrained(llm_name)

    # Prepare filtering if needed
    stopword_ids = None
    token_idf = None
    filter_mode = args_dict['filter_mode']

    if filter_mode == 'stopwords':
        stopword_ids = get_stopword_token_ids(tokenizer)
    elif filter_mode == 'high_freq':
        token_counts = compute_corpus_token_frequencies(data_slice, tokenizer)
        stopword_ids = get_high_frequency_tokens(token_counts, args_dict['high_freq_percentile'])
    elif filter_mode == 'idf':
        if args_dict['idf_file']:
            token_idf = load_idf_file(args_dict['idf_file'])
        else:
            default_idf = Path(__file__).parent.parent / 'head_data' / args_dict['llm'] / 'corpus_idf.json'
            if default_idf.exists():
                token_idf = load_idf_file(default_idf)
            else:
                print(f"[GPU {gpu_id}] Error: No IDF file found", flush=True)
                return

    # Initialize extractor
    if filter_mode == 'none':
        extractor = HFFeatureExtractor(
            llm_name,
            prune=args_dict['prune'],
            quantize=args_dict['quantize'],
            calibrate=calibrate
        )
    else:
        extractor = IDFFeatureExtractor(
            llm_name,
            prune=args_dict['prune'],
            quantize=args_dict['quantize'],
            calibrate=calibrate,
            filter_mode=filter_mode,
            stopword_ids=stopword_ids,
            token_idf=token_idf
        )

    print(f"[GPU {gpu_id}] Model loaded, extracting features...", flush=True)

    # Load qrels if provided
    qrels = None
    if args_dict['qrels'] is not None:
        qrels_path = Path(args_dict['qrels'])
        if qrels_path.exists():
            qrels = load_qrels(qrels_path)

    # Detect format
    input_format = detect_input_format(data_slice)

    # Extract features
    all_features, all_labels, all_query_ids, all_doc_ids, docs_per_query = extract_with_batching(
        extractor, data_slice, args_dict['batch_size'], args_dict['max_doc_tokens'],
        args_dict['max_query_tokens'], input_format, qrels, args_dict['relevance_threshold'],
        args_dict['max_docs'], reverse_order=args_dict['reverse_order'],
        desc=f"GPU {gpu_id}"
    )

    if all_features is None:
        print(f"[GPU {gpu_id}] No features extracted!", flush=True)
        # Save empty result
        np.savez_compressed(output_file,
                           features=np.array([]),
                           labels=np.array([]),
                           query_ids=np.array([]),
                           doc_ids=np.array([]),
                           docs_per_query=np.array([]))
        return

    print(f"[GPU {gpu_id}] Extracted {all_features.shape[0]} document features", flush=True)

    # Save results to temporary file
    np.savez_compressed(
        output_file,
        features=all_features,
        labels=all_labels,
        query_ids=all_query_ids,
        doc_ids=all_doc_ids,
        docs_per_query=docs_per_query
    )

    print(f"[GPU {gpu_id}] Saved results to {output_file}", flush=True)

    # Clean up
    del extractor
    torch.cuda.empty_cache()
    gc.collect()


def run_multi_gpu(data, gpu_ids, args_dict):
    """
    Run feature extraction in parallel across multiple GPUs.

    Args:
        data: Full dataset to process
        gpu_ids: List of GPU IDs to use
        args_dict: Dictionary of arguments

    Returns:
        Merged results from all GPUs
    """
    num_gpus = len(gpu_ids)
    num_samples = len(data)
    samples_per_gpu = (num_samples + num_gpus - 1) // num_gpus

    print(f"\nDistributing {num_samples} samples across {num_gpus} GPUs")
    print(f"Approximately {samples_per_gpu} samples per GPU")

    # Create temporary directory for intermediate results
    temp_dir = tempfile.mkdtemp(prefix='extract_features_')
    temp_files = []

    # Split data and create processes
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        start_idx = i * samples_per_gpu
        end_idx = min((i + 1) * samples_per_gpu, num_samples)

        if start_idx >= num_samples:
            break

        data_slice = data[start_idx:end_idx]
        temp_file = os.path.join(temp_dir, f'gpu_{gpu_id}_results.npz')
        temp_files.append(temp_file)

        print(f"GPU {gpu_id}: samples {start_idx} to {end_idx} ({len(data_slice)} samples)")

        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, data_slice, args_dict, temp_file)
        )
        processes.append(p)

    # Start all processes
    print("\nStarting workers...")
    for p in processes:
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("\nAll workers completed. Merging results...")

    # Merge results from all GPUs in order (preserves input order since we split sequentially)
    all_features = []
    all_labels = []
    all_query_ids = []
    all_doc_ids = []
    all_docs_per_query = []
    total_samples_processed = 0
    failed_gpus = []

    for i, temp_file in enumerate(temp_files):
        gpu_id = gpu_ids[i] if i < len(gpu_ids) else i
        if os.path.exists(temp_file):
            gpu_results = np.load(temp_file, allow_pickle=True)
            if len(gpu_results['features']) > 0:
                all_features.append(gpu_results['features'])
                all_labels.append(gpu_results['labels'])
                all_query_ids.extend(gpu_results['query_ids'])
                all_doc_ids.extend(gpu_results['doc_ids'])
                all_docs_per_query.extend(gpu_results['docs_per_query'])
                total_samples_processed += len(gpu_results['docs_per_query'])
                print(f"  GPU {gpu_id}: {len(gpu_results['docs_per_query'])} queries, {len(gpu_results['features'])} docs")
            else:
                failed_gpus.append(gpu_id)
                print(f"  GPU {gpu_id}: WARNING - no features extracted")
            # Clean up temp file
            os.remove(temp_file)
        else:
            failed_gpus.append(gpu_id)
            print(f"  GPU {gpu_id}: WARNING - results file not found")

    # Clean up temp directory
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass

    if failed_gpus:
        print(f"\nWARNING: GPUs {failed_gpus} failed or produced no results.")
        print("Output order may not match input order if middle GPUs failed!")

    print(f"\nTotal: {total_samples_processed} queries processed")

    if not all_features:
        return None, None, None, None, None

    return (
        np.vstack(all_features),
        np.concatenate(all_labels),
        np.array(all_query_ids, dtype=object),
        np.array(all_doc_ids, dtype=object),
        np.array(all_docs_per_query)
    )


def main():
    log_command()

    parser = argparse.ArgumentParser(
        description='Extract attention features with optional IDF/stopword filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard extraction (no filtering)
  python extract_features.py --llm mistral --input_file data.json

  # With stopword filtering
  python extract_features.py --llm mistral --input_file data.json --filter_mode stopwords

  # Compute corpus IDF
  python extract_features.py --llm mistral --compute_idf --corpus_dir /path/to/beir
        """
    )

    # Model and input/output
    parser.add_argument('--llm', type=str, default='mistral',
                        choices=['mistral', 'llama', 'phi', 'granite'],
                        help='LLM to use for feature extraction')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input JSON file (default: head_data/nq_core.json)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: head_data/{llm}/)')
    parser.add_argument('--output_name', '-o', type=str, default=None,
                        help='Output filename (without extension)')

    # Processing limits
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to process (default: all)')
    parser.add_argument('--max_docs', type=int, default=None,
                        help='Maximum documents per query (default: all)')
    parser.add_argument('--max_doc_tokens', type=int, default=300,
                        help='Maximum tokens per document (default: 300)')
    parser.add_argument('--max_query_tokens', type=int, default=None,
                        help='Maximum tokens per query (default: None)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for feature extraction (default: 1)')

    # Labels
    parser.add_argument('--qrels', type=str, default=None,
                        help='TREC qrels file for relevance labels')
    parser.add_argument('--relevance_threshold', type=int, default=1,
                        help='Minimum qrels score to be considered positive (default: 1)')

    # Model configuration
    parser.add_argument('--quantize', type=str, default=None, choices=[None, '4bit', '8bit'],
                        help='Quantization mode')
    parser.add_argument('--prune', type=float, default=0.0,
                        help='Layer pruning ratio (default: 0.0)')
    parser.add_argument('--no_calibration', action='store_true',
                        help='Disable calibration (subtracting N/A attention)')
    parser.add_argument('--reverse_order', action='store_true',
                        help='Reverse document order (least relevant first)')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU IDs to use (e.g., "0,1,2,3"). Default: single GPU (cuda:0)')

    # Filtering options
    parser.add_argument('--filter_mode', type=str, default='none',
                        choices=['none', 'stopwords', 'idf', 'high_freq'],
                        help='Token filtering mode (default: none)')
    parser.add_argument('--high_freq_percentile', type=float, default=95,
                        help='Percentile for high-frequency token filtering (default: 95)')
    parser.add_argument('--idf_file', type=str, default=None,
                        help='Pre-computed IDF file (for filter_mode=idf)')

    # IDF computation mode
    parser.add_argument('--compute_idf', action='store_true',
                        help='Compute corpus IDF and exit')
    parser.add_argument('--corpus_dir', type=str, default=None,
                        help='BEIR corpus dir or retriever_output dir (for --compute_idf)')
    parser.add_argument('--max_docs_idf', type=int, default=10000,
                        help='Max documents per dataset for IDF computation (default: 10000)')

    args = parse_args_with_config(parser)

    # Initialize tokenizer
    llm_name = LLM_NAMES[args.llm]
    print(f"Loading tokenizer: {llm_name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(llm_name)

    # Handle --compute_idf mode
    if args.compute_idf:
        if args.corpus_dir is None:
            print("Error: --corpus_dir required for --compute_idf")
            print("  Use BEIR dir (e.g., /path/to/beir)")
            print("  Or retriever_output dir")
            return

        corpus_dir = Path(args.corpus_dir)
        output_file = Path(__file__).parent.parent / 'head_data' / args.llm / 'corpus_idf.json'

        print(f"\nComputing corpus IDF from {corpus_dir}")
        compute_corpus_idf(
            corpus_dir,
            tokenizer,
            max_docs_per_dataset=args.max_docs_idf,
            output_file=output_file
        )
        return

    # Determine input file
    if args.input_file is None:
        input_file = Path(__file__).parent.parent / 'head_data' / 'nq_core.json'
    else:
        input_file = Path(args.input_file)

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    # Load data
    print(f"Loading data from {input_file}...")
    with open_file(input_file, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    # Detect input format
    input_format = detect_input_format(data)
    print(f"Detected format: {input_format}")

    # Load qrels if provided
    qrels = None
    if args.qrels is not None:
        qrels_path = Path(args.qrels)
        if qrels_path.exists():
            qrels = load_qrels(qrels_path)
            print(f"Loaded {len(qrels)} qrels from {qrels_path}")
        else:
            print(f"Warning: qrels file not found: {qrels_path}")

    # Limit samples
    if args.max_samples is not None:
        data = data[:args.max_samples]
        print(f"Limited to {len(data)} samples")

    # Prepare filtering if needed
    stopword_ids = None
    token_idf = None

    if args.filter_mode == 'stopwords':
        print("Computing stopword token IDs...")
        stopword_ids = get_stopword_token_ids(tokenizer)
        print(f"Found {len(stopword_ids)} stopword token IDs")

    elif args.filter_mode == 'high_freq':
        print("Computing corpus token frequencies...")
        token_counts = compute_corpus_token_frequencies(data, tokenizer)
        stopword_ids = get_high_frequency_tokens(token_counts, args.high_freq_percentile)
        print(f"Found {len(stopword_ids)} high-frequency token IDs (top {100-args.high_freq_percentile}%)")

    elif args.filter_mode == 'idf':
        if args.idf_file:
            token_idf = load_idf_file(args.idf_file)
        else:
            default_idf = Path(__file__).parent.parent / 'head_data' / args.llm / 'corpus_idf.json'
            if default_idf.exists():
                token_idf = load_idf_file(default_idf)
            else:
                print(f"No IDF file found. Run with --compute_idf first, or provide --idf_file")
                print(f"Expected: {default_idf}")
                return

    calibrate = not args.no_calibration
    print(f"Filter mode: {args.filter_mode}")
    print(f"Calibration: {calibrate}")

    # Check for multi-GPU mode
    if args.gpus is not None:
        # Parse GPU IDs
        gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]
        print(f"\nMulti-GPU mode: using GPUs {gpu_ids}")

        # Prepare args dict for workers
        args_dict = {
            'llm': args.llm,
            'prune': args.prune,
            'quantize': args.quantize,
            'no_calibration': args.no_calibration,
            'filter_mode': args.filter_mode,
            'high_freq_percentile': args.high_freq_percentile,
            'idf_file': args.idf_file,
            'batch_size': args.batch_size,
            'max_doc_tokens': args.max_doc_tokens,
            'max_query_tokens': args.max_query_tokens,
            'max_docs': args.max_docs,
            'reverse_order': args.reverse_order,
            'qrels': str(args.qrels) if args.qrels else None,
            'relevance_threshold': args.relevance_threshold,
        }

        # Set multiprocessing start method
        mp.set_start_method('spawn', force=True)

        # Run multi-GPU extraction
        all_features, all_labels, all_query_ids, all_doc_ids, docs_per_query = run_multi_gpu(
            data, gpu_ids, args_dict
        )
    else:
        # Single GPU mode
        print(f"\nInitializing feature extractor...")

        if args.filter_mode == 'none':
            # Use standard HFFeatureExtractor
            extractor = HFFeatureExtractor(
                llm_name,
                prune=args.prune,
                quantize=args.quantize,
                calibrate=calibrate
            )
        else:
            # Use IDFFeatureExtractor with filtering
            extractor = IDFFeatureExtractor(
                llm_name,
                prune=args.prune,
                quantize=args.quantize,
                calibrate=calibrate,
                filter_mode=args.filter_mode,
                stopword_ids=stopword_ids,
                token_idf=token_idf
            )

        # Extract features
        print(f"\nExtracting features...")
        if args.reverse_order:
            print("Document order: REVERSED (least relevant first)")
        all_features, all_labels, all_query_ids, all_doc_ids, docs_per_query = extract_with_batching(
            extractor, data, args.batch_size, args.max_doc_tokens, args.max_query_tokens,
            input_format, qrels, args.relevance_threshold, args.max_docs,
            reverse_order=args.reverse_order, desc="Processing"
        )

    if all_features is None:
        print("No features extracted!")
        return

    print(f"\nExtracted features: {all_features.shape}")
    print(f"Labels: {all_labels.shape}")

    # Count label statistics
    n_positive = int((all_labels == 1).sum())
    n_negative = int((all_labels == 0).sum())
    n_unknown = int((all_labels == -1).sum())
    print(f"Label distribution: {n_positive} positive, {n_negative} negative, {n_unknown} unknown")

    # Print truncation statistics (single-GPU mode only)
    truncation_stats = None
    if args.gpus is None and hasattr(extractor, 'get_truncation_stats'):
        truncation_stats = extractor.get_truncation_stats()
        print(f"\nTruncation statistics (max_doc_tokens={args.max_doc_tokens}, max_query_tokens={args.max_query_tokens}):")
        print(f"  Documents truncated: {truncation_stats['docs_truncated']}/{truncation_stats['docs_total']} ({truncation_stats['docs_truncated_pct']:.1f}%)")
        print(f"  Queries truncated: {truncation_stats['queries_truncated']}/{truncation_stats['queries_total']} ({truncation_stats['queries_truncated_pct']:.1f}%)")

    # Determine output directory and name
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / 'head_data' / args.llm

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_name is not None:
        output_name = args.output_name
    else:
        # Derive from input file name
        input_stem = input_file.stem
        if input_stem.endswith('.json'):
            input_stem = input_stem[:-5]
        n_samples = len(docs_per_query)
        quant_suffix = f'_{args.quantize}' if args.quantize else ''
        filter_suffix = f'_{args.filter_mode}' if args.filter_mode != 'none' else ''
        reverse_suffix = '_reversed' if args.reverse_order else ''
        output_name = f'attention_features_{input_stem}_n{n_samples}{filter_suffix}{reverse_suffix}{quant_suffix}'

    if output_name.endswith('.npz'):
        output_name = output_name[:-4]

    output_file = output_dir / f'{output_name}.npz'

    # Save features
    np.savez_compressed(
        output_file,
        features=all_features,
        labels=all_labels,
        query_ids=all_query_ids,
        doc_ids=all_doc_ids,
        docs_per_query=docs_per_query
    )
    print(f"\nSaved features to {output_file}")

    # Save metadata
    metadata_file = output_dir / f'{output_name}.meta.json'
    has_labels = bool(n_positive > 0 or n_negative > 0)
    # Get num_layers and num_heads (may not have extractor in multi-GPU mode)
    if args.gpus is None:
        num_layers = extractor.num_layer
        num_heads = extractor.num_head
    else:
        # In multi-GPU mode, infer from feature shape
        # Features have shape (n_docs, num_layers * num_heads)
        # For supported models: mistral/llama/granite have 32 layers, 32 heads
        # phi has 40 layers, 40 heads
        num_features = all_features.shape[1]
        if args.llm == 'phi':
            num_layers = 40
            num_heads = 40
        else:
            num_layers = 32
            num_heads = 32

    gpu_ids = [int(g.strip()) for g in args.gpus.split(',')] if args.gpus else None

    metadata = {
        'input_file': str(input_file),
        'input_format': input_format,
        'llm': args.llm,
        'num_queries': len(docs_per_query),
        'num_documents': len(all_labels),
        'num_features': all_features.shape[1],
        'num_layers': num_layers,
        'num_heads': num_heads,
        'max_doc_tokens': args.max_doc_tokens,
        'max_query_tokens': args.max_query_tokens,
        'max_docs': args.max_docs,
        'batch_size': args.batch_size,
        'prune': args.prune,
        'calibrate': calibrate,
        'quantize': args.quantize,
        'filter_mode': args.filter_mode,
        'reverse_order': args.reverse_order,
        'gpus': gpu_ids,
        'has_labels': has_labels,
        'qrels_file': str(args.qrels) if args.qrels else None,
        'relevance_threshold': args.relevance_threshold if args.qrels else None,
        'label_stats': {
            'positive': n_positive,
            'negative': n_negative,
            'unknown': n_unknown
        },
        'truncation_stats': truncation_stats,  # None in multi-GPU mode
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file}")


if __name__ == '__main__':
    main()
