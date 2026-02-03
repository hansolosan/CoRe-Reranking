#!/usr/bin/env python3
"""
Unified feature extraction script with optional IDF/stopword filtering.

This script extracts attention features from LLM heads for document reranking.
When --filter_mode is 'none' (default), it behaves like extract_head_features.py.
When --filter_mode is 'stopwords', 'idf', or 'high_freq', it applies token filtering.

Usage:
    # Standard extraction (no filtering)
    python extract_features.py --llm mistral --input_file data.json

    # With stopword filtering
    python extract_features.py --llm mistral --input_file data.json --filter_mode stopwords

    # With IDF weighting (requires pre-computed IDF file)
    python extract_features.py --llm mistral --input_file data.json --filter_mode idf

    # Compute corpus IDF first
    python extract_features.py --llm mistral --compute_idf --corpus_dir /path/to/beir
"""

import gc
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter

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
                          input_format, qrels, relevance_threshold, max_docs, desc="Processing"):
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

        queries.append(query)
        documents_list.append(documents)
        query_ids_list.append(query_id)

        # Get document IDs
        doc_ids = [d.get('idx', f'doc_{i}') for i, d in enumerate(documents)]
        doc_ids_list.append(doc_ids)

        # Get labels based on input format
        if input_format == 'head_detection':
            labels = np.array([1 if d.get('is_positive', False) else 0 for d in documents], dtype=np.int32)
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

    # Initialize extractor
    print(f"\nInitializing feature extractor...")
    calibrate = not args.no_calibration

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

    print(f"Filter mode: {args.filter_mode}")
    print(f"Calibration: {calibrate}")

    # Extract features
    print(f"\nExtracting features...")
    all_features, all_labels, all_query_ids, all_doc_ids, docs_per_query = extract_with_batching(
        extractor, data, args.batch_size, args.max_doc_tokens, args.max_query_tokens,
        input_format, qrels, args.relevance_threshold, args.max_docs, desc="Processing"
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
        output_name = f'attention_features_{input_stem}_n{n_samples}{filter_suffix}{quant_suffix}'

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
    metadata = {
        'input_file': str(input_file),
        'input_format': input_format,
        'llm': args.llm,
        'num_queries': len(docs_per_query),
        'num_documents': len(all_labels),
        'num_features': all_features.shape[1],
        'num_layers': extractor.num_layer,
        'num_heads': extractor.num_head,
        'max_doc_tokens': args.max_doc_tokens,
        'max_query_tokens': args.max_query_tokens,
        'max_docs': args.max_docs,
        'batch_size': args.batch_size,
        'prune': args.prune,
        'calibrate': calibrate,
        'quantize': args.quantize,
        'filter_mode': args.filter_mode,
        'has_labels': has_labels,
        'qrels_file': str(args.qrels) if args.qrels else None,
        'relevance_threshold': args.relevance_threshold if args.qrels else None,
        'label_stats': {
            'positive': n_positive,
            'negative': n_negative,
            'unknown': n_unknown
        },
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file}")


if __name__ == '__main__':
    main()
