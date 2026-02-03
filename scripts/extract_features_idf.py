#!/usr/bin/env python3
"""
Extract attention features with IDF-weighted or stopword-filtered aggregation.

This script tests whether filtering high-frequency tokens (stopwords) from
the attention aggregation improves the discriminative power of attention features.

Approach:
1. Compute token frequencies across the corpus
2. Filter out high-frequency tokens (stopwords) when computing attention scores
3. Compare with baseline (sum all tokens) features
"""

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
from extract_head_features import HFFeatureExtractor, open_file, detect_input_format, LLM_NAMES
from utils import log_command

# Common English stopwords
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
    'she', 'we', 'they', 'what', 'which', 'who', 'whom', 'when', 'where',
    'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 's', 't', 'just', 'don', 'now', 'also',
    '.', ',', '!', '?', ';', ':', '-', '--', '(', ')', '[', ']', '{', '}',
    '"', "'", '`', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '+',
    '=', '<', '>', '~', '\n', '\t', ' '
}


def get_stopword_token_ids(tokenizer, extra_stopwords=None):
    """
    Get token IDs that correspond to stopwords.

    Args:
        tokenizer: HuggingFace tokenizer
        extra_stopwords: Additional stopwords to include

    Returns:
        set of token IDs
    """
    stopwords = STOPWORDS.copy()
    if extra_stopwords:
        stopwords.update(extra_stopwords)

    stopword_ids = set()

    # Method 1: Direct encoding of stopwords
    for word in stopwords:
        # Try different variations
        for variant in [word, f' {word}', f'{word} ', f' {word} ', word.capitalize(), word.upper()]:
            try:
                ids = tokenizer.encode(variant, add_special_tokens=False)
                stopword_ids.update(ids)
            except:
                pass

    # Method 2: Check vocabulary directly for subword tokens
    vocab = tokenizer.get_vocab()
    for token, token_id in vocab.items():
        # Clean token (remove special prefixes like 'Ġ' in GPT-style tokenizers)
        clean_token = token.replace('Ġ', ' ').replace('▁', ' ').strip().lower()
        if clean_token in stopwords or len(clean_token) <= 1:
            stopword_ids.add(token_id)

    return stopword_ids


def compute_corpus_token_frequencies(data, tokenizer, max_samples=1000):
    """
    Compute token frequencies across the corpus.

    Args:
        data: List of samples with 'paragraphs'
        tokenizer: HuggingFace tokenizer
        max_samples: Maximum samples to process

    Returns:
        Counter of token frequencies
    """
    token_counts = Counter()

    for sample in tqdm(data[:max_samples], desc="Computing token frequencies"):
        for doc in sample.get('paragraphs', []):
            text = doc.get('paragraph_text', '')
            if text:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                token_counts.update(tokens)

    return token_counts


def get_high_frequency_tokens(token_counts, percentile=95):
    """
    Get tokens above a frequency percentile.

    Args:
        token_counts: Counter of token frequencies
        percentile: Percentile threshold (e.g., 95 means top 5% most frequent)

    Returns:
        set of high-frequency token IDs
    """
    if not token_counts:
        return set()

    counts = list(token_counts.values())
    threshold = np.percentile(counts, percentile)

    high_freq_ids = {token_id for token_id, count in token_counts.items() if count >= threshold}
    return high_freq_ids


def compute_corpus_idf(corpus_dir, tokenizer, datasets=None, max_docs_per_dataset=10000, output_file=None):
    """
    Compute corpus-level IDF from BEIR corpus files or retriever output files.

    Args:
        corpus_dir: Path to BEIR directory (with subdirs like nq/, hotpotqa/)
                    or retriever_output directory
        tokenizer: HuggingFace tokenizer
        datasets: List of dataset names (default: all BEIR datasets)
        max_docs_per_dataset: Maximum documents to process per dataset
        output_file: Path to save IDF file (optional)

    Returns:
        dict mapping token_id -> IDF score
    """
    corpus_dir = Path(corpus_dir)

    if datasets is None:
        # Default BEIR datasets
        datasets = [
            'nq', 'hotpotqa', 'fiqa', 'scifact', 'scidocs', 'nfcorpus',
            'trec-covid', 'dbpedia-entity', 'fever', 'climate-fever',
            'msmarco', 'quora', 'arguana', 'webis-touche2020'
        ]

    # Document frequency: how many documents contain each token
    doc_freq = Counter()
    total_docs = 0

    for dataset in tqdm(datasets, desc="Processing datasets"):
        # Try BEIR corpus format first (corpus_dir/dataset/corpus.jsonl.bz2)
        corpus_file = None
        is_beir_format = False

        for fname in ['corpus.jsonl.bz2', 'corpus.jsonl', 'corpus.json.bz2', 'corpus.json']:
            candidate = corpus_dir / dataset / fname
            if candidate.exists():
                corpus_file = candidate
                is_beir_format = True
                break

        # Fall back to retriever output format (corpus_dir/dataset.json.bz2)
        if corpus_file is None:
            for fname in [f'{dataset}.json.bz2', f'{dataset}.json']:
                candidate = corpus_dir / fname
                if candidate.exists():
                    corpus_file = candidate
                    is_beir_format = False
                    break

        if corpus_file is None:
            print(f"  Skipping {dataset} - file not found")
            continue

        docs_processed = 0
        try:
            with open_file(corpus_file, 'r') as f:
                if is_beir_format and 'jsonl' in corpus_file.name:
                    # BEIR JSONL format: one doc per line with _id, title, text
                    for line in f:
                        if docs_processed >= max_docs_per_dataset:
                            break
                        try:
                            doc = json.loads(line.strip())
                        except json.JSONDecodeError:
                            continue

                        text = doc.get('text', '')
                        title = doc.get('title', '')
                        full_text = f"{title} {text}".strip()

                        if not full_text:
                            continue

                        tokens = tokenizer.encode(full_text[:2000], add_special_tokens=False)
                        unique_tokens = set(tokens)

                        for token_id in unique_tokens:
                            doc_freq[token_id] += 1

                        total_docs += 1
                        docs_processed += 1
                else:
                    # Retriever output or JSON format
                    data = json.load(f)
                    seen_docs = set()

                    for sample in data:
                        if docs_processed >= max_docs_per_dataset:
                            break

                        for doc in sample.get('paragraphs', []):
                            if docs_processed >= max_docs_per_dataset:
                                break

                            text = doc.get('paragraph_text', '')
                            if not text:
                                continue

                            doc_hash = hash(text[:500])
                            if doc_hash in seen_docs:
                                continue
                            seen_docs.add(doc_hash)

                            tokens = tokenizer.encode(text[:2000], add_special_tokens=False)
                            unique_tokens = set(tokens)

                            for token_id in unique_tokens:
                                doc_freq[token_id] += 1

                            total_docs += 1
                            docs_processed += 1

        except Exception as e:
            print(f"  Error processing {dataset}: {e}")
            continue

        tqdm.write(f"  {dataset}: {docs_processed} documents")

    print(f"\nTotal documents: {total_docs}")
    print(f"Unique tokens: {len(doc_freq)}")

    # Compute IDF: log(N / df)
    # Use smoothed IDF: log((N + 1) / (df + 1)) + 1
    token_idf = {}
    for token_id, df in doc_freq.items():
        token_idf[token_id] = np.log((total_docs + 1) / (df + 1)) + 1

    # Normalize to [0, 1] range
    if token_idf:
        min_idf = min(token_idf.values())
        max_idf = max(token_idf.values())
        idf_range = max_idf - min_idf if max_idf > min_idf else 1
        token_idf = {tid: (idf - min_idf) / idf_range for tid, idf in token_idf.items()}

    # Save if output file specified
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        idf_data = {
            'total_docs': total_docs,
            'num_tokens': len(token_idf),
            'idf': {str(k): v for k, v in token_idf.items()}
        }

        with open(output_file, 'w') as f:
            json.dump(idf_data, f)
        print(f"\nSaved IDF to {output_file}")

    return token_idf


def load_idf_file(idf_file):
    """
    Load pre-computed IDF from file.

    Args:
        idf_file: Path to IDF JSON file

    Returns:
        dict mapping token_id (int) -> IDF score (float)
    """
    with open(idf_file, 'r') as f:
        data = json.load(f)

    # Convert string keys back to integers
    token_idf = {int(k): v for k, v in data['idf'].items()}
    print(f"Loaded IDF for {len(token_idf)} tokens from {idf_file}")
    return token_idf


class IDFFeatureExtractor(HFFeatureExtractor):
    """Feature extractor with IDF-weighted or stopword-filtered aggregation."""

    def __init__(self, llm_name, prune=0.0, quantize=None, calibrate=True,
                 filter_mode='none', stopword_ids=None, token_idf=None):
        """
        Initialize the IDF feature extractor.

        Args:
            llm_name: HuggingFace model name
            prune: Layer pruning ratio
            quantize: Quantization mode
            calibrate: Whether to subtract N/A attention
            filter_mode: 'none', 'stopwords', 'idf', or 'high_freq'
            stopword_ids: Set of stopword token IDs (for stopwords mode)
            token_idf: Dict of token_id -> IDF weight (for idf mode)
        """
        super().__init__(llm_name, prune=prune, quantize=quantize, calibrate=calibrate)

        self.filter_mode = filter_mode
        self.stopword_ids = stopword_ids or set()
        self.token_idf = token_idf or {}

        print(f"Filter mode: {filter_mode}")
        if filter_mode == 'stopwords':
            print(f"Stopword token IDs: {len(self.stopword_ids)}")

    def _extract_raw_features_filtered(self, query, truncated_docs, doc_spans,
                                       input_ids, kv_cache=None, context_start_idx=0):
        """
        Extract features with token filtering applied.

        This is a modified version of _extract_raw_features that applies
        stopword/IDF filtering when aggregating attention over document tokens.
        """
        prompt, _, query_span = self.prepare_input(query, truncated_docs)

        # Get attention weights
        tokenized_input = self.tokenizer(prompt, return_tensors='pt').to(self.llm.device)
        _input_ids = tokenized_input.input_ids[:, context_start_idx:]
        full_input_ids = tokenized_input.input_ids[0]  # For token filtering
        _query_indices = list(range(query_span[0] - context_start_idx, query_span[1] - context_start_idx + 1))

        if kv_cache is None:
            kv_cache = self.DynamicCacheWithQuery(query_indices=_query_indices)
        else:
            kv_cache.query_cache = []
            kv_cache._query_indices = _query_indices

        with torch.no_grad():
            output = self.llm(
                input_ids=_input_ids,
                use_cache=True,
                past_key_values=kv_cache,
                output_attentions=True
            )
        kv_cache = output.past_key_values

        # Collect key and query caches
        all_key_cache = []
        all_query_cache = []
        for i in range(self.num_layer):
            all_key_cache.append(kv_cache.key_cache[i][:, :, :query_span[1] + 1])
            all_query_cache.append(kv_cache.query_cache[i])
        all_key_cache = torch.stack(all_key_cache)
        all_query_cache = torch.stack(all_query_cache)

        del tokenized_input, output
        torch.cuda.empty_cache()

        # Compute attention weights
        attn_weights = self._get_attn_weights(all_key_cache, all_query_cache).to('cuda').squeeze(1)
        del all_key_cache, all_query_cache
        torch.cuda.empty_cache()

        # Average over query tokens
        attn_weights = attn_weights.mean(-2)  # (num_layer, num_head, seq_len)

        # Extract document-level scores with filtering
        num_docs = len(doc_spans)
        features = np.zeros((num_docs, self.num_layer * self.num_head), dtype=np.float32)

        for doc_idx, (start, end) in enumerate(doc_spans):
            # Get token IDs for this document span
            doc_token_ids = full_input_ids[start:end].cpu().numpy()

            if self.filter_mode == 'stopwords':
                # Create mask excluding stopwords
                mask = torch.tensor(
                    [1.0 if tid not in self.stopword_ids else 0.0 for tid in doc_token_ids],
                    device=attn_weights.device
                )
                # Apply mask and sum
                doc_attn = attn_weights[:, :, start:end]  # (num_layer, num_head, doc_len)
                masked_attn = doc_attn * mask.unsqueeze(0).unsqueeze(0)
                doc_scores = masked_attn.sum(-1)  # (num_layer, num_head)

            elif self.filter_mode == 'idf':
                # Weight by IDF
                weights = torch.tensor(
                    [self.token_idf.get(tid, 1.0) for tid in doc_token_ids],
                    device=attn_weights.device
                )
                doc_attn = attn_weights[:, :, start:end]
                weighted_attn = doc_attn * weights.unsqueeze(0).unsqueeze(0)
                doc_scores = weighted_attn.sum(-1)

            else:
                # No filtering - standard sum
                doc_scores = attn_weights[:, :, start:end].sum(-1)

            features[doc_idx] = doc_scores.cpu().numpy().flatten()

        del attn_weights
        torch.cuda.empty_cache()

        return features, kv_cache, full_input_ids

    def extract_features(self, query, documents, max_doc_tokens=300, max_query_tokens=None):
        """Extract features with optional token filtering."""

        if self.filter_mode == 'none':
            # Use parent class method for baseline
            return super().extract_features(query, documents, max_doc_tokens, max_query_tokens)

        # Truncate query if needed
        if max_query_tokens is not None:
            query_words = query.split()
            if len(query_words) > max_query_tokens:
                query = ' '.join(query_words[:max_query_tokens])

        # Truncate documents
        truncated_docs = []
        for doc in documents:
            text = doc.get('paragraph_text', '')
            words = text.split()[:max_doc_tokens] if isinstance(text, str) else []
            truncated_docs.append({'paragraph_text': ' '.join(words)})

        # Get document spans
        _, doc_spans, query_span = self.prepare_input(query, truncated_docs)

        # Extract features with filtering
        features, kv_cache, input_ids = self._extract_raw_features_filtered(
            query, truncated_docs, doc_spans, input_ids=None
        )

        # Calibration if enabled
        if self.calibrate:
            query_start_idx = query_span[0]
            for i in range(len(kv_cache.key_cache)):
                kv_cache.key_cache[i] = kv_cache.key_cache[i][:, :, :query_start_idx, :]
                kv_cache.value_cache[i] = kv_cache.value_cache[i][:, :, :query_start_idx, :]
            kv_cache._seen_tokens = query_start_idx

            features_na, _, _ = self._extract_raw_features_filtered(
                'N/A', truncated_docs, doc_spans, input_ids,
                kv_cache=kv_cache, context_start_idx=query_start_idx
            )
            features = features - features_na

            del kv_cache
            torch.cuda.empty_cache()
        else:
            del kv_cache
            torch.cuda.empty_cache()

        return features


def evaluate_features(features, labels, docs_per_query, weights, top_k=8):
    """
    Quick evaluation of feature quality using NDCG@10.

    Args:
        features: (n_docs, n_heads) features
        labels: (n_docs,) binary labels
        docs_per_query: (n_queries,) docs per query
        weights: (n_heads,) head weights
        top_k: Number of top heads to use

    Returns:
        dict with evaluation metrics
    """
    from sklearn.metrics import roc_auc_score

    # Select top-k heads
    top_indices = np.argsort(np.abs(weights))[-top_k:]

    # Compute scores
    scores = features[:, top_indices] @ weights[top_indices]

    # Compute metrics per query
    ndcg_scores = []
    doc_offset = 0

    for n_docs in docs_per_query:
        q_scores = scores[doc_offset:doc_offset + n_docs]
        q_labels = labels[doc_offset:doc_offset + n_docs]
        doc_offset += n_docs

        if q_labels.sum() == 0:
            continue

        # Compute NDCG@10
        ranking = np.argsort(-q_scores)
        ranked_labels = q_labels[ranking][:10]

        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ranked_labels))
        ideal_labels = np.sort(q_labels)[::-1][:10]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_labels))

        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)

    # Compute AUC
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0

    return {
        'ndcg@10': np.mean(ndcg_scores) if ndcg_scores else 0,
        'auc': auc,
        'n_queries': len(ndcg_scores)
    }


def main():
    log_command()

    parser = argparse.ArgumentParser(description='Extract features with IDF/stopword filtering')
    parser.add_argument('--llm', type=str, default='mistral',
                        choices=['mistral', 'llama', 'phi', 'granite'])
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input JSON file (default: head_data/nq_core.json)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum samples to process (default: 100)')
    parser.add_argument('--max_doc_tokens', type=int, default=300,
                        help='Maximum tokens per document')
    parser.add_argument('--filter_mode', type=str, default='stopwords',
                        choices=['none', 'stopwords', 'idf', 'high_freq'],
                        help='Token filtering mode')
    parser.add_argument('--high_freq_percentile', type=float, default=95,
                        help='Percentile for high-frequency token filtering')
    parser.add_argument('--weight_file', type=str, default=None,
                        help='Head weight file for evaluation (optional)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare filtered vs baseline features')
    parser.add_argument('--quantize', type=str, default=None, choices=[None, '4bit', '8bit'])
    parser.add_argument('--compute_idf', action='store_true',
                        help='Compute corpus IDF from retriever outputs and exit')
    parser.add_argument('--idf_file', type=str, default=None,
                        help='Pre-computed IDF file to load (for filter_mode=idf)')
    parser.add_argument('--corpus_dir', type=str, default=None,
                        help='BEIR corpus dir (e.g., /path/to/beir) or retriever_output dir (for --compute_idf)')
    parser.add_argument('--max_docs_idf', type=int, default=5000,
                        help='Max documents per dataset for IDF computation')
    args = parser.parse_args()

    # Initialize tokenizer
    llm_name = LLM_NAMES[args.llm]
    print(f"Loading tokenizer: {llm_name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(llm_name)

    # Handle --compute_idf mode: compute corpus IDF and exit
    if args.compute_idf:
        if args.corpus_dir is None:
            print("Error: --corpus_dir required for --compute_idf")
            print("  Use BEIR dir (e.g., /local3/raduf/sandbox2/beir)")
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

    if args.max_samples:
        data = data[:args.max_samples]
        print(f"Limited to {len(data)} samples")

    # Get stopword token IDs or IDF weights
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
            # Load pre-computed IDF
            token_idf = load_idf_file(args.idf_file)
        else:
            # Try default location
            default_idf = Path(__file__).parent.parent / 'head_data' / args.llm / 'corpus_idf.json'
            if default_idf.exists():
                token_idf = load_idf_file(default_idf)
            else:
                print(f"No IDF file found. Run with --compute_idf first, or provide --idf_file")
                print(f"Expected: {default_idf}")
                return

    # Initialize extractor (start with 'none' filter mode if comparing)
    initial_filter_mode = 'none' if args.compare else args.filter_mode
    print(f"\nInitializing feature extractor...")
    extractor = IDFFeatureExtractor(
        llm_name,
        quantize=args.quantize,
        calibrate=True,
        filter_mode=initial_filter_mode,
        stopword_ids=stopword_ids,
        token_idf=token_idf
    )

    # Extract baseline features first if comparing
    baseline_features = None
    if args.compare:
        print(f"\nExtracting baseline features (no filtering)...")
        baseline_features = []
        all_labels = []
        docs_per_query = []

        for sample in tqdm(data, desc="Processing samples (baseline)"):
            query = sample.get('question', sample.get('query', ''))
            documents = sample.get('paragraphs', [])

            if not documents:
                continue

            try:
                features = extractor.extract_features(query, documents, max_doc_tokens=args.max_doc_tokens)
                labels = np.array([1 if d.get('is_positive', False) else 0 for d in documents], dtype=np.int32)

                baseline_features.append(features)
                all_labels.append(labels)
                docs_per_query.append(len(documents))
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue

            torch.cuda.empty_cache()

        baseline_features = np.vstack(baseline_features)
        all_labels = np.concatenate(all_labels)
        docs_per_query = np.array(docs_per_query)

        print(f"Baseline features: {baseline_features.shape}")

        # Now switch to filtered mode (reuse same model)
        print(f"\nSwitching to filter_mode={args.filter_mode}...")
        extractor.filter_mode = args.filter_mode
        extractor.stopword_ids = stopword_ids or set()
        extractor.token_idf = token_idf or {}

    # Extract features (filtered if compare, or whatever mode was requested)
    print(f"\nExtracting features with filter_mode={extractor.filter_mode}...")
    all_features = []
    if not args.compare:
        all_labels = []
        docs_per_query = []

    for sample in tqdm(data, desc="Processing samples"):
        query = sample.get('question', sample.get('query', ''))
        documents = sample.get('paragraphs', [])

        if not documents:
            continue

        try:
            features = extractor.extract_features(query, documents, max_doc_tokens=args.max_doc_tokens)
            all_features.append(features)

            if not args.compare:
                labels = np.array([1 if d.get('is_positive', False) else 0 for d in documents], dtype=np.int32)
                all_labels.append(labels)
                docs_per_query.append(len(documents))
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue

        torch.cuda.empty_cache()

    if not all_features:
        print("No features extracted!")
        return

    all_features = np.vstack(all_features)
    if not args.compare:
        all_labels = np.concatenate(all_labels)
        docs_per_query = np.array(docs_per_query)

    print(f"\nExtracted features: {all_features.shape}")
    print(f"Labels: {all_labels.shape}, positives: {all_labels.sum()}")

    # Compare with baseline if requested
    if args.compare and baseline_features is not None:
        print("\n" + "="*60)
        print("Feature Comparison")
        print("="*60)
        print(f"{'Metric':<30} {'Baseline':>15} {'Filtered':>15}")
        print("-"*60)
        print(f"{'Mean':<30} {baseline_features.mean():>15.4f} {all_features.mean():>15.4f}")
        print(f"{'Std':<30} {baseline_features.std():>15.4f} {all_features.std():>15.4f}")
        print(f"{'Min':<30} {baseline_features.min():>15.4f} {all_features.min():>15.4f}")
        print(f"{'Max':<30} {baseline_features.max():>15.4f} {all_features.max():>15.4f}")

        # Correlation between baseline and filtered
        correlations = []
        for i in range(all_features.shape[1]):
            corr = np.corrcoef(baseline_features[:, i], all_features[:, i])[0, 1]
            correlations.append(corr)
        print(f"{'Avg correlation with baseline':<30} {'-':>15} {np.mean(correlations):>15.4f}")

        # Evaluate if weight file provided
        if args.weight_file:
            print("\n" + "="*60)
            print("Evaluation with Head Weights")
            print("="*60)

            with open(args.weight_file) as f:
                weight_data = json.load(f)

            # Load weights
            n_heads = extractor.num_layer * extractor.num_head
            weights = np.zeros(n_heads)

            if 'all_weights' in weight_data:
                for key, w in weight_data['all_weights'].items():
                    layer, head = map(int, key.split('-'))
                    idx = layer * extractor.num_head + head
                    if idx < n_heads:
                        weights[idx] = w
            else:
                for key, scores in weight_data.items():
                    layer, head = map(int, key.split('-'))
                    idx = layer * extractor.num_head + head
                    if idx < n_heads:
                        weights[idx] = np.mean(scores) if isinstance(scores, list) else scores

            # Evaluate both
            baseline_metrics = evaluate_features(baseline_features, all_labels, docs_per_query, weights)
            filtered_metrics = evaluate_features(all_features, all_labels, docs_per_query, weights)

            print(f"{'Metric':<30} {'Baseline':>15} {'Filtered':>15} {'Diff':>15}")
            print("-"*60)
            for metric in ['ndcg@10', 'auc']:
                b = baseline_metrics[metric]
                f = filtered_metrics[metric]
                d = f - b
                print(f"{metric:<30} {b:>15.4f} {f:>15.4f} {d:>+15.4f}")

    # Save features
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / 'head_data' / args.llm

    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = f'attention_features_{args.filter_mode}_n{len(docs_per_query)}'
    output_file = output_dir / f'{output_name}.npz'

    np.savez_compressed(
        output_file,
        features=all_features,
        labels=all_labels,
        docs_per_query=docs_per_query
    )
    print(f"\nSaved features to {output_file}")


if __name__ == '__main__':
    main()
