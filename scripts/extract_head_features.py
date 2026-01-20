#!/usr/bin/env python3
"""
Extract attention features from all heads for BCE optimization or evaluation.
This script runs the model on detection data or retriever output and saves per-head attention scores.

Supports two input formats:
1. Head detection data (nq_core.json): has 'question', 'paragraphs' with 'is_positive'/'is_negative'
2. Retriever output (retriever_output/*.json): has 'idx', 'question', 'paragraphs' with 'idx', 'paragraph_text'
"""

import json
import os
import argparse
import gzip
import bz2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from src.custom.custom_cache import DynamicCacheWithQuery
import transformers
from transformers import BitsAndBytesConfig

LLM_NAMES = {
    'granite': 'ibm-granite/granite-3.2-8b-instruct',
    'llama': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'phi': 'microsoft/phi-4',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.2'
}


class FeatureExtractor:
    """Extract attention features from all heads for optimization."""

    def __init__(self, llm_name, prune=0.0, quantize=None):
        """
        Initialize the feature extractor.

        Args:
            llm_name: HuggingFace model name
            prune: Layer pruning ratio (0.0 = no pruning)
            quantize: Quantization mode - None, '4bit', or '8bit'
        """
        print(f"Loading model: {llm_name}...", flush=True)
        if quantize:
            print(f"Using {quantize} quantization", flush=True)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(llm_name)
        config = transformers.AutoConfig.from_pretrained(llm_name)
        config.num_hidden_layers = int(config.num_hidden_layers * (1 - prune))

        # Ensure head_dim is set (some models don't have it explicitly)
        if not hasattr(config, 'head_dim') or config.head_dim is None:
            config.head_dim = config.hidden_size // config.num_attention_heads

        if 'granite' in llm_name.lower():
            from src.custom.modeling_granite_attn import GraniteForCausalLM
            BaseLLMClass = GraniteForCausalLM
        elif 'llama' in llm_name.lower():
            from src.custom.modeling_llama_attn import LlamaForCausalLM
            BaseLLMClass = LlamaForCausalLM
        elif 'mistral' in llm_name.lower():
            from src.custom.modeling_mistral_attn import MistralForCausalLM
            BaseLLMClass = MistralForCausalLM
        elif 'phi' in llm_name.lower():
            from src.custom.modeling_phi_attn import Phi3ForCausalLM
            BaseLLMClass = Phi3ForCausalLM
        else:
            raise ValueError(f'Model {llm_name} not supported')

        # Setup quantization config
        quantization_config = None

        if quantize == '4bit':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif quantize == '8bit':
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

        # Load model - keep flash_attention_2 for memory efficiency
        load_kwargs = {
            'config': config,
            'device_map': 'cuda',
            'attn_implementation': 'flash_attention_2'
        }

        if quantization_config is not None:
            load_kwargs['quantization_config'] = quantization_config
        else:
            load_kwargs['torch_dtype'] = torch.float16

        try:
            self.llm = BaseLLMClass.from_pretrained(llm_name, **load_kwargs)
        except Exception as e:
            # Fall back to eager attention if flash attention fails with quantization
            print(f"Flash attention failed ({e}), falling back to eager attention", flush=True)
            load_kwargs['attn_implementation'] = 'eager'
            self.llm = BaseLLMClass.from_pretrained(llm_name, **load_kwargs)

        self.quantize = quantize

        # Setup prompts based on model
        self.offset = 0
        if 'granite' in llm_name.lower():
            self.prompt_prefix = '<|start_of_role|>user<|end_of_role|>'
            self.prompt_suffix = '<|end_of_text|><|start_of_role|>assistant<|end_of_role|>'
        elif 'llama' in llm_name.lower():
            self.prompt_prefix = '<|start_header_id|>user<|end_header_id|>'
            self.prompt_suffix = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        elif 'mistral' in llm_name.lower():
            self.prompt_prefix = '[INST]'
            self.prompt_suffix = '[/INST]'
            self.offset = 1
        elif 'phi' in llm_name.lower():
            self.prompt_prefix = '<|im_start|>user<|im_sep|>'
            self.prompt_suffix = '<|im_end|><|im_start|>assistant<|im_sep|>'

        self.retrieval_instruction = ' Here are some paragraphs:\n\n'
        self.retrieval_instruction_late = 'Please find information that are relevant to the following query in the paragraphs above.\n\nQuery: '

        self.num_layer = self.llm.config.num_hidden_layers
        self.num_head = self.llm.config.num_attention_heads
        print(f"Model loaded: {self.num_layer} layers, {self.num_head} heads", flush=True)

    def extract_features(self, query, documents, max_doc_tokens=300):
        """
        Extract attention features for each document from all heads.

        Args:
            query: query text
            documents: list of document dicts with 'paragraph_text'
            max_doc_tokens: maximum tokens per document (truncate longer docs)

        Returns:
            features: np.array of shape (num_docs, num_layers * num_heads)
        """
        # Truncate documents
        truncated_docs = []
        for doc in documents:
            text = doc.get('paragraph_text', '')
            # Simple word-based truncation
            words = text.split()[:max_doc_tokens]
            truncated_docs.append({'paragraph_text': ' '.join(words)})

        prompt, doc_spans, query_span = self.prepare_input(query, truncated_docs)

        # Get attention weights
        tokenized_input = self.tokenizer(prompt, return_tensors='pt').to(self.llm.device)
        _input_ids = tokenized_input.input_ids
        _query_indices = list(range(query_span[0], query_span[1] + 1))
        kv_cache = DynamicCacheWithQuery(query_indices=_query_indices)

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

        # Extract document-level scores for each head
        num_docs = len(doc_spans)
        features = np.zeros((num_docs, self.num_layer * self.num_head), dtype=np.float32)

        for doc_idx, (start, end) in enumerate(doc_spans):
            # Sum attention over document tokens for each head
            doc_attn = attn_weights[:, :, start:end].sum(-1)  # (num_layer, num_head)
            features[doc_idx] = doc_attn.cpu().numpy().flatten()

        del attn_weights
        torch.cuda.empty_cache()

        return features

    def prepare_input(self, query, documents):
        """Prepare input prompt and compute token spans for each document."""
        doc_spans = []
        llm_prompt = self.prompt_prefix + self.retrieval_instruction

        for i, doc in enumerate(documents):
            llm_prompt += f'[document {i + 1}]'
            start_len = len(self.tokenizer(llm_prompt).input_ids)

            llm_prompt += ' ' + doc['paragraph_text']
            end_len = len(self.tokenizer(llm_prompt).input_ids) - self.offset

            doc_spans.append((start_len, end_len))
            llm_prompt += '\n\n'

        start_len = len(self.tokenizer(llm_prompt).input_ids)
        llm_prompt += self.retrieval_instruction_late + f'{query.strip()}'
        end_len = len(self.tokenizer(llm_prompt).input_ids) - self.offset
        llm_prompt += self.prompt_suffix

        query_span = (start_len, end_len)

        return llm_prompt, doc_spans, query_span

    @classmethod
    def _get_attn_weights(cls, key_states, query_states):
        """Compute attention weights from key and query states."""
        import math
        num_layer, bsz, num_heads, q_len, head_dim = query_states.size()
        num_key_value_heads = key_states.size(2)
        num_key_value_groups = num_heads // num_key_value_heads
        kv_seq_len = key_states.size(-2)

        key_states = key_states.unsqueeze(3).expand(
            num_layer, bsz, num_key_value_heads, num_key_value_groups, kv_seq_len, head_dim
        )
        key_states = key_states.reshape(num_layer, bsz, num_heads, kv_seq_len, head_dim)
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(head_dim)

        del key_states, query_states
        torch.cuda.empty_cache()

        causal_mask = cls._get_causal_mask(attn_weights).to(attn_weights.device)
        attn_weights += causal_mask.unsqueeze(1)
        attn_lses = torch.logsumexp(attn_weights, dim=-1, keepdim=True)
        attn_weights = torch.exp(attn_weights - attn_lses)

        del causal_mask, attn_lses
        torch.cuda.empty_cache()

        return attn_weights

    @classmethod
    def _get_causal_mask(cls, attn_weights):
        """Create causal attention mask."""
        query_len, seq_len = attn_weights.size(-2), attn_weights.size(-1)
        causal_mask = torch.ones_like(attn_weights.transpose(-1, -2).squeeze(1))
        causal_mask = torch.triu(causal_mask, diagonal=-(seq_len - query_len))
        causal_mask = causal_mask.transpose(-1, -2)
        causal_mask = (1 - causal_mask) * torch.finfo(causal_mask.dtype).min
        return causal_mask


def open_file(filepath, mode='r', encoding='utf-8'):
    """
    Open a file, automatically handling compression based on extension.

    Supports:
        - .gz (gzip compression)
        - .bz2 (bzip2 compression)
        - uncompressed files

    Args:
        filepath: Path to the file
        mode: File mode ('r' for text, 'rb' for binary)
        encoding: Text encoding (default: utf-8)

    Returns:
        File handle
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix == '.gz':
        if 'b' in mode:
            return gzip.open(filepath, mode)
        return gzip.open(filepath, mode + 't', encoding=encoding)
    elif suffix == '.bz2':
        if 'b' in mode:
            return bz2.open(filepath, mode)
        return bz2.open(filepath, mode + 't', encoding=encoding)
    else:
        if 'b' in mode:
            return open(filepath, mode)
        return open(filepath, mode, encoding=encoding)


def detect_input_format(data):
    """Detect whether input is head detection format or retriever output format."""
    if len(data) == 0:
        return 'unknown'

    sample = data[0]

    # Head detection format has 'is_positive' in paragraphs
    if 'paragraphs' in sample and len(sample['paragraphs']) > 0:
        if 'is_positive' in sample['paragraphs'][0]:
            return 'head_detection'

    # Retriever output has 'idx' at top level
    if 'idx' in sample:
        return 'retriever_output'

    return 'unknown'


def load_qrels(qrels_file):
    """
    Load TREC-format qrels file.

    Supports formats:
    - 3 columns: query-id, corpus-id, score
    - 4 columns: query-id, iteration, corpus-id, score (TREC standard)

    Args:
        qrels_file: Path to qrels file

    Returns:
        dict: {(query_id, doc_id): relevance_score}
    """
    qrels = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) == 3:
                # Format: query-id, corpus-id, score
                query_id, doc_id, score = parts
            elif len(parts) >= 4:
                # Format: query-id, iteration, corpus-id, score (TREC standard)
                query_id, _, doc_id, score = parts[:4]
            else:
                continue

            try:
                score = int(score)
            except ValueError:
                try:
                    score = float(score)
                except ValueError:
                    continue

            qrels[(query_id, doc_id)] = score

    return qrels


def get_label_from_qrels(query_id, doc_id, qrels, relevance_threshold=1):
    """
    Get label for a (query, doc) pair from qrels.

    Args:
        query_id: Query identifier
        doc_id: Document identifier
        qrels: Dict from load_qrels()
        relevance_threshold: Minimum score to be considered positive (default: 1)

    Returns:
        1 if positive (in qrels with score >= threshold), 0 otherwise (negative or not in qrels)
    """
    key = (str(query_id), str(doc_id))
    if key in qrels:
        return 1 if qrels[key] >= relevance_threshold else 0
    return 0  # Not in qrels - treat as negative


def main():
    parser = argparse.ArgumentParser(description='Extract head attention features')
    parser.add_argument('--llm', type=str, default='mistral',
                        choices=['mistral', 'llama', 'phi', 'granite'])
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input JSON file (default: head_data/nq_core.json). Supports .gz and .bz2 compression.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: head_data/{llm}/)')
    parser.add_argument('--output_name', '-o', type=str, default=None,
                        help='Output filename (without extension). Default: attention_features_{input}_{n}_{quantize}')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (default: all)')
    parser.add_argument('--max_docs', type=int, default=None,
                        help='Maximum documents per query (default: all)')
    parser.add_argument('--max_doc_tokens', type=int, default=300,
                        help='Maximum tokens per document (default: 300)')
    parser.add_argument('--prune', type=float, default=0.0,
                        help='Layer pruning ratio (default: 0.0)')
    parser.add_argument('--qrels', type=str, default=None,
                        help='TREC qrels file for relevance labels (format: query-id doc-id score)')
    parser.add_argument('--relevance_threshold', type=int, default=1,
                        help='Minimum qrels score to be considered positive (default: 1)')
    parser.add_argument('--quantize', type=str, default=None, choices=[None, '4bit', '8bit'],
                        help='Quantization mode: 4bit, 8bit, or None (default: None)')
    args = parser.parse_args()

    # Determine input file
    if args.input_file is None:
        input_file = Path(__file__).parent.parent / 'head_data' / 'nq_core.json'
    else:
        input_file = Path(args.input_file)

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    # Load data (supports .gz and .bz2 compressed files)
    print(f"Loading data from {input_file}...", flush=True)
    with open_file(input_file, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples", flush=True)

    # Detect format
    input_format = detect_input_format(data)
    print(f"Detected format: {input_format}", flush=True)

    # Load qrels if provided
    qrels = None
    if args.qrels is not None:
        qrels_path = Path(args.qrels)
        if qrels_path.exists():
            qrels = load_qrels(qrels_path)
            print(f"Loaded {len(qrels)} qrels from {qrels_path}", flush=True)
        else:
            print(f"Warning: qrels file not found: {qrels_path}", flush=True)

    # Limit samples
    if args.max_samples is not None:
        data = data[:args.max_samples]
        print(f"Limited to {len(data)} samples", flush=True)

    # Initialize extractor
    extractor = FeatureExtractor(LLM_NAMES[args.llm], prune=args.prune, quantize=args.quantize)
    print(f"Total features per document: {extractor.num_layer * extractor.num_head}", flush=True)

    # Extract features
    all_features = []
    all_labels = []
    all_query_ids = []
    all_doc_ids = []
    docs_per_query = []

    for sample in tqdm(data, desc="Extracting features"):
        # Get query
        query = sample.get('question', sample.get('query', ''))
        query_id = sample.get('idx', '')

        # Get documents
        documents = sample.get('paragraphs', [])

        # Limit documents per query
        if args.max_docs is not None:
            documents = documents[:args.max_docs]

        if len(documents) == 0:
            continue

        # Extract features
        try:
            features = extractor.extract_features(query, documents, max_doc_tokens=args.max_doc_tokens)
        except torch.cuda.OutOfMemoryError:
            print(f"Warning: OOM for query {query_id} with {len(documents)} docs, skipping", flush=True)
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"Warning: Failed to extract features for query {query_id}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue

        # Get document IDs
        doc_ids = [d.get('idx', f'doc_{i}') for i, d in enumerate(documents)]

        # Get labels
        if input_format == 'head_detection':
            # Use is_positive field from head detection data
            labels = np.array([1 if d.get('is_positive', False) else 0 for d in documents], dtype=np.int32)
        elif qrels is not None:
            # Use qrels file for labels
            labels = np.array([
                get_label_from_qrels(query_id, doc_id, qrels, args.relevance_threshold)
                for doc_id in doc_ids
            ], dtype=np.int32)
        else:
            # No labels available
            labels = np.full(len(documents), -1, dtype=np.int32)

        all_features.append(features)
        all_labels.append(labels)
        all_query_ids.extend([query_id] * len(documents))
        all_doc_ids.extend(doc_ids)
        docs_per_query.append(len(documents))

    # Stack all features
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)

    print(f"\nExtracted features shape: {all_features.shape}", flush=True)
    print(f"Labels shape: {all_labels.shape}", flush=True)

    # Print label statistics
    n_positive = int((all_labels == 1).sum())
    n_negative = int((all_labels == 0).sum())
    n_unknown = int((all_labels == -1).sum())
    if n_positive > 0 or n_negative > 0:
        print(f"Positive samples: {n_positive}, Negative samples: {n_negative}", flush=True)
    else:
        print(f"No labels available (all {n_unknown} samples unlabeled)", flush=True)

    print(f"Queries processed: {len(docs_per_query)}", flush=True)
    print(f"Docs per query: min={min(docs_per_query)}, max={max(docs_per_query)}, avg={np.mean(docs_per_query):.1f}", flush=True)

    # Determine output path
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / 'head_data' / args.llm

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output name
    if args.output_name is not None:
        output_name = args.output_name
    else:
        # Derive from input file name (strip compression extensions)
        input_stem = input_file.stem  # e.g., 'nq_core' or 'nq'
        # Handle double extensions like .json.gz
        if input_stem.endswith('.json'):
            input_stem = input_stem[:-5]
        n_samples = len(docs_per_query)
        quant_suffix = f'_{args.quantize}' if args.quantize else ''
        output_name = f'attention_features_{input_stem}_n{n_samples}{quant_suffix}'

    # Save features
    output_file = output_dir / f'{output_name}.npz'
    np.savez(
        output_file,
        features=all_features,
        labels=all_labels,
        query_ids=np.array(all_query_ids, dtype=object),
        doc_ids=np.array(all_doc_ids, dtype=object),
        docs_per_query=np.array(docs_per_query, dtype=np.int32)
    )
    print(f"\nSaved features to {output_file}", flush=True)

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
        'prune': args.prune,
        'quantize': args.quantize,
        'has_labels': has_labels,
        'qrels_file': str(args.qrels) if args.qrels else None,
        'relevance_threshold': args.relevance_threshold if args.qrels else None,
        'label_stats': {
            'positive': int(n_positive),
            'negative': int(n_negative),
            'unknown': int(n_unknown)
        }
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file}", flush=True)


if __name__ == '__main__':
    main()
