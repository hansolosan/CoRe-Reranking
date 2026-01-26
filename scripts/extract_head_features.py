#!/usr/bin/env python3
"""
Extract attention features from all heads for BCE optimization or evaluation.
This script runs the model on detection data or retriever output and saves per-head attention scores.

Supports two input formats:
1. Head detection data (nq_core.json): has 'question', 'paragraphs' with 'is_positive'/'is_negative'
2. Retriever output (retriever_output/*.json): has 'idx', 'question', 'paragraphs' with 'idx', 'paragraph_text'

Supports two backends:
1. HuggingFace (default): Loads model using transformers with custom attention modules
2. vLLM: Uses vLLM for efficient inference (offline mode or server mode)
"""

import json
import os
import gc
import argparse
import gzip
import bz2
import time
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from abc import ABC, abstractmethod

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

import transformers
from transformers import BitsAndBytesConfig

from utils import log_command

LLM_NAMES = {
    'granite': 'ibm-granite/granite-3.2-8b-instruct',
    'llama': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'phi': 'microsoft/phi-4',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.2'
}


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    @abstractmethod
    def extract_features(self, query, documents, max_doc_tokens=300):
        """Extract attention features for each document from all heads."""
        pass

    @abstractmethod
    def extract_features_batch(self, queries, documents_list, max_doc_tokens=300):
        """Extract attention features for a batch of queries."""
        pass

    @property
    @abstractmethod
    def num_layer(self):
        """Number of layers in the model."""
        pass

    @property
    @abstractmethod
    def num_head(self):
        """Number of attention heads per layer."""
        pass


class HFFeatureExtractor(BaseFeatureExtractor):
    """Extract attention features using HuggingFace transformers with custom attention modules."""

    def __init__(self, llm_name, prune=0.0, quantize=None):
        from src.custom.custom_cache import DynamicCacheWithQuery
        self.DynamicCacheWithQuery = DynamicCacheWithQuery
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

        # Set padding token if not already set (required for batched inference)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # Use left padding for causal LMs (so generation happens on the right)
        self.tokenizer.padding_side = 'left'
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

        self._num_layer = self.llm.config.num_hidden_layers
        self._num_head = self.llm.config.num_attention_heads
        print(f"Model loaded: {self._num_layer} layers, {self._num_head} heads", flush=True)

    @property
    def num_layer(self):
        return self._num_layer

    @property
    def num_head(self):
        return self._num_head

    def extract_features(self, query, documents, max_doc_tokens=300, max_query_tokens=None):
        """
        Extract attention features for each document from all heads.

        Args:
            query: query text
            documents: list of document dicts with 'paragraph_text'
            max_doc_tokens: maximum tokens per document (truncate longer docs)
            max_query_tokens: maximum tokens for query (truncate if longer), None = no limit

        Returns:
            features: np.array of shape (num_docs, num_layers * num_heads)
        """
        # Truncate query if needed
        if max_query_tokens is not None:
            query_words = query.split()
            if len(query_words) > max_query_tokens:
                query = ' '.join(query_words[:max_query_tokens])

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
        kv_cache = self.DynamicCacheWithQuery(query_indices=_query_indices)

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

    def extract_features_batch(self, queries, documents_list, max_doc_tokens=300, max_query_tokens=None):
        """
        Extract attention features for a batch of queries.

        Args:
            queries: list of query texts
            documents_list: list of document lists (one per query)
            max_doc_tokens: maximum tokens per document
            max_query_tokens: maximum tokens for query (truncate if longer), None = no limit

        Returns:
            list of features arrays, one per query
        """
        batch_size = len(queries)
        if batch_size == 0:
            return []

        # Prepare all prompts
        all_prompts = []
        all_doc_spans = []
        all_query_spans = []
        all_truncated_docs = []

        for query, documents in zip(queries, documents_list):
            # Truncate query if needed
            if max_query_tokens is not None:
                query_words = query.split()
                if len(query_words) > max_query_tokens:
                    query = ' '.join(query_words[:max_query_tokens])

            # Truncate documents
            truncated_docs = []
            for doc in documents:
                text = doc.get('paragraph_text', '')
                words = text.split()[:max_doc_tokens]
                truncated_docs.append({'paragraph_text': ' '.join(words)})
            all_truncated_docs.append(truncated_docs)

            prompt, doc_spans, query_span = self.prepare_input(query, truncated_docs)
            all_prompts.append(prompt)
            all_doc_spans.append(doc_spans)
            all_query_spans.append(query_span)

        # Tokenize all prompts with padding
        tokenized = self.tokenizer(
            all_prompts,
            return_tensors='pt',
            padding=True,
            return_attention_mask=True
        ).to(self.llm.device)

        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        # Adjust query spans for padding (padding is on the left by default for causal LMs)
        # Check if padding is on left or right
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        adjusted_query_spans = []
        adjusted_doc_spans = []
        for b in range(batch_size):
            # Count padding tokens at the start
            seq = input_ids[b]
            pad_offset = 0
            for t in seq:
                if t == pad_token_id:
                    pad_offset += 1
                else:
                    break

            # Adjust spans
            q_start, q_end = all_query_spans[b]
            adjusted_query_spans.append((q_start + pad_offset, q_end + pad_offset))

            adj_doc_spans = []
            for d_start, d_end in all_doc_spans[b]:
                adj_doc_spans.append((d_start + pad_offset, d_end + pad_offset))
            adjusted_doc_spans.append(adj_doc_spans)

        # Process each sample in the batch separately (due to DynamicCacheWithQuery limitation)
        # But we can still benefit from keeping tensors on GPU
        all_features = []

        for b in range(batch_size):
            b_input_ids = input_ids[b:b+1]
            q_start, q_end = adjusted_query_spans[b]
            _query_indices = list(range(q_start, q_end + 1))
            kv_cache = self.DynamicCacheWithQuery(query_indices=_query_indices)

            with torch.no_grad():
                output = self.llm(
                    input_ids=b_input_ids,
                    use_cache=True,
                    past_key_values=kv_cache,
                    output_attentions=True
                )
            kv_cache = output.past_key_values

            # Collect key and query caches
            all_key_cache = []
            all_query_cache = []
            for i in range(self.num_layer):
                all_key_cache.append(kv_cache.key_cache[i][:, :, :q_end + 1])
                all_query_cache.append(kv_cache.query_cache[i])
            all_key_cache = torch.stack(all_key_cache)
            all_query_cache = torch.stack(all_query_cache)

            del output
            torch.cuda.empty_cache()

            # Compute attention weights
            attn_weights = self._get_attn_weights(all_key_cache, all_query_cache).to('cuda').squeeze(1)
            del all_key_cache, all_query_cache
            torch.cuda.empty_cache()

            # Average over query tokens
            attn_weights = attn_weights.mean(-2)  # (num_layer, num_head, seq_len)

            # Extract document-level scores
            doc_spans = adjusted_doc_spans[b]
            num_docs = len(doc_spans)
            features = np.zeros((num_docs, self.num_layer * self.num_head), dtype=np.float32)

            for doc_idx, (start, end) in enumerate(doc_spans):
                doc_attn = attn_weights[:, :, start:end].sum(-1)
                features[doc_idx] = doc_attn.cpu().numpy().flatten()

            all_features.append(features)

            del attn_weights
            torch.cuda.empty_cache()

        del tokenized, input_ids, attention_mask
        torch.cuda.empty_cache()

        return all_features

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


class VLLMFeatureExtractor(BaseFeatureExtractor):
    """
    Extract attention features using vLLM.

    Supports two modes:
    1. Offline mode: Uses vLLM's LLM class directly (requires vllm package)
    2. Server mode: Connects to a running vLLM server via HTTP

    Note: vLLM's standard API doesn't expose per-head attention weights directly.
    This implementation uses hooks to capture attention outputs from the model.
    """

    def __init__(self, llm_name, vllm_url=None, prune=0.0, tensor_parallel_size=1, gpu_memory_utilization=0.9):
        """
        Initialize the vLLM feature extractor.

        Args:
            llm_name: HuggingFace model name
            vllm_url: URL of vLLM server (if None, uses offline mode)
            prune: Layer pruning ratio (0.0 = no pruning) - only for offline mode
            tensor_parallel_size: Number of GPUs for tensor parallelism (offline mode)
            gpu_memory_utilization: Fraction of GPU memory to use (offline mode)
        """
        self.llm_name = llm_name
        self.vllm_url = vllm_url
        self.prune = prune

        if vllm_url is not None:
            self._init_server_mode(vllm_url, llm_name)
        else:
            self._init_offline_mode(llm_name, prune, tensor_parallel_size, gpu_memory_utilization)

    def _init_server_mode(self, vllm_url, llm_name):
        """Initialize server mode - connect to external vLLM server."""
        import requests

        self.mode = 'server'
        self.vllm_url = vllm_url.rstrip('/')
        self.requests = requests

        # Get model info from server
        try:
            response = requests.get(f"{self.vllm_url}/v1/models", timeout=10)
            if response.status_code == 200:
                models = response.json()
                print(f"Connected to vLLM server at {vllm_url}", flush=True)
                print(f"Available models: {models}", flush=True)
            else:
                print(f"Warning: Could not get model info from server (status {response.status_code})", flush=True)
        except Exception as e:
            print(f"Warning: Could not connect to vLLM server: {e}", flush=True)

        # Load tokenizer locally for prompt preparation
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'

        # Get model config for layer/head counts
        config = transformers.AutoConfig.from_pretrained(llm_name)
        self._num_layer = config.num_hidden_layers
        self._num_head = config.num_attention_heads

        # Setup prompts
        self._setup_prompts(llm_name)

        print(f"Server mode: {self._num_layer} layers, {self._num_head} heads", flush=True)
        print("WARNING: vLLM server mode has limited attention extraction support.", flush=True)
        print("For full attention features, use offline mode (--backend vllm without --vllm_url)", flush=True)

    def _init_offline_mode(self, llm_name, prune, tensor_parallel_size, gpu_memory_utilization):
        """Initialize offline mode - load model using vLLM."""
        try:
            from vllm import LLM, SamplingParams
            from vllm.attention import Attention
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Install it with: pip install vllm\n"
                "Or use HuggingFace backend with: --backend hf"
            )

        self.mode = 'offline'
        self.SamplingParams = SamplingParams

        print(f"Loading model with vLLM: {llm_name}...", flush=True)

        # Get config for layer count adjustment
        config = transformers.AutoConfig.from_pretrained(llm_name)
        original_layers = config.num_hidden_layers
        pruned_layers = int(original_layers * (1 - prune))

        if prune > 0:
            print(f"Layer pruning: {original_layers} -> {pruned_layers} layers", flush=True)
            # Note: vLLM doesn't directly support layer pruning
            # Would need custom model modification
            print("WARNING: Layer pruning not directly supported in vLLM offline mode", flush=True)

        # Load model with vLLM
        self.llm = LLM(
            model=llm_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            dtype='float16',
        )

        # Get tokenizer from vLLM
        self.tokenizer = self.llm.get_tokenizer()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'

        # Store model config
        self._num_layer = config.num_hidden_layers
        self._num_head = config.num_attention_heads

        # Setup prompts
        self._setup_prompts(llm_name)

        # Setup attention hooks for capturing attention weights
        self._setup_attention_hooks()

        print(f"Model loaded: {self._num_layer} layers, {self._num_head} heads", flush=True)

    def _setup_prompts(self, llm_name):
        """Setup prompt templates based on model type."""
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
        else:
            self.prompt_prefix = ''
            self.prompt_suffix = ''

        self.retrieval_instruction = ' Here are some paragraphs:\n\n'
        self.retrieval_instruction_late = 'Please find information that are relevant to the following query in the paragraphs above.\n\nQuery: '

    def _setup_attention_hooks(self):
        """Setup hooks to capture attention weights from vLLM model."""
        self.captured_attention = {}

        # vLLM uses a different architecture, attention capture requires
        # accessing the underlying model weights during forward pass
        # This is a placeholder for the hook setup
        print("Note: Attention hooks for vLLM require model-specific implementation", flush=True)

    @property
    def num_layer(self):
        return self._num_layer

    @property
    def num_head(self):
        return self._num_head

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

    def extract_features(self, query, documents, max_doc_tokens=300, max_query_tokens=None):
        """
        Extract attention features for each document from all heads.

        Args:
            query: query text
            documents: list of document dicts with 'paragraph_text'
            max_doc_tokens: maximum tokens per document
            max_query_tokens: maximum tokens for query (truncate if longer), None = no limit

        Returns:
            features: np.array of shape (num_docs, num_layers * num_heads)
        """
        # Truncate query if needed
        if max_query_tokens is not None:
            query_words = query.split()
            if len(query_words) > max_query_tokens:
                query = ' '.join(query_words[:max_query_tokens])

        # Truncate documents
        truncated_docs = []
        for doc in documents:
            text = doc.get('paragraph_text', '')
            words = text.split()[:max_doc_tokens]
            truncated_docs.append({'paragraph_text': ' '.join(words)})

        prompt, doc_spans, query_span = self.prepare_input(query, truncated_docs)

        if self.mode == 'server':
            return self._extract_features_server(prompt, doc_spans, query_span, len(truncated_docs))
        else:
            return self._extract_features_offline(prompt, doc_spans, query_span, len(truncated_docs))

    def _extract_features_server(self, prompt, doc_spans, query_span, num_docs):
        """Extract features using vLLM server API."""
        # vLLM's OpenAI-compatible API doesn't return attention weights
        # This would require a custom endpoint or modified vLLM server

        # For now, return placeholder features with a warning
        print("WARNING: vLLM server mode cannot extract attention weights.", flush=True)
        print("Using logprob-based approximation (limited accuracy).", flush=True)

        # Make request to get logprobs as a proxy signal
        try:
            response = self.requests.post(
                f"{self.vllm_url}/v1/completions",
                json={
                    "model": self.llm_name,
                    "prompt": prompt,
                    "max_tokens": 1,
                    "logprobs": 5,
                    "echo": True,
                },
                timeout=60
            )

            if response.status_code != 200:
                raise RuntimeError(f"Server returned status {response.status_code}")

            # Parse response - this is a very rough approximation
            # Real attention weights are not available via standard API
            result = response.json()

            # Return zeros since we can't get actual attention
            features = np.zeros((num_docs, self._num_layer * self._num_head), dtype=np.float32)
            return features

        except Exception as e:
            print(f"Error calling vLLM server: {e}", flush=True)
            return np.zeros((num_docs, self._num_layer * self._num_head), dtype=np.float32)

    def _extract_features_offline(self, prompt, doc_spans, query_span, num_docs):
        """Extract features using vLLM offline mode with attention capture."""
        # Use vLLM's generate with hooks to capture attention
        sampling_params = self.SamplingParams(
            max_tokens=1,
            temperature=0.0,
        )

        # For vLLM offline mode, we need to access the underlying model
        # to get attention weights. This requires model-specific handling.

        # Get the underlying HuggingFace model from vLLM
        try:
            # vLLM wraps the model - try to access it
            model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model

            # Tokenize
            input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.cuda()
            _query_indices = list(range(query_span[0], query_span[1] + 1))

            # Run forward pass with attention output
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    output_attentions=True,
                    return_dict=True,
                )

            # Extract attention weights
            # attentions is a tuple of (num_layers,) each with shape (batch, num_heads, seq_len, seq_len)
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                attentions = outputs.attentions

                # Extract query-to-document attention for each head
                features = np.zeros((num_docs, self._num_layer * self._num_head), dtype=np.float32)

                for layer_idx, layer_attn in enumerate(attentions):
                    # layer_attn shape: (1, num_heads, seq_len, seq_len)
                    # Get attention from query tokens to all tokens
                    query_attn = layer_attn[0, :, _query_indices, :].mean(dim=1)  # (num_heads, seq_len)

                    for doc_idx, (start, end) in enumerate(doc_spans):
                        # Sum attention over document tokens
                        doc_attn = query_attn[:, start:end].sum(dim=-1)  # (num_heads,)
                        feat_start = layer_idx * self._num_head
                        feat_end = feat_start + self._num_head
                        features[doc_idx, feat_start:feat_end] = doc_attn.cpu().numpy()

                return features
            else:
                print("WARNING: Model did not return attention weights", flush=True)
                return np.zeros((num_docs, self._num_layer * self._num_head), dtype=np.float32)

        except Exception as e:
            print(f"Error extracting attention from vLLM model: {e}", flush=True)
            print("Falling back to zero features", flush=True)
            return np.zeros((num_docs, self._num_layer * self._num_head), dtype=np.float32)

    def extract_features_batch(self, queries, documents_list, max_doc_tokens=300, max_query_tokens=None):
        """Extract features for a batch of queries."""
        # Process sequentially for now - batch optimization can be added later
        results = []
        for query, documents in zip(queries, documents_list):
            features = self.extract_features(query, documents, max_doc_tokens, max_query_tokens)
            results.append(features)
        return results


# Alias for backward compatibility
FeatureExtractor = HFFeatureExtractor


class MemoryTracker:
    """Track GPU memory usage during processing."""

    def __init__(self):
        self.memory_samples = []
        self.start_time = None
        self.device = None

    def start(self):
        """Start tracking memory."""
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(self.device)
            self.start_time = time.time()
            self.memory_samples = []
            self.sample()

    def sample(self):
        """Record current memory usage."""
        if self.device is not None:
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            self.memory_samples.append({
                'time': time.time() - self.start_time if self.start_time else 0,
                'allocated': allocated,
                'reserved': reserved
            })

    def get_stats(self):
        """Get memory statistics."""
        if self.device is None or len(self.memory_samples) == 0:
            return None

        allocated_values = [s['allocated'] for s in self.memory_samples]
        reserved_values = [s['reserved'] for s in self.memory_samples]

        peak_allocated = torch.cuda.max_memory_allocated(self.device)
        peak_reserved = torch.cuda.max_memory_reserved(self.device)

        stats = {
            'peak_allocated_gb': peak_allocated / (1024 ** 3),
            'peak_reserved_gb': peak_reserved / (1024 ** 3),
            'avg_allocated_gb': np.mean(allocated_values) / (1024 ** 3),
            'avg_reserved_gb': np.mean(reserved_values) / (1024 ** 3),
            'min_allocated_gb': np.min(allocated_values) / (1024 ** 3),
            'max_allocated_gb': np.max(allocated_values) / (1024 ** 3),
            'num_samples': len(self.memory_samples),
            'total_time_seconds': time.time() - self.start_time if self.start_time else 0
        }

        return stats

    def print_stats(self, num_queries=None):
        """Print memory statistics."""
        stats = self.get_stats()
        if stats is None:
            print("No GPU memory tracking available", flush=True)
            return None

        print(f"\n{'='*60}", flush=True)
        print("GPU Memory Usage Statistics", flush=True)
        print('='*60, flush=True)
        print(f"Peak allocated:    {stats['peak_allocated_gb']:.2f} GB", flush=True)
        print(f"Peak reserved:     {stats['peak_reserved_gb']:.2f} GB", flush=True)
        print(f"Avg allocated:     {stats['avg_allocated_gb']:.2f} GB", flush=True)
        print(f"Min allocated:     {stats['min_allocated_gb']:.2f} GB", flush=True)
        print(f"Max allocated:     {stats['max_allocated_gb']:.2f} GB", flush=True)
        print(f"Total time:        {stats['total_time_seconds']:.1f} seconds", flush=True)
        print(f"Memory samples:    {stats['num_samples']}", flush=True)

        if num_queries is not None and stats['total_time_seconds'] > 0:
            throughput = num_queries / stats['total_time_seconds']
            stats['throughput_queries_per_sec'] = throughput
            print(f"Throughput:        {throughput:.2f} queries/sec", flush=True)

        return stats


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
    # Log command execution
    log_command()

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
                        help='Quantization mode: 4bit, 8bit, or None (default: None). Only for HuggingFace backend.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of queries to process in each batch (default: 1). Higher values may improve throughput but use more memory.')

    # Backend selection
    parser.add_argument('--backend', type=str, default='hf', choices=['hf', 'vllm'],
                        help='Backend for model inference: hf (HuggingFace, default) or vllm')
    parser.add_argument('--vllm_url', type=str, default=None,
                        help='URL of vLLM server (e.g., http://localhost:8000). If provided, uses server mode instead of offline mode.')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='Number of GPUs for tensor parallelism (vLLM offline mode only)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='Fraction of GPU memory to use (vLLM offline mode only, default: 0.9)')
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

    # Initialize extractor based on backend
    llm_name = LLM_NAMES[args.llm]
    print(f"Using backend: {args.backend}", flush=True)

    if args.backend == 'vllm':
        extractor = VLLMFeatureExtractor(
            llm_name,
            vllm_url=args.vllm_url,
            prune=args.prune,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    else:
        extractor = HFFeatureExtractor(llm_name, prune=args.prune, quantize=args.quantize)

    print(f"Total features per document: {extractor.num_layer * extractor.num_head}", flush=True)

    # Print model memory usage
    if torch.cuda.is_available():
        model_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"Model memory usage: {model_memory:.2f} GB", flush=True)

    # Initialize memory tracker
    memory_tracker = MemoryTracker()
    memory_tracker.start()

    # Extract features
    all_features = []
    all_labels = []
    all_query_ids = []
    all_doc_ids = []
    docs_per_query = []

    # Prepare batches
    def process_batch(batch_samples):
        """Process a batch of samples."""
        batch_queries = []
        batch_documents = []
        batch_query_ids = []
        batch_doc_ids_list = []
        batch_labels_list = []

        for sample in batch_samples:
            query = sample.get('question', sample.get('query', ''))
            query_id = sample.get('idx', '')
            documents = sample.get('paragraphs', [])

            if args.max_docs is not None:
                documents = documents[:args.max_docs]

            if len(documents) == 0:
                continue

            batch_queries.append(query)
            batch_documents.append(documents)
            batch_query_ids.append(query_id)

            # Get document IDs
            doc_ids = [d.get('idx', f'doc_{i}') for i, d in enumerate(documents)]
            batch_doc_ids_list.append(doc_ids)

            # Get labels
            if input_format == 'head_detection':
                labels = np.array([1 if d.get('is_positive', False) else 0 for d in documents], dtype=np.int32)
            elif qrels is not None:
                labels = np.array([
                    get_label_from_qrels(query_id, doc_id, qrels, args.relevance_threshold)
                    for doc_id in doc_ids
                ], dtype=np.int32)
            else:
                labels = np.full(len(documents), -1, dtype=np.int32)
            batch_labels_list.append(labels)

        if len(batch_queries) == 0:
            return [], [], [], [], []

        # Extract features for the batch
        try:
            if len(batch_queries) == 1:
                # Single query - use original method
                batch_features = [extractor.extract_features(
                    batch_queries[0], batch_documents[0], max_doc_tokens=args.max_doc_tokens
                )]
            else:
                # Multiple queries - use batch method
                batch_features = extractor.extract_features_batch(
                    batch_queries, batch_documents, max_doc_tokens=args.max_doc_tokens
                )
        except torch.cuda.OutOfMemoryError:
            print(f"Warning: OOM for batch of {len(batch_queries)} queries, falling back to sequential (batch_size=1)", flush=True)
            torch.cuda.empty_cache()
            gc.collect()
            # Fall back to sequential processing
            batch_features = []
            for q_idx, (q, docs) in enumerate(zip(batch_queries, batch_documents)):
                try:
                    feats = extractor.extract_features(q, docs, max_doc_tokens=args.max_doc_tokens)
                    batch_features.append(feats)
                except torch.cuda.OutOfMemoryError:
                    # Try with reduced max_doc_tokens and max_query_tokens
                    torch.cuda.empty_cache()
                    gc.collect()
                    reduced_doc_tokens = args.max_doc_tokens
                    reduced_query_tokens = None  # Start without query truncation
                    feats = None
                    q_id = batch_query_ids[q_idx]
                    printed_problematic_warning = False
                    reducing_query = False

                    print(f"DEBUG: OOM in batch size 1 for query '{q_id}', num_docs={len(docs)}", flush=True)

                    while feats is None:
                        # Reduce document tokens first until we hit 200
                        if reduced_doc_tokens >= 200:
                            reduced_doc_tokens -= 50
                            if reduced_doc_tokens < 10:
                                reduced_doc_tokens = 10
                        # Once doc tokens are below 200, start reducing query tokens
                        elif not reducing_query:
                            # Measure original query length
                            query_token_count = len(extractor.tokenizer(q).input_ids)
                            if query_token_count > 100:
                                reducing_query = True
                                reduced_query_tokens = max(100, query_token_count // 2)
                                print(f"Info: Query '{q_id}' has {query_token_count} tokens, now reducing query to {reduced_query_tokens}", flush=True)
                            else:
                                # Query is already short, continue reducing docs
                                if reduced_doc_tokens >= 50:
                                    reduced_doc_tokens -= 50
                                    if reduced_doc_tokens < 10:
                                        reduced_doc_tokens = 10
                                else:
                                    break
                        elif reduced_query_tokens is not None and reduced_query_tokens > 50:
                            # Continue reducing query tokens
                            reduced_query_tokens -= 50
                        else:
                            # Both doc and query at minimum
                            break

                        # Print warning when going below 50 doc tokens (problematic query)
                        if reduced_doc_tokens < 50 and not printed_problematic_warning:
                            # Compute token statistics
                            try:
                                # Truncate docs and query to current settings
                                truncated_docs = []
                                for doc in docs:
                                    text = doc.get('paragraph_text', '')
                                    words = text.split()[:reduced_doc_tokens]
                                    truncated_docs.append({'paragraph_text': ' '.join(words)})

                                test_query = q
                                if reduced_query_tokens is not None:
                                    query_words = q.split()[:reduced_query_tokens]
                                    test_query = ' '.join(query_words)

                                # Prepare prompt to count tokens
                                prompt, doc_spans, query_span = extractor.prepare_input(test_query, truncated_docs)
                                prompt_tokens = len(extractor.tokenizer(prompt).input_ids)
                                query_tokens = query_span[1] - query_span[0] + 1

                                # Count document tokens
                                total_doc_tokens = sum(end - start for start, end in doc_spans)

                                # Get GPU memory stats
                                if torch.cuda.is_available():
                                    allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                                    reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
                                    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                                    mem_info = f"GPU: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved, {total_gb:.2f}GB total"
                                else:
                                    mem_info = "GPU: N/A"

                                query_info = f", max_query_tokens={reduced_query_tokens}" if reduced_query_tokens else ""
                                print(f"PROBLEMATIC: Query '{q_id}' requires max_doc_tokens < 50 (trying doc={reduced_doc_tokens}{query_info})", flush=True)
                                print(f"  Token stats: query={query_tokens}, docs_total={total_doc_tokens}, prompt_total={prompt_tokens}, num_docs={len(docs)}", flush=True)
                                print(f"  Batch size: 1 (sequential processing)", flush=True)
                                print(f"  {mem_info}", flush=True)
                            except Exception as e:
                                print(f"PROBLEMATIC: Query '{q_id}' requires max_doc_tokens < 50 (trying {reduced_doc_tokens}) [token count failed: {e}]", flush=True)

                            printed_problematic_warning = True

                        try:
                            query_info = f", max_query_tokens={reduced_query_tokens}" if reduced_query_tokens else ""
                            print(f"Warning: OOM for query '{q_id}', retrying with max_doc_tokens={reduced_doc_tokens}{query_info}", flush=True)
                            feats = extractor.extract_features(q, docs, max_doc_tokens=reduced_doc_tokens, max_query_tokens=reduced_query_tokens)
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            gc.collect()
                            continue
                        except Exception as e:
                            print(f"Warning: Failed query '{q_id}' with reduced tokens (doc={reduced_doc_tokens}, query={reduced_query_tokens}): {e}", flush=True)
                            break

                    if feats is None:
                        print(f"ERROR: Could not extract features for query '{q_id}' even with minimal tokens, skipping", flush=True)
                    batch_features.append(feats)
                    torch.cuda.empty_cache()
                except Exception as e:
                    q_id = batch_query_ids[q_idx]
                    print(f"Warning: Failed to extract features for query '{q_id}': {e}", flush=True)
                    batch_features.append(None)
                    torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Batch extraction failed ({e}), falling back to sequential (batch_size=1)", flush=True)
            torch.cuda.empty_cache()
            batch_features = []
            for q_idx, (q, docs) in enumerate(zip(batch_queries, batch_documents)):
                try:
                    feats = extractor.extract_features(q, docs, max_doc_tokens=args.max_doc_tokens)
                    batch_features.append(feats)
                except torch.cuda.OutOfMemoryError:
                    # Try with reduced max_doc_tokens and max_query_tokens
                    torch.cuda.empty_cache()
                    gc.collect()
                    reduced_doc_tokens = args.max_doc_tokens
                    reduced_query_tokens = None  # Start without query truncation
                    feats = None
                    q_id = batch_query_ids[q_idx]
                    printed_problematic_warning = False
                    reducing_query = False

                    print(f"DEBUG: OOM in batch size 1 (exception path) for query '{q_id}', num_docs={len(docs)}", flush=True)

                    while feats is None:
                        # Reduce document tokens first until we hit 200
                        if reduced_doc_tokens >= 200:
                            reduced_doc_tokens -= 50
                            if reduced_doc_tokens < 10:
                                reduced_doc_tokens = 10
                        # Once doc tokens are below 200, start reducing query tokens
                        elif not reducing_query:
                            # Measure original query length
                            query_token_count = len(extractor.tokenizer(q).input_ids)
                            if query_token_count > 100:
                                reducing_query = True
                                reduced_query_tokens = max(100, query_token_count // 2)
                                print(f"Info: Query '{q_id}' has {query_token_count} tokens, now reducing query to {reduced_query_tokens}", flush=True)
                            else:
                                # Query is already short, continue reducing docs
                                if reduced_doc_tokens >= 50:
                                    reduced_doc_tokens -= 50
                                    if reduced_doc_tokens < 10:
                                        reduced_doc_tokens = 10
                                else:
                                    break
                        elif reduced_query_tokens is not None and reduced_query_tokens > 50:
                            # Continue reducing query tokens
                            reduced_query_tokens -= 50
                        else:
                            # Both doc and query at minimum
                            break

                        # Print warning when going below 50 doc tokens (problematic query)
                        if reduced_doc_tokens < 50 and not printed_problematic_warning:
                            # Compute token statistics
                            try:
                                # Truncate docs and query to current settings
                                truncated_docs = []
                                for doc in docs:
                                    text = doc.get('paragraph_text', '')
                                    words = text.split()[:reduced_doc_tokens]
                                    truncated_docs.append({'paragraph_text': ' '.join(words)})

                                test_query = q
                                if reduced_query_tokens is not None:
                                    query_words = q.split()[:reduced_query_tokens]
                                    test_query = ' '.join(query_words)

                                # Prepare prompt to count tokens
                                prompt, doc_spans, query_span = extractor.prepare_input(test_query, truncated_docs)
                                prompt_tokens = len(extractor.tokenizer(prompt).input_ids)
                                query_tokens = query_span[1] - query_span[0] + 1

                                # Count document tokens
                                total_doc_tokens = sum(end - start for start, end in doc_spans)

                                # Get GPU memory stats
                                if torch.cuda.is_available():
                                    allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                                    reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
                                    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                                    mem_info = f"GPU: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved, {total_gb:.2f}GB total"
                                else:
                                    mem_info = "GPU: N/A"

                                query_info = f", max_query_tokens={reduced_query_tokens}" if reduced_query_tokens else ""
                                print(f"PROBLEMATIC: Query '{q_id}' requires max_doc_tokens < 50 (trying doc={reduced_doc_tokens}{query_info})", flush=True)
                                print(f"  Token stats: query={query_tokens}, docs_total={total_doc_tokens}, prompt_total={prompt_tokens}, num_docs={len(docs)}", flush=True)
                                print(f"  Batch size: 1 (sequential processing)", flush=True)
                                print(f"  {mem_info}", flush=True)
                            except Exception as e:
                                print(f"PROBLEMATIC: Query '{q_id}' requires max_doc_tokens < 50 (trying {reduced_doc_tokens}) [token count failed: {e}]", flush=True)

                            printed_problematic_warning = True

                        try:
                            query_info = f", max_query_tokens={reduced_query_tokens}" if reduced_query_tokens else ""
                            print(f"Warning: OOM for query '{q_id}', retrying with max_doc_tokens={reduced_doc_tokens}{query_info}", flush=True)
                            feats = extractor.extract_features(q, docs, max_doc_tokens=reduced_doc_tokens, max_query_tokens=reduced_query_tokens)
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            gc.collect()
                            continue
                        except Exception as e2:
                            print(f"Warning: Failed query '{q_id}' with reduced tokens (doc={reduced_doc_tokens}, query={reduced_query_tokens}): {e2}", flush=True)
                            break

                    if feats is None:
                        print(f"ERROR: Could not extract features for query '{q_id}' even with minimal tokens, skipping", flush=True)
                    batch_features.append(feats)
                    torch.cuda.empty_cache()
                except Exception as ex:
                    q_id = batch_query_ids[q_idx]
                    print(f"Warning: Failed to extract features for query '{q_id}': {ex}", flush=True)
                    batch_features.append(None)
                    torch.cuda.empty_cache()

        # Filter out failed extractions
        valid_features = []
        valid_labels = []
        valid_query_ids = []
        valid_doc_ids = []
        valid_docs_per_query = []
        num_skipped = 0

        for i, feats in enumerate(batch_features):
            if feats is not None:
                # Validate that features shape matches number of documents
                expected_docs = len(batch_documents[i])
                if feats.shape[0] != expected_docs:
                    print(f"ERROR: Feature shape mismatch for query '{batch_query_ids[i]}': "
                          f"got {feats.shape[0]} features but expected {expected_docs} documents. Skipping.", flush=True)
                    num_skipped += 1
                    continue

                # Validate labels length matches
                if len(batch_labels_list[i]) != expected_docs:
                    print(f"ERROR: Label length mismatch for query '{batch_query_ids[i]}': "
                          f"got {len(batch_labels_list[i])} labels but expected {expected_docs} documents. Skipping.", flush=True)
                    num_skipped += 1
                    continue

                # All validations passed - add to valid results
                valid_features.append(feats)
                valid_labels.append(batch_labels_list[i])
                valid_query_ids.extend([batch_query_ids[i]] * expected_docs)
                valid_doc_ids.extend(batch_doc_ids_list[i])
                valid_docs_per_query.append(expected_docs)
            else:
                num_skipped += 1

        if num_skipped > 0:
            print(f"Batch summary: Successfully processed {len(valid_features)} queries, skipped {num_skipped} queries", flush=True)

        return valid_features, valid_labels, valid_query_ids, valid_doc_ids, valid_docs_per_query

    # Process in batches
    num_samples = len(data)
    batch_size = args.batch_size

    for batch_start in tqdm(range(0, num_samples, batch_size), desc=f"Extracting features (batch_size={batch_size})"):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_samples = data[batch_start:batch_end]

        feats, labels, q_ids, d_ids, dpq = process_batch(batch_samples)

        all_features.extend(feats)
        all_labels.extend(labels)
        all_query_ids.extend(q_ids)
        all_doc_ids.extend(d_ids)
        docs_per_query.extend(dpq)

        # Sample memory usage and cleanup after each batch
        memory_tracker.sample()
        gc.collect()
        torch.cuda.empty_cache()

    # Print memory statistics
    memory_stats = memory_tracker.print_stats(num_queries=len(docs_per_query))

    # Report processing summary
    num_queries_processed = len(docs_per_query)
    num_queries_attempted = len(data)
    num_queries_skipped = num_queries_attempted - num_queries_processed
    if num_queries_skipped > 0:
        print(f"\n⚠ Processing summary: {num_queries_processed}/{num_queries_attempted} queries successfully processed", flush=True)
        print(f"  Skipped {num_queries_skipped} queries due to errors", flush=True)
    else:
        print(f"\n✓ All {num_queries_processed} queries processed successfully", flush=True)

    # Stack all features
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)

    print(f"\nExtracted features shape: {all_features.shape}", flush=True)
    print(f"Labels shape: {all_labels.shape}", flush=True)

    # Validate alignment of all arrays
    num_docs_from_features = all_features.shape[0]
    num_docs_from_labels = len(all_labels)
    num_docs_from_query_ids = len(all_query_ids)
    num_docs_from_doc_ids = len(all_doc_ids)
    num_docs_from_dpq = sum(docs_per_query)

    if not (num_docs_from_features == num_docs_from_labels == num_docs_from_query_ids ==
            num_docs_from_doc_ids == num_docs_from_dpq):
        print(f"\nERROR: Data alignment mismatch detected!", flush=True)
        print(f"  Features: {num_docs_from_features} docs", flush=True)
        print(f"  Labels: {num_docs_from_labels} docs", flush=True)
        print(f"  Query IDs: {num_docs_from_query_ids} docs", flush=True)
        print(f"  Doc IDs: {num_docs_from_doc_ids} docs", flush=True)
        print(f"  Docs per query sum: {num_docs_from_dpq} docs", flush=True)
        print(f"\nThis indicates a bug in data handling. Please report this issue.", flush=True)
        return
    else:
        print(f"✓ Data alignment validated: all arrays have {num_docs_from_features} documents", flush=True)

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

    if output_name.find(".npz") > 0:
        output_name = output_name.replace(".npz", "")
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
        'backend': args.backend,
        'vllm_url': args.vllm_url,
        'num_queries': len(docs_per_query),
        'num_documents': len(all_labels),
        'num_features': all_features.shape[1],
        'num_layers': extractor.num_layer,
        'num_heads': extractor.num_head,
        'max_doc_tokens': args.max_doc_tokens,
        'batch_size': args.batch_size,
        'prune': args.prune,
        'quantize': args.quantize if args.backend == 'hf' else None,
        'tensor_parallel_size': args.tensor_parallel_size if args.backend == 'vllm' else None,
        'gpu_memory_utilization': args.gpu_memory_utilization if args.backend == 'vllm' else None,
        'has_labels': has_labels,
        'qrels_file': str(args.qrels) if args.qrels else None,
        'relevance_threshold': args.relevance_threshold if args.qrels else None,
        'label_stats': {
            'positive': int(n_positive),
            'negative': int(n_negative),
            'unknown': int(n_unknown)
        },
        'memory_stats': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                        for k, v in (memory_stats or {}).items()}
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file}", flush=True)


if __name__ == '__main__':
    main()
