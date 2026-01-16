#!/usr/bin/env python3
"""
Extract attention features from all heads for BCE optimization.
This script runs the model on head detection data and saves per-head attention scores.
"""

import json
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from src.custom.custom_cache import DynamicCacheWithQuery
import transformers

LLM_NAMES = {
    'granite': 'ibm-granite/granite-3.2-8b-instruct',
    'llama': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'phi': 'microsoft/phi-4',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.2'
}

class FeatureExtractor:
    """Extract attention features from all heads for optimization."""

    def __init__(self, llm_name, prune=0.0):
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

        self.llm = BaseLLMClass.from_pretrained(
            llm_name,
            config=config,
            torch_dtype=torch.float16,
            attn_implementation='flash_attention_2',
            device_map='cuda'
        )

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

    def extract_features(self, query, documents):
        """
        Extract attention features for each document from all heads.

        Returns:
            features: np.array of shape (num_docs, num_layers * num_heads)
            labels: np.array of shape (num_docs,) with 1 for positive, 0 for negative
        """
        prompt, doc_spans, query_span = self.prepare_input(query, documents)

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


def main():
    parser = argparse.ArgumentParser(description='Extract head attention features')
    parser.add_argument('--llm', type=str, default='mistral',
                        choices=['mistral', 'llama', 'phi', 'granite'])
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Maximum number of samples to process')
    parser.add_argument('--prune', type=float, default=0.0)
    args = parser.parse_args()

    # Load head detection data
    data_file = Path(__file__).parent.parent / 'head_data' / 'nq_core.json'
    with open(data_file, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {data_file}")

    # Limit samples
    data = data[:args.max_samples]
    print(f"Processing {len(data)} samples")

    # Initialize extractor
    print(f"Loading model: {LLM_NAMES[args.llm]}")
    extractor = FeatureExtractor(LLM_NAMES[args.llm], prune=args.prune)
    print(f"Model has {extractor.num_layer} layers, {extractor.num_head} heads")
    print(f"Total features per document: {extractor.num_layer * extractor.num_head}")

    # Extract features
    all_features = []
    all_labels = []

    for sample in tqdm(data, desc="Extracting features"):
        query = sample['question']
        documents = sample['paragraphs']

        features = extractor.extract_features(query, documents)
        labels = np.array([1 if d['is_positive'] else 0 for d in documents], dtype=np.int32)

        all_features.append(features)
        all_labels.append(labels)

    # Stack all features
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)

    print(f"\nExtracted features shape: {all_features.shape}")
    print(f"Labels shape: {all_labels.shape}")
    print(f"Positive samples: {all_labels.sum()}, Negative samples: {(1 - all_labels).sum()}")

    # Save features
    output_dir = Path(__file__).parent.parent / 'head_data' / args.llm
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'attention_features_n{args.max_samples}.npz'
    np.savez(output_file, features=all_features, labels=all_labels)
    print(f"\nSaved features to {output_file}")


if __name__ == '__main__':
    main()
