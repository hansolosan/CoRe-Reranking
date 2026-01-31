"""
Custom cache implementation for storing query states alongside key-value states.

Extends HuggingFace's DynamicCache to additionally cache query states during
the forward pass, enabling attention weight computation for head detection and
reranking.
"""
from typing import Any, Dict, Optional, Tuple
from transformers.cache_utils import DynamicCache
import torch

class DynamicCacheWithQuery(DynamicCache):
    """
    Dynamic cache that stores query states in addition to key-value states.

    Extends transformers.cache_utils.DynamicCache to cache query states during
    the forward pass. This enables extraction of query states for computing
    attention weights externally, which is needed for head detection and reranking.

    Attributes:
        key_cache: List of key tensors, one per layer
        value_cache: List of value tensors, one per layer
        query_cache: List of query tensors, one per layer (custom addition)
        _seen_tokens: Counter for total tokens processed
        _query_indices: Token indices to cache (for single-query mode). If empty, caches all.
        _capture_all_queries: If True, caches all query states (for batched inference)
    """

    def __init__(self, query_indices=[], capture_all_queries=False) -> None:
        """
        Initialize the cache with query state capturing.

        Args:
            query_indices: List of token indices to cache query states for.
                           Used in single-query mode to cache only query tokens.
                           If empty and capture_all_queries=False, caches all tokens.
            capture_all_queries: If True, caches all query states regardless of
                                 query_indices. Used for batched inference where
                                 different batch items may have queries at different
                                 positions (default: False).
        """
        super().__init__()
        self._seen_tokens = 0  # Initialize for compatibility with newer transformers
        self.key_cache = []  # Initialize for compatibility with newer transformers
        self.value_cache = []  # Initialize for compatibility with newer transformers
        self._query_indices = query_indices # indices for query vectors to save
        self._capture_all_queries = capture_all_queries  # If True, cache all query states (for batched inference)
        self.query_cache = []

    def update(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with new key, value, and query states.

        Called during the forward pass of each layer to accumulate cached states.
        Automatically concatenates new states with existing cached states for
        autoregressive generation.

        Args:
            query_states: Query tensor of shape (batch_size, num_heads, seq_len, head_dim).
                          Can be None if not provided by the attention module.
            key_states: Key tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
            value_states: Value tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
            layer_idx: Index of the transformer layer (0-based)
            cache_kwargs: Optional kwargs for future extensions (not currently used)

        Returns:
            Tuple of (updated_key_cache, updated_value_cache) for this layer.
            Shape: (batch_size, num_kv_heads, total_seq_len, head_dim)

        Side effects:
            Updates self.key_cache, self.value_cache, self.query_cache, and self._seen_tokens
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        if query_states is not None:
            if len(self.query_cache) <= layer_idx:
                self.query_cache.append(query_states)
            else:
                self.query_cache[layer_idx] = torch.cat([self.query_cache[layer_idx], query_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        """
        Convert legacy cache format to DynamicCacheWithQuery.

        Converts the tuple-based cache format used in older HuggingFace transformers
        versions to the DynamicCacheWithQuery format.

        Args:
            past_key_values: Legacy cache format - tuple of (key, value) tuples,
                             one per layer. Each key/value has shape
                             (batch_size, num_heads, seq_len, head_dim).
                             If None, returns an empty cache.

        Returns:
            DynamicCacheWithQuery instance populated with the provided key-value states.
            Note: query_cache will be empty since legacy format doesn't include queries.
        """
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(None, key_states, value_states, layer_idx)
        return cache