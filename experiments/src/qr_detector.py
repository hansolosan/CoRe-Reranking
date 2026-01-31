"""
Query-Relevance (QR) Head Detector.

This module implements the QR detector that identifies retrieval heads by
measuring how much attention each head pays to the positive document when
processing a query.
"""
import math
import transformers
import torch
from .custom.custom_cache import DynamicCacheWithQuery

class HeadDetector():
    """
    Query-Relevance (QR) Head Detector.

    Identifies retrieval-relevant attention heads by measuring the total attention
    each head pays to the positive document tokens. Unlike CoRe detector, QR does
    not use contrastive scoring with negative documents.

    Attributes:
        tokenizer: HuggingFace tokenizer for the LLM
        llm: Custom LLM with attention caching capabilities
        prompt_prefix: Model-specific prompt prefix (e.g., '[INST]' for Mistral)
        prompt_suffix: Model-specific prompt suffix (e.g., '[/INST]' for Mistral)
        retrieval_instruction: Instruction text before documents
        retrieval_instruction_late: Instruction text before query
        offset: Tokenization offset (1 for Mistral, 0 for others)
        num_layer: Number of transformer layers
        num_head: Number of attention heads per layer
        num_query: Counter for number of queries processed
        head_score: Dict mapping "{layer}-{head}" to accumulated scores
    """

    def __init__(self, llm_name) -> None:
        """
        Initialize the QR head detector.

        Args:
            llm_name: HuggingFace model name/path (e.g., 'mistralai/Mistral-7B-Instruct-v0.2')
        """
        # set up LLM
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(llm_name)

        if 'granite' in llm_name.lower():
            from .custom.modeling_granite_attn import GraniteForCausalLM
            BaseLLMClass = GraniteForCausalLM
        elif 'llama' in llm_name.lower():
            from .custom.modeling_llama_attn import LlamaForCausalLM
            BaseLLMClass = LlamaForCausalLM
        elif 'mistral' in llm_name.lower():
            from .custom.modeling_mistral_attn import MistralForCausalLM
            BaseLLMClass = MistralForCausalLM
        elif 'phi' in llm_name.lower():
            from .custom.modeling_phi_attn import Phi3ForCausalLM
            BaseLLMClass = Phi3ForCausalLM
        else:
            print(f'base model {llm_name} not supported')

        self.llm = BaseLLMClass.from_pretrained(
            llm_name,
            torch_dtype=torch.float16,
            attn_implementation='flash_attention_2',
            device_map='cuda'
        )

        # setup prompts
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

        # layer info
        self.num_query = 0
        self.num_layer = self.llm.config.num_hidden_layers
        self.num_head = self.llm.config.num_attention_heads
        self.head_score = {}
        for layer in range(self.num_layer):
            for head in range(self.num_head):
                self.head_score[f"{layer}-{head}"] = 0

    def get_head_score(self):
        """
        Get normalized head scores averaged over all queries.

        Divides accumulated scores by the number of queries processed to get
        average scores per head across all queries.

        Returns:
            dict: Mapping from "{layer}-{head}" to average score.
                  Higher scores indicate heads that pay more attention to
                  positive documents.
        """
        for head in self.head_score.keys():
            self.head_score[head] /= self.num_query
        return self.head_score

    def compute_retrieval_score(self, query, documents, pos_idx, neg_idx):
        """
        Compute and accumulate retrieval scores for all heads on one query.

        Processes a query with documents, computes attention scores to the
        positive document for each head, and accumulates them into head_score.

        Args:
            query: Query text string
            documents: List of document text strings
            pos_idx: Index of the positive document in documents list
            neg_idx: List of indices of hard negative documents (not used in QR,
                     but kept for API compatibility with CoRe detector)

        Side effects:
            Updates self.head_score by adding scores for this query
            Increments self.num_query counter
        """
        prompt, pos_span, query_span = self.prepare_input(query, documents, pos_idx, neg_idx)
        score = self.score_documents(prompt, pos_span, query_span)
        for layer in range(self.num_layer):
            for head in range(self.num_head):
                self.head_score[f"{layer}-{head}"] += score[layer, head].item()
        self.num_query += 1

    def score_documents(self, prompt, pos_span, query_span):
        """
        Score heads by their attention to the positive document.

        Runs a forward pass through the LLM to compute attention weights, then
        sums attention paid to the positive document tokens by each head.

        Args:
            prompt: Full formatted prompt string including documents and query
            pos_span: Tuple (start_idx, end_idx) of positive document tokens
            query_span: Tuple (start_idx, end_idx) of query tokens

        Returns:
            torch.Tensor: Attention scores of shape (num_layer, num_head).
                          Each value is the sum of attention weights to positive
                          document tokens from query tokens.
        """
        tokenized_input = self.tokenizer(prompt,return_tensors='pt').to(self.llm.device)
        _input_ids = tokenized_input.input_ids
        _query_indices = list(range(query_span[0], query_span[1]+1))
        kv_cache=DynamicCacheWithQuery(query_indices=_query_indices)

        with torch.no_grad():
            output = self.llm(
                input_ids=_input_ids,
                use_cache=True,
                past_key_values=kv_cache,
                output_attentions=True
                )
        kv_cache = output.past_key_values

        # loop through all layers and compute attention scores
        all_key_cache = []
        all_query_cache = []
        for i in range(self.num_layer):
            all_key_cache.append(kv_cache.key_cache[i][:,:,:query_span[1]+1])
            all_query_cache.append(kv_cache.query_cache[i])
        all_key_cache = torch.stack(all_key_cache)
        all_query_cache = torch.stack(all_query_cache)

        attn_weights = self._get_attn_weights(all_key_cache, all_query_cache).to('cuda').squeeze(1)
        attn_weights = attn_weights.mean(-2)

        # compute contrastive score
        pos_score = attn_weights[:,:,pos_span[0]:pos_span[1]].sum(-1)
        head_scores = pos_score.to('cpu')

        return head_scores

    def prepare_input(self, query, documents, pos_idx, neg_idx):
        """
        Prepare formatted prompt and compute token spans for positive document and query.

        Formats the prompt with model-specific prefixes/suffixes and computes
        token indices for the positive document and query. Unlike CoRe detector,
        neg_idx is not used in QR scoring.

        Args:
            query: Query text string
            documents: List of document text strings
            pos_idx: Index of the positive document in documents list
            neg_idx: List of indices of hard negative documents (not used in QR,
                     but kept for API compatibility)

        Returns:
            tuple: (llm_prompt, pos_span, query_span) where:
                - llm_prompt: Full formatted prompt string
                - pos_span: Tuple (start_idx, end_idx) of positive document tokens
                - query_span: Tuple (start_idx, end_idx) of query tokens
        """
        llm_prompt = self.prompt_prefix + self.retrieval_instruction

        for i, doc in enumerate(documents):
            llm_prompt += f'[document {i+1}]'
            start_len = len(self.tokenizer(llm_prompt).input_ids)

            llm_prompt += ' ' + doc
            end_len = len(self.tokenizer(llm_prompt).input_ids) - self.offset

            if i == pos_idx:
                pos_span = (start_len, end_len)
            llm_prompt += '\n\n'

        start_len = len(self.tokenizer(llm_prompt).input_ids)

        llm_prompt += self.retrieval_instruction_late + f'{query.strip()}'
        end_len = len(self.tokenizer(llm_prompt).input_ids) - self.offset
        llm_prompt += self.prompt_suffix

        query_span = (start_len, end_len)

        return llm_prompt, pos_span, query_span

    @classmethod
    def _get_attn_weights(cls, key_states, query_states):
        """
        Compute attention weights from key and query states.

        Implements the attention computation: softmax(QK^T / sqrt(d_k)) with causal masking.
        Supports Grouped Query Attention (GQA) where key/value heads may be fewer than
        query heads.

        Args:
            key_states: Cached key states, shape (num_layer, bsz, num_kv_heads, seq_len, head_dim)
            query_states: Cached query states, shape (num_layer, bsz, num_heads, q_len, head_dim)

        Returns:
            torch.Tensor: Attention weights of shape (num_layer, bsz, num_heads, q_len, seq_len).
                          Values are in [0, 1] and sum to 1 over the seq_len dimension.
        """
        num_layer, bsz, num_heads, q_len, head_dim = query_states.size()
        num_key_value_heads = key_states.size(2)
        num_key_value_groups = num_heads // num_key_value_heads
        kv_seq_len = key_states.size(-2)

        key_states = key_states.unsqueeze(3).expand(num_layer, bsz, num_key_value_heads, num_key_value_groups, kv_seq_len, head_dim)
        key_states = key_states.reshape(num_layer, bsz, num_heads, kv_seq_len, head_dim)
        attn_weights = torch.matmul(query_states, key_states.transpose(-2,-1)) / math.sqrt(head_dim)

        causal_mask = cls._get_causal_mask(attn_weights).to(attn_weights.device)
        attn_weights += causal_mask.unsqueeze(1)
        attn_lses = torch.logsumexp(attn_weights, dim=-1, keepdim=True)
        attn_weights = torch.exp(attn_weights - attn_lses)

        return attn_weights

    @classmethod
    def _get_causal_mask(cls, attn_weights):
        """
        Create causal attention mask preventing attention to future tokens.

        Generates a mask where valid positions are 0 and invalid (future) positions
        are set to a large negative value (min float) so they become ~0 after softmax.

        Args:
            attn_weights: Attention weight tensor, used only for shape and dtype.
                          Expected shape: (..., query_len, seq_len)

        Returns:
            torch.Tensor: Causal mask of same shape as attn_weights.
                          Valid positions are 0, invalid positions are -inf.
        """
        query_len, seq_len = attn_weights.size(-2), attn_weights.size(-1)
        causal_mask = torch.ones_like(attn_weights.transpose(-1,-2).squeeze(1))
        causal_mask = torch.triu(causal_mask, diagonal=-(seq_len-query_len))
        causal_mask = causal_mask.transpose(-1,-2)
        causal_mask = (1-causal_mask) * torch.finfo(causal_mask.dtype).min
        return causal_mask
