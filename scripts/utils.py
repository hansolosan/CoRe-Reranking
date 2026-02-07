#!/usr/bin/env python3
"""
Utility functions shared across scripts.
"""

import sys
import getpass
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# Model configurations: (num_layers, num_heads_per_layer)
MODEL_CONFIGS = {
    'mistral': (32, 32),   # 32 layers, 32 heads
    'llama': (32, 32),     # 32 layers, 32 heads
    'phi': (40, 40),       # 40 layers, 40 heads
    'granite': (40, 32),   # 40 layers, 32 heads
}


def get_head_info(llm_name):
    """
    Get number of layers and heads for a model.

    Args:
        llm_name: Model name ('mistral', 'llama', 'phi', 'granite')

    Returns:
        (num_layers, num_heads_per_layer)
    """
    return MODEL_CONFIGS.get(llm_name, (32, 32))


def load_features(feature_file=None, llm_name=None, num_samples=None, return_ids=False):
    """
    Load extracted attention features from .npz file.

    Args:
        feature_file: Path to feature file (.npz). If provided, llm_name and num_samples are ignored.
        llm_name: LLM name for default path construction
        num_samples: Number of samples for default path construction
        return_ids: If True, also return query_ids and doc_ids

    Returns:
        features: (n_docs, n_heads) attention features array
        labels: (n_docs,) binary labels array
        docs_per_query: (n_queries,) array of docs per query, or None if not available
        query_ids: (n_docs,) query IDs (only if return_ids=True)
        doc_ids: (n_docs,) document IDs (only if return_ids=True)
    """
    if feature_file is not None:
        path = Path(feature_file)
    else:
        if llm_name is None:
            raise ValueError("Either feature_file or llm_name must be provided")
        path = Path(__file__).parent.parent / 'head_data' / llm_name / f'attention_features_n{num_samples}.npz'

    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    data = np.load(path, allow_pickle=True)

    features = data['features']
    labels = data['labels']

    # Get docs_per_query if available
    docs_per_query = None
    if 'docs_per_query' in data:
        docs_per_query = data['docs_per_query']

    if return_ids:
        # Get query_ids and doc_ids if available
        query_ids = data.get('query_ids', None)
        doc_ids = data.get('doc_ids', None)

        # Generate default IDs if not available
        if query_ids is None:
            if docs_per_query is not None:
                # Generate query IDs based on docs_per_query
                query_ids = []
                for q_idx, n_docs in enumerate(docs_per_query):
                    query_ids.extend([f'q{q_idx}'] * n_docs)
                query_ids = np.array(query_ids, dtype=object)
            else:
                # Assume fixed docs_per_query (e.g., 50)
                query_ids = np.array([f'q{i // 50}' for i in range(len(labels))], dtype=object)

        if doc_ids is None:
            # Generate sequential doc IDs
            doc_ids = np.array([f'd{i}' for i in range(len(labels))], dtype=object)

        return features, labels, docs_per_query, query_ids, doc_ids

    return features, labels, docs_per_query


def log_command(logfile='logfile'):
    """
    Log the command execution to a file.

    Args:
        logfile: Name of the log file (default: 'logfile')
                 File is created in the repository root directory.

    Log format: YYYY-MM-DD HH:MM:SS | username | full command
    """
    try:
        username = getpass.getuser()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if '-h' in sys.argv or '--help' in sys.argv:
            # Don't print anything if the user's looking for help
            return
        command = ' '.join(sys.argv)

        log_entry = f"{timestamp} | {username} | python {command}\n"

        logfile_path = Path(__file__).parent.parent / logfile
        with open(logfile_path, 'a') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Warning: Could not write to logfile: {e}", file=sys.stderr)


def parse_args_with_config(parser, args=None):
    """
    Parse arguments with optional YAML config file support.

    If --config is provided, loads options from the YAML file as defaults.
    Command-line arguments override YAML values.

    Args:
        parser: An argparse.ArgumentParser instance (should already have arguments added)
        args: Arguments to parse (default: sys.argv[1:])

    Returns:
        Parsed namespace object

    Usage:
        parser = argparse.ArgumentParser()
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--batch_size', type=int, default=32)
        # --config is added automatically
        args = parse_args_with_config(parser)

    YAML config file format:
        learning_rate: 0.01
        batch_size: 64
    """
    if not YAML_AVAILABLE:
        print("Warning: PyYAML not installed. Config file support disabled.", file=sys.stderr)
        return parser.parse_args(args)

    # Add --config argument if not already present
    config_action = None
    for action in parser._actions:
        if '--config' in action.option_strings:
            config_action = action
            break

    if config_action is None:
        parser.add_argument('--config', type=str, default=None,
                            help='Path to YAML config file. CLI args override config values.')

    if args is None:
        args = sys.argv[1:]

    # First pass: extract just the config file path
    # Use parse_known_args to avoid errors from other required arguments
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, default=None)
    config_args, _ = config_parser.parse_known_args(args)

    # If config file specified, load it and set as defaults
    if config_args.config is not None:
        config_path = Path(config_args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            config = {}

        # Get valid argument names from parser
        valid_args = set()
        for action in parser._actions:
            valid_args.update(action.option_strings)
            if action.dest != 'help':
                valid_args.add(action.dest)

        # Filter config to only include valid arguments
        filtered_config = {}
        for key, value in config.items():
            # Convert underscores to match argparse dest names
            dest_key = key.replace('-', '_')
            if dest_key in valid_args or f'--{key}' in valid_args or f'--{dest_key}' in valid_args:
                filtered_config[dest_key] = value
            else:
                print(f"Warning: Unknown config key '{key}' ignored", file=sys.stderr)

        # Set defaults from config file
        parser.set_defaults(**filtered_config)

    # Second pass: parse all arguments (CLI overrides config defaults)
    return parser.parse_args(args)


def load_cross_encoder(model_name, device=None, trust_remote_code=True, verbose=True, model_kwargs=None):
    """
    Load a CrossEncoder model with proper padding token configuration.

    Some models (e.g., jina-reranker) don't have a padding token defined,
    which causes errors with batch_size > 1. This function sets pad_token
    and pad_token_id to eos_token/eos_token_id if not defined.

    Args:
        model_name: HuggingFace model name or path
        device: Device to load model on ('cuda', 'cpu', or None for auto)
        trust_remote_code: Whether to trust remote code (required for some models)
        verbose: Whether to print status messages
        model_kwargs: Additional keyword arguments to pass to CrossEncoder constructor

    Returns:
        CrossEncoder model instance
    """
    if model_kwargs is None:
        model_kwargs = {}
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Install with: pip install sentence-transformers"
        )

    import torch
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CrossEncoder(model_name, device=device, trust_remote_code=trust_remote_code, model_kwargs=model_kwargs)

    # Set pad_token and pad_token_id if not defined (required for batch_size > 1)
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
        if verbose:
            print(f"Set tokenizer pad_token to eos_token for batched inference")

    # Also set pad_token_id in model config (some models check this)
    if model.model.config.pad_token_id is None:
        model.model.config.pad_token_id = model.tokenizer.pad_token_id
        if verbose:
            print(f"Set model.config.pad_token_id to {model.tokenizer.pad_token_id}")

    return model


# =============================================================================
# Reranker Class Hierarchy
# =============================================================================

class BaseReranker:
    """Abstract base class for rerankers."""

    def __init__(self, model_name, device=None, verbose=True, model_kwargs=None):
        self.model_name = model_name
        self.verbose = verbose
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

        import torch
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

    def predict(self, pairs, batch_size=32):
        """
        Compute relevance scores for query-document pairs.

        Args:
            pairs: List of (query, document) tuples
            batch_size: Batch size for inference

        Returns:
            numpy array of scores, one per pair
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(model_name='{self.model_name}', device='{self.device}')"


class CrossEncoderReranker(BaseReranker):
    """Reranker using sentence-transformers CrossEncoder."""

    def __init__(self, model_name, device=None, trust_remote_code=True, verbose=True, model_kwargs=None):
        super().__init__(model_name, device, verbose, model_kwargs)

        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name,
                                  device=self.device,
                                  trust_remote_code=trust_remote_code,
                                  model_kwargs=self.model_kwargs)

        # Set pad_token and pad_token_id if not defined (required for batch_size > 1)
        if self.model.tokenizer.pad_token is None:
            self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
            self.model.tokenizer.pad_token_id = self.model.tokenizer.eos_token_id
            if verbose:
                print(f"Set tokenizer pad_token to eos_token for batched inference")

        # Also set pad_token_id in model config (some models check this)
        if self.model.model.config.pad_token_id is None:
            self.model.model.config.pad_token_id = self.model.tokenizer.pad_token_id
            if verbose:
                print(f"Set model.config.pad_token_id to {self.model.tokenizer.pad_token_id}")

    def predict(self, pairs, batch_size=32):
        return self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)


class BiEncoderReranker(BaseReranker):
    """Reranker using sentence-transformers SentenceTransformer (bi-encoder)."""

    def __init__(self, model_name, device=None, verbose=True, model_kwargs=None):
        super().__init__(model_name, device, verbose, model_kwargs)

        from sentence_transformers import SentenceTransformer
        import numpy as np

        self.model = SentenceTransformer(model_name, device=self.device, model_kwargs=self.model_kwargs)
        self.np = np

    def predict(self, pairs, batch_size=32):
        queries = [p[0] for p in pairs]
        docs = [p[1] for p in pairs]

        # Encode queries and documents
        query_embs = self.model.encode(queries, batch_size=batch_size, show_progress_bar=False)
        doc_embs = self.model.encode(docs, batch_size=batch_size, show_progress_bar=False)

        # Compute cosine similarity
        query_norms = query_embs / self.np.linalg.norm(query_embs, axis=1, keepdims=True)
        doc_norms = doc_embs / self.np.linalg.norm(doc_embs, axis=1, keepdims=True)
        scores = (query_norms * doc_norms).sum(axis=1)

        return scores


class JinaReranker(BaseReranker):
    """Reranker for Jina reranker models (v3+) that use AutoModel with built-in rerank method."""

    def __init__(self, model_name, device=None, trust_remote_code=True, verbose=True,
                 max_doc_length=2048, max_query_length=512, model_kwargs=None):
        super().__init__(model_name, device, verbose, model_kwargs)

        from transformers import AutoModel
        import numpy as np

        self.np = np
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length

        if verbose:
            print(f"Loading Jina reranker: {model_name}")

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype='auto',
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, pairs, batch_size=32):
        """
        Compute relevance scores for query-document pairs.

        Note: Jina reranker v3 uses its own rerank() method which processes
        one query at a time with multiple documents. The batch_size parameter
        is ignored since the model doesn't support batched decoding in the
        traditional sense.

        Args:
            pairs: List of (query, document) tuples
            batch_size: Ignored for Jina reranker (kept for API compatibility)

        Returns:
            numpy array of scores, one per pair
        """
        import numpy as np

        # Group pairs by query
        query_to_docs = {}
        query_to_indices = {}
        for idx, (query, doc) in enumerate(pairs):
            if query not in query_to_docs:
                query_to_docs[query] = []
                query_to_indices[query] = []
            query_to_docs[query].append(doc)
            query_to_indices[query].append(idx)

        # Process each query
        scores = np.zeros(len(pairs))
        for query, docs in query_to_docs.items():
            results = self.model.rerank(
                query, docs,
                max_doc_length=self.max_doc_length,
                max_query_length=self.max_query_length
            )
            # Map scores back to original indices
            indices = query_to_indices[query]
            for result in results:
                original_idx = indices[result['index']]
                scores[original_idx] = result['relevance_score']

        return scores


# Models that require specific reranker classes
JINA_RERANKER_MODELS = [
    'jina-reranker-v3',
]


def detect_reranker_type(model_name):
    """
    Detect the appropriate reranker type for a model.

    Args:
        model_name: HuggingFace model name

    Returns:
        str: 'jina', 'cross-encoder', or 'bi-encoder'
    """
    model_lower = model_name.lower()

    # Check for Jina reranker models
    for pattern in JINA_RERANKER_MODELS:
        if pattern in model_lower:
            return 'jina'

    # Check for cross-encoder patterns
    cross_encoder_patterns = ['cross-encoder', 'cross_encoder', 'rerank', 'bge-reranker']
    if any(p in model_lower for p in cross_encoder_patterns):
        return 'cross-encoder'

    # Default to bi-encoder
    return 'bi-encoder'


def load_reranker(model_name, device=None, trust_remote_code=True, verbose=True,
                  max_length=512, max_doc_length=2048, max_query_length=512, model_kwargs=None):
    """
    Load the appropriate reranker for a model.

    Automatically detects whether to use CrossEncoder, JinaReranker,
    or BiEncoder based on the model name.

    Args:
        model_name: HuggingFace model name or path
        device: Device to load model on ('cuda', 'cpu', or None for auto)
        trust_remote_code: Whether to trust remote code
        verbose: Whether to print status messages
        max_length: Max sequence length (for CrossEncoder)
        max_doc_length: Max document length (for JinaReranker)
        max_query_length: Max query length (for JinaReranker)
        model_kwargs: Additional keyword arguments to pass to model constructors

    Returns:
        BaseReranker instance
    """
    reranker_type = detect_reranker_type(model_name)

    if verbose:
        print(f"Detected reranker type: {reranker_type}")

    if reranker_type == 'jina':
        return JinaReranker(
            model_name, device=device, trust_remote_code=trust_remote_code,
            verbose=verbose, max_doc_length=max_doc_length, max_query_length=max_query_length,
            model_kwargs=model_kwargs
        )
    elif reranker_type == 'cross-encoder':
        return CrossEncoderReranker(
            model_name, device=device, trust_remote_code=trust_remote_code, verbose=verbose,
            model_kwargs=model_kwargs
        )
    else:
        return BiEncoderReranker(model_name, device=device, verbose=verbose, model_kwargs=model_kwargs)
