#!/usr/bin/env python3
"""
Utility functions shared across scripts.
"""

import sys
import getpass
import numpy as np
from pathlib import Path
from datetime import datetime


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


def load_features(feature_file=None, llm_name=None, num_samples=None):
    """
    Load extracted attention features from .npz file.

    Args:
        feature_file: Path to feature file (.npz). If provided, llm_name and num_samples are ignored.
        llm_name: LLM name for default path construction
        num_samples: Number of samples for default path construction

    Returns:
        features: (n_docs, n_heads) attention features array
        labels: (n_docs,) binary labels array
        docs_per_query: (n_queries,) array of docs per query, or None if not available
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
