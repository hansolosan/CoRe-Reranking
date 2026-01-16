# Head Weight Optimization Progress Report

## Goal

Implement learned head weighting optimization for the CoRe (Contrastive Retrieval) reranking system. Instead of using greedy selection of top-k heads with equal weights, we aim to learn optimal weights for attention heads using:

1. **Binary Cross-Entropy (BCE) Loss** with L1 regularization (Lasso) for sparse head selection
2. **Listwise Cross-Entropy Loss** (InfoNCE) as an alternative that better matches the contrastive nature of ranking

The optimization should:
- Learn which heads are most important for retrieval
- Automatically determine the optimal number of heads via L1 sparsity
- Potentially improve reranking performance over greedy selection

## Background

The CoRe paper identifies retrieval heads using a contrastive scoring metric (SCoRe):

```
SCoRe(h) = exp(s_pos/t) / (exp(s_pos/t) + Σ exp(s_neg/t))
```

Currently, heads are selected greedily by:
1. Computing average SCoRe across 5000 detection samples
2. Sorting heads by score
3. Selecting top-k heads (typically k=8)

## Work Completed

### 1. LaTeX Report (`docs/head_selection_report.tex`)
- Formal description of CoRe head selection with equations
- Proposed BCE optimization with logistic regression
- Proposed listwise cross-entropy (InfoNCE) loss
- L1/Elastic Net/Group Lasso regularization strategies
- Experimental plan
- Citation to CoRe paper (Tran et al., arXiv:2510.02219)

### 2. Data Analysis (`scripts/analyze_head_data.py`)
Analyzed the head detection data:
- **5,000 questions** (1,000 base queries × 5 position variations)
- **50 paragraphs per question** (1 positive + 49 hard negatives)
- Data source: NQ training set with hard negatives from granite-embedding-30m-english

### 3. Feature Extraction Script (`scripts/extract_head_features.py`)
Created script to extract attention features from all heads:
- Loads LLM with custom attention modules
- Processes head detection data
- Extracts per-document attention scores for each head
- Saves features as numpy arrays for optimization

### 4. BCE Training Script (`scripts/train_head_weights_bce.py`)
Created script to train logistic regression with L1 regularization:
- Loads extracted features
- Trains with multiple λ values: [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
- Reports metrics: accuracy, precision, recall, F1, AUC-ROC, sparsity
- Compares learned heads with original CoRe heads
- Saves results to JSON

## Current Blockers

### Feature Extraction Runtime Error
The `extract_head_features.py` script encounters a compatibility issue:

```
AttributeError: 'DynamicCacheWithQuery' object has no attribute '_seen_tokens'
```

**Cause**: The custom `DynamicCacheWithQuery` class in `experiments/src/custom/custom_cache.py` inherits from `DynamicCache` but the parent class API has changed in newer versions of transformers. The `_seen_tokens` attribute initialization is missing.

**Fix needed**: Update `custom_cache.py` to initialize `_seen_tokens` in `__init__`:
```python
def __init__(self, query_indices=[]) -> None:
    super().__init__()
    self._seen_tokens = 0  # Add this line
    self._query_indices = query_indices
    self.query_cache = []
```

## Next Steps

1. **Fix the cache compatibility issue** - Update `custom_cache.py` to work with current transformers version

2. **Run feature extraction** - Extract attention features for 1000 samples:
   ```bash
   python scripts/extract_head_features.py --llm mistral --max_samples 1000
   ```

3. **Train BCE model** - Run optimization with various L1 penalties:
   ```bash
   python scripts/train_head_weights_bce.py --llm mistral --num_samples 1000
   ```

4. **Implement listwise cross-entropy** - Create `train_head_weights_ce.py` with InfoNCE loss

5. **Evaluate on BEIR** - Compare learned weights vs greedy selection on benchmark datasets

6. **Analyze results** - Compare:
   - Which heads are selected by each method
   - Sparsity patterns across layers
   - Reranking performance (NDCG@1/5/10)

## File Structure

```
CoRe-Reranking/
├── docs/
│   ├── head_selection_report.tex    # LaTeX report with equations
│   └── optimization_progress.md     # This document
├── scripts/
│   ├── analyze_head_data.py         # Data analysis script
│   ├── extract_head_features.py     # Feature extraction from LLM
│   └── train_head_weights_bce.py    # BCE + L1 optimization
├── head_data/
│   ├── nq_core.json                 # Detection data (5000 samples)
│   └── {llm}/
│       ├── core_temp*.json          # Pre-computed CoRe head scores
│       └── attention_features_*.npz # Extracted features (to be created)
└── experiments/
    └── src/
        └── custom/
            └── custom_cache.py      # Needs fix for _seen_tokens
```

## References

- Tran, L., Li, Y., Florian, R., & Sun, W. (2025). Less is More: Contrastive Retrieval Heads Improve Attention-Based Re-Ranking. arXiv:2510.02219
- Chen et al. (2025). In-Context Reranking (ICR)
