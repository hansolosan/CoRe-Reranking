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

### 3. Feature Extraction (`scripts/extract_head_features.py`)
Comprehensive feature extraction with multiple input/output options:

**Features:**
- ✅ Fixed `DynamicCacheWithQuery` compatibility issue
- ✅ Support for multiple input formats (head detection data, retriever output)
- ✅ TREC qrels file support for relevance labels (`--qrels`)
- ✅ 4-bit and 8-bit quantization support (`--quantize`)
- ✅ Custom input/output paths (`--input_file`, `-o`)
- ✅ Quantization mode in output filename
- ✅ Metadata JSON file with configuration details
- ✅ OOM error handling with automatic skip and cache clearing

**Supported input formats:**
1. Head detection data (`nq_core.json`): has `is_positive`/`is_negative` fields
2. Retriever output (`retriever_output/*.json`): uses qrels for labels

### 4. BCE Training (`scripts/train_head_weights_bce.py`)
- ✅ Trained logistic regression with L1 regularization
- ✅ Parallel training support (`--n_jobs` flag, uses joblib)
- ✅ Configurable max iterations (`--max_iter`)
- ✅ Command logging in output JSON
- ✅ Numeric suffix to avoid overwriting files
- ✅ `--save_all` flag to save all lambda results

### 5. Evaluation Script (`scripts/evaluate_head_weights.py`)
Computes ranking metrics on head detection data:
- **NDCG@k** (k=1,5,10)
- **Precision@k** (k=1,5,10)
- **MAP** (Mean Average Precision)
- **MRR** (Mean Reciprocal Rank)
- Supports comparison of learned vs equal weights
- Works with both BCE and CoRe weight files

### 6. Feature Comparison (`scripts/compare_features.py`) — NEW
Compares two feature files to measure quantization effects:
- **Ranking metrics**: NDCG@k, P@k, Kendall's τ, Spearman's ρ, RBO, Top-1 match
- **Feature metrics**: MSE, RMSE, MAE, Pearson r, Cosine similarity
- **Per-head analysis**: Identifies worst-performing attention heads
- Useful for validating quantized model outputs against full precision

### 7. Analysis Scripts
- `scripts/compare_head_selection.py` - Compare CoRe vs BCE methods, compute AUC-ROC
- `scripts/analyze_sparsity.py` - Analyze sparsity across lambda values with plotting
- `scripts/get_top_heads.py` - Quick inspection of top heads from CoRe or BCE

## Results (Mistral, n=1000)

### Sparsity vs Performance (Parallel Training, 6 jobs, ~6.6 min)

| Lambda | Non-zero Heads | Sparsity | AUC-ROC |
|--------|---------------|----------|---------|
| 1.5 | 868 | 15.2% | 0.9995 |
| 2.0 | 811 | 20.8% | 0.9995 |
| 5.0 | 640 | 37.5% | 0.9993 |
| 10.0 | 479 | 53.2% | 0.9991 |
| 50.0 | 173 | 83.1% | 0.9986 |
| **100.0** | **105** | **89.8%** | **0.9980** |

### Comparison with CoRe

| Method | Heads Used | AUC-ROC |
|--------|-----------|---------|
| CoRe Top 8 (equal weights) | 8 | 0.9871 |
| BCE λ=100 (learned weights) | 105 | 0.9980 |

**Key insight**: Even at 90% sparsity (105 heads), BCE significantly outperforms CoRe's 8 heads (+1.1% AUC-ROC).

### Top 20 BCE Heads (λ=2.0)

| Rank | Layer | Head | Weight |
|------|-------|------|--------|
| 1 | 12 | 11 | 0.567 |
| 2 | 9 | 2 | **-0.500** |
| 3 | 9 | 26 | 0.500 |
| 4 | 15 | 7 | 0.500 |
| 5 | 12 | 10 | 0.499 |
| 6 | 16 | 13 | **-0.462** |
| 7 | 7 | 18 | 0.449 |
| 8 | 15 | 21 | 0.401 |

**Negative weights** (bolded) indicate anti-correlated heads — high attention suggests document is NOT relevant.

### Key Findings

1. **BCE learns negative weights** - Some heads anti-correlate with relevance, which CoRe cannot capture
2. **More heads help** - 105 heads with learned weights outperform 8 heads with equal weights
3. **Sparsity-performance tradeoff** - Can achieve 90% sparsity with only 0.15% AUC drop
4. **Parallel training works** - 6 lambda values trained in ~6.6 minutes with 6 workers

### Evaluation Note

Current AUC-ROC metrics are computed on a **20% held-out validation split** from the NQ head detection data. This is in-distribution evaluation. True generalization should be tested on out-of-distribution BEIR datasets.

## Next Steps

1. **Evaluate with ranking metrics** on head detection data
2. **Try higher λ values** to approach 8 heads
3. **Integrate with reranker** - Modify `reranking.py` to use learned BCE weights
4. **Evaluate on BEIR** - Test generalization on out-of-distribution datasets
5. **Implement listwise cross-entropy** - Create `train_head_weights_ce.py` with InfoNCE loss

## File Structure

```
CoRe-Reranking/
├── docs/
│   ├── head_selection_report.tex    # LaTeX report with equations
│   └── optimization_progress.md     # This document
├── scripts/
│   ├── analyze_head_data.py         # Data analysis script
│   ├── extract_head_features.py     # Feature extraction (+ quantization)
│   ├── train_head_weights_bce.py    # BCE + L1 optimization (parallel)
│   ├── evaluate_head_weights.py     # Ranking metrics evaluation
│   ├── compare_features.py          # Compare feature files (NEW)
│   ├── compare_head_selection.py    # Compare methods (AUC-ROC)
│   ├── analyze_sparsity.py          # Sparsity analysis with plotting
│   └── get_top_heads.py             # Quick head inspection
├── head_data/
│   ├── nq_core.json                 # Detection data (5000 samples)
│   └── mistral/
│       ├── core_temp0.001_prune0.0.json      # Pre-computed CoRe head scores
│       ├── attention_features_*.npz          # Extracted features
│       ├── bce_weights_lambda*.json          # BCE results for various λ
│       └── *.meta.json                       # Feature extraction metadata
└── experiments/
    └── src/
        └── custom/
            └── custom_cache.py      # Fixed ✅
```

## Usage Examples

### Feature Extraction

```bash
# Basic extraction from head detection data
CUDA_VISIBLE_DEVICES=0 python scripts/extract_head_features.py \
    --llm mistral --max_samples 1000

# From retriever output with qrels labels
CUDA_VISIBLE_DEVICES=0 python scripts/extract_head_features.py \
    --llm mistral \
    --input_file retriever_output/nq.json \
    --qrels path/to/qrels.tsv \
    --max_samples 100

# With 4-bit quantization (reduces memory ~4x)
CUDA_VISIBLE_DEVICES=0 python scripts/extract_head_features.py \
    --llm mistral --quantize 4bit --max_samples 100

# Custom output name
CUDA_VISIBLE_DEVICES=0 python scripts/extract_head_features.py \
    --llm mistral --quantize 4bit -o my_features
```

**Output naming:** `attention_features_{input}_{n}_{quantize}.npz`
- `attention_features_nq_core_n100.npz` (full precision)
- `attention_features_nq_core_n100_4bit.npz` (4-bit quantized)

**Memory usage (Mistral-7B):**
| Mode | VRAM |
|------|------|
| fp16 | ~14 GB |
| 8bit | ~8 GB |
| 4bit | ~5 GB |

### Train BCE Weights

```bash
# Parallel training with multiple lambda values
python scripts/train_head_weights_bce.py --llm mistral \
    --lambda_l1 1.0 5.0 10.0 50.0 100.0 \
    --n_jobs -1 --save_all

# Higher lambda for more sparsity
python scripts/train_head_weights_bce.py --llm mistral \
    --lambda_l1 200 500 1000 2000 --n_jobs -1 --save_all
```

### Evaluate Head Weights

```bash
# Evaluate BCE weights with ranking metrics
python scripts/evaluate_head_weights.py --llm mistral \
    --weight_file head_data/mistral/bce_weights_lambda100.0_n1000.json \
    --top_k_heads 8 16 32 --compare_equal

# Compare with CoRe heads
python scripts/evaluate_head_weights.py --llm mistral \
    --weight_file head_data/mistral/core_temp0.001_prune0.0.json \
    --top_k_heads 8
```

### Compare Quantized vs Full Precision

```bash
# Measure quantization effects on features
python scripts/compare_features.py \
    --reference head_data/mistral/attention_features_nq_core_n100.npz \
    --system head_data/mistral/attention_features_nq_core_n100_4bit.npz \
    -o comparison_results.json

# With specific head weights for scoring
python scripts/compare_features.py \
    -r features_fp16.npz -s features_4bit.npz \
    -w head_data/mistral/bce_weights_lambda100.0_n1000.json
```

### Analyze Sparsity

```bash
# Analyze sparsity-performance tradeoff
python scripts/analyze_sparsity.py --llm mistral --plot
```

## References

- Tran, L., Li, Y., Florian, R., & Sun, W. (2025). Less is More: Contrastive Retrieval Heads Improve Attention-Based Re-Ranking. arXiv:2510.02219
- Chen et al. (2025). In-Context Reranking (ICR)
