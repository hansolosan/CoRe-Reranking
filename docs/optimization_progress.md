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
- ✅ **Smart OOM retry logic** with progressive token reduction:
  - Automatically reduces `max_doc_tokens` from 300→200→150→...→10 on CUDA OOM
  - When doc tokens drop below 200, reduces query tokens if query > 100 tokens
  - Query reduction: half original length, then -50 increments
  - Temporary per-query reduction (doesn't affect other queries)
  - Comprehensive logging for problematic queries (token stats, GPU memory)
  - Data alignment validation when skipping queries
- ✅ Compressed input file support (`.gz`, `.bz2`)
- ✅ **vLLM backend support** (`--backend vllm`)
- ✅ **Batch extraction script** (`extract_features_batch.sh`) for parallel processing
- ✅ **Calibration support** (matches `reranker_calib.py` behavior):
  - Subtracts attention from "N/A" query to remove query-independent patterns
  - Enabled by default; use `--no_calibration` for raw attention features
  - Calibration status recorded in metadata JSON

**Supported backends:**
1. HuggingFace (default): `--backend hf` - Uses transformers with custom attention modules
2. vLLM offline: `--backend vllm` - Uses vLLM for efficient inference
3. vLLM server: `--backend vllm --vllm_url http://localhost:8000` - Connects to running vLLM server

**Supported input formats:**
1. Head detection data (`nq_core.json`): has `is_positive`/`is_negative` fields
2. Retriever output (`retriever_output/*.json`): uses qrels for labels

**OOM handling example:**
```
DEBUG: CUDA OOM on query 'test-free-speech-debate-ldhwbmclg-pro01a'
  Attempting with reduced tokens: doc=250, query=607
PROBLEMATIC: Query 'test-free-speech-debate-ldhwbmclg-pro01a' requires max_doc_tokens < 50
  Token stats: query=1214, docs_total=236, prompt_total=1636, num_docs=20
  GPU: 45.23GB allocated, 46.12GB reserved, 80.00GB total
```

### 4. Modular Trainer Architecture (`scripts/trainers.py`)
Modular trainer classes for easy loss function swapping:
- **BaseTrainer**: Abstract base class with common interface
- **BCETrainer**: Binary Cross-Entropy with L1 regularization (sklearn LogisticRegression)
- **InfoNCETrainer**: Contrastive loss for listwise ranking (proximal gradient descent)
- Factory function: `get_trainer(name, **kwargs)` returns trainer instance
- All trainers implement: `fit()`, `predict_proba()`, `predict()`, `evaluate()`, `get_weights()`

### 5. Head Weight Training (`scripts/train_head_weights_bce.py`)
- ✅ **Modular loss functions**: `--loss bce` or `--loss infonce`
- ✅ **Normalized loss**: `(1/n) * sum(loss) + λ * ||w||_1`
- ✅ **K-fold cross-validation**: `--cv 5` for 5-fold CV at base query level
- ✅ Parallel training support (`--n_jobs` flag, uses joblib)
- ✅ Configurable max iterations (`--max_iter`)
- ✅ Command logging in output JSON
- ✅ Numeric suffix to avoid overwriting files
- ✅ Saves all lambda results by default (`--save_best_only` to save only best)
- ✅ Direct feature file input (`--feature_file` / `-f`)
- ✅ Query-level train/val split (no data leakage between base queries)
- ✅ JSON-based query grouping (`--input_file` to load query variations)
- ✅ **Custom output template** (`--output` / `-o`) with placeholders: `{lambda}`, `{n}`, `{loss}`, `{llm}`

**Note on normalization**: The loss is normalized by the number of samples, making λ comparable across different dataset sizes.

**Note on data splitting**: Train/val split is done at the BASE QUERY level. The NQ data has 5 position variations per base query, and all variations stay together in the same split to prevent data leakage.

### 6. Evaluation Script (`scripts/rerank_with_head_weights.py`)
Computes ranking metrics on head detection data:
- **NDCG@k** (k=1,5,10)
- **Precision@k** (k=1,5,10)
- **Match@k** (hit rate: 1 if any relevant in top-k)
- **MAP** (Mean Average Precision)
- **MRR** (Mean Reciprocal Rank)
- Supports comparison of learned vs equal weights
- Works with both BCE and CoRe weight files
- ✅ **Baseline retriever evaluation** - Always computes baseline performance (original retriever ranking)
- ✅ **Oracle evaluation** - Computes upper bound performance (moves gold doc to rank 1 if present)
- ✅ **BEIR evaluator support** - `--evaluator beir` (default) uses BEIR's official metrics
- ✅ **External qrels support** - `--qrels` loads qrels file for proper NDCG computation
  - Uses full qrels (may contain docs not in retrieved set) for accurate IDCG
  - Validates query/doc ID matching between .npz and qrels files
  - Shows warnings with sample IDs when mismatches detected
- ✅ Multiple feature files (`-f file1.npz file2.npz ...`)
- ✅ **Enhanced color highlighting**:
  - Green (bold): Global maximum per metric column
  - Cyan (bold): Per-file maximum per metric column
  - Yellow (bold): Second-highest per file
  - Oracle results excluded from color highlighting (not compared with actual methods)
- ✅ Selectable metrics (`--metrics ndcg p m map mrr`)
- ✅ Optional JSON output (`--output` / `-o`)
- ✅ `--no_baseline` flag to skip baseline evaluation
- ✅ `--no_oracle` flag to skip oracle (upper bound) evaluation
- ✅ **Save ranked results** - `--save_ranked` outputs ranked document lists to `reranked_results/<llm>/k<k>/`
- ✅ **RRF Fusion** - `--fusion` combines baseline retriever and attention head rankings using Reciprocal Rank Fusion
  - Formula: `RRF(d) = Σ 1/(k + rank(d))` where k defaults to 60
  - Configurable via `--rrf_k` parameter
  - Results appear as `top-8+rrf` in output
- ✅ Uses shared utilities from `utils.py`

### 7. Feature Comparison (`scripts/compare_features.py`)
Compares two feature files to measure quantization effects:
- **Ranking metrics**: NDCG@k, P@k, Kendall's τ, Spearman's ρ, RBO, Top-1 match
- **Feature metrics**: MSE, RMSE, MAE, Pearson r, Cosine similarity
- **Per-head analysis**: Identifies worst-performing attention heads
- Useful for validating quantized model outputs against full precision

### 8. Head Correlation Analysis (`scripts/analyze_head_correlations.py`)
Analyzes correlations between attention heads relative to relevance:
- **Head-to-relevance correlation**: Point-biserial correlation with binary labels
- **Inter-head correlation matrix**: Pearson/Spearman between all head pairs
- **Conditional correlation**: Separate analysis for positive vs negative documents
- **Partial correlation**: Inter-head correlation controlling for relevance
- **Head clustering**: Hierarchical clustering based on correlation patterns
- **Diverse head selection**: Greedy algorithm balancing relevance + low redundancy
- ✅ Visualization plots (heatmaps, dendrograms, bar charts)
- ✅ JSON output for programmatic analysis

### 9. Analysis Scripts
- `scripts/compare_head_selection.py` - Compare CoRe vs BCE methods, compute AUC-ROC
- `scripts/analyze_sparsity.py` - Analyze sparsity across lambda values with plotting
- `scripts/get_top_heads.py` - Quick inspection of top heads from CoRe or BCE

### 10. BEIR Aggregate Evaluation (`scripts/evaluate_beir_aggregate.py`)
Computes aggregate BEIR scores across all 26 BEIR datasets:
- ✅ **Parallel execution** - Concurrent dataset evaluation with configurable workers (`--n_jobs`)
- ✅ **Dataset validation** - Ensures all 14 main BEIR datasets present
- ✅ **CQADupStack aggregation** - Averages 12 cqadupstack domains into single score
- ✅ **BEIR averaging** - Final score across 15 datasets (14 main + 1 cqadupstack)
- ✅ **Numerical sorting** - Results sorted by top-k value (baseline, then top-1, top-2, top-8...)
- ✅ **Color highlighting** - Green (best), yellow (second-best) per metric
- ✅ **Comprehensive output** - Individual JSON per dataset + aggregate results
- ✅ **Flexible cqadupstack** - Warns but continues if some domains missing
- ✅ **External qrels support** - `--beir_dir` loads qrels from BEIR directory structure
  - Loads from `{beir_dir}/{corpus}/qrels/test.tsv`
  - Supports both cqadupstack formats: `cqadupstack/android/` and `cqadupstack-android/`
  - BEIR evaluator is now the default for proper NDCG computation
- ✅ **Per-corpus breakdown table** - Shows baseline, oracle, and best config per dataset
  - `--display_metrics` selects which metrics to show (default: `NDCG@10`)
  - `--no_corpus_breakdown` skips the breakdown table
  - Row highlighting: green for best non-oracle value per row
- ✅ **RRF Fusion support** - `--fusion` and `--rrf_k` passed to reranking subprocess

**BEIR datasets (15 total after aggregation):**
- 14 main: trec-covid, nfcorpus, dbpedia-entity, scifact, scidocs, fiqa, nq, fever, climate-fever, hotpotqa, webis-touche2020, msmarco, quora, arguana
- 1 aggregated: cqadupstack (average of android, english, gaming, gis, mathematica, physics, programmers, stats, tex, unix, webmasters, wordpress)

**Usage example:**
```bash
python scripts/evaluate_beir_aggregate.py \
    --llm mistral \
    --weight_file head_data/mistral/bce_weights_lambda0.0001_n5000.json \
    --feature_dir head_data/mistral \
    --beir_dir /path/to/beir \
    --k 10 \
    --top_k_heads 1 2 4 8 16 32 \
    --n_jobs 8 \
    --output_dir results/beir_k10
```

**Output:**
```
======================================================================
BEIR Aggregate Results
======================================================================
Config               Weights    NDCG@1   NDCG@5   NDCG@10  MAP      MRR
----------------------------------------------------------------------
baseline             retriever  0.3245   0.4123   0.4567   0.3890   0.4234
oracle               gold@1     0.9756   0.9812   0.9845   0.9801   0.9823
top-1                bce        0.3312   0.4234   0.4678   0.3956   0.4345
top-8                bce        0.3456   0.4389   0.4823   0.4123   0.4567  <- green
top-16               bce        0.3512   0.4445   0.4891   0.4189   0.4623  <- yellow
======================================================================
```

**Note**: Oracle shows upper bound performance (gold document moved to rank 1 if present). Oracle results are excluded from color highlighting to avoid skewing comparisons between actual reranking methods.

**Per-corpus breakdown output:**
```
======================================================================
Per-Corpus Breakdown (k=40 documents)
======================================================================
Dataset              baseline      oracle  best (top-8)
------------------------------------------------------------
arguana                 0.412       0.981         0.439
climate-fever           0.345       0.976         0.356
dbpedia-entity          0.389       0.980         0.412
...
cqadupstack             0.456       0.984         0.482
------------------------------------------------------------
BEIR Average            0.423       0.982         0.457
```
- Rows show per-dataset performance for baseline, oracle, and best reranking config
- Best non-oracle value per row is highlighted in green
- `--display_metrics` controls which metrics are shown (can specify multiple)

### 11. Batch Feature Extraction (`scripts/extract_features_batch.sh`)
Shell script for batch feature extraction across multiple datasets and k values:
- ✅ **Parallel extraction** - Processes all combinations of input files × max_docs values
- ✅ **Conda environment management** - Automatic activation
- ✅ **Progress tracking** - Shows completed/skipped/failed jobs
- ✅ **Smart skipping** - Skips existing output files
- ✅ **OOM handling** - Supports `--max_query_tokens` for long queries
- ✅ **Flexible configuration** - All extract_head_features.py options supported

**Usage example:**
```bash
./scripts/extract_features_batch.sh \
    --conda core_env \
    --llm mistral \
    --qrels data/qrels/nq-test.tsv \
    --max_doc_tokens 300 \
    --max_query_tokens 500 \
    --max_docs 10 20 40 100 \
    --output_dir head_data/mistral \
    --files retriever_output/nq.json retriever_output/hotpotqa.json
```

**Output naming:** `{output_dir}/attention_features_{dataset}_k{max_docs}.npz`

### 12. Oracle Evaluation (Upper Bound Performance)
Added oracle evaluation mode to measure theoretical upper bound performance:

**What it does:**
- Moves the first gold (relevant) document to rank 1 if it exists in the retrieved set
- Shows the maximum possible performance achievable with perfect ranking
- Helps quantify the performance gap between current methods and theoretical maximum

**Implementation:**
- Added `use_oracle` parameter to `evaluate_ranking()` and `evaluate_ranking_beir()`
- Finds gold documents (label > 0) and assigns them the highest score
- Works with both custom and BEIR evaluators
- Supports both fixed and variable docs-per-query

**Display behavior:**
- Oracle results appear after baseline in evaluation output
- Excluded from color highlighting to avoid skewing method comparisons
- Sorted order: baseline → oracle → top-k methods (numerically)
- Can be disabled with `--no_oracle` flag

**Scripts updated:**
- `scripts/rerank_with_head_weights.py` - Added oracle evaluation mode
- `scripts/evaluate_beir_aggregate.py` - Added oracle support with `--no_oracle` flag

**Example output:**
```
File                           Config       Weights    NDCG@1   NDCG@5   NDCG@10
------------------------------------------------------------------------------
attention_features_nq_k10.npz  baseline     retriever  0.3245   0.4123   0.4567
                               oracle       gold@1     0.9812   0.9856   0.9892
                               top-8        bce        0.3456   0.4389   0.4823  <- green
                               top-16       bce        0.3512   0.4445   0.4891  <- yellow
```

**Interpretation:**
- Gap between baseline (0.3245) and oracle (0.9812) shows maximum possible improvement
- Gap between top-8 (0.3456) and oracle shows remaining headroom for improvement
- Oracle NDCG@1 < 1.0 means some queries don't have gold docs in retrieved set

## Results (Mistral, n=1000)

> **Note**: Results below were obtained with the **unnormalized** BCE loss formulation.
> With the current normalized loss `(1/n)*sum(BCE) + λ*||w||_1`, equivalent λ values are approximately `λ_new ≈ λ_old / n_samples` (where n_samples ≈ 40,000).

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
5. **Implement listwise cross-entropy** - ✅ InfoNCE implemented in `trainers.py`

## File Structure

```
CoRe-Reranking/
├── docs/
│   ├── head_selection_report.tex    # LaTeX report with equations
│   └── optimization_progress.md     # This document
├── scripts/
│   ├── utils.py                     # Shared utilities (feature loading, model configs, command logging)
│   ├── trainers.py                  # Modular trainer classes (BCE, InfoNCE)
│   ├── analyze_head_data.py         # Data analysis script
│   ├── extract_head_features.py     # Feature extraction (+ quantization, OOM retry)
│   ├── extract_features_batch.sh    # Batch feature extraction script
│   ├── train_head_weights_bce.py    # Head weight training (supports BCE/InfoNCE, CV)
│   ├── rerank_with_head_weights.py  # Ranking metrics evaluation (multi-file, baseline, oracle, BEIR)
│   ├── evaluate_beir_aggregate.py   # BEIR aggregate evaluation (parallel, oracle support)
│   ├── analyze_head_correlations.py # Head correlation analysis (+ clustering, plots)
│   ├── compare_features.py          # Compare feature files
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

# With max query tokens to handle long queries (OOM prevention)
CUDA_VISIBLE_DEVICES=0 python scripts/extract_head_features.py \
    --llm mistral \
    --input_file retriever_output/nq.json \
    --qrels path/to/qrels.tsv \
    --max_doc_tokens 300 \
    --max_query_tokens 500

# From compressed input file (.gz or .bz2)
CUDA_VISIBLE_DEVICES=0 python scripts/extract_head_features.py \
    --llm mistral \
    --input_file retriever_output/nq.json.gz \
    --max_samples 100

# With 4-bit quantization (reduces memory ~4x)
CUDA_VISIBLE_DEVICES=0 python scripts/extract_head_features.py \
    --llm mistral --quantize 4bit --max_samples 100

# Custom output name
CUDA_VISIBLE_DEVICES=0 python scripts/extract_head_features.py \
    --llm mistral --quantize 4bit -o my_features

# Using vLLM backend (offline mode)
CUDA_VISIBLE_DEVICES=0 python scripts/extract_head_features.py \
    --llm mistral --backend vllm --max_samples 100

# Using vLLM server mode (requires running vLLM server)
python scripts/extract_head_features.py \
    --llm mistral --backend vllm --vllm_url http://localhost:8000 \
    --max_samples 100

# Batch extraction across multiple datasets and k values
./scripts/extract_features_batch.sh \
    --conda core_env \
    --llm mistral \
    --qrels data/qrels/nq-test.tsv \
    --max_doc_tokens 300 \
    --max_query_tokens 500 \
    --max_docs 10 20 40 100 \
    --output_dir head_data/mistral \
    --files retriever_output/*.json
```

**Output naming:** `attention_features_{input}_{n}_{quantize}.npz`
- `attention_features_nq_core_n100.npz` (full precision)
- `attention_features_nq_core_n100_4bit.npz` (4-bit quantized)
- `attention_features_nq_k10.npz` (from batch script with max_docs=10)

**Memory usage (Mistral-7B):**
| Mode | VRAM |
|------|------|
| fp16 | ~14 GB |
| 8bit | ~8 GB |
| 4bit | ~5 GB |

**OOM handling:**
- Automatically reduces `max_doc_tokens` in 50-token increments on CUDA OOM
- When doc tokens < 200, reduces query tokens if query > 100 tokens
- Temporary per-query adjustment (doesn't affect other queries)
- Logs problematic queries with detailed token and memory statistics

### Train Head Weights

```bash
# BCE loss with parallel training (default)
python scripts/train_head_weights_bce.py --llm mistral \
    --lambda_l1 1e-5 1e-4 1e-3 1e-2 1e-1 \
    --n_jobs -1

# InfoNCE (contrastive) loss
python scripts/train_head_weights_bce.py --llm mistral \
    --loss infonce \
    --lambda_l1 1e-3 1e-2 1e-1 --n_jobs -1

# K-fold cross-validation (5-fold)
python scripts/train_head_weights_bce.py --llm mistral \
    --lambda_l1 1e-3 1e-2 1e-1 \
    --cv 5 --n_jobs -1

# Higher lambda for more sparsity
python scripts/train_head_weights_bce.py --llm mistral \
    --lambda_l1 0.1 0.5 1.0 2.0 --n_jobs -1

# Direct feature file input
python scripts/train_head_weights_bce.py --llm mistral \
    -f head_data/mistral/attention_features_custom.npz \
    --lambda_l1 1e-3 1e-2 1e-1 --n_jobs -1

# With query groupings from original JSON (for proper train/val split)
python scripts/train_head_weights_bce.py --llm mistral \
    --input_file head_data/nq_core.json \
    --lambda_l1 1e-3 1e-2 1e-1 --n_jobs -1

# Custom output template with placeholders
python scripts/train_head_weights_bce.py --llm mistral \
    --lambda_l1 1e-3 1e-2 1e-1 --cv 5 \
    --output "models/{llm}/weights_{loss}_lambda{lambda}.json"
```

**Note**: Lambda values are for the normalized loss `(1/n)*sum(loss) + λ*||w||_1`. Typical range: `1e-5` to `1.0`.

**Available loss functions**: `bce` (Binary Cross-Entropy, default), `infonce` (contrastive ranking loss)

### Evaluate Head Weights

```bash
# Evaluate BCE weights with ranking metrics (includes baseline and oracle)
python scripts/rerank_with_head_weights.py --llm mistral \
    --weight_file head_data/mistral/bce_weights_lambda100.0_n1000.json \
    --top_k_heads 8 16 32 --compare_equal

# Compare with CoRe heads (baseline and oracle computed automatically)
python scripts/rerank_with_head_weights.py --llm mistral \
    --weight_file head_data/mistral/core_temp0.001_prune0.0.json \
    --top_k_heads 8

# Skip baseline and oracle evaluation (reranked results only)
python scripts/rerank_with_head_weights.py --llm mistral \
    --weight_file head_data/mistral/bce_weights_lambda0.01_n1000.json \
    --top_k_heads 8 --no_baseline --no_oracle

# Use BEIR evaluator instead of custom evaluator
python scripts/rerank_with_head_weights.py --llm mistral \
    --weight_file head_data/mistral/bce_weights_lambda0.01_n1000.json \
    -f head_data/mistral/attention_features_nq_k10.npz \
    --top_k_heads 8 --evaluator beir

# Evaluate on multiple feature files (e.g., different k values)
python scripts/rerank_with_head_weights.py --llm mistral \
    --weight_file head_data/mistral/core_temp0.001_prune0.5.json \
    -f head_data/mistral/attention_features_nq_n100_k*.npz \
    --top_k_heads 8 --ks 1 3 5 10 100

# Save results to JSON file
python scripts/rerank_with_head_weights.py --llm mistral \
    --weight_file head_data/mistral/bce_weights_lambda0.01_n1000.json \
    -f head_data/mistral/features_train.npz head_data/mistral/features_test.npz \
    --top_k_heads 8 32 -o results/eval_metrics.json

# BEIR aggregate evaluation across all datasets
python scripts/evaluate_beir_aggregate.py \
    --llm mistral \
    --weight_file head_data/mistral/bce_weights_lambda0.0001_n5000.json \
    --feature_dir head_data/mistral \
    --k 10 \
    --top_k_heads 1 2 4 8 16 32 \
    --n_jobs 8 \
    --output_dir results/beir_k10

# With RRF fusion (combines retriever + attention head rankings)
python scripts/rerank_with_head_weights.py --llm mistral \
    --weight_file head_data/mistral/bce_weights_lambda0.01_n1000.json \
    -f head_data/mistral/attention_features_nq_k40.npz \
    --top_k_heads 8 16 \
    --fusion --rrf_k 60

# BEIR aggregate with fusion and per-corpus breakdown
python scripts/evaluate_beir_aggregate.py \
    --llm mistral \
    --weight_file head_data/mistral/bce_weights_lambda0.0001_n5000.json \
    --feature_dir head_data/mistral \
    --k 40 \
    --top_k_heads 8 16 \
    --fusion \
    --display_metrics NDCG@1 NDCG@5 NDCG@10 \
    --output_dir results/beir_fusion
```

**Evaluation modes:**
- **Baseline**: Original retriever ranking (no reranking)
- **Oracle**: Upper bound performance (gold document at rank 1 if present in top-k)
- **Reranking**: Using learned weights with various top-k head configurations

**Output color coding:**
- **Green (bold)**: Global maximum per metric column
- **Cyan (bold)**: Per-file maximum per metric column
- **Yellow (bold)**: Second-highest per file
- **Oracle results**: Displayed but excluded from color highlighting (not compared with actual methods)

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

### Analyze Head Correlations

```bash
# Basic correlation analysis
python scripts/analyze_head_correlations.py --llm mistral --num_samples 1000

# With plots and JSON output
python scripts/analyze_head_correlations.py --llm mistral \
    -f head_data/mistral/attention_features_n1000.npz \
    --plot --output head_data/mistral/correlation_analysis.json

# Spearman correlation and more diverse heads
python scripts/analyze_head_correlations.py --llm mistral \
    --method spearman --diverse_k 16 --plot
```

**Generated plots** (in `head_data/{llm}/plots/`):
- `inter_head_corr_pearson.png` - Full correlation matrix heatmap
- `head_relevance_corr.png` - Bar chart of head-to-relevance correlations
- `head_clustering_dendrogram.png` - Hierarchical clustering tree
- `conditional_corr_diff.png` - Positive vs negative document correlations
- `partial_corr.png` - Correlation after controlling for relevance

### Analyze Sparsity

```bash
# Analyze sparsity-performance tradeoff
python scripts/analyze_sparsity.py --llm mistral --plot
```

## References

- Tran, L., Li, Y., Florian, R., & Sun, W. (2025). Less is More: Contrastive Retrieval Heads Improve Attention-Based Re-Ranking. arXiv:2510.02219
- Chen et al. (2025). In-Context Reranking (ICR)
