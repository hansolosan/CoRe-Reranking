# Head Configuration Comparison Guide

This guide explains how to compare two attention head configurations using statistical significance testing.

## Overview

The `compare_head_configs.py` script compares two head weight configurations by:

1. Loading attention features directly from .npz files
2. Computing per-query NDCG scores for both configurations
3. Performing proper randomization tests on per-query scores for statistical significance

## Statistical Significance Testing

The script implements a **randomization test** (permutation test) on **per-query scores**, which is the gold standard for statistical significance testing in information retrieval.

### How it Works

1. **Per-query evaluation**: For each query in each dataset, compute the metric (e.g., NDCG@10) for both configurations
2. **Observed difference**: Calculate the mean difference in per-query scores between the two systems
3. **Random permutations**: For each query, randomly swap which system's score is labeled as "system 1" vs "system 2"
4. **Recompute difference**: Calculate the mean difference for each random permutation
5. **P-value**: The proportion of permutations where |permuted_diff| ≥ |observed_diff|

### Macro vs Micro Averages

- **Significance test**: Uses all per-query scores pooled together (proper statistical power)
- **Displayed averages**: Macro averages (average of per-dataset means)
  - CQADupstack: Average of per-domain means
  - BEIR: Average of per-dataset means (14 main + cqadupstack as one entry)

## Quick Start

```bash
cd scripts/

python compare_head_configs.py \
    --config1 ../head_data/mistral/core_temp0.001_prune0.0.json \
    --config2 ../head_data/mistral/bce_weights_lambda0.001_n5000.json \
    --llm mistral \
    --feature_dir ../head_data/mistral \
    --k 10 \
    --num_heads 8 \
    --num_permutations 10000 \
    --beir_dir /path/to/beir \
    --output results.json \
    -v
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config1` | str | required | Path to first head configuration file |
| `--config2` | str | required | Path to second head configuration file |
| `--llm` | str | `mistral` | LLM model (mistral/llama/phi/granite) |
| `--feature_dir` | str | required | Directory containing feature .npz files |
| `--k` | int | `10` | K value for feature files (*_k{K}.npz pattern) |
| `--num_heads` | int | `8` | Number of top heads to use |
| `--num_permutations` | int | `10000` | Number of random permutations for significance test |
| `--metric` | str | `NDCG@10` | Metric for comparison |
| `--beir_dir` | str | `None` | Path to BEIR data directory for qrels (required for proper NDCG) |
| `--output` | str | `None` | Output file for detailed JSON results |
| `--n_jobs` | int | `4` | Number of parallel workers |
| `--verbose` | flag | `False` | Print detailed progress |

## Output

### Console Output

The script prints several sections with color-coded results:

1. **Per-Dataset Results**: Shows scores for each dataset with statistical significance
   ```
   Dataset                        Config1              Config2              Diff         p-value       Sig
   --------------------------------------------------------------------------------------------------------
   Note: Green values indicate statistically significant improvements (p < 0.05)
   nq                             0.523                0.499                +0.025       0.001234      **
   scifact                        0.689                0.654                +0.035       0.000123      **
   quora                          0.834                0.856                -0.022       0.003456      **
   --------------------------------------------------------------------------------------------------------
   cqadupstack-average            0.325                0.319                +0.006       0.045678      *
   --------------------------------------------------------------------------------------------------------
   BEIR Macro Average             0.452                0.431                +0.021       0.001234      **
   ```

   - **Green highlighting**: Indicates the configuration that is statistically significantly better
   - **Significance markers**: `*` for p < 0.05, `**` for p < 0.01

2. **Overall BEIR Statistical Test**: Details of the significance test
   ```
   Overall BEIR Statistical Significance Test
   ================================================================================
   Config 1 mean: 0.452
   Config 2 mean: 0.431
   Observed difference: +0.021
   Number of queries: 12345
   Number of permutations: 10000
   P-value (two-tailed): 0.001234
   Significant at α=0.05: Yes
   Significant at α=0.01: Yes
   ```

3. **Interpretation**: Plain English summary with win counts
   ```
   Interpretation
   ================================================================================
   Config 'core_temp0.001_prune0.0' performs better by 0.021 on average.
   This difference is HIGHLY SIGNIFICANT (p < 0.01).

   Per-dataset significant wins (p < 0.05):
     Config 1 (core_temp0.001_prune0.0): 8
     Config 2 (bce_weights_lambda0.001_n5000): 2
   ```

### JSON Output

When using `--output`, a detailed JSON file is saved:

```json
{
  "config1": "core_temp0.001_prune0.0",
  "config2": "bce_weights_lambda0.001_n5000",
  "llm": "mistral",
  "metric": "NDCG@10",
  "k": 10,
  "num_heads": 8,
  "num_permutations": 10000,
  "beir_overall": {
    "mean_score1": 0.452,
    "mean_score2": 0.431,
    "observed_diff": 0.021,
    "p_value": 0.001234,
    "num_queries": 12345,
    "significant_at_0.05": true,
    "significant_at_0.01": true
  },
  "per_dataset": {
    "nq": {
      "mean_score1": 0.523,
      "mean_score2": 0.499,
      "observed_diff": 0.024,
      "p_value": 0.001234,
      "num_queries": 3452,
      "significant_at_0.05": true,
      "significant_at_0.01": true
    },
    ...
  }
}
```

## Examples

### Example 1: Compare CoRe vs BCE Weights

```bash
python compare_head_configs.py \
    --config1 ../head_data/mistral/core_temp0.001_prune0.0.json \
    --config2 ../head_data/mistral/bce_weights_lambda0.001_n5000.json \
    --llm mistral \
    --feature_dir ../head_data/mistral \
    --k 10 \
    --num_heads 8 \
    --beir_dir /path/to/beir \
    --output core_vs_bce.json
```

### Example 2: Compare Different Lambda Values

```bash
python compare_head_configs.py \
    --config1 ../head_data/mistral/bce_weights_lambda0.0001_n5000.json \
    --config2 ../head_data/mistral/bce_weights_lambda0.001_n5000.json \
    --llm mistral \
    --feature_dir ../head_data/mistral \
    --beir_dir /path/to/beir \
    --output lambda_comparison.json
```

### Example 3: High-Precision Test with Many Permutations

```bash
# Use 100,000 permutations for very precise p-value
python compare_head_configs.py \
    --config1 ../head_data/mistral/core_temp0.001_prune0.0.json \
    --config2 ../head_data/mistral/bce_weights_lambda0.001_n5000.json \
    --llm mistral \
    --feature_dir ../head_data/mistral \
    --beir_dir /path/to/beir \
    --num_permutations 100000
```

### Example 4: Verbose Output

```bash
python compare_head_configs.py \
    --config1 ../head_data/mistral/core_temp0.001_prune0.0.json \
    --config2 ../head_data/mistral/bce_weights_lambda0.001_n5000.json \
    --llm mistral \
    --feature_dir ../head_data/mistral \
    --beir_dir /path/to/beir \
    -v
```

## Interpreting Results

### P-value Guidelines

- **p < 0.01**: Highly significant - very strong evidence that the difference is real
- **p < 0.05**: Significant - reasonable evidence that the difference is real
- **p ≥ 0.05**: Not significant - difference could be due to random chance

### Number of Permutations

- **1,000**: Quick test, approximate p-value
- **10,000** (default): Good balance of precision and speed
- **50,000+**: High precision for publication-quality results

The minimum detectable p-value is `1/num_permutations`, so use at least 10,000 permutations if you expect p < 0.001.

## Requirements

### Feature Files

Feature files must be named `*_k{K}.npz` (e.g., `attention_features_nq_k10.npz`) and contain:
- `features`: (num_docs, num_heads) attention features
- `labels`: (num_docs,) binary relevance labels
- `docs_per_query`: (num_queries,) docs per query
- `query_ids`: (num_docs,) query IDs
- `doc_ids`: (num_docs,) document IDs

### BEIR Qrels

For proper NDCG computation, provide `--beir_dir` pointing to a directory with qrels:
```
beir_dir/
  nq/qrels/test.tsv
  scifact/qrels/test.tsv
  cqadupstack/android/qrels/test.tsv
  ...
```

Without external qrels, queries without matching relevance judgments will be skipped.

## Troubleshooting

### Missing Feature Files

**Error**: `Found 0 datasets` or `No BEIR datasets found`

**Solution**: Ensure feature files follow the naming pattern `attention_features_{dataset}_k{K}.npz`:
```bash
ls ../head_data/mistral/*_k10.npz
```

### No Queries Evaluated

**Error**: All datasets show 0 queries

**Solution**: Ensure query IDs in feature files match those in qrels. Check ID format (string vs int).

### Low Number of Permutations Warning

If you use fewer than 1,000 permutations, results may be imprecise. For reliable results, use at least 10,000 permutations.

## References

### Randomization Test

The randomization test (permutation test) is based on:

- Smucker, M. D., Allan, J., & Carterette, B. (2007). A comparison of statistical significance tests for information retrieval evaluation. *CIKM 2007*.
- Carterette, B. (2012). Multiple testing in statistical analysis of systems-based information retrieval experiments. *TOIS*.

### BEIR Benchmark

- Thakur, N., et al. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. *NeurIPS 2021*.

## Tips

1. **Always use --beir_dir** for proper NDCG computation with external qrels
2. **Use more permutations** (50,000+) when differences are small or borderline significant
3. **Check verbose output** (-v) to verify queries are being evaluated correctly
4. **Save JSON output** for detailed analysis and generating tables/plots later
5. **Compare multiple num_heads values** to understand sensitivity to head count
