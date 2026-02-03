# Head Configuration Comparison Guide

This guide explains how to compare two attention head configurations using statistical significance testing.

## Overview

The comparison pipeline consists of two main scripts:

1. **`compare_head_configs.py`**: Core comparison script that evaluates two head configurations and performs statistical significance testing
2. **`run_head_comparison.sh`**: Convenience script that automates the full pipeline (reranking + comparison)

## Statistical Significance Testing

The scripts implement a **randomization test** (also called permutation test), which is the gold standard for statistical significance testing in information retrieval.

### How it Works

1. **Compute observed difference**: Calculate the actual difference in metrics (e.g., NDCG@10) between the two systems
2. **Random permutations**: For each query, randomly swap which system's score is labeled as "system 1" vs "system 2"
3. **Recompute difference**: Calculate the difference for each random permutation
4. **P-value**: The proportion of permutations where the difference is as extreme as the observed difference

### Stratification

Use `--stratify` to perform stratified randomization by dataset. This ensures that permutations preserve the dataset structure, which is important when datasets have different characteristics or difficulties.

## Quick Start

### Option 1: Using the Helper Script (Recommended)

The helper script handles both reranking and comparison:

```bash
cd experiments/

# Compare two configurations on all BEIR datasets
./run_head_comparison.sh \
    ../head_data/mistral/core_temp0.001_prune0.0.json \
    ../head_data/mistral/core_temp0.001_prune0.3.json \
    mistral

# Compare on specific datasets with custom permutations
./run_head_comparison.sh \
    ../head_data/mistral/core_temp0.001_prune0.0.json \
    ../head_data/mistral/core_temp0.001_prune0.3.json \
    mistral \
    "nq scifact hotpotqa" \
    20000
```

### Option 2: Manual Pipeline

If you want more control, run each step manually:

#### Step 1: Run Reranking for Both Configurations

```bash
cd experiments/

# Rerank with configuration 1
for dataset in nq scifact hotpotqa; do
    python reranking.py \
        --llm mistral \
        --data $dataset \
        --reranker core \
        --temp 0.001 \
        --prune 0.0 \
        --num_head 8
done

# Rerank with configuration 2
for dataset in nq scifact hotpotqa; do
    python reranking.py \
        --llm mistral \
        --data $dataset \
        --reranker core \
        --temp 0.001 \
        --prune 0.3 \
        --num_head 8
done
```

#### Step 2: Run Comparison

```bash
python compare_head_configs.py \
    --config1 ../head_data/mistral/core_temp0.001_prune0.0.json \
    --config2 ../head_data/mistral/core_temp0.001_prune0.3.json \
    --llm mistral \
    --datasets nq scifact hotpotqa \
    --num_permutations 10000 \
    --metric NDCG@10 \
    --stratify \
    --skip_reranking \
    --output results.json
```

## Command-Line Arguments

### `compare_head_configs.py`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config1` | str | required | Path to first head configuration file |
| `--config2` | str | required | Path to second head configuration file |
| `--llm` | str | `mistral` | LLM model (mistral/llama/phi/granite) |
| `--datasets` | str+ | all BEIR | List of BEIR datasets to evaluate |
| `--num_permutations` | int | `10000` | Number of random permutations for significance test |
| `--num_head` | int | `8` | Number of top heads to use |
| `--top_k` | int | `40` | Number of documents to rerank |
| `--metric` | str | `NDCG@10` | Metric for comparison (NDCG@1/5/10) |
| `--stratify` | flag | `False` | Stratify randomization by dataset |
| `--skip_reranking` | flag | `False` | Skip reranking (assume results exist) |
| `--output` | str | `None` | Output file for detailed JSON results |

### `run_head_comparison.sh`

```bash
./run_head_comparison.sh <config1> <config2> <llm> [datasets] [num_permutations]
```

| Argument | Description | Example |
|----------|-------------|---------|
| `config1` | Path to first config | `../head_data/mistral/core_temp0.001_prune0.0.json` |
| `config2` | Path to second config | `../head_data/mistral/core_temp0.001_prune0.3.json` |
| `llm` | LLM model | `mistral` |
| `datasets` | Space-separated datasets (optional) | `"nq scifact hotpotqa"` |
| `num_permutations` | Number of permutations (optional) | `20000` |

## Output

### Console Output

The script prints several sections with color-coded results:

1. **Per-Dataset Results**: Shows scores for each dataset with statistical significance
   ```
   Dataset                        Config1              Config2              Diff         p-value       Sig
   --------------------------------------------------------------------------------------------------------
   Note: Green values indicate statistically significant improvements (p < 0.05)
   nq                             0.5234               0.4987               +0.0247      0.002341      **
   scifact                        0.6891               0.6542               +0.0349      0.048723      *
   --------------------------------------------------------------------------------------------------------
   cqadupstack-average            0.3245               0.3189               +0.0056      0.234567
   --------------------------------------------------------------------------------------------------------
   BEIR Macro Average             0.4523               0.4312               +0.0211      0.001234      **
   ```

   - **Green highlighting**: Indicates the configuration that is statistically significantly better
   - **Significance markers**: `*` for p < 0.05, `**` for p < 0.01

2. **Overall BEIR Statistical Test**: Details of the macro-average significance test
   ```
   Overall BEIR Statistical Significance Test
   ================================================================================
   Config 1 macro average: 0.5234
   Config 2 macro average: 0.4987
   Observed difference: +0.0247
   Number of dataset groups: 15
   Number of permutations: 10000
   P-value (two-tailed): 0.0023
   Significant at α=0.05: Yes
   Significant at α=0.01: Yes
   ```

3. **Interpretation**: Plain English summary with win counts
   ```
   Interpretation
   ================================================================================
   Config 'core_temp0.001_prune0.0' performs better by 0.0247 on BEIR macro average.
   This difference is HIGHLY SIGNIFICANT (p < 0.01).

   Per-dataset significant wins (p < 0.05):
     Config 1 (core_temp0.001_prune0.0): 8
     Config 2 (core_temp0.001_prune0.3): 2
   ```

### JSON Output

When using `--output`, a detailed JSON file is saved with:

```json
{
  "config1": "core_temp0.001_prune0.0",
  "config2": "core_temp0.001_prune0.3",
  "llm": "mistral",
  "metric": "NDCG@10",
  "num_permutations": 10000,
  "stratified": true,
  "macro_average": {
    "config1": 0.5234,
    "config2": 0.4987,
    "difference": 0.0247
  },
  "statistical_test": {
    "observed_diff": 0.0247,
    "mean_score1": 0.5234,
    "mean_score2": 0.4987,
    "p_value": 0.0023,
    "num_queries": 1250,
    "num_permutations": 10000,
    "significant_at_0.05": true,
    "significant_at_0.01": true
  },
  "per_dataset_results": { ... }
}
```

## Examples

### Example 1: Compare Layer Pruning Strategies

```bash
# Compare no pruning vs 30% pruning
./run_head_comparison.sh \
    ../head_data/mistral/core_temp0.001_prune0.0.json \
    ../head_data/mistral/core_temp0.001_prune0.3.json \
    mistral \
    "nq fever hotpotqa scifact"
```

### Example 2: Compare Different Loss Functions

```bash
# Compare CoRe vs BCE weights
python compare_head_configs.py \
    --config1 ../head_data/mistral/core_temp0.001_prune0.0.json \
    --config2 ../head_data/mistral/bce_weights_lambda0.001_n1000.json \
    --llm mistral \
    --datasets nq scifact \
    --num_permutations 10000 \
    --stratify \
    --output core_vs_bce.json
```

### Example 3: High-Precision Test with Many Permutations

```bash
# Use 100,000 permutations for very precise p-value
python compare_head_configs.py \
    --config1 ../head_data/mistral/core_temp0.001_prune0.0.json \
    --config2 ../head_data/mistral/core_temp0.001_prune0.3.json \
    --llm mistral \
    --num_permutations 100000 \
    --stratify
```

### Example 4: Compare Across All BEIR Datasets

```bash
# Full BEIR evaluation (26 datasets)
./run_head_comparison.sh \
    ../head_data/mistral/core_temp0.001_prune0.0.json \
    ../head_data/mistral/core_temp0.001_prune0.3.json \
    mistral
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

### When to Use Stratification

Use `--stratify` when:
- Comparing across multiple diverse datasets
- Datasets have different sizes or difficulties
- You want to control for dataset-level variance

Don't use stratification when:
- Only evaluating on a single dataset
- All datasets are similar in nature

## Advanced Usage

### Comparing Trained Weights

If you've trained head weights using `train_head_weights_bce.py`, compare them:

```bash
python compare_head_configs.py \
    --config1 ../head_data/mistral/bce_weights_lambda0.001_n1000.json \
    --config2 ../head_data/mistral/infonce_weights_lambda0.01_n1000.json \
    --llm mistral \
    --datasets nq scifact hotpotqa \
    --num_permutations 20000 \
    --stratify \
    --output bce_vs_infonce.json
```

### Batch Comparison

To compare multiple configurations, use a loop:

```bash
# Compare different pruning ratios
for prune2 in 0.3 0.4 0.5 0.6; do
    ./run_head_comparison.sh \
        ../head_data/mistral/core_temp0.001_prune0.0.json \
        ../head_data/mistral/core_temp0.001_prune${prune2}.json \
        mistral \
        "nq scifact hotpotqa" \
        10000
done
```

### Different Metrics

Compare using different ranking metrics:

```bash
# Use NDCG@1 (focuses on top-1 accuracy)
python compare_head_configs.py \
    --config1 config1.json \
    --config2 config2.json \
    --llm mistral \
    --metric NDCG@1

# Use NDCG@5 (focuses on top-5)
python compare_head_configs.py \
    --config1 config1.json \
    --config2 config2.json \
    --llm mistral \
    --metric NDCG@5
```

## Troubleshooting

### Missing Reranking Results

**Error**: `Reranking results not found: ...`

**Solution**: Either run reranking first, or check that the file paths are correct:

```bash
# Run reranking
python reranking.py --llm mistral --data nq --reranker core --temp 0.001 --prune 0.0

# Or use the helper script which runs reranking automatically
./run_head_comparison.sh config1.json config2.json mistral "nq"
```

### No Common Queries

**Error**: `No common queries between the two systems`

**Solution**: Ensure both configurations were evaluated on the same datasets and queries.

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

1. **Use stratification** for multi-dataset comparisons to account for dataset variance
2. **Use more permutations** (50,000+) when differences are small or borderline significant
3. **Check multiple metrics** (NDCG@1, NDCG@5, NDCG@10) to understand trade-offs
4. **Save JSON output** for detailed analysis and generating tables/plots later
5. **Compare on diverse datasets** to ensure improvements generalize
