# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements an attention-based reranker with Contrastive Retrieval (CoRe) head detection for information retrieval tasks. The system identifies the most relevant attention heads (roughly 1% of all heads) using contrastive learning between positive and hard negative documents, which significantly improves reranking performance compared to aggregating over all heads.

The implementation is based on In-Context-Reranking (https://github.com/OSU-NLP-Group/In-Context-Reranking).

## Supported LLMs

The codebase supports four LLM architectures:
- **Granite**: `ibm-granite/granite-3.2-8b-instruct` (temperature: 0.001)
- **Llama**: `meta-llama/Meta-Llama-3.1-8B-Instruct` (temperature: 0.1)
- **Mistral**: `mistralai/Mistral-7B-Instruct-v0.2` (temperature: 0.001)
- **Phi**: `microsoft/phi-4` (temperature: 0.1)

Note: Temperature parameters differ between models - Phi and Llama use 0.1, while Granite and Mistral use 0.001.

## Architecture

### Core Components

1. **Head Detectors** (`experiments/src/`):
   - `core_detector.py`: Contrastive Retrieval (CoRe) detector that identifies retrieval heads by contrasting attention scores between positive and hard negative documents
   - `qr_detector.py`: Query-Relevance (QR) detector that scores heads based on query-document attention
   - Both detectors use custom attention modules and extend base LLM classes

2. **Reranker** (`experiments/src/reranker_calib.py`):
   - Implements calibrated reranking using selected retrieval heads
   - Uses two forward passes: one with actual query, one with content-free query ("N/A")
   - Calibrated score = score(query) - score("N/A")
   - Applies token masking based on mean and standard deviation threshold

3. **Custom Model Extensions** (`experiments/src/custom/`):
   - Model-specific attention modifications for Granite, Llama, Mistral, and Phi
   - `custom_cache.py`: `DynamicCacheWithQuery` extends HuggingFace's `DynamicCache` to additionally store query states for attention computation
   - Custom modeling files patch the forward pass to capture and cache query states

4. **Main Scripts** (`experiments/`):
   - `head_detection.py`: Detects and scores retrieval heads using CoRe or QR detectors
   - `reranking.py`: Performs document reranking using detected heads
   - `evaluate_beir.py`: Evaluates reranking results on BEIR benchmark datasets
   - `evaluate_mldr.py`: Evaluates on multilingual MLDR datasets

### Data Flow

1. **Head Detection**: Query + positive/negative documents → HeadDetector → head scores saved to `head_data/{llm}/`
2. **Reranking**: Query + documents + top-k heads → Reranker → ranked results saved to `reranking_output/{llm}/`
3. **Evaluation**: Reranked results + qrels → metrics (NDCG@1/5/10)

### Prompt Structure

All models use a consistent prompt format:
```
{prompt_prefix} Here are some paragraphs:

[document 1] {doc1_text}

[document 2] {doc2_text}

...

Please find information that are relevant to the following query in the paragraphs above.

Query: {query_text}{prompt_suffix}
```

Each model has specific prefix/suffix tokens (e.g., `[INST]`/`[/INST]` for Mistral).

## Common Commands

All commands should be run from the `experiments/` directory.

### Head Detection

Reproduce CoRe head scores for a model:
```bash
python head_detection.py --llm mistral --detector core --temp 0.001
```

Arguments:
- `--llm`: Model choice (`mistral`, `llama`, `phi`, `granite`)
- `--detector`: Detection method (`core`, `qr`)
- `--temp`: Temperature for softmax in contrastive scoring (default: 0.001)
- `--prune`: Layer pruning ratio (default: 0.0)

Output: `../head_data/{llm}/{detector}_temp{temp}_prune{prune}.json`

### Reranking

Rerank documents using detected retrieval heads:
```bash
python reranking.py --llm mistral --data hotpotqa --reranker core --num_head 8
```

Arguments:
- `--llm`: Model choice
- `--data`: Dataset name (e.g., `hotpotqa`, `nq`, `scifact`)
- `--reranker`: Reranker type (`icr`, `qr`, `core`)
  - `icr`: In-Context Reranking (all heads)
  - `qr`: Query-Relevance heads
  - `core`: Contrastive Retrieval heads
- `--top_k`: Number of documents to rerank (default: 40)
- `--num_head`: Number of top retrieval heads to use (default: 8)
- `--temp`: Temperature (auto-set based on model)
- `--prune`: Layer pruning ratio (default: 0.0)

Output: `../reranking_output/{llm}/top{top_k}/{data}_{reranker}_temp{temp}_prune{prune}.json`

### Evaluation

Evaluate on BEIR benchmark:
```bash
python evaluate_beir.py
```
This evaluates all models and configurations on 26 BEIR datasets including cqadupstack variants.

Evaluate on MLDR (multilingual):
```bash
python evaluate_mldr.py
```
This evaluates on 6 languages: de, en, es, fr, it, pt.

## Data Requirements

### Required Data Downloads

1. **Retriever Outputs**: Download from https://drive.google.com/drive/folders/1nYDB1J03g8O9AlU3Zw6d6m2xQ1tTd1aQ and place in `./retriever_output/`
   - Contains top-40 retrieved documents from granite-embedding models
   - Format: JSON files named `{dataset}.json`

2. **Head Detection Data**: Download from https://drive.google.com/drive/folders/11CxygqHC_sPoYQHdRSfU-aUrihSuVTki and place in `./head_data/`
   - Contains query-document pairs with positive/negative labels for head detection
   - Pre-computed head scores are already included in the repository

### Data Formats

**Retriever output** (`retriever_output/{dataset}.json`):
```json
[
  {
    "idx": "query_id",
    "question": "query text",
    "paragraphs": [
      {
        "idx": "doc_id",
        "paragraph_text": "document text",
        ...
      }
    ]
  }
]
```

**Head detection data** (`head_data/nq_core.json`):
```json
[
  {
    "question": "query text",
    "paragraphs": [
      {
        "paragraph_text": "document text",
        "is_positive": true/false,
        "is_negative": true/false
      }
    ]
  }
]
```

**Head scores** (`head_data/{llm}/core_temp{temp}_prune{prune}.json`):
```json
{
  "layer-head": [score1, score2, ...],
  "0-0": [0.023, 0.019, ...],
  "0-1": [0.031, 0.028, ...]
}
```

## Key Implementation Details

### Layer Pruning
The `prune` parameter removes top layers from the model:
```python
config.num_hidden_layers = int(config.num_hidden_layers * (1-prune))
```
Example: `prune=0.3` removes 30% of layers from the top.

### Document Truncation
Documents are truncated to 300 tokens in `reranking.py`:
```python
p['paragraph_text'] = ' '.join(p['paragraph_text'].split(' ')[:300])
```

### Model-Specific Tokenization Offsets
Mistral requires a +1 offset when computing token spans due to tokenization behavior. Other models use offset=0.

### Attention Computation
- Uses Flash Attention 2 for efficiency
- Custom cache stores both key/value and query states
- Causal masking is applied during attention weight computation
- Grouped Query Attention (GQA) support for multi-query architectures

### Calibration Mechanism
The reranker uses contrastive calibration:
1. Forward pass with actual query → `tok_scores`
2. Forward pass with "N/A" query → `tok_scores_na`
3. Calibrated score = `tok_scores - tok_scores_na`
4. Token-level filtering: keep tokens where score > mean - 2*std

## BEIR Datasets

The evaluation covers 15 main datasets (26 including cqadupstack variants):
- `trec-covid`, `nfcorpus`, `dbpedia-entity`, `scifact`, `scidocs`, `fiqa`, `nq`, `fever`
- `climate-fever`, `hotpotqa`, `webis-touche2020`, `msmarco`, `quora`, `arguana`
- 12 cqadupstack domains (android, english, gaming, gis, mathematica, physics, programmers, stats, tex, unix, webmasters, wordpress)

## GPU Requirements

All experiments require CUDA-capable GPUs. The code uses:
- `torch.float16` for model precision
- `flash_attention_2` for efficient attention computation
- `device_map='cuda'` for automatic GPU placement
