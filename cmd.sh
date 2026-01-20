#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python scripts/extract_head_features.py --llm mistral --input_file retriever_output/nq.json --qrels path/to/nq-qrels.tsv --max_samples 100