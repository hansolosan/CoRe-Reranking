#!/bin/bash
#
# Helper script to run full head configuration comparison pipeline
#
# This script:
# 1. Runs reranking for two head configurations on specified datasets
# 2. Performs statistical significance testing
#
# Usage:
#   ./run_head_comparison.sh config1.json config2.json mistral "nq scifact hotpotqa"
#

set -e  # Exit on error

# Check arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <config1_path> <config2_path> <llm> [datasets] [num_permutations]"
    echo ""
    echo "Arguments:"
    echo "  config1_path        Path to first head configuration file"
    echo "  config2_path        Path to second head configuration file"
    echo "  llm                 LLM model (mistral/llama/phi/granite)"
    echo "  datasets (optional) Space-separated dataset names (default: all BEIR)"
    echo "  num_permutations    Number of permutations for significance test (default: 10000)"
    echo ""
    echo "Example:"
    echo "  $0 ../head_data/mistral/core_temp0.001_prune0.0.json \\"
    echo "     ../head_data/mistral/core_temp0.001_prune0.3.json \\"
    echo "     mistral \\"
    echo "     \"nq scifact hotpotqa\" \\"
    echo "     10000"
    exit 1
fi

CONFIG1="$1"
CONFIG2="$2"
LLM="$3"
DATASETS="${4:-}"
NUM_PERM="${5:-10000}"

# Extract config names for reranking
CONFIG1_NAME=$(basename "$CONFIG1" .json)
CONFIG2_NAME=$(basename "$CONFIG2" .json)

echo "================================================================"
echo "Head Configuration Comparison Pipeline"
echo "================================================================"
echo "Config 1: $CONFIG1_NAME"
echo "Config 2: $CONFIG2_NAME"
echo "LLM: $LLM"
echo "Datasets: ${DATASETS:-all BEIR datasets}"
echo "Permutations: $NUM_PERM"
echo "================================================================"
echo ""

# Determine datasets to process
if [ -z "$DATASETS" ]; then
    # Use all BEIR datasets
    DATASET_LIST=(
        trec-covid nfcorpus dbpedia-entity scifact scidocs fiqa nq fever
        climate-fever hotpotqa webis-touche2020 msmarco quora arguana
        cqadupstack-android cqadupstack-english cqadupstack-gaming cqadupstack-gis
        cqadupstack-mathematica cqadupstack-physics cqadupstack-programmers cqadupstack-stats
        cqadupstack-tex cqadupstack-unix cqadupstack-webmasters cqadupstack-wordpress
    )
else
    # Use provided datasets
    read -ra DATASET_LIST <<< "$DATASETS"
fi

# Parse config files to extract parameters
extract_param() {
    local config_name="$1"
    local param="$2"

    if [[ "$config_name" =~ $param([0-9.]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo ""
    fi
}

# Extract temp and prune parameters for both configs
TEMP1=$(extract_param "$CONFIG1_NAME" "temp")
PRUNE1=$(extract_param "$CONFIG1_NAME" "prune")
TEMP2=$(extract_param "$CONFIG2_NAME" "temp")
PRUNE2=$(extract_param "$CONFIG2_NAME" "prune")

# Set default temp if not found
if [ -z "$TEMP1" ]; then
    if [[ "$LLM" == "phi" ]] || [[ "$LLM" == "llama" ]]; then
        TEMP1="0.1"
    else
        TEMP1="0.001"
    fi
fi

if [ -z "$TEMP2" ]; then
    if [[ "$LLM" == "phi" ]] || [[ "$LLM" == "llama" ]]; then
        TEMP2="0.1"
    else
        TEMP2="0.001"
    fi
fi

# Set default prune if not found
PRUNE1="${PRUNE1:-0.0}"
PRUNE2="${PRUNE2:-0.0}"

echo "Step 1: Running reranking for Config 1..."
echo "Parameters: temp=$TEMP1, prune=$PRUNE1"
echo ""

for dataset in "${DATASET_LIST[@]}"; do
    output_file="../reranking_output/$LLM/top40/${dataset}_${CONFIG1_NAME}.json"

    if [ -f "$output_file" ]; then
        echo "  [SKIP] $dataset (already exists)"
    else
        echo "  [RUN] $dataset"
        python reranking.py \
            --llm "$LLM" \
            --data "$dataset" \
            --reranker core \
            --temp "$TEMP1" \
            --prune "$PRUNE1" \
            --num_head 8 \
            --top_k 40 || echo "  [FAILED] $dataset"
    fi
done

echo ""
echo "Step 2: Running reranking for Config 2..."
echo "Parameters: temp=$TEMP2, prune=$PRUNE2"
echo ""

for dataset in "${DATASET_LIST[@]}"; do
    output_file="../reranking_output/$LLM/top40/${dataset}_${CONFIG2_NAME}.json"

    if [ -f "$output_file" ]; then
        echo "  [SKIP] $dataset (already exists)"
    else
        echo "  [RUN] $dataset"
        python reranking.py \
            --llm "$LLM" \
            --data "$dataset" \
            --reranker core \
            --temp "$TEMP2" \
            --prune "$PRUNE2" \
            --num_head 8 \
            --top_k 40 || echo "  [FAILED] $dataset"
    fi
done

echo ""
echo "Step 3: Running statistical comparison..."
echo ""

# Build dataset argument
DATASET_ARGS=""
if [ -n "$DATASETS" ]; then
    DATASET_ARGS="--datasets ${DATASET_LIST[@]}"
fi

# Run comparison
python compare_head_configs.py \
    --config1 "$CONFIG1" \
    --config2 "$CONFIG2" \
    --llm "$LLM" \
    $DATASET_ARGS \
    --num_permutations "$NUM_PERM" \
    --metric NDCG@10 \
    --stratify \
    --skip_reranking \
    --output "comparison_${CONFIG1_NAME}_vs_${CONFIG2_NAME}.json"

echo ""
echo "================================================================"
echo "Pipeline completed!"
echo "Results saved to: comparison_${CONFIG1_NAME}_vs_${CONFIG2_NAME}.json"
echo "================================================================"
