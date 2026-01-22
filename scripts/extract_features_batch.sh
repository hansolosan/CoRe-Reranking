#!/bin/bash
#
# Batch feature extraction from retrieved document files.
#
# Extracts attention features for all combinations of:
#   - Input JSON files
#   - Max documents per query values
#
# Output files are named: attention_features_{input_name}_k{max_docs}_{llm}.npz
#
# Usage:
#   ./scripts/extract_features_batch.sh \
#       --conda myenv \
#       --llm mistral \
#       --qrels path/to/qrels.tsv \
#       --max_doc_tokens 300 \
#       --max_docs 10 20 50 100 \
#       --files retriever_output/nq.json retriever_output/hotpotqa.json
#
# Example with all options:
#   ./scripts/extract_features_batch.sh \
#       --conda core_env \
#       --llm mistral \
#       --qrels data/qrels/nq-test.tsv \
#       --max_doc_tokens 300 \
#       --max_samples 1000 \
#       --max_docs 10 50 100 \
#       --output_dir head_data/mistral \
#       --files retriever_output/*.json
#

set -e  # Exit on error

# Default values
CONDA_ENV=""
LLM="mistral"
QRELS=""
MAX_DOC_TOKENS=300
MAX_SAMPLES=""
BATCH_SIZE=""
OUTPUT_DIR=""
DRY_RUN=false
EXTRA_ARGS=""

# Arrays for multi-value arguments
MAX_DOCS_LIST=()
INPUT_FILES=()

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Required arguments:
  --conda ENV           Conda environment name to activate
  --llm LLM             LLM model: mistral, llama, phi, granite (default: mistral)
  --qrels FILE          Path to qrels file for relevance labels
  --max_docs K [K ...]  List of max documents per query values
  --files F [F ...]     List of input JSON files (supports glob patterns)

Optional arguments:
  --max_doc_tokens N    Max tokens per document (default: 300)
  --max_samples N       Max number of query samples to process
  --batch_size N        Batch size for inference (default: 1)
  --output_dir DIR      Output directory (default: head_data/{llm})
  --dry_run             Print commands without executing
  --extra_args "ARGS"   Additional arguments to pass to extract_head_features.py
  -h, --help            Show this help message

Output naming:
  Files are saved as: {output_dir}/attention_features_{input_name}_k{max_docs}.npz

Example:
  $0 --conda myenv --llm mistral --qrels qrels.tsv \\
     --max_docs 10 50 100 --files retriever_output/nq.json
EOF
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --conda)
            CONDA_ENV="$2"
            shift 2
            ;;
        --llm)
            LLM="$2"
            shift 2
            ;;
        --qrels)
            QRELS="$2"
            shift 2
            ;;
        --max_doc_tokens)
            MAX_DOC_TOKENS="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        --extra_args)
            EXTRA_ARGS="$2"
            shift 2
            ;;
        --max_docs)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MAX_DOCS_LIST+=("$1")
                shift
            done
            ;;
        --files)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                # Expand glob patterns
                for f in $1; do
                    if [[ -f "$f" ]]; then
                        INPUT_FILES+=("$f")
                    else
                        log_warn "File not found: $f"
                    fi
                done
                shift
            done
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
MISSING_ARGS=()

if [[ -z "$CONDA_ENV" ]]; then
    MISSING_ARGS+=("--conda")
fi

if [[ -z "$QRELS" ]]; then
    MISSING_ARGS+=("--qrels")
fi

if [[ ${#MAX_DOCS_LIST[@]} -eq 0 ]]; then
    MISSING_ARGS+=("--max_docs")
fi

if [[ ${#INPUT_FILES[@]} -eq 0 ]]; then
    MISSING_ARGS+=("--files")
fi

if [[ ${#MISSING_ARGS[@]} -gt 0 ]]; then
    log_error "Missing required arguments: ${MISSING_ARGS[*]}"
    echo ""
    print_usage
    exit 1
fi

# Validate LLM choice
if [[ ! "$LLM" =~ ^(mistral|llama|phi|granite)$ ]]; then
    log_error "Invalid LLM: $LLM. Must be one of: mistral, llama, phi, granite"
    exit 1
fi

# Validate qrels file exists
if [[ ! -f "$QRELS" ]]; then
    log_error "Qrels file not found: $QRELS"
    exit 1
fi

# Set default output directory
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="head_data/$LLM"
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Print configuration
echo ""
echo "========================================"
echo "Batch Feature Extraction Configuration"
echo "========================================"
echo "Conda environment:  $CONDA_ENV"
echo "LLM:                $LLM"
echo "Qrels file:         $QRELS"
echo "Max doc tokens:     $MAX_DOC_TOKENS"
echo "Max samples:        ${MAX_SAMPLES:-all}"
echo "Batch size:         ${BATCH_SIZE:-1}"
echo "Output directory:   $OUTPUT_DIR"
echo "Max docs values:    ${MAX_DOCS_LIST[*]}"
echo "Input files:        ${#INPUT_FILES[@]} file(s)"
for f in "${INPUT_FILES[@]}"; do
    echo "                    - $(basename "$f")"
done
echo "Dry run:            $DRY_RUN"
if [[ -n "$EXTRA_ARGS" ]]; then
    echo "Extra args:         $EXTRA_ARGS"
fi
echo "========================================"
echo ""

# Calculate total jobs
TOTAL_JOBS=$((${#INPUT_FILES[@]} * ${#MAX_DOCS_LIST[@]}))
log_info "Total extraction jobs: $TOTAL_JOBS"
echo ""

# Activate conda environment
if [[ "$DRY_RUN" == false ]]; then
    log_info "Activating conda environment: $CONDA_ENV"

    # Try different conda initialization methods
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
        source "/opt/conda/etc/profile.d/conda.sh"
    elif command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
    else
        log_error "Could not find conda. Please ensure conda is installed and in PATH."
        exit 1
    fi

    conda activate "$CONDA_ENV" || {
        log_error "Failed to activate conda environment: $CONDA_ENV"
        exit 1
    }
    log_success "Conda environment activated"
fi

# Create output directory
if [[ "$DRY_RUN" == false ]]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Track progress
COMPLETED=0
FAILED=0
SKIPPED=0

# Run extraction for all combinations
for INPUT_FILE in "${INPUT_FILES[@]}"; do
    # Get base name without extension
    INPUT_NAME=$(basename "$INPUT_FILE" .json)
    INPUT_NAME=$(basename "$INPUT_NAME" .json.gz)
    INPUT_NAME=$(basename "$INPUT_NAME" .json.bz2)

    for MAX_DOCS in "${MAX_DOCS_LIST[@]}"; do
        COMPLETED=$((COMPLETED + 1))

        # Construct output filename
        OUTPUT_NAME="attention_features_${INPUT_NAME}_k${MAX_DOCS}"
        OUTPUT_PATH="${OUTPUT_DIR}/${OUTPUT_NAME}.npz"

        echo ""
        log_info "[$COMPLETED/$TOTAL_JOBS] Processing: $(basename "$INPUT_FILE") with max_docs=$MAX_DOCS"

        # Check if output already exists
        if [[ -f "$OUTPUT_PATH" && "$DRY_RUN" == false ]]; then
            log_warn "Output exists, skipping: $OUTPUT_PATH"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        # Build command
        CMD="python ${SCRIPT_DIR}/extract_head_features.py"
        CMD="$CMD --llm $LLM"
        CMD="$CMD --input_file \"$INPUT_FILE\""
        CMD="$CMD --qrels \"$QRELS\""
        CMD="$CMD --max_docs $MAX_DOCS"
        CMD="$CMD --max_doc_tokens $MAX_DOC_TOKENS"
        CMD="$CMD -o \"$OUTPUT_NAME\""

        if [[ -n "$MAX_SAMPLES" ]]; then
            CMD="$CMD --max_samples $MAX_SAMPLES"
        fi

        if [[ -n "$BATCH_SIZE" ]]; then
            CMD="$CMD --batch_size $BATCH_SIZE"
        fi

        if [[ -n "$EXTRA_ARGS" ]]; then
            CMD="$CMD $EXTRA_ARGS"
        fi

        echo "Command: $CMD"

        if [[ "$DRY_RUN" == true ]]; then
            log_info "[DRY RUN] Would execute above command"
        else
            # Execute command
            if eval "$CMD"; then
                log_success "Completed: $OUTPUT_PATH"
            else
                log_error "Failed: $INPUT_FILE with max_docs=$MAX_DOCS"
                FAILED=$((FAILED + 1))
            fi
        fi
    done
done

# Summary
echo ""
echo "========================================"
echo "Extraction Summary"
echo "========================================"
echo "Total jobs:    $TOTAL_JOBS"
echo "Completed:     $((COMPLETED - FAILED - SKIPPED))"
echo "Skipped:       $SKIPPED"
echo "Failed:        $FAILED"
echo "========================================"

if [[ $FAILED -gt 0 ]]; then
    log_error "Some jobs failed. Check the output above for details."
    exit 1
else
    log_success "All jobs completed successfully!"
fi
