#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 <gpu> <output> [options]

Required positional arguments:
  gpu                 CUDA device id
  output              Model/output folder

Options:
  --model <model>     Checkpoint id, model filename, or model path
                      (default: <output>/model.pt)
  --seed <seed>       Non-negative random seed (default: 0)
  --record             Save one rollout NPZ
  -h, --help           Show this help message

Examples:
  $0 0 output_amp_g1_location_1 --seed 123
  $0 0 output_amp_g1_location_1 --record --model 6200 --seed 123
EOF
}

error() {
    echo "Error: $1" >&2
    echo >&2
    usage >&2
    exit 2
}

if [[ $# -eq 1 && ("$1" == "-h" || "$1" == "--help") ]]; then
    usage
    exit 0
fi

if [[ $# -lt 2 ]]; then
    error "gpu and output are required"
fi

CUDA_ID=$1
MODEL_FOLDER=$2
shift 2

MODEL_FILE=""
RAND_SEED=0
RECORD=false
MODEL_SET=false
SEED_SET=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            if [[ "$MODEL_SET" == true ]]; then
                error "--model may only be specified once"
            fi
            if [[ $# -lt 2 ]]; then
                error "--model requires a value"
            fi
            MODEL_FILE=$2
            MODEL_SET=true
            shift 2
            ;;
        --model=*)
            if [[ "$MODEL_SET" == true ]]; then
                error "--model may only be specified once"
            fi
            MODEL_FILE=${1#--model=}
            if [[ -z "$MODEL_FILE" ]]; then
                error "--model requires a value"
            fi
            MODEL_SET=true
            shift
            ;;
        --seed)
            if [[ "$SEED_SET" == true ]]; then
                error "--seed may only be specified once"
            fi
            if [[ $# -lt 2 ]]; then
                error "--seed requires a value"
            fi
            RAND_SEED=$2
            SEED_SET=true
            shift 2
            ;;
        --seed=*)
            if [[ "$SEED_SET" == true ]]; then
                error "--seed may only be specified once"
            fi
            RAND_SEED=${1#--seed=}
            if [[ -z "$RAND_SEED" ]]; then
                error "--seed requires a value"
            fi
            SEED_SET=true
            shift
            ;;
        --record)
            if [[ "$RECORD" == true ]]; then
                error "--record may only be specified once"
            fi
            RECORD=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "unknown argument: $1"
            ;;
    esac
done

if [[ ! "$RAND_SEED" =~ ^[0-9]+$ ]]; then
    error "--seed must be a non-negative integer"
fi

ARGS=(--model_folder "$MODEL_FOLDER" --rand_seed "$RAND_SEED")
if [[ -n "$MODEL_FILE" ]]; then
    ARGS+=(--model_file "$MODEL_FILE")
fi
if [[ "$RECORD" == true ]]; then
    ARGS+=(--record true)
fi

if [[ -n "$MODEL_FILE" ]]; then
    MODEL_DISPLAY=$MODEL_FILE
else
    MODEL_DISPLAY="default (model.pt)"
fi

printf 'GPU:    %s\n' "$CUDA_ID"
printf 'Output: %s\n' "$MODEL_FOLDER"
printf 'Model:  %s\n' "$MODEL_DISPLAY"
printf 'Seed:   %s\n' "$RAND_SEED"
printf 'Record: %s\n' "$RECORD"

CUDA_VISIBLE_DEVICES="$CUDA_ID" \
    python mimickit/test_viser.py "${ARGS[@]}"
