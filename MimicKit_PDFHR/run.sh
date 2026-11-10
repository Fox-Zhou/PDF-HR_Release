#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <cuda_id> <arg_file>" >&2
    exit 2
fi

CUDA_ID="$1"
ARG_FILE="$2"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "$ARG_FILE" ]]; then
    echo "Argument file not found: $ARG_FILE" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="$CUDA_ID"
exec python mimickit/run.py --arg_file "$ARG_FILE" --visualize false
