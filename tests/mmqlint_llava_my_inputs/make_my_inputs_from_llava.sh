#!/usr/bin/env bash
set -euo pipefail

LLAVA_JSON="$1"
IMAGE_ROOT="$2"
OUT_DIR="${3:-./my_inputs}"
MAX_RECORDS="${4:-500}"

python tests/mmqlint_llava_my_inputs/make_my_inputs_from_llava.py \
  --llava-json "$LLAVA_JSON" \
  --image-root "$IMAGE_ROOT" \
  --out-dir "$OUT_DIR" \
  --max-records "$MAX_RECORDS"
