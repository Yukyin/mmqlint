#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# mmqlint user-only checker for your own inputs
#
# This is a trimmed version of run_doc_bug_coverage_user.sh.
# It removes the built-in regression suite that intentionally creates failing data.
# It only checks the provided USER_DIR inputs.
#
# Usage
#   bash tests/run_doc_bug_coverage_user.sh USER_DIR
#
# USER_DIR should contain
#   infer.jsonl     required
#   train.jsonl     optional but recommended (enables train-mode checks)
#   dataset_dir     optional but recommended (enables dataset-on-disk checks)
#
# Environment variables
#   PROFILE         default: my-vlm
#   COORD_FIELD     default: coordinates
#   COORD_KEYS      default: x0,y0,x1,y1
#   KEEP_ARTIFACTS  default: 1  keep the temp dir under tests/.artifacts
#
# ==============================================================================

if [[ $# -ne 1 ]]; then
  echo "Usage: bash tests/run_doc_bug_coverage_user.sh USER_DIR"
  echo "USER_DIR should contain infer.jsonl and optionally train.jsonl and dataset_dir"
  exit 2
fi

USER_DIR="$1"
if [[ ! -d "$USER_DIR" ]]; then
  echo "ERROR: USER_DIR is not a directory: $USER_DIR"
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTS_DIR="$ROOT/tests"
ART_DIR="$TESTS_DIR/.artifacts"
PROFILES_YAML="$ROOT/profiles.yaml"

PROFILE="${PROFILE:-my-vlm}"
COORD_FIELD="${COORD_FIELD:-coordinates}"
COORD_KEYS="${COORD_KEYS:-x0,y0,x1,y1}"
KEEP_ARTIFACTS="${KEEP_ARTIFACTS:-1}"

mkdir -p "$ART_DIR"

# --- Ensure gitignore contains artifacts path ---
GITIGNORE="$ROOT/.gitignore"
if [[ -f "$GITIGNORE" ]]; then
  if ! grep -qE '^tests/\.artifacts/?$' "$GITIGNORE"; then
    echo "tests/.artifacts/" >> "$GITIGNORE"
    echo "[gitignore] appended: tests/.artifacts/"
  fi
fi

WORK_DIR="$(mktemp -d "$ART_DIR/tmp.XXXXXX")"
cleanup() {
  if [[ "$KEEP_ARTIFACTS" == "0" ]]; then
    rm -rf "$WORK_DIR"
  else
    echo "[artifacts] kept at: $WORK_DIR"
  fi
}
trap cleanup EXIT

banner() { echo "== $* =="; }
run_ok() { "$@"; }

USER_INFER_JSONL="$USER_DIR/infer.jsonl"
USER_TRAIN_JSONL="$USER_DIR/train.jsonl"
USER_DATASET_DIR="$USER_DIR/dataset_dir"

banner "[0] Sanity: editable install"
run_ok python -m pip install -e "$ROOT" >/dev/null

banner "[1] CLI sanity: commands exist"
run_ok mmqlint --help >/dev/null
run_ok mmqlint validate-profiles --help >/dev/null
run_ok mmqlint check --help >/dev/null
run_ok mmqlint verify-system --help >/dev/null
run_ok mmqlint check-dataset --help >/dev/null
echo "OK"

banner "[2] Profiles policy: require system + visible (schema-level policy)"
run_ok mmqlint validate-profiles --profile-file "$PROFILES_YAML"
echo "OK"

banner "[U0] User inputs: validate paths"
if [[ ! -f "$USER_INFER_JSONL" ]]; then
  echo "ERROR: missing required file: $USER_INFER_JSONL"
  echo "Expected format: JSONL, one JSON object per line, with fields id and messages"
  exit 2
fi
if [[ -f "$USER_TRAIN_JSONL" ]]; then
  echo "Found: $USER_TRAIN_JSONL"
else
  echo "Note: train.jsonl not found. Train mode checks on your data will be skipped."
fi
if [[ -d "$USER_DATASET_DIR" ]]; then
  echo "Found: $USER_DATASET_DIR"
else
  echo "Note: dataset_dir not found. Dataset on disk checks on your data will be skipped."
fi
echo "OK"

banner "[U1] User JSONL infer: check structure and system policy (mode=infer, profile=$PROFILE)"
mmqlint check "$USER_INFER_JSONL" --mode infer \
  --profile "$PROFILE" --profile-file "$PROFILES_YAML" --fail-on ERROR
echo "OK"

banner "[U2] User JSONL train: check supervision policy (mode=train, profile=$PROFILE)"
if [[ -f "$USER_TRAIN_JSONL" ]]; then
  mmqlint check "$USER_TRAIN_JSONL" --mode train \
    --profile "$PROFILE" --profile-file "$PROFILES_YAML" --fail-on ERROR
  echo "OK"
else
  echo "SKIP"
fi

banner "[U3] User JSONL infer: verify system is visible after rendering (profile=$PROFILE)"
mmqlint verify-system "$USER_INFER_JSONL" \
  --profile "$PROFILE" --profile-file "$PROFILES_YAML" \
  --render-plugin "$TESTS_DIR/render_plugin.py" \
  --fail-on ERROR
echo "OK"

banner "[U4] User dataset on disk: nested None and coordinate sanity"
if [[ -d "$USER_DATASET_DIR" ]]; then
  mmqlint check-dataset "$USER_DATASET_DIR" --fail-on ERROR

  mmqlint check-dataset "$USER_DATASET_DIR" \
    --coord-field "$COORD_FIELD" --coord-keys "$COORD_KEYS" \
    --fail-on WARN

  echo "OK"
else
  echo "SKIP"
fi

banner "ALL DONE: user checks completed"
