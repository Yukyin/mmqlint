#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# mmqlint doc-bug coverage runner
# - All demo artifacts go under: tests/.artifacts/
# - Auto-add "tests/.artifacts/" to repo root .gitignore (if missing)
# ==============================================================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTS_DIR="$ROOT/tests"
ART_DIR="$TESTS_DIR/.artifacts"
PROFILES_YAML="$ROOT/profiles.yaml"

# Keep artifacts by default (so you can inspect). Set KEEP_ARTIFACTS=0 to cleanup.
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

# --- helpers ---
banner() { echo "== $* =="; }

run_ok() {
  "$@"
}

run_expect_fail_with_code() {
  # usage: run_expect_fail_with_code E001_SOMETHING cmd args...
  local want_code="$1"; shift
  local out
  set +e
  out="$("$@" 2>&1)"
  local rc=$?
  set -e
  echo "$out"
  if [[ $rc -eq 0 ]]; then
    echo "ERROR: expected failure (non-zero exit), but command succeeded."
    exit 1
  fi
  if ! echo "$out" | grep -q "$want_code"; then
    echo "ERROR: expected to find code '$want_code' in output, but not found."
    exit 1
  fi
}

# ==============================================================================
banner "[0] Sanity: editable install"
# ==============================================================================
run_ok python -m pip install -e "$ROOT" >/dev/null

# ==============================================================================
banner "[1] CLI sanity: commands exist"
# ==============================================================================
run_ok mmqlint --help >/dev/null
run_ok mmqlint validate-profiles --help >/dev/null
run_ok mmqlint check --help >/dev/null
run_ok mmqlint verify-system --help >/dev/null
run_ok mmqlint check-dataset --help >/dev/null
echo "OK"

# ==============================================================================
banner "[2] Profiles policy: require system + visible (schema-level policy)"
# ==============================================================================
run_ok mmqlint validate-profiles --profile-file "$PROFILES_YAML"
echo "OK"

# ==============================================================================
banner "[3] A1: infer must have system; missing system -> ERROR E002_SYSTEM_REQUIRED"
# ==============================================================================
cat > "$WORK_DIR/no_system.jsonl" <<'EOF'
{"id":"t_no_system","messages":[{"role":"user","content":"hi"}]}
EOF
run_expect_fail_with_code "E002_SYSTEM_REQUIRED" \
  mmqlint check "$WORK_DIR/no_system.jsonl" --mode infer \
    --profile my-vlm --profile-file "$PROFILES_YAML" --fail-on ERROR
echo "OK"

# ==============================================================================
banner "[4] A2: system blank/empty -> ERROR E002_SYSTEM_EMPTY"
# ==============================================================================
cat > "$WORK_DIR/system_blank.jsonl" <<'EOF'
{"id":"t_blank","messages":[{"role":"system","content":"   "},{"role":"user","content":"hi"}]}
{"id":"t_empty","messages":[{"role":"system","content":""},{"role":"user","content":"hi"}]}
EOF
run_expect_fail_with_code "E002_SYSTEM_EMPTY" \
  mmqlint check "$WORK_DIR/system_blank.jsonl" --mode infer \
    --profile my-vlm --profile-file "$PROFILES_YAML" --fail-on ERROR
echo "OK"

# ==============================================================================
banner "[5] A3: train must have assistant label; user-only in train -> ERROR E001_TRAIN_MISSING_ASSISTANT"
# ==============================================================================
cat > "$WORK_DIR/train_missing_assistant.jsonl" <<'EOF'
{"id":"t_train_missing_assistant","messages":[{"role":"system","content":"sys"},{"role":"user","content":"hi"}]}
EOF
run_expect_fail_with_code "E001_TRAIN_MISSING_ASSISTANT" \
  mmqlint check "$WORK_DIR/train_missing_assistant.jsonl" --mode train \
    --profile my-vlm --profile-file "$PROFILES_YAML" --fail-on ERROR
echo "OK"

# ==============================================================================
banner "[6] A4: infer must NOT contain assistant; assistant present -> ERROR E001_INFER_HAS_ASSISTANT"
# ==============================================================================
cat > "$WORK_DIR/infer_has_assistant.jsonl" <<'EOF'
{"id":"t_infer_has_assistant","messages":[{"role":"system","content":"sys"},{"role":"user","content":"hi"},{"role":"assistant","content":"leak"}]}
EOF
run_expect_fail_with_code "E001_INFER_HAS_ASSISTANT" \
  mmqlint check "$WORK_DIR/infer_has_assistant.jsonl" --mode infer \
    --profile my-vlm --profile-file "$PROFILES_YAML" --fail-on ERROR
echo "OK"

# ==============================================================================
banner "[7] A5: strict typed content passes"
# ==============================================================================
cat > "$WORK_DIR/typed_ok.jsonl" <<'EOF'
{"id":"t_typed_ok","messages":[{"role":"system","content":[{"type":"text","text":"SYSTEM_RULE: must answer in JSON only"}]},{"role":"user","content":[{"type":"text","text":"hi"}]}]}
EOF
run_ok mmqlint check "$WORK_DIR/typed_ok.jsonl" --mode infer \
  --profile my-vlm --profile-file "$PROFILES_YAML" --strict-typed --fail-on ERROR >/dev/null
echo "OK"

# ==============================================================================
banner "[8] B1: verify-system passes with normal render plugin"
# ==============================================================================
run_ok mmqlint verify-system "$WORK_DIR/typed_ok.jsonl" \
  --profile my-vlm --profile-file "$PROFILES_YAML" \
  --render-plugin "$TESTS_DIR/render_plugin.py" \
  --strict-typed --fail-on ERROR >/dev/null
echo "OK"

# ==============================================================================
banner "[9] B2: verify-system fails if pipeline drops system -> ERROR E002_SYSTEM_NOT_VISIBLE"
# ==============================================================================
run_expect_fail_with_code "E002_SYSTEM_NOT_VISIBLE" \
  mmqlint verify-system "$WORK_DIR/typed_ok.jsonl" \
    --profile my-vlm --profile-file "$PROFILES_YAML" \
    --render-plugin "$TESTS_DIR/render_plugin_drop_system.py" \
    --strict-typed --fail-on ERROR
echo "OK"

# ==============================================================================
banner "[10] C1: build demo datasets under tests/.artifacts (HF datasets on-disk)"
# ==============================================================================
export MMQLINT_ART_WORKDIR="$WORK_DIR"
run_ok python - <<'PY'
import os
from pathlib import Path
from datasets import Dataset, Features, Value, Image as HFImage
from PIL import Image
import numpy as np

work = Path(os.environ["MMQLINT_ART_WORKDIR"])
img_dir = work / "demo_images"
img_dir.mkdir(parents=True, exist_ok=True)

def save_img(name: str, w: int, h: int) -> str:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    p = img_dir / name
    Image.fromarray(arr).save(p)
    return str(p)

features = Features({
    "id": Value("string"),
    "image": HFImage(decode=True),
    "image_w": Value("int32"),
    "image_h": Value("int32"),
    "meta": {"a": Value("int32"), "b": Value("int32")},
    "coordinates": {"x0": Value("int32"), "y0": Value("int32"), "x1": Value("int32"), "y1": Value("int32")},
})

# 1) Bad None dataset: missing nested keys -> filled as None
rows_bad_none = [
    {
        "id": "ok",
        "image": save_img("ok.png", 512, 512),
        "image_w": 512, "image_h": 512,
        "meta": {"a": 1, "b": 2},
        "coordinates": {"x0": 10, "y0": 10, "x1": 100, "y1": 100},
    },
    {
        "id": "missing_meta_b",
        "image": save_img("m1.png", 512, 512),
        "image_w": 512, "image_h": 512,
        "meta": {"a": 1},  # missing b
        "coordinates": {"x0": 10, "y0": 10, "x1": 100, "y1": 100},
    },
    {
        "id": "missing_coordinates_y1",
        "image": save_img("m2.png", 512, 512),
        "image_w": 512, "image_h": 512,
        "meta": {"a": 1, "b": 2},
        "coordinates": {"x0": 10, "y0": 10, "x1": 100},  # missing y1
    },
]
ds_bad_none = Dataset.from_list(rows_bad_none).cast(features)
out_bad_none = work / "demo_ds_img_bad_none"
ds_bad_none.save_to_disk(str(out_bad_none))

# 2) Coord warn dataset: all keys present, coords out of bounds
rows_coord_warn = [
    {
        "id": "coords_look_like_1000_space",
        "image": save_img("c1.png", 512, 512),
        "image_w": 512, "image_h": 512,
        "meta": {"a": 1, "b": 2},
        "coordinates": {"x0": 50, "y0": 80, "x1": 900, "y1": 950},  # out of bounds
    },
]
ds_coord_warn = Dataset.from_list(rows_coord_warn).cast(features)
out_coord_warn = work / "demo_ds_coord_warn"
ds_coord_warn.save_to_disk(str(out_coord_warn))

# 3) All ok dataset
rows_all_ok = [
    {
        "id": "ok1",
        "image": save_img("a1.png", 640, 480),
        "image_w": 640, "image_h": 480,
        "meta": {"a": 1, "b": 2},
        "coordinates": {"x0": 10, "y0": 10, "x1": 200, "y1": 200},
    },
    {
        "id": "ok2",
        "image": save_img("a2.png", 512, 512),
        "image_w": 512, "image_h": 512,
        "meta": {"a": 3, "b": 4},
        "coordinates": {"x0": 0, "y0": 0, "x1": 511, "y1": 511},
    },
]
ds_all_ok = Dataset.from_list(rows_all_ok).cast(features)
out_all_ok = work / "demo_ds_all_ok"
ds_all_ok.save_to_disk(str(out_all_ok))

print("Saved datasets to:")
print(" -", out_bad_none)
print(" -", out_coord_warn)
print(" -", out_all_ok)
PY
echo "OK"

# ==============================================================================
banner "[11] C2: dataset nested None check -> ERROR E102_DATASET_NONE_VALUE"
# ==============================================================================
run_expect_fail_with_code "E102_DATASET_NONE_VALUE" \
  mmqlint check-dataset "$WORK_DIR/demo_ds_img_bad_none" --fail-on ERROR
echo "OK"

# ==============================================================================
banner "[12] C3: coord sanity (generic keys via --coord-field/--coord-keys) -> WARN W301_COORD_OUT_OF_BOUNDS"
# ==============================================================================
run_expect_fail_with_code "W301_COORD_OUT_OF_BOUNDS" \
  mmqlint check-dataset "$WORK_DIR/demo_ds_coord_warn" \
    --coord-field coordinates --coord-keys x0,y0,x1,y1 \
    --fail-on WARN
echo "OK"

# ==============================================================================
banner "[13] C4 (positive): all-ok dataset should be 0 issues (even with coord check enabled)"
# ==============================================================================
run_ok mmqlint check-dataset "$WORK_DIR/demo_ds_all_ok" \
  --coord-field coordinates --coord-keys x0,y0,x1,y1 \
  --fail-on ERROR >/dev/null
echo "OK"

banner "ALL DONE: doc-bug coverage tests completed"
