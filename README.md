# mmqlint

mmqlint is a lightweight quality gate for LLM and VLM training and inference datasets, and for the prompt rendering pipeline that consumes them.

It supports common data formats for SFT datasets, inference inputs, and VQA or VLM style chat conversations.

It is meant to run before expensive fine-tuning, RLHF style training, or large scale batch inference, so you can catch format drift and silent data corruption early.

## What this tool does

mmqlint focuses on problems that are easy to miss when you only eyeball a few samples, but can meaningfully affect model behavior and training signal.

It can help you validate

- Chat style data stored as JSONL, as commonly used for LLM and VLM training and inference
  - Structure and required fields
  - Train versus infer conventions
  - Typed content formats for multimodal messages
- Prompt rendering pipelines used to build model inputs
  - Confirms that the final rendered prompt still contains required system content
  - Detects pipelines that accidentally drop system messages
- Hugging Face datasets saved on disk, often used in large scale training and evaluation
  - Detects nested fields that became None after casting or saving
  - Coordinate sanity checks for image related metadata

## Why you may need this tool

Large runs fail late, and some bugs do not fail at all. They silently reduce model quality or invalidate evaluation.

Common examples

- A system message exists in JSONL, but the rendering pipeline removes it, changing LLM or VLM behavior.
- A dataset declares nested keys in its features, but some rows are missing those keys. After casting, missing keys are filled with None and then saved. Downstream training or evaluation code may crash, or the model may learn from corrupted supervision.
- Inference inputs accidentally include assistant turns, leaking labels and inflating apparent quality.
- Coordinate fields look reasonable but are in a different coordinate system than the image size, breaking vision grounding.

mmqlint is designed to turn these into fast, deterministic checks with stable issue codes you can gate on in CI.


## Quickstart

These steps assume you are in the repo root.

### 1 Install relative packages

```bash
conda create -y -n mmqlint python=3.10
conda activate mmqlint
pip install -r requirements.txt
pip install -e .
```

### 2 Run the one command with samples

```bash
bash tests/run_doc_bug_coverage.sh
```

What the runner checks

- Installation and CLI sanity
- Profile validation
- JSONL policy checks for train and infer conventions
- System policy in two layers
  - system required in JSONL and system content not empty
  - system still visible in the final rendered prompt
- Dataset on disk checks, including nested None after casting or saving, and coordinate sanity

Two runner settings

- KEEP_ARTIFACTS = 1
  - Default. Keeps `tests/.artifacts/` so you can inspect generated inputs and outputs.
  - Runs the same checks as KEEP_ARTIFACTS equals 0.
- KEEP_ARTIFACTS = 0
  - Deletes the work directory at exit.

```bash
KEEP_ARTIFACTS=0 bash tests/run_doc_bug_coverage.sh
```

How to read the output

- Each step prints a banner like [3] A1 and then prints OK on success.
- Some steps are expected to fail. The runner confirms the expected issue code appears in the output.
- If anything behaves differently than expected, the runner exits non zero.

When an issue is printed, the most important fields are

- level is WARN or ERROR
- code is the stable issue code you can gate on
- sample id tells you which record failed
- path points to the failing field


## Separate check steps

This section explains each check performed by `tests/run_doc_bug_coverage.sh`.

### A1: infer must have system

How it is done  
- Runs a JSONL check in infer mode on a demo file that is missing the system message.

What it checks  
- System required in JSONL for inference inputs.

Expected result  
- Fail with issue code E002_SYSTEM_REQUIRED.

Commands  
```bash
# Generate fixtures (run once per WORK_DIR)
WORK_DIR=tests/.artifacts/tmp.docbug
python tests/generate_doc_bug_jsonl_fixtures.py --out "$WORK_DIR" --overwrite

# Run the check
mmqlint check "$WORK_DIR/no_system.jsonl" --mode infer \
  --profile my-vlm --profile-file profiles.yaml --fail-on ERROR
```
### A2: system blank or empty

How it is done  
- Runs a JSONL check in infer mode on a demo file where system content is empty.

What it checks  
- System exists but has no usable content.

Expected result  
- Fail with issue code E002_SYSTEM_EMPTY.

Commands  
```bash
# Generate fixtures (run once per WORK_DIR)
WORK_DIR=tests/.artifacts/tmp.docbug
python tests/generate_doc_bug_jsonl_fixtures.py --out "$WORK_DIR" --overwrite

# Run the check
mmqlint check "$WORK_DIR/system_blank.jsonl" --mode infer \
  --profile my-vlm --profile-file profiles.yaml --fail-on ERROR
```
### A3: train must have assistant label

How it is done  
- Runs a JSONL check in train mode on a demo file that has only user turns.

What it checks  
- Training samples must contain at least one assistant turn.

Expected result  
- Fail with issue code E001_TRAIN_MISSING_ASSISTANT.

Commands  
```bash
# Generate fixtures (run once per WORK_DIR)
WORK_DIR=tests/.artifacts/tmp.docbug
python tests/generate_doc_bug_jsonl_fixtures.py --out "$WORK_DIR" --overwrite

# Run the check
mmqlint check "$WORK_DIR/train_missing_assistant.jsonl" --mode train \
  --profile my-vlm --profile-file profiles.yaml --fail-on ERROR
```
### A4: infer must not contain assistant

How it is done  
- Runs a JSONL check in infer mode on a demo file that includes an assistant turn.

What it checks  
- Inference inputs must not contain assistant content, to avoid label leakage.

Expected result  
- Fail with issue code E001_INFER_HAS_ASSISTANT.

Commands  
```bash
# Generate fixtures (run once per WORK_DIR)
WORK_DIR=tests/.artifacts/tmp.docbug
python tests/generate_doc_bug_jsonl_fixtures.py --out "$WORK_DIR" --overwrite

# Run the check
mmqlint check "$WORK_DIR/infer_has_assistant.jsonl" --mode infer \
  --profile my-vlm --profile-file profiles.yaml --fail-on ERROR
```
### A5: strict typed content passes

How it is done  
- Runs a JSONL check under strict typed content rules on a well formed typed sample.

What it checks  
- Typed content format is accepted and does not regress.

Expected result  
- Pass.

Commands  
```bash
# Generate fixtures (run once per WORK_DIR)
WORK_DIR=tests/.artifacts/tmp.docbug
python tests/generate_doc_bug_jsonl_fixtures.py --out "$WORK_DIR" --overwrite

# Run the check
mmqlint check "$WORK_DIR/typed_ok.jsonl" --mode infer \
  --profile my-vlm --profile-file profiles.yaml --strict-typed --fail-on ERROR
```
### B1: system visible after rendering

How it is done  
- Runs a system visibility check using a normal render plugin that preserves system messages.

What it checks  
- Rendering keeps system content present in the final prompt string.

Expected result  
- Pass.

Commands  
```bash
# Generate fixtures (run once per WORK_DIR)
WORK_DIR=tests/.artifacts/tmp.docbug
python tests/generate_doc_bug_jsonl_fixtures.py --out "$WORK_DIR" --overwrite

# Run the check
mmqlint verify-system "$WORK_DIR/typed_ok.jsonl" \
  --profile my-vlm --profile-file profiles.yaml \
  --render-plugin tests/render_plugin.py \
  --strict-typed --fail-on ERROR
```
### B2: pipeline drops system

How it is done  
- Runs the same system visibility check using a render plugin that intentionally drops system messages.

What it checks  
- Detects render pipelines that accidentally remove system messages.

Expected result  
- Fail with issue code E002_SYSTEM_NOT_VISIBLE.

Commands  
```bash
# Generate fixtures (run once per WORK_DIR)
WORK_DIR=tests/.artifacts/tmp.docbug
python tests/generate_doc_bug_jsonl_fixtures.py --out "$WORK_DIR" --overwrite

# Run the check
mmqlint verify-system "$WORK_DIR/typed_ok.jsonl" \
  --profile my-vlm --profile-file profiles.yaml \
  --render-plugin tests/render_plugin_drop_system.py \
  --strict-typed --fail-on ERROR
```
### C1: build demo datasets on disk

How it is done  
- Runs a small builder that creates Hugging Face datasets and saves them to disk under `tests/.artifacts/`.

What it checks  
- Reproducible dataset fixtures exist for downstream checks.

Expected result  
- Pass.

Commands  
```bash
# Generate fixtures (run once per WORK_DIR)
WORK_DIR=tests/.artifacts/tmp.docbug
python tests/generate_doc_bug_jsonl_fixtures.py --out "$WORK_DIR" --overwrite

# Optional sanity: show the dataset fixture directories
ls -la "$WORK_DIR/demo_ds_img_bad_none" "$WORK_DIR/demo_ds_coord_warn" "$WORK_DIR/demo_ds_all_ok"
```
### C2: dataset nested None check

How it is done  
- Runs a dataset check on a dataset where nested keys were missing in some rows and became None after casting or saving.

What it checks  
- Detects the common cast pitfall where declared nested keys become None values.

Expected result  
- Fail with issue code E102_DATASET_NONE_VALUE.

Commands  
```bash
# Generate fixtures (run once per WORK_DIR)
WORK_DIR=tests/.artifacts/tmp.docbug
python tests/generate_doc_bug_jsonl_fixtures.py --out "$WORK_DIR" --overwrite

# Run the check
mmqlint check-dataset "$WORK_DIR/demo_ds_img_bad_none" --fail-on ERROR
```
### C3: coordinate sanity

How it is done  
- Runs a dataset check with coordinate checking enabled and a fail on WARN threshold.

What it checks  
- Coordinates are compared against the real image size. Out of bounds coordinates are flagged.

Expected result  
- Warn with code W301_COORD_OUT_OF_BOUNDS and the step fails because fail on WARN is used.

Commands  
```bash
# Generate fixtures (run once per WORK_DIR)
WORK_DIR=tests/.artifacts/tmp.docbug
python tests/generate_doc_bug_jsonl_fixtures.py --out "$WORK_DIR" --overwrite

# Run the check
mmqlint check-dataset "$WORK_DIR/demo_ds_coord_warn" \
  --coord-field coordinates --coord-keys x0,y0,x1,y1 \
  --fail-on WARN
```

## Quickstart with your own data

Run the one-shot check on an existing `my_inputs` folder:

```bash
MY_INPUT=/path/to/my_inputs
mkdir -p "$MY_INPUT"
bash tests/run_doc_bug_coverage_user.sh "$MY_INPUT"
```

`/path/to/my_inputs` is a directory with the following layout:

```text
my_inputs/
  infer.jsonl                  # required
  train.jsonl                  # optional but recommended (enables train-mode checks)
  dataset_dir/                 # optional but recommended (enables dataset-on-disk checks)
    dataset_info.json
    state.json
    data-00000-of-00001.arrow
    ...

```

### Example: build `my_inputs` from LLaVA style VQA JSON

If you start from a VQA dataset like [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), you can first convert a JSON sample into a `my_inputs` folder, then run the same one-shot check:

```bash
bash tests/mmqlint_llava_my_inputs/make_my_inputs_from_llava.sh \
     tests/mmqlint_llava_my_inputs/llava_sample.json \
     tests/mmqlint_llava_my_inputs \
     tests/my_inputs 500
bash tests/run_doc_bug_coverage_user.sh tests/my_inputs
```

- The first command converts the LLaVA style JSON sample into `tests/my_inputs`.
- The last argument `500` is the max number of records to export.
- Make sure the image paths referenced by the JSON can be resolved on your machine.


## Real world scenarios

The fastest way to learn is to map each check to situations you actually hit.

#### Scenario 1 Before training a chat model

Run the JSONL checks in train mode to ensure each sample includes an assistant turn and required system content.

What you gain  
- Prevents silent training on user only logs  
- Keeps system policy consistent across the dataset

#### Scenario 2 Before batch inference

Run the JSONL checks in infer mode to ensure inference inputs include a valid system message and do not include assistant content.

What you gain  
- Avoids label leakage  
- Avoids inconsistent prompting

#### Scenario 3 When you change your chat template or prompt builder

Run the system visibility checks.

What you gain  
- Catches rendering changes that remove the system message

#### Scenario 4 After casting and saving a Hugging Face dataset

Run the dataset on disk checks.

What you gain  
- Detects nested None values caused by missing nested keys  
- Detects coordinate mismatches when coordinate sanity is enabled


#### 💡For more real-world scenarios, see [EXAMPLES.md](EXAMPLES.md).

