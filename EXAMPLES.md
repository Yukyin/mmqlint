# Examples for mmqlint

This file holds longer examples so the README can stay short.

If you are new, start with the one command runner in the README Quickstart.

## 0 One command runner

Create an environment and install

```bash
conda create -y -n mmqlint python=3.10
conda activate mmqlint
pip install -e .
pip install -r requirements.txt
```

Run the regression runner

```bash
bash tests/run_doc_bug_coverage.sh
```

Two runner settings

- KEEP_ARTIFACTS equals 1 keeps tests dot artifacts for inspection.
- KEEP_ARTIFACTS equals 0 cleans up at exit.

```bash
KEEP_ARTIFACTS=0 bash tests/run_doc_bug_coverage.sh
```

## 1 Profiles file example

A minimal profiles yaml that turns on system required and system visible policies.

```yaml
profiles:
  generic:
    require_system: true
    system_visible: true
    system_invisible_level: ERROR
    fold_system_into_user: false

  my-vlm:
    require_system: true
    system_visible: true
    system_invisible_level: ERROR
    fold_system_into_user: false
```

Validate it

```bash
mmqlint validate-profiles --profile-file profiles.yaml
mmqlint list-profiles --profile-file profiles.yaml
```

## 2 Minimal JSONL examples

### Train style example

One sample with a system message, a user turn, and an assistant turn.

```json
{"id":"train-1","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Say hello."},{"role":"assistant","content":"Hello."}]}
```

### Infer style example

One sample with a system message and a user turn.

```json
{"id":"infer-1","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Say hello."}]}
```

## 3 Check your own JSONL

Train mode check

```bash
mmqlint check train.jsonl --mode train --profile my-vlm --profile-file profiles.yaml --fail-on ERROR
```

Infer mode check

```bash
mmqlint check infer.jsonl --mode infer --profile my-vlm --profile-file profiles.yaml --fail-on ERROR
```

Strict CI style

```bash
mmqlint check infer.jsonl --mode infer --profile my-vlm --profile-file profiles.yaml --fail-on WARN
```

## 4 System visibility after rendering

When to use

- You have a chat template or prompt builder that converts messages into a prompt string.
- You want to ensure the system message is still present after rendering.

Run with your render plugin

```bash
mmqlint verify-system infer.jsonl --profile my-vlm --profile-file profiles.yaml --render-plugin render_plugin.py --fail-on ERROR
```

## 5 Demo dataset on disk

Goal

- Build a tiny Hugging Face dataset, save it to disk, and run checks on the saved directory.

```python
from pathlib import Path
from datasets import Dataset, Features, Value, Image as HFImage
from PIL import Image
import numpy as np

def save_img(dir_path: Path, name: str, w: int, h: int) -> str:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    p = dir_path / name
    Image.fromarray(arr).save(p)
    return str(p)

work = Path("artifacts")
img_dir = work / "demo_images"
img_dir.mkdir(parents=True, exist_ok=True)

features = Features({
    "id": Value("string"),
    "image": HFImage(decode=True),
    "image_w": Value("int32"),
    "image_h": Value("int32"),
    "meta": {"a": Value("int32"), "b": Value("int32")},
    "coordinates": {"x0": Value("int32"), "y0": Value("int32"), "x1": Value("int32"), "y1": Value("int32")},
})

rows = [
    {
        "id": "ok",
        "image": save_img(img_dir, "ok.png", 512, 512),
        "image_w": 512, "image_h": 512,
        "meta": {"a": 1, "b": 2},
        "coordinates": {"x0": 10, "y0": 10, "x1": 100, "y1": 100},
    },
]
Dataset.from_list(rows).cast(features).save_to_disk(str(work / "demo_ds_all_ok"))
```

Check the dataset directory

```bash
mmqlint check-dataset artifacts/demo_ds_all_ok --fail-on ERROR
```

## 6 Nested None pitfall after cast

This reproduces the common pitfall where missing nested keys become None after casting and saving.

```python
from datasets import Dataset

rows_bad_none = [
    {
        "id": "missing_meta_b",
        "image": "artifacts/demo_images/m1.png",
        "image_w": 512, "image_h": 512,
        "meta": {"a": 1},
        "coordinates": {"x0": 10, "y0": 10, "x1": 100, "y1": 100},
    },
    {
        "id": "missing_coordinates_y1",
        "image": "artifacts/demo_images/m2.png",
        "image_w": 512, "image_h": 512,
        "meta": {"a": 1, "b": 2},
        "coordinates": {"x0": 10, "y0": 10, "x1": 100},
    },
]

Dataset.from_list(rows_bad_none).cast(features).save_to_disk(str(work / "demo_ds_img_bad_none"))
```

Expected outcome

- Some nested fields are filled as None after cast.
- The dataset check should report E102_DATASET_NONE_VALUE.

Run

```bash
mmqlint check-dataset artifacts/demo_ds_img_bad_none --fail-on ERROR
```

## 7 Coordinate sanity scenario

When your images are 512 by 512 but coordinates look like 0 to 1000.

Build a dataset with out of bounds coordinates

```python
rows_coord_warn = [
    {
        "id": "coords_look_like_1000_space",
        "image": "artifacts/demo_images/c1.png",
        "image_w": 512, "image_h": 512,
        "meta": {"a": 1, "b": 2},
        "coordinates": {"x0": 50, "y0": 80, "x1": 900, "y1": 950},
    },
]
Dataset.from_list(rows_coord_warn).cast(features).save_to_disk(str(work / "demo_ds_coord_warn"))
```

Run with coordinate checks enabled

```bash
mmqlint check-dataset artifacts/demo_ds_coord_warn --coord-field coordinates --coord-keys x0,y0,x1,y1 --fail-on WARN
```

Expected outcome

- The tool reports W301_COORD_OUT_OF_BOUNDS.
- Because fail-on is WARN, the command exits non zero.

## 8 Merge your own JSON samples with a public dataset for SFT

Goal

- Take your own small set of samples in memory.
- Load a public dataset from Hugging Face.
- Convert both into the same messages based chat format.
- Concatenate and save.

Minimal example

```python
from datasets import load_dataset, Dataset, concatenate_datasets

my_rows = [
    {"id": "mine-1", "messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}
]
ds_mine = Dataset.from_list(my_rows)

ds_public = load_dataset("HuggingFaceH4/ultrachat_200k", split="train[:100]")
def to_messages(ex, i):
    return {"id": f"pub-{i}", "messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": ex["prompt"]}, {"role": "assistant", "content": ex["response"]}]}
ds_public2 = ds_public.map(to_messages, with_indices=True, remove_columns=ds_public.column_names)

ds = concatenate_datasets([ds_mine, ds_public2])
ds.save_to_disk("artifacts/sft_merged_ds")
```

Then run your gate

```bash
mmqlint check-dataset artifacts/sft_merged_ds --fail-on ERROR
```

## 9 CI style gate

Example CI steps

- install
- run runner once
- then run strict checks on your own data

```bash
bash tests/run_doc_bug_coverage.sh
mmqlint check train.jsonl --mode train --profile my-vlm --profile-file profiles.yaml --fail-on WARN
mmqlint check infer.jsonl --mode infer --profile my-vlm --profile-file profiles.yaml --fail-on WARN
```

