#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create my_inputs files for run_doc_bug_coverage_user.sh from LLaVA v1.5 style JSON.

Input JSON format
- A list of items. Each item contains:
  - id: string
  - image: string, path relative to an image root
  - conversations: list of turns
    - from: "human" or "gpt"
    - value: string, may contain "<image>" token

Outputs under --out-dir
- infer.jsonl
  One record per item.
  Contains only system + the first human turn, no assistant turns.
  This matches infer mode rules used by the doc-bug runner.

- train.jsonl
  One record per item.
  Contains system + all turns mapped to user and assistant roles.

- typed_infer.jsonl and typed_train.jsonl
  Same records but all message contents are typed lists.
  User turns become a list that includes the image path and the text.
  Assistant turns become a typed text list.

- dataset_dir
  A Hugging Face dataset saved to disk.
  Contains image, image_w, image_h, and a coordinates struct that is within bounds.
  This lets the coordinate sanity check run without failing.

Notes
- You must provide --image-root and the images must exist, because image size is read from disk.
- If you do not want dataset_dir generation, pass --skip-dataset-dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

def load_llava_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("Expected the input JSON to be a list.")
    return data

def normalize_text(value: str) -> str:
    if value is None:
        return ""
    s = str(value)
    s = s.replace("<image>", "").strip()
    # Remove a leading newline if the original value started with "<image>\n"
    while s.startswith("\n"):
        s = s[1:]
    return s.strip()

def abs_image_path(image_root: Path, rel_path: str) -> str:
    p = Path(rel_path)
    if p.is_absolute():
        return str(p)
    return str((image_root / p).resolve())

def make_system(system_text: str) -> Dict[str, Any]:
    return {"role": "system", "content": system_text}

def make_user_text(text: str) -> Dict[str, Any]:
    return {"role": "user", "content": text}

def make_assistant_text(text: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": text}

def make_user_typed(image_path: str, text: str) -> Dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": text},
        ],
    }

def make_text_typed(role: str, text: str) -> Dict[str, Any]:
    return {"role": role, "content": [{"type": "text", "text": text}]}

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_image_size(image_path: str) -> Tuple[int, int]:
    try:
        from PIL import Image
    except Exception as e:
        raise SystemExit("Pillow is required to read image sizes. Install with: pip install pillow") from e

    with Image.open(image_path) as im:
        w, h = im.size
    return int(w), int(h)

def build_dataset_dir(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    try:
        from datasets import Dataset, Features, Value, Image as HFImage
    except Exception as e:
        raise SystemExit("datasets is required to build dataset_dir. Install with: pip install datasets") from e

    features = Features(
        {
            "id": Value("string"),
            "image": HFImage(decode=True),
            "image_w": Value("int32"),
            "image_h": Value("int32"),
            "coordinates": {
                "x0": Value("int32"),
                "y0": Value("int32"),
                "x1": Value("int32"),
                "y1": Value("int32"),
            },
            "prompt": Value("string"),
            "answer": Value("string"),
        }
    )

    ds = Dataset.from_list(rows).cast(features)
    ds.save_to_disk(str(out_dir))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llava-json", required=True, help="Path to llava_v1_5_mix665k.json or a subset of it")
    ap.add_argument("--image-root", required=True, help="Root directory that contains the image files")
    ap.add_argument("--out-dir", default="my_inputs", help="Output directory, default my_inputs")
    ap.add_argument("--max-records", type=int, default=500, help="Limit the number of items")
    ap.add_argument(
        "--system-text",
        default="You are a helpful assistant.",
        help="System message to prepend",
    )
    ap.add_argument(
        "--skip-dataset-dir",
        action="store_true",
        help="Do not generate dataset_dir",
    )
    args = ap.parse_args()

    llava_json = Path(args.llava_json)
    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_llava_json(llava_json)
    if args.max_records and args.max_records > 0:
        items = items[: args.max_records]

    infer_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    typed_infer_rows: List[Dict[str, Any]] = []
    typed_train_rows: List[Dict[str, Any]] = []
    dataset_rows: List[Dict[str, Any]] = []

    for it in items:
        sid = str(it.get("id", ""))
        rel_img = str(it.get("image", ""))
        conv = it.get("conversations", [])
        if not sid or not rel_img or not isinstance(conv, list) or not conv:
            continue

        img_path = abs_image_path(image_root, rel_img)

        # First human and first gpt for dataset prompt/answer.
        first_human = ""
        first_gpt = ""
        for turn in conv:
            if turn.get("from") == "human" and not first_human:
                first_human = normalize_text(turn.get("value", ""))
            if turn.get("from") == "gpt" and not first_gpt:
                first_gpt = normalize_text(turn.get("value", ""))
            if first_human and first_gpt:
                break

        # infer.jsonl uses only the first human turn and no assistant turns.
        infer_messages = [make_system(args.system_text), make_user_text(first_human)]
        infer_rows.append({"id": sid, "messages": infer_messages})

        typed_infer_messages = [make_text_typed("system", args.system_text), make_user_typed(img_path, first_human)]
        typed_infer_rows.append({"id": sid, "messages": typed_infer_messages})

        # train.jsonl includes full conversation with roles mapped.
        train_messages: List[Dict[str, Any]] = [make_system(args.system_text)]
        typed_train_messages: List[Dict[str, Any]] = [make_text_typed("system", args.system_text)]

        for t in conv:
            who = t.get("from")
            val = normalize_text(t.get("value", ""))
            if who == "human":
                train_messages.append(make_user_text(val))
                typed_train_messages.append(make_user_typed(img_path, val))
            elif who == "gpt":
                train_messages.append(make_assistant_text(val))
                typed_train_messages.append(make_text_typed("assistant", val))

        train_rows.append({"id": sid, "messages": train_messages})
        typed_train_rows.append({"id": sid, "messages": typed_train_messages})

        # dataset rows
        if not args.skip_dataset_dir:
            w, h = read_image_size(img_path)
            dataset_rows.append(
                {
                    "id": sid,
                    "image": img_path,
                    "image_w": w,
                    "image_h": h,
                    # within bounds coordinates
                    "coordinates": {"x0": 0, "y0": 0, "x1": max(0, w - 1), "y1": max(0, h - 1)},
                    "prompt": first_human,
                    "answer": first_gpt,
                }
            )

    write_jsonl(out_dir / "infer.jsonl", infer_rows)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "typed_infer.jsonl", typed_infer_rows)
    write_jsonl(out_dir / "typed_train.jsonl", typed_train_rows)

    if not args.skip_dataset_dir:
        build_dataset_dir(dataset_rows, out_dir / "dataset_dir")

    print("Wrote:")
    print(" -", out_dir / "infer.jsonl")
    print(" -", out_dir / "train.jsonl")
    print(" -", out_dir / "typed_infer.jsonl")
    print(" -", out_dir / "typed_train.jsonl")
    if not args.skip_dataset_dir:
        print(" -", out_dir / "dataset_dir")

if __name__ == "__main__":
    main()
