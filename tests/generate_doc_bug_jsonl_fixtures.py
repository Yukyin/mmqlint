#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate synthetic fixtures for the doc-bug regression checks.

This single generator produces:
- JSONL fixtures for checks A1-A5 and B1-B2
- Hugging Face dataset-on-disk fixtures for checks C1-C3

It does NOT print any fixture contents. It only writes files/directories.

Outputs (under --out):
JSONL
- no_system.jsonl
- system_blank.jsonl
- train_missing_assistant.jsonl
- infer_has_assistant.jsonl
- typed_ok.jsonl

Datasets (HF save_to_disk directories)
- demo_ds_img_bad_none
- demo_ds_coord_warn
- demo_ds_all_ok
- demo_images (helper images for the datasets)

Default output directory:
- tests/.artifacts/tmp.<random>

Usage:
  python tests/generate_doc_bug_jsonl_fixtures.py --out tests/.artifacts/tmp.docbug --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import string
import secrets
from pathlib import Path
from typing import Any, Dict, List


def _rand_suffix(n: int = 6) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(n))


def _ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise SystemExit(f"ERROR: output path exists: {path} (pass --overwrite to replace)")
        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def generate_jsonl_fixtures(out_dir: Path) -> None:
    # A1: missing system (infer)
    _write_jsonl(
        out_dir / "no_system.jsonl",
        [{"id": "t_no_system", "messages": [{"role": "user", "content": "hi"}]}],
    )

    # A2: system blank / empty (infer)
    _write_jsonl(
        out_dir / "system_blank.jsonl",
        [
            {"id": "t_blank", "messages": [{"role": "system", "content": "   "}, {"role": "user", "content": "hi"}]},
            {"id": "t_empty", "messages": [{"role": "system", "content": ""}, {"role": "user", "content": "hi"}]},
        ],
    )

    # A3: train missing assistant (train)
    _write_jsonl(
        out_dir / "train_missing_assistant.jsonl",
        [{"id": "t_train_missing_assistant", "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]}],
    )

    # A4: infer has assistant (infer)
    _write_jsonl(
        out_dir / "infer_has_assistant.jsonl",
        [{"id": "t_infer_has_assistant", "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "leak"}]}],
    )

    # A5 / B1-B2: strict typed ok sample (infer)
    # Typed content format: content is a list of blocks. Minimal "text" blocks only.
    _write_jsonl(
        out_dir / "typed_ok.jsonl",
        [
            {
                "id": "t_typed_ok",
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": "sys policy"}]},
                    {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                ],
            }
        ],
    )


def generate_dataset_fixtures(out_dir: Path) -> None:
    # Keep imports local so users who only want JSONL fixtures can still import the module.
    from datasets import Dataset, Features, Value, Image as HFImage  # type: ignore
    from PIL import Image  # type: ignore
    import numpy as np  # type: ignore

    img_dir = out_dir / "demo_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    def save_img(name: str, w: int, h: int) -> str:
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        p = img_dir / name
        Image.fromarray(arr).save(p)
        return str(p)

    features = Features(
        {
            "id": Value("string"),
            "image": HFImage(),
            "image_w": Value("int32"),
            "image_h": Value("int32"),
            "meta": {"a": Value("int32"), "b": Value("int32")},
            "coordinates": {"x0": Value("int32"), "y0": Value("int32"), "x1": Value("int32"), "y1": Value("int32")},
        }
    )

    # C2 fixture: nested keys missing -> cast fills None
    rows_bad_none = [
        {
            "id": "ok0",
            "image": save_img("ok0.png", 512, 512),
            "image_w": 512,
            "image_h": 512,
            "meta": {"a": 1, "b": 2},
            "coordinates": {"x0": 10, "y0": 10, "x1": 100, "y1": 120},
        },
        {
            "id": "missing_meta_b",
            "image": save_img("m0.png", 512, 512),
            "image_w": 512,
            "image_h": 512,
            "meta": {"a": 9},  # missing b -> cast fills None
            "coordinates": {"x0": 5, "y0": 5, "x1": 60, "y1": 60},
        },
        {
            "id": "missing_coordinates_y1",
            "image": save_img("m1.png", 512, 512),
            "image_w": 512,
            "image_h": 512,
            "meta": {"a": 1, "b": 2},
            "coordinates": {"x0": 10, "y0": 10, "x1": 100},  # missing y1 -> cast fills None
        },
    ]
    ds_bad_none = Dataset.from_list(rows_bad_none).cast(features)
    out_bad_none = out_dir / "demo_ds_img_bad_none"
    if out_bad_none.exists():
        shutil.rmtree(out_bad_none)
    ds_bad_none.save_to_disk(str(out_bad_none))

    # C3 fixture: coords out of bounds for 512x512
    rows_coord_warn = [
        {
            "id": "coords_look_like_1000_space",
            "image": save_img("c1.png", 512, 512),
            "image_w": 512,
            "image_h": 512,
            "meta": {"a": 1, "b": 2},
            "coordinates": {"x0": 50, "y0": 80, "x1": 900, "y1": 950},
        }
    ]
    ds_coord_warn = Dataset.from_list(rows_coord_warn).cast(features)
    out_coord_warn = out_dir / "demo_ds_coord_warn"
    if out_coord_warn.exists():
        shutil.rmtree(out_coord_warn)
    ds_coord_warn.save_to_disk(str(out_coord_warn))

    # C4-style positive fixture (used in some suites): all-ok dataset
    rows_all_ok = [
        {
            "id": "ok1",
            "image": save_img("a1.png", 640, 480),
            "image_w": 640,
            "image_h": 480,
            "meta": {"a": 1, "b": 2},
            "coordinates": {"x0": 10, "y0": 10, "x1": 200, "y1": 200},
        },
        {
            "id": "ok2",
            "image": save_img("a2.png", 512, 512),
            "image_w": 512,
            "image_h": 512,
            "meta": {"a": 3, "b": 4},
            "coordinates": {"x0": 0, "y0": 0, "x1": 511, "y1": 511},
        },
    ]
    ds_all_ok = Dataset.from_list(rows_all_ok).cast(features)
    out_all_ok = out_dir / "demo_ds_all_ok"
    if out_all_ok.exists():
        shutil.rmtree(out_all_ok)
    ds_all_ok.save_to_disk(str(out_all_ok))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="",
        help="Output directory to write fixtures. Default: tests/.artifacts/tmp.<random>",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it exists.")
    args = ap.parse_args()

    if args.out.strip():
        out_dir = Path(args.out)
        _ensure_empty_dir(out_dir, overwrite=args.overwrite)
    else:
        out_dir = Path("tests") / ".artifacts" / f"tmp.{_rand_suffix()}"
        out_dir.mkdir(parents=True, exist_ok=True)

    generate_jsonl_fixtures(out_dir)
    generate_dataset_fixtures(out_dir)

    print(f"[mmqlint] fixtures written under: {out_dir}")


if __name__ == "__main__":
    main()
