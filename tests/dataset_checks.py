# src/mmqlint/dataset_checks.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

def _is_blank_text(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False

def _get_by_path(obj: Any, path: str) -> Any:
    cur = obj
    for key in path.split("."):
        if not isinstance(cur, dict):
            return None
        if key not in cur:
            return None
        cur = cur[key]
    return cur

def _list_leaf_paths(features: Any, prefix: str = "") -> List[str]:
    """
    HF datasets Features is dict-like; nested structs show as dict of sub-features.
    We traverse dicts; leaf nodes are non-dict feature objects.
    """
    out: List[str] = []
    if isinstance(features, dict):
        for k, v in features.items():
            p = f"{prefix}.{k}" if prefix else k
            out.extend(_list_leaf_paths(v, p))
    else:
        # leaf
        out.append(prefix)
    return out

@dataclass(frozen=True)
class DatasetIssue:
    level: str          # "ERROR" | "WARN" | "INFO"
    code: str           # e.g., "E101_DATASET_NONE_FILLED"
    row: int            # 0-based row index
    sample_id: str      # from a column (default "id") if present, else "row_{i}"
    path: str           # e.g., "meta.b"
    message: str

def lint_dataset_on_disk(
    path: str,
    *,
    id_field: str = "id",
    forbid_none_paths: Optional[List[str]] = None,
    check_all_nested_leaf_nones: bool = True,
    image_w_field: str = "image_w",
    image_h_field: str = "image_h",
    coordinates_prefix: str = "coordinates",
    coordinates_fields: Tuple[str, str, str, str] = ("x0", "y0", "x1", "y1"),
    max_rows: Optional[int] = None,
) -> List[DatasetIssue]:
    """
    Dataset-level checks:
        1) The features declare nested keys, but during writing or saving some keys are missing -> after cast or save they get filled in as None
        2) The coordinate system or resolution for coordinates is inconsistent: coordinates exceed image_w or image_h for example 900,950 appears in a 512x512 image
    """
    try:
        from datasets import load_from_disk, DatasetDict
    except Exception as e:
        return [DatasetIssue(
            level="ERROR",
            code="E100_DATASET_DEP_MISSING",
            row=0,
            sample_id="dataset",
            path="$",
            message=f"Failed to import datasets.load_from_disk. Please `pip install datasets`. Details: {e}",
        )]

    ds_obj = load_from_disk(path)

    # Normalize to list of (split_name, dataset)
    splits: List[Tuple[str, Any]] = []
    if ds_obj.__class__.__name__ == "DatasetDict":  # avoid hard import typing issues
        for k in ds_obj.keys():
            splits.append((k, ds_obj[k]))
    else:
        splits.append(("data", ds_obj))

    issues: List[DatasetIssue] = []

    for split_name, ds in splits:
        # Build leaf paths from features (only for nested structs)
        leaf_paths: List[str] = []
        if check_all_nested_leaf_nones:
            # ds.features is dict-like mapping of columns to Feature
            try:
                feats = ds.features
                # find nested dict columns
                for col, f in feats.items():
                    if isinstance(f, dict):
                        leaf_paths.extend(_list_leaf_paths({col: f}))
            except Exception:
                leaf_paths = []

        if forbid_none_paths:
            leaf_paths.extend([p for p in forbid_none_paths if p not in leaf_paths])

        n = len(ds) if max_rows is None else min(len(ds), max_rows)

        for i in range(n):
            row = ds[i]
            sid = str(row.get(id_field, f"{split_name}:row_{i}"))

            # (1) nested leaf paths None check
            for p in leaf_paths:
                v = _get_by_path(row, p)
                if v is None:
                    issues.append(DatasetIssue(
                        level="ERROR",
                        code="E101_DATASET_NONE_FILLED",
                        row=i,
                        sample_id=sid,
                        path=p,
                        message=f"{p} is None (likely filled due to missing nested key during cast/save).",
                    ))

            # (2) coordinates out-of-bounds check
            iw = row.get(image_w_field, None)
            ih = row.get(image_h_field, None)
            coordinates = row.get(coordinates_prefix, None)

            if isinstance(iw, int) and isinstance(ih, int) and isinstance(coordinates, dict):
                x0 = coordinates.get(coordinates_fields[0], None)
                y0 = coordinates.get(coordinates_fields[1], None)
                x1 = coordinates.get(coordinates_fields[2], None)
                y1 = coordinates.get(coordinates_fields[3], None)

                coords = [x0, y0, x1, y1]
                if any(c is None for c in coords):
                    # If subkeys of coordinates are missing or None, do not report repeatedly here (it will be caught by E101)
                    continue

                # basic sanity
                if not all(isinstance(c, (int, float)) for c in coords):
                    issues.append(DatasetIssue(
                        level="ERROR",
                        code="E102_coordinates_BAD_TYPE",
                        row=i,
                        sample_id=sid,
                        path=coordinates_prefix,
                        message=f"coordinates coords must be numeric; got {coords}",
                    ))
                    continue

                if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
                    issues.append(DatasetIssue(
                        level="ERROR",
                        code="E103_coordinates_NEGATIVE",
                        row=i,
                        sample_id=sid,
                        path=coordinates_prefix,
                        message=f"coordinates has negative coords: {coords}",
                    ))

                if x1 > iw or y1 > ih:
                    # Further judge whether it looks like a 1000-space
                    looks_like_1000 = (iw <= 600 and ih <= 600 and max(x1, y1) >= 800 and max(x1, y1) <= 1200)
                    code = "E105_coordinates_SPACE_MISMATCH" if looks_like_1000 else "E104_coordinates_OUT_OF_BOUNDS"
                    hint = " (looks like 1000-space coords; expected pixel space)" if looks_like_1000 else ""
                    issues.append(DatasetIssue(
                        level="ERROR",
                        code=code,
                        row=i,
                        sample_id=sid,
                        path=coordinates_prefix,
                        message=f"coordinates exceeds image size {iw}x{ih}: ({x0}, {y0}, {x1}, {y1}){hint}",
                    ))

    return issues

