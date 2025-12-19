from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import json
import importlib.util
from pathlib import Path

from .profiles import Profile


# -----------------------------
# Issue model / severity policy
# -----------------------------
_LEVEL_ORDER = {"INFO": 0, "WARN": 1, "ERROR": 2}


@dataclass
class Issue:
    level: str        # INFO|WARN|ERROR
    code: str
    line: int
    sample_id: str
    path: str
    message: str

    @property
    def line_no(self) -> int:
        # backwards-compat with earlier CLI that used .line_no
        return self.line


def _issue(level: str, code: str, line: int, sample_id: str, path: str, message: str) -> Issue:
    if level not in _LEVEL_ORDER:
        level = "ERROR"
    return Issue(level=level, code=code, line=line, sample_id=sample_id, path=path, message=message)


def should_fail(issues: List[Issue], fail_on: str) -> bool:
    thr = _LEVEL_ORDER.get((fail_on or "ERROR").upper(), 2)
    return any(_LEVEL_ORDER.get(i.level, 2) >= thr for i in issues)


# -----------------------------
# JSONL chat schema lint
# -----------------------------
_ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


def _is_blank_text(x: Any) -> bool:
    return isinstance(x, str) and x.strip() == ""


def _typed_text_of_content(content: Any) -> str:
    """Best-effort extract concatenated text from content which may be:
    - raw str
    - list[{"type":"text","text":...}, ...]
    - dict with {"type":"text","text":...}
    """
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if content.get("type") == "text" and isinstance(content.get("text"), str):
            return content["text"]
        return ""
    if isinstance(content, list):
        out = []
        for it in content:
            if isinstance(it, dict) and it.get("type") == "text" and isinstance(it.get("text"), str):
                out.append(it["text"])
        return "\n".join(out)
    return ""


def _validate_messages_schema(messages: Any, line_no: int, sample_id: str, strict_typed: bool) -> List[Issue]:
    issues: List[Issue] = []
    if not isinstance(messages, list):
        return [_issue("ERROR", "E003_SCHEMA_MISMATCH", line_no, sample_id, "messages", "`messages` must be a list.")]

    for mi, m in enumerate(messages):
        base = f"messages[{mi}]"
        if not isinstance(m, dict):
            issues.append(_issue("ERROR", "E003_SCHEMA_MISMATCH", line_no, sample_id, base, "Each message must be an object."))
            continue
        role = m.get("role")
        if role not in _ALLOWED_ROLES:
            issues.append(_issue("ERROR", "E003_SCHEMA_MISMATCH", line_no, sample_id, f"{base}.role", f"Invalid role: {role!r}."))
        if "content" not in m:
            issues.append(_issue("ERROR", "E003_SCHEMA_MISMATCH", line_no, sample_id, f"{base}.content", "Missing required field `content`."))
            continue

        content = m.get("content")
        if strict_typed:
            if not isinstance(content, list):
                issues.append(_issue("ERROR", "E003_SCHEMA_MISMATCH", line_no, sample_id, f"{base}.content", "With --strict-typed, `content` must be a list of typed items."))
            else:
                for ci, it in enumerate(content):
                    p = f"{base}.content[{ci}]"
                    if not isinstance(it, dict):
                        issues.append(_issue("ERROR", "E003_SCHEMA_MISMATCH", line_no, sample_id, p, "Typed content item must be an object."))
                        continue
                    t = it.get("type")
                    if not t:
                        issues.append(_issue("ERROR", "E003_SCHEMA_MISMATCH", line_no, sample_id, p, "Typed content item missing key `type`."))
                        continue
                    if t == "text":
                        if not isinstance(it.get("text"), str):
                            issues.append(_issue("ERROR", "E012_BAD_FIELD_TYPE", line_no, sample_id, f"{p}.text", "type='text' field 'text' must be str."))
                    elif t == "image":
                        if "image" not in it:
                            issues.append(_issue("ERROR", "E011_MISSING_TYPE_FIELDS", line_no, sample_id, p, "type='image' missing required fields: ['image']"))
                    # Unknown types are allowed for extensibility.
        else:
            # non-strict: allow raw strings or typed list
            if not (isinstance(content, str) or isinstance(content, list) or isinstance(content, dict)):
                issues.append(_issue("ERROR", "E012_BAD_FIELD_TYPE", line_no, sample_id, f"{base}.content", "content must be str or typed list/object."))

    return issues


def _system_presence_and_nonempty(messages: List[Dict[str, Any]], line_no: int, sample_id: str, profile_obj: Profile) -> List[Issue]:
    issues: List[Issue] = []
    if not profile_obj.require_system:
        return issues

    sys_msgs = [m for m in messages if m.get("role") == "system"]
    if not sys_msgs:
        issues.append(_issue("ERROR", "E002_SYSTEM_REQUIRED", line_no, sample_id, "messages",
                             "Profile requires a role='system' message, but none is present."))
        return issues

    # Must be non-empty
    for sm in sys_msgs:
        c = sm.get("content")
        txt = _typed_text_of_content(c)
        if _is_blank_text(txt):
            issues.append(_issue("ERROR", "E002_SYSTEM_EMPTY", line_no, sample_id, "messages",
                                 "Profile requires a non-empty role='system' message, but system content is empty/blank."))
            break
    return issues


def _mode_specific_checks(messages: List[Dict[str, Any]], line_no: int, sample_id: str, mode: str) -> List[Issue]:
    issues: List[Issue] = []
    mode = (mode or "train").lower()

    has_assistant = any(m.get("role") == "assistant" for m in messages)
    if mode == "infer":
        if has_assistant:
            # label leakage risk
            # point to first assistant if possible
            idx = next((i for i, m in enumerate(messages) if m.get("role") == "assistant"), None)
            path = f"messages[{idx}].role" if idx is not None else "messages"
            issues.append(_issue("ERROR", "E001_INFER_HAS_ASSISTANT", line_no, sample_id, path,
                                 "Inference sample contains an assistant turn (risk of label leakage / wrong prompt)."))
    else:  # train
        if not has_assistant:
            issues.append(_issue("ERROR", "E001_TRAIN_MISSING_ASSISTANT", line_no, sample_id, "messages",
                                 "Train sample contains no assistant turn (no supervision label)."))
    return issues


def lint_jsonl(
    jsonl_path: str,
    *,
    mode: str = "train",
    profile_obj: Profile,
    strict_typed: bool = False,
) -> List[Issue]:
    """Lint a JSONL file containing {"id":..., "messages":[...]} lines."""
    issues: List[Issue] = []
    path = Path(jsonl_path)

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                issues.append(_issue("ERROR", "E004_BAD_JSON", line_no, f"line_{line_no}", "$", f"Invalid JSON: {e}"))
                continue

            sample_id = str(obj.get("id", f"line_{line_no}"))
            messages = obj.get("messages", None)

            # schema
            sch = _validate_messages_schema(messages, line_no, sample_id, strict_typed)
            issues.extend(sch)
            if sch and any(i.code in {"E003_SCHEMA_MISMATCH", "E004_BAD_JSON", "E012_BAD_FIELD_TYPE"} for i in sch):
                # If messages isn't valid list, skip deeper checks.
                if not isinstance(messages, list):
                    continue

            assert isinstance(messages, list)
            issues.extend(_system_presence_and_nonempty(messages, line_no, sample_id, profile_obj))
            issues.extend(_mode_specific_checks(messages, line_no, sample_id, mode))

    return issues


def fix_jsonl(
    jsonl_path: str,
    out_path: str,
    *,
    strict_typed: bool = False,
) -> Tuple[str, List[Issue]]:
    """Apply conservative, 'safe' fixes.

    This is intentionally limited: it won't invent data.
    It can fix obvious typos like role='usr' -> 'user',
    and add missing type='text' when item has a 'text' field.
    """
    issues: List[Issue] = []
    in_p = Path(jsonl_path)
    out_p = Path(out_path)

    with in_p.open("r", encoding="utf-8") as fin, out_p.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except Exception as e:
                issues.append(_issue("ERROR", "E004_BAD_JSON", line_no, f"line_{line_no}", "$", f"Invalid JSON: {e}"))
                continue

            sample_id = str(obj.get("id", f"line_{line_no}"))
            msgs = obj.get("messages")
            if isinstance(msgs, list):
                for m in msgs:
                    if isinstance(m, dict):
                        if m.get("role") == "usr":
                            m["role"] = "user"
                            issues.append(_issue("INFO", "I901_FIX_ROLE_USR", line_no, sample_id, "messages[].role", "Fixed role 'usr' -> 'user'."))

                        if strict_typed and isinstance(m.get("content"), list):
                            for it in m["content"]:
                                if isinstance(it, dict) and "type" not in it and "text" in it:
                                    it["type"] = "text"
                                    issues.append(_issue("INFO", "I902_FIX_ADD_TYPE_TEXT", line_no, sample_id, "messages[].content[].type", "Added missing type='text' for item with 'text' field."))

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return str(out_p), issues


# -----------------------------
# verify-system: render-level check
# -----------------------------
def _load_render_plugin(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("mmqlint_render_plugin", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load plugin {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "render"):
        raise RuntimeError("render plugin must define a function: render(messages, **kwargs) -> str")
    return mod.render


def verify_system_visibility_jsonl(
    jsonl_path: str,
    *,
    profile_obj: Profile,
    render_plugin_path: str,
    strict_typed: bool = False,
) -> List[Issue]:
    """Verify that the system message exists AND appears in the final rendered prompt."""
    issues: List[Issue] = []
    render_fn = _load_render_plugin(render_plugin_path)

    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                issues.append(_issue("ERROR", "E004_BAD_JSON", line_no, f"line_{line_no}", "$", f"Invalid JSON: {e}"))
                continue

            sample_id = str(obj.get("id", f"line_{line_no}"))
            messages = obj.get("messages", None)

            sch = _validate_messages_schema(messages, line_no, sample_id, strict_typed)
            issues.extend(sch)
            if not isinstance(messages, list):
                continue

            # schema-level: require system and non-empty
            issues.extend(_system_presence_and_nonempty(messages, line_no, sample_id, profile_obj))
            sys_texts = [_typed_text_of_content(m.get("content")) for m in messages if m.get("role") == "system"]
            sys_text = "\n".join([t for t in sys_texts if isinstance(t, str)])

            if not profile_obj.system_visible:
                # If profile doesn't require visibility, skip.
                continue

            # render prompt
            try:
                prompt = render_fn(messages=messages, sample=obj)
            except TypeError:
                # plugin signature without kwargs
                prompt = render_fn(messages)
            except Exception as e:
                issues.append(_issue("ERROR", "E901_RENDER_PLUGIN_ERROR", line_no, sample_id, "messages", f"Render plugin error: {e}"))
                continue

            # check containment (best-effort substring)
            if sys_text.strip() and sys_text.strip() not in str(prompt):
                level = profile_obj.system_invisible_level
                issues.append(_issue(level, "E002_SYSTEM_NOT_VISIBLE", line_no, sample_id, "messages",
                                     "System message exists but is not present in the rendered prompt (likely dropped by the pipeline)."))

    return issues


# -----------------------------
# HF Datasets / Arrow checks
# -----------------------------
def _iter_splits(ds_any):
    # datasets.Dataset or datasets.DatasetDict
    if hasattr(ds_any, "items") and not hasattr(ds_any, "features"):
        # DatasetDict-like
        for k, v in ds_any.items():
            yield k, v
    else:
        yield "data", ds_any


def _find_image_columns(features: Dict[str, Any]) -> List[str]:
    cols: List[str] = []
    for k, v in features.items():
        if v.__class__.__name__ == "Image":
            cols.append(k)
    return cols


def _get_size_from_row(row: Dict[str, Any], image_cols: List[str], w_col: Optional[str], h_col: Optional[str]) -> Optional[Tuple[int, int]]:
    # Prefer explicit w/h columns if present (fast, no decode)
    if w_col and h_col and isinstance(row.get(w_col), int) and isinstance(row.get(h_col), int):
        return int(row[w_col]), int(row[h_col])

    # Else try Image column (may decode)
    for c in image_cols:
        img = row.get(c)
        # datasets.Image returns PIL.Image.Image when decode=True
        if hasattr(img, "size"):
            w, h = img.size
            return int(w), int(h)
        # could be dict with 'width'/'height'
        if isinstance(img, dict):
            if isinstance(img.get("width"), int) and isinstance(img.get("height"), int):
                return int(img["width"]), int(img["height"])
    return None


def _walk_none(obj: Any, prefix: str = "") -> Iterable[str]:
    """Yield paths where value is None (nested)."""
    if obj is None:
        yield prefix or "$"
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            yield from _walk_none(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            yield from _walk_none(v, p)


def check_dataset_on_disk(
    dataset_path: str,
    *,
    expect_size: Optional[Tuple[int, int]] = None,
    size_policy: str = "any",  # any|consistent
    fail_on_level: str = "ERROR",
    coord_field: Optional[str] = None,
    coord_keys: Optional[List[str]] = None,
) -> List[Issue]:
    """Check HF dataset saved on disk.

    - Nested None detection: catches "features declares nested keys but rows got filled with None"
    - Size sanity: (A) --size-policy consistent -> all examples have same size (w,h)
                 (B) --expect-size W,H -> strict match
    - coord_field/coord_keys (optional): if provided, ensures coords are within image size.
      This is useful to catch 512-vs-1000 coordinate system mixups without hardcoding coordinates.* names.
    """
    issues: List[Issue] = []

    try:
        from datasets import load_from_disk  # type: ignore
    except Exception as e:
        return [_issue("ERROR", "E900_DATASETS_IMPORT", 1, "dataset", "$", f"Cannot import datasets: {e}")]

    ds_any = load_from_disk(dataset_path)

    for split_name, ds in _iter_splits(ds_any):
        features = getattr(ds, "features", {}) or {}
        image_cols = _find_image_columns(features)

        # heuristic w/h columns
        w_col = "image_w" if "image_w" in features else ("width" if "width" in features else None)
        h_col = "image_h" if "image_h" in features else ("height" if "height" in features else None)

        seen_sizes: Dict[Tuple[int, int], int] = {}

        for idx, row in enumerate(ds):
            row_no = idx + 1
            sample_id = str(row.get("id", f"{split_name}:{row_no}"))

            # (1) nested None check (but do not flag top-level 'image' missing — only actual None values)
            for p in _walk_none(row):
                issues.append(_issue("ERROR", "E102_DATASET_NONE_VALUE", row_no, sample_id, p,
                                     f"{p} is None (often happens when missing nested key gets filled as None after cast/save)."))

            # (2) image size checks
            size = _get_size_from_row(row, image_cols=image_cols, w_col=w_col, h_col=h_col)
            if size is not None:
                seen_sizes[size] = seen_sizes.get(size, 0) + 1
                if expect_size is not None and size != expect_size:
                    issues.append(_issue("ERROR", "E201_IMAGE_SIZE_MISMATCH", row_no, sample_id, "image",
                                         f"Image size {size[0]}x{size[1]} != expected {expect_size[0]}x{expect_size[1]}."))
            else:
                # No size info available -> warn (can't check)
                issues.append(_issue("WARN", "W202_IMAGE_SIZE_UNKNOWN", row_no, sample_id, "image",
                                     "Cannot determine image size (no Image column and no width/height columns)."))

            # (3) optional coord sanity
            if coord_field and coord_keys:
                # row field could be nested "a.b.c"
                cur = row
                ok = True
                for part in coord_field.split("."):
                    if isinstance(cur, dict) and part in cur:
                        cur = cur[part]
                    else:
                        ok = False
                        break
                if ok and isinstance(cur, dict):
                    vals = []
                    for k in coord_keys:
                        vals.append(cur.get(k))
                    if all(isinstance(v, (int, float)) for v in vals) and size is not None:
                        w, h = size
                        # generic rule: coords must be within [0,w] / [0,h]
                        bad = False
                        for k, v in zip(coord_keys, vals):
                            if k.lower().endswith(("x", "x0", "x1", "cx")):
                                if v < 0 or v > w:
                                    bad = True
                            if k.lower().endswith(("y", "y0", "y1", "cy")):
                                if v < 0 or v > h:
                                    bad = True
                        if bad:
                            issues.append(_issue("WARN", "W301_COORD_OUT_OF_BOUNDS", row_no, sample_id, coord_field,
                                                 f"Coordinates out of bounds for image {w}x{h}: { {k: cur.get(k) for k in coord_keys} }"))

        # After iterating split: size_policy=consistent
        if (size_policy or "any").lower() == "consistent":
            if len(seen_sizes) > 1:
                # report one aggregated warning per split
                # choose top 5 sizes
                top = sorted(seen_sizes.items(), key=lambda kv: kv[1], reverse=True)[:5]
                msg = "Inconsistent image sizes in split "
                msg += f"{split_name}: " + ", ".join([f"{w}x{h} ({c})" for (w, h), c in top])
                issues.append(_issue("WARN", "W201_IMAGE_SIZE_INCONSISTENT", 1, f"{split_name}", "image", msg))
            elif len(seen_sizes) == 0:
                issues.append(_issue("WARN", "W202_IMAGE_SIZE_UNKNOWN", 1, f"{split_name}", "image",
                                     "Cannot determine image sizes for this split."))

    return issues
