# src/mmqlint/types.py
from __future__ import annotations

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}

# Built-in modality types and their required keys.
# You can extend this later or allow user registration.
TYPE_REQUIRED_KEYS = {
    "text":  {"text"},
    "image": {"image"},
    "audio": {"audio"},
    "video": {"video"},
    "file":  {"file"},
}

# Optional: simple type checks for required keys
TYPE_KEY_TYPES = {
    ("text", "text"): str,
    # For image/audio/video/file we allow str by default (path/url),
    # but users may store dicts (e.g., {"path":..., "sha256":...}); keep it permissive for now.
}
