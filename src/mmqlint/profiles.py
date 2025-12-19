from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import json

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class Profile:
    """A profile describes *runtime assumptions* of a model / framework pipeline.

    - require_system: dataset must contain at least one role='system' message.
    - system_visible: system messages are expected to be passed to the model.
    - fold_system_into_user: legacy workaround where system is concatenated into user.
      (Not allowed when require_system=True in our policy.)
    - system_invisible_level: if system is present but not visible, severity in verify-system.
    """
    name: str
    require_system: bool = True
    system_visible: bool = True
    fold_system_into_user: bool = False
    system_invisible_level: str = "ERROR"  # "WARN" or "ERROR"

    def validate(self) -> None:
        if self.system_invisible_level not in {"WARN", "ERROR"}:
            raise ValueError(f"system_invisible_level must be WARN|ERROR, got {self.system_invisible_level}")
        if self.require_system and not self.system_visible:
            # Your stated policy: system must be present AND visible.
            raise ValueError(f"profile {self.name}: require_system=True implies system_visible=True")
        if self.require_system and self.fold_system_into_user:
            raise ValueError(f"profile {self.name}: fold_system_into_user must be False when require_system=True")
        if self.fold_system_into_user and self.system_visible:
            # If you're folding, you *usually* don't rely on system visibility.
            # We don't forbid it, but it is confusing; keep strict here.
            raise ValueError(f"profile {self.name}: fold_system_into_user=True conflicts with system_visible=True")


DEFAULT_PROFILES: Dict[str, Profile] = {
    "generic": Profile(name="generic", require_system=True, system_visible=True, fold_system_into_user=False, system_invisible_level="ERROR"),
    "my-vlm": Profile(name="my-vlm", require_system=True, system_visible=True, fold_system_into_user=False, system_invisible_level="ERROR"),
    "qwen3-vl": Profile(name="qwen3-vl", require_system=True, system_visible=True, fold_system_into_user=False, system_invisible_level="ERROR"),
}


def _load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load YAML profiles. `pip install pyyaml`")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_profiles(profile_file: Optional[str] = None) -> Dict[str, Profile]:
    """Load built-in profiles + optional overrides from YAML/JSON.

    Expected file format:
      profiles:
        my-vlm:
          require_system: true
          system_visible: true
          fold_system_into_user: false
          system_invisible_level: ERROR
    """
    profs: Dict[str, Profile] = dict(DEFAULT_PROFILES)

    if not profile_file:
        # validate defaults
        for p in profs.values():
            p.validate()
        return profs

    if profile_file.endswith((".yaml", ".yml")):
        data = _load_yaml(profile_file)
    elif profile_file.endswith(".json"):
        data = _load_json(profile_file)
    else:
        raise ValueError("profile_file must be .yaml/.yml or .json")

    raw = data.get("profiles", data)  # allow either top-level "profiles" or direct mapping
    if not isinstance(raw, dict):
        raise ValueError("Profile file must contain a mapping under key 'profiles'")

    for name, cfg in raw.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"profile {name} must be a mapping of fields")
        merged = {
            "name": name,
            "require_system": cfg.get("require_system", profs.get(name, Profile(name=name)).require_system),
            "system_visible": cfg.get("system_visible", profs.get(name, Profile(name=name)).system_visible),
            "fold_system_into_user": cfg.get("fold_system_into_user", profs.get(name, Profile(name=name)).fold_system_into_user),
            "system_invisible_level": cfg.get("system_invisible_level", profs.get(name, Profile(name=name)).system_invisible_level),
        }
        p = Profile(**merged)
        p.validate()
        profs[name] = p

    # validate all
    for p in profs.values():
        p.validate()

    return profs
