"""
    Tiny SCOUT-wide utilities.
"""
from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent


def load_paths() -> dict[str, Path]:
    cfg = REPO_ROOT / "paths.yaml"
    if not cfg.is_file():
        raise FileNotFoundError(f"paths.yaml not found at {REPO_ROOT}.")
    raw = yaml.safe_load(cfg.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{cfg} must be a YAML mapping.")
    out: dict[str, Path] = {}
    for k, v in raw.items():
        p = Path(v).expanduser()
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve(strict=False)
        out[k] = p
    return out
