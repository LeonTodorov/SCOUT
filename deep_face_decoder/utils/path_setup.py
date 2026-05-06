from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root path."""
    return Path(__file__).resolve().parents[2]


def ensure_recognizers_on_path() -> Path:
    """Ensure repo root is on sys.path so `feature_extractors.*` imports resolve."""
    root = repo_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
