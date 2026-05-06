"""Path setup for sae_analysis: makes SCOUT subdirs importable."""
from __future__ import annotations

import sys
from pathlib import Path


SCOUT_ROOT = Path(__file__).resolve().parents[1]
DFD_ROOT = SCOUT_ROOT / "deep_face_decoder"


def setup_paths() -> None:
    """Insert SCOUT root + deep_face_decoder on sys.path.

    NOTE: batchtopk/ is intentionally not added — it contains a 'utils.py' that
    would shadow deep_face_decoder/utils/ when importing 'utils.path_setup'
    inside the DFD wrappers. Use 'from batchtopk.batchtopk import ...' instead.
    """
    for p in [SCOUT_ROOT, DFD_ROOT]:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
