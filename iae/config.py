"""Repository paths.

Each path is taken from its environment variable, else from configs/paths.yaml (a copy of
configs/paths.example.yaml), else from the default below. Relative paths are resolved against the repository root.
"""
from __future__ import annotations
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = REPO_ROOT / "configs" / "paths.yaml"


def _file_values() -> dict:
    """Reads the flat `key: value` file without a YAML dependency."""
    values = {}
    if _CONFIG.exists():
        for line in _CONFIG.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if ":" in line:
                key, value = line.split(":", 1)
                values[key.strip()] = value.strip().strip("'\"")
    return values


_VALUES = _file_values()


def _path(env: str, key: str, default: str) -> Path:
    p = Path(os.environ.get(env) or _VALUES.get(key) or default).expanduser()
    return (p if p.is_absolute() else REPO_ROOT / p).resolve()


DATA_ROOT = _path("IAE_DATA_ROOT", "data_root", "data")                      # WatchAct release, manifests, models/
OUT_ROOT = _path("IAE_OUT_ROOT", "out_root", str(DATA_ROOT / "v2"))           # pipeline outputs and runs/
WATCHACT_ROOT = _path("IAE_WATCHACT_ROOT", "watchact_root", str(DATA_ROOT / "external" / "WatchAct"))  # official code
MODEL_ROOT = DATA_ROOT / "models"                                             # model weights, see README
WATCHACT_SCRIPTS = [WATCHACT_ROOT / "video_planning" / "scripts", WATCHACT_ROOT / "video_planning"]
TEB_ROOT = Path(__file__).resolve().parent / "third_party" / "teb"
