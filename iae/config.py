"""Repository paths loaded from environment variables."""
from __future__ import annotations
import os
from pathlib import Path

def _path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()

DATA_ROOT = _path("IAE_DATA_ROOT", "data")
OUT_ROOT = _path("IAE_OUT_ROOT", str(DATA_ROOT / "v2"))
WATCHACT_ROOT = _path("IAE_WATCHACT_ROOT", str(DATA_ROOT / "external" / "WatchAct"))
MODEL_ROOT = _path("IAE_MODEL_ROOT", "models")
WATCHACT_SCRIPTS = [WATCHACT_ROOT / "video_planning" / "scripts", WATCHACT_ROOT / "video_planning"]
TEB_ROOT = Path(__file__).resolve().parent / "third_party" / "teb"
