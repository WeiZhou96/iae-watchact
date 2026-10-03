# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from matplotlib import font_manager as fm

want = [
    "Calibri Math",
    "Calibri",
    "Cambria Math",
    "Cambria",
    "Arial",
    "Helvetica",
    "DejaVu Sans",
]
print("=== findfont ===")
for name in want:
    try:
        path = fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
        print(f"OK  {name:16s} -> {path}")
    except Exception as exc:
        print(f"NO  {name:16s} -> {type(exc).__name__}: {exc}")

print("=== Windows Fonts candidates ===")
fonts = Path(r"C:\Windows\Fonts")
keys = ("calibri", "cambria", "arial", "segoe")
if fonts.is_dir():
    for p in sorted(fonts.iterdir()):
        low = p.name.lower()
        if any(k in low for k in keys):
            print(p)
