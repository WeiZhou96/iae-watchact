# -*- coding: utf-8 -*-
"""Heatmap conventions for metric matrices and regime scans.

Default ramp is cividis (Coordination fig8 / fig9). Cells may carry numeric
text with luminance-aware ink. Heatmaps keep a thin full frame; cartesian
despine does not apply.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

from .style_helpers import luminance, style_colorbar
from .style_palettes import SEQUENTIAL


def sequential_cmap(name: str, colors: Sequence[str], n: int = 256) -> LinearSegmentedColormap:
    """Build a paper-anchored sequential map (crack fig10 pattern)."""
    return LinearSegmentedColormap.from_list(name, list(colors), N=n)


def heatmap(
    ax: Axes,
    mat,
    *,
    cmap: str = SEQUENTIAL,
    vmin=None,
    vmax=None,
    xticklabels=None,
    yticklabels=None,
    annotate: bool = False,
    fmt: str = ".2f",
    colorbar: bool = True,
    mark_best: Optional[str] = None,
    cell_lw: float = 0.7,
    fontsize: float = 6.5,
):
    """Draw a 2-d matrix. ``mark_best`` in {None, 'col', 'row'} stars the max."""
    mat = np.asarray(mat, dtype=float)
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)
        sp.set_color("#6B7280")
    ax.tick_params(length=0, pad=3)
    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(list(xticklabels))
    if yticklabels is not None:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(list(yticklabels))
    if cell_lw:
        for i in range(mat.shape[0] + 1):
            ax.axhline(i - 0.5, color="white", lw=cell_lw, zorder=3)
        for j in range(mat.shape[1] + 1):
            ax.axvline(j - 0.5, color="white", lw=cell_lw, zorder=3)
    best = None
    if mark_best == "col" and np.isfinite(mat).any():
        best = ("col", np.nanargmax(mat, axis=0))
    elif mark_best == "row" and np.isfinite(mat).any():
        best = ("row", np.nanargmax(mat, axis=1))
    if annotate or best is not None:
        cmap_obj = im.cmap
        norm = im.norm
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not np.isfinite(v):
                    ax.text(j, i, "\u2013", ha="center", va="center",
                            fontsize=fontsize, color="#AAAAAA")
                    continue
                rgba = cmap_obj(norm(v))
                ink = "white" if luminance(rgba) < 0.55 else "#111111"
                is_best = False
                if best is not None:
                    kind, idx = best
                    is_best = (kind == "col" and idx[j] == i) or (kind == "row" and idx[i] == j)
                if annotate:
                    ax.text(
                        j, i, format(v, fmt), ha="center", va="center",
                        fontsize=fontsize, color=ink,
                        fontweight="bold" if is_best else "normal", zorder=4,
                    )
                if is_best:
                    ax.text(
                        j - 0.32, i - 0.32, "$\\bigstar$",
                        ha="center", va="center", fontsize=fontsize,
                        color=ink, zorder=4,
                    )
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        style_colorbar(cb)
    return im
