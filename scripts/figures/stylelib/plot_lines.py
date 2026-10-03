# -*- coding: utf-8 -*-
"""Line / curve conventions.

Ours is thicker (1.6–2.0) than comparators (1.1–1.3). Markers carry a white
edge. Reference rules are dashed grey. Optional fill_between bands use
SynCrash alpha ~ 0.13.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from matplotlib.axes import Axes

from .style_helpers import BAND_ALPHA, legend_out, style_ax
from .style_palettes import ROLE, categorical

OURS_LW = 1.8
BASE_LW = 1.2
REF_LW = 0.9
MARKER_EDGE = 0.8


def plot_curves(
    ax: Axes,
    xs,
    ys_list,
    labels=None,
    colors=None,
    lws=None,
    emphasize: int = 0,
    markers: Optional[Sequence[str]] = None,
    bands=None,
    band_alpha: float = BAND_ALPHA,
    grid: Optional[str] = "y",
    legend: bool = True,
):
    """Plot multiple curves; ``emphasize`` gets the ours-weight stroke.

    ``bands`` is an optional sequence of y-error arrays (same length as
    ``ys_list``); None entries skip the band for that series.
    """
    n = len(ys_list)
    colors = list(colors) if colors is not None else categorical(n)
    labels = list(labels) if labels is not None else ["series-%d" % i for i in range(n)]
    lws = list(lws) if lws is not None else [
        OURS_LW if i == emphasize else BASE_LW for i in range(n)
    ]
    xs = np.asarray(xs, dtype=float)
    for i, ys in enumerate(ys_list):
        ys = np.asarray(ys, dtype=float)
        if bands is not None and i < len(bands) and bands[i] is not None:
            err = np.asarray(bands[i], dtype=float)
            ax.fill_between(
                xs, ys - err, ys + err,
                color=colors[i], alpha=band_alpha, linewidth=0, zorder=2,
            )
        kw = dict(color=colors[i], lw=lws[i], label=labels[i], zorder=3)
        if markers is not None and i < len(markers) and markers[i]:
            kw.update(
                marker=markers[i],
                ms=3.6,
                markerfacecolor="white",
                markeredgecolor=colors[i],
                markeredgewidth=MARKER_EDGE,
            )
        ax.plot(xs, ys, **kw)
    style_ax(ax, grid=grid)
    if legend:
        legend_out(ax)
    return ax


def ref_hline(ax: Axes, y: float, label: Optional[str] = None, color: str = "#B0B0B0"):
    ax.axhline(y, color=color, lw=REF_LW, ls=(0, (4, 3)), label=label, zorder=1)
    return ax


def ref_vline(ax: Axes, x: float, label: Optional[str] = None, color: str = "#B0B0B0"):
    ax.axvline(x, color=color, lw=REF_LW, ls=(0, (4, 3)), label=label, zorder=1)
    return ax
