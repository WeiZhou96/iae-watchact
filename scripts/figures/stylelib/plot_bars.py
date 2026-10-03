# -*- coding: utf-8 -*-
"""Bar, lollipop, dumbbell and forest-interval helpers.

White bar edges (lw 0.4) come from RAL / SynCrash. Forest intervals follow
Coordination fig4 (line + end ticks + hollow-edged point, not a bar).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from matplotlib.axes import Axes

from .style_helpers import legend_out, legend_outside, style_ax
from .style_palettes import ROLE, categorical

BAR_EDGE_LW = 0.4
LOLLIPOP_LW = 2.4
FOREST_LW = 0.9


def grouped_bars(
    ax: Axes,
    data,
    labels,
    series_labels=None,
    colors=None,
    width: float = 0.8,
    edgecolor: str = "white",
    yerr=None,
    legend: str = "out",
):
    """``data``: shape (n_groups, n_series) or a list of series arrays.

    ``yerr`` matches ``data``. ``legend`` is 'out', 'outside', or None.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    n_groups, n_series = arr.shape
    colors = list(colors) if colors is not None else categorical(n_series)
    series_labels = list(series_labels) if series_labels is not None else [
        "s%d" % i for i in range(n_series)
    ]
    err = None if yerr is None else np.asarray(yerr, dtype=float)
    if err is not None and err.ndim == 1:
        err = err[:, None]
    x = np.arange(n_groups)
    bar_w = width / max(n_series, 1)
    for i in range(n_series):
        offset = (i - (n_series - 1) / 2.0) * bar_w
        kw = dict(
            color=colors[i],
            edgecolor=edgecolor,
            linewidth=BAR_EDGE_LW,
            label=series_labels[i],
        )
        if err is not None:
            kw["yerr"] = err[:, i]
            kw["error_kw"] = dict(ecolor="#333333", elinewidth=0.9, capsize=2.5)
        ax.bar(x + offset, arr[:, i], bar_w * 0.92, **kw)
    ax.set_xticks(x)
    ax.set_xticklabels(list(labels))
    style_ax(ax, grid="y")
    ax.grid(axis="x", visible=False)
    if n_series > 1 and legend == "out":
        legend_out(ax)
    elif n_series > 1 and legend == "outside":
        legend_outside(ax)
    return ax


def stacked_bars(
    ax: Axes,
    fractions,
    labels,
    series_labels,
    colors=None,
    annotate: bool = False,
):
    """100% stacked composition (RAL fig2). ``fractions`` shape (n_groups, n_series)."""
    arr = np.asarray(fractions, dtype=float)
    n_groups, n_series = arr.shape
    colors = list(colors) if colors is not None else categorical(n_series)
    x = np.arange(n_groups)
    bottom = np.zeros(n_groups)
    for i in range(n_series):
        ax.bar(
            x, arr[:, i], bottom=bottom, color=colors[i],
            edgecolor="white", linewidth=BAR_EDGE_LW, label=series_labels[i],
        )
        if annotate:
            for j, v in enumerate(arr[:, i]):
                if v >= 8:
                    ax.text(
                        j, bottom[j] + v / 2.0, "%.0f%%" % v,
                        ha="center", va="center", fontsize=7.2,
                        color="white", fontweight="bold",
                    )
        bottom = bottom + arr[:, i]
    ax.set_xticks(x)
    ax.set_xticklabels(list(labels))
    style_ax(ax, grid="y")
    ax.grid(axis="x", visible=False)
    legend_outside(ax)
    return ax


def lollipop(
    ax: Axes,
    labels,
    values,
    colors=None,
    *,
    horizontal: bool = False,
    ours_index: Optional[int] = -1,
):
    """Stem + marker. ``ours_index`` (-1 = last) uses ROLE['ours'] if colors omitted."""
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    if colors is None:
        colors = [ROLE["gray"]] * n
        if ours_index is not None:
            colors[ours_index] = ROLE["ours"]
    idx = np.arange(n)
    sizes = [42] * n
    if ours_index is not None:
        sizes[ours_index] = 80
    if horizontal:
        ax.hlines(idx, 0, vals, color=colors, lw=LOLLIPOP_LW, alpha=0.55)
        ax.scatter(vals, idx, s=sizes, color=colors, zorder=3, edgecolors="white", linewidths=0.8)
        ax.set_yticks(idx)
        ax.set_yticklabels(list(labels))
        style_ax(ax, grid="x")
        ax.grid(axis="y", visible=False)
    else:
        ax.vlines(idx, 0, vals, color=colors, lw=LOLLIPOP_LW, alpha=0.55)
        ax.scatter(idx, vals, s=sizes, color=colors, zorder=3, edgecolors="white", linewidths=0.8)
        ax.set_xticks(idx)
        ax.set_xticklabels(list(labels))
        style_ax(ax, grid="y")
        ax.grid(axis="x", visible=False)
    return ax


def dumbbell(ax: Axes, y, left, right, left_color=None, right_color=None, lw: float = 3.4):
    """Paired points joined by a grey stem (RAL composite / crack ablation)."""
    y = np.asarray(y, dtype=float)
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left_color = left_color or ROLE["baseline"]
    right_color = right_color or ROLE["ours"]
    for yi, a, b in zip(y, left, right):
        ax.plot([a, b], [yi, yi], color="#C9CCD6", lw=lw, zorder=1, solid_capstyle="round")
    ax.scatter(left, y, s=55, color=left_color, zorder=4, edgecolors="white", linewidths=0.8)
    ax.scatter(right, y, s=55, color=right_color, zorder=4, edgecolors="white", linewidths=0.8)
    style_ax(ax, grid="x")
    return ax


def forest_intervals(
    ax: Axes,
    y,
    lo,
    hi,
    point,
    color=None,
    cap: float = 0.075,
):
    """Coordination fig4 interval: butt-capped line, end ticks, white-edged point."""
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    point = np.asarray(point, dtype=float)
    color = color or ROLE["blue"]
    for yi, a, b, p in zip(y, lo, hi, point):
        ax.plot([a, b], [yi, yi], color=color, lw=FOREST_LW, solid_capstyle="butt", zorder=3)
        ax.plot([a, a], [yi - cap, yi + cap], color=color, lw=0.7, zorder=3)
        ax.plot([b, b], [yi - cap, yi + cap], color=color, lw=0.7, zorder=3)
        ax.plot(
            [p], [yi], marker="o", ms=2.9, color=color,
            markeredgecolor="white", markeredgewidth=0.35, zorder=4,
        )
    style_ax(ax, grid="x")
    return ax


def errorbar_points(ax: Axes, x, y, yerr, color=None, fmt="o-", **kwargs):
    color = color or ROLE["ours"]
    kw = dict(
        fmt=fmt, color=color, lw=1.2, capsize=2.0,
        markerfacecolor="white", markeredgewidth=0.7, elinewidth=0.8,
    )
    kw.update(kwargs)
    ax.errorbar(x, y, yerr=yerr, **kw)
    style_ax(ax, grid="y")
    return ax
