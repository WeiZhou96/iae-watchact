# -*- coding: utf-8 -*-
import sys as _iae_sys
if '--help' in _iae_sys.argv:
    print('Usage: python %s [options]' % __file__)
    raise SystemExit(0)
"""Learning curve and sensitivity of IAE (real runs; seeds 0-2; all 455 NC/RD requests, missing = failure).

Input: figs_src/data/sens_summary.json (sens_summary.py). (a)-(c) retrain with seeds 0-2 (mean +- std);
(d) re-decodes the ensemble's frame logits with each test-time cost lambda (no retraining; NC only is affected).
Output: figs/fig_sensitivity.pdf (+ .png, 300 dpi)
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from stylelib import use_style, save_figure, style_ax, panel_tag  # noqa: E402
from stylelib.style_base import IEEE_TEXT_WIDTH_IN  # noqa: E402

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "data" / "sens_summary.json").read_text())
OUT = HERE.parent.parent / "figs" / "fig_sensitivity.pdf"
C_ALL, C_NC = "#D55E00", "#0072B2"
GREY = "#9A9A9A"
use_style()


def series(keys, metric):
    v = np.array([[p[metric] for p in S["trained"][k]] for k in keys])
    return v.mean(1), v.std(1)


def draw(ax, x, keys, chosen, xlabel, xticks=None, xticklabels=None, logx=False):
    for metric, col, ls, mk in (("ALL_SR", C_ALL, "-", "o"), ("ALL_strict", C_ALL, (0, (3, 2)), "s"),
                                ("No_SR", C_NC, "-", "o"), ("No_strict", C_NC, (0, (3, 2)), "s")):
        m, s = series(keys, metric)
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.13, lw=0, zorder=1)
        # individual seeds (three per setting), drawn as small filled dots behind the mean
        per_seed = np.array([[p[metric] for p in S["trained"][k]] for k in keys])
        for sd in range(per_seed.shape[1]):
            ax.scatter(x, per_seed[:, sd], s=4, color=col, alpha=0.45, lw=0, zorder=2)
        ax.plot(x, m, color=col, ls=ls, lw=1.3 if metric.startswith("ALL") else 1.0, marker=mk, ms=3.2,
                mfc="white", mec=col, mew=0.8, zorder=3)
    ax.axvline(chosen, color=GREY, lw=0.6, ls=(0, (1, 1.5)), zorder=0)
    if logx: ax.set_xscale("log")
    if xticks is not None:
        ax.set_xticks(xticks); ax.set_xticklabels(xticklabels or [str(t) for t in xticks]); ax.minorticks_off()
    ax.set_xlabel(xlabel, fontsize=7.4, labelpad=1)
    ax.set_ylim(0, 90)
    style_ax(ax, grid="y")


fig, axs = plt.subplots(1, 4, figsize=(IEEE_TEXT_WIDTH_IN, 2.25), gridspec_kw=dict(wspace=0.32))
fig.subplots_adjust(left=0.065, right=0.995, top=0.84, bottom=0.27)

# (a) learning curve
ax = axs[0]
draw(ax, np.array([25, 50, 75, 100]), ["frac25", "frac50", "frac75", "frac100"], 100,
     "Training activities per fold (%)", [25, 50, 75, 100])
ax.axhline(27.5, color="#56B4E9", lw=0.8, ls=(0, (4, 3)), zorder=0)
ax.text(27, 26.0, "32B direct, all SR 27.5", fontsize=6.2, color="#3A8FC4", va="top")
ax.set_ylabel("Success (%)", fontsize=7.4)
ax.set_title("Training data", fontsize=8.0)
m, _ = series(["frac25", "frac100"], "ALL_SR")
ax.annotate("%.1f" % m[0], (25, m[0]), xytext=(2, 4), textcoords="offset points", ha="left", fontsize=6.4, color=C_ALL)
ax.annotate("%.1f" % m[1], (100, m[1]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=6.4, color=C_ALL)
panel_tag(ax, "(a)", dx=-0.3, dy=1.16)

# (b) top-k pooling
ax = axs[1]
draw(ax, np.array([1, 3, 5, 10, 20]), ["k1", "k3", "k5", "k10", "k20"], 5, "Pooled frames $k$",
     [1, 3, 5, 10, 20], logx=True)
ax.set_title("Video-score pooling", fontsize=8.0)
panel_tag(ax, "(b)", dx=-0.2, dy=1.16)

# (c) structured-loss weight (0 drawn at the left end of a symlog-like categorical axis)
ax = axs[2]
xs = np.arange(5)
draw(ax, xs, ["mu0", "mu003", "mu01", "mu03", "mu1"], 2, r"Structured-loss weight $\gamma$", list(xs),
     ["0", ".03", ".1", ".3", "1"])
ax.set_title("Program-level loss", fontsize=8.0)
panel_tag(ax, "(c)", dx=-0.2, dy=1.16)

# (d) test-time decoding cost on the ensemble (no retraining)
ax = axs[3]
lams = [-2, -1, -0.5, 0, 0.5, 1, 2]
for metric, col, ls, mk in (("ALL_SR", C_ALL, "-", "o"), ("ALL_strict", C_ALL, (0, (3, 2)), "s"),
                            ("No_SR", C_NC, "-", "o"), ("No_strict", C_NC, (0, (3, 2)), "s")):
    v = [S["lambda"][str(l)][metric] for l in lams]
    ax.plot(lams, v, color=col, ls=ls, lw=1.3 if metric.startswith("ALL") else 1.0, marker=mk, ms=3.2,
            mfc="white", mec=col, mew=0.8, zorder=3)
ax.axvline(0, color=GREY, lw=0.6, ls=(0, (1, 1.5)), zorder=0)
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_xlabel(r"Decoding cost $\lambda$ (test time)", fontsize=7.4, labelpad=1)
ax.set_ylim(0, 90)
style_ax(ax, grid="y")
ax.set_title("Event cost (ensemble)", fontsize=8.0)
ax.text(-2.1, 86, "lower $\\lambda$: more events,\nplan SR up, strict down", fontsize=6.0, color="#555555", va="top")
panel_tag(ax, "(d)", dx=-0.2, dy=1.16)

h = [Line2D([], [], color=C_ALL, lw=1.3, marker="o", ms=3.2, mfc="white", label="All, plan SR"),
     Line2D([], [], color=C_ALL, lw=1.3, ls=(0, (3, 2)), marker="s", ms=3.2, mfc="white", label="All, strict"),
     Line2D([], [], color=C_NC, lw=1.0, marker="o", ms=3.2, mfc="white", label="NC, plan SR"),
     Line2D([], [], color=C_NC, lw=1.0, ls=(0, (3, 2)), marker="s", ms=3.2, mfc="white", label="NC, strict"),
     Line2D([], [], color=GREY, lw=0.6, ls=(0, (1, 1.5)), label="setting used in the paper")]
fig.legend(handles=h, loc="lower center", ncol=5, fontsize=6.6, frameon=False, bbox_to_anchor=(0.5, -0.01),
           handlelength=2.0, columnspacing=1.4)
save_figure(fig, OUT)
print(OUT)
