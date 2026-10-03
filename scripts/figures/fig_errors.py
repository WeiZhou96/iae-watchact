# -*- coding: utf-8 -*-
"""Where plans fail: gold-pair outcomes, extra objects and destination-error types (real per-request results).

Input: release/predictions/pairs_all.json (scripts/analysis/export_release.py; canonical-frame gold and predicted pairs for 455 NC/RD requests).
Output: figs/fig_errors.pdf (+ .png, 300 dpi)
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from stylelib import use_style, save_figure, style_ax, panel_tag  # noqa: E402
from stylelib.style_base import IEEE_TEXT_WIDTH_IN  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PRED = REPO / "release" / "predictions"      # released per-request predictions and scores
FDATA = REPO / "release" / "figure_data"     # released summaries used by the figures
D = json.loads((PRED / "pairs_all.json").read_text())
OUT = REPO / "figs" / "fig_errors.pdf"
R = {r["uid"]: r for r in D["requests"]}
METHODS = [("d8", "8B direct"), ("d32", "32B direct"), ("d8o", "8B + overlays"), ("d32o", "32B + overlays"),
           ("dvl", "InternVL3.5-8B direct"), ("handcrafted", "Hand-crafted evid."), ("iae_ens", "IAE (ensemble)")]
C_OK, C_DST, C_MISS = "#0072B2", "#E69F00", "#D9D9D9"
OURS = "#D55E00"
use_style()


def outcomes(m, task):
    M = D["methods"][m]; c = collections.Counter(); nreq = 0; extra_req = 0
    for k, r in R.items():
        if r["task"] != task: continue
        nreq += 1; g = dict(map(tuple, r["gold"])); p = dict(map(tuple, M[k]["pred"]))
        for o, d in g.items():
            c["ok" if p.get(o) == d else ("dst" if o in p else "miss")] += 1
        extra_req += any(o not in g for o in p)
    n = sum(c.values())
    return {k: 100 * c[k] / n for k in ("ok", "dst", "miss")}, 100 * extra_req / nreq, n, nreq


def dtype(d):
    if "drawer" in d or "cabinet" in d: return "drawer"
    if d.endswith("_contain_region"): return "container"
    return "region"


def dest_errors(m):
    M = D["methods"][m]; c = collections.Counter()
    for k, r in R.items():
        if r["task"] != "No": continue
        g = dict(map(tuple, r["gold"])); p = dict(map(tuple, M[k]["pred"]))
        for o, d in g.items():
            if o not in p or p[o] == d: continue
            a, b = dtype(d), dtype(p[o])
            if a == b == "container":
                kinds = {d.split("_")[0], p[o].split("_")[0]}
                c["tray \u2194 basket" if kinds == {"wooden", "basket"} else "same-kind container"] += 1
            elif a == b == "drawer": c["wrong drawer"] += 1
            elif a == b == "region": c["wrong table region"] += 1
            elif a == "region" and b == "container": c["region \u2192 container"] += 1
            elif a == "container" and b == "region": c["container \u2192 region"] += 1
            else: c["other"] += 1
    return c


fig = plt.figure(figsize=(IEEE_TEXT_WIDTH_IN, 2.8))
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 0.8], wspace=0.06, left=0.135, right=0.645, top=0.88, bottom=0.25)
y = np.arange(len(METHODS))[::-1]
stats = {}
for j, (task, name) in enumerate([("No", "NC"), ("Re", "RD")]):
    ax = fig.add_subplot(gs[0, j])
    ext_x = 118
    for i, (m, lab) in enumerate(METHODS):
        o, ext, n, nreq = outcomes(m, task); stats[(task, m)] = (o, ext)
        left = 0
        for key, col in (("ok", C_OK), ("dst", C_DST), ("miss", C_MISS)):
            ax.barh(y[i], o[key], left=left, color=col, height=0.62, lw=0, zorder=2)
            if o[key] >= 9:
                ax.text(left + o[key] / 2, y[i], "%.0f" % o[key], ha="center", va="center", fontsize=6.4,
                        color="white" if key != "miss" else "#333333", zorder=3)
            left += o[key]
        ax.plot([ext_x], [y[i]], marker="o", ms=4.2, mfc=OURS if m == "iae_ens" else "white",
                mec=OURS if m == "iae_ens" else "#555555", mew=0.9, zorder=3)
        ax.text(ext_x + 5, y[i], "%.0f" % ext, va="center", ha="left", fontsize=6.4,
                color=OURS if m == "iae_ens" else "#333333")
    ax.axvline(106, color="#BBBBBB", lw=0.5, zorder=0)
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100, ext_x])
    ax.set_xticklabels(["0", "25", "50", "75", "100", "extra\nobj."])
    ax.set_ylim(-0.6, len(METHODS) - 0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in METHODS] if j == 0 else [], fontsize=7.0)
    for t in ax.get_yticklabels():
        if t.get_text().startswith("IAE"): t.set_color(OURS); t.set_fontweight("bold")
    style_ax(ax, grid=None)
    ax.set_xlabel("Goal pairs (%)", fontsize=7.4, labelpad=1)
    ax.set_title("%s: %d goal pairs in %d requests" % (name, n, nreq), fontsize=8.0)
    panel_tag(ax, "(%s)" % "ab"[j], dx=-0.36 if j == 0 else -0.05, dy=1.14)
    if j == 0:
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        h = [Patch(color=C_OK, label="correct pair"), Patch(color=C_DST, label="object found, wrong destination"),
             Patch(color=C_MISS, label="object missed"),
             Line2D([], [], marker="o", ls="", mfc="white", mec="#555555", label="requests adding non-goal objects (%)")]
        fig.legend(handles=h, loc="lower center", ncol=4, fontsize=6.6, frameon=False, bbox_to_anchor=(0.5, -0.015),
                   handlelength=1.2, columnspacing=1.2)

# (c) NC destination-error types
ax = fig.add_axes([0.80, 0.27, 0.19, 0.60])
ci, cd = dest_errors("iae_ens"), dest_errors("d32")
cats = [k for k, _ in (ci + cd).most_common()]
yy = np.arange(len(cats))[::-1]
for i, k in enumerate(cats):
    ax.plot([cd[k], ci[k]], [yy[i], yy[i]], color="#CCCCCC", lw=1.2, zorder=1)
    ax.plot(cd[k], yy[i], "o", ms=4.2, mfc="white", mec="#0072B2", mew=0.9, zorder=2)
    ax.plot(ci[k], yy[i], "o", ms=4.4, color=OURS, zorder=3)
    ax.text(max(ci[k], cd[k]) + 2.5, yy[i], "%d / %d" % (ci[k], cd[k]), va="center", fontsize=6.2, color="#444444")
ax.set_yticks(yy)
ax.set_yticklabels(cats, fontsize=6.8)
ax.tick_params(axis="y", length=0, pad=2)
ax.set_xlim(0, max(max(ci.values()), max(cd.values())) * 1.35)
ax.set_ylim(-0.6, len(cats) - 0.4)
style_ax(ax, grid="x")
ax.spines["left"].set_visible(False)
ax.set_xlabel("NC goal pairs (count)", fontsize=7.4, labelpad=1)
ax.set_title("NC destination errors", fontsize=8.0, loc="right")
ax.plot([], [], "o", color=OURS, ms=4.4, label="IAE (ensemble)")
ax.plot([], [], "o", mfc="white", mec="#0072B2", ms=4.2, label="32B direct")
ax.legend(loc="lower right", fontsize=6.4, handletextpad=0.2, borderaxespad=0.1)
panel_tag(ax, "(c)", dx=-0.72, dy=1.14)
ax.tick_params(axis="y", labelsize=6.8)

save_figure(fig, OUT)
print(OUT)
for k, v in stats.items(): print(k, {a: round(b, 1) for a, b in v[0].items()}, "extra %.1f" % v[1])
print("IAE dest errors", dict(ci)); print("32B dest errors", dict(cd))
