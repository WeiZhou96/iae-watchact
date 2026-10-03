# -*- coding: utf-8 -*-
"""Selective prediction and activity-level paired gains (real per-request results).

Inputs (release/predictions):
  iae_ensemble_rows.json per-request output of the 3-seed IAE ensemble (conf = decoding/selection margin)
  direct_rows.json     per-request scores of the direct VLM baselines (same 455 NC/RD requests)
Output: figs/fig_selective.pdf (+ .png, 300 dpi)
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from stylelib import use_style, save_figure, ROLE, style_ax, panel_tag  # noqa: E402
from stylelib.style_base import IEEE_TEXT_WIDTH_IN, FONT_STATUS  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PRED = REPO / "release" / "predictions"      # released per-request predictions and scores
FDATA = REPO / "release" / "figure_data"     # released summaries used by the figures
OUT = REPO / "figs" / "fig_selective.pdf"

OURS = "#D55E00"
BLUE = "#0072B2"
SKY = "#56B4E9"
GREY = "#8C8C8C"
FAIL = "#E3E3E3"
TXT = "#555555"
TASK_NAME = {"No": "NC", "Re": "RD"}
rng = np.random.default_rng(0)
MIN_K = 10

use_style()
ours = json.loads((PRED / "iae_ensemble_rows.json").read_text())
direct = json.loads((PRED / "direct_rows.json").read_text())
d32 = {r["uid"]: r for r in direct["direct_32b"]}
d8o = {r["uid"]: r for r in direct["direct_8b_overlay"]}


def coverage_curve(rows, key="strict"):
    order = sorted(rows, key=lambda r: -r["conf"])
    y = np.cumsum([r[key] for r in order]) / np.arange(1, len(order) + 1)
    return np.arange(1, len(order) + 1) / len(order), y, order


def boot_band(rows, key="strict", B=1000):
    """Activity-cluster bootstrap of the risk-coverage curve, evaluated on a fixed coverage grid."""
    acts = sorted({r["activity"] for r in rows})
    by = {a: [r for r in rows if r["activity"] == a] for a in acts}
    grid = np.linspace(0.05, 1.0, 96)
    curves = []
    for _ in range(B):
        samp = [r for a in rng.choice(acts, len(acts)) for r in by[a]]
        c, y, _ = coverage_curve(samp, key)
        curves.append(np.interp(grid, c, y))
    lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
    return grid, lo, hi


def at(c, y, cov):
    return y[int(np.ceil(cov * len(c))) - 1]


fig = plt.figure(figsize=(IEEE_TEXT_WIDTH_IN, 2.35))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 0.075], width_ratios=[1, 1, 1.3],
              hspace=0.08, wspace=0.34, left=0.06, right=0.995, top=0.90, bottom=0.17)

summary = {}
for j, task in enumerate(["Re", "No"]):
    rows = [r for r in ours if r["task"] == task]
    ax = fig.add_subplot(gs[0, j])
    c, y, order = coverage_curve(rows)
    cs, ys, _ = coverage_curve(rows, "succ")
    g, lo, hi = boot_band(rows)
    ax.fill_between(g, 100 * lo, 100 * hi, color=OURS, alpha=0.13, lw=0, zorder=2)
    k0 = MIN_K - 1  # curves start at MIN_K committed requests (small denominators are noise)
    ax.plot(c[k0:], 100 * y[k0:], color=OURS, lw=1.8, zorder=4, label="IAE strict")
    ax.plot(cs[k0:], 100 * ys[k0:], color=OURS, lw=1.0, ls=(0, (3, 2)), zorder=3, label="IAE plan SR")
    # references for the ranking itself: oracle (all strict successes first) and random order (constant rate)
    n_all = len(rows); n_ok = sum(r["strict"] for r in rows); kk = np.arange(1, n_all + 1)
    y_or = np.minimum(1.0, n_ok / kk); y_rd = np.full(n_all, n_ok / n_all)
    ax.plot(kk[k0:] / n_all, 100 * y_or[k0:], color="#555555", lw=0.8, ls=(0, (1, 1.2)), zorder=2,
            label="oracle ranking")
    ax.plot(kk / n_all, 100 * y_rd, color="#999999", lw=0.8, ls=(0, (5, 2, 1, 2)), zorder=2, label="random ranking")
    auc = {nm: 100 * float(np.mean(v)) for nm, v in (("IAE", y), ("oracle", y_or), ("random", y_rd))}
    auc_txt = "area under strict curve\nIAE %.1f | oracle %.1f | random %.1f" % (auc["IAE"], auc["oracle"], auc["random"])
    ax.text(0.98 if task == "Re" else 0.97, 0.03 if task == "Re" else 0.97, auc_txt, transform=ax.transAxes,
            fontsize=6.0, color="#444444", ha="right",
            va="bottom" if task == "Re" else "top", zorder=6,
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.85))
    summary.setdefault("auc", {})[task] = {k_: round(v_, 1) for k_, v_ in auc.items()}
    # baselines at full coverage (they expose no confidence)
    b32 = 100 * np.mean([d32[r["uid"]]["strict"] for r in rows])
    b8o = 100 * np.mean([d8o[r["uid"]]["strict"] for r in rows])
    for val, col, name in [(b32, BLUE, "32B direct"), (b8o, SKY, "8B + overlay")]:
        ax.axhline(val, color=col, lw=0.9, ls=(0, (4, 3)), zorder=1)
    # direct labels stacked above the upper reference line, in the same vertical order as the lines
    upper = max(b32, b8o)
    if task == "Re":
        ax.text(0.02, upper + 6.2, "8B + overlay %.1f" % b8o, color=SKY, fontsize=6.6, va="bottom", ha="left")
        ax.text(0.02, upper + 1.2, "32B direct %.1f" % b32, color=BLUE, fontsize=6.6, va="bottom", ha="left")
    else:  # NC: one line between the baseline lines and the random-ranking line
        ax.text(0.24, upper + 1.2, "8B + overlay %.1f" % b8o, color=SKY, fontsize=6.6, va="bottom", ha="left")
        ax.text(0.68, upper + 1.2, "32B direct %.1f" % b32, color=BLUE, fontsize=6.6, va="bottom", ha="left")
    for cov in (0.4, 0.6, 1.0):
        v = 100 * at(c, y, cov)
        ax.plot([cov], [v], marker="o", ms=3.6, mfc="white", mec=OURS, mew=0.9, zorder=5)
        end_off = (-2, -11) if task == "Re" else (-2, 4)
        ax.annotate("%.1f" % v, (cov, v), xytext=end_off if cov == 1.0 else (0, 4), textcoords="offset points",
                    ha="right" if cov == 1.0 else "center",
                    va="bottom", fontsize=6.8, color=OURS)
    summary[task] = {cov: round(100 * at(c, y, cov), 1) for cov in (0.2, 0.4, 0.6, 0.8, 1.0)}
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 105)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("Success of committed requests (%)" if j == 0 else "")
    style_ax(ax, grid="y")
    ax.set_title("%s: %d requests ranked by margin" % (TASK_NAME[task], len(rows)), fontsize=8.5)
    panel_tag(ax, "(%s)" % "ab"[j], dx=-0.2 if j == 0 else -0.14, dy=1.13)
    if j == 0:
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.40), ncol=2, fontsize=6.4, handlelength=1.8, columnspacing=0.9)
    # one tile per request, ordered by margin; filled = strict success
    tx = fig.add_subplot(gs[1, j], sharex=ax)
    n = len(order)
    cols = [OURS if r["strict"] else FAIL for r in order]
    tx.bar((np.arange(n) + 0.5) / n, np.ones(n), width=1.0 / n, color=cols, lw=0, align="center")
    tx.set_ylim(0, 1)
    tx.set_yticks([])
    for s in ("left", "top", "right"):
        tx.spines[s].set_visible(False)
    tx.set_xlabel("Coverage (fraction of requests committed)")
    tx.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    tx.tick_params(labelbottom=True)
    tx.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])

# (c) activity-level paired gains over the 32B direct VLM
ax = fig.add_subplot(gs[:, 2])
stats = {}
for k, task in enumerate(["Re", "No"]):
    rows = [r for r in ours if r["task"] == task]
    acts = sorted({r["activity"] for r in rows})
    diff = []
    for a in acts:
        rs = [r for r in rows if r["activity"] == a]
        diff.append(100 * (np.mean([r["succ"] for r in rs]) - np.mean([d32[r["uid"]]["succ"] for r in rs])))
    diff = np.array(diff)
    # deterministic dodge of tied values (layout only; values unchanged)
    xs = np.zeros(len(diff))
    for v in np.unique(diff):
        idx = np.where(diff == v)[0]
        xs[idx] = (np.arange(len(idx)) - (len(idx) - 1) / 2) * 0.024
    x0 = 1.45 * k
    ax.scatter(x0 + xs, diff, s=9, color=np.where(diff > 0, OURS, np.where(diff < 0, BLUE, GREY)),
               alpha=0.85, lw=0, zorder=3)
    boots = [np.mean(rng.choice(diff, len(diff))) for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    m = diff.mean()
    ax.plot([x0 + 0.52, x0 + 0.52], [lo, hi], color="#222222", lw=1.3, zorder=4)
    ax.plot([x0 + 0.46, x0 + 0.58], [m, m], color="#222222", lw=1.3, zorder=4)
    ax.text(x0 + 0.61, m, "%+.1f\n[%+.1f, %+.1f]" % (m, lo, hi), fontsize=6.6, va="center", ha="left",
            color="#222222")
    w, t, l = int((diff > 0).sum()), int((diff == 0).sum()), int((diff < 0).sum())
    stats[task] = dict(n=len(diff), mean=round(m, 1), lo=round(lo, 1), hi=round(hi, 1), win=w, tie=t, loss=l)
ax.axhline(0, color="#B0B0B0", lw=0.9, ls=(0, (4, 3)), zorder=1)
ax.set_xlim(-0.45, 2.45)
ax.set_ylim(-108, 108)
ax.set_xticks([0, 1.45])
ax.set_xticklabels(["RD, %d activities\n%d win / %d tie / %d loss" % tuple(stats["Re"][x] for x in ("n", "win", "tie", "loss")),
                    "NC, %d activities\n%d win / %d tie / %d loss" % tuple(stats["No"][x] for x in ("n", "win", "tie", "loss"))],
                   fontsize=7.0)
ax.set_ylabel("IAE $-$ 32B direct, plan SR (pp)")
style_ax(ax, grid="y")
ax.set_title("Per-activity gain over the 32B VLM", fontsize=8.5)
panel_tag(ax, "(c)", dx=-0.19, dy=1.075)

pdf = save_figure(fig, OUT)
print(pdf)
print("fonts:", FONT_STATUS)
print("coverage:", summary)
print("activity gains:", stats)
