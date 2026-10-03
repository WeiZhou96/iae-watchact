# -*- coding: utf-8 -*-
"""Qualitative cases from real WatchAct videos and real IAE / VLM outputs.

Input: data/iae_cases (exported by scripts/figures/export_cases.py from the pipeline outputs; selection rule and
seed are recorded there; the frames are WatchAct images and are not distributed with this repository). Frames are shown with the tracked instance masks, hand landmarks and forearm lines that
IAE actually computed at those frames; curves are the frame logits of the three-seed ensemble.
Processing: every frame is cropped to rows 0-600 of 720 (floor removed), identically for all cases.
Output: figs/fig_cases.pdf (+ .png, 300 dpi)
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from stylelib import use_style, save_figure, style_ax  # noqa: E402
from stylelib.style_base import IEEE_TEXT_WIDTH_IN  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PRED = REPO / "release" / "predictions"      # released per-request predictions and scores
FDATA = REPO / "release" / "figure_data"     # released summaries used by the figures
CASES = REPO / "data" / "iae_cases"
OUT = REPO / "figs" / "fig_cases.pdf"
CROP_Y = 600
PAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]
GREY = "#BDBDBD"
OK, BAD = "#009E73", "#C0392B"
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]

use_style()
M = json.loads((CASES / "cases.json").read_text())


def short(x):
    return x.replace("_contain_region", "").replace("_region", "")


def prog_str(pairs):
    if not pairs:
        return "no plan"
    items = [(p["object_id"], p["destination_region"]) if isinstance(p, dict) else tuple(p) for p in pairs]
    return ";  ".join("%s$\\rightarrow$%s" % (short(o), short(d)) for o, d in items)


def color_of(case, cid):
    labels = case["labels"]
    return PAL[labels.index(cid) % len(PAL)] if cid in labels else None


def draw_frame(ax, case, cdir, fr, num):
    im = np.asarray(Image.open(cdir / fr["file"]).convert("RGB")).astype(float) / 255.0
    H, W = im.shape[:2]
    shape = fr["mask_shape"]
    mk = np.unpackbits(np.load(cdir / fr["mask"]), axis=-1)[..., :shape[-1]].astype(bool)
    over = im.copy()
    for j, lab in enumerate(case["labels"]):
        m = np.kron(mk[j], np.ones((4, 4), bool))[:H, :W]
        rgb = np.array([int(PAL[j % len(PAL)][i:i + 2], 16) / 255 for i in (1, 3, 5)])
        over[m] = 0.55 * over[m] + 0.45 * rgb
    ax.imshow(over[:CROP_Y], interpolation="bilinear")
    for j, lab in enumerate(case["labels"]):
        b = fr["boxes"][j] if j < len(fr["boxes"]) else None
        if b is None or b["box"][1] > CROP_Y:
            continue
        x1, y1 = b["box"][0], b["box"][1]
        ax.text(x1, max(y1 - 6, 14), short(lab), fontsize=4.6, color="white", va="bottom", ha="left",
                bbox=dict(boxstyle="square,pad=0.12", fc=PAL[j % len(PAL)], ec="none", alpha=0.9))
    body = fr.get("body")
    sc = fr.get("body_sc") or []
    if body:
        for s, e, w in ((5, 7, 9), (6, 8, 10)):
            if min(sc[s], sc[e], sc[w]) < 0.3:
                continue
            ax.plot([body[s][0], body[e][0], body[w][0]], [body[s][1], body[e][1], body[w][1]], color="white", lw=0.8)
            d = np.array(body[w]) - np.array(body[e])
            d = d / (np.linalg.norm(d) + 1e-6)
            tip = np.array(body[w]) + 260 * d
            ax.plot([body[w][0], tip[0]], [body[w][1], tip[1]], color="#FFD400", lw=0.8, ls=(0, (2.5, 1.5)))
    for h in fr.get("hands", []):
        kp = np.array(h["kp"])
        for a, b in HAND_EDGES:
            ax.plot(kp[[a, b], 0], kp[[a, b], 1], color="white", lw=0.45)
        ax.plot(kp[8, 0], kp[8, 1], "o", ms=1.6, color="#FFD400", mec="none")
    ax.set_xlim(0, W)
    ax.set_ylim(CROP_Y, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.4)
        s.set_color("#666666")
    ax.text(10, CROP_Y - 10, "%d  t = %.1f s" % (num, fr["sec"]), fontsize=5.6, color="white", va="bottom", ha="left",
            bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.55))


def draw_curves(ax, case, shown):
    z = np.array(case["z"])
    tsec = np.array(case["fps_frames"])[: len(z)]
    gold = {o for o, _ in case["gold"]} | {d.replace("_contain_region", "") for _, d in case["gold"]}
    if case["activity"].startswith("eliminating"):
        ev_ids = {c for c, kd in zip(case["cand_ids"], case["cand_kind"]) if kd == "movable"}
    else:
        ev_ids = {e[1] for e in case["events"]}
    keep = [k for k, c in enumerate(case["cand_ids"]) if c in gold or c in ev_ids]
    for k, c in enumerate(case["cand_ids"]):
        if k not in keep:
            ax.plot(tsec, z[:, k], color=GREY, lw=0.4, alpha=0.7, zorder=1)
    for k in keep:
        c = case["cand_ids"][k]
        col = color_of(case, c) or "#555555"
        ls = "-" if c in gold else (0, (2, 1.2))
        ax.plot(tsec, z[:, k], color=col, lw=1.0, ls=ls, zorder=3, label=short(c))
    # decoded deictic events exist only for NC; the RD program selects instances without events
    events = [] if case["activity"].startswith("eliminating") else case["events"]
    for t, cid, role, v in events:
        ax.plot([tsec[t]], [z[t, case["cand_ids"].index(cid)]], marker="o" if role == "o" else "s", ms=3.0,
                mfc="white", mec="#222222", mew=0.7, zorder=6)
    ymin = z.min() - 0.5
    for i, t in enumerate(shown):
        ax.annotate(str(i + 1), (tsec[t], ymin), fontsize=5.6, ha="center", va="bottom", color="#222222")
        ax.axvline(tsec[t], color="#999999", lw=0.5, ls=(0, (1, 1.5)), zorder=0)
    ax.axhline(0, color="#B0B0B0", lw=0.6, ls=(0, (4, 3)), zorder=0)
    ax.set_xlim(tsec[0], tsec[-1])
    ax.set_ylim(ymin, z.max() + 0.35 * (z.max() - ymin))
    ax.set_xlabel("Time (s)", fontsize=7.0, labelpad=1)
    ax.set_ylabel("Frame logit", fontsize=7.0, labelpad=1)
    ax.tick_params(labelsize=6.2)
    style_ax(ax, grid=None)
    ax.legend(loc="upper left", ncol=3, fontsize=5.0, handlelength=1.4, columnspacing=0.8, borderaxespad=0.1,
              handletextpad=0.3)


ROWS = [
    ("A", "(a) NC, correct: two object\u2013destination cues; the 32B VLM sends all three objects to the tray"),
    ("C", "(b) RD, correct: \u201cPlease place the other one closer to the person\u2019s basket\u201d, with two identical milk cartons"),
    ("B", "(c) NC, failure: both objects correct, but their destinations are swapped between the tray and the basket"),
]

# shown frames: NC = first object cue, first destination cue, last destination cue (decoded events);
# RD = evidence peaks of the two identical cartons and the anchor (final) frame
SHOW = {"A": (59, 90, 191), "B": (88, 135, 154), "C": (91, 140, 332)}

# fixed layout in inches so that frames, program text and curves align across rows
FW, GAP, X0 = 1.53, 0.035, 0.02
FH = FW * CROP_Y / 1280
TITLE_H, TEXT_H, ROW_GAP = 0.17, 0.36, 0.10
ROW_H = TITLE_H + FH + TEXT_H + ROW_GAP
FIG_W, FIG_H = IEEE_TEXT_WIDTH_IN, len(ROWS) * ROW_H + 0.02
fig = plt.figure(figsize=(FIG_W, FIG_H))


def box(x, y, w, h):
    """Axes rectangle given in inches from the top-left corner of the figure."""
    return [x / FIG_W, 1 - (y + h) / FIG_H, w / FIG_W, h / FIG_H]


for r, (name, title) in enumerate(ROWS):
    case = M[name]
    cdir = CASES / name
    frames = [f for f in case["frames"]]
    # distinct frames in time order; keep at most three (the last is the anchor frame)
    uniq = []
    for f in frames:
        if f["t"] not in [u["t"] for u in uniq]:
            uniq.append(f)
    uniq = [u for u in uniq if u["t"] in SHOW[name]]
    y_top = r * ROW_H
    fig.text(X0 / FIG_W, 1 - (y_top + 0.02) / FIG_H, title, fontsize=7.2, ha="left", va="top", fontweight="bold")
    yf = y_top + TITLE_H
    for i, fr in enumerate(uniq):
        ax = fig.add_axes(box(X0 + i * (FW + GAP), yf, FW, FH))
        ax.set_aspect("auto")
        draw_frame(ax, case, cdir, fr, i + 1)
    cx = X0 + 3 * (FW + GAP) + 0.36
    ax = fig.add_axes(box(cx, yf, FIG_W - cx - 0.03, FH + TEXT_H - 0.24))
    draw_curves(ax, case, [f["t"] for f in uniq])
    lines = [("Goal", case["gold"], "#222222", ""),
             ("IAE", case["iae"], OK if case["iae_succ"] else BAD, "(correct)" if case["iae_succ"] else "(wrong)"),
             ("32B VLM", case["d32"], OK if case["d32_succ"] else BAD, "(correct)" if case["d32_succ"] else "(wrong)")]
    for j, (lab, pairs, col, mark) in enumerate(lines):
        fig.text((X0 + 0.01) / FIG_W, 1 - (yf + FH + 0.04 + j * 0.105) / FIG_H,
                 "%s %s:  %s" % (lab, mark, prog_str(pairs)), fontsize=6.0, color=col, ha="left", va="top")

save_figure(fig, OUT)
print(OUT)
