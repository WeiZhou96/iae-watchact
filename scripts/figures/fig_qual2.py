# -*- coding: utf-8 -*-
"""Episodic examples and registration/identity tracking from real WatchAct videos and real IAE outputs.

Input: data/qual2 (exported by scripts/figures/export_qual2.py from the pipeline outputs; selection rules and seed
recorded there; the frames are WatchAct images and are not distributed with this repository).
(a,b) Episodic requests: first and last frames with the tracked instance masks and public identifiers at those
frames; arrows on the anchor frame show the container destinations of the program; text gives goal, IAE, 32B.
(c) One RD activity: last frames of the three cameras with the registered identifiers; registered ground points
pi(p(b)) of all instances against their canonical grid positions; image x-coordinate of the two identical
butter boxes over time in the front camera (tracks), with a frame in which one of them is handled.
Frames are cropped to rows 0-600 of 720 for every camera. Output: figs/fig_qual2.pdf (+ .png, 300 dpi)
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image
from stylelib import use_style, save_figure, style_ax  # noqa: E402
from stylelib.style_base import IEEE_TEXT_WIDTH_IN  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PRED = REPO / "release" / "predictions"      # released per-request predictions and scores
FDATA = REPO / "release" / "figure_data"     # released summaries used by the figures
Q = REPO / "data" / "qual2"
OUT = REPO / "figs" / "fig_qual2.pdf"
M = json.loads((Q / "qual2.json").read_text())
CROP_Y = 600
PAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]
OK, BAD = "#009E73", "#C0392B"
CAM = {"front": "#222222", "side": "#0072B2", "oblique": "#D55E00"}
use_style()


def short(x):
    return x.replace("_contain_region", "").replace("_region", "")


def prog_str(pairs):
    items = [(p["object_id"], p["destination_region"]) if isinstance(p, dict) else tuple(p) for p in pairs]
    return ";  ".join("%s$\\rightarrow$%s" % (short(o), short(d)) for o, d in items) if items else "no plan"


def rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def draw_frame(ax, cdir, fr, labels, colors, title, label_size=4.6):
    im = np.asarray(Image.open(cdir / fr["file"]).convert("RGB")).astype(float) / 255.0
    H, W = im.shape[:2]
    mk = np.unpackbits(np.load(cdir / fr["mask"]), axis=-1)[..., :fr["mask_shape"][-1]].astype(bool)
    over = im.copy()
    for j, lab in enumerate(labels):
        if lab is None or j >= mk.shape[0]: continue
        m = np.kron(mk[j], np.ones((4, 4), bool))[:H, :W]
        over[m] = 0.55 * over[m] + 0.45 * rgb(colors[lab])
    ax.imshow(over[:CROP_Y], interpolation="bilinear")
    for j, lab in enumerate(labels):
        b = fr["boxes"][j] if j < len(fr["boxes"]) else None
        if lab is None or b is None or b["box"][1] > CROP_Y: continue
        ax.text(b["box"][0], max(b["box"][1] - 5, 14), short(lab), fontsize=label_size, color="white", va="bottom",
                ha="left", bbox=dict(boxstyle="square,pad=0.1", fc=colors[lab], ec="none", alpha=0.9))
    ax.set_xlim(0, W); ax.set_ylim(CROP_Y, 0); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.4); s.set_color("#666666")
    ax.text(10, CROP_Y - 10, title, fontsize=5.6, color="white", va="bottom", ha="left",
            bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.55))
    return W


FIG_W, FIG_H = IEEE_TEXT_WIDTH_IN, 3.72
fig = plt.figure(figsize=(FIG_W, FIG_H))


def box(x, y, w, h):
    return [x / FIG_W, 1 - (y + h) / FIG_H, w / FIG_W, h / FIG_H]


# ---------------- (a), (b): episodic examples
FW = 1.72; FH = FW * CROP_Y / 1280; GAP = 0.03
TASKN = {"Res": "Restore", "Rev": "Reversal", "Imi": "Imitation"}
for k, name in enumerate(("A1", "A2")):
    case = M[name]; cdir = Q / name
    x0 = 0.02 + k * (2 * FW + GAP + 0.14)
    colors = {lab: PAL[j % len(PAL)] for j, lab in enumerate(case["labels"]) if lab}
    verdict = "correct" if case["iae_succ"] else "failure"
    fig.text(x0 / FIG_W, 1 - 0.02 / FIG_H, "(%s) %s, %s" % ("ab"[k], TASKN[case["task"]], verdict), fontsize=7.2,
             fontweight="bold", ha="left", va="top")
    first, last = case["frames"]
    anchor = case["anchor"]
    axes = []
    for i, (fr, tag) in enumerate(((first, "start"), (last, "end"))):
        ax = fig.add_axes(box(x0 + i * (FW + GAP), 0.2, FW, FH))
        is_anchor = (anchor == "first" and i == 0) or (anchor == "last" and i == 1)
        draw_frame(ax, cdir, fr, case["labels"], colors,
                   "%s  t = %.1f s%s" % (tag, fr["sec"], "  (anchor)" if is_anchor else ""))
        axes.append((ax, fr))
    # instances invisible in the start frame (hidden in a container)
    hidden = [lab for j, lab in enumerate(case["labels"]) if lab and first["boxes"][j] is None]
    # program arrows on the anchor frame towards container destinations
    ax_a, fr_a = axes[1] if anchor == "last" else axes[0]
    lab_idx = {lab: j for j, lab in enumerate(case["labels"]) if lab}
    for p in case["iae"]:
        o, d = p["object_id"], p["destination_region"]
        if not d.endswith("_contain_region"): continue
        c = d.replace("_contain_region", "")
        bo, bc = fr_a["boxes"][lab_idx[o]], fr_a["boxes"][lab_idx.get(c, -1)] if c in lab_idx else None
        if bo is None or bc is None: continue
        s = ((bo["box"][0] + bo["box"][2]) / 2, bo["box"][1]); e = ((bc["box"][0] + bc["box"][2]) / 2, bc["box"][1] + 10)
        ax_a.add_patch(FancyArrowPatch(s, e, arrowstyle="-|>", mutation_scale=6, lw=1.0, color=colors[o],
                                       connectionstyle="arc3,rad=-0.3", zorder=5))
    merged = []
    for a_ in range(len(case["labels"])):
        for b_ in range(a_ + 1, len(case["labels"])):
            ba, bb = first["boxes"][a_], first["boxes"][b_]
            if ba is not None and bb is not None and ba["box"] == bb["box"]:
                merged.append((case["labels"][a_], case["labels"][b_]))
    for la, lb in merged:
        ax0 = axes[0][0]
        ax0.text(1270, 14, "tracks of %s and %s coincide" % (short(la), short(lb)), fontsize=4.8, ha="right",
                 va="top", color="white", bbox=dict(boxstyle="square,pad=0.12", fc=BAD, ec="none", alpha=0.85))
    if hidden:
        ax0 = axes[0][0]
        ax0.text(1270, 14, "hidden at start: " + ", ".join(short(h) for h in hidden), fontsize=4.8, ha="right",
                 va="top", color="white", bbox=dict(boxstyle="square,pad=0.12", fc="#444444", ec="none", alpha=0.8))
    lines = [("Goal", case["gold"], "#222222", ""),
             ("IAE", case["iae"], OK if case["iae_succ"] else BAD, "(correct)" if case["iae_succ"] else "(wrong)"),
             ("32B VLM", case["d32"], OK if case["d32_succ"] else BAD, "(correct)" if case["d32_succ"] else "(wrong)")]
    for j, (lab, pairs, col, mark) in enumerate(lines):
        fig.text((x0 + 0.01) / FIG_W, 1 - (0.2 + FH + 0.035 + j * 0.1) / FIG_H,
                 "%s %s:  %s" % (lab, mark, prog_str(pairs)), fontsize=5.6, color=col, ha="left", va="top")

# ---------------- (c): registration across cameras and identity over time
B = M["B"]; bdir = Q / "B"
yB = 0.2 + FH + 0.6
fig.text(0.02 / FIG_W, 1 - (yB - 0.14) / FIG_H,
         "(c) Registration and identity: RD, \u201c%s\u201d" % B["instruction"].split(": ", 1)[-1], fontsize=7.2,
         fontweight="bold", ha="left", va="top")
TW = 1.13; TH = TW * CROP_Y / 1280
labs_all = sorted({l for v in B["views"].values() for l in v["labels"] if l})
bcol = {lab: PAL[j % len(PAL)] for j, lab in enumerate(labs_all)}
for i, v in enumerate(("front", "side", "oblique")):
    e = B["views"][v]
    ax = fig.add_axes(box(0.02 + i * (TW + 0.025), yB + 0.02, TW, TH))
    draw_frame(ax, bdir, e["frame"], e["labels"], bcol, "%s, end (anchor)" % v, label_size=4.0)
# outcomes of all requests on this activity (from the per-request results used for Table 1)
PA = json.loads((PRED / "pairs_all.json").read_text())
reqs = [r for r in PA["requests"] if r["activity"] == B["activity"]]
n_i = sum(PA["methods"]["iae_ens"][r["uid"]]["succ"] for r in reqs)
n_d = sum(PA["methods"]["d32"][r["uid"]]["succ"] for r in reqs)
wrong32 = sorted({(tuple(map(tuple, PA["methods"]["d32"][r["uid"]]["pred"])), r["view"]) for r in reqs
                  if not PA["methods"]["d32"][r["uid"]]["succ"]})
views32 = sorted({v for _, v in wrong32})
pred32 = wrong32[0][0] if wrong32 else ()
iae_pred = tuple(map(tuple, PA["methods"]["iae_ens"][reqs[0]["uid"]]["pred"]))
ytxt = yB + 0.02 + TH + 0.05
blines = [("Goal (%d requests)" % len(reqs), prog_str([tuple(g) for g in reqs[0]["gold"]]), "#222222"),
          ("IAE (%d/%d correct)" % (n_i, len(reqs)), prog_str(list(iae_pred)), OK),
          ("32B VLM (%d/%d correct)" % (n_d, len(reqs)),
           prog_str(list(pred32)) + " on %s requests" % " and ".join(views32), BAD)]
for j, (lab, txt, col) in enumerate(blines):
    fig.text(0.03 / FIG_W, 1 - (ytxt + j * 0.1) / FIG_H, "%s:  %s" % (lab, txt), fontsize=5.6, color=col,
             ha="left", va="top")
# registration scatter in canonical coordinates
ax = fig.add_axes(box(0.02 + 3 * (TW + 0.025) + 0.3, yB + 0.02, 1.25, TH + 0.33))
for gx in (-0.5, 0.5, 1.5, 2.5):
    ax.plot([gx, gx], [-0.5, 2.5], color="#DDDDDD", lw=0.5, zorder=0)
for gy in (-0.5, 0.5, 1.5, 2.5):
    ax.plot([-0.5, 2.5], [gy, gy], color="#DDDDDD", lw=0.5, zorder=0)
ax.add_patch(Rectangle((-1.5, -0.5), 1.0, 3.0, fc="#F4F4F4", ec="#DDDDDD", lw=0.5, zorder=0))
for lab, q in B["canonical"].items():
    ax.plot(q[0], q[1], marker="s", ms=6.5, mfc="none", mec=bcol.get(lab, "#555555"), mew=1.0, zorder=2)
    ax.text(q[0] + 0.26, q[1] - 0.22, short(lab), fontsize=4.6, ha="left" if q[0] < 0 else "center", va="bottom",
            color=bcol.get(lab, "#555555"))
for v, mk in (("front", "o"), ("side", "^"), ("oblique", "D")):
    for lab, pt in B["views"][v]["registered_ground_points"].items():
        q = B["canonical"].get(lab)
        if q is None: continue
        ax.plot([q[0], pt[0]], [q[1], pt[1]], color=bcol[lab], lw=0.5, alpha=0.7, zorder=1)
        ax.plot(pt[0], pt[1], marker=mk, ms=2.8, color=bcol[lab], mec="white", mew=0.3, ls="", zorder=3)
ax.set_xlim(-1.6, 2.6); ax.set_ylim(2.75, -0.6)
ax.set_xticks([-1, 0, 1, 2]); ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["back", "mid", "front"], fontsize=5.6)
ax.set_xticklabels(["fix.", "left", "center", "right"], fontsize=5.6)
ax.tick_params(length=0, pad=1)
for s in ax.spines.values(): s.set_visible(False)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="s", ls="", mfc="none", mec="#555555", ms=5, label="public position"),
     Line2D([], [], marker="o", ls="", color="#555555", ms=3, label="front"),
     Line2D([], [], marker="^", ls="", color="#555555", ms=3, label="side"),
     Line2D([], [], marker="D", ls="", color="#555555", ms=3, label="oblique")]
ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=5.0, handletextpad=0.2,
          columnspacing=0.6, borderaxespad=0)
res = {v: B["views"][v]["residual"] for v in B["views"]}
ax.set_title("registered ground points\n(RMS residual %.2f\u2013%.2f cells)" % (min(res.values()), max(res.values())),
             fontsize=6.0, loc="center")
# identity over time (front camera): x of the two same-category instances
e = B["views"]["front"]; tsec = np.array(e["tsec"])
cats = e["cats"]; dup = [j for j, c in enumerate(cats) if cats.count(c) >= 2 and c not in ("basket", "wooden_tray")]
ax = fig.add_axes(box(FIG_W - 1.72, yB + 0.02, 1.68, TH + 0.33))
for j in dup:
    cs = e["centroids"][j]; cT = next(c for c in reversed(cs) if c is not None)
    dist = np.array([np.hypot(c[0] - cT[0], c[1] - cT[1]) if c is not None else np.nan for c in cs], float)
    ax.plot(tsec[:len(dist)], dist, color=bcol[e["labels"][j]], lw=1.1, label=short(e["labels"][j]))
for g in e.get("mid_frames", []):
    ax.axvline(g["sec"], color="#999999", lw=0.5, ls=(0, (1, 1.5)))
ax.set_xlabel("Time (s)", fontsize=6.4, labelpad=1)
ax.set_ylabel("distance to final position (px)", fontsize=6.0, labelpad=1)
ax.tick_params(labelsize=5.6)
style_ax(ax, grid=None)
ax.legend(loc="center left", fontsize=5.4, handlelength=1.2)
ax.set_title("front-camera tracks of the identical boxes", fontsize=6.0, loc="center")
# inset: the frame in which a box is carried, with its identity
if e.get("mid_frames"):
    g = e["mid_frames"][-1]
    ins = fig.add_axes(box(FIG_W - 1.72 + 0.93, yB + 0.08, 0.72, 0.72 * CROP_Y / 1280))
    draw_frame(ins, bdir, g, e["labels"], bcol, "t = %.1f s" % g["sec"], label_size=3.6)

save_figure(fig, OUT)
print(OUT, {v: round(r, 3) for v, r in res.items()})
