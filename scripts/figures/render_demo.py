"""Render IAE demonstration videos from real pipeline outputs.

Every overlay comes from the exported pipeline outputs of one request (cases/<tag>/meta.json, masks.npz):
registered instance IDs, SAM 2.1 masks (stored at 1/4 resolution; upsampled and smoothed for display),
MediaPipe hand keypoints, per-frame evidence of the three-seed ensemble (held-out fold), the decoded
program, and WatchAct scores. Overlays are sampled at 10 fps and shown on the 29.97 fps source video:
keypoints are linearly interpolated between samples, masks and scores use the nearest sample.

Usage: python render_demo.py <case_dir> <out.mp4> [--still N ...]
"""
from __future__ import annotations
import json, math, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------- style
W_OUT, H_OUT = 1920, 1080
FPS = 30000 / 1001
FONTS = 'C:/Windows/Fonts/'
_MPL = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
# Segoe UI / Consolas where available (the released clips), DejaVu (bundled with matplotlib) elsewhere
FONT_FILES = {'regular': ('segoeui.ttf', 'DejaVuSans.ttf'), 'semibold': ('seguisb.ttf', 'DejaVuSans-Bold.ttf'),
              'bold': ('segoeuib.ttf', 'DejaVuSans-Bold.ttf'), 'light': ('segoeuil.ttf', 'DejaVuSans.ttf'),
              'mono': ('consola.ttf', 'DejaVuSansMono.ttf'), 'monob': ('consolab.ttf', 'DejaVuSansMono-Bold.ttf')}
_fc = {}


def font(size, w='regular'):
    key = (size, w)
    if key not in _fc:
        win, fallback = FONT_FILES[w]
        path = Path(FONTS) / win
        _fc[key] = ImageFont.truetype(str(path if path.exists() else _MPL / fallback), size)
    return _fc[key]


def hexrgb(h):
    h = h.lstrip('#'); return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


BG = hexrgb('#F7F7F4'); PANEL = (255, 255, 255); BORDER = hexrgb('#E2E2DC'); INK = hexrgb('#1E2227')
MUTED = hexrgb('#6A717B'); FAINT = hexrgb('#B9BEC6'); OK = hexrgb('#1A7F37'); BAD = hexrgb('#C0392B')
OBJ_COLORS = ['#0072B2', '#D55E00', '#009E73', '#882255']   # movable objects
CONT_COLORS = ['#CC79A7', '#E69F00', '#56B4E9', '#999933']  # containers and fixtures
VX, VY, VW, VH = 40, 112, 1280, 720            # video panel
RX, RY, RW, RH = 1352, 112, 528, 720           # right panel
TX, TY, TW, TH = 40, 852, 1840, 196            # timeline panel

RAY_POSE = 0.75  # the fingertip ray is drawn only when the hand forms a pointing pose
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

TASK_TITLE = {'Nonverbal_Cue': 'Nonverbal cue: follow the pointing gestures',
              'Reference_Disambiguation': 'Reference disambiguation: which of the identical objects?',
              'Imitation': 'Imitation: repeat the demonstrated rearrangement',
              'Reversal': 'Reversal: undo the demonstrated actions',
              'Restore_Previous_State': 'Restore: put the moved objects back'}


def sig(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


def pretty_dest(d):
    if d.endswith('_contain_region'): return d[:-len('_contain_region')]
    if d.startswith('main_'):
        r, c = d[5:-7].split('_'); return f'table {r}-{c}'
    if d.startswith('fixture_'): return 'fixture ' + d.split('_')[1]
    for lv in ('top', 'middle', 'bottom'):
        if d.endswith(f'_{lv}_region'): return d[:-len(f'_{lv}_region')] + f' {lv} drawer'
    return d


def ease(x):
    x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)


# ---------------------------------------------------------------- data
class Case:
    def __init__(self, d: Path):
        self.dir = d
        self.m = m = json.loads((d / 'meta.json').read_text())
        self.task = m['task']; self.episodic = 'cands' not in m
        tr = m['tracks']; self.tr = tr; self.T = len(tr['per_frame'])
        self.labels, self.cats = tr['labels'], tr['cats']
        idx = m['frame_index']; self.natives = np.array([f['native'] for f in idx['frames']], float)
        self.native_fps = idx['native_fps']
        z = np.load(d / 'masks.npz'); shp = tuple(z['shape']); bits = np.unpackbits(z['masks'], axis=-1)
        self.masks = bits[..., :shp[-1]].astype(bool)
        self.hands = {r['i']: r['hands'] for r in (m.get('hands') or [])}
        self.body = {r['i']: r for r in (m.get('body') or []) if r.get('kp')}
        # colours by track index for labelled instances (movable objects first, then containers)
        lab = [k for k, l in enumerate(self.labels) if l is not None]
        mov = sorted([k for k in lab if self.kind(k) == 'object'], key=lambda k: self.labels[k])
        oth = sorted([k for k in lab if k not in mov], key=lambda k: self.labels[k])
        self.color = {k: hexrgb(OBJ_COLORS[i % len(OBJ_COLORS)]) for i, k in enumerate(mov)}
        self.color.update({k: hexrgb(CONT_COLORS[i % len(CONT_COLORS)]) for i, k in enumerate(oth)})
        self.order = mov + oth
        if not self.episodic:
            self.cands = m['cands']; self.z = np.asarray(m['z']); self.s = np.asarray(m['s'])
            self.cand_of_track = {}
            for j, c in enumerate(self.cands):
                if 'track' in c and c['kind'] != 'drawer': self.cand_of_track[c['track']] = j
        self._mask_cache = {}
        self.anchor_end = 0 if tr.get('anchor') == 0 else -1
        self.scene_t = 0 if self.anchor_end == 0 else self.T - 1
        self.frames = None

    def kind(self, k):
        c = self.cats[k]
        return 'container' if c in ('basket', 'wooden_tray') else 'fixture' if c == 'wooden_cabinet' else 'object'

    def load_video(self):
        ff = shutil.which('ffmpeg')
        cmd = [ff, '-v', 'error', '-i', str(self.dir / 'source.mp4'), '-vf', f'scale={VW}:{VH}:flags=lanczos',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        n = len(raw) // (VW * VH * 3)
        self.frames = np.frombuffer(raw, np.uint8)[:n * VW * VH * 3].reshape(n, VH, VW, 3)
        print('decoded', n, 'frames', flush=True)

    def u_of_native(self, n):
        return float(np.interp(n, self.natives, np.arange(self.T)))

    def native_of_u(self, u):
        return float(np.interp(u, np.arange(self.T), self.natives))

    def mask(self, t, k):
        key = (t, k)
        if key not in self._mask_cache:
            m = self.masks[t, k].astype(np.float32)
            m = cv2.resize(m, (VW, VH), interpolation=cv2.INTER_LINEAR)
            m = cv2.GaussianBlur(m, (0, 0), 2.0) > 0.5
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            self._mask_cache[key] = (m, cs)
        return self._mask_cache[key]

    def box(self, t, k):
        b = self.tr['per_frame'][t][k]
        return None if b is None else b['box']

    def center(self, t, k):
        b = self.tr['per_frame'][t][k]
        if b is None:  # nearest visible sample
            for dt in range(1, self.T):
                for tt in (t - dt, t + dt):
                    if 0 <= tt < self.T and self.tr['per_frame'][tt][k] is not None:
                        return self.center(tt, k)
            return None
        x0, y0, x1, y1 = b['box']; return np.array([(x0 + x1) / 2, (y0 + y1) / 2])

    def track_of(self, label):
        return self.labels.index(label) if label in self.labels else None

    def dest_point(self, d, t):
        if d.endswith('_contain_region'):
            k = self.track_of(d[:-len('_contain_region')])
            if k is not None:
                b = self.box(t, k) or self.box(self.T - 1, k)
                return np.array([(b[0] + b[2]) / 2, b[1] + 0.40 * (b[3] - b[1])])
        for j, lv in enumerate(('top', 'middle', 'bottom')):
            if d.endswith(f'_{lv}_region') and not d.startswith('main_'):
                k = self.track_of(d[:-len(f'_{lv}_region')])
                if k is not None:
                    b = self.box(t, k); h = (b[3] - b[1]) / 3
                    return np.array([(b[0] + b[2]) / 2, b[1] + (j + 0.5) * h])
        p = self.m['region_points'].get(d)
        return None if p is None else np.array(p)

    def hands_at(self, u):
        """Hand keypoints at fractional sample u: linear interpolation between samples, matched by handedness."""
        t0 = int(math.floor(u)); t1 = min(t0 + 1, self.T - 1); a = u - t0
        h0 = {h['handed']: h for h in self.hands.get(t0, [])}; h1 = {h['handed']: h for h in self.hands.get(t1, [])}
        out = []
        for key in set(h0) | set(h1):
            if key in h0 and key in h1:
                kp = (1 - a) * np.asarray(h0[key]['kp']) + a * np.asarray(h1[key]['kp'])
            elif key in h0 and a < 0.5: kp = np.asarray(h0[key]['kp'])
            elif key in h1 and a >= 0.5: kp = np.asarray(h1[key]['kp'])
            else: continue
            out.append(kp)
        return out


# ---------------------------------------------------------------- drawing helpers
def rounded(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, r, fill=fill, outline=outline, width=width)


def overlay_masks(img, case, t, alpha=0.32, highlight=None, only=None, fade=1.0):
    """img: HxWx3 uint8 RGB (video panel). Fill + contour for each labelled instance."""
    out = img.astype(np.float32)
    ks = [k for k in case.order if only is None or k in only]
    for k in ks:
        m, _ = case.mask(t, k)
        if not m.any(): continue
        c = np.array(case.color[k], np.float32)
        a = alpha * fade * (1.6 if highlight is not None and k == highlight else 1.0) * (1.0 if case.kind(k) == 'object' else 0.55)
        out[m] = out[m] * (1 - a) + c * a
    out = out.astype(np.uint8)
    for k in ks:
        _, cs = case.mask(t, k)
        if not cs: continue
        col = tuple(int(v) for v in case.color[k])
        if fade < 1:
            col = tuple(int(v * fade + 255 * (1 - fade) * 0) for v in col)
        th = 4 if highlight is not None and k == highlight else 2
        if th > 2: cv2.drawContours(out, cs, -1, (255, 255, 255), th + 3, cv2.LINE_AA)
        cv2.drawContours(out, cs, -1, col, th, cv2.LINE_AA)
    return out


def draw_hands(img, kps, ray=True):
    for kp in kps:
        p = kp.astype(int)
        for a, b in HAND_EDGES:
            cv2.line(img, tuple(p[a]), tuple(p[b]), (30, 34, 39), 4, cv2.LINE_AA)
        for a, b in HAND_EDGES:
            cv2.line(img, tuple(p[a]), tuple(p[b]), (255, 255, 255), 2, cv2.LINE_AA)
        for q in p:
            cv2.circle(img, tuple(q), 3, (255, 214, 102), -1, cv2.LINE_AA)
        if ray:
            g = hand_geometry(kp)
            if pointing_pose(g) > RAY_POSE:
                tip = g['tip']; dvec = g['dir']
                n = 12
                for i in range(n):
                    s0 = 14 + i * 34; s1 = s0 + 20
                    a0 = tip + dvec * s0; a1 = tip + dvec * s1
                    alpha = 1 - i / n
                    col = (int(255), int(214 * alpha + 255 * (1 - alpha)), int(102 * alpha + 255 * (1 - alpha)))
                    cv2.line(img, tuple(a0.astype(int)), tuple(a1.astype(int)), (30, 34, 39), 6, cv2.LINE_AA)
                    cv2.line(img, tuple(a0.astype(int)), tuple(a1.astype(int)), col, 3, cv2.LINE_AA)
    return img


def hand_geometry(kp):  # identical to iae.evidence.hand_geometry
    kp = np.asarray(kp, float)
    wrist, mcp, pip, tip = kp[0], kp[5], kp[6], kp[8]
    a, b = pip - mcp, tip - pip
    straight = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
    palm = np.linalg.norm(kp[9] - wrist) + 1e-6
    ext = float(np.linalg.norm(tip - mcp) / palm)
    d = tip - mcp; d = d / (np.linalg.norm(d) + 1e-6)
    return {'tip': tip, 'dir': d, 'straight': straight, 'ext': ext}


def pointing_pose(g):  # identical to iae.evidence.pointing_pose
    s1 = 1 / (1 + math.exp(-12 * (g['straight'] - 0.8)))
    s2 = 1 / (1 + math.exp(-6 * (g['ext'] - 1.0)))
    return s1 * s2


def clock(d, text):
    f = font(17, 'semibold'); tw = d.textlength(text, font=f)
    x1, y0 = VX + VW - 16, VY + 16
    rounded(d, (x1 - tw - 24, y0, x1, y0 + 32), 16, fill=(30, 34, 39))
    d.text((x1 - 12, y0 + 5), text, font=f, fill=(255, 255, 255), anchor='ra')


def id_tag(d, xy, text, color, size=17):
    f = font(size, 'monob'); x, y = xy
    tw = d.textlength(text, font=f); h = size + 10
    x0 = int(x - tw / 2 - 8); y0 = int(y - h)
    rounded(d, (x0, y0, x0 + tw + 16, y0 + h), 6, fill=color)
    d.text((x0 + 8, y0 + 3), text, font=f, fill=(255, 255, 255))


def draw_tags(d, case, t, only=None, alpha=1.0):
    for k in case.order:
        if only is not None and k not in only: continue
        b = case.box(t, k)
        if b is None: continue
        x = VX + (b[0] + b[2]) / 2; y = VY + max(b[1] - 6, 30)
        col = case.color[k] if alpha >= 1 else tuple(int(c * alpha + 255 * (1 - alpha)) for c in case.color[k])
        id_tag(d, (x, y), case.labels[k], col)


def arrow(img_pil, p0, p1, color, width=6, progress=1.0, bend=-0.28):
    """Quadratic Bezier arrow on the full canvas (coordinates in canvas space)."""
    d = ImageDraw.Draw(img_pil)
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    mid = (p0 + p1) / 2; v = p1 - p0; nrm = np.array([v[1], -v[0]])
    if nrm[1] > 0: nrm = -nrm
    ctrl = mid + nrm * abs(bend)
    ts = np.linspace(0, progress, 60)
    pts = [tuple((1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1) for t in ts]
    if len(pts) < 2: return
    d.line(pts, fill=(255, 255, 255), width=width + 6, joint='curve')
    d.line(pts, fill=color, width=width, joint='curve')
    d.ellipse((p0[0] - 7, p0[1] - 7, p0[0] + 7, p0[1] + 7), fill=color, outline=(255, 255, 255), width=3)
    if progress >= 0.999:
        e = np.asarray(pts[-1]); s = np.asarray(pts[-6]); u = (e - s) / (np.linalg.norm(e - s) + 1e-6); w = np.array([-u[1], u[0]])
        L = 22; tri = [tuple(e + u * 4), tuple(e - u * L + w * L * 0.55), tuple(e - u * L - w * L * 0.55)]
        tri_o = [tuple(e + u * 8), tuple(e - u * (L + 4) + w * (L + 4) * 0.6), tuple(e - u * (L + 4) - w * (L + 4) * 0.6)]
        d.polygon(tri_o, fill=(255, 255, 255)); d.polygon(tri, fill=color)


# ---------------------------------------------------------------- timeline
class Timeline:
    def __init__(self, case: Case, mode):
        self.case = case; self.mode = mode
        have_segoe = (Path(FONTS) / 'segoeui.ttf').exists()
        if have_segoe:
            for f in ('segoeui.ttf', 'seguisb.ttf'):
                font_manager.fontManager.addfont(FONTS + f)
        plt.rcParams.update({'font.family': 'Segoe UI' if have_segoe else 'DejaVu Sans', 'font.size': 13, 'axes.edgecolor': '#9AA0A6',
                             'axes.labelcolor': '#4A5058', 'xtick.color': '#6A717B', 'ytick.color': '#6A717B'})
        tsec = np.arange(case.T) * float(np.median(np.diff(case.natives))) / case.native_fps
        self.tsec = tsec
        series = []
        if mode == 'evidence':
            y = sig(case.z)
            for j, c in enumerate(case.cands):
                k = c.get('track')
                if k is not None and c['kind'] != 'drawer' and k in case.color:
                    series.append((y[:, j], case.color[k], 2.4, 1.0, 3))
                else:
                    series.append((y[:, j], hexrgb('#B9BEC6'), 1.0, 0.8, 1))
            ylim, ylabel = (0, 1.02), 'frame evidence  \u03c3(z)'
            yt = [0, 0.5, 1]
        else:  # displacement of each object from its anchor position, % of image width
            W = case.tr['size'][0]
            for k in case.order:
                if case.kind(k) != 'object': continue
                a = case.center(case.scene_t, k)
                y = np.array([np.nan if case.tr['per_frame'][t][k] is None else
                              np.linalg.norm(case.center(t, k) - a) / W * 100 for t in range(case.T)])
                series.append((y, case.color[k], 2.4, 1.0, 3))
            mx = np.nanmax([np.nanmax(s[0]) for s in series]) if series else 1
            ylim, ylabel = (0, max(10, mx * 1.12)), 'displacement\n(% image width)'
            yt = None
        self.ylim = ylim
        dpi = 100
        fig = plt.figure(figsize=(TW / dpi, TH / dpi), dpi=dpi); fig.patch.set_alpha(0)
        ax = fig.add_axes([0.075, 0.24, 0.905, 0.68]); ax.set_facecolor('none')
        ax.set_xlim(0, tsec[-1]); ax.set_ylim(*ylim)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        ax.set_xlabel('video time (s)', labelpad=2); ax.set_ylabel(ylabel, labelpad=6, fontsize=12.5)
        if yt: ax.set_yticks(yt)
        ax.grid(axis='y', color='#ECEDEF', lw=0.8); ax.set_axisbelow(True)
        fig.canvas.draw(); self.bg = np.asarray(fig.canvas.buffer_rgba()).copy()
        lines = []
        for y, col, lw, al, zo in sorted(series, key=lambda s: s[4]):
            ln, = ax.plot(tsec, y, color=np.array(col) / 255, lw=lw, alpha=al, zorder=zo, solid_capstyle='round')
            lines.append(ln)
        fig.canvas.draw(); self.fg = np.asarray(fig.canvas.buffer_rgba()).copy()
        self.tr = ax.transData; self.figh = TH
        x0 = self.tr.transform((0, 0))[0]; x1 = self.tr.transform((tsec[-1], 0))[0]
        self.x0, self.x1 = x0, x1
        plt.close(fig)

    def xy(self, u, val):
        t = float(np.interp(u, np.arange(self.case.T), self.tsec))
        x, y = self.tr.transform((t, val)); return TX + x, TY + (self.figh - y)

    def image(self, u_reveal=None):
        """RGBA timeline revealed up to sample u (None = full)."""
        if u_reveal is None: return self.fg
        x = int(self.xy(u_reveal, 0)[0] - TX)
        img = self.bg.copy(); img[:, :x] = self.fg[:, :x]
        return img


# ---------------------------------------------------------------- frame composer
class Composer:
    def __init__(self, case: Case, steps):
        self.case = case; self.steps = steps
        mode = 'displacement' if case.episodic else 'evidence'
        self.tl = Timeline(case, mode)

    def base(self, step_idx):
        img = Image.new('RGB', (W_OUT, H_OUT), BG); d = ImageDraw.Draw(img); m = self.case.m
        d.text((VX, 22), TASK_TITLE.get(m['task'], m['task']), font=font(32, 'semibold'), fill=INK)
        sub = m['instruction'].replace('Start executing the instruction from the final scene of the video:', '').replace(
            'Start executing the instruction from the initial scene of the video:', '').strip()
        sub = sub[0].upper() + sub[1:]
        d.text((VX, 68), f'\u201c{sub}\u201d', font=font(19), fill=MUTED)
        # step pills (right aligned)
        x = W_OUT - 40; f = font(16, 'semibold')
        for i in range(len(self.steps) - 1, -1, -1):
            s = f'{i + 1}  {self.steps[i]}'; tw = d.textlength(s, font=f); x0 = x - tw - 24
            active = i == step_idx; done = step_idx is not None and i < step_idx
            fill = INK if active else (232, 233, 228) if done else (240, 240, 236)
            col = (255, 255, 255) if active else MUTED if done else FAINT
            rounded(d, (x0, 30, x, 62), 16, fill=fill)
            d.text((x0 + 12, 35), s, font=f, fill=col); x = x0 - 8
        # panels
        rounded(d, (RX, RY, RX + RW, RY + RH), 12, fill=PANEL, outline=BORDER)
        rounded(d, (TX - 4, TY - 6, TX + TW + 4, TY + TH + 2), 12, fill=PANEL, outline=BORDER)
        src = 'front' if m['view'] == 'front' else m['view']
        d.text((W_OUT - 40, H_OUT - 26), f'Video: WatchAct benchmark (Li et al., 2026), {src} view, activity {m["activity"]}.  '
               f'Overlays: IAE outputs for this request.', font=font(14), fill=FAINT, anchor='ra')
        return img, d

    def put_video(self, img, frame):
        img.paste(Image.fromarray(frame), (VX, VY))
        d = ImageDraw.Draw(img); rounded(d, (VX - 1, VY - 1, VX + VW, VY + VH), 4, outline=BORDER, width=1)

    def caption(self, d, text, sub=None):
        f = font(21, 'semibold'); tw = d.textlength(text, font=f)
        w = tw + 32; h = 44 if sub is None else 72
        if sub is not None: w = max(w, d.textlength(sub, font=font(17)) + 32)
        x0, y0 = VX + 18, VY + VH - h - 18
        rounded(d, (x0, y0, x0 + w, y0 + h), 10, fill=(255, 255, 255))
        d.text((x0 + 16, y0 + 8), text, font=f, fill=INK)
        if sub is not None: d.text((x0 + 16, y0 + 40), sub, font=font(17), fill=MUTED)

    def timeline(self, img, u=None, alpha=1.0):
        tl = Image.fromarray(self.tl.image(u))
        if alpha < 1:
            a = np.asarray(tl).copy(); a[..., 3] = (a[..., 3] * alpha).astype(np.uint8); tl = Image.fromarray(a)
        img.paste(tl, (TX, TY), tl)

    def cursor(self, d, u):
        x, _ = self.tl.xy(u, 0); y0 = self.tl.xy(u, self.tl.ylim[1])[1]; y1 = self.tl.xy(u, 0)[1]
        d.line([(x, y0), (x, y1)], fill=INK, width=2)

    # right panel: instance list
    def instance_list(self, d, u=None, upto=None, chosen=None, y=None):
        c = self.case; y = RY + 22 if y is None else y
        d.text((RX + 24, y), 'INSTANCES', font=font(15, 'bold'), fill=MUTED)
        d.text((RX + 24 + 100, y + 1), 'public IDs from the final scene' if c.anchor_end == -1 else 'public IDs from the initial scene',
               font=font(15), fill=FAINT)
        y += 34
        for i, k in enumerate(c.order):
            if upto is not None and i >= upto: break
            col = c.color[k]
            d.rounded_rectangle((RX + 24, y + 6, RX + 42, y + 24), 4, fill=col)
            d.text((RX + 54, y + 1), c.labels[k], font=font(19, 'monob'), fill=INK)
            d.text((RX + 54 + d.textlength(c.labels[k], font=font(19, 'monob')) + 10, y + 3), c.kind(k), font=font(16), fill=MUTED)
            if u is not None and not c.episodic and k in c.cand_of_track:
                v = float(sig(c.z[int(round(u)), c.cand_of_track[k]]))
                bx0, bx1 = RX + 330, RX + RW - 70
                d.rounded_rectangle((bx0, y + 9, bx1, y + 21), 6, fill=(238, 239, 235))
                d.rounded_rectangle((bx0, y + 9, bx0 + max(12, (bx1 - bx0) * v), y + 21), 6, fill=col)
                d.text((bx1 + 10, y + 2), f'{v:.2f}', font=font(16, 'mono'), fill=MUTED)
            if chosen and k in chosen:
                d.text((RX + RW - 24, y + 3), chosen[k], font=font(15, 'semibold'), fill=col, anchor='ra')
            y += 40
        return y

    def program_block(self, d, pairs, y, title='DECODED PROGRAM', n_show=None, sub=None):
        c = self.case
        d.text((RX + 24, y), title, font=font(15, 'bold'), fill=MUTED)
        if sub: d.text((RX + 24 + d.textlength(title, font=font(15, 'bold')) + 10, y + 1), sub, font=font(15), fill=FAINT)
        y += 32
        for i, (o, dst) in enumerate(pairs):
            if n_show is not None and i >= n_show: break
            k = c.track_of(o); col = c.color.get(k, INK)
            d.text((RX + 24, y), o, font=font(19, 'monob'), fill=col)
            x = RX + 24 + d.textlength(o, font=font(19, 'monob')) + 10
            d.text((x, y), '\u2192', font=font(19, 'mono'), fill=MUTED)
            d.text((x + 30, y), pretty_dest(dst), font=font(19, 'mono'), fill=INK)
            y += 34
        return y

    def result_block(self, d, prog=1.0):
        m = self.case.m; y = RY + 22
        d.text((RX + 24, y), 'RESULT', font=font(15, 'bold'), fill=MUTED)
        d.text((RX + 24 + 70, y + 1), 'scored by WatchAct symbolic execution', font=font(15), fill=FAINT)
        y += 40
        blocks = [('Ground truth', [tuple(p) for p in m['gold']], None),
                  ('IAE (ours)', [(p['object_id'], p['destination_region']) for p in m['iae']['pairs']], m['iae']),
                  ('Qwen3-VL-32B, direct', [(p['object_id'], p['destination_region']) for p in m['qwen32b']['pairs']], m['qwen32b'])]
        gold = {tuple(p) for p in m['gold']}
        for i, (name, pairs, res) in enumerate(blocks):
            if prog < (i + 1) / len(blocks) * 0.999 and i > 0 and prog < 1: break
            d.text((RX + 24, y), name, font=font(19, 'semibold'), fill=INK)
            if res is not None:
                okk = res['succ']; lab = ('success' if okk else 'failure') + ('' if not okk else (', strict' if res['strict'] else ''))
                col = OK if okk else BAD; f = font(16, 'semibold'); tw = d.textlength(lab, font=f)
                rounded(d, (RX + RW - 24 - tw - 22, y - 1, RX + RW - 24, y + 27), 14, fill=col)
                d.text((RX + RW - 24 - tw - 11, y + 2), lab, font=f, fill=(255, 255, 255))
            y += 34
            if not pairs:
                d.text((RX + 36, y), '(no valid plan)', font=font(18), fill=MUTED); y += 30
            for o, dst in pairs:
                good = (o, dst) in gold
                col = INK if res is None else (OK if good else BAD)
                d.text((RX + 36, y), o, font=font(18, 'mono'), fill=col)
                x = RX + 36 + d.textlength(o, font=font(18, 'mono')) + 8
                d.text((x, y), '\u2192 ' + pretty_dest(dst), font=font(18, 'mono'), fill=col)
                y += 28
            y += 18
        return y

    def stats_block(self, d, alpha=1.0):
        """Reported success rates over all requests of this task, for context (bottom of the right panel)."""
        st = TASK_STATS.get(self.case.task)
        if st is None: return
        name, rows = st
        y = RY + RH - 40 - 52 * len(rows) - 44
        d.line([(RX + 24, y - 16), (RX + RW - 24, y - 16)], fill=BORDER, width=1)
        d.text((RX + 24, y), 'ACROSS ' + name.upper(), font=font(15, 'bold'), fill=MUTED)
        d.text((RX + RW - 24, y + 1), 'plan success (%)', font=font(15), fill=FAINT, anchor='ra'); y += 34
        for i, (lab, sr, strict) in enumerate(rows):
            col = INK if i == 0 else MUTED; bar = hexrgb('#0072B2') if i == 0 else hexrgb('#AEB4BC')
            d.text((RX + 24, y), lab, font=font(17, 'semibold' if i == 0 else 'regular'), fill=col)
            bx0, bx1 = RX + 24, RX + RW - 90
            d.rounded_rectangle((bx0, y + 26, bx1, y + 36), 5, fill=(238, 239, 235))
            d.rounded_rectangle((bx0, y + 26, bx0 + (bx1 - bx0) * sr / 100 * alpha, y + 36), 5, fill=bar)
            txt = f'{sr:.1f}' + (f'  ({strict:.1f} strict)' if strict is not None else '')
            d.text((RX + RW - 24, y + 4), txt, font=font(16, 'mono'), fill=col, anchor='ra')
            y += 52

    def layout_block(self, d, y, upto=None):
        """Public task layout (top view): every object's initial region from the WatchAct task definition."""
        c = self.case; m = c.m
        d.text((RX + 24, y), 'PUBLIC TASK LAYOUT', font=font(15, 'bold'), fill=MUTED)
        d.text((RX + 24 + 176, y + 1), 'top view, from the task definition', font=font(15), fill=FAINT); y += 34
        gx0, gy0, cell = RX + 150, y + 22, 100; fx = RX + 36
        d.rounded_rectangle((gx0, gy0, gx0 + 3 * cell, gy0 + 3 * cell), 10, fill=(246, 241, 233), outline=(214, 204, 188), width=2)
        for i in (1, 2):
            d.line([(gx0 + i * cell, gy0 + 6), (gx0 + i * cell, gy0 + 3 * cell - 6)], fill=(228, 219, 204), width=1)
            d.line([(gx0 + 6, gy0 + i * cell), (gx0 + 3 * cell - 6, gy0 + i * cell)], fill=(228, 219, 204), width=1)
        d.text((gx0 + 1.5 * cell, gy0 - 4), 'back', font=font(13), fill=FAINT, anchor='md')
        d.text((gx0 + 1.5 * cell, gy0 + 3 * cell + 4), 'front', font=font(13), fill=FAINT, anchor='ma')
        d.text((fx + 40, gy0 - 4), 'fixtures', font=font(13), fill=FAINT, anchor='md')
        on = {r[1]: r[2] for r in m['initial_states'] if len(r) == 3}
        shown = set(c.labels[k] for i, k in enumerate(c.order) if upto is None or i < upto)
        slots = {}
        for o, reg in on.items():
            k = c.track_of(o)
            if reg.startswith('main_'):
                r, cc = reg[5:-7].split('_'); px = gx0 + (['left', 'center', 'right'].index(cc) + 0.5) * cell
                py = gy0 + (['back', 'middle', 'front'].index(r) + 0.5) * cell
            elif reg.startswith('fixture_'):
                px = fx + 40; py = gy0 + (0.85 if 'back' in reg else 2.15) * cell
            else:
                continue
            n = slots.get((px, py), 0); slots[(px, py)] = n + 1; py += n * 26
            col = c.color.get(k, FAINT) if o in shown else (222, 223, 219)
            d.ellipse((px - 9, py - 9, px + 9, py + 9), fill=col)
            above = reg.startswith('main_') and reg.endswith('_center_region')  # alternate labels within a row
            d.text((px, py - 12) if above else (px, py + 12), o, font=font(12, 'mono'), fill=INK if o in shown else FAINT,
                   anchor='md' if above else 'ma')
        return gy0 + 3 * cell + 30

    def legend_block(self, d, items):
        y = RY + RH - 30 - 34 * len(items) - 34
        d.line([(RX + 24, y - 16), (RX + RW - 24, y - 16)], fill=BORDER, width=1)
        d.text((RX + 24, y), 'HOW TO READ', font=font(15, 'bold'), fill=MUTED); y += 34
        for kind, text in items:
            x = RX + 30; cy = y + 11
            if kind == 'mask':
                d.rounded_rectangle((x, cy - 9, x + 30, cy + 9), 5, fill=(204, 227, 240), outline=hexrgb('#0072B2'), width=2)
            elif kind == 'hand':
                pts = [(x, cy + 8), (x + 10, cy - 2), (x + 20, cy - 6), (x + 30, cy - 9)]
                d.line(pts, fill=(30, 34, 39), width=5); d.line(pts, fill=(255, 255, 255), width=2)
                for q in pts: d.ellipse((q[0] - 3, q[1] - 3, q[0] + 3, q[1] + 3), fill=(255, 214, 102))
            elif kind == 'ray':
                for i in range(3):
                    d.line([(x + i * 11, cy), (x + i * 11 + 7, cy)], fill=(255, 214, 102), width=4)
            elif kind == 'curve':
                d.line([(x, cy + 6), (x + 10, cy - 4), (x + 20, cy + 2), (x + 30, cy - 8)], fill=hexrgb('#0072B2'), width=3)
            elif kind == 'trail':
                d.line([(x, cy + 6), (x + 15, cy), (x + 30, cy - 6)], fill=hexrgb('#D55E00'), width=3)
            d.text((x + 44, y), text, font=font(16), fill=INK)
            y += 34


# ---------------------------------------------------------------- storyboards
def segments(case: Case):
    """List of (step_index, n_frames, fn(i, n) -> PIL.Image)."""
    c = case; T = c.T; nat_last = len(c.frames) - 1
    epis = c.episodic
    steps = (['Register', 'Track', 'Compare ends', 'Program', 'Result'] if epis else
             ['Register', 'Track back', 'Evidence', 'Program', 'Result'])
    comp = Composer(c, steps)
    scene_word = 'final' if c.anchor_end == -1 else 'initial'
    prog_word = {'Nonverbal_Cue': 'Decode an object–destination program', 'Reference_Disambiguation': 'Resolve the reference with the evidence'}.get(
        c.task, 'Read the program off the tracks (no learning)')
    long_steps = ([f'Register the {scene_word} scene to public IDs', 'Track every instance through the video',
                   'Compare where each instance starts and ends', prog_word, 'Score the plan with WatchAct'] if epis else
                  [f'Register the {scene_word} scene to public IDs', 'Track identities back through the video',
                   'Score hand–instance evidence per frame', prog_word, 'Score the plan with WatchAct'])
    st = c.scene_t; scene_native = 0 if c.anchor_end == 0 else nat_last
    m = c.m
    if epis:
        exec_pairs = [(p['object_id'], p['destination_region']) for p in m['iae']['pairs']]
    else:
        exec_pairs = [(p['object_id'], p['destination_region']) for p in m['iae']['pairs']]
    segs = []

    def frame_at_native(n):
        return c.frames[int(np.clip(round(n), 0, nat_last))]

    # 0. title card
    sc = 0.80; tw_, th_ = int(VW * sc), int(VH * sc)
    poster = Image.fromarray(frame_at_native(scene_native)).resize((tw_, th_), Image.LANCZOS)
    poster_mask = Image.new('L', (tw_, th_), 0); ImageDraw.Draw(poster_mask).rounded_rectangle((0, 0, tw_, th_), 14, fill=255)

    def title(i, n):
        img = Image.new('RGB', (W_OUT, H_OUT), BG); d = ImageDraw.Draw(img)
        a = ease(i / 18)
        col = lambda rgb: tuple(int(v * a + b * (1 - a)) for v, b in zip(rgb, BG))
        px, py = 70, (H_OUT - th_) // 2
        shadow = Image.new('RGBA', (tw_ + 40, th_ + 40), (0, 0, 0, 0))
        ImageDraw.Draw(shadow).rounded_rectangle((20, 26, tw_ + 20, th_ + 26), 16, fill=(0, 0, 0, int(28 * a)))
        img.paste(shadow, (px - 20, py - 20), shadow)
        pv = Image.blend(Image.new('RGB', poster.size, BG), poster, a)
        img.paste(pv, (px, py), poster_mask)
        x0 = px + tw_ + 70; wcol = W_OUT - 70 - x0; y = 250
        d.text((x0, y), 'IAE  \u00b7  instance-anchored interaction evidence', font=font(20, 'semibold'), fill=col(hexrgb('#0072B2'))); y += 46
        for line in wrap(d, TASK_TITLE.get(m['task'], m['task']), font(40, 'semibold'), wcol):
            d.text((x0, y), line, font=font(40, 'semibold'), fill=col(INK)); y += 52
        y += 18
        sub = m['instruction'].split(':', 1)[1].strip() if ':' in m['instruction'] else m['instruction']
        for line in wrap(d, f'\u201c{sub[0].upper() + sub[1:]}\u201d', font(22), wcol):
            d.text((x0, y), line, font=font(22), fill=col(INK)); y += 32
        scene = 'final' if c.anchor_end == -1 else 'initial'
        y += 8; d.text((x0, y), f'The robot acts from the {scene} scene of the video (left).', font=font(19), fill=col(MUTED)); y += 60
        for j, s_ in enumerate(long_steps):
            d.ellipse((x0, y, x0 + 30, y + 30), fill=col(INK)); d.text((x0 + 15, y + 15), str(j + 1), font=font(16, 'bold'), fill=col((255, 255, 255)), anchor='mm')
            d.text((x0 + 44, y + 1), s_, font=font(20, 'semibold'), fill=col(INK)); y += 42
        y += 30
        d.text((x0, y), f'WatchAct  \u00b7  {m["activity"]}  \u00b7  {m["view"]} view', font=font(16), fill=col(FAINT))
        d.text((x0, y + 24), f'request {m["uid"]}', font=font(16), fill=col(FAINT))
        d.text((W_OUT - 40, H_OUT - 26), 'Video: WatchAct benchmark (Li et al., 2026).  Overlays: IAE outputs for this request.',
               font=font(14), fill=col(FAINT), anchor='ra')
        return img
    segs.append((None, 105, title))

    # 1. register the anchor scene: masks and tags appear one by one
    def register(i, n):
        img, d = comp.base(0); fr = frame_at_native(scene_native)
        k_n = len(c.order); per = n * 0.7 / max(k_n, 1)
        shown = [k for j, k in enumerate(c.order) if i >= j * per]
        vid = overlay_masks(fr, c, st, only=shown)
        comp.put_video(img, vid); d = ImageDraw.Draw(img)
        draw_tags(d, c, st, only=shown)
        y = comp.instance_list(d, upto=len(shown))
        comp.layout_block(d, y + 28, upto=len(shown))
        scene = 'final' if c.anchor_end == -1 else 'initial'
        comp.caption(d, f'Register the {scene} scene', 'Detected instances are matched to the public object IDs of the task layout.')
        comp.timeline(img, u=0, alpha=0.35)
        return img
    segs.append((0, 100, register))

    # 2. propagate identities (backwards for final-scene tasks)
    speed = 6.0
    if c.anchor_end == -1:
        n2 = int(nat_last / speed)

        def trackback(i, n):
            img, d = comp.base(1)
            nn = nat_last * (1 - i / max(n - 1, 1)); u = c.u_of_native(nn); t = int(round(u))
            vid = overlay_masks(frame_at_native(nn), c, t)
            comp.put_video(img, vid); d = ImageDraw.Draw(img); draw_tags(d, c, t)
            comp.instance_list(d)
            comp.legend_block(d, [('mask', 'SAM 2.1 mask, coloured by public ID')])
            clock(d, f'{nn / c.native_fps:4.1f} s  \u00b7  reverse 6\u00d7')
            comp.caption(d, 'Propagate identities backwards  (SAM 2.1, played at 6\u00d7 in reverse)',
                         'Every instance keeps its final-scene ID, even where it was moved or occluded earlier.')
            comp.timeline(img, u=0, alpha=0.35)
            return img
        segs.append((1, n2, trackback))
        play_speed = 1.25
        n3 = int(nat_last / play_speed)

        def evidence(i, n):
            img, d = comp.base(2)
            nn = nat_last * i / max(n - 1, 1); u = c.u_of_native(nn); t = int(round(u))
            hl = None
            row = sig(c.z[t])
            best = max(((row[j], k) for k, j in c.cand_of_track.items()), default=(0, None))
            if best[0] > 0.5: hl = best[1]
            vid = overlay_masks(frame_at_native(nn), c, t, alpha=0.26, highlight=hl)
            vid = draw_hands(vid.copy(), c.hands_at(u))
            comp.put_video(img, vid); d = ImageDraw.Draw(img); draw_tags(d, c, t)
            comp.instance_list(d, u=u)
            comp.legend_block(d, [('mask', 'tracked instance; bold outline = highest evidence'), ('hand', 'hand keypoints (MediaPipe)'),
                                  ('ray', 'fingertip ray, shown in pointing poses'), ('curve', 'per-frame evidence of each instance')])
            comp.caption(d, 'Score hand\u2013instance interaction evidence in every frame',
                         'Hand keypoints and the fingertip ray are related to each instance; the network scores every frame.')
            comp.timeline(img, u=u); d = ImageDraw.Draw(img); comp.cursor(d, u)
            clock(d, f'{nn / c.native_fps:4.1f} s  \u00b7  {play_speed:g}\u00d7')
            return img
        segs.append((2, n3, evidence))
    else:
        play_speed = 1.5
        n3 = int(nat_last / play_speed)

        def trackfwd(i, n):
            img, d = comp.base(1)
            nn = nat_last * i / max(n - 1, 1); u = c.u_of_native(nn); t = int(round(u))
            vid = overlay_masks(frame_at_native(nn), c, t, alpha=0.28)
            vid = vid.copy()
            for k in c.order:  # centroid trails of objects
                if c.kind(k) != 'object': continue
                pts = [c.center(tt, k) for tt in range(0, t + 1) if c.tr['per_frame'][tt][k] is not None]
                if len(pts) > 1:
                    cv2.polylines(vid, [np.array(pts, np.int32)], False, (255, 255, 255), 6, cv2.LINE_AA)
                    cv2.polylines(vid, [np.array(pts, np.int32)], False, c.color[k], 3, cv2.LINE_AA)
            vid = draw_hands(vid, c.hands_at(u), ray=False)
            comp.put_video(img, vid); d = ImageDraw.Draw(img); draw_tags(d, c, t)
            comp.instance_list(d)
            clock(d, f'{nn / c.native_fps:4.1f} s  \u00b7  {play_speed:g}\u00d7')
            comp.legend_block(d, [('mask', 'SAM 2.1 mask, coloured by public ID'), ('trail', 'path of the object centre'),
                                  ('hand', 'hand keypoints (MediaPipe)')])
            comp.caption(d, f'Track every instance through the video  (SAM 2.1, {play_speed:g}\u00d7)',
                         'Identities are propagated forward from the initial scene; lines show each object\u2019s path.')
            comp.timeline(img, u=u); d = ImageDraw.Draw(img); comp.cursor(d, u)
            return img
        segs.append((1, n3, trackfwd))

    # 3/4. program
    if not epis and c.task == 'Nonverbal_Cue':
        ev = m['iae']['events']; n4 = 150

        def decode(i, n):
            img, d = comp.base(3)
            fr = frame_at_native(nat_last); vid = overlay_masks(fr, c, st, alpha=0.22)
            comp.put_video(img, vid); d = ImageDraw.Draw(img); draw_tags(d, c, st)
            comp.timeline(img); d = ImageDraw.Draw(img)
            k_ev = int(min(len(ev), (i / (n * 0.55)) * len(ev) + 1))
            for j, (t, cid, role, v) in enumerate(ev[:k_ev]):
                x, y = comp.tl.xy(t, float(sig(v)))
                tk = c.track_of(cid); col = c.color.get(tk, INK)
                d.ellipse((x - 9, y - 9, x + 9, y + 9), fill=col, outline=(255, 255, 255), width=3)
                lab = ('object ' if role == 'o' else 'destination ') + cid
                d.text((x, max(y - 16, TY + 26)), lab, font=font(14, 'semibold'), fill=INK, anchor='md', stroke_width=3, stroke_fill=(255, 255, 255))
            pairs = exec_pairs; npairs = len(pairs)
            pa = max(0.0, (i - n * 0.45) / (n * 0.45))
            for j, (o, dst) in enumerate(pairs):
                pr = ease(pa * npairs - j)
                if pr <= 0: continue
                k = c.track_of(o); p0 = c.center(st, k); p1 = c.dest_point(dst, st)
                if p0 is None or p1 is None: continue
                arrow(img, (VX + p0[0], VY + p0[1]), (VX + p1[0], VY + p1[1]), c.color[k], progress=pr)
            d = ImageDraw.Draw(img); draw_tags(d, c, st)
            chosen = {c.track_of(cid): ('object' if role == 'o' else 'destination') for t, cid, role, v in ev[:k_ev] if c.track_of(cid) is not None}
            y = comp.instance_list(d, chosen=chosen)
            comp.program_block(d, pairs, y + 26, n_show=int(math.ceil(pa * npairs)) if pa > 0 else 0,
                               sub='object \u2192 destination')
            comp.caption(d, 'Decode an object\u2013destination program',
                         'A grammar-constrained dynamic program selects alternating object and destination events.')
            return img
        segs.append((3, n4, decode))
    elif not epis:  # reference disambiguation
        sel = m['iae']['selected']; dest = m['iae']['dest']; n4 = 170

        def rdprog(i, n):
            img, d = comp.base(3)
            fr = frame_at_native(nat_last); vid = overlay_masks(fr, c, st, alpha=0.22)
            comp.put_video(img, vid); d = ImageDraw.Draw(img); draw_tags(d, c, st)
            comp.timeline(img); d = ImageDraw.Draw(img)
            y = RY + 22
            d.text((RX + 24, y), 'VIDEO-LEVEL EVIDENCE', font=font(15, 'bold'), fill=MUTED)
            d.text((RX + 24 + 196, y + 1), 'pooled over the top frames', font=font(15), fill=FAINT); y += 36
            movs = [k for k in c.order if c.kind(k) == 'object']
            a = ease(i / (n * 0.3))
            for k in sorted(movs, key=lambda k: -c.s[c.cand_of_track[k]]):
                v = float(sig(c.s[c.cand_of_track[k]])); col = c.color[k]
                d.text((RX + 24, y), c.labels[k], font=font(19, 'monob'), fill=INK)
                bx0, bx1 = RX + 210, RX + RW - 80
                d.rounded_rectangle((bx0, y + 8, bx1, y + 20), 6, fill=(238, 239, 235))
                d.rounded_rectangle((bx0, y + 8, bx0 + max(12, (bx1 - bx0) * v * a), y + 20), 6, fill=col)
                d.text((bx1 + 10, y + 1), f'{v:.2f}', font=font(16, 'mono'), fill=MUTED)
                y += 38
            y += 14
            steps_txt = []
            text = m['instruction'].lower()
            tg = [p['object_id'] for p in m['iae']['pairs']]
            if 'other' in text:
                steps_txt.append(('\u201cthe other one\u201d: least evidence among the identical objects', ', '.join(tg)))
            else:
                steps_txt.append(('Objects with positive evidence', ', '.join(tg)))
            if 'closer' in text or 'farther' in text:
                w_ = 'closer to' if 'closer' in text else 'farther from'
                steps_txt.append((f'\u201c{w_} the person\u201d: the basket {w_} the person\u2019s position', dest))
            else:
                steps_txt.append(('Destination named in the instruction', dest))
            d.text((RX + 24, y), 'REFERENCE PROGRAM', font=font(15, 'bold'), fill=MUTED); y += 32
            k_s = int(min(len(steps_txt), max(0, (i - n * 0.25) / (n * 0.2)) + (1 if i > n * 0.25 else 0)))
            for lab, val in steps_txt[:k_s]:
                d.text((RX + 24, y), lab, font=font(17, 'semibold'), fill=MUTED); y += 26
                d.text((RX + 36, y), val, font=font(18, 'monob'), fill=INK); y += 34
            pa = max(0.0, (i - n * 0.7) / (n * 0.25))
            for j, p in enumerate(m['iae']['pairs']):
                pr = ease(pa * len(m['iae']['pairs']) - j)
                if pr <= 0: continue
                k = c.track_of(p['object_id']); p0 = c.center(st, k); p1 = c.dest_point(p['destination_region'], st)
                arrow(img, (VX + p0[0], VY + p0[1]), (VX + p1[0], VY + p1[1]), c.color[k], progress=pr)
            d = ImageDraw.Draw(img); draw_tags(d, c, st)
            comp.caption(d, 'Resolve the reference with the evidence',
                         'Instances the person handled carry high evidence; the instruction selects among them.')
            return img
        segs.append((3, n4, rdprog))
    else:  # episodic
        moved = m['iae']['moved']; n4 = 120; n5 = 140
        other_t = c.T - 1 if c.anchor_end == 0 else 0

        def ends(i, n):
            img, d = comp.base(2)
            fr = frame_at_native(nat_last if c.anchor_end == 0 else 0)
            vid = overlay_masks(fr, c, other_t, alpha=0.26)
            comp.put_video(img, vid); d = ImageDraw.Draw(img); draw_tags(d, c, other_t)
            comp.timeline(img); d = ImageDraw.Draw(img)
            y = comp.instance_list(d)
            y += 26
            d.text((RX + 24, y), 'MOVED INSTANCES', font=font(15, 'bold'), fill=MUTED)
            d.text((RX + 24 + 150, y + 1), 'start region \u2192 end region', font=font(15), fill=FAINT); y += 32
            kk = int(min(len(moved), i / (n * 0.6) * len(moved) + 1))
            for o, pa_, po, dist in moved[:kk]:
                k = c.track_of(o)
                d.text((RX + 24, y), o, font=font(18, 'monob'), fill=c.color[k]); y += 26
                d.text((RX + 36, y), f'{pretty_dest(pa_)}  \u2192  {pretty_dest(po)}', font=font(17, 'mono'), fill=INK); y += 32
            comp.caption(d, 'Compare where each instance starts and ends',
                         'Positions are mapped to public table regions or containers through the registration.')
            return img
        segs.append((2, n4, ends))

        def eprog(i, n):
            img, d = comp.base(3)
            fr = frame_at_native(scene_native); vid = overlay_masks(fr, c, st, alpha=0.22)
            comp.put_video(img, vid); d = ImageDraw.Draw(img); draw_tags(d, c, st)
            comp.timeline(img)
            pa = i / (n * 0.7)
            for j, (o, dst) in enumerate(exec_pairs):
                pr = ease(pa * len(exec_pairs) - j)
                if pr <= 0: continue
                k = c.track_of(o); p0 = c.center(st, k); p1 = c.dest_point(dst, st)
                if p0 is None or p1 is None: continue
                arrow(img, (VX + p0[0], VY + p0[1]), (VX + p1[0], VY + p1[1]), c.color[k], progress=pr)
            d = ImageDraw.Draw(img); draw_tags(d, c, st)
            y = comp.instance_list(d)
            comp.program_block(d, exec_pairs, y + 26, n_show=int(math.ceil(pa * len(exec_pairs))), title='PROGRAM',
                               sub='executed from the initial scene')
            comp.caption(d, 'Imitation program: repeat every move from the initial scene',
                         'No learning is involved: the program is read off the identity-preserving tracks.')
            return img
        segs.append((3, n5, eprog))

    # 5. result
    def result(i, n):
        img, d = comp.base(4)
        fr = frame_at_native(scene_native); vid = overlay_masks(fr, c, st, alpha=0.22)
        comp.put_video(img, vid)
        for o, dst in exec_pairs:
            k = c.track_of(o); p0 = c.center(st, k); p1 = c.dest_point(dst, st)
            if p0 is None or p1 is None: continue
            arrow(img, (VX + p0[0], VY + p0[1]), (VX + p1[0], VY + p1[1]), c.color[k])
        d = ImageDraw.Draw(img); draw_tags(d, c, st)
        comp.timeline(img, alpha=0.5); d = ImageDraw.Draw(img)
        comp.result_block(d, prog=min(1.0, i / (n * 0.5)))
        if i > n * 0.45: comp.stats_block(d, alpha=ease((i - n * 0.45) / (n * 0.25)))
        res = m['iae']
        if res['succ']:
            comp.caption(d, 'The plan is executed and scored by WatchAct', 'Arrows show the IAE program on the scene where it is executed.')
        else:
            comp.caption(d, 'Failure case: the plan is scored as a failure', FAILURE_NOTE.get(m['uid'], 'Arrows show the IAE program on the scene where it is executed.'))
        return img
    segs.append((4, 170, result))
    return segs


FAILURE_NOTE = {'417af5a3547b3b334bde23d5': 'Both boxes and both containers are found, but the two identical boxes are paired with the wrong containers.'}

# Reported results on all requests of the task (paper, Tables 1 and 4), shown for context in the last step.
TASK_STATS = {
    'Nonverbal_Cue': ('all 195 nonverbal-cue requests', [('IAE (ours)', 42.6, 16.9), ('Qwen3-VL-32B, direct', 22.6, 4.6)]),
    'Reference_Disambiguation': ('all 260 reference requests', [('IAE (ours)', 80.4, 74.2), ('Qwen3-VL-32B, direct', 31.2, 23.5)]),
    'Imitation': ('all 290 imitation requests', [('IAE (ours)', 46.6, None), ('Qwen3-VL-32B, direct', 37.6, None)]),
    'Restore_Previous_State': ('all 285 restore requests', [('IAE (ours)', 46.3, None), ('Qwen3-VL-32B, direct', 16.5, None)]),
    'Reversal': ('all 225 reversal requests', [('IAE (ours)', 46.2, None), ('Qwen3-VL-32B, direct', 26.7, None)]),
}


def wrap(d, text, f, width):
    words = text.split(); lines = []; cur = ''
    for w in words:
        t = (cur + ' ' + w).strip()
        if d.textlength(t, font=f) <= width: cur = t
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines


def render(case_dir, out, stills=None):
    c = Case(Path(case_dir)); c.load_video()
    segs = segments(c)
    total = sum(n for _, n, _ in segs); print('frames', total, f'{total / FPS:.1f} s', flush=True)
    if stills:
        k = 0
        for si, (step, n, fn) in enumerate(segs):
            for i in range(n):
                if k in stills:
                    fn(i, n).save(Path(out).with_name(Path(out).stem + f'_still_{k:04d}.png'))
                k += 1
        return
    ff = shutil.which('ffmpeg')
    p = subprocess.Popen([ff, '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W_OUT}x{H_OUT}',
                          '-r', '30000/1001', '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '19',
                          '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)], stdin=subprocess.PIPE)
    prev = None; xf = 12
    for si, (step, n, fn) in enumerate(segs):
        for i in range(n):
            im = np.asarray(fn(i, n).convert('RGB'))
            if si == 1 and i < xf and prev is not None:  # cross-fade from the title card
                a = (i + 1) / (xf + 1); im = (prev * (1 - a) + im * a).astype(np.uint8)
            p.stdin.write(im.tobytes())
            if si == 0 and i == n - 1: prev = im.astype(np.float32)
        print('segment', si, 'done', flush=True)
    p.stdin.close(); p.wait()
    print('wrote', out)


if __name__ == '__main__':
    args = sys.argv[1:]
    stills = [int(x) for x in args[args.index('--still') + 1:]] if '--still' in args else None
    render(args[0], args[1], stills)
