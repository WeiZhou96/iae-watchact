"""Final-scene-anchored interaction evidence (goal-free).

Builds, per video, a candidate set (tracked instances + table regions projected from the registration map)
and per-frame hand evidence: pointing pose, fingertip ray, per-candidate pointing log-scores, contact and
instance motion. Everything is computed from pixels, public layout and the registration map.
"""
from __future__ import annotations
import json, math
import numpy as np
from .common import OUT
from .register import canonical_xy

MAIN_REGIONS = [f'main_{r}_{c}_region' for r in ('back', 'middle', 'front') for c in ('left', 'center', 'right')]
FIXTURE_REGIONS = ['fixture_front_region', 'fixture_back_region']
CONTAINERS = ('basket', 'wooden_tray')


def _load(kind, vk):
    p = OUT / kind / f'{vk}.json'
    return json.loads(p.read_text()) if p.exists() else None


def region_points(reg, W):
    """Image points of canonical region centres via the inverse registration map."""
    if reg.get('status') != 'ok': return {}
    th = math.radians(reg['angle_deg']); sx, sy, u, v = reg['scale_shift']; c, s = math.cos(th), math.sin(th)
    pts = {}
    for name in MAIN_REGIONS + FIXTURE_REGIONS:
        q = np.array(canonical_xy(name)); rq = np.array([c * q[0] + s * q[1], -s * q[0] + c * q[1]])  # R^T q
        px, py = (rq[0] - u) / sx, (rq[1] - v) / sy
        pts[name] = (px * W, py * W)
    return pts


def hand_geometry(kp):
    kp = np.asarray(kp, float)
    wrist, mcp, pip, tip = kp[0], kp[5], kp[6], kp[8]
    a, b = pip - mcp, tip - pip
    straight = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
    palm = np.linalg.norm(kp[9] - wrist) + 1e-6
    curl = float(np.mean([np.linalg.norm(kp[j] - wrist) / palm for j in (12, 16, 20)]))
    ext = float(np.linalg.norm(tip - mcp) / palm)
    d = tip - mcp; d = d / (np.linalg.norm(d) + 1e-6)
    return {'tip': tip, 'dir': d, 'straight': straight, 'curl': curl, 'ext': ext, 'palm': float(palm)}


def pointing_pose(g):
    """Soft pointing-pose score in [0,1]: straight, extended index finger with other fingers flexed."""
    s1 = 1 / (1 + math.exp(-12 * (g['straight'] - 0.8)))
    s2 = 1 / (1 + math.exp(-6 * (g['ext'] - 1.0)))
    return s1 * s2


def build(vk, sigma_ang=22.0, sigma_dist=0.35, region_prior=-1.5):
    tr, hands, person, reg = _load('tracks', vk), _load('hands', vk), _load('person', vk), _load('register', vk)
    if tr is None or hands is None: return None
    W, H = tr['size']; T = len(tr['per_frame'])
    labels = tr['labels']; cats = tr['cats']
    cands = []
    for k, (lab, cat) in enumerate(zip(labels, cats)):
        if lab is None: continue
        kind = 'container' if cat in CONTAINERS else 'fixture' if cat == 'wooden_cabinet' else 'movable'
        cands.append({'id': lab, 'kind': kind, 'track': k})
    rp = region_points(reg, W)
    for name, (x, y) in rp.items():
        if -0.2 * W < x < 1.2 * W and -0.2 * H < y < 1.2 * H: cands.append({'id': name, 'kind': 'region', 'point': [x, y]})
    hand_by_t = {r['i']: r['hands'] for r in hands}
    frames = []
    prev_c = None
    for t in range(T):
        pf = tr['per_frame'][t]
        pos = []
        for c in cands:
            if 'track' in c:
                b = pf[c['track']]
                pos.append(None if b is None else ((b['box'][0] + b['box'][2]) / 2, (b['box'][1] + b['box'][3]) / 2, b['box']))
            else:
                pos.append((c['point'][0], c['point'][1], None))
        hs = []
        for h in hand_by_t.get(t, []):
            g = hand_geometry(h['kp']); pp = pointing_pose(g)
            ls, contact = [], []
            for p in pos:
                if p is None: ls.append(-1e9); contact.append(0.0); continue
                v = np.array([p[0], p[1]]) - g['tip']; dist = np.linalg.norm(v) + 1e-6
                ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v, g['dir']) / dist)))))
                ls.append(-(ang / sigma_ang) ** 2 / 2 - dist / (sigma_dist * W) + (region_prior if p[2] is None else 0.0))
                box = p[2]
                if box is not None:
                    m = 0.15 * max(box[2] - box[0], box[3] - box[1])
                    inside = any(box[0] - m <= q[0] <= box[2] + m and box[1] - m <= q[1] <= box[3] + m for q in (np.asarray(h['kp'])[[4, 8, 12]]))
                    contact.append(1.0 if inside else 0.0)
                else: contact.append(0.0)
            hs.append({'pose': pp, 'tip': g['tip'].tolist(), 'logscore': ls, 'contact': contact})
        cen = [None if p is None else (p[0], p[1]) for p in pos]
        frames.append({'t': t, 'hands': hs, 'centers': cen})
    # instance motion relative to final position
    final = frames[-1]['centers']
    for f in frames:
        f['disp'] = [None if (c is None or fc is None) else float(math.hypot(c[0] - fc[0], c[1] - fc[1]) / W)
                     for c, fc in zip(f['centers'], final)]
    per = {r['i']: r for r in person} if person else {}
    ppts = []
    for t in range(T):
        r = per.get(t)
        if r and r['box'] and r['score'] >= 0.35: ppts.append(((r['box'][0] + r['box'][2]) / 2, r['box'][3]))
    return {'video_key': vk, 'W': W, 'H': H, 'T': T, 'cands': cands, 'frames': frames, 'person_points': ppts, 'register': reg}
