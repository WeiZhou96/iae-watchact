"""Tensorised hand-candidate interaction features for the evidence network (goal-free).

Candidates: registered instances (movable / container), cabinet drawers (cabinet box split into thirds),
and table regions projected through the registration map. For every frame, each detected hand and each
candidate gets a geometric feature vector; the network pools over hands and time.
"""
from __future__ import annotations
import json, math
import numpy as np
from .common import OUT, stem
from .evidence import region_points, hand_geometry

KINDS = ['movable', 'container', 'drawer', 'region']
CONTAINERS = ('basket', 'wooden_tray')
NF = 28  # 22 hand/candidate features + 6 arm/head ray features
MAX_HANDS = 2


def _load(kind, vk):
    p = OUT / kind / f'{vk}.json'
    return json.loads(p.read_text()) if p.exists() else None


def candidates(tr, reg, W, H):
    cands = []
    for k, (lab, cat) in enumerate(zip(tr['labels'], tr['cats'])):
        if lab is None: continue
        if cat == 'wooden_cabinet':
            for j, lv in enumerate(('top', 'middle', 'bottom')):
                cands.append({'id': f'{lab}_{lv}_region', 'kind': 'drawer', 'track': k, 'third': j})
        else:
            cands.append({'id': lab, 'kind': 'container' if cat in CONTAINERS else 'movable', 'track': k})
    for name, (x, y) in region_points(reg, W).items():
        if -0.1 * W < x < 1.1 * W and -0.1 * H < y < 1.1 * H: cands.append({'id': name, 'kind': 'region', 'point': [x, y]})
    return cands


def _cand_geom(c, pf):
    """(center, box) of a candidate at one frame; None if not visible."""
    if 'point' in c: return np.array(c['point']), None
    b = pf[c['track']]
    if b is None: return None, None
    x0, y0, x1, y1 = b['box']
    if c['kind'] == 'drawer':
        h = (y1 - y0) / 3; y0, y1 = y0 + c['third'] * h, y0 + (c['third'] + 1) * h
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2]), [x0, y0, x1, y1]


def _arm_rays(b, wrist_hand, tip, min_sc=0.3):
    """Elbow->fingertip and head->fingertip directions (COCO-17), matching the hand to the nearer body wrist."""
    out = {'arm': None, 'head': None}
    if not b or not b.get('kp'): return out
    kp = np.asarray(b['kp']); sc = np.asarray(b['sc'])
    side = min((9, 10), key=lambda w: np.linalg.norm(kp[w] - wrist_hand))
    elbow = 7 if side == 9 else 8
    if sc[elbow] >= min_sc and np.linalg.norm(tip - kp[elbow]) > 1:
        d = tip - kp[elbow]; out['arm'] = d / np.linalg.norm(d)
    head = [j for j in (0, 1, 2) if sc[j] >= 0.4]
    if head:
        hp = kp[head].mean(0); d = tip - hp
        if np.linalg.norm(d) > 1: out['head'] = d / np.linalg.norm(d)
    return out


def _ray_feats(arm, tip, cen, W):
    f = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # defaults: max angle, no ray
    v = cen - tip; n = np.linalg.norm(v) + 1e-6
    if arm['arm'] is not None:
        u = arm['arm']; c = float(v @ u) / n
        f[0] = math.degrees(math.acos(max(-1.0, min(1.0, c)))) / 90
        f[1] = min(abs(u[0] * v[1] - u[1] * v[0]) / W * 5, 3.0)  # perpendicular distance to the arm line
        f[2] = max(min(float(v @ u) / W * 3, 3.0), -3.0)          # signed distance along the line beyond the tip
        f[3] = 1.0
    if arm['head'] is not None:
        f[4] = math.degrees(math.acos(max(-1.0, min(1.0, float(v @ arm['head']) / n)))) / 90; f[5] = 1.0
    return f


def build(vk, static=False, tracks='tracks'):
    tr, hands, reg, body = _load(tracks, vk), _load('hands', vk), _load('register', vk), _load('body', vk)
    if tr is None or hands is None or reg is None: return None
    if static:  # ablation: no identity-preserving tracking; every frame uses the final-frame boxes
        tr = dict(tr); tr['per_frame'] = [tr['per_frame'][-1]] * len(tr['per_frame'])
    bb = {r['i']: r for r in body} if body else {}
    W, H = tr['size']; T = len(tr['per_frame'])
    keep = tr.get('keep')  # subset-of-frames tracks: per_frame[t] belongs to original frame keep[t]
    fi = (lambda t: keep[t]) if keep else (lambda t: t)
    cands = candidates(tr, reg, W, H); K = len(cands)
    if K == 0: return None
    X = np.zeros((T, MAX_HANDS, K, NF), np.float32); M = np.zeros((T, MAX_HANDS), bool)
    hb = {r['i']: r['hands'] for r in hands}
    final = [_cand_geom(c, tr['per_frame'][-1])[0] for c in cands]
    prev_tip = {}
    geo_prev = [None] * K
    for t in range(T):
        pf = tr['per_frame'][t]
        geo = [_cand_geom(c, pf) for c in cands]
        disp = np.array([0.0 if g[0] is None or f is None else np.linalg.norm(g[0] - f) / W for g, f in zip(geo, final)])
        vel = np.array([0.0 if g[0] is None or p is None or p[0] is None else np.linalg.norm(g[0] - p[0]) / W for g, p in zip(geo, geo_prev)])
        geo_prev = geo
        hs = sorted(hb.get(fi(t), []), key=lambda h: -h['score'])[:MAX_HANDS]
        slots = {}
        for h in hs:  # stable slots by handedness so temporal models see the same hand across frames
            j = 0 if h['handed'] == 'Left' else 1
            if j in slots: j = 1 - j
            slots[j] = h
        for j, h in sorted(slots.items()):
            g = hand_geometry(h['kp']); tip = g['tip']
            key = h['handed']; sp = 0.0 if key not in prev_tip else np.linalg.norm(tip - prev_tip[key]) / W
            prev_tip[key] = tip
            angs = np.full(K, 180.0); dists = np.full(K, 2.0)
            for k, (cen, box) in enumerate(geo):
                if cen is None: continue
                v = cen - tip; d = np.linalg.norm(v) + 1e-6
                angs[k] = math.degrees(math.acos(max(-1.0, min(1.0, float(v @ g['dir']) / d)))); dists[k] = d / W
            ar = np.argsort(np.argsort(angs)); dr = np.argsort(np.argsort(dists))
            cone = np.where(angs < 25, dists, 9.0); first = np.argmin(cone) if cone.min() < 9 else -1
            kp = np.asarray(h['kp'])
            arm = _arm_rays(bb.get(fi(t)), kp[0], tip)
            for k, (cen, box) in enumerate(geo):
                if cen is None: continue
                contact = 0.0; rel = 0.0
                if box is not None:
                    m = 0.15 * max(box[2] - box[0], box[3] - box[1]); size = max(box[2] - box[0], box[3] - box[1]) + 1e-6
                    contact = float(any(box[0] - m <= q[0] <= box[2] + m and box[1] - m <= q[1] <= box[3] + m for q in kp[[4, 8, 12]]))
                    rel = dists[k] * W / size
                X[t, j, k] = [g['straight'], g['ext'], g['curl'], h['score'], angs[k] / 90, dists[k], math.log(dists[k] + 1e-3),
                              contact, min(rel, 20) / 10, ar[k] / max(K - 1, 1), dr[k] / max(K - 1, 1), float(k == first), min(sp * 20, 2.0),
                              (cen[1] - tip[1]) / W, *[float(cands[k]['kind'] == x) for x in KINDS], min(disp[k] * 10, 3.0),
                              min(vel[k] * 50, 3.0), t / max(T - 1, 1), 1.0, *_ray_feats(arm, tip, cen, W)]
            M[t, j] = True
    if keep:
        Tf = tr['T_full']; ka = np.asarray(keep)
        near = np.abs(np.arange(Tf)[:, None] - ka[None, :]).argmin(1)
        X, M, T = X[near], M[near], Tf
    return {'video_key': vk, 'X': X, 'M': M, 'cands': cands, 'W': W, 'H': H, 'T': T, 'register': reg, 'keep': keep}


def candidate_labels(cands, gold_pairs, task, instruction, objects):
    """Evaluation-derived training targets (used only inside training folds).

    NC: objects and destinations appearing in gold pairs are positive; every other candidate negative.
    RD: the selected set is recovered from the gold targets through the instruction semantics; only movable
    candidates are supervised (containers/regions masked).
    """
    y = np.zeros(len(cands), np.float32); w = np.ones(len(cands), np.float32)
    ids = [c['id'] for c in cands]
    if task == 'Nonverbal_Cue':
        pos = {o for o, _ in gold_pairs} | {d.replace('_contain_region', '') if d.endswith('_contain_region') else d for _, d in gold_pairs}
        for k, i in enumerate(ids): y[k] = float(i in pos)
    else:
        import re
        targets = {o for o, _ in gold_pairs}
        comp = re.search(r'\b(other|others|remaining|unselected)\b', instruction.lower()) is not None
        cats = {stem(o) for o in targets}
        movable = [o for o in objects if not o.startswith(('basket', 'wooden_tray', 'wooden_cabinet'))]
        same = [o for o in movable if stem(o) in cats]
        domain = same if len(same) > len(targets) else movable
        selected = ({o for o in domain if o not in targets} if comp else targets)
        pointed_dest = {d.replace('_contain_region', '') for _, d in gold_pairs} if 'that container' in instruction.lower() else None
        for k, c in enumerate(cands):
            if c['kind'] == 'movable': y[k] = float(c['id'] in selected)
            elif c['kind'] == 'container' and pointed_dest is not None: y[k] = float(c['id'] in pointed_dest)
            else: w[k] = 0.0
    return y, w
