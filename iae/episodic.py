"""Episodic programs over identity-preserving tracks (no learning, goal-free).

Restore / Reversal (anchor = final scene): every instance whose position at the start of the video differs from
its final position is moved back to its start region. Imitation (anchor = initial scene): every instance whose
final position differs from its initial position is moved to its final region. Positions are mapped to public
regions through the per-video registration map; containment in a tracked container or cabinet drawer takes
precedence over table regions.
"""
from __future__ import annotations
import math
import numpy as np

CONTAINERS = ('basket', 'wooden_tray')


def img_to_canon(reg, W, x, y):
    th = math.radians(reg['angle_deg']); sx, sy, u, v = reg['scale_shift']; c, s = math.cos(th), math.sin(th)
    a = np.array([x / W * sx + u, y / W * sy + v]); return np.array([c * a[0] - s * a[1], s * a[0] + c * a[1]])


def canon_region(q):
    x, y = q
    if x < -0.5: return 'fixture_front_region' if y > 1.0 else 'fixture_back_region'
    col = ['left', 'center', 'right'][int(np.clip(round(x), 0, 2))]; row = ['back', 'middle', 'front'][int(np.clip(round(y), 0, 2))]
    return f'main_{row}_{col}_region'


def _pos(per_frame, j, end, span=15):
    """Box of track j at the given end (0 or -1), taking the nearest visible frame within `span`."""
    rng = range(0, min(span, len(per_frame))) if end == 0 else range(len(per_frame) - 1, max(-1, len(per_frame) - 1 - span), -1)
    for t in rng:
        b = per_frame[t][j]
        if b is not None: return b['box'], t
    return None, None


def _inside(pt, box, m=0.1):
    w, h = box[2] - box[0], box[3] - box[1]
    return box[0] - m * w <= pt[0] <= box[2] + m * w and box[1] - m * h <= pt[1] <= box[3] + m * h


def _hidden_in_container(tr, j, end):
    """If track j is invisible near one end of the video, return the container it emerges from / disappears into:
    the container closest to its first (or last) visible position, when that position lies within 1.5 container sizes."""
    pf = tr['per_frame']; order = range(len(pf)) if end == 0 else range(len(pf) - 1, -1, -1)
    t = next((t for t in order if pf[t][j] is not None), None)
    if t is None: return None, None
    b = pf[t][j]['box']; c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]); best = None
    for k, (lab, cat) in enumerate(zip(tr['labels'], tr['cats'])):
        if lab is None or cat not in CONTAINERS: continue
        cb = pf[t][k] or next((pf[u][k] for u in order if pf[u][k] is not None), None)
        if cb is None: continue
        cb = cb['box']; cc = np.array([(cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2]); size = max(cb[2] - cb[0], cb[3] - cb[1])
        d = np.linalg.norm(c - cc) / size
        if d < 1.5 and (best is None or d < best[0]): best = (d, lab)
    return (best[1] + '_contain_region', b) if best else (None, None)


def place_of(tr, reg, j, end):
    """Public region of track j at one end of the video."""
    pf = tr['per_frame']; W = tr['size'][0]
    box, t = _pos(pf, j, end)
    if box is None: return _hidden_in_container(tr, j, end)
    foot = ((box[0] + box[2]) / 2, box[3]); cen = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    for k, (lab, cat) in enumerate(zip(tr['labels'], tr['cats'])):
        if lab is None or k == j: continue
        cb = pf[t][k]
        if cb is None: continue
        if cat in CONTAINERS and _inside(cen, cb['box'], 0.05): return lab + '_contain_region', box
        if cat == 'wooden_cabinet' and _inside(cen, cb['box'], 0.0):
            y0, y1 = cb['box'][1], cb['box'][3]; lv = ['top', 'middle', 'bottom'][int(np.clip((cen[1] - y0) / (y1 - y0) * 3, 0, 2))]
            return f'{lab}_{lv}_region', box
    return canon_region(img_to_canon(reg, W, *foot)), box


def place_of_canon(tr, reg, j, end, containers, radius=0.6):
    """v2: containment decided on the table plane. `containers` maps public container ID -> canonical (x, y) from the
    public scene. The object's foot point (or, if hidden at that end, its nearest visible foot point) is mapped to the
    table plane; within `radius` of a container centre -> that container, otherwise the table/fixture region."""
    pf = tr['per_frame']; W = tr['size'][0]
    box, t = _pos(pf, j, end)
    hidden = box is None
    if hidden:
        order = range(len(pf)) if end == 0 else range(len(pf) - 1, -1, -1)
        t = next((u for u in order if pf[u][j] is not None), None)
        if t is None: return None, None
        box = pf[t][j]['box']
    q = img_to_canon(reg, W, (box[0] + box[2]) / 2, box[3])
    if containers:
        cid, cq = min(containers.items(), key=lambda kv: np.linalg.norm(q - kv[1]))
        if hidden or np.linalg.norm(q - cq) < radius: return cid + '_contain_region', box
    for k, (lab, cat) in enumerate(zip(tr['labels'], tr['cats'])):
        cb = pf[t][k] if lab is not None and cat == 'wooden_cabinet' else None
        if cb is not None and _inside(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), cb['box'], 0.0):
            y0, y1 = cb['box'][1], cb['box'][3]; lv = ['top', 'middle', 'bottom'][int(np.clip(((box[1] + box[3]) / 2 - y0) / (y1 - y0) * 3, 0, 2))]
            return f'{lab}_{lv}_region', box
    return canon_region(q), box


def program_v2(task, tr, reg, objects, initial_states, min_move=0.5):
    from .register import canonical_xy
    if reg.get('status') != 'ok': return {'pairs': [], 'valid': False, 'moved': []}
    on = {r[1]: r[2] for r in initial_states if len(r) == 3}
    containers = {o: np.array(canonical_xy(on[o])) for o in objects if o.startswith(CONTAINERS) and canonical_xy(on.get(o))}
    W = tr['size'][0]; anchor_end = 0 if tr.get('anchor', len(tr['per_frame']) - 1) == 0 else -1; other = -1 if anchor_end == 0 else 0
    pairs, moved = [], []
    for j, (lab, cat) in enumerate(zip(tr['labels'], tr['cats'])):
        if lab is None or cat in CONTAINERS or cat == 'wooden_cabinet': continue
        pa, ba = place_of_canon(tr, reg, j, anchor_end, containers); po, bo = place_of_canon(tr, reg, j, other, containers)
        if pa is None or po is None: continue
        qa = img_to_canon(reg, W, (ba[0] + ba[2]) / 2, ba[3]); qo = img_to_canon(reg, W, (bo[0] + bo[2]) / 2, bo[3])
        dist = float(np.linalg.norm(qa - qo))
        if po != pa and (dist >= min_move or po.endswith('_contain_region') or pa.endswith('_contain_region') or 'cabinet' in po + pa):
            pairs.append({'object_id': lab, 'destination_region': po}); moved.append((lab, pa, po, round(dist, 2)))
    return {'pairs': pairs, 'valid': bool(pairs), 'moved': moved}


def program(task, tr, reg, objects, min_move=0.5):
    if reg.get('status') != 'ok': return {'pairs': [], 'valid': False, 'moved': []}
    W = tr['size'][0]; anchor_end = 0 if tr.get('anchor', len(tr['per_frame']) - 1) == 0 else -1; other = -1 if anchor_end == 0 else 0
    pairs, moved = [], []
    for j, (lab, cat) in enumerate(zip(tr['labels'], tr['cats'])):
        if lab is None or cat in CONTAINERS or cat == 'wooden_cabinet': continue
        pa, ba = place_of(tr, reg, j, anchor_end); po, bo = place_of(tr, reg, j, other)
        if pa is None or po is None: continue
        qa = img_to_canon(reg, W, (ba[0] + ba[2]) / 2, ba[3]); qo = img_to_canon(reg, W, (bo[0] + bo[2]) / 2, bo[3])
        dist = float(np.linalg.norm(qa - qo))
        if po != pa and (dist >= min_move or po.endswith('_contain_region') or pa.endswith('_contain_region') or 'cabinet' in po + pa):
            pairs.append({'object_id': lab, 'destination_region': po}); moved.append((lab, pa, po, round(dist, 2)))
    return {'pairs': pairs, 'valid': bool(pairs), 'moved': moved}
