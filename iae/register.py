"""Final-frame instance registration: detected boxes -> public IDs, per video, using only public layout.

Each public object with an 'On <region>' initial state has a canonical table coordinate. For every
assignment of same-category detections to public IDs, a 5-DoF map (rotation, anisotropic scale,
translation) from image ground points to canonical coordinates is fitted by grid search over the
angle and linear least squares; the assignment with the smallest residual wins. The margin to the
second-best assignment is kept as a confidence signal.
"""
from __future__ import annotations
import itertools, math, re
import numpy as np
from .common import stem

ROW = {'back': 0.0, 'middle': 1.0, 'front': 2.0}
COL = {'left': 0.0, 'center': 1.0, 'right': 2.0}


def canonical_xy(region: str):
    m = re.fullmatch(r'main_(back|middle|front)_(left|center|right)_region', region or '')
    if m: return COL[m.group(2)], ROW[m.group(1)]
    m = re.fullmatch(r'fixture_(front|back)_region', region or '')
    if m: return -1.0, 1.5 if m.group(1) == 'front' else 0.5
    return None


def ground_point(box, W):
    return np.array([(box[0] + box[2]) / 2 / W, box[3] / W])


VIEW_ANGLES = {'front': (-30, 30), 'side': (230, 310), 'oblique': (0, 360)}  # degrees; public camera_perspective prior


def _fit(P, Q, view='oblique'):
    """Min over angle of || Q - (R(th) diag(s) P + t) ||^2 with s,t free; returns residual RMS and params."""
    best = (math.inf, None); lo, hi = VIEW_ANGLES.get(view, (0, 360))
    for deg in range(lo, hi, 5):
        th = math.radians(deg % 360); c, s = math.cos(th), math.sin(th)
        # Q = R diag(sx,sy) P + t  ->  R^T (Q - t) = diag(sx,sy) P ; solve linear in (sx, sy, u, v) with u,v = R^T t
        Qr = Q @ np.array([[c, -s], [s, c]])  # rows: R^T q
        A = np.zeros((2 * len(P), 4)); b = np.zeros(2 * len(P))
        A[0::2, 0] = P[:, 0]; A[0::2, 2] = 1; b[0::2] = Qr[:, 0]
        A[1::2, 1] = P[:, 1]; A[1::2, 3] = 1; b[1::2] = Qr[:, 1]
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        if x[0] <= 0 or x[1] <= 0: continue
        res = float(np.sqrt(np.mean((A @ x - b) ** 2)))
        if res < best[0]: best = (res, (th, x.tolist()))
    return best


def register(instances, objects, initial_states, W, view='oblique', max_assignments=20000):
    on = {r[1]: r[2] for r in initial_states if len(r) == 3 and r[0] == 'On'}
    anchored = [o for o in objects if canonical_xy(on.get(o)) is not None and not o.startswith('wooden_cabinet')]
    by_cat = {}
    for i, x in enumerate(instances): by_cat.setdefault(x['cat'], []).append(i)
    cats = sorted({stem(o) for o in anchored})
    ids_by_cat = {c: [o for o in anchored if stem(o) == c] for c in cats}
    options = []
    for c in cats:
        dets = by_cat.get(c, []); ids = ids_by_cat[c]
        k = min(len(dets), len(ids))
        options.append([(c, tuple(zip(perm_ids, perm_dets))) for perm_ids in itertools.permutations(ids, k)
                        for perm_dets in [tuple(dets[:k])]] if k else [(c, ())])
    # permute detections instead of ids when detections outnumber: handled by count-constrained detection upstream
    results = []
    for n, combo in enumerate(itertools.product(*options)):
        if n >= max_assignments: break
        pairs = [p for _, ps in combo for p in ps]
        if len(pairs) < 3: continue
        P = np.array([ground_point(instances[d]['box'], W) for _, d in pairs]); Q = np.array([canonical_xy(on[o]) for o, _ in pairs])
        res, params = _fit(P, Q, view)
        if params is not None: results.append((res, {o: d for o, d in pairs}, params))
    if not results:
        return {'status': 'insufficient_points', 'assignment': {}, 'residual': None, 'margin': None}
    results.sort(key=lambda r: r[0])
    best = results[0]; second = next((r for r in results[1:] if r[1] != best[1]), None)
    margin = (second[0] - best[0]) if second else None
    assignment = dict(best[1])
    cab_dets = [i for i, x in enumerate(instances) if x['cat'] == 'wooden_cabinet']
    for o, i in zip(sorted(o for o in objects if o.startswith('wooden_cabinet')), cab_dets): assignment[o] = i
    return {'status': 'ok', 'assignment': assignment, 'residual': best[0], 'margin': margin,
            'angle_deg': round(math.degrees(best[2][0]), 1), 'scale_shift': best[2][1],
            'unanchored_public_ids': sorted(set(objects) - set(anchored)), 'n_assignments': len(results)}
