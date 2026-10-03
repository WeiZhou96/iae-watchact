"""Intent programs over learned evidence scores (goal-free at inference).

NC: indicated candidates (video score > theta) are placed in time by their frame-score peaks, then parsed with
the deictic grammar (O+ D)+ ; each destination takes all pending objects.
RD: the selected set is the movable candidates with score > theta (at least the best one); the instruction picks
selected or complement; the destination is named, person-relative, or indicated.
"""
from __future__ import annotations
import json, math, re
import numpy as np
from .common import OUT, stem
from .register import canonical_xy

COMPLEMENT = re.compile(r'\b(other|others|remaining|unselected)\b', re.I)


def _dest(c):
    return c['id'] + '_contain_region' if c['kind'] == 'container' else c['id'] if c['kind'] in ('region', 'drawer') else None


def peaks(zk, drop=1.0, sep=15):
    top = zk.max(); cand = [t for t in range(len(zk)) if zk[t] >= top - drop and zk[t] == zk[max(0, t - 3):t + 4].max()]
    out = []
    for t in sorted(cand, key=lambda t: -zk[t]):
        if all(abs(t - u) >= sep for u in out): out.append(t)
    return sorted(out)


def nc_program_v1(f, d, theta=0.0):
    s, z, cands = d['s'], d['z'], f['cands']
    ind = [k for k in range(len(cands)) if s[k] > theta]
    events = []
    for k in ind:
        kind = cands[k]['kind']
        if kind == 'movable':
            events.append((int(np.argmax(z[:, k])), k, 'o'))
        elif kind in ('container', 'region', 'drawer'):
            for t in peaks(z[:, k]): events.append((t, k, 'd'))
    events.sort(); pending = []; pairs = {}
    for t, k, r in events:
        if r == 'o':
            if k not in pending: pending.append(k)
        elif pending:
            for o in pending: pairs[cands[o]['id']] = _dest(cands[k])
            pending = []
    ok = bool(pairs) and not pending and len(pairs) <= 3
    used = [k for k in ind]
    conf = float(min(abs(s[k] - theta) for k in range(len(cands)))) if ok else 0.0
    return {'pairs': [{'object_id': o, 'destination_region': g} for o, g in pairs.items()], 'valid': ok, 'confidence': conf,
            'indicated': [cands[k]['id'] for k in used]}


def _person_canonical(vk, reg, W, keep=None):
    p = OUT / 'person' / f'{vk}.json'
    if not p.exists() or reg.get('status') != 'ok': return None
    th = math.radians(reg['angle_deg']); sx, sy, u, v = reg['scale_shift']; c, s = math.cos(th), math.sin(th); qs = []
    ks = set(keep) if keep else None
    for r in json.loads(p.read_text()):
        if ks is not None and r.get('i') not in ks: continue
        if r['box'] and r['score'] >= 0.35:
            x, y = (r['box'][0] + r['box'][2]) / 2, r['box'][3]
            a = np.array([x / W * sx + u, y / W * sy + v]); qs.append([c * a[0] - s * a[1], s * a[0] + c * a[1]])
    return np.median(np.array(qs), 0) if qs else None


def rd_program_v1(f, d, req, theta=0.0):
    s, cands = d['s'], f['cands']; text = req['instruction'].lower(); objects = req['objects']
    mov = [k for k, c in enumerate(cands) if c['kind'] == 'movable']
    if not mov: return {'pairs': [], 'valid': False, 'confidence': 0.0}
    plural = re.search(r'\b(others|ones|objects|remaining)\b', text) is not None
    top = max(mov, key=lambda k: s[k]); cat = stem(cands[top]['id'])
    same = sorted([k for k in mov if stem(cands[k]['id']) == cat], key=lambda k: -s[k])
    if len(same) < 2: same = sorted(mov, key=lambda k: -s[k])  # reference domain backs off to all movable objects
    selected = [cands[k]['id'] for k in mov if s[k] > theta] or [cands[top]['id']]
    if COMPLEMENT.search(text):
        if plural:
            targets = [cands[k]['id'] for k in same if s[k] <= theta] or [cands[same[-1]]['id']]
            if len(targets) == len(same): targets = [cands[k]['id'] for k in same[1:]]
        else:
            targets = [cands[same[-1]]['id']] if len(same) > 1 else []
        selected = [cands[k]['id'] for k in same if cands[k]['id'] not in targets]
    else:
        targets = selected if plural else [cands[top]['id']]
    containers = [o for o in objects if o.startswith(('basket', 'wooden_tray'))]
    want = 'wooden_tray' if 'tray' in text else 'basket' if 'basket' in text else None
    pool = [o for o in containers if want is None or o.startswith(want)]
    on = {r[1]: r[2] for r in req['initial_states'] if len(r) == 3}
    dest = pool[0] if len(pool) == 1 else None
    if dest is None and pool and ('closer' in text or 'farther' in text):
        pc = _person_canonical(f['video_key'], f['register'], f['W'], f.get('keep'))
        if pc is not None:
            dd = {o: float(np.linalg.norm(np.array(canonical_xy(on[o])) - pc)) for o in pool if canonical_xy(on.get(o))}
            if dd: dest = (min if 'closer' in text else max)(dd, key=dd.get)
    if dest is None and pool:
        ck = [k for k, c in enumerate(cands) if c['id'] in pool]
        dest = cands[max(ck, key=lambda k: s[k])]['id'] if ck else pool[0]
    vals = sorted((s[k] for k in mov), reverse=True)
    conf = float(min(abs(v - theta) for v in vals))
    pairs = [{'object_id': o, 'destination_region': dest + '_contain_region'} for o in targets] if dest else []
    return {'pairs': pairs, 'valid': bool(pairs) and len(pairs) <= 3, 'confidence': conf, 'selected': selected, 'dest': dest}
