"""Intent programs over interaction evidence (goal-free).

NC: indication episodes -> roles by candidate kind -> operations (grouped or alternating).
RD: selected set from interaction -> selected / complement -> destination by name, person proximity or indication.
Every output carries a confidence used by the commitment gate.
"""
from __future__ import annotations
import math, re
import numpy as np
from .common import stem


def softmax(x):
    x = np.asarray(x, float); m = x.max(); e = np.exp(x - m); return e / e.sum()


def frame_posteriors(ev, min_pose=0.5):
    """Per frame: best pointing hand -> (pose, posterior over candidates)."""
    out = []
    for f in ev['frames']:
        best = None
        for h in f['hands']:
            if h['pose'] >= min_pose and (best is None or h['pose'] > best['pose']): best = h
        out.append(None if best is None else (best['pose'], softmax(best['logscore'])))
    return out


def episodes(ev, min_pose=0.5, min_len=3, max_gap=2):
    """Contiguous pointing runs with a stable argmax target; returns list of {start, end, post, target}."""
    fp = frame_posteriors(ev, min_pose); eps = []; cur = None; gap = 0
    for t, x in enumerate(fp):
        if x is None:
            gap += 1
            if cur and gap > max_gap: eps.append(cur); cur = None
            continue
        pose, post = x; k = int(post.argmax())
        if cur and (k == cur['target'] or post[cur['target']] > 0.3) and gap <= max_gap:
            cur['end'] = t; cur['acc'] += post * pose; cur['n'] += 1
        else:
            if cur: eps.append(cur)
            cur = {'start': t, 'end': t, 'target': k, 'acc': post * pose, 'n': 1}
        gap = 0
    if cur: eps.append(cur)
    res = []
    for e in eps:
        if e['n'] < min_len: continue
        post = e['acc'] / e['acc'].sum(); k = int(post.argmax()); srt = np.sort(post)[::-1]
        res.append({'start': e['start'], 'end': e['end'], 'target': k, 'post': post, 'margin': float(srt[0] - (srt[1] if len(srt) > 1 else 0))})
    merged = []  # merge consecutive episodes pointing at the same target
    for e in res:
        if merged and merged[-1]['target'] == e['target'] and e['start'] - merged[-1]['end'] <= 10:
            m = merged[-1]; m['end'] = e['end']; m['post'] = (m['post'] + e['post']) / 2; m['margin'] = max(m['margin'], e['margin'])
        else: merged.append(dict(e))
    return merged


def dest_region(c):
    return c['id'] + '_contain_region' if c['kind'] == 'container' else c['id'] if c['kind'] == 'region' else None


def nc_program(ev, **kw):
    eps = episodes(ev, **kw); cands = ev['cands']; pending = []; pairs = []; conf = []
    for e in eps:
        c = cands[e['target']]
        if c['kind'] == 'movable':
            if c['id'] not in [p[0] for p in pending]: pending.append((c['id'], e['margin']))
        elif c['kind'] in ('container', 'region') and pending:
            for oid, m in pending:
                pairs = [p for p in pairs if p['object_id'] != oid]
                pairs.append({'object_id': oid, 'destination_region': dest_region(c)}); conf.append(min(m, e['margin']))
            pending = []
    ok = bool(pairs) and not pending and len(pairs) <= 3
    return {'pairs': pairs, 'confidence': (min(conf) if conf and ok else 0.0), 'valid': ok,
            'episodes': [{'start': e['start'], 'end': e['end'], 'target': cands[e['target']]['id'], 'margin': round(e['margin'], 3)} for e in eps]}


def handled_scores(ev, contact_w=1.0, motion_w=1.0, point_w=0.5):
    """Per movable candidate: evidence that the person interacted with it (contact, displacement, pointing)."""
    cands = ev['cands']; fp = frame_posteriors(ev)
    s = {}
    for k, c in enumerate(cands):
        if c['kind'] != 'movable': continue
        contact = sum(max((h['contact'][k] for h in f['hands']), default=0.0) for f in ev['frames']) / 10.0
        disp = max((f['disp'][k] or 0.0) for f in ev['frames'])
        point = sum(x[1][k] * x[0] for x in fp if x is not None) / 10.0
        s[c['id']] = contact_w * min(contact, 1.5) + motion_w * min(disp / 0.05, 2.0) + point_w * min(point, 1.5)
    return s


def person_canonical(ev):
    reg = ev['register']
    if not ev['person_points'] or reg.get('status') != 'ok': return None
    th = math.radians(reg['angle_deg']); sx, sy, u, v = reg['scale_shift']; c, s = math.cos(th), math.sin(th); W = ev['W']
    qs = []
    for x, y in ev['person_points']:
        p = np.array([x / W * sx + u, y / W * sy + v])  # R^T q
        qs.append([c * p[0] - s * p[1], s * p[0] + c * p[1]])
    return np.median(np.array(qs), axis=0)


COMPLEMENT = re.compile(r'\b(other|others|remaining|unselected)\b', re.I)
SELECTED = re.compile(r'\b(selected|this object|this one)\b', re.I)


def rd_program(ev, instruction, objects, initial_states, thresh=1.0):
    from .register import canonical_xy
    text = instruction.lower()
    hs = handled_scores(ev)
    if not hs: return {'pairs': [], 'confidence': 0.0, 'valid': False}
    ranked = sorted(hs.items(), key=lambda x: -x[1])
    selected = [o for o, v in hs.items() if v >= thresh] or [ranked[0][0]]
    sel_cats = {stem(o) for o in selected}
    if COMPLEMENT.search(text):
        targets = [o for o in objects if stem(o) in sel_cats and o not in selected and not o.startswith(('basket', 'wooden_tray', 'wooden_cabinet'))]
    else:
        targets = selected
    # destination
    containers = [o for o in objects if o.startswith(('basket', 'wooden_tray'))]
    want = 'wooden_tray' if 'tray' in text else 'basket' if 'basket' in text else None
    pool = [o for o in containers if want is None or o.startswith(want)]
    on = {r[1]: r[2] for r in initial_states if len(r) == 3}
    dest = None
    if len(pool) == 1: dest = pool[0]
    elif pool and ('closer' in text or 'farther' in text):
        pc = person_canonical(ev)
        if pc is not None:
            d = {o: float(np.linalg.norm(np.array(canonical_xy(on[o])) - pc)) for o in pool if canonical_xy(on.get(o))}
            if d: dest = (min if 'closer' in text else max)(d, key=d.get)
    elif pool and 'that container' in text or (pool and len(pool) > 1):
        eps = episodes(ev); cand_ids = [c['id'] for c in ev['cands']]
        pointed = [cand_ids[e['target']] for e in eps if cand_ids[e['target']] in pool]
        if pointed: dest = pointed[-1]
    if dest is None and pool: dest = pool[0]
    vals = sorted(hs.values(), reverse=True)
    gap = vals[len(selected) - 1] - (vals[len(selected)] if len(vals) > len(selected) else 0.0)
    pairs = [{'object_id': o, 'destination_region': dest + '_contain_region'} for o in targets] if dest else []
    return {'pairs': pairs, 'confidence': float(gap), 'valid': bool(pairs) and len(pairs) <= 3,
            'selected': selected, 'handled': {k: round(v, 3) for k, v in hs.items()}, 'dest': dest}
