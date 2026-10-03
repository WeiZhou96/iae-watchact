import sys as _iae_sys
if '--help' in _iae_sys.argv:
    print('Usage: python %s [options]' % __file__)
    raise SystemExit(0)
"""Score direct predictions or an episodic run on Restore / Reversal / Imitation (800 requests; missing = failure),
with activity-cluster bootstrap CIs and a paired comparison against a reference.
Usage: score_episodic.py name=spec [...] [--ref spec]   spec: run:<run> | direct:<dir>
"""
import collections, json, math, random, sys
from pathlib import Path
import numpy as np
from iae.common import requests, OUT, EPISODIC
from iae.score import score_pairs

REQS = requests(EPISODIC)


def preds(spec):
    kind, x = spec.split(':', 1); p = {}
    if kind == 'run':
        for r in json.load(open(OUT / 'runs' / x / 'rows.json')): p[r['uid']] = r['pairs']
    else:
        for f in Path(x).glob('pred_*.jsonl'):
            for s in open(f):
                r = json.loads(s); p[r['uid']] = (r.get('pairs') if r['status'] == 'ok' else None) or []
    return p


def scores(spec):
    p = preds(spec); return {r['uid']: (r['task'][:3], r['activity_id'], score_pairs(r, p.get(r['uid']) or [], max_pairs=8)) for r in REQS}


def ci(vals_by_act, rng, B=5000):
    acts = list(vals_by_act); n = len(acts)
    res = [np.mean([x for a in (acts[rng.randrange(n)] for _ in range(n)) for x in vals_by_act[a]]) for _ in range(B)]
    return np.percentile(res, [2.5, 97.5])


args = [a for a in sys.argv[1:] if not a.startswith('--ref')]
ref = next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--ref=')), None)
R = scores(ref) if ref else None; rng = random.Random(0)
for item in args:
    name, spec = item.split('=', 1); S = scores(spec); line = []
    for t in ('Imi', 'Res', 'Rev', 'ALL'):
        by = collections.defaultdict(list)
        for uid, (task, act, s) in S.items():
            if t == 'ALL' or task == t: by[act].append(float(s['success']))
        m = 100 * np.mean([x for v in by.values() for x in v]); lo, hi = 100 * ci(by, rng)
        strict = 100 * np.mean([float(s['strict']) for uid, (task, act, s) in S.items() if t == 'ALL' or task == t])
        line.append(f'{t}: SR {m:.1f} [{lo:.1f},{hi:.1f}] strict {strict:.1f}')
    print(f'{name:22s} ' + ' | '.join(line))
    if R is not None:
        by = collections.defaultdict(list)
        for uid, (task, act, s) in S.items(): by[act].append(float(s['success']) - float(R[uid][2]['success']))
        d = 100 * np.mean([x for v in by.values() for x in v]); lo, hi = 100 * ci(by, rng)
        means = [np.mean(v) for v in by.values()]; pos = int(sum(x > 0 for x in means)); neg = int(sum(x < 0 for x in means))
        n = pos + neg; pv = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(pos, neg) + 1)) / 2 ** n) if n else 1.0
        print(f"{'':22s} vs ref: dSR {d:+.1f} [{lo:+.1f},{hi:+.1f}]  activities +{pos}/-{neg}/={len(means) - n}  sign-test p={pv:.2g}")
