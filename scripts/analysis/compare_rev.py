import sys as _iae_sys
if '--help' in _iae_sys.argv:
    print('Usage: python %s [options]' % __file__)
    raise SystemExit(0)
"""Seed-averaged metrics of run prefixes and activity-level paired statistics against a reference prefix.
Usage: compare_rev.py <ref_prefix> <prefix> [<prefix> ...] [--out file.json]
Requests missing from a run count as failures (NC-only runs are compared on NC only)."""
import collections, json, math, random, sys
from pathlib import Path
import numpy as np
from iae.common import requests, OUT
from iae.score import score_pairs

args = sys.argv[1:]; out = None
if '--out' in args: i = args.index('--out'); out = args[i + 1]; args = args[:i] + args[i + 2:]
REQS = [r for r in requests() if r['task'] in ('Nonverbal_Cue', 'Reference_Disambiguation')]


def runs_of(prefix):
    return [prefix] if (OUT / 'runs' / prefix / 'rows.json').exists() else [f'{prefix}_s{s}' for s in (0, 1, 2)]


def seed_avg(prefix):
    runs = runs_of(prefix); acc = collections.defaultdict(lambda: [0.0, 0.0, 0.0]); tasks = set()
    for run in runs:
        rows = json.load(open(OUT / 'runs' / run / 'rows.json'))
        P = {x['uid']: x['pairs'] for x in rows}; tasks |= {x['task'] for x in rows}
        for r in REQS:
            s = score_pairs(r, P.get(r['uid']) or [])
            acc[r['uid']][0] += s['success'] / len(runs); acc[r['uid']][1] += s['strict'] / len(runs)
            acc[r['uid']][2] += s['progress'] / len(runs)
    return acc, tasks


def paired(a, b, k, task=None, B=5000):
    by = collections.defaultdict(list)
    for r in REQS:
        if task and r['task'][:2] != task: continue
        by[r['activity_id']].append(a[r['uid']][k] - b[r['uid']][k])
    acts = list(by); rng = random.Random(0); n = len(acts)
    d = 100 * np.mean([x for v in by.values() for x in v])
    bs = [np.mean([x for a_ in (acts[rng.randrange(n)] for _ in range(n)) for x in by[a_]]) for _ in range(B)]
    lo, hi = 100 * np.percentile(bs, [2.5, 97.5])
    means = [np.mean(v) for v in by.values()]; pos = int(sum(m > 1e-9 for m in means)); neg = int(sum(m < -1e-9 for m in means))
    nz = pos + neg; p = min(1.0, 2 * sum(math.comb(nz, j) for j in range(min(pos, neg) + 1)) / 2 ** nz) if nz else 1.0
    return dict(d=round(d, 1), lo=round(lo, 1), hi=round(hi, 1), p=float(f'{p:.3g}'), pos=pos, neg=neg)


ref, others = args[0], args[1:]
R, _ = seed_avg(ref); res = {}
for pre in [ref] + others:
    A, tasks = seed_avg(pre) if pre != ref else (R, None); m = {}
    for t in ('No', 'Re', None):
        u = [r['uid'] for r in REQS if t is None or r['task'][:2] == t]
        for j, key in enumerate(('SR', 'strict', 'prog')): m[f'{t or "ALL"}_{key}'] = round(100 * np.mean([A[x][j] for x in u]), 1)
    if pre != ref:
        for t in ('No', 'Re', None):
            for j, key in ((0, 'SR'), (1, 'strict')): m[f'vs_ref_{t or "ALL"}_{key}'] = paired(A, R, j, t)
    res[pre] = m
    print(pre, json.dumps({k: v for k, v in m.items() if not k.startswith('vs_ref') or k.startswith('vs_ref_No')}), flush=True)
if out: Path(out).write_text(json.dumps(res, indent=1))
