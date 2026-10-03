"""Per-activity paired statistics of each ablation against the final model (seed-averaged per request).

For every run prefix (seeds 0-2): score all 455 NC/RD requests (missing = failure), average success and strict
over seeds per request, then compare with the final model (abl_notemporal) by activity: mean difference,
5,000-sample activity bootstrap CI, two-sided sign test. Also prints means so rows can be matched to Table III.
"""
import collections, json, math, random, sys
from pathlib import Path
import numpy as np
from iae.common import requests, OUT
from iae.score import score_pairs

REQS = requests()
PREFIX = {'full': 'abl_notemporal', 'noray': 'abl_final', 'temporal': 'abl_full', 'nostruct': 'abl_m_nostruct',
          'nogrammar': 'abl_m_nogrammar', 'notrack': 'abl_m_notrack', 'blocked': 'abl_m_blocked',
          'handcrafted': 'rule_heuristic_eval_c0', 'hamming': 'rev_ham', 'sub32': 'rev_sub32', 'relation': 'rel'}


def seed_avg(prefix):
    runs = [prefix] if prefix.startswith('rule') else [f'{prefix}_s{s}' for s in (0, 1, 2)]
    acc = collections.defaultdict(lambda: [0.0, 0.0])
    for run in runs:
        P = {x['uid']: x['pairs'] for x in json.load(open(OUT / 'runs' / run / 'rows.json'))}
        for r in REQS:
            s = score_pairs(r, P.get(r['uid']) or [])
            acc[r['uid']][0] += s['success'] / len(runs); acc[r['uid']][1] += s['strict'] / len(runs)
    return acc


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
    return dict(d=round(d, 1), lo=round(lo, 1), hi=round(hi, 1), p=p, pos=pos, neg=neg)


A = {k: seed_avg(v) for k, v in PREFIX.items()}
out = {}
for k, acc in A.items():
    m = {}
    for t in ('No', 'Re', None):
        u = [r['uid'] for r in REQS if t is None or r['task'][:2] == t]
        m[(t or 'ALL') + '_SR'] = round(100 * np.mean([acc[x][0] for x in u]), 1)
        m[(t or 'ALL') + '_strict'] = round(100 * np.mean([acc[x][1] for x in u]), 1)
    if k != 'full':
        m['vs_full_ALL_SR'] = paired(acc, A['full'], 0); m['vs_full_ALL_strict'] = paired(acc, A['full'], 1)
        m['vs_full_NC_strict'] = paired(acc, A['full'], 1, 'No'); m['vs_full_RD_strict'] = paired(acc, A['full'], 1, 'Re')
    out[k] = m
    print(k, json.dumps(m), flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
