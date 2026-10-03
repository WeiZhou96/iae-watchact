"""Average held-out frame/video logits of several CV runs (same folds), decode, run RD programs, and score.

Usage: ensemble.py <out_name> <cost> <run1> <run2> ...
"""
import collections, json, pickle, sys
from pathlib import Path
import numpy as np
from iae.common import requests, OUT
from iae.decode import decode, program_margin
from iae.program_v1 import rd_program_v1
from iae.score import score_pairs

name, cost, runs = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
S = [pickle.load(open(OUT / 'runs' / r / 'scores.pkl', 'rb')) for r in runs]
CACHE = json.load(open(OUT / 'runs' / runs[0] / 'config.json')).get('cache', 'feat_cache')
vks = set.intersection(*[set(s) for s in S])
avg = {vk: {'z': np.mean([s[vk]['z'] for s in S], 0), 's': np.mean([s[vk]['s'] for s in S], 0)} for vk in vks}
rows = []; agg = collections.defaultdict(collections.Counter); dec = {}
for r in requests():
    vk = r['video_key']; pairs = []; conf = 0.0; valid = False
    if vk in avg:
        f = pickle.loads((OUT / CACHE / f'{vk}.pkl').read_bytes())
        if r['task'] == 'Nonverbal_Cue':
            if vk not in dec:
                dec[vk] = decode(avg[vk]['z'], f['cands'], cost=cost); dec[vk]['margin'] = program_margin(avg[vk]['z'], f['cands'], cost=cost)
            p = dec[vk]; pairs, conf, valid = p['pairs'], p['margin'], p['valid']
        else:
            p = rd_program_v1(f, avg[vk], r); pairs, conf, valid = p['pairs'], p['confidence'], p['valid']
    s = score_pairs(r, pairs)
    rows.append({'uid': r['uid'], 'task': r['task'][:2], 'view': r['view'], 'ref': r['reference'], 'activity': r['activity_id'],
                 'pairs': pairs, 'conf': conf, 'valid': valid, 'succ': s['success'], 'strict': s['strict'], 'prog': s['progress']})
    for key in ['ALL', r['task'][:2]]:
        a = agg[key]; a['n'] += 1; a['SR'] += s['success']; a['strict'] += s['strict']; a['prog'] += s['progress']
out = OUT / 'runs' / name; out.mkdir(parents=True, exist_ok=True)
(out / 'rows.json').write_text(json.dumps(rows, default=str)); pickle.dump(avg, open(out / 'scores.pkl', 'wb'))
(out / 'config.json').write_text(json.dumps({'ensemble_of': runs, 'cost': cost}))
for k in ['No', 'Re', 'ALL']:
    a = agg[k]; n = a['n']
    print(f"{k:3s} n={n} SR={a['SR']} ({100*a['SR']/n:.1f}%) strict={a['strict']} ({100*a['strict']/n:.1f}%) progress={100*a['prog']/n:.1f}%")
