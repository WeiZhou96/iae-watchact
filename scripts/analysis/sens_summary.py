"""Summaries for the learning curve and sensitivity grid (all 455 NC/RD requests; missing = failure).

Trained settings: seeds 0-2 of each run prefix. Test-time decoding cost lambda: the three final seeds' frame
logits are averaged (as for the ensemble) and NC programs are re-decoded with each lambda.
Output: JSON with per-setting per-seed metrics.
"""
import json, pickle, sys
from pathlib import Path
import numpy as np
from iae.common import requests, OUT
from iae.score import score_pairs
from iae.decode import decode

REQS = requests()
NC = [r for r in REQS if r['task'] == 'Nonverbal_Cue']; N = {'No': 195, 'Re': 260}


def metrics(pairs_of):
    out = {}
    agg = {'No': [0, 0], 'Re': [0, 0]}
    for r in REQS:
        s = score_pairs(r, pairs_of.get(r['uid']) or [])
        t = r['task'][:2]; agg[t][0] += s['success']; agg[t][1] += s['strict']
    for t in agg: out[t + '_SR'], out[t + '_strict'] = 100 * agg[t][0] / N[t], 100 * agg[t][1] / N[t]
    out['ALL_SR'] = 100 * (agg['No'][0] + agg['Re'][0]) / 455; out['ALL_strict'] = 100 * (agg['No'][1] + agg['Re'][1]) / 455
    return out


def run_metrics(run):
    rows = json.load(open(OUT / 'runs' / run / 'rows.json'))
    return metrics({x['uid']: x['pairs'] for x in rows})


SETTINGS = {'frac25': 'sens_frac25', 'frac50': 'sens_frac50', 'frac75': 'sens_frac75', 'frac100': 'abl_notemporal',
            'k1': 'sens_k1', 'k3': 'sens_k3', 'k5': 'abl_notemporal', 'k10': 'sens_k10', 'k20': 'sens_k20',
            'mu0': 'abl_m_nostruct', 'mu003': 'sens_mu003', 'mu01': 'abl_notemporal', 'mu03': 'sens_mu03', 'mu1': 'sens_mu1'}
res = {'trained': {}, 'lambda': {}}
cache = {}
for key, prefix in SETTINGS.items():
    per = []
    for s in (0, 1, 2):
        run = f'{prefix}_s{s}'
        if run not in cache: cache[run] = run_metrics(run)
        per.append(cache[run])
    res['trained'][key] = per
    print(key, {m: round(float(np.mean([p[m] for p in per])), 1) for m in per[0]}, flush=True)

# test-time lambda on the ensemble logits
sc = [pickle.load(open(OUT / 'runs' / f'abl_notemporal_s{s}' / 'scores.pkl', 'rb')) for s in (0, 1, 2)]
cfg = json.load(open(OUT / 'runs' / 'abl_notemporal_s0' / 'config.json')); cache_name = cfg['cache']
ens_rows = {x['uid']: x['pairs'] for x in json.load(open(OUT / 'runs' / 'ens_final' / 'rows.json'))}
feats = {}
for lam in (-2, -1, -0.5, 0, 0.5, 1, 2):
    pairs = dict(ens_rows)
    for r in NC:
        vk = r['video_key']
        if vk not in sc[0]: pairs[r['uid']] = []; continue
        if vk not in feats: feats[vk] = pickle.loads((OUT / cache_name / f'{vk}.pkl').read_bytes())['cands']
        z = np.mean([s[vk]['z'] for s in sc], 0)
        pairs[r['uid']] = decode(z, feats[vk], cost=lam)['pairs']
    res['lambda'][str(lam)] = metrics(pairs)
    print('lambda', lam, {m: round(v, 1) for m, v in res['lambda'][str(lam)].items()}, flush=True)
Path(sys.argv[1]).write_text(json.dumps(res))
