"""Summary of the layout-dependence experiment (all 455 NC/RD requests and 800 episodic requests; missing = failure)."""
import json, re, sys
from pathlib import Path
import numpy as np
from iae.common import requests, OUT
from iae.score import score_pairs

REQS = requests(); N = {'No': 195, 'Re': 260}
LOG = OUT / 'logs'


def metrics(run):
    P = {x['uid']: x['pairs'] for x in json.load(open(OUT / 'runs' / run / 'rows.json'))}
    agg = {'No': [0, 0], 'Re': [0, 0]}
    for r in REQS:
        s = score_pairs(r, P.get(r['uid']) or []); t = r['task'][:2]
        agg[t][0] += s['success']; agg[t][1] += s['strict']
    m = {f'{t}_{k}': 100 * agg[t][i] / N[t] for t in agg for i, k in enumerate(('SR', 'strict'))}
    m['ALL_SR'] = 100 * (agg['No'][0] + agg['Re'][0]) / 455; m['ALL_strict'] = 100 * (agg['No'][1] + agg['Re'][1]) / 455
    return m


def epi(run):
    rows = json.load(open(OUT / 'runs' / run / 'rows.json'))
    return 100 * np.mean([x['succ'] for x in rows])


def changed(cond):
    t = (LOG / f'pert_{cond}.log').read_text()
    m = re.search(r"'changed': (\d+)", t); n = re.search(r'videos (\d+)', t)
    return 100 * int(m.group(1)) / int(n.group(1))


res = {'clean': {'implicit': [metrics(f'abl_notemporal_s{s}') for s in (0, 1, 2)], 'episodic': [epi('episodic_v1')] * 3,
                 'changed': [0.0] * 3}}
for name in ('noise025', 'noise050', 'noise100', 'randid'):
    res[name] = {'implicit': [metrics(f'layout_{name}_s{s}') for s in (0, 1, 2)],
                 'episodic': [epi(f'episodic_{name}_s{s}') for s in (0, 1, 2)],
                 'changed': [changed(f'{name}_s{s}') for s in (0, 1, 2)]}
for k, v in res.items():
    im = {m: (round(float(np.mean([x[m] for x in v['implicit']])), 1), round(float(np.std([x[m] for x in v['implicit']])), 1))
          for m in v['implicit'][0]}
    print(k, im, 'episodic', round(float(np.mean(v['episodic'])), 1), round(float(np.std(v['episodic'])), 1),
          'changed%', round(float(np.mean(v['changed'])), 1), flush=True)
Path(sys.argv[1]).write_text(json.dumps(res))
