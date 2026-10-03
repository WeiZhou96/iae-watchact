"""Per-request gold and predicted pairs (canonical frame), reference frame, camera and success for NC/RD,
for IAE (ensemble and single seeds), the hand-crafted variant and all direct baselines."""
import json, sys
from pathlib import Path
from iae.common import requests, OUT
from iae.score import score_pairs, gold_pairs

REQS = requests()
out = {'requests': [{'uid': r['uid'], 'task': r['task'][:2], 'view': r['view'], 'ref': r['reference'],
                     'activity': r['activity_id'], 'gold': gold_pairs(r['activity_id'])} for r in REQS], 'methods': {}}


def from_run(run):
    return {x['uid']: x['pairs'] for x in json.load(open(OUT / 'runs' / run / 'rows.json'))}


def from_direct(d):
    p = {}
    for f in (OUT / d).glob('pred_*.jsonl'):
        for s in open(f):
            x = json.loads(s); p[x['uid']] = (x.get('pairs') if x['status'] == 'ok' else None) or []
    return p


SRC = {'iae_ens': ('run', 'ens_final'), 'iae_s0': ('run', 'abl_notemporal_s0'), 'iae_s1': ('run', 'abl_notemporal_s1'),
       'iae_s2': ('run', 'abl_notemporal_s2'), 'handcrafted': ('run', 'rule_heuristic_eval_c0'),
       'd8': ('dir', 'direct_8b'), 'd32': ('dir', 'direct_32b'), 'd8o': ('dir', 'direct_8b_overlay'), 'd32o': ('dir', 'direct_32b_overlay'), 'dvl': ('dir', 'direct_internvl8b'), 's32t': ('dir', 'struct32b_text'), 's32f': ('dir', 'struct32b_frames'), 's32t2': ('dir', 'struct32b_text_v2')}
for name, (kind, x) in SRC.items():
    P = from_run(x) if kind == 'run' else from_direct(x); rows = {}
    for r in REQS:
        pairs = P.get(r['uid']) or []
        s = score_pairs(r, pairs)
        rows[r['uid']] = {'pred': [(p['object_id'], p['destination_region']) for p in pairs if isinstance(p, dict)],
                          'succ': bool(s['success']), 'strict': bool(s['strict']), 'valid': s['failure'] is None}
    out['methods'][name] = rows
    print(name, sum(v['succ'] for v in rows.values()), sum(v['strict'] for v in rows.values()), flush=True)
Path(sys.argv[1]).write_text(json.dumps(out))
