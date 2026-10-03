import sys as _iae_sys
if '--help' in _iae_sys.argv:
    print('Usage: python %s [options]' % __file__)
    raise SystemExit(0)
"""Stage 11: episodic programs (Restore / Reversal / Imitation) for all requests, scored with the official scorer."""
import collections, json, sys
from pathlib import Path
from iae.common import requests, OUT, EPISODIC
from iae.episodic import program, program_v2
from iae.score import score_pairs

name = sys.argv[1] if len(sys.argv) > 1 else 'episodic_v1'
version = sys.argv[2] if len(sys.argv) > 2 else 'v1'
rows = []; agg = collections.defaultdict(collections.Counter); cache = {}
for r in requests(EPISODIC):
    vk = r['video_key']
    if vk not in cache:
        tp, rp = OUT / 'tracks' / f'{vk}.json', OUT / 'register' / f'{vk}.json'
        if not (tp.exists() and rp.exists()): cache[vk] = {'pairs': [], 'valid': False, 'moved': []}
        elif version == 'v2': cache[vk] = program_v2(r['task'], json.loads(tp.read_text()), json.loads(rp.read_text()), r['objects'], r['initial_states'])
        else: cache[vk] = program(r['task'], json.loads(tp.read_text()), json.loads(rp.read_text()), r['objects'])
    p = cache[vk]; s = score_pairs(r, p['pairs'], max_pairs=8)
    rows.append({'uid': r['uid'], 'task': r['task'][:3], 'view': r['view'], 'activity': r['activity_id'], 'pairs': p['pairs'], 'moved': p['moved'],
                 'succ': s['success'], 'strict': s['strict'], 'prog': s['progress'], 'fail': s['failure'], 'conf': 0.0})
    for key in [r['task'][:3], (r['task'][:3], r['view']), 'ALL']:
        a = agg[key]; a['n'] += 1; a['SR'] += s['success']; a['strict'] += s['strict']; a['prog'] += s['progress']
out = OUT / 'runs' / name; out.mkdir(parents=True, exist_ok=True); (out / 'rows.json').write_text(json.dumps(rows))
for k in sorted(agg, key=str):
    a = agg[k]; n = a['n']; print(k, f"n={n} SR={a['SR']} ({100*a['SR']/n:.1f}%) strict={a['strict']} ({100*a['strict']/n:.1f}%) prog={100*a['prog']/n:.1f}%")
