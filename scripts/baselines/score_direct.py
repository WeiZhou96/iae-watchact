import sys as _iae_sys
if '--help' in _iae_sys.argv:
    print('Usage: python %s [options]' % __file__)
    raise SystemExit(0)
"""Score direct-baseline predictions for all NC/RD requests (missing/corrupt count as failures)."""
import collections, json, sys
from pathlib import Path
from iae.common import requests
from iae.score import score_pairs

d = Path(sys.argv[1]); pred = {}
for f in d.glob('pred_*.jsonl'):
    for s in open(f):
        x = json.loads(s); pred[x['uid']] = x
agg = collections.defaultdict(collections.Counter); rows = []
for r in requests():
    x = pred.get(r['uid']); pairs = (x.get('pairs') if x and x['status'] == 'ok' else None) or []
    s = score_pairs(r, pairs); rows.append({'uid': r['uid'], 'succ': s['success'], 'prog': s['progress'], 'fail': s['failure'], 'have': x is not None})
    for key in [(r['task'][:2],), (r['task'][:2], r['view'])]:
        a = agg[key]; a['n'] += 1; a['succ'] += s['success']; a['prog_x100'] += round(100 * s['progress']); a['have'] += x is not None
(d / 'scored.json').write_text(json.dumps(rows))
for k in sorted(agg): print(k, dict(agg[k]))
