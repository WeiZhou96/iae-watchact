"""Apply VLM destination verification to a decoded run and score (NC re-decoded; RD rows copied).
Usage: apply_verify.py <src_run> <verify_run>
"""
import collections, json, pickle, sys
from pathlib import Path
from iae.common import requests, OUT
from iae.decode import decode, dest_of
from iae.score import score_pairs

src, ver = sys.argv[1], sys.argv[2]
cfg = json.load(open(OUT / 'runs' / src / 'config.json')); cache = cfg.get('cache', 'feat_cache')
scores = pickle.load(open(OUT / 'runs' / src / 'scores.pkl', 'rb'))
changes = {}
for f in (OUT / 'runs' / ver).glob('verify_*.jsonl'):
    for s in open(f):
        x = json.loads(s); changes[x['video_key']] = {c['t']: c['to'] for c in x['changes']}
src_rows = {x['uid']: x for x in json.load(open(OUT / 'runs' / src / 'rows.json'))}
rows = []; agg = collections.defaultdict(collections.Counter); newpairs = {}
for r in requests():
    vk = r['video_key']
    if r['task'] == 'Nonverbal_Cue' and vk in scores:
        if vk not in newpairs:
            f = pickle.loads((OUT / cache / f'{vk}.pkl').read_bytes()); cands = f['cands']; byid = {c['id']: c for c in cands}
            d = decode(scores[vk]['z'], cands, cost=0); pending = []; pairs = {}
            for (t, cid, role, v) in d['events']:
                if role == 'o': pending.append(cid)
                else:
                    tgt = changes.get(vk, {}).get(t, cid)
                    for o in pending: pairs[o] = dest_of(byid[tgt])
                    pending = []
            newpairs[vk] = [{'object_id': o, 'destination_region': g} for o, g in pairs.items()]
        pairs = newpairs[vk]
    else:
        pairs = src_rows.get(r['uid'], {}).get('pairs', [])
    s = score_pairs(r, pairs); rows.append({'uid': r['uid'], 'task': r['task'][:2], 'view': r['view'], 'pairs': pairs, 'succ': s['success'], 'strict': s['strict'], 'prog': s['progress']})
    for key in ['ALL', r['task'][:2]]:
        a = agg[key]; a['n'] += 1; a['SR'] += s['success']; a['strict'] += s['strict']; a['prog'] += s['progress']
(OUT / 'runs' / ver / 'rows.json').write_text(json.dumps(rows))
for k in ['No', 'Re', 'ALL']:
    a = agg[k]; n = a['n']; print(f"{k} n={n} SR={a['SR']} ({100*a['SR']/n:.1f}%) strict={a['strict']} ({100*a['strict']/n:.1f}%) prog={100*a['prog']/n:.1f}%")
print('slots changed', sum(len(v) for v in changes.values()), 'videos', len(changes))
