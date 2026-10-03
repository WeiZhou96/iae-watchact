import sys as _iae_sys
if '--help' in _iae_sys.argv:
    print('Usage: python %s [options]' % __file__)
    raise SystemExit(0)
"""Oracle diagnostics for NC programmes (analysis only; goals are never used in training or model selection).

On the averaged held-out frame logits of an ensemble run, decode NC programmes (i) as usual, (ii) with the gold
object set given (all gold objects must be used, destinations and pairing from the evidence), (iii) with
destination events limited to the gold destinations. Reports SR / strict over the 195 NC requests and pair
statistics per video: correct pairs, gold objects missed, objects sent to a wrong destination, extra objects.
Usage: diag_oracle.py <ensemble_run> <out.json>
"""
import json, pickle, sys
from pathlib import Path
import numpy as np
from iae.common import requests, OUT
from iae.decode import decode, decode_oracle
from iae.score import score_pairs, gold_pairs

run, out = sys.argv[1], sys.argv[2]
avg = pickle.load(open(OUT / 'runs' / run / 'scores.pkl', 'rb'))
cache = json.load(open(OUT / 'runs' / json.load(open(OUT / 'runs' / run / 'config.json'))['ensemble_of'][0] / 'config.json'))['cache']
NC = [r for r in requests() if r['task'] == 'Nonverbal_Cue']
res = {}
for mode in ('none', 'objects', 'destinations'):
    succ = strict = 0; pc = {'correct': 0, 'missed': 0, 'wrong_dst': 0, 'extra': 0, 'gold': 0}; seen = set(); dec = {}
    for r in NC:
        vk = r['video_key']
        if vk not in avg: continue
        if vk not in dec:
            f = pickle.loads((OUT / cache / f'{vk}.pkl').read_bytes()); G = gold_pairs(r['activity_id'])
            dec[vk] = decode(avg[vk]['z'], f['cands']) if mode == 'none' else decode_oracle(avg[vk]['z'], f['cands'], G, mode)
        pairs = dec[vk]['pairs']; s = score_pairs(r, pairs); succ += s['success']; strict += s['strict']
        if vk in seen: continue
        seen.add(vk); G = dict(gold_pairs(r['activity_id'])); P = {p['object_id']: p['destination_region'] for p in pairs}
        pc['gold'] += len(G)
        for o, d in G.items():
            if o not in P: pc['missed'] += 1
            elif P[o] == d: pc['correct'] += 1
            else: pc['wrong_dst'] += 1
        pc['extra'] += sum(o not in G for o in P)
    res[mode] = {'SR': round(100 * succ / len(NC), 1), 'strict': round(100 * strict / len(NC), 1), 'videos': len(seen), **pc}
    print(mode, res[mode], flush=True)
Path(out).write_text(json.dumps(res, indent=1))
