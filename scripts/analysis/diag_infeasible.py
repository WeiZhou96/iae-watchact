"""Why do some NC videos admit no goal-consistent programme? (diagnostics on the averaged held-out logits)

Classes, checked in order: (1) a goal object or destination is not among the candidates (not registered /
not visible), (2) the goal has more objects than N_max, (3) the programme becomes feasible when every local maximum
of every candidate is kept (pruning to three peaks within 3 logits), (4) otherwise the order of the peaks admits
no programme (an object peak must precede the destination peak that closes it).
Usage: diag_infeasible.py <ensemble_run>"""
import collections, json, pickle, sys
from iae.common import requests, OUT
from iae.decode import decode_constrained, dest_of
from iae.score import gold_pairs

run = sys.argv[1]
avg = pickle.load(open(OUT / 'runs' / run / 'scores.pkl', 'rb'))
seen = {}; cnt = collections.Counter()
for r in requests():
    if r['task'] != 'Nonverbal_Cue' or r['video_key'] in seen or r['video_key'] not in avg: continue
    vk = r['video_key']; f = pickle.loads((OUT / 'feat_cache_v3' / f'{vk}.pkl').read_bytes()); cands = f['cands']
    G = gold_pairs(r['activity_id']); z = avg[vk]['z']
    if decode_constrained(z, cands, G) is not None: seen[vk] = 'feasible'; cnt['feasible'] += 1; continue
    ids = {c['id'] for c in cands}; dests = {dest_of(c) for c in cands if c['kind'] in ('container', 'region', 'drawer')}
    if any(o not in ids for o, _ in G) or any(d not in dests for _, d in G): why = 'missing candidate'
    elif len(G) > 3: why = 'more than N_max objects'
    elif decode_constrained(z, cands, G, per_cand=10 ** 6) is not None: why = 'pruning (3 peaks per candidate)'
    else:
        import iae.decode as D
        old = D.candidate_events
        D.candidate_events = lambda z_, c_, per_cand=3, drop=3.0, sep=10: old(z_, c_, 10 ** 6, 1e9, 1)
        ok = decode_constrained(z, cands, G) is not None
        D.candidate_events = old
        why = 'pruning (3-logit window / 1-s spacing)' if ok else 'order of the peaks'
    seen[vk] = why; cnt[why] += 1
print(dict(cnt), 'videos', len(seen))
