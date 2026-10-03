"""Export real data for the qualitative case figure (selection rule fixed in advance, seeded).

Case A  NC, ensemble strictly correct, 32B direct wrong, front camera, >= 2 gold pairs.
Case B  NC failure: ensemble selects the gold objects but confuses tray and basket for a destination.
Case C  RD, ensemble strictly correct, 32B direct wrong, and the selected instance moves by > 60 px
        between its evidence peak and the anchor frame (identity must be carried by backward tracking).
For each case: frames at the decoded events (NC) or at the evidence peak and anchor (RD), instance masks
(1/4 resolution), labels, hand and body keypoints at those frames, frame logits of all candidates,
decoded events, gold pairs, IAE pairs and 32B pairs.
"""
import json, pickle, random, sys, shutil
from pathlib import Path
import numpy as np
from iae.common import requests, OUT, frames_dir, load_frame_index
from iae.decode import decode
from iae.score import gold_pairs, score_pairs

dst = Path(sys.argv[1]); dst.mkdir(parents=True, exist_ok=True)
R = {r['uid']: r for r in requests()}
print('request keys', sorted(next(iter(R.values())).keys()))
ens = {r['uid']: r for r in json.load(open(OUT / 'runs/ens_final/rows.json'))}
sc = pickle.load(open(OUT / 'runs/ens_final/scores.pkl', 'rb'))
cfg = json.load(open(OUT / 'runs/ens_final/config.json')); cache = cfg.get('cache', 'feat_cache_v3')
d32 = {}
for f in (OUT / 'direct_32b').glob('pred_*.jsonl'):
    for s in open(f):
        x = json.loads(s); d32[x['uid']] = (x.get('pairs') if x['status'] == 'ok' else None) or []
ok32 = {u: score_pairs(R[u], d32.get(u, []))['success'] for u in R}


def feat(vk):
    return pickle.loads((OUT / cache / f'{vk}.pkl').read_bytes())


def track(vk):
    return json.loads((OUT / 'tracks' / f'{vk}.json').read_text())


rng = random.Random(0)
A = [u for u, r in ens.items() if r['task'] == 'No' and r['strict'] and not ok32[u] and r['view'] == 'front'
     and len(gold_pairs(r['activity'])) >= 2 and R[u]['video_key'] in sc]
B = []
for u, r in ens.items():
    if r['task'] != 'No' or r['succ'] or R[u]['video_key'] not in sc: continue
    g = dict(gold_pairs(r['activity'])); p = {x['object_id']: x['destination_region'] for x in r['pairs']}
    if set(p) != set(g): continue
    bad = [(p[o], g[o]) for o in g if p[o] != g[o]]
    if any(('tray' in a and 'basket' in b) or ('basket' in a and 'tray' in b) for a, b in bad): B.append(u)
C = []
for u, r in ens.items():
    if r['task'] != 'Re' or not r['strict'] or ok32[u] or R[u]['video_key'] not in sc: continue
    vk = R[u]['video_key']; f = feat(vk); z = sc[vk]['z']; tr = track(vk); ids = [c['id'] for c in f['cands']]
    o = r['pairs'][0]['object_id'] if r['pairs'] else None
    if o not in ids or o not in tr['labels']: continue
    k = ids.index(o); j = tr['labels'].index(o); t = int(np.argmax(z[:, k]))
    a, b = tr['per_frame'][t][j], tr['per_frame'][-1][j]
    if a and b and np.hypot(a['c'][0] - b['c'][0], a['c'][1] - b['c'][1]) > 60: C.append(u)
print('pool sizes', len(A), len(B), len(C))
pick = {'A': rng.choice(sorted(A)), 'B': rng.choice(sorted(B)), 'C': rng.choice(sorted(C))}

meta = {}
for name, u in pick.items():
    r = ens[u]; req = R[u]; vk = req['video_key']; f = feat(vk); z = np.asarray(sc[vk]['z']); tr = track(vk)
    ids = [c['id'] for c in f['cands']]; d = decode(z, f['cands'], cost=0)
    idx = load_frame_index(vk)
    if True:
        mv = [k for k, c in enumerate(f['cands']) if c['kind'] == 'movable']
        times = [e[0] for e in d['events']] + [int(np.argmax(z[:, k])) for k in mv] + [len(idx['frames']) - 1]
    else:
        mv = [k for k, c in enumerate(f['cands']) if c['kind'] == 'movable']; kb = max(mv, key=lambda k: z[:, k].max()); o = r['pairs'][0]['object_id']; times = [int(np.argmax(z[:, kb])), int(np.argmax(z[:, ids.index(o)])), len(idx['frames']) - 1]
    times = sorted(set(times))
    out = dst / name; out.mkdir(exist_ok=True)
    npz = np.load(OUT / 'tracks' / f'{vk}.npz'); shape = tuple(npz['shape'])
    masks = np.unpackbits(npz['masks'], axis=-1)[..., :shape[-1]].astype(bool)
    hands = {x['i']: x['hands'] for x in json.loads((OUT / 'hands' / f'{vk}.json').read_text())}
    body = {x['i']: x for x in json.loads((OUT / 'body' / f'{vk}.json').read_text())}
    frames = []
    for t in times + [len(idx['frames']) - 1]:
        fi = idx['frames'][t]; shutil.copy(frames_dir(vk) / fi['file'], out / f'frame_{t:04d}.jpg')
        np.save(out / f'mask_{t:04d}.npy', np.packbits(masks[t], axis=-1))
        frames.append({'t': t, 'sec': fi['t'], 'file': f'frame_{t:04d}.jpg', 'mask': f'mask_{t:04d}.npy',
                       'mask_shape': list(masks[t].shape), 'hands': hands.get(fi['i'], []),
                       'body': body.get(fi['i'], {}).get('kp'), 'body_sc': body.get(fi['i'], {}).get('sc'),
                       'boxes': tr['per_frame'][t]})
    meta[name] = {'uid': u, 'video_key': vk, 'activity': r['activity'], 'view': r['view'], 'ref': r['ref'],
                  'instruction': req.get('instruction') or req.get('text') or req.get('query'),
                  'labels': tr['labels'], 'cats': tr['cats'], 'size': tr['size'], 'anchor': tr.get('anchor'),
                  'fps_frames': [x['t'] for x in idx['frames']], 'cand_ids': ids,
                  'cand_kind': [c['kind'] for c in f['cands']], 'z': z.round(3).tolist(),
                  'events': d['events'], 'gold': gold_pairs(r['activity']), 'iae': r['pairs'], 'd32': d32.get(u, []),
                  'iae_succ': r['succ'], 'iae_strict': r['strict'], 'd32_succ': bool(ok32[u]), 'frames': frames,
                  'pool_size': len({'A': A, 'B': B, 'C': C}[name])}
    print(name, u, vk, r['activity'], times, d['events'])
(dst / 'cases.json').write_text(json.dumps(meta))
