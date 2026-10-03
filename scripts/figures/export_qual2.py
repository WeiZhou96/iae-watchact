"""Export real data for the second qualitative figure (episodic cases + registration/tracking). Selection rules fixed
in advance, random choice with seed 0 inside each group.

A1  episodic success: Restore or Reversal, front camera, >= 2 goal pairs, IAE strictly correct, 32B direct wrong.
A2  episodic failure: any episodic task, front camera, IAE wrong, and a goal involves a container
    (a goal destination is a container interior or an initial place is a container interior).
B   registration/tracking: an RD activity with >= 2 movable instances of one category, registration status ok on all
    three cameras, and a same-category instance that moves by > 60 px in the front-camera track.
Success for A uses the scorer with up to 8 pairs, as in Table IV.
"""
import json, random, shutil, sys
from pathlib import Path
import numpy as np
from iae.common import requests, OUT, EPISODIC, frames_dir, load_frame_index, stem
from iae.score import score_pairs, gold_pairs
from iae.episodic import img_to_canon
from iae.register import canonical_xy

dst = Path(sys.argv[1]); dst.mkdir(parents=True, exist_ok=True)
RE = {r['uid']: r for r in requests(EPISODIC)}
rows = {x['uid']: x for x in json.load(open(OUT / 'runs/episodic_v1/rows.json'))}
d32 = {}
for f in (OUT / 'direct_32b_episodic').glob('pred_*.jsonl'):
    for s in open(f):
        x = json.loads(s); d32[x['uid']] = (x.get('pairs') if x['status'] == 'ok' else None) or []
ok = {u: score_pairs(RE[u], rows[u]['pairs'] or [], max_pairs=8) for u in rows}
ok32 = {u: score_pairs(RE[u], d32.get(u, []), max_pairs=8)['success'] for u in rows}
has_track = lambda vk: (OUT / 'tracks' / f'{vk}.json').exists()

A1 = sorted(u for u, r in rows.items() if r['task'] in ('Res', 'Rev') and r['view'] == 'front' and ok[u]['strict']
            and not ok32[u] and len(gold_pairs(r['activity'])) >= 2 and has_track(RE[u]['video_key']))
def involves_container(u):
    g = gold_pairs(rows[u]['activity'])
    init = {s[1]: s[2] for s in RE[u]['initial_states'] if len(s) == 3}
    return any(d.endswith('_contain_region') or str(init.get(o, '')).endswith('_contain_region') for o, d in g)
A2 = sorted(u for u, r in rows.items() if r['view'] == 'front' and not ok[u]['success'] and involves_container(u)
            and has_track(RE[u]['video_key']))

RD = [r for r in requests() if r['task'] == 'Reference_Disambiguation']
by_act = {}
for r in RD: by_act.setdefault(r['activity_id'], {})[r['view']] = r['video_key']
Bpool = []
for act, views in sorted(by_act.items()):
    if set(views) != {'front', 'side', 'oblique'}: continue
    regs = {v: json.loads((OUT / 'register' / f'{vk}.json').read_text()) for v, vk in views.items()
            if (OUT / 'register' / f'{vk}.json').exists()}
    if len(regs) < 3 or any(x.get('status') != 'ok' for x in regs.values()): continue
    tr = json.loads((OUT / 'tracks' / f"{views['front']}.json").read_text()) if has_track(views['front']) else None
    if tr is None: continue
    cats = [c for c in tr['cats'] if c not in ('basket', 'wooden_tray', 'wooden_cabinet')]
    dup = {c for c in cats if cats.count(c) >= 2}
    if not dup: continue
    moved = False
    for j, (lab, cat) in enumerate(zip(tr['labels'], tr['cats'])):
        if cat not in dup: continue
        cs = [p[j]['c'] for p in tr['per_frame'] if p[j] is not None]
        if cs and max(np.hypot(c[0] - cs[-1][0], c[1] - cs[-1][1]) for c in cs) > 60: moved = True
    if moved: Bpool.append(act)
print('pool sizes', len(A1), len(A2), len(Bpool))
rng = random.Random(0)
pick = {'A1': rng.choice(A1), 'A2': rng.choice(A2), 'B': rng.choice(Bpool)}


def export_frame(vk, t, out, tag):
    idx = load_frame_index(vk); fi = idx['frames'][t]
    shutil.copy(frames_dir(vk) / fi['file'], out / f'{tag}.jpg')
    npz = np.load(OUT / 'tracks' / f'{vk}.npz'); shape = tuple(npz['shape'])
    m = np.unpackbits(npz['masks'][t], axis=-1)[..., :shape[-1]]
    np.save(out / f'{tag}_mask.npy', np.packbits(m.astype(bool), axis=-1))
    return {'t': t, 'sec': fi['t'], 'file': f'{tag}.jpg', 'mask': f'{tag}_mask.npy', 'mask_shape': list(m.shape)}


meta = {}
for name in ('A1', 'A2'):
    u = pick[name]; r = rows[u]; req = RE[u]; vk = req['video_key']
    out = dst / name; out.mkdir(exist_ok=True)
    tr = json.loads((OUT / 'tracks' / f'{vk}.json').read_text()); T = len(tr['per_frame'])
    fr = [export_frame(vk, 0, out, 'first'), export_frame(vk, T - 1, out, 'last')]
    for f_, t in zip(fr, (0, T - 1)): f_['boxes'] = tr['per_frame'][t]
    meta[name] = {'uid': u, 'task': r['task'], 'activity': r['activity'], 'video_key': vk,
                  'instruction': req.get('instruction'), 'labels': tr['labels'], 'cats': tr['cats'], 'size': tr['size'],
                  'anchor': 'first' if r['task'] == 'Imi' else 'last', 'frames': fr, 'moved': r['moved'],
                  'iae': r['pairs'], 'gold': gold_pairs(r['activity']), 'd32': d32.get(u, []),
                  'iae_succ': bool(ok[u]['success']), 'iae_strict': bool(ok[u]['strict']), 'd32_succ': bool(ok32[u]),
                  'initial_states': req['initial_states'], 'pool_size': len(A1 if name == 'A1' else A2)}
    print(name, u, vk, r['moved'], 'gold', meta[name]['gold'], 'iae_ok', ok[u]['success'])

act = pick['B']; views = by_act[act]; outB = dst / 'B'; outB.mkdir(exist_ok=True); mb = {'activity': act, 'views': {}}
req0 = next(r for r in RD if r['activity_id'] == act)
canon = {s[1]: canonical_xy(s[2]) for s in req0['initial_states'] if len(s) == 3}
for v, vk in views.items():
    reg = json.loads((OUT / 'register' / f'{vk}.json').read_text()); tr = json.loads((OUT / 'tracks' / f'{vk}.json').read_text())
    T = len(tr['per_frame']); W = tr['size'][0]
    f_ = export_frame(vk, T - 1, outB, v); f_['boxes'] = tr['per_frame'][T - 1]
    pts = {}
    for j, lab in enumerate(tr['labels']):
        b = tr['per_frame'][T - 1][j]
        if lab is None or b is None: continue
        q = img_to_canon(reg, W, (b['box'][0] + b['box'][2]) / 2, b['box'][3])
        pts[lab] = [float(q[0]), float(q[1])]
    ent = {'video_key': vk, 'frame': f_, 'labels': tr['labels'], 'cats': tr['cats'], 'size': tr['size'],
           'registered_ground_points': pts, 'angle_deg': reg.get('angle_deg'), 'residual': reg.get('residual')}
    if v == 'front':
        idx = load_frame_index(vk)
        ent['tsec'] = [x['t'] for x in idx['frames']]
        ent['centroids'] = [[(p[j]['c'] if p[j] is not None else None) for p in tr['per_frame']] for j in range(len(tr['labels']))]
        cats = tr['cats']; dup = [c for c in cats if cats.count(c) >= 2 and c not in ('basket', 'wooden_tray', 'wooden_cabinet')]
        js = [j for j, c in enumerate(cats) if c in dup]
        # frames at which a same-category instance is farthest from its final position (being handled)
        far = []
        for j in js:
            cs = [(t, p[j]['c']) for t, p in enumerate(tr['per_frame']) if p[j] is not None]
            if cs:
                t_far = max(cs, key=lambda tc: np.hypot(tc[1][0] - cs[-1][1][0], tc[1][1] - cs[-1][1][1]))[0]
                far.append(t_far)
        ent['mid_frames'] = []
        for k, t in enumerate(sorted(set(far))[:2]):
            g = export_frame(vk, t, outB, f'front_mid{k}'); g['boxes'] = tr['per_frame'][t]; ent['mid_frames'].append(g)
    mb['views'][v] = ent
mb['canonical'] = canon; mb['instruction'] = req0['instruction']; mb['pool_size'] = len(Bpool)
meta['B'] = mb
print('B', act, {v: e['video_key'] for v, e in mb['views'].items()}, 'residuals', {v: e['residual'] for v, e in mb['views'].items()})
(dst / 'qual2.json').write_text(json.dumps(meta))
