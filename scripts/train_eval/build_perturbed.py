import sys as _iae_sys
if '--help' in _iae_sys.argv:
    print('Usage: python %s [options]' % __file__)
    raise SystemExit(0)
"""Test-time perturbation of the public layout used for registration.

mode 'noise': Gaussian noise (std sigma, in grid cells) is added to the canonical coordinates of every public object
              used to fit the registration map; registration (assignment + map) is re-run.
mode 'randid': the fitted map is kept, but identifiers are permuted at random among detections of the same
              category (identity at chance level).
For every video the new registration is written, track labels are rewritten accordingly (masks unchanged), and the
28-d feature cache is rebuilt. Episodic programs are evaluated directly on the perturbed inputs.
Usage: build_perturbed.py <cond_name> <mode> <sigma> <seed> [workers]
"""
import collections, hashlib, json, random, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import iae.register as R
import iae.features as F
from iae.common import requests, OUT, ALL_TASKS, EPISODIC, stem
from iae.episodic import program
from iae.score import score_pairs

cond, mode, sigma, seed = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
workers = int(sys.argv[5]) if len(sys.argv) > 5 else 32
BASE = OUT / 'pert' / cond
for sub in ('register', 'tracks'): (BASE / sub).mkdir(parents=True, exist_ok=True)
CACHE = OUT / f'feat_cache_v3_{cond}'; CACHE.mkdir(exist_ok=True)
_orig_cxy = R.canonical_xy


def rng_for(*key):
    h = hashlib.sha256(('|'.join(map(str, key))).encode()).hexdigest()
    return np.random.default_rng(int(h[:12], 16))


def perturb_registration(vk, r):
    d = json.loads((OUT / 'det_final' / f'{vk}.json').read_text()); W = d['size'][0]
    reg0 = json.loads((OUT / 'register' / f'{vk}.json').read_text())
    if mode == 'noise':
        on = {s[1]: s[2] for s in r['initial_states'] if len(s) == 3 and s[0] == 'On'}
        noisy = {}
        for o, reg_name in on.items():
            q = _orig_cxy(reg_name)
            if q is not None:
                noisy[reg_name + '#' + o] = tuple(np.array(q) + rng_for(vk, o, seed).normal(0, sigma, 2))
        # register() looks up canonical_xy(on[o]); give each object its own noisy coordinate
        states = [[s[0], s[1], s[2] + '#' + s[1]] if len(s) == 3 and s[0] == 'On' and _orig_cxy(s[2]) is not None else s
                  for s in r['initial_states']]
        R.canonical_xy = lambda name: noisy.get(name) if name in noisy else _orig_cxy(name)
        try:
            reg = R.register(d['instances'], r['objects'], states, W, view=r['view'])
        finally:
            R.canonical_xy = _orig_cxy
    else:
        reg = dict(reg0)
        if reg0.get('status') == 'ok':
            g = random.Random(int(hashlib.sha256(f'{vk}|{seed}'.encode()).hexdigest()[:12], 16))
            by = collections.defaultdict(list)
            for o, i in reg0['assignment'].items(): by[stem(o)].append((o, i))
            asg = {}
            for c, lst in by.items():
                ids = [o for o, _ in lst]; dets = [i for _, i in lst]; g.shuffle(dets)
                asg.update(dict(zip(ids, dets)))
            reg['assignment'] = asg
    reg.update(video_key=vk, activity_id=r['activity_id'], view=r['view'], instances=d['instances'])
    return reg


def work(item):
    vk, r = item
    try:
        reg = perturb_registration(vk, r)
        (BASE / 'register' / f'{vk}.json').write_text(json.dumps(reg))
        tp = OUT / 'tracks' / f'{vk}.json'
        if not tp.exists(): return vk, 'no_track'
        tr = json.loads(tp.read_text())
        lab = {i: o for o, i in reg.get('assignment', {}).items()}
        tr['labels'] = [lab.get(j) for j in range(len(tr['labels']))]
        (BASE / 'tracks' / f'{vk}.json').write_text(json.dumps(tr))
        if r['task'] in ('Nonverbal_Cue', 'Reference_Disambiguation'):
            orig = F._load
            F._load = lambda kind, v: (json.loads((BASE / kind / f'{v}.json').read_text())
                                       if kind in ('register', 'tracks') else orig(kind, v))
            try:
                f = F.build(vk)
            finally:
                F._load = orig
            if f is not None:
                import pickle
                (CACHE / f'{vk}.pkl').write_bytes(pickle.dumps(f))
        same = reg.get('assignment') == json.loads((OUT / 'register' / f'{vk}.json').read_text()).get('assignment')
        return vk, 'same' if same else 'changed'
    except Exception as e:
        return vk, 'error:' + str(e)[:80]


if __name__ == '__main__':
    vids = {}
    for r in requests(ALL_TASKS): vids.setdefault(r['video_key'], r)
    with Pool(workers) as p:
        res = p.map(work, sorted(vids.items()), chunksize=2)
    st = collections.Counter(x.split(':')[0] for _, x in res)
    print(cond, 'videos', len(res), dict(st), flush=True)
    # episodic programs on the perturbed inputs
    agg = collections.Counter(); rows = []
    for r in requests(EPISODIC):
        vk = r['video_key']; tp, rp = BASE / 'tracks' / f'{vk}.json', BASE / 'register' / f'{vk}.json'
        pr = program(r['task'], json.loads(tp.read_text()), json.loads(rp.read_text()), r['objects']) if tp.exists() and rp.exists() else {'pairs': []}
        s = score_pairs(r, pr['pairs'], max_pairs=8)
        agg['n'] += 1; agg['succ'] += s['success']; agg['strict'] += s['strict']
        rows.append({'uid': r['uid'], 'task': r['task'][:3], 'activity': r['activity_id'], 'succ': s['success'], 'strict': s['strict']})
    out = OUT / 'runs' / f'episodic_{cond}'; out.mkdir(parents=True, exist_ok=True); (out / 'rows.json').write_text(json.dumps(rows))
    print(cond, 'episodic SR %.1f strict %.1f' % (100 * agg['succ'] / agg['n'], 100 * agg['strict'] / agg['n']), flush=True)
