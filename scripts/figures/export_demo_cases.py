"""Select demo cases by a fixed rule and export everything the renderer needs (read-only on the pipeline outputs).

Selection (seed 2026, pools sorted by uid before shuffling):
  V1  Nonverbal_Cue, IAE ensemble success and strict success
  V2  Reference_Disambiguation, IAE ensemble success and strict success
  V3  episodic tasks (episodic_v1, the reported run), success and strict success
  V4  Nonverbal_Cue, IAE ensemble failure
The first three of each shuffled pool are exported; the released clips use the first draw of every pool.
Usage (after the pipeline and the runs ens_final / episodic_v1 exist): python scripts/figures/export_demo_cases.py
"""
import json, pickle, random, shutil
from pathlib import Path
import numpy as np

from iae.common import requests, OUT, TASKS, EPISODIC, load_frame_index, frames_dir
from iae.decode import decode
from iae.program_v1 import rd_program_v1
from iae.evidence import region_points
from iae.score import score_pairs, gold_pairs
from iae.episodic import program

DST = OUT / 'demo' / 'cases'
DST.mkdir(parents=True, exist_ok=True)
REQS = {r['uid']: r for r in requests(TASKS + EPISODIC)}
ENS = json.loads((OUT / 'runs/ens_final/rows.json').read_text())
EPI = json.loads((OUT / 'runs/episodic_v1/rows.json').read_text())
SC = pickle.load(open(OUT / 'runs/ens_final/scores.pkl', 'rb'))
CFG = json.loads((OUT / 'runs/ens_final/config.json').read_text())


def direct(d):
    p = {}
    for f in Path(d).glob('pred_*.jsonl'):
        for s in open(f):
            x = json.loads(s); p[x['uid']] = (x.get('pairs') if x['status'] == 'ok' else None) or []
    return p


D32 = direct(OUT / 'direct_32b'); D32E = direct(OUT / 'direct_32b_episodic')


def load(kind, vk):
    p = OUT / kind / f'{vk}.json'
    return json.loads(p.read_text()) if p.exists() else None


def pool(rows, cond):
    return sorted([r for r in rows if cond(r)], key=lambda r: r['uid'])


POOLS = {
    'V1_nc_success': pool(ENS, lambda r: r['task'] == 'No' and r['succ'] and r['strict']),
    'V2_rd_success': pool(ENS, lambda r: r['task'] == 'Re' and r['succ'] and r['strict']),
    'V3_episodic_success': pool(EPI, lambda r: r['succ'] and r['strict']),
    'V4_nc_failure': pool(ENS, lambda r: r['task'] == 'No' and not r['succ']),
}


def canon(pairs):
    return sorted((p['object_id'], p['destination_region']) for p in pairs or [])


def export(tag, rank, row):
    req = REQS[row['uid']]; vk = req['video_key']
    out = DST / f'{tag}_{rank}'; out.mkdir(exist_ok=True)
    idx = load_frame_index(vk); tr = load('tracks', vk); reg = load('register', vk)
    meta = {'tag': tag, 'rank': rank, 'uid': row['uid'], 'task': req['task'], 'view': req['view'], 'reference': req['reference'],
            'activity': req['activity_id'], 'instruction': req['instruction'], 'video_key': vk,
            'source_video': req['video_path'], 'frame_index': idx, 'gold': gold_pairs(req['activity_id']),
            'objects': req['objects'], 'initial_states': req['initial_states']}
    epis = req['task'] in EPISODIC
    meta['iae'] = {k: row[k] for k in ('pairs', 'succ', 'strict', 'prog', 'conf') if k in row}
    if epis:
        p = program(req['task'], tr, reg, req['objects'])
        assert canon(p['pairs']) == canon(row['pairs']), (row['uid'], p['pairs'], row['pairs'])
        meta['iae']['moved'] = p['moved']; meta['anchor'] = tr.get('anchor')
        dp = D32E.get(row['uid'], []); s = score_pairs(req, dp, max_pairs=8)
    else:
        f = pickle.load(open(OUT / 'feat_cache_v3' / f'{vk}.pkl', 'rb')); d = SC[vk]
        meta['cands'] = f['cands']; meta['z'] = np.asarray(d['z']).round(4).tolist(); meta['s'] = np.asarray(d['s']).round(4).tolist()
        if req['task'] == 'Nonverbal_Cue':
            q = decode(d['z'], f['cands'], cost=CFG['cost'])
            assert canon(q['pairs']) == canon(row['pairs']), (row['uid'], q['pairs'], row['pairs'])
            meta['iae']['events'] = [list(e) for e in q['events']]
        else:
            q = rd_program_v1(f, d, req)
            assert canon(q['pairs']) == canon(row['pairs']), (row['uid'], q['pairs'], row['pairs'])
            meta['iae']['selected'] = q['selected']; meta['iae']['dest'] = q['dest']
        dp = D32.get(row['uid'], []); s = score_pairs(req, dp)
    meta['qwen32b'] = {'pairs': dp, 'succ': s['success'], 'strict': s['strict'], 'prog': s['progress']}
    meta['region_points'] = {k: list(v) for k, v in region_points(reg, tr['size'][0]).items()} if reg else {}
    meta['register'] = reg
    for kind in ('tracks', 'hands', 'body', 'person'):
        meta[kind] = load(kind, vk)
    (out / 'meta.json').write_text(json.dumps(meta))
    shutil.copy2(OUT / 'tracks' / f'{vk}.npz', out / 'masks.npz')
    shutil.copy2(req['video_path'], out / 'source.mp4')
    n = len(idx['frames'])
    for j, t in enumerate([0, n // 3, 2 * n // 3, n - 1]):
        shutil.copy2(frames_dir(vk) / idx['frames'][t]['file'], out / f'peek_{j}.jpg')
    print(tag, rank, row['uid'], req['task'], req['view'], req['activity_id'], 'iae', row['succ'], row['strict'],
          '32b', s['success'], s['strict'], flush=True)


rng = random.Random(2026)
summary = {}
for tag, rows in POOLS.items():
    order = rows[:]; rng.shuffle(order)
    summary[tag] = {'pool_size': len(rows), 'order': [r['uid'] for r in order[:10]]}
    for rank, row in enumerate(order[:3]):
        export(tag, rank, row)
(DST / 'selection.json').write_text(json.dumps({'seed': 2026, 'rule': __doc__, 'pools': summary}, indent=1))
print(json.dumps({k: v['pool_size'] for k, v in summary.items()}))
