"""Export the public evaluation artifacts of the paper (read-only on the pipeline outputs).

Writes to <IAE_OUT_ROOT>/release_build/ (copy its content into release/):
  requests.json                 all 1,255 requests: uid, task, WatchAct example id, activity, view, reference, video, gold
  predictions/pairs_all.json    NC/RD (455): predicted pairs, success and strict success of every Table 1 method
  predictions/ablations.json    NC/RD: every Table 3 variant and seed
  predictions/sensitivity.json  NC/RD: the training-fraction, top-k and gamma runs (Fig. sensitivity) and layout runs (Table 6)
  predictions/episodic.json     episodic (800): IAE and the direct VLM baselines (Table 4)
  predictions/iae_ensemble_rows.json   ensemble rows incl. confidence (Fig. selective prediction)
  predictions/direct_rows.json         per-request flags of the direct baselines (Fig. selective prediction)
"""
import json, shutil, sys
from pathlib import Path
from iae.common import requests, OUT, TASKS, EPISODIC, DATA
from iae.score import score_pairs, gold_pairs

DST = OUT / 'release_build'; (DST / 'predictions').mkdir(parents=True, exist_ok=True)
NCRD = requests(TASKS); EPI = requests(EPISODIC)
ROUTE = {json.loads(s)['uid']: json.loads(s) for s in open(DATA / 'manifests/routing.jsonl')}
EPI_ROWS = {}
for t in EPISODIC:
    for s in open(DATA / 'data_full/WatchAct/data' / f'{t}.jsonl'):
        r = json.loads(s); EPI_ROWS[r['id']] = r

reqs = []
for r in NCRD:
    ro = ROUTE[r['uid']]
    reqs.append({'uid': r['uid'], 'task': r['task'], 'watchact_id': ro['id'], 'activity': r['activity_id'], 'view': r['view'],
                 'reference': r['reference'], 'video': f"{r['task']}/{ro['video']}", 'gold': gold_pairs(r['activity_id'])})
for r in EPI:
    e = EPI_ROWS[r['uid']]
    reqs.append({'uid': r['uid'], 'task': r['task'], 'watchact_id': e['id'], 'activity': r['activity_id'], 'view': r['view'],
                 'reference': r['reference'], 'video': f"{r['task']}/{e['video']}", 'gold': gold_pairs(r['activity_id'])})
(DST / 'requests.json').write_text(json.dumps(reqs))
print('requests', len(reqs))

_cache = {}


def scored(req_list, P, max_pairs=3):
    rows = {}
    for r in req_list:
        pairs = P.get(r['uid']) or []
        key = (r['uid'], json.dumps(pairs, sort_keys=True), max_pairs)
        if key not in _cache:
            s = score_pairs(r, pairs, max_pairs=max_pairs); _cache[key] = (bool(s['success']), bool(s['strict']), s['failure'] is None)
        a, b, v = _cache[key]
        rows[r['uid']] = {'pred': [(p['object_id'], p['destination_region']) for p in pairs if isinstance(p, dict)], 'succ': a, 'strict': b, 'valid': v}
    return rows


def from_run(run):
    return {x['uid']: x['pairs'] for x in json.load(open(OUT / 'runs' / run / 'rows.json'))}


def from_direct(d):
    p = {}
    for f in (OUT / d).glob('pred_*.jsonl'):
        for s in open(f):
            x = json.loads(s); p[x['uid']] = (x.get('pairs') if x['status'] == 'ok' else None) or []
    return p


def summary(rows, req_list):
    n = len(req_list); return round(100 * sum(rows[r['uid']]['succ'] for r in req_list) / n, 1), round(100 * sum(rows[r['uid']]['strict'] for r in req_list) / n, 1)


# Table 1 / Table 2 (same keys and format as the figure and table scripts expect)
SRC = {'iae_ens': ('run', 'ens_final'), 'iae_s0': ('run', 'abl_notemporal_s0'), 'iae_s1': ('run', 'abl_notemporal_s1'),
       'iae_s2': ('run', 'abl_notemporal_s2'), 'handcrafted': ('run', 'rule_heuristic_eval_c0'),
       'd8': ('dir', 'direct_8b'), 'd32': ('dir', 'direct_32b'), 'd8o': ('dir', 'direct_8b_overlay'), 'd32o': ('dir', 'direct_32b_overlay'),
       'dvl': ('dir', 'direct_internvl8b'), 's32t': ('dir', 'struct32b_text'), 's32f': ('dir', 'struct32b_frames'), 's32t2': ('dir', 'struct32b_text_v2')}
pa = {'requests': [{'uid': r['uid'], 'task': r['task'][:2], 'view': r['view'], 'ref': r['reference'], 'activity': r['activity_id'],
                    'gold': gold_pairs(r['activity_id'])} for r in NCRD], 'methods': {}}
for name, (kind, x) in SRC.items():
    rows = scored(NCRD, from_run(x) if kind == 'run' else from_direct(x)); pa['methods'][name] = rows
    print('table1', name, summary(rows, NCRD))
(DST / 'predictions/pairs_all.json').write_text(json.dumps(pa))

# Table 3: ablations and controls (all seeds)
ABL = {'iae': 'abl_notemporal', 'no_rays': 'abl_final', 'temporal_conv': 'abl_full', 'no_program_loss': 'abl_m_nostruct',
       'event_loss': 'rev_ham', 'no_grammar': 'abl_m_nogrammar', 'no_backward_tracking': 'abl_m_notrack', 'forward_tracking': 'rev_fwd',
       'no_learning': 'rule_heuristic_eval_c0', 'frames32_evidence': 'rev_sub32', 'frames32_all_stages': 'rev_s32full',
       'blocked_folds': 'abl_m_blocked', 'relation_model': 'rel'}
abl = {'runs': ABL, 'seeds': {}}
for key, pre in ABL.items():
    runs = [pre] if (OUT / 'runs' / pre / 'rows.json').exists() else [f'{pre}_s{s}' for s in (0, 1, 2)]
    for run in runs:
        P = from_run(run); req = [r for r in NCRD if r['uid'] in P] if key == 'relation_model' else NCRD
        abl['seeds'][run] = scored(req, P)
    print('table3', key, runs)
(DST / 'predictions/ablations.json').write_text(json.dumps(abl))

# Fig. sensitivity and Table 6 (layout perturbation)
sens = {}
for d in sorted((OUT / 'runs').iterdir()):
    if d.name.startswith(('sens_', 'layout_', 'nested_lambda_')) and (d / 'rows.json').exists():
        sens[d.name] = scored(NCRD, from_run(d.name))
(DST / 'predictions/sensitivity.json').write_text(json.dumps(sens)); print('sensitivity runs', len(sens))

# Table 4: episodic
EP = {'iae': ('run', 'episodic_v1'), 'd8': ('dir', 'direct_8b_episodic'), 'd32': ('dir', 'direct_32b_episodic'), 'dvl': ('dir', 'direct_internvl8b_episodic')}
ep = {}
for name, (kind, x) in EP.items():
    rows = scored(EPI, from_run(x) if kind == 'run' else from_direct(x), max_pairs=8); ep[name] = rows
    print('table4', name, summary(rows, EPI), {t[:3]: summary(rows, [r for r in EPI if r['task'] == t])[0] for t in EPISODIC})
for name, pre in {'iae_noise025': 'episodic_noise025', 'iae_noise050': 'episodic_noise050', 'iae_noise100': 'episodic_noise100', 'iae_randid': 'episodic_randid'}.items():
    for s in (0, 1, 2):  # perturbed-layout runs store only the official flags
        rows = json.load(open(OUT / 'runs' / f'{pre}_s{s}' / 'rows.json'))
        ep[f'{name}_s{s}'] = {x['uid']: {'succ': bool(x['succ']), 'strict': bool(x['strict'])} for x in rows}
(DST / 'predictions/episodic.json').write_text(json.dumps(ep))

# rows used by the selective-prediction figure
shutil.copy2(OUT / 'runs/ens_final/rows.json', DST / 'predictions/iae_ensemble_rows.json')
dr = {}
for name in ['direct_8b', 'direct_32b', 'direct_8b_overlay', 'direct_32b_overlay']:
    P = from_direct(name); rows = scored(NCRD, P)
    dr[name] = [{'uid': r['uid'], 'task': r['task'][:2], 'view': r['view'], 'activity': r['activity_id'],
                 'succ': rows[r['uid']]['succ'], 'strict': rows[r['uid']]['strict']} for r in NCRD]
(DST / 'predictions/direct_rows.json').write_text(json.dumps(dr))
print('done', len(_cache))
