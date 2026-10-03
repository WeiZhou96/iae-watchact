"""Recompute the reported results from the released per-request files only (no data, models or simulator needed).

Checks Table 1 (plan success and strict success per task, single model and ensemble), Table 2 (paired differences of
the ensemble to every baseline: mean, activities better/equal/worse, sign test), Table 3 (seed-averaged ablations and
controls, and their differences to IAE) and Table 4 (episodic tasks). With --ci, the 95% activity-cluster bootstrap
intervals of Tables 1 and 2 are recomputed as well (5,000 resamples, seed 0, as in scripts/figures/table_ci.py and
tables_extra.py). Exits with status 1 if any number differs from the paper.
Usage: python scripts/analysis/check_release.py [--ci]
"""
import collections, json, math, random, sys
from pathlib import Path
import numpy as np

REL = Path(__file__).resolve().parents[2] / 'release'
PA = json.loads((REL / 'predictions/pairs_all.json').read_text())
R = PA['requests']; M = PA['methods']
NC = [r['uid'] for r in R if r['task'] == 'No']; RD = [r['uid'] for r in R if r['task'] == 'Re']; ALL = NC + RD
fails = []


def check(label, got, want, tol=0.051):
    ok = want is None or abs(got - want) < tol
    if not ok: fails.append(f'{label}: got {got}, paper {want}')
    return f'{got:5.1f}' + ('' if ok else ' (!)')


def pct(vals): return round(100 * float(np.mean(vals)), 1)


def per_request(name):
    if name == 'iae_single':
        return {u: (np.mean([M[f'iae_s{s}'][u]['succ'] for s in range(3)]), np.mean([M[f'iae_s{s}'][u]['strict'] for s in range(3)])) for u in ALL}
    return {u: (float(M[name][u]['succ']), float(M[name][u]['strict'])) for u in ALL}


def boot(by, B=5000):
    acts = list(by); rng = random.Random(0); n = len(acts)
    bs = [np.mean([x for a in (acts[rng.randrange(n)] for _ in range(n)) for x in by[a]]) for _ in range(B)]
    return [round(100 * float(x), 1) for x in np.percentile(bs, [2.5, 97.5])]


# ---------------------------------------------------------------- Table 1
T1 = {  # NC SR, NC strict, RD SR, RD strict, All SR [CI], All strict [CI]
    'd8': (16.9, 4.1, 27.7, 20.8, 23.1, (16.3, 30.5), 13.6, (7.9, 20.0)),
    'd32': (22.6, 4.6, 31.2, 23.5, 27.5, (20.9, 34.5), 15.4, (9.9, 21.1)),
    'd8o': (25.6, 7.2, 34.6, 27.3, 30.8, (23.1, 38.9), 18.7, (12.3, 25.3)),
    'd32o': (21.5, 3.1, 34.2, 24.2, 28.8, (22.0, 36.0), 15.2, (9.7, 21.1)),
    'dvl': (10.8, 4.1, 34.6, 17.3, 24.4, (18.5, 31.0), 11.6, (7.3, 16.5)),
    's32t': (37.4, 17.4, 25.4, 20.8, 30.5, (23.1, 38.5), 19.3, (13.0, 26.2)),
    's32f': (29.7, 12.8, 26.2, 18.8, 27.7, (20.9, 34.7), 16.3, (11.0, 22.0)),
    's32t2': (35.9, 12.3, 73.5, 68.5, 57.4, (49.0, 65.3), 44.4, (35.8, 52.7)),
    'handcrafted': (2.6, 2.6, 75.4, 70.0, 44.2, (34.9, 53.6), 41.1, (31.4, 50.5)),
    'iae_single': (37.6, 15.7, 78.7, 73.2, 61.1, (52.5, 69.2), 48.6, (39.8, 57.7)),
    'iae_ens': (42.6, 16.9, 80.4, 74.2, 64.2, (55.4, 72.5), 49.7, (40.4, 58.9)),
}
NAMES = {'d8': 'Qwen3-VL-8B, direct', 'd32': 'Qwen3-VL-32B, direct', 'd8o': 'Qwen3-VL-8B + overlays', 'd32o': 'Qwen3-VL-32B + overlays',
         'dvl': 'InternVL3.5-8B, direct', 's32t': 'Qwen3-VL-32B + evidence', 's32f': '  + frames', 's32t2': '  explained',
         'handcrafted': 'Hand-crafted evidence', 'iae_single': 'IAE, single model', 'iae_ens': 'IAE, three-seed ensemble'}
CI = '--ci' in sys.argv
print('Table 1  (455 NC/RD requests, %)        NC SR  strict  RD SR  strict  All SR  strict' + ('   CI SR         CI strict' if CI else ''))
for k, w in T1.items():
    v = per_request(k)
    cells = [check(f'T1 {k} NC SR', pct([v[u][0] for u in NC]), w[0]), check(f'T1 {k} NC strict', pct([v[u][1] for u in NC]), w[1]),
             check(f'T1 {k} RD SR', pct([v[u][0] for u in RD]), w[2]), check(f'T1 {k} RD strict', pct([v[u][1] for u in RD]), w[3]),
             check(f'T1 {k} All SR', pct([v[u][0] for u in ALL]), w[4]), check(f'T1 {k} All strict', pct([v[u][1] for u in ALL]), w[6])]
    line = f'  {NAMES[k]:28s} ' + '  '.join(cells)
    if CI:
        for j, want in ((0, w[5]), (1, w[7])):
            by = collections.defaultdict(list)
            for r in R: by[r['activity']].append(v[r['uid']][j])
            lo, hi = boot(by)
            if abs(lo - want[0]) > 0.051 or abs(hi - want[1]) > 0.051: fails.append(f'T1 {k} CI {j}: [{lo}, {hi}] vs {want}')
            line += f'   [{lo:4.1f}, {hi:4.1f}]'
    print(line)

# ---------------------------------------------------------------- Table 2
T2 = {  # SR diff [CI], better/equal/worse activities, strict diff [CI]
    'd8': (41.1, (31.2, 50.5), (53, 31, 7), 36.0, (26.6, 45.3)), 'd32': (36.7, (27.0, 45.9), (53, 27, 11), 34.3, (25.7, 43.1)),
    'd8o': (33.4, (24.2, 42.6), (46, 39, 6), 31.0, (22.0, 40.2)), 'd32o': (35.4, (26.4, 44.2), (55, 28, 8), 34.5, (25.9, 42.9)),
    'dvl': (39.8, (31.0, 48.1), (59, 27, 5), 38.0, (29.0, 46.8)), 's32t': (33.6, (23.7, 43.5), (49, 31, 11), 30.3, (20.4, 40.7)),
    's32f': (36.5, (27.0, 45.9), (51, 30, 10), 33.4, (24.0, 43.1)), 's32t2': (6.8, (1.3, 12.7), (27, 56, 8), 5.3, (0.0, 11.0)),
    'handcrafted': (20.0, (12.5, 27.9), (26, 61, 4), 8.6, (3.1, 14.7)),
}
print('\nTable 2  (IAE ensemble minus baseline, overall)   dSR    better/equal/worse  sign test p   dStrict')
for k, w in T2.items():
    res = []
    for metric in ('succ', 'strict'):
        by = collections.defaultdict(list)
        for r in R: by[r['activity']].append(float(M['iae_ens'][r['uid']][metric]) - float(M[k][r['uid']][metric]))
        means = [np.mean(x) for x in by.values()]; pos = int(sum(m > 0 for m in means)); neg = int(sum(m < 0 for m in means)); nz = pos + neg
        p = min(1.0, 2 * sum(math.comb(nz, j) for j in range(min(pos, neg) + 1)) / 2 ** nz) if nz else 1.0
        res.append((pct([x for v in by.values() for x in v]), (pos, len(means) - nz, neg), p, boot(by) if CI else None))
    (d, wtl, p, ci), (ds, _, _, cis) = res
    cells = check(f'T2 {k} dSR', d, w[0]) + f'   {wtl[0]:3d}/{wtl[1]:3d}/{wtl[2]:3d}' + ('' if wtl == w[2] else ' (!)')
    if wtl != w[2]: fails.append(f'T2 {k} activities {wtl} vs {w[2]}')
    line = f'  {NAMES[k]:28s}  {cells}       {p:9.2g}   ' + check(f'T2 {k} dStrict', ds, w[3])
    if CI:
        for got, want, lab in ((ci, w[1], 'SR'), (cis, w[4], 'strict')):
            if abs(got[0] - want[0]) > 0.051 or abs(got[1] - want[1]) > 0.051: fails.append(f'T2 {k} CI {lab}: {got} vs {want}')
        line += f'   SR [{ci[0]:4.1f}, {ci[1]:4.1f}]  strict [{cis[0]:4.1f}, {cis[1]:4.1f}]'
    print(line)

# ---------------------------------------------------------------- Table 3
AB = json.loads((REL / 'predictions/ablations.json').read_text())
T3 = {  # NC SR, NC strict, RD strict, All SR, All strict, delta All SR, delta All strict
    'iae': (37.6, 15.7, 73.2, 61.1, 48.6, None, None), 'no_rays': (36.1, 14.0, 74.2, 60.4, 48.4, -0.7, -0.1),
    'temporal_conv': (33.5, 14.7, 72.7, 59.2, 47.8, -1.9, -0.7), 'no_program_loss': (37.3, 12.3, 73.6, 60.7, 47.3, -0.4, -1.2),
    'event_loss': (37.3, 18.1, 73.6, 61.2, 49.8, 0.1, 1.2), 'no_grammar': (21.2, 8.9, 73.2, 54.1, 45.6, -7.0, -2.9),
    'no_backward_tracking': (34.2, 13.3, 70.1, 57.2, 45.8, -3.9, -2.8), 'forward_tracking': (31.1, 14.2, 71.2, 56.9, 46.7, -4.2, -1.8),
    'no_learning': (2.6, 2.6, 70.0, 44.2, 41.1, -16.9, -7.5), 'frames32_evidence': (35.2, 11.6, 71.7, 57.9, 45.9, -3.2, -2.6),
    'frames32_all_stages': (29.2, 10.9, 69.9, 55.3, 44.6, -5.8, -4.0), 'blocked_folds': (32.3, 14.0, 70.8, 56.0, 46.4, -5.1, -2.1),
    'relation_model': (19.7, 7.9, None, None, None, None, None),
}


def seed_avg(key):
    pre = AB['runs'][key]; runs = [r for r in AB['seeds'] if r == pre or r.startswith(pre + '_s')]
    out = {}
    for u in ALL:
        out[u] = (np.mean([AB['seeds'][r][u]['succ'] if u in AB['seeds'][r] else 0.0 for r in runs]),
                  np.mean([AB['seeds'][r][u]['strict'] if u in AB['seeds'][r] else 0.0 for r in runs]))
    return out


base = seed_avg('iae')
print('\nTable 3  (seed-averaged, %)        NC SR  strict  RD strict  All SR  strict   dAll SR  dAll strict')
for k, w in T3.items():
    v = seed_avg(k)
    cells = [check(f'T3 {k} NC SR', pct([v[u][0] for u in NC]), w[0]), check(f'T3 {k} NC strict', pct([v[u][1] for u in NC]), w[1])]
    if w[2] is not None:
        cells += [check(f'T3 {k} RD strict', pct([v[u][1] for u in RD]), w[2]), check(f'T3 {k} All SR', pct([v[u][0] for u in ALL]), w[3]),
                  check(f'T3 {k} All strict', pct([v[u][1] for u in ALL]), w[4])]
        if w[5] is not None:
            cells += [check(f'T3 {k} dSR', pct([v[u][0] - base[u][0] for u in ALL]), w[5]),
                      check(f'T3 {k} dStrict', pct([v[u][1] - base[u][1] for u in ALL]), w[6])]
    print(f'  {k:22s} ' + '   '.join(cells))

# ---------------------------------------------------------------- Table 4
EP = json.loads((REL / 'predictions/episodic.json').read_text())
REQ = {r['uid']: r for r in json.loads((REL / 'requests.json').read_text())}
T4 = {'d8': (37.6, 0.4, 5.8, 15.4, 14.1), 'd32': (37.6, 16.5, 26.7, 27.0, 24.8), 'dvl': (31.7, 9.8, 8.9, 17.5, 15.6), 'iae': (46.6, 46.3, 46.2, 46.4, 46.4)}
print('\nTable 4  (800 episodic requests, %)   Imit.  Restore  Reversal   All   strict')
for k, w in T4.items():
    rows = EP[k]; by = lambda t: [rows[u]['succ'] for u in rows if REQ[u]['task'] == t]
    cells = [check(f'T4 {k} imit', pct(by('Imitation')), w[0]), check(f'T4 {k} restore', pct(by('Restore_Previous_State')), w[1]),
             check(f'T4 {k} reversal', pct(by('Reversal')), w[2]), check(f'T4 {k} all', pct([x['succ'] for x in rows.values()]), w[3]),
             check(f'T4 {k} strict', pct([x['strict'] for x in rows.values()]), w[4])]
    print(f'  {k:6s}                          ' + '   '.join(cells))

print('\nall numbers match the paper' if not fails else '\nMISMATCHES:\n  ' + '\n  '.join(fails))
sys.exit(1 if fails else 0)
