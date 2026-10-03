import sys as _iae_sys
if '--help' in _iae_sys.argv:
    print('Usage: python %s [options]' % __file__)
    raise SystemExit(0)
"""Exhaustive check of the grammar-constrained DP (iev.decode._dp) against brute-force enumeration.

Random event sequences (up to 12 events, 1-4 objects, 1-4 destinations, random roles/times/scores) are
generated; every subsequence is enumerated, parsed with the grammar (O+ D)+ (distinct objects, at most
max_obj objects, no pending object at the end, at least one operation) and scored. The DP optimum must equal
the brute-force optimum for (a) free decoding, (b) gold-constrained decoding (operations equal G) and
(c) operation-level loss-augmented decoding, F + Delta(A, G). Also checks that the gold-constrained
programme reproduces G exactly.
Usage: test_dp_bruteforce.py [n_cases]
"""
import itertools, random, sys
from pathlib import Path
from iae.decode import _dp, _dp_differ, _to_program, dest_of, OBJ, DST


def parse(ev, sel, cands, max_obj):
    used, pending, pairs = set(), [], {}
    for i in sel:
        t, k, role, v = ev[i]
        if role == OBJ:
            if k in used or k in pending or len(used) + len(pending) >= max_obj: return None
            pending.append(k)
        else:
            if not pending: return None
            for o in pending: pairs[o] = dest_of(cands[k])
            used |= set(pending); pending = []
    if pending or not used: return None
    return pairs


def brute(ev, cands, cost, max_obj, gold=None, mode='free'):
    best = -1e9
    for r in range(1, len(ev) + 1):
        for sel in itertools.combinations(range(len(ev)), r):
            pairs = parse(ev, sel, cands, max_obj)
            if pairs is None: continue
            A = {(o, d) for o, d in pairs.items()}
            if mode == 'gold' and A != gold: continue
            s = sum(ev[i][3] - cost for i in sel)
            if mode == 'aug': s += len(A - gold) + len(gold - A) - len(gold)   # DP adds Delta without the constant |G|
            best = max(best, s)
    return best


def margin_example():
    """Events O_a:5, O_b:-1, D_d:5: exact margin 1 (a->d vs {a->d, b->d}); the removal heuristic gives 6."""
    import numpy as np
    from iae.decode import program_margin, program_margin_approx
    cands = [{'id': 'a', 'kind': 'movable'}, {'id': 'b', 'kind': 'movable'}, {'id': 'd', 'kind': 'region'}]
    z = np.full((60, 3), -20.0); z[10, 0] = 5; z[25, 1] = -1; z[40, 2] = 5
    ex, ap = program_margin(z, cands), program_margin_approx(z, cands)
    assert abs(ex - 1) < 1e-9 and abs(ap - 6) < 1e-9, (ex, ap)
    print('margin example: exact', ex, 'approx', ap)


def main(n_cases=3000):
    margin_example()
    rng = random.Random(0); checked = {'free': 0, 'gold': 0, 'aug': 0}; infeasible = 0
    for case in range(n_cases):
        no, nd = rng.randint(1, 4), rng.randint(1, 4)
        cands = [{'id': f'o{i}', 'kind': 'movable'} for i in range(no)] + [{'id': f'd{j}', 'kind': 'region'} for j in range(nd)]
        n = rng.randint(1, 12)
        ev = sorted((rng.randint(0, 200), rng.randrange(no + nd), None, round(rng.gauss(0, 2), 3)) for _ in range(n))
        ev = [(t, k, OBJ if k < no else DST, v) for t, k, _, v in ev]
        cost = rng.choice([0.0, 0.0, 1.0, -1.0]); max_obj = 3
        gobj = rng.sample(range(no), rng.randint(1, min(no, 3)))
        gold = {(o, dest_of(cands[no + rng.randrange(nd)])) for o in gobj}
        # (a) free
        s, path = _dp(ev, cands, cost, max_obj)
        b = brute(ev, cands, cost, max_obj)
        assert abs(max(s, -1e9) - b) < 1e-6 or (s < -1e8 and b < -1e8), (case, 'free', s, b)
        checked['free'] += 1
        # (a') exact margin: best programme whose operations differ from the decoded ones
        if s > -1e8:
            pairs, _ = _to_program(ev, path, cands)
            ref = {int(o[1:]): d for o, d in pairs.items()}
            A0 = set(ref.items()); bd = -1e9
            for r_ in range(1, len(ev) + 1):
                for sel in itertools.combinations(range(len(ev)), r_):
                    pp = parse(ev, sel, cands, max_obj)
                    if pp is None or set(pp.items()) == A0: continue
                    bd = max(bd, sum(ev[i][3] - cost for i in sel))
            dd = _dp_differ(ev, cands, cost, max_obj, ref)
            assert abs(max(dd, -1e9) - bd) < 1e-6 or (dd < -1e8 and bd < -1e8), (case, 'margin', dd, bd)
            checked['margin'] = checked.get('margin', 0) + 1
        # (b) gold-constrained, as in decode_constrained
        gd = {o: d for o, d in gold}; req = set(gd)
        s, path = _dp(ev, cands, cost, max_obj, allow_obj=req, gold_dest=gd, required=req)
        b = brute(ev, cands, cost, max_obj, gold, 'gold')
        assert abs(max(s, -1e9) - b) < 1e-6 or (s < -1e8 and b < -1e8), (case, 'gold', s, b)
        if s > -1e8:
            pairs, _ = _to_program(ev, path, cands)
            assert {(int(o[1:]), d) for o, d in pairs.items()} == gold, (case, 'gold-pairs')
        else:
            infeasible += 1
        checked['gold'] += 1
        # (c) operation-level loss-augmented
        s, path = _dp(ev, cands, cost, max_obj, pair_aug=gd)
        b = brute(ev, cands, cost, max_obj, gold, 'aug')
        assert abs(max(s, -1e9) - b) < 1e-6 or (s < -1e8 and b < -1e8), (case, 'aug', s, b)
        checked['aug'] += 1
    print('all optima agree', checked, 'gold-infeasible cases', infeasible)


def test_dp_bruteforce():
    main(3000)

