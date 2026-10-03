"""Grammar-constrained temporal decoding of deictic programs.

Candidate events are temporal peaks of each candidate's frame logits. A dynamic programme over time-sorted
events selects the subsequence maximising summed evidence minus a per-event cost, subject to the deictic
grammar (O+ D)+, distinct objects and at most `max_obj` objects.

`decode_constrained` restricts programmes to those whose operations equal a set of gold pairs (training only):
objects are limited to the gold objects, every gold object must be used, and a destination event may only
close pending objects whose gold destination it is.
`decode(..., augment=gold, aug_mode=...)` is loss-augmented inference (training only):
  'pair'    operation-level loss  Delta(A, G) = |A \\ G| + |G \\ A|, added at destination events (decomposable,
            since the pending objects are part of the DP state); used by the final model;
  'hamming' event-level Hamming loss to the gold-consistent programme (unit bonus for events outside it);
  'event'   unit bonus for events whose candidate is not a gold object / gold destination (earlier variant).
"""
from __future__ import annotations
import numpy as np

OBJ, DST = 'o', 'd'


def dest_of(c):
    return c['id'] + '_contain_region' if c['kind'] == 'container' else c['id']


def candidate_events(z, cands, per_cand=3, drop=3.0, sep=10):
    ev = []
    for k, c in enumerate(cands):
        role = OBJ if c['kind'] == 'movable' else DST if c['kind'] in ('container', 'region', 'drawer') else None
        if role is None: continue
        zk = z[:, k]; top = zk.max(); picked = []
        for t in np.argsort(-zk):
            if zk[t] < top - drop or len(picked) >= per_cand: break
            if zk[t] < zk[max(0, t - 2):t + 3].max(): continue
            if all(abs(t - u) >= sep for u in picked): picked.append(int(t))
        for t in picked: ev.append((t, k, role, float(zk[t])))
    ev.sort()
    return ev


def _dp(ev, cands, cost, max_obj, allow_obj=None, gold_dest=None, required=None, bonus=None, pair_aug=None):
    n = len(ev); memo = {}

    def f(i, used, pending):
        key = (i, used, pending)
        if key in memo: return memo[key]
        if i == n:
            ok = not pending and used and (required is None or set(used) == required)
            res = (0.0, ()) if ok else (-1e9, ())
            memo[key] = res; return res
        best = f(i + 1, used, pending)
        t, k, role, v = ev[i]
        v = v - cost + (bonus[i] if bonus is not None else 0.0)
        if role == OBJ and k not in used and k not in pending and len(used) + len(pending) < max_obj and (allow_obj is None or k in allow_obj):
            s, p = f(i + 1, used, tuple(sorted(pending + (k,))))
            if s + v > best[0]: best = (s + v, (i,) + p)
        if role == DST and pending:
            ok = gold_dest is None or all(gold_dest.get(o) == dest_of(cands[k]) for o in pending)
            if ok:
                w = v
                if pair_aug is not None:      # +1 per wrong pair, -1 per correct pair (the constant |G| is added outside)
                    d = dest_of(cands[k]); w += sum(-1.0 if pair_aug.get(o) == d else 1.0 for o in pending)
                s, p = f(i + 1, tuple(sorted(used + pending)), ())
                if s + w > best[0]: best = (s + w, (i,) + p)
        memo[key] = best; return best

    return f(0, (), ())


def _to_program(ev, path, cands):
    pairs = {}; pending = []; chosen = [ev[i] for i in path]
    for t, k, role, v in chosen:
        if role == OBJ: pending.append(k)
        else:
            for o in pending: pairs[cands[o]['id']] = dest_of(cands[k])
            pending = []
    return pairs, chosen


def op_loss(pairs, gold):
    """Delta(A, G) = |A \\ G| + |G \\ A| over (object, destination) operations."""
    A = {(p['object_id'], p['destination_region']) for p in pairs}; G = {(o, d) for o, d in gold}
    return len(A - G) + len(G - A)


def decode(z, cands, cost=0.0, max_obj=3, per_cand=3, augment=None, aug_mode='event', gold_idx=None):
    ev = candidate_events(z, cands, per_cand)
    bonus = None; pair_aug = None
    if augment is not None:
        if aug_mode == 'event':
            gobj = {o for o, _ in augment}; gdst = {d for _, d in augment}
            bonus = [0.0 if ((role == OBJ and cands[k]['id'] in gobj) or (role == DST and dest_of(cands[k]) in gdst)) else 1.0
                     for t, k, role, v in ev]
        elif aug_mode == 'hamming':
            gi = set(gold_idx or [])
            bonus = [0.0 if (t, k) in gi else 1.0 for t, k, role, v in ev]
        elif aug_mode == 'pair':
            idx = {c['id']: k for k, c in enumerate(cands)}
            pair_aug = {idx[o]: d for o, d in augment if o in idx}
        else:
            raise ValueError(aug_mode)
    score, path = _dp(ev, cands, cost, max_obj, bonus=bonus, pair_aug=pair_aug)
    if score < -1e8: return {'pairs': [], 'score': score, 'events': [], 'valid': False, 'idx': []}
    pairs, chosen = _to_program(ev, path, cands)
    return {'pairs': [{'object_id': o, 'destination_region': d} for o, d in pairs.items()], 'score': score, 'valid': bool(pairs),
            'events': [(t, cands[k]['id'], role, round(v, 2)) for t, k, role, v in chosen], 'idx': [(t, k) for t, k, _, _ in chosen]}


def decode_constrained(z, cands, gold_pairs, cost=0.0, max_obj=3, per_cand=3):
    ev = candidate_events(z, cands, per_cand)
    idx = {c['id']: k for k, c in enumerate(cands)}
    gold = {o: d for o, d in gold_pairs}
    if any(o not in idx for o in gold): return None
    required = {idx[o] for o in gold}
    gd = {idx[o]: d for o, d in gold.items()}
    score, path = _dp(ev, cands, cost, max_obj, allow_obj=required, gold_dest=gd, required=required)
    if score < -1e8: return None
    return {'score': score, 'idx': [(ev[i][0], ev[i][1]) for i in path]}


def decode_oracle(z, cands, gold_pairs, oracle='objects', cost=0.0, max_obj=3, per_cand=3):
    """Diagnostics only: decode with the gold object set (all of it must be used) or with destination events
    limited to the gold destinations; the rest of the programme is chosen by the evidence."""
    ev = candidate_events(z, cands, per_cand)
    idx = {c['id']: k for k, c in enumerate(cands)}
    if oracle == 'objects':
        req = {idx[o] for o, _ in gold_pairs if o in idx}
        score, path = _dp(ev, cands, cost, max_obj, allow_obj=req, required=req)
    else:
        gd = {d for _, d in gold_pairs}
        ev = [e for e in ev if e[2] == OBJ or dest_of(cands[e[1]]) in gd]
        score, path = _dp(ev, cands, cost, max_obj)
    if score < -1e8: return {'pairs': [], 'valid': False}
    pairs, _ = _to_program(ev, path, cands)
    return {'pairs': [{'object_id': o, 'destination_region': d} for o, d in pairs.items()], 'valid': bool(pairs)}


def _dp_differ(ev, cands, cost, max_obj, ref):
    """Best score over programmes whose operations differ from `ref` (dict object index -> destination).
    The state is extended by a flag that records whether a pair outside `ref` has been produced; since objects
    are distinct, a programme reproduces `ref` exactly iff the flag is unset and the assigned objects equal ref."""
    n = len(ev); memo = {}; target = frozenset(ref)

    def f(i, used, pending, wrong):
        key = (i, used, pending, wrong)
        if key in memo: return memo[key]
        if i == n:
            ok = not pending and used and (wrong or frozenset(used) != target)
            memo[key] = 0.0 if ok else -1e9; return memo[key]
        best = f(i + 1, used, pending, wrong)
        t, k, role, v = ev[i]; v = v - cost
        if role == OBJ and k not in used and k not in pending and len(used) + len(pending) < max_obj:
            best = max(best, v + f(i + 1, used, tuple(sorted(pending + (k,))), wrong))
        if role == DST and pending:
            d = dest_of(cands[k]); w2 = wrong or any(ref.get(o) != d for o in pending)
            best = max(best, v + f(i + 1, tuple(sorted(used + pending)), (), w2))
        memo[key] = best; return best

    return f(0, (), (), False)


def program_margin(z, cands, cost=0.0, max_obj=3, per_cand=3):
    """Exact margin of the decoded programme (Eq. 14): best score minus the best score of any programme whose
    set of operations differs; F(omega*) if no such programme exists."""
    ev = candidate_events(z, cands, per_cand)
    best_s, path = _dp(ev, cands, cost, max_obj)
    if best_s < -1e8: return 0.0
    best_pairs, _ = _to_program(ev, path, cands)
    idx = {c['id']: k for k, c in enumerate(cands)}
    alt = _dp_differ(ev, cands, cost, max_obj, {idx[o]: d for o, d in best_pairs.items()})
    return float(best_s - alt) if alt > -1e8 else float(best_s)


def program_margin_approx(z, cands, cost=0.0, max_obj=3, per_cand=3):
    """Earlier approximation of the margin: re-decode with one chosen event removed at a time (can overestimate)."""
    ev = candidate_events(z, cands, per_cand)
    best_s, path = _dp(ev, cands, cost, max_obj)
    if best_s < -1e8: return 0.0
    best_pairs, _ = _to_program(ev, path, cands); alt = -1e9
    for i in path:
        bonus = [0.0] * len(ev); bonus[i] = -1e6
        s, p = _dp(ev, cands, cost, max_obj, bonus=bonus)
        if s < -1e8: continue
        pr, _ = _to_program(ev, p, cands)
        if pr != best_pairs: alt = max(alt, s)
    return float(best_s - alt) if alt > -1e8 else float(best_s)
