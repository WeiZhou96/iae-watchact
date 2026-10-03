import sys as _iae_sys
if '--help' in _iae_sys.argv:
    print('Usage: python %s [options]' % __file__)
    raise SystemExit(0)
"""Stage 8: activity-grouped 5-fold CV of the evidence network with MIL and program-level structured losses.

Training uses task goals of training folds only (MIL candidate labels; structured hinge between the best
gold-consistent programme and the loss-augmented best programme). Per fold, the decoding cost is chosen on
training videos. Held-out predictions are decoded goal-free and scored with the official scorer.
Usage: s8_cv.py --run NAME [--struct W] [--epochs E] [--struct-epochs S] [--seed S] [--direct DIR ...]
"""
import argparse, collections, json, pickle, random, sys, time
from pathlib import Path
import numpy as np
from iae.common import requests, OUT
from iae.features import build, candidate_labels
from iae.score import score_pairs, gold_pairs
from iae.decode import decode, decode_constrained, op_loss
from iae.program_v1 import rd_program_v1

COSTS = [-4, -3, -2, -1, 0, 1]


def load_features(vks, cache_name='feat_cache'):
    cache = OUT / cache_name; cache.mkdir(exist_ok=True); out = {}; static = cache_name.endswith('_static')
    for vk in vks:
        p = cache / f'{vk}.pkl'
        if p.exists(): out[vk] = pickle.loads(p.read_bytes()); continue
        f = build(vk, static=static, tracks='tracks_fwd' if cache_name.endswith('_fwd') else 'tracks_s32' if cache_name.endswith('_s32') else 'tracks')
        if f is not None and cache_name.startswith('feat_cache_v2'): f['X'] = f['X'][..., :22]  # v2 = features without arm/head rays
        if f is not None: p.write_bytes(pickle.dumps(f)); out[vk] = f
    return out


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s); pos = s[y > 0.5]; neg = s[y < 0.5]
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def main():
    import torch
    import torch.nn.functional as F
    from iae.model import EvidenceNet, EvidenceNetT
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--struct', type=float, default=0.0)
    ap.add_argument('--epochs', type=int, default=150); ap.add_argument('--struct-epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--direct', type=Path, action='append', default=[])
    ap.add_argument('--hidden', type=int, default=48); ap.add_argument('--fixed-cost', type=float, default=None)
    ap.add_argument('--folds', choices=['random', 'blocked'], default='random')
    ap.add_argument('--nc-program', choices=['decode', 'threshold'], default='decode')
    ap.add_argument('--cache', default='feat_cache'); ap.add_argument('--arch', choices=['mlp', 'temporal', 'temporal_compete'], default='mlp')
    ap.add_argument('--train-frac', type=float, default=1.0)  # fraction of training activities kept per fold and task
    ap.add_argument('--topk', type=int, default=5)  # frames pooled into the video score
    ap.add_argument('--test-cache', default=None)  # evaluate held-out folds on another (e.g. perturbed) feature cache
    ap.add_argument('--aug', choices=['event', 'hamming', 'pair'], default='event')  # task loss of the structured hinge
    ap.add_argument('--subsample', type=int, default=0)  # >0: every frame holds the features of the nearest of N uniform frames
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    reqs = requests(); feats = load_features(sorted({r['video_key'] for r in reqs}), a.cache)
    feats_te = load_features(sorted(feats), a.test_cache) if a.test_cache else feats
    if a.subsample > 0:
        def hold(f):
            T = f['X'].shape[0]; keep = np.round(np.linspace(0, T - 1, a.subsample)).astype(int)
            near = keep[np.abs(np.arange(T)[:, None] - keep[None, :]).argmin(1)]
            return {**f, 'X': f['X'][near], 'M': f['M'][near]}
        feats = {vk: hold(f) for vk, f in feats.items()}
        feats_te = feats if not a.test_cache else {vk: hold(f) for vk, f in feats_te.items()}
    reqs = [r for r in reqs if r['video_key'] in feats]
    if a.folds == 'blocked':
        # recording-session blocks: order activities by their front-view video number, cut into 5 contiguous blocks per task
        fold = {}
        for task in ('Nonverbal_Cue', 'Reference_Disambiguation'):
            fv = {}
            for r in reqs:
                if r['task'] == task and r['view'] == 'front': fv[r['activity_id']] = r['video_key']
            order = sorted(fv, key=lambda x: fv[x]); n = len(order)
            for i, x in enumerate(order): fold[x] = min(4, i * 5 // n)
    else:
        acts = sorted({r['activity_id'] for r in reqs}); random.Random(0).shuffle(acts); fold = {x: i % 5 for i, x in enumerate(acts)}
    vid = {}
    for r in reqs: vid.setdefault(r['video_key'], r)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    G = {vk: gold_pairs(r['activity_id']) for vk, r in vid.items()}

    def tensors(vk, src=None):
        f = (src or feats)[vk]; r = vid[vk]
        y, w = candidate_labels(f['cands'], G[vk], r['task'], r['instruction'], r['objects'])
        task = torch.tensor([1.0, 0.0] if r['task'] == 'Nonverbal_Cue' else [0.0, 1.0])
        return torch.tensor(f['X']).to(dev), torch.tensor(f['M']).to(dev), task.to(dev), torch.tensor(y).to(dev), torch.tensor(w).to(dev)

    T = {vk: tensors(vk) for vk in vid}
    T_te = {vk: tensors(vk, feats_te) for vk in vid if vk in feats_te} if a.test_cache else T
    scores, fold_cost, infeasible = {}, {}, {}
    t0 = time.time()
    for k in range(5):
        train = [vk for vk, r in vid.items() if fold[r['activity_id']] != k]; test = [vk for vk, r in vid.items() if fold[r['activity_id']] == k]
        if a.train_frac < 1.0:
            rs = random.Random(1000 * a.seed + k); keep = set()
            for task in ('Nonverbal_Cue', 'Reference_Disambiguation'):
                ta = sorted({vid[v]['activity_id'] for v in train if vid[v]['task'] == task}); rs.shuffle(ta)
                keep |= set(ta[:max(1, round(a.train_frac * len(ta)))])
            train = [v for v in train if vid[v]['activity_id'] in keep]
        nf = next(iter(T.values()))[0].shape[-1]
        net = (EvidenceNet(nf=nf, hidden=a.hidden) if a.arch == 'mlp' else EvidenceNetT(nf=nf, hidden=a.hidden, compete=a.arch == 'temporal_compete')).to(dev); net.topk = a.topk; opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        pos = sum(float((T[v][3] * T[v][4]).sum()) for v in train); neg = sum(float(((1 - T[v][3]) * T[v][4]).sum()) for v in train)
        pw = torch.tensor(neg / max(pos, 1)).to(dev)
        total = a.epochs + (a.struct_epochs if a.struct > 0 else 0)
        n_inf = n_nc = 0
        for ep in range(total):
            opt.zero_grad(); loss = 0.0; sl = 0.0
            for vk in train:
                X, M, task, y, w = T[vk]; z, s = net(X, M, task)
                loss = loss + F.binary_cross_entropy_with_logits(s, y, weight=w, pos_weight=pw)
                if a.struct > 0 and ep >= a.epochs and vid[vk]['task'] == 'Nonverbal_Cue':
                    zn = z.detach().cpu().numpy(); cands = feats[vk]['cands']
                    gc = decode_constrained(zn, cands, G[vk])
                    if ep == a.epochs: n_nc += 1; n_inf += gc is None
                    if gc is not None:
                        pr = decode(zn, cands, augment=G[vk], aug_mode=a.aug, gold_idx=gc['idx'])
                        sg = sum(z[t, kk] for t, kk in gc['idx']); sp = sum(z[t, kk] for t, kk in pr['idx']) if pr['idx'] else z.sum() * 0
                        if a.aug == 'pair': margin = float(op_loss(pr['pairs'], G[vk]))
                        else: margin = sum(1.0 for t, kk in pr['idx'] if (t, kk) not in set(gc['idx']))
                        h = F.relu(sp - sg + margin); sl = sl + h
            obj = (loss + a.struct * sl) / len(train)
            obj.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            for vk in train + [v for v in test if v in T_te]:
                X, M, task, y, w = (T[vk] if vk in train else T_te[vk]); z, s = net(X, M, task)
                scores.setdefault(('tr', k) if vk in train else 'te', {})[vk] = {'z': z.cpu().numpy(), 's': s.cpu().numpy(), 'y': y.cpu().numpy(), 'w': w.cpu().numpy()}
        # choose decoding cost on training NC videos (exact pair match)
        best = None
        for c in COSTS:
            ok = 0
            for vk in train:
                if vid[vk]['task'] != 'Nonverbal_Cue': continue
                d = decode(scores[('tr', k)][vk]['z'], feats[vk]['cands'], cost=c)
                ok += sorted((p['object_id'], p['destination_region']) for p in d['pairs']) == G[vk]
            if best is None or ok > best[1]: best = (c, ok)
        fold_cost[k] = best[0] if a.fixed_cost is None else a.fixed_cost
        infeasible[k] = [n_inf, n_nc]
        print(f'fold {k} loss {float(loss) / len(train):.4f} struct {float(sl) if a.struct else 0:.2f} cost {best} infeasible {infeasible[k]} {time.time() - t0:.0f}s', flush=True)
    te = scores['te']
    for task in ['Nonverbal_Cue', 'Reference_Disambiguation']:
        ys, ss = [], []
        for vk, d in te.items():
            if vid[vk]['task'] != task: continue
            m = d['w'] > 0; ys += list(d['y'][m]); ss += list(d['s'][m])
        print(task[:2], 'held-out candidate AUC', round(auc(ys, ss), 3), 'n', len(ys))
    direct = {}
    for dd in a.direct:
        name = dd.name; direct[name] = {}
        for f in dd.glob('pred_*.jsonl'):
            for s in open(f):
                x = json.loads(s); direct[name][x['uid']] = x.get('pairs') if x['status'] == 'ok' else None
    rows = []; dec_cache = {}
    for r in reqs:
        vk = r['video_key']
        if vk not in te: continue  # no perturbed features (registration failed): counted as failure by the analysis
        d = te[vk]; f = feats_te[vk]
        if r['task'] == 'Nonverbal_Cue':
            if vk not in dec_cache:
                if a.nc_program == 'decode': dec_cache[vk] = decode(d['z'], f['cands'], cost=fold_cost[fold[r['activity_id']]])
                else:
                    from iae.program_v1 import nc_program_v1
                    q = nc_program_v1(f, d); q['score'] = q['confidence']; dec_cache[vk] = q
            prog = dec_cache[vk]; conf = prog['score']
        else:
            prog = rd_program_v1(f, d, r); conf = prog['confidence']
        s = score_pairs(r, prog['pairs'])
        row = {'uid': r['uid'], 'task': r['task'][:2], 'view': r['view'], 'ref': r['reference'], 'activity': r['activity_id'],
               'succ': s['success'], 'prog': s['progress'], 'fail': s['failure'], 'conf': conf, 'valid': prog['valid'], 'pairs': prog['pairs']}
        for name, dm in direct.items():
            if r['uid'] in dm:
                ds = score_pairs(r, dm[r['uid']] or []); row[f'{name}_succ'] = ds['success']; row[f'{name}_prog'] = ds['progress']
        rows.append(row)
    out = OUT / 'runs' / a.run; out.mkdir(parents=True, exist_ok=True)
    (out / 'rows.json').write_text(json.dumps(rows, default=str)); pickle.dump(te, open(out / 'scores.pkl', 'wb'))
    (out / 'config.json').write_text(json.dumps({**vars(a), 'direct': [str(x) for x in a.direct], 'fold_cost': fold_cost, 'infeasible': infeasible}))
    agg = collections.defaultdict(collections.Counter)
    for x in rows:
        for key in [(x['task'],), (x['task'], x['view'])]:
            g = agg[key]; g['n'] += 1; g['succ'] += x['succ']
            for name in direct:
                if f'{name}_succ' in x: g[name] += x[f'{name}_succ']
    for key in sorted(agg): print(key, dict(agg[key]))


if __name__ == '__main__':
    main()
