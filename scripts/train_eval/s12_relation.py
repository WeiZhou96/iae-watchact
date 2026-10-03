import sys as _iae_sys
if '--help' in _iae_sys.argv:
    print('Usage: python %s [options]' % __file__)
    raise SystemExit(0)
"""Same-supervision relation baseline for nonverbal cues (NC): direct object-destination pair classification.

Same inputs, folds and supervision as the evidence network: the 28-d hand-candidate features of the
registered and tracked candidates (feat_cache_v3), activity-grouped folds, and the goal operations of the
training activities as the only labels. No grammar and no decoding: a shared per-frame encoder embeds every
(hand, candidate) feature vector (max over hands); for every (movable object, destination) pair the two
embedding sequences are concatenated and passed through a dilated temporal CNN with max and mean pooling to a
pair logit (BCE, class-balanced). At test time each object takes its best destination, objects whose best
logit exceeds theta are kept (at most 3, at least 1); theta is chosen on the training videos of each fold.
Usage: s12_relation.py --run NAME --seed S [--epochs 150] [--cache feat_cache_v3]
"""
import argparse, json, pickle, random, sys, time
from pathlib import Path
import numpy as np
from iae.common import requests, OUT
from iae.score import score_pairs, gold_pairs
from iae.decode import dest_of

THETAS = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]


def main():
    import torch, torch.nn as nn, torch.nn.functional as F
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=150); ap.add_argument('--cache', default='feat_cache_v3')
    ap.add_argument('--hidden', type=int, default=48)
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    reqs = requests()
    cache = OUT / a.cache
    feats = {vk: pickle.loads((cache / f'{vk}.pkl').read_bytes()) for vk in sorted({r['video_key'] for r in reqs})
             if (cache / f'{vk}.pkl').exists()}
    reqs = [r for r in reqs if r['video_key'] in feats]
    acts = sorted({r['activity_id'] for r in reqs}); random.Random(0).shuffle(acts); fold = {x: i % 5 for i, x in enumerate(acts)}
    vid = {}
    for r in reqs:
        if r['task'] == 'Nonverbal_Cue': vid.setdefault(r['video_key'], r)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    G = {vk: gold_pairs(r['activity_id']) for vk, r in vid.items()}

    def pack(vk):
        f = feats[vk]; cands = f['cands']
        O = [k for k, c in enumerate(cands) if c['kind'] == 'movable']
        D = [k for k, c in enumerate(cands) if c['kind'] in ('container', 'region', 'drawer')]
        pairs = [(o, d) for o in O for d in D]
        gold = set(G[vk]); y = [1.0 if (cands[o]['id'], dest_of(cands[d])) in gold else 0.0 for o, d in pairs]
        return dict(X=torch.tensor(f['X']).float().to(dev), M=torch.tensor(f['M']).to(dev), pairs=pairs,
                    po=torch.tensor([o for o, _ in pairs]).to(dev), pd=torch.tensor([d for _, d in pairs]).to(dev),
                    y=torch.tensor(y).to(dev), cands=cands)

    P = {vk: pack(vk) for vk in vid}
    nf = next(iter(P.values()))['X'].shape[-1]; H = a.hidden

    class RelNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(nf, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
            self.tcn = nn.Sequential(nn.Conv1d(2 * H, 64, 9, padding=4), nn.ReLU(), nn.Conv1d(64, 64, 9, padding=8, dilation=2), nn.ReLU())
            self.head = nn.Linear(128, 1)

        def forward(self, d):
            e = self.enc(d['X'])                                         # T, Hh, K, H
            e = e.masked_fill(~d['M'][:, :, None, None], -1e4).max(1).values
            e = torch.where(e > -1e3, e, torch.zeros_like(e))            # frames without a hand -> 0
            seq = torch.cat([e[:, d['po']], e[:, d['pd']]], -1).permute(1, 2, 0)   # P, 2H, T
            h = self.tcn(seq)
            return self.head(torch.cat([h.max(-1).values, h.mean(-1)], -1)).squeeze(-1)

    def program(d, logit, theta):
        best = {}
        for (o, dd), v in zip(d['pairs'], logit):
            if o not in best or v > best[o][0]: best[o] = (v, dd)
        sel = sorted(((v, o, dd) for o, (v, dd) in best.items()), reverse=True)
        keep = [x for x in sel if x[0] > theta][:3] or sel[:1]
        return [{'object_id': d['cands'][o]['id'], 'destination_region': dest_of(d['cands'][dd])} for v, o, dd in keep]

    rows = []; t0 = time.time(); chosen = {}
    for k in range(5):
        train = [vk for vk, r in vid.items() if fold[r['activity_id']] != k]; test = [vk for vk, r in vid.items() if fold[r['activity_id']] == k]
        net = RelNet().to(dev); opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        pos = sum(float(P[v]['y'].sum()) for v in train); neg = sum(float((1 - P[v]['y']).sum()) for v in train)
        pw = torch.tensor(neg / max(pos, 1)).to(dev)
        for ep in range(a.epochs):
            opt.zero_grad(); loss = 0.0
            for vk in train:
                loss = loss + F.binary_cross_entropy_with_logits(net(P[vk]), P[vk]['y'], pos_weight=pw)
            (loss / len(train)).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            L = {vk: net(P[vk]).cpu().numpy() for vk in train + test}
        best = None
        for th in THETAS:
            ok = sum(sorted((p['object_id'], p['destination_region']) for p in program(P[vk], L[vk], th)) == G[vk] for vk in train)
            if best is None or ok > best[1]: best = (th, ok)
        chosen[k] = best
        for r in reqs:
            if r['task'] != 'Nonverbal_Cue' or fold[r['activity_id']] != k: continue
            pairs = program(P[r['video_key']], L[r['video_key']], best[0]); s = score_pairs(r, pairs)
            rows.append({'uid': r['uid'], 'task': 'No', 'view': r['view'], 'activity': r['activity_id'], 'pairs': pairs,
                         'succ': s['success'], 'strict': s['strict'], 'prog': s['progress']})
        print(f'fold {k} loss {float(loss) / len(train):.4f} theta {best} {time.time() - t0:.0f}s', flush=True)
    out = OUT / 'runs' / a.run; out.mkdir(parents=True, exist_ok=True)
    (out / 'rows.json').write_text(json.dumps(rows, default=str))
    (out / 'config.json').write_text(json.dumps({**vars(a), 'theta': chosen}))
    n = len(rows)
    print(f"NC n={n} SR={100 * np.mean([x['succ'] for x in rows]):.1f} strict={100 * np.mean([x['strict'] for x in rows]):.1f}")


if __name__ == '__main__':
    main()
