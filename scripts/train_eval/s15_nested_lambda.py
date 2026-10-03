import sys as _iae_sys
if '--help' in _iae_sys.argv:
    print('Usage: python %s [options]' % __file__)
    raise SystemExit(0)
"""Nested selection of the decoding cost lambda (NC), without access to the outer test folds.

For each outer fold of the main run (same activity folds as s8_cv), the outer training activities are split into
four inner folds; the evidence network is retrained on three of them (same configuration as the main model:
MLP, MIL 150 epochs + operation-level structured loss gamma=0.1 for 20 epochs) and applied to the fourth. Lambda is
chosen on these inner held-out logits by the number of NC training videos whose decoded operations equal the goal,
and then applied to the outer test logits of the main run (runs/<main>_s<seed>/scores.pkl, unchanged).
Usage: s15_nested_lambda.py --seed S [--main abl_notemporal]
"""
import argparse, json, pickle, random, sys, time
from pathlib import Path
import numpy as np
from iae.common import requests, OUT
from iae.features import candidate_labels
from iae.score import score_pairs, gold_pairs
from iae.decode import decode, decode_constrained, op_loss

GRID = [-2, -1, -0.5, 0, 0.5, 1, 2]


def main():
    import torch, torch.nn.functional as F
    from iae.model import EvidenceNet
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--main', default='abl_notemporal')
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    reqs = requests(); cache = OUT / 'feat_cache_v3'
    feats = {vk: pickle.loads((cache / f'{vk}.pkl').read_bytes()) for vk in sorted({r['video_key'] for r in reqs}) if (cache / f'{vk}.pkl').exists()}
    reqs = [r for r in reqs if r['video_key'] in feats]
    acts = sorted({r['activity_id'] for r in reqs}); random.Random(0).shuffle(acts); fold = {x: i % 5 for i, x in enumerate(acts)}
    vid = {}
    for r in reqs: vid.setdefault(r['video_key'], r)
    G = {vk: gold_pairs(r['activity_id']) for vk, r in vid.items()}
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    def tensors(vk):
        f = feats[vk]; r = vid[vk]
        y, w = candidate_labels(f['cands'], G[vk], r['task'], r['instruction'], r['objects'])
        task = torch.tensor([1.0, 0.0] if r['task'] == 'Nonverbal_Cue' else [0.0, 1.0])
        return torch.tensor(f['X']).to(dev), torch.tensor(f['M']).to(dev), task.to(dev), torch.tensor(y).to(dev), torch.tensor(w).to(dev)

    T = {vk: tensors(vk) for vk in vid}

    def train_apply(train, apply_to):
        net = EvidenceNet(nf=next(iter(T.values()))[0].shape[-1], hidden=48).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        pos = sum(float((T[v][3] * T[v][4]).sum()) for v in train); neg = sum(float(((1 - T[v][3]) * T[v][4]).sum()) for v in train)
        pw = torch.tensor(neg / max(pos, 1)).to(dev)
        for ep in range(170):
            opt.zero_grad(); loss = 0.0; sl = 0.0
            for vk in train:
                X, M, task, y, w = T[vk]; z, s = net(X, M, task)
                loss = loss + F.binary_cross_entropy_with_logits(s, y, weight=w, pos_weight=pw)
                if ep >= 150 and vid[vk]['task'] == 'Nonverbal_Cue':
                    zn = z.detach().cpu().numpy(); cands = feats[vk]['cands']
                    gc = decode_constrained(zn, cands, G[vk])
                    if gc is not None:
                        pr = decode(zn, cands, augment=G[vk], aug_mode='pair')
                        sg = sum(z[t, kk] for t, kk in gc['idx']); sp = sum(z[t, kk] for t, kk in pr['idx']) if pr['idx'] else z.sum() * 0
                        sl = sl + F.relu(sp - sg + float(op_loss(pr['pairs'], G[vk])))
            ((loss + 0.1 * sl) / len(train)).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            return {vk: net(*T[vk][:3])[0].cpu().numpy() for vk in apply_to}

    outer = pickle.load(open(OUT / 'runs' / f'{a.main}_s{a.seed}' / 'scores.pkl', 'rb'))
    rows, chosen = [], {}; t0 = time.time()
    for k in range(5):
        tr_acts = sorted({vid[v]['activity_id'] for v in vid if fold[vid[v]['activity_id']] != k})
        random.Random(100 + k).shuffle(tr_acts); inner = {x: i % 4 for i, x in enumerate(tr_acts)}
        zin = {}
        for j in range(4):
            tr = [v for v in vid if vid[v]['activity_id'] in inner and inner[vid[v]['activity_id']] != j]
            va = [v for v in vid if vid[v]['activity_id'] in inner and inner[vid[v]['activity_id']] == j and vid[v]['task'] == 'Nonverbal_Cue']
            zin.update(train_apply(tr, va))
        score = {lam: sum(sorted((p['object_id'], p['destination_region']) for p in decode(zin[v], feats[v]['cands'], cost=lam)['pairs']) == G[v]
                          for v in zin) for lam in GRID}
        best = max(GRID, key=lambda l: (score[l], -abs(l)))           # ties broken towards lambda = 0
        chosen[k] = {'lambda': best, 'inner_exact': score}
        print(f'fold {k} chosen lambda {best} inner exact {score} {time.time() - t0:.0f}s', flush=True)
        for r in reqs:
            if fold[r['activity_id']] != k or r['task'] != 'Nonverbal_Cue' or r['video_key'] not in outer: continue
            pairs = decode(outer[r['video_key']]['z'], feats[r['video_key']]['cands'], cost=best)['pairs']
            s = score_pairs(r, pairs)
            rows.append({'uid': r['uid'], 'task': 'No', 'pairs': pairs, 'succ': s['success'], 'strict': s['strict']})
    out = OUT / 'runs' / f'nested_lambda_s{a.seed}'; out.mkdir(parents=True, exist_ok=True)
    # RD requests are unaffected by lambda: copy them from the main run so that the run covers all 455 requests
    for x in json.load(open(OUT / 'runs' / f'{a.main}_s{a.seed}' / 'rows.json')):
        if x['task'] != 'No': rows.append(x)
    (out / 'rows.json').write_text(json.dumps(rows, default=str))
    (out / 'config.json').write_text(json.dumps({'seed': a.seed, 'main': a.main, 'chosen': chosen, 'grid': GRID}))
    nc = [x for x in rows if x['task'] == 'No']
    print('NC SR %.1f strict %.1f (n=%d)' % (100 * np.mean([x['succ'] for x in nc]) * len(nc) / 195, 100 * np.mean([x['strict'] for x in nc]) * len(nc) / 195, len(nc)))


if __name__ == '__main__':
    main()
