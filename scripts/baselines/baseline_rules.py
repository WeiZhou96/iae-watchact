"""Hand-crafted evidence baselines on the same features (no learning), written as a run for ensemble.py scoring.

NC: frame score = log pointing-pose - angular and distance penalties (max over hands); same grammar decoding.
RD: selected score = maximum displacement of the instance from its final position (motion heuristic).
Usage: baseline_rules.py <out_run> <cache>
"""
import json, math, pickle, sys
from pathlib import Path
import numpy as np
from iae.common import requests, OUT

name, cache = sys.argv[1], sys.argv[2]
task_of = {r['video_key']: r['task'] for r in requests()}
scores = {}
for p in (OUT / cache).glob('*.pkl'):
    f = pickle.loads(p.read_bytes()); X, M = f['X'], f['M']; vk = f['video_key']
    st, ext, ang, dist = X[..., 0], X[..., 1], X[..., 4] * 90, X[..., 5]
    pose = 1 / (1 + np.exp(-12 * (st - 0.8))) / (1 + np.exp(-6 * (ext - 1.0)))
    g = np.log(pose + 1e-6) - (ang / 22.0) ** 2 / 2 - dist / 0.35
    g = np.where(M[:, :, None], g, -30.0); z = g.max(1)
    if task_of.get(vk) == 'Reference_Disambiguation':
        disp = X[..., 18].max(1)  # [T,K]
        s = disp.max(0); z = np.repeat(s[None], X.shape[0], 0)
    else:
        s = np.sort(z, 0)[-5:].mean(0)
    scores[vk] = {'z': z.astype(np.float32), 's': s.astype(np.float32)}
out = OUT / 'runs' / name; out.mkdir(parents=True, exist_ok=True)
pickle.dump(scores, open(out / 'scores.pkl', 'wb')); (out / 'config.json').write_text(json.dumps({'cache': cache, 'rule': True}))
print(len(scores))
