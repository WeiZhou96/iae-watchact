# -*- coding: utf-8 -*-
import sys as _iae_sys
if '--help' in _iae_sys.argv:
    print('Usage: python %s [options]' % __file__)
    raise SystemExit(0)
"""Numbers for the camera/reference table and the paired-comparison table (from data/pairs_all.json).

Paired comparison: per-activity mean difference of IAE (ensemble) minus a baseline over the 455 NC/RD requests;
95% interval from 5,000 activity-cluster bootstrap resamples; two-sided sign test over activities with a
non-zero difference. Writes data/tables_extra.json and prints LaTeX-ready rows.
"""
import collections
import json
import math
import random
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "data" / "pairs_all.json").read_text())
R = {r["uid"]: r for r in D["requests"]}
M = D["methods"]
out = {}

# camera x reference breakdown (plan SR)
cams = {}
for m in ("d32", "d8o", "iae_ens"):
    row = {}
    for t in ("No", "Re"):
        for key, f in (("front", lambda r: r["view"] == "front"), ("side", lambda r: r["view"] == "side"),
                       ("oblique", lambda r: r["view"] == "oblique"), ("camera", lambda r: r["ref"] == "camera"),
                       ("human", lambda r: r["ref"] == "human")):
            u = [k for k, r in R.items() if r["task"] == t and f(r)]
            row[f"{t}_{key}"] = (round(100 * sum(M[m][k]["succ"] for k in u) / len(u), 1), len(u))
    cams[m] = row
    print(m, {k: v[0] for k, v in row.items()})
out["camera_reference"] = cams


def paired(a, b, metric, task=None, B=5000, seed=0):
    by = collections.defaultdict(list)
    for k, r in R.items():
        if task and r["task"] != task: continue
        by[r["activity"]].append(float(M[a][k][metric]) - float(M[b][k][metric]))
    acts = list(by); vals = [x for v in by.values() for x in v]
    d = 100 * np.mean(vals)
    rng = random.Random(seed); n = len(acts); bs = []
    for _ in range(B):
        s = [x for a_ in (acts[rng.randrange(n)] for _ in range(n)) for x in by[a_]]
        bs.append(np.mean(s))
    lo, hi = 100 * np.percentile(bs, [2.5, 97.5])
    means = [np.mean(v) for v in by.values()]
    pos = int(sum(x > 0 for x in means)); neg = int(sum(x < 0 for x in means)); nz = pos + neg
    p = min(1.0, 2 * sum(math.comb(nz, k) for k in range(min(pos, neg) + 1)) / 2 ** nz) if nz else 1.0
    return dict(d=round(d, 1), lo=round(lo, 1), hi=round(hi, 1), win=pos, tie=len(means) - nz, loss=neg, p=p)


pc = {}
for b in ("d8", "d32", "d8o", "d32o", "dvl", "handcrafted", "s32t", "s32f", "s32t2"):
    pc[b] = {"SR": paired("iae_ens", b, "succ"), "strict": paired("iae_ens", b, "strict")}
    print(b, pc[b])
out["paired"] = pc
(HERE / "data" / "tables_extra.json").write_text(json.dumps(out, indent=1))
