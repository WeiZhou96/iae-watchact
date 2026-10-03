# -*- coding: utf-8 -*-
"""95% activity-cluster bootstrap intervals of overall plan SR and strict success for every Table 1 method
(release/predictions/pairs_all.json; the single-model row averages the three seeds per request).
Writes release/figure_data/table_ci.json."""
import collections
import json
import random
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PRED = REPO / "release" / "predictions"      # released per-request predictions and scores
FDATA = REPO / "release" / "figure_data"     # released summaries used by the figures
D = json.loads((PRED / "pairs_all.json").read_text())
R = D["requests"]; M = D["methods"]


def per_request(name):
    if name == "iae_single":
        return {r["uid"]: (np.mean([M[f"iae_s{s}"][r["uid"]]["succ"] for s in range(3)]),
                           np.mean([M[f"iae_s{s}"][r["uid"]]["strict"] for s in range(3)])) for r in R}
    return {r["uid"]: (float(M[name][r["uid"]]["succ"]), float(M[name][r["uid"]]["strict"])) for r in R}


def ci(vals, k, B=5000):
    by = collections.defaultdict(list)
    for r in R: by[r["activity"]].append(vals[r["uid"]][k])
    acts = list(by); rng = random.Random(0); n = len(acts)
    bs = [np.mean([x for a in (acts[rng.randrange(n)] for _ in range(n)) for x in by[a]]) for _ in range(B)]
    return [round(100 * float(np.mean([v[k] for v in vals.values()])), 1)] + [round(100 * float(x), 1) for x in np.percentile(bs, [2.5, 97.5])]


out = {}
for name in ("d8", "d32", "d8o", "d32o", "dvl", "s32t", "s32f", "s32t2", "handcrafted", "iae_single", "iae_ens"):
    v = per_request(name)
    out[name] = {"SR": ci(v, 0), "strict": ci(v, 1)}
    print(name, out[name])
(FDATA / "table_ci.json").write_text(json.dumps(out, indent=1))
