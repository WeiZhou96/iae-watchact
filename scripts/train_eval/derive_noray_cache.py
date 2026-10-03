"""Feature cache without the forearm and head rays (Table 3, "- arm/head rays").

The 28-d hand-candidate features end with the 6 ray features (iae.features, NF = 22 + 6). The ablation uses the
first 22 dimensions of the same cache; every other field is copied unchanged.
Usage: derive_noray_cache.py [src=feat_cache_v3] [dst=feat_cache_v2]
"""
import pickle, sys
from iae.common import OUT

src = OUT / (sys.argv[1] if len(sys.argv) > 1 else 'feat_cache_v3')
dst = OUT / (sys.argv[2] if len(sys.argv) > 2 else 'feat_cache_v2'); dst.mkdir(exist_ok=True)
n = 0
for p in sorted(src.glob('*.pkl')):
    f = pickle.loads(p.read_bytes())
    if f is not None:
        f = dict(f); f['X'] = f['X'][..., :22]
    (dst / p.name).write_bytes(pickle.dumps(f)); n += 1
print(n, 'videos ->', dst)
