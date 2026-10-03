"""Build a feature cache in parallel shards: build_cache.py <cache_name> <tracks_dir> <shard> <nshards>."""
import pickle, sys
from pathlib import Path
from iae.common import requests, OUT
from iae.features import build

name, tracks, shard, nsh = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
cache = OUT / name; cache.mkdir(exist_ok=True)
vks = sorted({r['video_key'] for r in requests()})
for n, vk in enumerate(vks):
    if n % nsh != shard or (cache / f'{vk}.pkl').exists(): continue
    f = build(vk, tracks=tracks)
    if f is not None: (cache / f'{vk}.pkl').write_bytes(pickle.dumps(f))
print('done', shard, flush=True)
