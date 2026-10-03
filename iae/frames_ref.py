"""Request-frame <-> canonical region remapping, using the official WatchAct mapping tables."""
import json
from functools import lru_cache
from .common import DATA

CFG = DATA / 'external/WatchAct/video_planning/config'


@lru_cache(None)
def _tables():
    d = json.loads((CFG / 'human2camera_reference.json').read_text())
    return {k: dict(v) if not isinstance(v, dict) else v for k, v in d.items()}


@lru_cache(None)
def region_descriptions():
    return json.loads((CFG / 'region_descriptions.json').read_text())


def _to_request_table(view, ref):
    t = _tables()
    if ref == 'human': return t['camera2human_reference']
    if ref == 'camera' and view == 'side': return t['camera2leftCamera_reference']
    return {}


def to_request(region, view, ref):
    return _to_request_table(view, ref).get(region, region)


def to_canonical(region, view, ref):
    inv = {v: k for k, v in _to_request_table(view, ref).items()}
    return inv.get(region, region)


def request_states(states, view, ref):
    return [[*s[:-1], to_request(s[-1], view, ref)] if len(s) == 3 else list(s) for s in states]


def region_text(view, ref):
    d = region_descriptions()
    key = 'ALL_REGION_DESCRIPTIONS_LEFT_CAMERA' if (ref == 'camera' and view == 'side') else \
          'ALL_REGION_DESCRIPTIONS_HUMAN' if ref == 'human' else 'ALL_REGION_DESCRIPTIONS_CAMERA'
    return d[key]
