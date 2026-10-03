"""Shared paths and public-input loading for the v2 interaction-evidence pipeline.

Only public fields are exposed by `requests()`; goals are loaded exclusively by `goals()` for scoring.
"""
from __future__ import annotations
import json, re, os
from pathlib import Path
from .config import DATA_ROOT, OUT_ROOT, WATCHACT_ROOT, WATCHACT_SCRIPTS, TEB_ROOT

DATA = DATA_ROOT
OUT = OUT_ROOT
REPO = Path(os.environ.get('IAE_SOURCE_REPO', '.')).expanduser().resolve()
TEB_SRC = TEB_ROOT
TASKS = ('Nonverbal_Cue', 'Reference_Disambiguation')
EPISODIC = ('Restore_Previous_State', 'Reversal', 'Imitation')  # manipulation-tracking tasks (no learning)
FULL = DATA / 'data_full/WatchAct'
ALL_TASKS = TASKS + EPISODIC
INITIAL_ANCHOR = ('Imitation',)  # tasks executed from the initial scene of the video
FRAME_FPS = 10.0
FRAME_WIDTH = 1280


def stem(public_id: str) -> str:
    return re.sub(r'_\d+$', '', public_id)


def video_key(path: str) -> str:
    p = Path(path)
    return f'{p.parts[-3][:3]}_{p.parts[-2]}_{p.stem}'


def _meta():
    m = {}
    for t in TASKS + EPISODIC:
        for e in json.loads((FULL / f'meta_data/{t}.json').read_text())['examples']:
            m[e['id']] = (t, e['simulation_task'])
    return m


def requests(tasks=TASKS):
    """All requests with public fields only: uid, task, activity, view, reference, instruction, video, objects, initial states."""
    meta = _meta(); rows = []
    for t in tasks:
        if t not in EPISODIC: continue
        for s in open(FULL / f'data/{t}.jsonl'):
            r = json.loads(s); _, st = meta[r['original_id']]; vp = str(FULL / 'videos' / t / r['video'])
            rows.append({'uid': r['id'], 'task': t, 'activity_id': r['original_id'], 'view': r['camera_perspective'],
                         'reference': r['spatial_reference'], 'instruction': r['language_instruction'], 'video_path': vp,
                         'video_key': video_key(vp), 'objects': st['objects'], 'initial_states': st['initial_states']})
    for s in open(DATA / 'manifests/routing.jsonl'):
        r = json.loads(s)
        if r['task'] not in tasks: continue
        _, st = meta[r['original_id']]
        rows.append({'uid': r['uid'], 'task': r['task'], 'activity_id': r['original_id'], 'view': r['camera_perspective'],
                     'reference': r['spatial_reference'], 'instruction': r['language_instruction'], 'video_path': r['video_path'],
                     'video_key': video_key(r['video_path']), 'objects': st['objects'], 'initial_states': st['initial_states']})
    return rows


def goals():
    """Evaluation only."""
    return {k: v[1] for k, v in _meta().items()}


def frames_dir(vkey: str) -> Path:
    return OUT / 'frames10' / vkey


def load_frame_index(vkey: str):
    return json.loads((frames_dir(vkey) / 'index.json').read_text())
