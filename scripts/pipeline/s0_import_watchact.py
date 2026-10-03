"""Stage 0: build the request manifest from the WatchAct release.

Reads <IAE_DATA_ROOT>/data_full/WatchAct/{data,videos} (the Hugging Face release of WatchAct) and writes
<IAE_DATA_ROOT>/manifests/{public,routing}.jsonl and import_manifest.json for the two implicit-intent tasks.
Every request receives a 24-character ID derived from its task, WatchAct ID, activity, instruction and video;
these IDs are the keys of all files in release/. Goal annotations (meta_data/) are not read.
Usage: s0_import_watchact.py
"""
from iae.config import DATA_ROOT
from iae.third_party.teb.watchact import import_dataset

if __name__ == '__main__':
    r = import_dataset(DATA_ROOT / 'data_full' / 'WatchAct', DATA_ROOT / 'manifests', tasks=['Nonverbal_Cue', 'Reference_Disambiguation'])
    print(r['examples'], 'requests ->', r['routing'])
