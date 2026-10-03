"""Correct destinations (request level, NC) and NC SR/strict for a run and its VLM-verified variants.
Usage: verify_stats.py <src_run> <verified_run> ..."""
import json, sys
from iae.common import requests, OUT
from iae.score import score_pairs, gold_pairs

NC = [r for r in requests() if r['task'] == 'Nonverbal_Cue']
for run in sys.argv[1:]:
    P = {x['uid']: x['pairs'] for x in json.load(open(OUT / 'runs' / run / 'rows.json'))}
    sel = ok = succ = strict = 0
    for r in NC:
        pairs = P.get(r['uid']) or []; pm = {p['object_id']: p['destination_region'] for p in pairs}
        for o, d in gold_pairs(r['activity_id']):
            if o in pm: sel += 1; ok += pm[o] == d
        s = score_pairs(r, pairs); succ += s['success']; strict += s['strict']
    print(run, 'selected', sel, 'correct dest', ok, 'NC SR %.1f strict %.1f' % (100 * succ / len(NC), 100 * strict / len(NC)))
    for f in sorted((OUT / 'runs' / run).glob('verify_*.jsonl')):
        pass
    vs = list((OUT / 'runs' / run).glob('verify_*.jsonl'))
    if vs:
        ch = [c for f in vs for s_ in open(f) for c in json.loads(s_)['changes']]
        print('   queried slots', len(ch), 'changed', sum(c['from'] != c['to'] for c in ch))
