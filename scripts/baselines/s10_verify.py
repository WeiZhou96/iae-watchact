import sys as _iae_sys
if '--help' in _iae_sys.argv:
    print('Usage: python %s [options]' % __file__)
    raise SystemExit(0)
"""Evidence-guided VLM verification of ambiguous destination slots (NC), goal-free at inference.

For every decoded destination event, destination candidates whose frame logit at that time is within `delta`
of the chosen one form a competitor set. If there is more than one, 7 frames around the event are marked
Set-of-Mark style (letters on candidate boxes / region points) and a frozen VLM is asked which one the person
indicates. The answer replaces the destination only for that slot. Writes a new run directory.
Usage: s10_verify.py <src_run> <out_run> <model_dir> [delta] [shard nshards]
"""
import json, pickle, string, sys, time
from pathlib import Path
import numpy as np
from iae.common import requests, OUT, frames_dir, load_frame_index
from iae.decode import decode, dest_of
from iae.features import _cand_geom

SYSTEM = 'You judge where a person is pointing in short video clips. Answer with a single letter.'
PROMPT = ('The frames show a person indicating a destination with a pointing gesture. Candidate destinations are marked '
          'with letters {letters}. Which marked destination is the person pointing at? Consider the direction of the finger '
          'and arm, not only which mark is closest to the fingertip in the image. Answer with exactly one letter.')
COLORS = [(0, 0, 255), (0, 200, 0), (255, 120, 0), (200, 0, 200)]


def main():
    import cv2
    from iae.vlm import VLM
    src, dst, model = sys.argv[1], sys.argv[2], sys.argv[3]
    delta = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
    shard, nsh = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (0, 1)
    cfg = json.load(open(OUT / 'runs' / src / 'config.json')); cache = cfg.get('cache', 'feat_cache')
    scores = pickle.load(open(OUT / 'runs' / src / 'scores.pkl', 'rb'))
    out = OUT / 'runs' / dst; out.mkdir(parents=True, exist_ok=True); tmp = out / 'marked'; tmp.mkdir(exist_ok=True)
    vlm = VLM(model, device_map={'': 0})
    reqs = requests(('Nonverbal_Cue',)); vks = sorted({r['video_key'] for r in reqs if r['video_key'] in scores})
    log = []
    for n, vk in enumerate(vks):
        if n % nsh != shard: continue
        f = pickle.loads((OUT / cache / f'{vk}.pkl').read_bytes()); cands = f['cands']; z = scores[vk]['z']
        tr = json.loads((OUT / 'tracks' / f'{vk}.json').read_text()); idx = load_frame_index(vk)
        d = decode(z, cands, cost=cfg.get('fold_cost_fixed', 0) or 0)
        ids = [c['id'] for c in cands]; changes = []
        for (t, cid, role, v) in d['events']:
            if role != 'd': continue
            k = ids.index(cid)
            comp = [j for j, c in enumerate(cands) if c['kind'] in ('container', 'region', 'drawer') and z[t, j] >= z[t, k] - delta]
            comp = sorted(comp, key=lambda j: -z[t, j])[:4]
            if len(comp) < 2: continue
            letters = string.ascii_uppercase[:len(comp)]; frames = []
            for dt in (-6, -4, -2, 0, 2, 4, 6):
                tt = min(max(t + dt, 0), len(idx['frames']) - 1)
                im = cv2.imread(str(frames_dir(vk) / idx['frames'][tt]['file']))
                for L, j, col in zip(letters, comp, COLORS):
                    cen, box = _cand_geom(cands[j], tr['per_frame'][tt])
                    if cen is None: continue
                    if box is not None: cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), col, 3); p = (int(box[0]), int(box[1]) - 8)
                    else: cv2.circle(im, (int(cen[0]), int(cen[1])), 14, col, 3); p = (int(cen[0]) + 16, int(cen[1]))
                    cv2.putText(im, L, p, cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 4)
                im = cv2.resize(im, (896, 504)); fp = tmp / f'{vk}_{t}_{dt}.jpg'; cv2.imwrite(str(fp), im); frames.append((str(fp), idx['frames'][tt]['t']))
            ans, cost = vlm.generate(SYSTEM, frames, PROMPT.format(letters=', '.join(letters)), max_new_tokens=4)
            a = ans.strip()[:1].upper()
            if a in letters:
                j = comp[letters.index(a)]
                changes.append({'t': t, 'from': cid, 'to': ids[j], 'competitors': [ids[j2] for j2 in comp], 'answer': a})
        log.append({'video_key': vk, 'changes': changes})
        with open(out / f'verify_{shard}.jsonl', 'a') as fo: fo.write(json.dumps(log[-1]) + '\n')
        print(vk, len(changes), flush=True)


if __name__ == '__main__':
    main()
