import sys as _iae_sys
if '--help' in _iae_sys.argv:
    print('Usage: python %s [options]' % __file__)
    raise SystemExit(0)
"""Direct VLM baseline (D1 contract) for all NC/RD requests, in each request's own spatial frame.

32 uniformly sampled 10-fps frames (last frame included), max side 896. The public scene is remapped to the
request frame with the official tables; the predicted regions are mapped back to canonical before scoring.
Usage: run_direct.py --model <dir> --out <dir> --shard i --nshards n [--tasks ...]
"""
import argparse, dataclasses, json, sys, time
from pathlib import Path
from iae.common import requests, frames_dir, load_frame_index, OUT, TASKS, EPISODIC
from iae.frames_ref import request_states, region_text, to_canonical

SYSTEM = ("You interpret a human demonstration video for a robot that will act AFTER the video, starting from the final scene. "
          "The person may select objects or locations by pointing, touching, picking up, or moving them. "
          "Use only the supplied frames and public scene information. Return one JSON object following the user's schema.")
OVERLAY_NOTE = ('Frames are annotated: coloured boxes mark the tracked objects and are labelled with their exact public object IDs; '
                'yellow lines show the detected hand skeletons and the forearm direction. ')
USER = ('Task instruction: {instruction}\n'
        'Infer every object placement the robot should perform. Return one JSON object {{"pairs":[{{"object_id":"exact public object ID",'
        '"destination_region":"exact public region ID"}}]}} with one to three pairs. Always return your best-supported answer; '
        'an empty answer counts as a failure. Region IDs follow the spatial reference described below.\nPUBLIC_SCENE\n{scene}')


_OVL = {}


def draw_overlay(im, vk, i):
    """Same perception outputs as the method: registered/tracked instance boxes with public IDs, hands, forearm."""
    from PIL import ImageDraw, ImageFont
    font = ImageFont.load_default(size=26)
    from iae.common import OUT
    if vk not in _OVL:
        L = lambda k: json.loads((OUT / k / f'{vk}.json').read_text()) if (OUT / k / f'{vk}.json').exists() else None
        _OVL[vk] = (L('tracks'), {r['i']: r for r in (L('hands') or [])}, {r['i']: r for r in (L('body') or [])})
    tr, hands, body = _OVL[vk]; d = ImageDraw.Draw(im)
    cols = [(255, 60, 60), (60, 200, 60), (60, 120, 255), (255, 160, 0), (200, 60, 200), (0, 200, 200)]
    if tr:
        for j, (lab, b) in enumerate(zip(tr['labels'], tr['per_frame'][i])):
            if lab is None or b is None: continue
            x0, y0, x1, y1 = b['box']; c = cols[j % len(cols)]
            d.rectangle([x0, y0, x1, y1], outline=c, width=3); d.text((x0 + 2, max(0, y0 - 30)), lab, fill=c, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    CH = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
    for h in hands.get(i, {}).get('hands', []):
        kp = h['kp']
        for u, v in CH: d.line([tuple(kp[u]), tuple(kp[v])], fill=(255, 230, 0), width=2)
        d.ellipse([kp[8][0] - 5, kp[8][1] - 5, kp[8][0] + 5, kp[8][1] + 5], outline=(255, 230, 0), width=3)
    b = body.get(i)
    if b and b.get('kp'):
        for e, w in ((7, 9), (8, 10)):
            if b['sc'][e] > 0.3 and b['sc'][w] > 0.3: d.line([tuple(b['kp'][e]), tuple(b['kp'][w])], fill=(255, 230, 0), width=3)
    return im


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1); ap.add_argument('--frames', type=int, default=32)
    ap.add_argument('--multi-gpu', action='store_true', help='use iev.vlm (device_map=auto) instead of teb LocalVLM')
    ap.add_argument('--tasks', choices=['implicit', 'episodic'], default='implicit')
    ap.add_argument('--overlay', action='store_true', help='draw registered instance IDs, hand skeletons and forearm lines on frames')
    ap.add_argument('--family', choices=['qwen', 'internvl'], default='qwen')
    a = ap.parse_args()
    from iae.third_party.teb.schema import Frame, json_object_with_audit
    from iae.third_party.teb.backends import make_backend, Request
    import iae.third_party.teb.backends as bm
    from iae.third_party.teb.nc_plan_compiler import region_catalog
    from PIL import Image
    bm.SYSTEM = SYSTEM
    if a.multi_gpu:
        from iae.vlm import VLM
        vlm = VLM(a.model, family=a.family)
    backend = None if a.multi_gpu else make_backend({'type': 'local', 'model': a.model, 'device': 'cuda:0', 'allow_download': False, 'temperature': 0.,
                            'seed': 42, 'max_pixels': 1638400})
    a.out.mkdir(parents=True, exist_ok=True); dst = a.out / f'pred_{a.shard}.jsonl'
    done = {json.loads(s)['uid'] for s in open(dst)} if dst.exists() else set()
    rows = [r for k, r in enumerate(requests(TASKS if a.tasks == 'implicit' else EPISODIC)) if k % a.nshards == a.shard and r['uid'] not in done]

    class R(Request):
        def text(self): return self.payload['_text']

    cache = a.out / ('frames896_overlay' if a.overlay else 'frames896'); cache.mkdir(exist_ok=True)
    for r in rows:
        rec = {'uid': r['uid'], 'activity_id': r['activity_id'], 'view': r['view'], 'reference': r['reference'], 'status': 'error'}
        idx = load_frame_index(r['video_key'])
        if idx.get('corrupt'):
            rec['error'] = 'corrupt_video'
        else:
            F = idx['frames']; n = a.frames; pick = [F[round(k * (len(F) - 1) / (n - 1))] for k in range(n)]
            frames = []
            for k, f in enumerate(pick):
                p = cache / f"{r['video_key']}_{f['i']:05d}.jpg"
                def _ok(q):
                    try:
                        with Image.open(q) as t: t.verify()
                        return True
                    except Exception: return False
                if not p.exists() or not _ok(p):
                    import os
                    im = Image.open(frames_dir(r['video_key']) / f['file']).convert('RGB')
                    if a.overlay: im = draw_overlay(im, r['video_key'], f['i'])
                    im.thumbnail((896, 896))
                    tmp = p.with_suffix(f'.{os.getpid()}.tmp.jpg'); im.save(tmp, quality=92); os.replace(tmp, p)
                import hashlib
                w, h = Image.open(p).size
                frames.append(Frame(index=k, timestamp=f['t'], path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), width=w, height=h))
            st = request_states(r['initial_states'], r['view'], r['reference'])
            scene = {'objects': sorted(r['objects']), 'initial_states': st,
                     'allowed_regions': sorted({s[-1] for s in request_states([['On', 'x', g] for g in region_catalog(r['objects'], r['initial_states'])], r['view'], r['reference'])}),
                     'region_descriptions': region_text(r['view'], r['reference'])}
            user = USER if a.tasks == 'implicit' else USER.replace('with one to three pairs', 'with every object that must be moved (one to eight pairs)')
            text = (OVERLAY_NOTE if a.overlay else '') + user.format(instruction=r['instruction'].replace('Start executing the instruction from the final scene of the video: ', ''),
                               scene=json.dumps(scene, ensure_ascii=False, sort_keys=True))
            t0 = time.time()
            if a.multi_gpu:
                raw, cost = vlm.generate(SYSTEM, [(fr.path, fr.timestamp) for fr in frames], text)
            else:
                req = R('direct', {'_text': text}, frames, 1024); prep = backend.prepare(req); rep = backend.generate(prep)
                raw, cost = rep.text, dataclasses.asdict(rep.cost)
            rec.update(raw=raw, seconds=round(time.time() - t0, 2), cost=cost)
            try:
                parsed, _ = json_object_with_audit(raw)
                pairs = [{'object_id': p['object_id'], 'destination_region': to_canonical(p['destination_region'], r['view'], r['reference'])}
                         for p in parsed.get('pairs', []) if isinstance(p, dict) and isinstance(p.get('object_id'), str) and isinstance(p.get('destination_region'), str)]
                rec.update(pairs=pairs, status='ok')
            except Exception as e:
                rec['error'] = repr(e)
        with open(dst, 'a') as f: f.write(json.dumps(rec) + '\n')
        print(r['uid'], rec['status'], flush=True)


if __name__ == '__main__':
    main()
