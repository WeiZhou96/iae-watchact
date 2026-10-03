import sys as _iae_sys
if '--help' in _iae_sys.argv:
    print('Usage: python %s [options]' % __file__)
    raise SystemExit(0)
"""Stage 5c: COCO-17 body keypoints (RTMPose-m, CPU via rtmlib) inside the person box of every 10-fps frame.

Output body/<vk>.json: per frame {i, kp: 17x[x,y], sc: 17 scores} or null when no person.
Usage (pose_rtmlib env): s5c_body.py <workers> [video_key ...]
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from iae.config import DATA_ROOT as ROOT, OUT_ROOT as OUT
URL = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip'


def run(vk):
    import cv2
    from rtmlib import RTMPose
    dst = OUT / 'body' / f'{vk}.json'
    if dst.exists(): return vk, 'exists'
    m = RTMPose(onnx_model=URL, model_input_size=(192, 256), backend='onnxruntime', device='cpu')
    idx = json.loads((OUT / 'frames10' / vk / 'index.json').read_text())
    persons = {r['i']: r for r in json.loads((OUT / 'person' / f'{vk}.json').read_text())}
    rows = []
    for f in idx['frames']:
        p = persons.get(f['i'])
        if not (p and p['box'] and p['score'] >= 0.35): rows.append({'i': f['i'], 'kp': None}); continue
        img = cv2.imread(str(OUT / 'frames10' / vk / f['file']))
        kp, sc = m(img, bboxes=[p['box']])
        rows.append({'i': f['i'], 'kp': kp[0].round(1).tolist(), 'sc': sc[0].round(3).tolist()})
    dst.write_text(json.dumps(rows))
    return vk, len(rows)


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    (OUT / 'body').mkdir(exist_ok=True)
    vks = sys.argv[2:] or sorted(p.stem for p in (OUT / 'person').glob('*.json'))
    t = time.time()
    with ProcessPoolExecutor(max_workers=int(sys.argv[1])) as ex:
        for vk, n in ex.map(run, vks): print(vk, n, round(time.time() - t), flush=True)
