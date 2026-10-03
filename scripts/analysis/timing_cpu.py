import sys as _iae_sys
if '--help' in _iae_sys.argv:
    print('Usage: python %s [options]' % __file__)
    raise SystemExit(0)
"""Time hand (MediaPipe, hand env) or body (RTMPose, rtmlib env) keypoints on one video, single process, no outputs.
Usage: timing_cpu.py hands|body <video_key>"""
import json, sys, time
from pathlib import Path
from iae.config import DATA_ROOT as ROOT, OUT_ROOT as OUT
kind, vk = sys.argv[1], sys.argv[2]
import cv2
idx = json.loads((OUT / 'frames10' / vk / 'index.json').read_text())
persons = {r['i']: r for r in json.loads((OUT / 'person' / f'{vk}.json').read_text())}
frames = [(f, persons.get(f['i'])) for f in idx['frames']]
if kind == 'hands':
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / 'models/mediapipe/hand_landmarker.task')), num_hands=2))
else:
    from rtmlib import RTMPose
    det = RTMPose(onnx_model='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip',
                  model_input_size=(192, 256), backend='onnxruntime', device='cpu')
t0 = time.time(); n = 0
for f, p in frames:
    if not (p and p['box'] and p['score'] >= 0.35): continue
    img = cv2.imread(str(OUT / 'frames10' / vk / f['file'])); n += 1
    if kind == 'hands':
        x1, y1, x2, y2 = [int(v) for v in p['box']]; crop = img[max(0, y1):y2, max(0, x1):x2]
        s = 960 / max(crop.shape[:2]); crop = cv2.resize(crop, None, fx=s, fy=s)
        det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
    else:
        det(img, bboxes=[p['box']])
print(json.dumps({'kind': kind, 'video_key': vk, 'frames_with_person': n, 'seconds': round(time.time() - t0, 2)}))
