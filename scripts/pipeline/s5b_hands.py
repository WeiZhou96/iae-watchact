"""Stage 5b: MediaPipe hand landmarks on person crops (hand_mediapipe env, CPU, multiprocess).

Output hands/<vk>.json: per frame list of hands {handed, score, kp: 21x[x,y] in 1280-px frame coordinates}.
Frames with person score < 0.35 are skipped (no person in view).
"""
import json, sys, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from iae.config import DATA_ROOT as ROOT, OUT_ROOT as OUT


def run(vk):
    import cv2, mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    dst = OUT / 'hands' / f'{vk}.json'
    if dst.exists(): return vk, 'exists'
    det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / 'models/mediapipe/hand_landmarker.task')),
        num_hands=2, min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3))
    idx = json.loads((OUT / 'frames10' / vk / 'index.json').read_text())
    persons = {r['i']: r for r in json.loads((OUT / 'person' / f'{vk}.json').read_text())}
    rows = []
    for f in idx['frames']:
        p = persons.get(f['i']); hands = []
        if p and p['box'] and p['score'] >= 0.35:
            img = cv2.imread(str(OUT / 'frames10' / vk / f['file'])); H, W = img.shape[:2]
            x1, y1, x2, y2 = p['box']; mx, my = 0.2 * (x2 - x1), 0.1 * (y2 - y1)
            x1, y1, x2, y2 = int(max(0, x1 - mx)), int(max(0, y1 - my)), int(min(W, x2 + mx)), int(min(H, y2 + my))
            crop = img[y1:y2, x1:x2]
            if crop.size:
                s = 960 / max(crop.shape[:2]); crop = cv2.resize(crop, None, fx=s, fy=s)
                r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                for lm, hd in zip(r.hand_landmarks, r.handedness):
                    hands.append({'handed': hd[0].category_name, 'score': round(hd[0].score, 3),
                                  'kp': [[round(x1 + q.x * crop.shape[1] / s, 1), round(y1 + q.y * crop.shape[0] / s, 1)] for q in lm]})
        rows.append({'i': f['i'], 'hands': hands})
    dst.write_text(json.dumps(rows)); det.close()
    return vk, sum(1 for r in rows if r['hands'])


if __name__ == '__main__':
    (OUT / 'hands').mkdir(exist_ok=True)
    vks = sorted(p.stem for p in (OUT / 'person').glob('*.json'))
    with ProcessPoolExecutor(max_workers=int(sys.argv[1])) as ex:
        for vk, n in ex.map(run, vks): print(vk, n, flush=True)
    os._exit(0)
