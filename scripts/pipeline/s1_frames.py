import sys as _iae_sys
if '--help' in _iae_sys.argv:
    print('Usage: python %s [options]' % __file__)
    raise SystemExit(0)
"""Stage 1: decode every NC/RD video at 10 fps to 1280-px JPEGs (sequential decode, parallel across videos).

index.json records native fps, native frame count, and the native frame index of every saved frame.
The last native frame is always saved (final scene anchor).
"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from iae.common import requests, frames_dir, FRAME_FPS, FRAME_WIDTH, ALL_TASKS


def decode(args):
    import cv2
    vpath, vkey = args
    out = frames_dir(vkey)
    if (out / 'index.json').exists(): return vkey, 'exists'
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(vpath); fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = fps / FRAME_FPS; want = set(round(k * step) for k in range(int(n / step) + 1)); frames = []; i = 0; last = None
    while True:
        ok, f = cap.read()
        if not ok: break
        last = (i, f)
        if i in want:
            s = FRAME_WIDTH / f.shape[1]; g = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            name = f'{len(frames):05d}.jpg'; cv2.imwrite(str(out / name), g, [cv2.IMWRITE_JPEG_QUALITY, 92])
            frames.append({'i': len(frames), 'native': i, 't': i / fps, 'file': name})
        i += 1
    if not frames:
        (out / 'index.json').write_text(json.dumps({'video_path': vpath, 'corrupt': True, 'frames': []}))
        return vkey, 'corrupt'
    if frames[-1]['native'] != last[0]:
        f = last[1]; s = FRAME_WIDTH / f.shape[1]; g = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        name = f'{len(frames):05d}.jpg'; cv2.imwrite(str(out / name), g, [cv2.IMWRITE_JPEG_QUALITY, 92])
        frames.append({'i': len(frames), 'native': last[0], 't': last[0] / fps, 'file': name})
    idx = {'video_path': vpath, 'native_fps': fps, 'native_frames': i, 'width': g.shape[1], 'height': g.shape[0],
           'scale_from_native': s, 'frames': frames}
    (out / 'index.json').write_text(json.dumps(idx))
    return vkey, len(frames)


if __name__ == '__main__':
    vids = sorted({(r['video_path'], r['video_key']) for r in requests(ALL_TASKS)})
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 16) as ex:
        for k, n in ex.map(decode, vids): print(k, n, flush=True)
