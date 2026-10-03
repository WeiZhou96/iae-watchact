import sys as _iae_sys
if '--help' in _iae_sys.argv:
    print('Usage: python %s [options]' % __file__)
    raise SystemExit(0)
"""Stage 5a: person box per 10-fps frame (GroundingDINO, batched). Usage: s5a_person.py <shard> <nshards>"""
import json, sys
from pathlib import Path
from iae.common import OUT, DATA, frames_dir, requests, load_frame_index


def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    md = str(DATA / 'models/grounding-dino-tiny')
    proc = AutoProcessor.from_pretrained(md); model = AutoModelForZeroShotObjectDetection.from_pretrained(md).cuda().eval()
    out = OUT / 'person'; out.mkdir(parents=True, exist_ok=True)
    vks = sorted({r['video_key'] for r in requests()}, reverse=len(sys.argv) > 3)
    for k, vk in enumerate(vks):
        if k % nsh != shard or (out / f'{vk}.json').exists(): continue
        idx = load_frame_index(vk)
        if idx.get('corrupt'): continue
        rows = []
        for s in range(0, len(idx['frames']), 16):
            fr = idx['frames'][s:s + 16]; imgs = [Image.open(frames_dir(vk) / f['file']).convert('RGB') for f in fr]
            W, H = imgs[0].size
            inp = proc(images=imgs, text=['person.'] * len(imgs), return_tensors='pt').to('cuda')
            with torch.no_grad(): o = model(**inp)
            res = proc.post_process_grounded_object_detection(o, inp.input_ids, threshold=0.3, text_threshold=0.25, target_sizes=[(H, W)] * len(imgs))
            for f, r in zip(fr, res):
                cand = [(b, s_) for b, s_ in zip(r['boxes'].tolist(), r['scores'].tolist())]
                best = max(cand, key=lambda x: x[1], default=None)
                rows.append({'i': f['i'], 'box': [round(v, 1) for v in best[0]] if best else None, 'score': round(best[1], 3) if best else 0.0})
        (out / f'{vk}.json').write_text(json.dumps(rows)); print(vk, len(rows), flush=True)


if __name__ == '__main__':
    main()
