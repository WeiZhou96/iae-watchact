"""Stage 2: count-constrained open-vocabulary detection on the final frame of every video.

Candidates from one query per public category are pooled; exactly the public count per category
is selected greedily by score with cross-category de-duplication (IoU >= 0.5). Only public object
lists are used.
"""
import collections, json, sys
from pathlib import Path
from iae.common import requests, frames_dir, load_frame_index, stem, OUT, DATA, ALL_TASKS, INITIAL_ANCHOR

PHRASE = {'basket': 'basket', 'wooden_tray': 'wooden tray', 'butter': 'butter box', 'cream_cheese': 'cream cheese box',
          'milk': 'milk carton', 'alphabet_soup': 'soup can', 'orange_juice': 'orange juice carton',
          'new_salad_dressing': 'salad dressing bottle', 'ketchup': 'ketchup bottle', 'bbq_sauce': 'bbq sauce bottle',
          'tomato_sauce': 'tomato sauce can', 'wooden_cabinet': 'black cabinet with drawers', 'porcelain_mug': 'mug',
          'tennis_ball': 'tennis ball', 'mango': 'mango', 'pear': 'pear'}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    md = str(DATA / 'models/grounding-dino-tiny')
    proc = AutoProcessor.from_pretrained(md); model = AutoModelForZeroShotObjectDetection.from_pretrained(md).cuda().eval()
    out = OUT / 'det_final'; out.mkdir(parents=True, exist_ok=True)
    vids = {}
    for r in requests(ALL_TASKS): vids.setdefault(r['video_key'], (r['objects'], r['task']))
    for vk, (objs, task) in sorted(vids.items()):
        if (out / f'{vk}.json').exists(): continue
        idx = load_frame_index(vk)
        if idx.get('corrupt'): continue
        f = idx['frames'][0] if task in INITIAL_ANCHOR else idx['frames'][-1]; img = Image.open(frames_dir(vk) / f['file']).convert('RGB'); W, H = img.size
        need = collections.Counter(stem(o) for o in objs); cands = []
        for c in need:
            inp = proc(images=img, text=PHRASE[c] + '.', return_tensors='pt').to('cuda')
            with torch.no_grad(): o = model(**inp)
            res = proc.post_process_grounded_object_detection(o, inp.input_ids, threshold=0.15, text_threshold=0.15, target_sizes=[(H, W)])[0]
            for b, s in zip(res['boxes'].tolist(), res['scores'].tolist()):
                area = (b[2] - b[0]) * (b[3] - b[1]) / (W * H)
                if area > (0.45 if c in ('wooden_cabinet',) else 0.2): continue
                cands.append({'cat': c, 'box': [round(v, 1) for v in b], 'score': round(s, 4)})
        cands.sort(key=lambda x: -x['score']); chosen = []; left = dict(need)
        for x in cands:
            if left.get(x['cat'], 0) <= 0: continue
            if any(iou(x['box'], y['box']) >= 0.5 for y in chosen): continue
            chosen.append(x); left[x['cat']] -= 1
        missing = {k: v for k, v in left.items() if v > 0}
        (out / f'{vk}.json').write_text(json.dumps({'video_key': vk, 'frame': f, 'size': [W, H], 'need': need,
                                                    'instances': chosen, 'missing': missing, 'n_candidates': len(cands)}))
        print(vk, 'missing', missing, flush=True)


if __name__ == '__main__':
    main()
