"""Control for backward anchoring: forward tracking from the first frame, matched to the registered instances at the end.

For every NC/RD video: count-constrained detection on the FIRST frame (same detector, phrases, counts and selection
rule as stage 2), SAM2 propagation forwards from the first frame (same model and settings as stage 4), and at the
last frame (the execution scene) each forward track is matched to a registered final-frame instance of the same
category by maximum total box IoU (exhaustive matching per category, IoU >= 0.1); a matched track takes that instance's public identifier,
unmatched tracks keep no identifier. Registered instances without a matching track have no candidate.
Writes tracks_fwd/<vk>.json in the format of stage 4 (features are built from it with cache name *_fwd).
Usage: s13_forward_track.py <shard> <nshards>
"""
import collections, itertools, json, sys
from pathlib import Path
import numpy as np
from iae.common import requests, OUT, DATA, frames_dir, load_frame_index, stem, TASKS
from scripts.s2_detect_final import PHRASE, iou


def detect(proc, model, img, need):
    import torch
    W, H = img.size; cands = []
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
    return chosen, {k: v for k, v in left.items() if v > 0}


def main():
    import torch, cv2
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2VideoModel, Sam2VideoProcessor
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    dev = 'cuda'; dt = torch.bfloat16
    dp = AutoProcessor.from_pretrained(str(DATA / 'models/grounding-dino-tiny'))
    dm = AutoModelForZeroShotObjectDetection.from_pretrained(str(DATA / 'models/grounding-dino-tiny')).cuda().eval()
    md = str(DATA / 'models/sam2-hiera-tiny')
    proc = Sam2VideoProcessor.from_pretrained(md, local_files_only=True)
    model = Sam2VideoModel.from_pretrained(md, local_files_only=True).to(dev, dtype=dt).eval()
    out = OUT / 'tracks_fwd'; out.mkdir(parents=True, exist_ok=True)
    vids = {}
    for r in requests(TASKS): vids.setdefault(r['video_key'], r)
    for n, (vk, r) in enumerate(sorted(vids.items())):
        if n % nsh != shard or (out / f'{vk}.json').exists(): continue
        rp = OUT / 'register' / f'{vk}.json'
        if not rp.exists(): continue
        reg = json.loads(rp.read_text())
        idx = load_frame_index(vk); T = len(idx['frames'])
        images = [Image.open(frames_dir(vk) / f['file']).convert('RGB') for f in idx['frames']]
        H, W = images[0].height, images[0].width
        need = collections.Counter(stem(o) for o in r['objects'])
        inst, missing = detect(dp, dm, images[0], need); K = len(inst)
        if K == 0: continue
        sess = proc.init_video_session(video=images, inference_device=dev, dtype=dt)
        proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=0, obj_ids=list(range(1, K + 1)),
                                             input_boxes=[[x['box'] for x in inst]])
        h, w = H // 4, W // 4; masks = np.zeros((T, K, h, w), dtype=bool)
        with torch.inference_mode(), torch.autocast('cuda', dtype=dt):
            model(inference_session=sess, frame_idx=0)
            for o in model.propagate_in_video_iterator(sess, start_frame_idx=0, reverse=False, show_progress_bar=False):
                m = proc.post_process_masks([o.pred_masks], original_sizes=[[H, W]], binarize=True)[0]
                m = m.reshape(m.shape[0], H, W).float().cpu().numpy() if m.dim() == 4 else m.float().cpu().numpy()
                for j in range(min(K, m.shape[0])):
                    masks[int(o.frame_idx), j] = cv2.resize(m[j].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        per = []
        for t in range(T):
            row = []
            for j in range(K):
                ys, xs = np.nonzero(masks[t, j])
                if len(xs) == 0: row.append(None); continue
                row.append({'box': [float(xs.min() * 4), float(ys.min() * 4), float(xs.max() * 4 + 4), float(ys.max() * 4 + 4)],
                            'c': [float(xs.mean() * 4), float(ys.mean() * 4)], 'area': int(len(xs)) * 16})
            per.append(row)
        # identity at the execution scene: match last-frame track boxes to registered final-frame instances
        lab_of = {i: o for o, i in reg['assignment'].items()}; rinst = reg['instances']
        labels = [None] * K
        for cat in {x['cat'] for x in inst}:
            tj = [j for j in range(K) if inst[j]['cat'] == cat]
            ri = [i for i, x in enumerate(rinst) if x['cat'] == cat and i in lab_of]
            S = {(j, i): (iou(per[-1][j]['box'], rinst[i]['box']) if per[-1][j] is not None else 0.0) for j in tj for i in ri}
            best = (0.0, ())
            small, large = (tj, ri) if len(tj) <= len(ri) else (ri, tj)
            for perm in itertools.permutations(large, len(small)):   # exhaustive matching (at most 3 per category)
                pairs = [(a, b) if small is tj else (b, a) for a, b in zip(small, perm)]
                v = sum(S[p] for p in pairs if S[p] >= 0.1)
                if v > best[0]: best = (v, tuple(pairs))
            for j, i in best[1]:
                if S[(j, i)] >= 0.1: labels[j] = lab_of[i]
        (out / f'{vk}.json').write_text(json.dumps({'video_key': vk, 'anchor': 0, 'labels': labels, 'cats': [x['cat'] for x in inst],
                                                    'size': [W, H], 'per_frame': per, 'first_frame_missing': missing}))
        print(vk, T, K, 'matched', sum(l is not None for l in labels), 'of', len(lab_of), 'missing', missing, flush=True)
        del sess, images; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
