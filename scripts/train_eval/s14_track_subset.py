import sys as _iae_sys
if '--help' in _iae_sys.argv:
    print('Usage: python %s [options]' % __file__)
    raise SystemExit(0)
"""Matched-sampling control with the full perception pipeline: SAM2 propagation on N uniformly sampled frames only.

Same anchor-frame boxes and identities (stage 2/3), same SAM2 model and settings as stage 4, but the video given to
the segmenter consists of N frames at round(linspace(0, T-1, N)) (the frames shown to the VLM baselines; the last
frame is the anchor). Writes tracks_s<N>/<vk>.json with per_frame rows for the kept frames and keys keep, T_full.
Usage: s14_track_subset.py <N> <shard> <nshards>
"""
import json, sys
from pathlib import Path
import numpy as np
from iae.common import requests, OUT, DATA, frames_dir, load_frame_index, TASKS


def main():
    import torch, cv2
    from PIL import Image
    from transformers import Sam2VideoModel, Sam2VideoProcessor
    N, shard, nsh = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    md = str(DATA / 'models/sam2-hiera-tiny'); dev = 'cuda'; dt = torch.bfloat16
    proc = Sam2VideoProcessor.from_pretrained(md, local_files_only=True)
    model = Sam2VideoModel.from_pretrained(md, local_files_only=True).to(dev, dtype=dt).eval()
    out = OUT / f'tracks_s{N}'; out.mkdir(parents=True, exist_ok=True)
    vids = sorted({r['video_key'] for r in requests(TASKS)})
    for n, vk in enumerate(vids):
        if n % nsh != shard or (out / f'{vk}.json').exists(): continue
        rp = OUT / 'register' / f'{vk}.json'
        if not rp.exists(): continue
        reg = json.loads(rp.read_text()); inst = reg['instances']; K = len(inst)
        if K == 0: continue
        idx = load_frame_index(vk); T = len(idx['frames'])
        anchor = json.loads((OUT / 'det_final' / f'{vk}.json').read_text())['frame']['i']
        assert anchor == T - 1, (vk, anchor, T)
        keep = sorted({int(round(x)) for x in np.linspace(0, T - 1, N)})
        label = {i: o for o, i in reg['assignment'].items()}
        images = [Image.open(frames_dir(vk) / idx['frames'][i]['file']).convert('RGB') for i in keep]
        H, W = images[0].height, images[0].width; L = len(keep)
        sess = proc.init_video_session(video=images, inference_device=dev, dtype=dt)
        proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=L - 1, obj_ids=list(range(1, K + 1)),
                                             input_boxes=[[x['box'] for x in inst]])
        h, w = H // 4, W // 4; masks = np.zeros((L, K, h, w), dtype=bool)
        with torch.inference_mode(), torch.autocast('cuda', dtype=dt):
            model(inference_session=sess, frame_idx=L - 1)
            for o in model.propagate_in_video_iterator(sess, start_frame_idx=L - 1, reverse=True, show_progress_bar=False):
                m = proc.post_process_masks([o.pred_masks], original_sizes=[[H, W]], binarize=True)[0]
                m = m.reshape(m.shape[0], H, W).float().cpu().numpy() if m.dim() == 4 else m.float().cpu().numpy()
                for j in range(min(K, m.shape[0])):
                    masks[int(o.frame_idx), j] = cv2.resize(m[j].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        per = []
        for t in range(L):
            row = []
            for j in range(K):
                ys, xs = np.nonzero(masks[t, j])
                if len(xs) == 0: row.append(None); continue
                row.append({'box': [float(xs.min() * 4), float(ys.min() * 4), float(xs.max() * 4 + 4), float(ys.max() * 4 + 4)],
                            'c': [float(xs.mean() * 4), float(ys.mean() * 4)], 'area': int(len(xs)) * 16})
            per.append(row)
        (out / f'{vk}.json').write_text(json.dumps({'video_key': vk, 'anchor': T - 1, 'labels': [label.get(j) for j in range(K)],
                                                    'cats': [x['cat'] for x in inst], 'size': [W, H], 'per_frame': per,
                                                    'keep': keep, 'T_full': T}))
        print(vk, T, L, K, flush=True)
        del sess, images; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
