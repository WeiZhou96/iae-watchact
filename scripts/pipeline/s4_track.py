"""Stage 4: SAM2 propagation from the final frame backwards; every registered instance keeps its identity.

Per video writes tracks/<vk>.npz with masks at 1/4 resolution (bool, [T, K, h, w]) and tracks/<vk>.json with
per-frame boxes, centroids and areas. Detections not registered to a public ID are tracked too (label null).
Usage: s4_track.py <shard> <nshards>
"""
import contextlib, json, sys
from pathlib import Path
import numpy as np
from iae.common import OUT, DATA, frames_dir, load_frame_index


def main():
    import torch, cv2
    from PIL import Image
    from transformers import Sam2VideoModel, Sam2VideoProcessor
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    md = str(DATA / 'models/sam2-hiera-tiny'); dev = 'cuda'; dt = torch.bfloat16
    proc = Sam2VideoProcessor.from_pretrained(md, local_files_only=True)
    model = Sam2VideoModel.from_pretrained(md, local_files_only=True).to(dev, dtype=dt).eval()
    out = OUT / 'tracks'; out.mkdir(parents=True, exist_ok=True)
    regs = sorted((OUT / 'register').glob('*.json'), reverse=len(sys.argv) > 3)
    for k, rp in enumerate(regs):
        if k % nsh != shard: continue
        reg = json.loads(rp.read_text()); vk = reg['video_key']
        if (out / f'{vk}.json').exists(): continue
        idx = load_frame_index(vk); T = len(idx['frames'])
        anchor = json.loads((OUT / 'det_final' / f'{vk}.json').read_text())['frame']['i']; forward = anchor == 0
        inst = reg['instances']; K = len(inst)
        if K == 0: continue
        label = {i: o for o, i in reg['assignment'].items()}
        images = [Image.open(frames_dir(vk) / f['file']).convert('RGB') for f in idx['frames']]
        H, W = images[0].height, images[0].width
        sess = proc.init_video_session(video=images, inference_device=dev, dtype=dt)
        obj_ids = list(range(1, K + 1))
        proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=anchor, obj_ids=list(obj_ids),
                                             input_boxes=[[x['box'] for x in inst]])
        h, w = H // 4, W // 4; masks = np.zeros((T, K, h, w), dtype=bool)
        with torch.inference_mode(), torch.autocast('cuda', dtype=dt):
            model(inference_session=sess, frame_idx=anchor)
            for o in model.propagate_in_video_iterator(sess, start_frame_idx=anchor, reverse=not forward, show_progress_bar=False):
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
        np.savez_compressed(out / f'{vk}.npz', masks=np.packbits(masks, axis=-1), shape=np.array(masks.shape))
        (out / f'{vk}.json').write_text(json.dumps({'video_key': vk, 'anchor': anchor, 'labels': [label.get(j) for j in range(K)],
                                                    'cats': [x['cat'] for x in inst], 'size': [W, H], 'per_frame': per}))
        print(vk, T, K, flush=True)
        del sess, images; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
