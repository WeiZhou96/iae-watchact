"""Per-stage wall-clock timing on single videos (idle GPU), writing nothing to the pipeline outputs.
Stages in the teb env: final-frame detection, registration, SAM2 tracking, per-frame person detection,
feature construction, network + decoding. Hand/body keypoints are timed separately in their envs.
Usage: timing.py <video_key> [...]
"""
import json, pickle, sys, time
from pathlib import Path
import numpy as np
from iae.common import OUT, DATA, frames_dir, load_frame_index, requests, stem
from iae.register import register


def main():
    import torch, collections
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2VideoModel, Sam2VideoProcessor
    md = str(DATA / 'models/grounding-dino-tiny'); proc = AutoProcessor.from_pretrained(md)
    det = AutoModelForZeroShotObjectDetection.from_pretrained(md).cuda().eval()
    sp = Sam2VideoProcessor.from_pretrained(str(DATA / 'models/sam2-hiera-tiny'), local_files_only=True)
    sm = Sam2VideoModel.from_pretrained(str(DATA / 'models/sam2-hiera-tiny'), local_files_only=True).to('cuda', dtype=torch.bfloat16).eval()
    R = {r['video_key']: r for r in requests()}
    for vk in sys.argv[1:]:
        r = R[vk]; idx = load_frame_index(vk); T = len(idx['frames']); t = {}
        imgs = [Image.open(frames_dir(vk) / f['file']).convert('RGB') for f in idx['frames']]
        torch.cuda.synchronize(); t0 = time.time()
        d = json.loads((OUT / 'det_final' / f'{vk}.json').read_text())
        for c in collections.Counter(stem(o) for o in r['objects']):
            inp = proc(images=imgs[-1], text='object.', return_tensors='pt').to('cuda')
            with torch.no_grad(): det(**inp)
        torch.cuda.synchronize(); t['final_detection'] = time.time() - t0
        t0 = time.time(); register(d['instances'], r['objects'], r['initial_states'], d['size'][0], view=r['view']); t['registration'] = time.time() - t0
        t0 = time.time()
        sess = sp.init_video_session(video=imgs, inference_device='cuda', dtype=torch.bfloat16)
        sp.add_inputs_to_inference_session(inference_session=sess, frame_idx=T - 1, obj_ids=list(range(1, len(d['instances']) + 1)), input_boxes=[[x['box'] for x in d['instances']]])
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            sm(inference_session=sess, frame_idx=T - 1)
            for o in sm.propagate_in_video_iterator(sess, start_frame_idx=T - 1, reverse=True, show_progress_bar=False):
                sp.post_process_masks([o.pred_masks], original_sizes=[[imgs[0].height, imgs[0].width]], binarize=True)
        torch.cuda.synchronize(); t['sam2_tracking'] = time.time() - t0
        t0 = time.time()
        for s in range(0, T, 16):
            inp = proc(images=imgs[s:s + 16], text=['person.'] * len(imgs[s:s + 16]), return_tensors='pt').to('cuda')
            with torch.no_grad(): det(**inp)
        torch.cuda.synchronize(); t['person_detection'] = time.time() - t0
        from iae.features import build
        t0 = time.time(); f = build(vk); t['features'] = time.time() - t0
        from iae.model import EvidenceNet
        from iae.decode import decode
        net = EvidenceNet(nf=f['X'].shape[-1]).cuda().eval(); X = torch.tensor(f['X']).cuda(); M = torch.tensor(f['M']).cuda()
        t0 = time.time()
        with torch.no_grad(): z, s = net(X, M, torch.tensor([1.0, 0.0]).cuda())
        decode(z.cpu().numpy(), f['cands']); torch.cuda.synchronize(); t['network_and_decoding'] = time.time() - t0
        print(json.dumps({'video_key': vk, 'frames_10fps': T, **{k: round(v, 2) for k, v in t.items()}}), flush=True)
        del sess, imgs; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
