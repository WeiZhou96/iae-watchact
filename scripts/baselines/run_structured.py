import sys as _iae_sys
if '--help' in _iae_sys.argv:
    print('Usage: python %s [options]' % __file__)
    raise SystemExit(0)
"""Structured-evidence VLM baseline: the VLM receives the same instance-anchored evidence that the programs of IAE
consume and produces the plan itself (it replaces the grammar decoder / symbolic programs).

Evidence (held-out, from the IAE ensemble run; goal-free):
  NC  all candidate events of Eq. (7) (time, public ID, object or destination, frame logit), in temporal order;
  RD  video-level evidence score s_c of every movable instance and container, and the distance of the person's
      median table position to every container (the quantities used by the RD program).
Prompt, public scene, region frame, output format and parsing are those of run_direct.py; destination IDs in the
evidence are given in the request's spatial frame. With --no-frames the request is text only; otherwise the same 32
frames as the direct baseline are attached.
Usage: run_structured.py --model <dir> --out <dir> --shard i --nshards n [--no-frames] [--evidence-run ens_final]
"""
import argparse, hashlib, json, pickle, sys, time
from pathlib import Path
import numpy as np
from iae.common import requests, frames_dir, load_frame_index, OUT, TASKS
from iae.frames_ref import request_states, region_text, to_canonical, to_request
from iae.decode import candidate_events, dest_of
from iae.program_v1 import _person_canonical
from iae.register import canonical_xy
from scripts.run_direct import SYSTEM, USER

EVID = ('STRUCTURED_EVIDENCE (computed from the video by a perception system that registers every object of the final scene '
        'to its public ID, tracks it through the video and scores hand-object interactions; higher scores mean stronger '
        'evidence that the person indicated or handled that candidate):\n{evidence}\n')


EVID_V2 = ('STRUCTURED_EVIDENCE (computed from the video by a perception system that registers every object of the final scene '
           'to its public ID, tracks it through the video and scores hand-object interactions).\n{evidence}\n'
           'How to read this evidence: {semantics}\n')
SEM_NC = ('each event is a moment at which the person indicated a candidate with a hand; a higher score means stronger evidence '
          'that the candidate was indicated at that time, and events with low scores are likely spurious. A demonstration usually '
          'indicates one or more objects and then the place where they should go, possibly several times.')
SEM_RD = ('an interaction score is evidence that the person touched, pointed at or handled that instance during the video; '
          'it is NOT the probability that the instance must be moved. Decide the target objects from the instruction: if it '
          'refers to the selected/indicated/handled object(s), the targets are the instances with high scores; if it refers to '
          '"the other one" or "the others", the targets are the instances of the same kind that the person did NOT select, i.e. '
          'the ones with low scores. A container score is evidence that the person indicated that container.')


def nc_evidence(z, cands, view, ref, fps=10.0):
    lines = []
    for t, k, role, v in candidate_events(z, cands):
        c = cands[k]
        name = c['id'] if role == 'o' else to_request(dest_of(c), view, ref)
        lines.append(f't={t / fps:.1f}s  {"object" if role == "o" else "destination"} {name}  score {v:.2f}')
    return 'Indication events in temporal order:\n' + ('\n'.join(lines) if lines else '(none)')


def rd_evidence(s, f, req):
    cands = f['cands']
    mov = sorted(((s[k], c['id']) for k, c in enumerate(cands) if c['kind'] == 'movable'), reverse=True)
    con = sorted(((s[k], c['id']) for k, c in enumerate(cands) if c['kind'] == 'container'), reverse=True)
    out = ['Interaction score of each movable object over the video:'] + [f'  {i}: {v:.2f}' for v, i in mov]
    out += ['Interaction score of each container over the video:'] + [f'  {i}: {v:.2f}' for v, i in con]
    pc = _person_canonical(f['video_key'], f['register'], f['W'])
    on = {r[1]: r[2] for r in req['initial_states'] if len(r) == 3}
    if pc is not None:
        dd = [(float(np.linalg.norm(np.array(canonical_xy(on[o])) - pc)), o) for o in req['objects']
              if o.startswith(('basket', 'wooden_tray')) and canonical_xy(on.get(o))]
        if dd: out += ["Distance from the person's position to each container (table grid cells):"] + [f'  {o}: {d:.2f}' for d, o in sorted(dd)]
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--no-frames', action='store_true'); ap.add_argument('--evidence-run', default='ens_final')
    ap.add_argument('--prompt', choices=['v1', 'v2'], default='v1')  # v2: explicit semantics of scores and of 'the other one'
    ap.add_argument('--max-mem-gib', type=int, default=0)  # per-GPU cap when the model is split over several GPUs
    a = ap.parse_args()
    import torch
    from PIL import Image
    from iae.third_party.teb.schema import json_object_with_audit
    from iae.third_party.teb.nc_plan_compiler import region_catalog
    from iae.vlm import VLM
    import torch as _t
    mm = {i: f'{a.max_mem_gib}GiB' for i in range(_t.cuda.device_count())} if a.max_mem_gib else None
    vlm = VLM(a.model, device_map='auto', max_memory=mm)
    S = pickle.load(open(OUT / 'runs' / a.evidence_run / 'scores.pkl', 'rb'))
    a.out.mkdir(parents=True, exist_ok=True); dst = a.out / f'pred_{a.shard}.jsonl'
    done = {json.loads(x)['uid'] for x in open(dst)} if dst.exists() else set()
    rows = [r for k, r in enumerate(requests(TASKS)) if k % a.nshards == a.shard and r['uid'] not in done]
    cache = OUT / 'direct_32b' / 'frames896'
    for r in rows:
        rec = {'uid': r['uid'], 'activity_id': r['activity_id'], 'view': r['view'], 'reference': r['reference'], 'status': 'error'}
        vk = r['video_key']; idx = load_frame_index(vk)
        if idx.get('corrupt') or vk not in S:
            rec['error'] = 'corrupt_video' if idx.get('corrupt') else 'no_evidence'
        else:
            f = pickle.loads((OUT / 'feat_cache_v3' / f'{vk}.pkl').read_bytes())
            ev = nc_evidence(S[vk]['z'], f['cands'], r['view'], r['reference']) if r['task'] == 'Nonverbal_Cue' else rd_evidence(S[vk]['s'], f, r)
            st = request_states(r['initial_states'], r['view'], r['reference'])
            scene = {'objects': sorted(r['objects']), 'initial_states': st,
                     'allowed_regions': sorted({s_[-1] for s_ in request_states([['On', 'x', g] for g in region_catalog(r['objects'], r['initial_states'])], r['view'], r['reference'])}),
                     'region_descriptions': region_text(r['view'], r['reference'])}
            head = EVID.format(evidence=ev) if a.prompt == 'v1' else EVID_V2.format(
                evidence=ev, semantics=SEM_NC if r['task'] == 'Nonverbal_Cue' else SEM_RD)
            text = head + USER.format(
                instruction=r['instruction'].replace('Start executing the instruction from the final scene of the video: ', ''),
                scene=json.dumps(scene, ensure_ascii=False, sort_keys=True))
            frames = []
            if not a.no_frames:
                F = idx['frames']; n = 32; pick = [F[round(k * (len(F) - 1) / (n - 1))] for k in range(n)]
                for fr in pick:
                    p = cache / f"{vk}_{fr['i']:05d}.jpg"
                    if not p.exists():
                        im = Image.open(frames_dir(vk) / fr['file']).convert('RGB'); im.thumbnail((896, 896)); im.save(p, quality=92)
                    frames.append((str(p), fr['t']))
            t0 = time.time()
            if frames:
                raw, cost = vlm.generate(SYSTEM, frames, text)
            else:
                msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': [{'type': 'text', 'text': text}]}]
                prompt = vlm.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inp = vlm.processor(text=[prompt], return_tensors='pt').to(vlm.model.device)
                with torch.inference_mode():
                    ids = vlm.model.generate(**inp, max_new_tokens=1024, do_sample=False)
                raw = vlm.processor.batch_decode(ids[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0]
                cost = {'input_tokens': int(inp['attention_mask'].sum())}
            rec.update(raw=raw, seconds=round(time.time() - t0, 2), cost=cost, evidence=ev)
            try:
                parsed, _ = json_object_with_audit(raw)
                pairs = [{'object_id': p['object_id'], 'destination_region': to_canonical(p['destination_region'], r['view'], r['reference'])}
                         for p in parsed.get('pairs', []) if isinstance(p, dict) and isinstance(p.get('object_id'), str) and isinstance(p.get('destination_region'), str)]
                rec.update(pairs=pairs, status='ok')
            except Exception as e:
                rec['error'] = repr(e)
        with open(dst, 'a') as fo: fo.write(json.dumps(rec) + '\n')
        print(r['uid'], rec['status'], flush=True)


if __name__ == '__main__':
    main()
