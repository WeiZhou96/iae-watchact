"""Stage 3: register final-frame detections to public IDs for every video; draw a few overlays for inspection."""
import json, sys
from pathlib import Path
from iae.common import requests, OUT, frames_dir, ALL_TASKS
from iae.register import register


def main():
    import cv2
    out = OUT / 'register'; out.mkdir(parents=True, exist_ok=True); viz = out / 'viz'; viz.mkdir(exist_ok=True)
    vids = {}
    for r in requests(ALL_TASKS): vids.setdefault(r['video_key'], r)
    stats = []
    for vk, r in sorted(vids.items()):
        p = OUT / 'det_final' / f'{vk}.json'
        if not p.exists() or (out / f'{vk}.json').exists(): continue
        d = json.loads(p.read_text()); W = d['size'][0]
        reg = register(d['instances'], r['objects'], r['initial_states'], W, view=r['view'])
        reg.update(video_key=vk, activity_id=r['activity_id'], view=r['view'], instances=d['instances'])
        (out / f'{vk}.json').write_text(json.dumps(reg))
        stats.append((vk, r['view'], reg['status'], reg['residual'], reg['margin'], reg.get('angle_deg')))
        if hash(vk) % 12 == 0:
            img = cv2.imread(str(frames_dir(vk) / d['frame']['file']))
            for o, i in reg['assignment'].items():
                b = [int(v) for v in d['instances'][i]['box']]
                cv2.rectangle(img, b[:2], b[2:], (0, 255, 0), 2); cv2.putText(img, o, (b[0], b[1] - 6), 0, 0.7, (0, 255, 255), 2)
            cv2.imwrite(str(viz / f'{vk}.jpg'), cv2.resize(img, (960, 540)))
    for view in ['front', 'side', 'oblique']:
        v = [s for s in stats if s[1] == view and s[2] == 'ok']
        import statistics
        print(view, 'n', len(v), 'median residual', round(statistics.median(x[3] for x in v), 3),
              'margin<0.02', sum(1 for x in v if x[4] is not None and x[4] < 0.02), 'angles', sorted({x[5] for x in v})[:12])


if __name__ == '__main__':
    main()
