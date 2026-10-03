"""WatchAct public manifest importer: request IDs and routing manifest; no goal access."""
from __future__ import annotations
from pathlib import Path
from .io import read_json,read_jsonl,write_json,write_jsonl,inside,digest_object,digest_file
from .schema import PublicInput

TASKS=['Imitation','Reversal','Temporal_Sort','Fine-Grained_Action','Count','Ordinal','State_Change','Moment',
       'Nonverbal_Cue','Reference_Disambiguation','Restore_Previous_State','Task_Continuation',
       'Error_Correction','Conditional_Execution']


def import_dataset(data_root,out,tasks=None):
    root=Path(data_root).resolve();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tasks=tasks or TASKS
    if set(tasks)-set(TASKS):raise ValueError('Unknown task names')
    public=[];routing=[];sources={}
    for task in tasks:
        source=root/'data'/f'{task}.jsonl'
        if not source.exists():raise FileNotFoundError(source)
        sources[task]=digest_file(source)
        for index,row in enumerate(read_jsonl(source)):
            for k in ['language_instruction','video','id','original_id']:
                if k not in row:raise ValueError(f'{source}:{index+1} missing required {k}')
            video=inside(root/'videos',str(row['video']))
            if not video.is_file():video=inside(root/'videos',f'{task}/{row["video"]}')
            if not video.is_file():raise FileNotFoundError(f'Unresolved WatchAct video: {row["video"]}')
            uid=digest_object({'task':task,'id':row['id'],'original_id':row['original_id'],
                               'instruction':row['language_instruction'],'video':row['video']})[:24]
            p=PublicInput(uid,str(row['language_instruction']),str(row.get('camera_perspective','front')),
                          str(row.get('spatial_reference','camera')),video_path=str(video))
            PublicInput.from_dict(p.__dict__);public.append(p.__dict__)
            routing.append({'uid':uid,'task':task,'id':row['id'],'original_id':row['original_id'],
                'video':row['video'],'language_instruction':row['language_instruction'],
                'camera_perspective':p.camera_perspective,'spatial_reference':p.spatial_reference,
                'video_path':str(video),'source_jsonl':str(source)})
    if len({p['uid'] for p in public})!=len(public):raise ValueError('Duplicate derived example IDs')
    write_jsonl(out/'public.jsonl',public);write_jsonl(out/'routing.jsonl',routing)
    write_json(out/'import_manifest.json',{'tasks':tasks,'examples':len(public),'source_hashes':sources,
        'input_whitelist':['language_instruction','video','camera_perspective','spatial_reference'],
        'never_read':'meta_data/*.json (oracle goals are not read by this importer)',
        'split_policy':'No supervised split inferred from Hugging Face train label'})
    return {'examples':len(public),'public_manifest':str(out/'public.jsonl'),'routing':str(out/'routing.jsonl')}
