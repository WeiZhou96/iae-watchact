"""Local Qwen-compatible VLM and an explicit synthetic fixture backend."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import time
from PIL import Image
from .schema import Frame,json_object
from .budget import Cost
from .io import digest_object,read_json,digest_file
from .prompts import SYSTEM,PROMPTS,PROMPT_VERSION,TRACKER_EVIDENCE_NOTE

@dataclass
class Request:
    task:str
    payload:dict
    frames:list[Frame]
    max_new_tokens:int=1024
    nonce:int=0
    def text(self)->str:
        note = TRACKER_EVIDENCE_NOTE if 'sam2_tracker_evidence' in self.payload else ''
        return PROMPTS[self.task]+note+'\nINPUT_JSON\n'+json.dumps(self.payload,ensure_ascii=False,sort_keys=True)
    def fingerprint(self)->dict:
        return {'prompt_version':PROMPT_VERSION,'prompt_sha256':digest_object(PROMPTS[self.task]),
                'system':SYSTEM,'task':self.task,'payload':self.payload,
                'frames':[{'index':f.index,'timestamp':f.timestamp,'sha256':f.sha256} for f in self.frames],
                'max_new_tokens':self.max_new_tokens,'nonce':self.nonce}

@dataclass
class Prepared:
    request:Request
    inputs:Any
    estimate:Cost

@dataclass
class Reply:
    text:str
    cost:Cost
    seconds:float

class Backend:
    signature:dict
    def prepare(self,request:Request)->Prepared:raise NotImplementedError
    def generate(self,prepared:Prepared)->Reply:raise NotImplementedError

class LocalVLM(Backend):
    def __init__(self,model:str,revision:str|None=None,device:str='cpu',allow_download:bool=False,
                 max_pixels:int=802816,temperature:float=0.,seed:int=42,**unused):
        if unused:raise ValueError(f'Unknown local backend options {set(unused)}')
        import torch
        try:from transformers import AutoProcessor,AutoModelForImageTextToText
        except ImportError as exc:raise RuntimeError('Install the [local] extra for real VLM inference') from exc
        if not allow_download and not Path(model).exists():raise FileNotFoundError('Local model path not found; explicitly download weights first')
        if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        dtype=torch.float32 if device=='cpu' else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        kwargs={'local_files_only':not allow_download,'trust_remote_code':False}
        if revision:kwargs['revision']=revision
        self.processor=AutoProcessor.from_pretrained(model,max_pixels=max_pixels,**kwargs)
        self.model=AutoModelForImageTextToText.from_pretrained(model,torch_dtype=dtype,**kwargs).to(device)
        self.model.eval();self.model.requires_grad_(False);self.device=device;self.temperature=temperature;self.seed=seed
        self.signature={'backend':'local','model':str(model),'revision':revision,
                        'resolved_commit':getattr(self.model.config,'_commit_hash',None),
                        'max_pixels':max_pixels,'temperature':temperature,'seed':seed,
                        'input_mode':'timestamped multi-image, not native video tokenization'}
        if Path(model).is_dir():
            root=Path(model)
            self.signature['local_artifact_audit']={
                'small_file_sha256':{f.name:digest_file(f) for f in root.glob('*.json') if f.stat().st_size<20_000_000},
                'weight_file_stats':[{ 'name':f.name,'bytes':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns}
                                     for f in sorted(root.glob('*.safetensors'))],
                'qualification':'Weight file metadata and optional download lock; not a full weight-byte checksum.'} 
        self.image_token_id=getattr(self.model.config,'image_token_id',None)
        if self.image_token_id is None:
            self.image_token_id=self.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        if self.image_token_id is None or self.image_token_id==self.processor.tokenizer.unk_token_id:
            raise RuntimeError('Image-token accounting is undefined for this processor; use a verified Qwen-compatible adapter')

    def prepare(self,request):
        images=[];content=[]
        for f in request.frames:
            content.append({'type':'text','text':f'FRAME {f.index} TIME {f.timestamp:.6f} seconds'})
            content.append({'type':'image'})
            with Image.open(f.path) as im:images.append(im.convert('RGB').copy())
        content.append({'type':'text','text':request.text()})
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':content}]
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        if images:inputs=self.processor(text=[text],images=images,padding=True,return_tensors='pt')
        else:inputs=self.processor(text=[text],padding=True,return_tensors='pt')
        n=int(inputs['attention_mask'].sum())
        visual=int((inputs['input_ids']==self.image_token_id).sum()) if self.image_token_id is not None else 0
        return Prepared(request,inputs,Cost(n,visual,request.max_new_tokens,len(request.frames),exact=True))

    def generate(self,prepared):
        import torch
        torch.manual_seed(self.seed+prepared.request.nonce)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(self.seed+prepared.request.nonce)
        inputs=prepared.inputs.to(self.device);n=inputs['input_ids'].shape[1]
        kwargs={'max_new_tokens':prepared.request.max_new_tokens,'do_sample':self.temperature>0}
        if self.temperature>0:kwargs['temperature']=self.temperature
        if self.device.startswith('cuda'):torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.inference_mode():ids=self.model.generate(**inputs,**kwargs)
        if self.device.startswith('cuda'):torch.cuda.synchronize()
        generated=ids[:,n:];text=self.processor.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        cost=Cost(prepared.estimate.input_tokens,prepared.estimate.visual_tokens,int(generated.numel()),len(prepared.request.frames),exact=True)
        return Reply(text,cost,time.perf_counter()-start)

class FixtureBackend(Backend):
    """Test double consuming explicit fixture observations, never real videos/benchmark examples."""
    def __init__(self,fixture:dict):
        self.fixture=fixture;self.signature={'backend':'synthetic_fixture','fixture_digest':digest_object(fixture)}
    def prepare(self,request):
        n=len(request.text().encode('utf-8'))//3+1
        return Prepared(request,None,Cost(n,len(request.frames)*16,request.max_new_tokens,len(request.frames),exact=False))
    def generate(self,prepared):
        r=prepared.request;f=self.fixture
        if r.task=='slots':obj={'slots':f['slots']}
        elif r.task=='events':obj={'objects':f['objects'],'events':f['events']}
        elif r.task=='candidates':obj={'candidates':f['candidates']}
        elif r.task in ['intent_map','destination_map']:obj={'moves':[]}
        elif r.task=='object_inventory':
            objects=[]
            frame_ids=[frame.index for frame in r.frames]
            for index, item in enumerate(f.get('objects', []), 1):
                description=str(item.get('description', 'object'))
                role='container' if any(word in description.casefold()
                                        for word in ('tray', 'basket', 'cabinet', 'drawer', 'container')) else 'movable'
                objects.append({'id':f'v{index}','role':role,
                                'description':description,'position':'unknown',
                                'frame_ids':frame_ids or [0]})
            obj={'objects':objects}
        elif r.task in ['identity_align','identity_align_retry']:
            event_objects=r.payload.get('event_objects', [])
            inventory=r.payload.get('inventory', [])
            obj={'mappings':[{'event_object_id':event['id'],
                              'inventory_id':inventory[index]['id'],
                              'evidence_frame_ids':list(event.get('frame_ids', []))[:1] or [r.frames[0].index]}
                             for index, event in enumerate(event_objects)
                             if index < len(inventory)]}
        elif r.task=='container_inventory':
            containers=[]
            for item in f.get('objects',[]):
                desc=str(item.get('description','')).casefold()
                if any(word in desc for word in ('tray','basket','cabinet','drawer','container')):
                    containers.append({'id':item['id'],'kind':'container','position':'unknown',
                                       'frame_ids':list(range(len(r.frames)))})
            obj={'containers':containers}
        elif r.task=='container_rank':obj={'selected_candidate_ids':[]}
        elif r.task in ['compare','global_compare']:
            # Deterministic preference solely for a transparent synthetic test transcript.
            def support(candidate):
                if 'bindings' not in candidate:return 0
                return sum(v==f['preferred_bindings'].get(k) for k,v in candidate['bindings'].items())
            aa=support(r.payload['A']);bb=support(r.payload['B'])
            obj={'choice':'A' if aa>bb else 'B' if bb>aa else 'TIE'}
        elif r.task=='plan':
            selected=r.payload.get('selected',{}).get('bindings')
            if selected is None:selected=f['candidates'][0]['bindings']
            target=selected.get('target',{}).get('value');dest=selected.get('destination',{}).get('value')
            objects=[{'name':o['id'],'description':o['description']} for o in f['objects']]
            obj={'Objects':objects,'Actions':[]}
            if target and dest:
                obj['Actions']=[{'command':'pick','object':target,'source_location':'table'},
                                {'command':'place','object':target,'target_location':dest}]
        else:raise ValueError('Fixture backend does not implement '+r.task)
        text=json.dumps(obj)
        return Reply(text,Cost(prepared.estimate.input_tokens,prepared.estimate.visual_tokens,max(1,len(text)//3),len(r.frames),exact=False),0.)


def make_backend(config:dict,fixture:dict|None=None)->Backend:
    options=dict(config);kind=options.pop('type','local')
    if kind=='local':return LocalVLM(**options)
    if kind=='fixture':
        if fixture is None:raise ValueError('Fixture backend requires explicit synthetic fixture; never falls back from real inference')
        return FixtureBackend(fixture)
    raise ValueError('Unknown backend '+kind)
