"""Typed public-only interfaces and strict structured-output validation."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from typing import Any
import copy
import json
import math

class SchemaError(ValueError):pass


def json_object_with_audit(text:str|dict)->tuple[dict,dict|None]:
    if isinstance(text,dict):return text,None
    if not isinstance(text,str):raise SchemaError('Model response must be JSON object or string')
    text=text.strip()
    # Some VLM responses append a closing Markdown fence without opening one.
    # Remove only that exact trailing fence; all other extra text remains invalid.
    if text.endswith('```') and not text.startswith('```') and '\n```' not in text[:-3]:
        text=text[:-3].rstrip()
    if text.startswith('```'):
        lines=text.splitlines()
        if len(lines)>=3 and lines[-1].strip()=='```':text='\n'.join(lines[1:-1])
    try:obj=json.loads(text);repair=None
    except json.JSONDecodeError as exc:
        # A model may echo the same object once more inside a trailing
        # ```json fence. Accept this only when both decoded objects are
        # byte-for-byte equivalent; conflicting extra JSON stays invalid.
        try:
            first,end=json.JSONDecoder().raw_decode(text)
            tail=text[end:].strip()
            if tail.startswith('```'):
                lines=tail.splitlines()
                if lines and lines[0].strip().startswith('```') and lines[-1].strip()=='```':
                    duplicate=json.loads('\n'.join(lines[1:-1]))
                    if duplicate==first:
                        return first,{'kind':'remove_duplicate_fenced_json','characters':'suffix'}
        except (json.JSONDecodeError,ValueError,TypeError):
            pass
        # Some local VLM responses repeat an identical top-level candidates
        # object after explanatory paragraphs. Accept this only when every
        # repeated top-level candidates object decodes equally and the prose
        # contains no other braces; conflicting JSON remains invalid.
        try:
            decoder=json.JSONDecoder()
            first,end=decoder.raw_decode(text)
            marker='{"candidates"'
            repeated=[]
            cursor=end
            dropped_incomplete_tail=False
            while True:
                second_pos=text.find(marker,cursor)
                if second_pos < 0:
                    break
                if '{' in text[cursor:second_pos] or '}' in text[cursor:second_pos]:
                    repeated=[]
                    break
                try:
                    second,second_end=decoder.raw_decode(text,second_pos)
                except json.JSONDecodeError:
                    tail=text[second_pos:]
                    if (repeated and tail.count('{') > tail.count('}')
                            and text.find(marker,second_pos+len(marker)) < 0):
                        dropped_incomplete_tail=True
                        break
                    repeated=[]
                    break
                repeated.append((second,second_end))
                cursor=second_end
            if (isinstance(first,dict) and set(first)=={'candidates'}
                    and isinstance(first['candidates'],list) and repeated
                    and all(second == first for second,_ in repeated)
                    and (dropped_incomplete_tail
                         or ('{' not in text[cursor:] and '}' not in text[cursor:]))):
                    repair={
                        'kind':'remove_duplicate_explanatory_suffix',
                        'objects':1+len(repeated),
                    }
                    if dropped_incomplete_tail:
                        repair['dropped_incomplete_tail']=True
                    return first,repair
        except (json.JSONDecodeError,ValueError,TypeError):
            pass
        # Qwen3-VL can emit one compact candidate object per line instead of
        # wrapping them in a single candidates array.  Repair this narrow,
        # unambiguous form only for consecutive top-level objects whose sole
        # key is ``candidates``; all other extra JSON remains invalid.  The
        # repair is recorded in the ledger and candidate parsing still applies
        # the configured candidate cap and schema checks.
        try:
            decoder=json.JSONDecoder();pos=0;objects=[];dropped_incomplete_tail=False
            while True:
                while pos<len(text) and text[pos].isspace():pos+=1
                if pos>=len(text):break
                try:
                    item,end=decoder.raw_decode(text,pos)
                except json.JSONDecodeError:
                    # If complete candidate objects precede a final truncated
                    # repetition, retain the complete set and record the drop.
                    tail=text[pos:].lstrip()
                    if (objects and tail.startswith('{') and '"candidates"' in tail
                            and tail.count('{')>tail.count('}')):
                        dropped_incomplete_tail=True
                        break
                    raise
                objects.append(item);pos=end
            if (len(objects)>=2 or dropped_incomplete_tail) and all(
                    isinstance(item,dict) and set(item)=={'candidates'}
                    and isinstance(item['candidates'],list) for item in objects):
                merged=[];seen=set()
                for item in objects:
                    for candidate in item['candidates']:
                        marker=json.dumps(candidate,sort_keys=True,ensure_ascii=False)
                        if marker not in seen:
                            seen.add(marker);merged.append(candidate)
                repair={'kind':'merge_consecutive_candidate_objects',
                        'objects':len(objects),
                        'deduplicated':sum(len(x['candidates']) for x in objects)-len(merged)}
                if dropped_incomplete_tail:
                    repair['dropped_incomplete_tail']=True
                return {'candidates':merged},repair
        except (json.JSONDecodeError,ValueError,TypeError):
            pass
        # Qwen3-VL can omit one closer either after a complete top-level array
        # or just before that array's final close. No other malformed JSON is repaired.
        if text.count('{')!=text.count('}')+1:raise SchemaError(f'Invalid JSON model output: {exc}') from exc
        if text.endswith(']'):
            candidate=text+'}';repair={'kind':'append_outer_object_closer','characters':'}'}
        elif text.endswith(']}'):
            candidate=text[:-2]+'}'+text[-2:];repair={'kind':'insert_object_closer_before_final_array','characters':'}'}
        else:raise SchemaError(f'Invalid JSON model output: {exc}') from exc
        try:obj=json.loads(candidate)
        except json.JSONDecodeError:raise SchemaError(f'Invalid JSON model output: {exc}') from exc
    if not isinstance(obj,dict):raise SchemaError('Expected a JSON object')
    return obj,repair

def json_object(text:str|dict)->dict:
    return json_object_with_audit(text)[0]

@dataclass(frozen=True)
class Frame:
    index:int
    timestamp:float
    path:str
    sha256:str
    width:int
    height:int

@dataclass(frozen=True)
class PublicInput:
    uid:str
    instruction:str
    camera_perspective:str
    spatial_reference:str
    video_path:str|None=None
    frames_manifest:str|None=None
    synthetic:bool=False
    fixture_path:str|None=None

    def prompt_context(self)->dict:
        # Routing identifiers, filenames, task categories and goals NEVER enter prompts.
        return {'instruction':self.instruction,'camera_perspective':self.camera_perspective,
                'spatial_reference':self.spatial_reference}

    @classmethod
    def from_dict(cls,d:dict)->'PublicInput':
        allowed=set(cls.__dataclass_fields__)
        unexpected=set(d)-allowed
        if unexpected:raise SchemaError(f'Forbidden/unknown public fields: {sorted(unexpected)}')
        x=cls(**d)
        if not x.uid or not x.instruction.strip():raise SchemaError('Empty uid/instruction')
        if x.spatial_reference not in ['camera','human']:raise SchemaError('Unsupported reference frame')
        if not x.video_path and not x.frames_manifest:raise SchemaError('Missing video/frames')
        if x.fixture_path and not x.synthetic:raise SchemaError('Fixture observations are forbidden for real data')
        return x

@dataclass(frozen=True)
class Slot:
    id:str
    role:str
    description:str
    kind:str='object'
    needs_video:bool=True
    explicit_value:str|None=None
    temporal:str|None=None
    event_action:str|None=None

@dataclass(frozen=True)
class ObjectRef:
    id:str
    description:str

@dataclass(frozen=True)
class Event:
    id:str
    start:float
    end:float
    action:str
    object_ids:tuple[str,...]
    frame_ids:tuple[int,...]
    relations:tuple[Any,...]=()
    uncertain:bool=False


_RELATION_ROLE_ALIASES={
    'actor':'actor','agent':'actor','person':'actor','human':'actor',
    'target_object':'manipulated_object','object':'manipulated_object',
    'manipulated_object':'manipulated_object','pointed_object':'manipulated_object',
    'target_container':'target_container','destination':'target_container',
    'destination_id':'target_container','target_location':'target_container',
    'location':'target_container','source_location':'source_location',
    # Existing WatchAct event tables commonly use ``source`` with an object ID
    # and an object description (not a spatial location).  Preserve that
    # observed schema as manipulated-object evidence.
    'source':'manipulated_object','source_object':'manipulated_object',
}

_CUE_ACTION_WORDS=('point','gesture','indicate','touch')
_MANIPULATION_ACTION_WORDS=('pick','place','move','put','transfer','bring','take','lift','grab')


def _action_contains(action:str,words:tuple[str,...])->bool:
    text=str(action or '').strip().casefold()
    return any(word in text for word in words)


def _relation_values(value:Any)->list[str]:
    if isinstance(value,(list,tuple,set)):return [str(item) for item in value if item is not None]
    return [] if value is None else [str(value)]


def normalize_event_roles(event:Event)->dict:
    """Normalize public event relations while retaining raw conversion evidence."""
    mentions=[{'value':str(value),'role':'untyped_object','source':'object_ids'}
              for value in event.object_ids]
    conversions=[];wire_field_differences=[]
    relations=event.relations or ()
    legacy_pair=(len(relations)==2 and all(isinstance(value,str) for value in relations)
                 and str(relations[0]).strip().casefold() in _RELATION_ROLE_ALIASES)
    if legacy_pair:
        relation_type=str(relations[0]).strip().casefold()
        role=_RELATION_ROLE_ALIASES[relation_type]
        value=str(relations[1])
        conversions.append({'index':0,'status':'normalized_legacy_pair',
                            'type':relation_type,'role':role,'values':[value],
                            'source':'legacy_pair[1]',
                            'raw':copy.deepcopy(list(relations))})
        mentions.append({'value':value,'role':role,
                         'source':'relations_legacy_pair',
                         'relation_type':relation_type})
        relations=()
    for index,relation in enumerate(relations):
        if not isinstance(relation,dict):
            conversions.append({'index':index,'status':'unknown_wire_format',
                                'raw':copy.deepcopy(relation)})
            continue
        relation_type=str(relation.get('type','')).strip().casefold()
        role=_RELATION_ROLE_ALIASES.get(relation_type,'unknown')
        values=[];source=None
        # When both fields exist, ``id`` is commonly the relation record ID
        # (for example r1) while ``object_id`` is the referenced entity.
        for key in ('object_id','id'):
            if key in relation:
                values=_relation_values(relation.get(key));source=key;break
        if 'object_id' in relation and 'id' in relation:
            object_id_values=_relation_values(relation.get('object_id'))
            id_values=_relation_values(relation.get('id'))
            if object_id_values!=id_values:
                wire_field_differences.append({
                    'index':index,
                    'object_id_values':object_id_values,
                    'id_values':id_values,
                    'resolution':'object_id_preferred_id_retained_as_raw',
                })
        if not values:
            for key in _RELATION_ROLE_ALIASES:
                if key in relation:
                    values=_relation_values(relation.get(key));source=key
                    if not relation_type:
                        relation_type=key;role=_RELATION_ROLE_ALIASES[key]
                    break
        status='normalized' if values and role!='unknown' else (
            'unknown_role' if values else 'missing_identity')
        conversions.append({'index':index,'status':status,'type':relation_type or None,
                            'role':role,'values':values,'source':source,
                            'raw':copy.deepcopy(relation)})
        mentions.extend({'value':value,'role':role,'source':f'relations[{index}]',
                         'relation_type':relation_type or None} for value in values)
    by_value={}
    for mention in mentions:by_value.setdefault(mention['value'],set()).add(mention['role'])
    conflicts=[{'value':value,'roles':sorted(roles-{'untyped_object'})}
               for value,roles in by_value.items()
               if len(roles-{'untyped_object','unknown'})>1]
    return {'event_id':event.id,'mentions':mentions,'conversions':conversions,
            'conflicts':conflicts,'wire_field_differences':wire_field_differences}


def expected_event_role(slot:Slot)->str:
    role=str(slot.role or '').strip().casefold()
    if role in _RELATION_ROLE_ALIASES:return _RELATION_ROLE_ALIASES[role]
    if slot.kind=='location':return 'target_container'
    if slot.kind=='object':return 'manipulated_object'
    return 'untyped_object'


def event_supports(event:Event,value:str,expected_role:str)->dict:
    """Return an auditable role-aware evidence decision for one event/value."""
    normalized=normalize_event_roles(event);value=str(value)
    mentions=[m for m in normalized['mentions'] if m['value']==value]
    explicit={m['role'] for m in mentions if m['role']!='untyped_object'}
    allowed={expected_role,'untyped_object'}
    explicit_allowed=explicit & allowed
    explicit_incompatible=explicit-allowed-{'unknown'}
    if explicit_allowed and explicit_incompatible:
        status='conflict';reason='evidence_role_conflict'
    elif explicit_allowed:
        status='supported';reason='explicit_role_match'
    elif explicit_incompatible:
        status='unsupported';reason='evidence_role_mismatch'
    elif 'unknown' in explicit:
        status='unknown';reason='unknown_relation_role'
    elif any(m['role']=='untyped_object' for m in mentions):
        other_explicit={m['value'] for m in normalized['mentions']
                        if m['role']==expected_role and m['value']!=value}
        if other_explicit:
            status='unsupported';reason='explicit_role_names_other_identity'
        elif _action_contains(event.action,_CUE_ACTION_WORDS):
            # In the historical event wire format, object_ids on pointing and
            # gesture rows frequently identify the acting person.  Without an
            # explicit target_object/target_container relation, promoting such
            # an untyped identity to an operation parameter recreates the
            # actor-as-target bug identified in the teacher audit.
            status='unknown';reason='ambiguous_cue_participant'
        elif expected_role=='manipulated_object' and _action_contains(
                event.action,_MANIPULATION_ACTION_WORDS):
            status='supported';reason='legacy_manipulation_object_match'
        else:
            status='unknown';reason='untyped_object_role_unknown'
    elif mentions:
        status='unknown';reason='unknown_relation_role'
    else:
        status='unsupported';reason='identity_not_mentioned'
    return {'event_id':event.id,'value':value,'expected_role':expected_role,
            'status':status,'reason_code':reason,'mentions':mentions,
            'conflicts':normalized['conflicts']}

@dataclass(frozen=True)
class Binding:
    value:str|None
    event_ids:tuple[str,...]=()

@dataclass
class Candidate:
    bindings:dict[str,Binding]

    def to_dict(self)->dict:
        return {'bindings':{k:{'value':v.value,'event_ids':list(v.event_ids)} for k,v in self.bindings.items()}}

    def signature(self)->str:
        return json.dumps(self.to_dict(),sort_keys=True)

    def replaced(self,slot:str,binding:Binding)->'Candidate':
        d=dict(self.bindings);d[slot]=binding;return Candidate(d)


def parse_slots(obj:dict,max_slots:int=24)->list[Slot]:
    raw=obj.get('slots')
    if not isinstance(raw,list) or not 1<=len(raw)<=max_slots:raise SchemaError('Expected 1..max_slots slots')
    slots=[]
    for r in raw:
        if not isinstance(r,dict):raise SchemaError('Invalid slot')
        try:s=Slot(**r)
        except TypeError as exc:raise SchemaError(str(exc)) from exc
        if not isinstance(s.needs_video,bool):raise SchemaError('needs_video must be boolean')
        if not s.id or s.kind not in ['object','location','state','action','text']:raise SchemaError('Invalid slot type/id')
        if s.temporal not in [None,'first','last','before','after','ordinal']:raise SchemaError('Invalid temporal constraint')
        if s.explicit_value is not None and not isinstance(s.explicit_value,str):raise SchemaError('explicit_value must be string/null')
        slots.append(s)
    if len({s.id for s in slots})!=len(slots):raise SchemaError('Duplicate slot id')
    return slots


def parse_events(obj:dict,frames:list[Frame],max_events:int=64)->tuple[list[ObjectRef],list[Event]]:
    try:objects=[ObjectRef(**r) for r in obj['objects']]
    except (TypeError,KeyError) as exc:raise SchemaError('Invalid object inventory') from exc
    oids={o.id for o in objects}
    if len(oids)!=len(objects):raise SchemaError('Duplicate object identity')
    raw=obj.get('events');events=[];allowed={f.index:f for f in frames}
    if not isinstance(raw,list) or len(raw)>max_events:raise SchemaError('Invalid/too many events')
    for r in raw:
        if not isinstance(r,dict):raise SchemaError('Invalid event')
        try:
            rr=dict(r)
            for field in ['object_ids','frame_ids','relations']:rr[field]=tuple(rr.get(field,[]))
            e=Event(**rr)
        except TypeError as exc:raise SchemaError(str(exc)) from exc
        if not e.id or not e.frame_ids or set(e.frame_ids)-set(allowed):raise SchemaError('Event references unavailable frames')
        if set(e.object_ids)-oids:raise SchemaError('Event references unknown object')
        if not math.isfinite(e.start) or not math.isfinite(e.end) or e.start>e.end or e.start<0:raise SchemaError('Invalid event time interval')
        # Frame references are the authoritative temporal evidence.  At sampling
        # rates other than 1 Hz, vision models sometimes copy frame indices or
        # an old time grid into start/end.  Keep rejecting non-finite/negative
        # intervals above, but canonicalize a finite inconsistent interval to
        # the cited frames' real timestamps so temporal binding remains valid.
        cited_times=[allowed[i].timestamp for i in e.frame_ids]
        if any(t<e.start-1.01 or t>e.end+1.01 for t in cited_times):
            rr['start']=min(cited_times);rr['end']=max(cited_times)
            e=Event(**rr)
        events.append(e)
    if len({e.id for e in events})!=len(events):raise SchemaError('Duplicate event id')
    return objects,events


def validate_candidate_details(candidate:Candidate,slots:list[Slot],objects:list[ObjectRef],events:list[Event])->list[dict]:
    errors=[];smap={s.id:s for s in slots};emap={e.id:e for e in events};oids={o.id for o in objects}
    def add(code,message,slot_id=None,event_ids=(),support=()):
        errors.append({'code':code,'message':message,'slot_id':slot_id,
                       'event_ids':list(event_ids),'support':list(support)})
    if set(candidate.bindings)!=set(smap):
        add('binding_coverage_mismatch','candidate must cover exactly the parsed slots')
        return errors
    for sid,b in candidate.bindings.items():
        s=smap[sid]
        if b.value is not None and not isinstance(b.value,str):
            add('invalid_value_type',f'{sid}: value must be string/null',sid);continue
        unknown=sorted(set(b.event_ids)-set(emap))
        if unknown:
            add('unknown_evidence',f'{sid}: unknown evidence',sid,unknown);continue
        if len(set(b.event_ids))!=len(b.event_ids):
            add('duplicate_evidence',f'{sid}: duplicate evidence',sid,b.event_ids)
        if s.explicit_value is not None and b.value!=s.explicit_value:
            add('changed_explicit_instruction',f'{sid}: changed explicit instruction',sid,b.event_ids)
        if b.value is None:continue
        if s.needs_video and not b.event_ids:
            add('missing_video_evidence',f'{sid}: video-dependent value lacks evidence',sid)
        if s.kind in {'object','location'} and s.needs_video and s.explicit_value is None:
            if b.value not in oids:
                add('unseen_object_identity',f'{sid}: unseen object identity',sid,b.event_ids)
            if b.event_ids:
                role=expected_event_role(s)
                support=[event_supports(emap[event_id],b.value,role) for event_id in b.event_ids]
                if any(item['status']=='conflict' for item in support):
                    add('evidence_role_conflict',f'{sid}: cited evidence has conflicting roles',
                        sid,b.event_ids,support)
                elif not any(item['status']=='supported' for item in support):
                    codes=sorted({item['reason_code'] for item in support})
                    code='evidence_role_mismatch' if 'evidence_role_mismatch' in codes else (
                        'unknown_relation_role' if 'unknown_relation_role' in codes else
                        'cited_events_do_not_support_role')
                    add(code,f'{sid}: cited events do not support {role} ({", ".join(codes)})',
                        sid,b.event_ids,support)
    return errors


def validate_candidate(candidate:Candidate,slots:list[Slot],objects:list[ObjectRef],events:list[Event])->list[str]:
    return [error['message'] for error in validate_candidate_details(candidate,slots,objects,events)]


def parse_candidates(obj,slots,objects,events,max_candidates=4)->tuple[list[Candidate],list[dict]]:
    raw=obj.get('candidates')
    if not isinstance(raw,list):raise SchemaError('Expected candidates list')
    result=[];rejected=[];seen=set()
    for i,r in enumerate(raw[:max_candidates]):
        details=[]
        try:
            if set(r)!={'bindings'}:raise SchemaError('Candidate must only contain bindings')
            c=Candidate({k:Binding(v['value'],tuple(v.get('event_ids',[]))) for k,v in r['bindings'].items()})
            details=validate_candidate_details(c,slots,objects,events)
            if details:raise SchemaError('; '.join(error['message'] for error in details))
            if c.signature() not in seen:seen.add(c.signature());result.append(c)
        except (TypeError,KeyError,SchemaError,AttributeError) as exc:
            item={'index':i,'reason':str(exc)}
            if details:
                item['reason_codes']=[error['code'] for error in details]
                item['errors']=details
            rejected.append(item)
    return result,rejected


def binding_matrix(c:Candidate,slots:list[Slot],events:list[Event])->list[list[int]]:
    return [[int(e.id in c.bindings[s.id].event_ids) for s in slots] for e in events]


def difference(a:Candidate,b:Candidate)->list[str]:
    return sorted(k for k in set(a.bindings)|set(b.bindings) if a.bindings.get(k)!=b.bindings.get(k))


def normalize_plan(obj:dict)->dict:
    # Syntax validation only; do not add Open/Pick actions or consult goal conditions.
    if set(obj)-{'Objects','Actions'}:raise SchemaError('Unknown plan fields')
    objects=obj.get('Objects');actions=obj.get('Actions')
    if not isinstance(objects,list) or not isinstance(actions,list) or not actions:raise SchemaError('Nonempty Actions and Objects list required')
    names=set();oo=[];aa=[]
    for o in objects:
        if not isinstance(o,dict) or set(o)!={'name','description'}:raise SchemaError('Invalid plan object')
        if not isinstance(o['name'],str) or not o['name'] or o['name'] in names:raise SchemaError('Duplicate/invalid plan object name')
        if not isinstance(o['description'],str) or not o['description'].strip():raise SchemaError('Missing object description')
        names.add(o['name']);oo.append(dict(o))
    for action in actions:
        if not isinstance(action,dict):raise SchemaError('Invalid action')
        c=str(action.get('command','')).lower()
        if c not in ['pick','place','open','close']:raise SchemaError('Unknown action primitive')
        key='source_location' if c=='pick' else 'target_location'
        if set(action)-{'command','object',key}:raise SchemaError('Unexpected action argument')
        if action.get('object') not in names:raise SchemaError('Action references unknown object')
        if not isinstance(action.get(key),str) or not action[key].strip():raise SchemaError('Missing action location')
        aa.append({'command':c,'object':action['object'],key:action[key]})
    return {'Objects':oo,'Actions':aa}
