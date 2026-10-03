"""Compile already resolved canonical operation pairs; never infer identities.

Callers must establish visual-to-canonical mappings independently. Region names
must already be in this request's coordinate frame (including public remapping).
This module
does not approve EvidenceFacts and does not read evaluation metadata or goals.
Supports one final placement per object, preserving initial drawer open state.
"""
from __future__ import annotations
import re


def region_catalog(objects, initial_states):
    regions = {f'main_{depth}_{side}_region' for depth in ('front','middle','back')
               for side in ('left','center','right')}
    regions |= {'fixture_front_region','fixture_back_region'}
    for name in objects:
        if name.startswith(('basket_', 'wooden_tray_')):
            regions.add(name+'_contain_region')
        elif name.startswith('wooden_cabinet_'):
            regions |= {name+'_'+tier+'_region' for tier in ('top','middle','bottom')}
            regions.add(name+'_top_side')
    regions |= {row[-1] for row in initial_states if len(row) in (2,3)}
    return regions


def drawer(region):
    return bool(re.fullmatch(r'wooden_cabinet_\d+_(top|middle|bottom)_region',region))


def relation_for(region):
    return 'In' if drawer(region) or region.endswith('_contain_region') else 'On'


def compile_resolved_pairs(*, objects, initial_states, pairs):
    """Return simulator-compatible actions and a literal translator payload.

    pairs: iterable of {object_id, destination_region, relation}. All three
    fields are mandatory; object IDs are canonical and regions use the request
    coordinate frame. Reject uncertain or conflicting input.
    No fallback guessing, category-to-instance matching, or partial success.
    """
    if not objects or len(objects)!=len(set(objects)):
        raise ValueError('object inventory must be nonempty and unique')
    positions, gates = {}, {}
    for row in initial_states:
        if len(row)==3 and row[0] in ('On','In'):
            _,obj,region=row
            if obj not in objects or obj in positions:
                raise ValueError('invalid or duplicate initial object')
            positions[obj]=region
        elif len(row)==2 and row[0] in ('Open','Close'):
            if row[1] in gates or not drawer(row[1]):
                raise ValueError('invalid or duplicate drawer state')
            gates[row[1]]=row[0]
        else:
            raise ValueError('unsupported initial state')
    regions=region_catalog(objects,initial_states)
    pairs=list(pairs)
    if not pairs:
        raise ValueError('empty pairs do not define a supported task')
    seen=set();actions=[]
    for pair in pairs:
        obj=pair['object_id'];target=pair['destination_region'];rel=pair['relation']
        if obj in seen or obj not in positions:
            raise ValueError('duplicate or unresolved object')
        if obj.startswith(('wooden_cabinet_','wooden_tray_','basket_')):
            raise ValueError('moving fixtures is outside compiler scope')
        seen.add(obj)
        if target not in regions or rel!=relation_for(target):
            raise ValueError('unknown region or incompatible relation')
        source=positions[obj]
        opened=[]
        for region in dict.fromkeys((source,target)):
            if drawer(region):
                if region not in gates:
                    raise ValueError('drawer initial state is unknown')
                if gates[region]=='Close':
                    actions.append(['open',region]);opened.append(region)
        actions.extend([['pick',obj,source],['place',obj,target]])
        for region in reversed(opened):actions.append(['close',region])
    payload_objects=[{'source_name':name,'source_description':name,
                     'unified_name':name,'task_object_id':name,
                     'unified_region':positions.get(name),'matched':name in positions}
                     for name in objects]
    payload_actions=[]
    for action in actions:
        cmd=action[0]
        if cmd in ('open','close'):
            region=action[1];obj=region.rsplit('_',2)[0]
            src=positions.get(obj);dest=region
        else:
            obj=action[1];src=positions[obj];dest=action[2] if cmd=='place' else None
        payload_actions.append({'command':cmd,'object_source_name':obj,
           'object_unified_name':obj,'source_location_text':src,
           'target_location_text':dest,'source_region_id':src,'target_region_id':dest,
           'source_location_matched':src is not None,'target_location_matched':dest is not None})
    return {'actions':actions,'translator_payload':{'objects':payload_objects,'actions':payload_actions}}


def compile_literal_action_pairs(*, objects, initial_states, actions):
    """Conservative NC transport repair, preserving every predicted pair.

    Accept already translated complete PICK/PLACE pairs only. Unknown actions,
    missing objects, repeated placements and source mismatches fail the whole
    proposal. Identity correctness remains the upstream predictor's responsibility.
    This function never verifies visual evidence or upgrades human approval.
    """
    positions={r[1]:r[2] for r in initial_states if len(r)==3}
    pending=None;pairs=[];explicit_gates=[]
    for action in actions:
        cmd=str(action.get('command','')).lower()
        obj=action.get('object_unified_name')
        if obj not in objects:
            raise ValueError('unresolved action object')
        if cmd in ('open','close'):
            if pending is not None:
                raise ValueError('interleaved action in pick/place pair')
            target=action.get('target_region_id')
            if not isinstance(target,str) or not drawer(target):
                raise ValueError('unsupported gate action')
            if target.rsplit('_',2)[0]!=obj:
                raise ValueError('gate identity mismatch')
            explicit_gates.append(target)
        elif cmd=='pick':
            if pending is not None or action.get('source_region_id')!=positions.get(obj):
                raise ValueError('missing placement or source mismatch')
            pending=obj
        elif cmd=='place':
            if pending!=obj:
                raise ValueError('unpaired placement')
            dest=action.get('target_region_id')
            if not isinstance(dest,str):raise ValueError('unresolved destination')
            pairs.append({'object_id':obj,'destination_region':dest,'relation':relation_for(dest)})
            pending=None
        else:raise ValueError('unsupported action')
    if pending is not None:raise ValueError('missing final placement')
    if len(pairs)>3:raise ValueError('outside NC first-version scope')
    touched={p['destination_region'] for p in pairs}|{positions.get(p['object_id']) for p in pairs}
    if any(g not in touched for g in explicit_gates):
        raise ValueError('independent gate instruction outside scope')
    return compile_resolved_pairs(objects=objects,initial_states=initial_states,pairs=pairs)
