"""Evaluation only: canonical pairs -> deterministic compile -> official WatchAct simulation and scoring."""
import sys
from .common import WATCHACT_SCRIPTS, goals
from .third_party.teb.nc_plan_compiler import compile_resolved_pairs, relation_for


def score_pairs(req, pairs, max_pairs=3):
    import sys
    for path in WATCHACT_SCRIPTS:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        import run_action_plan_simulation as sim
        import run_action_plan_scoring as scoring
    except ImportError as exc:
        raise RuntimeError('Install the pinned WatchAct repository and set IAE_WATCHACT_ROOT before scoring') from exc
    task = goals()[req['activity_id']]
    row = {'uid': req['uid'], 'success': False, 'strict': False, 'progress': 0.0, 'failure': None}
    try:
        if not isinstance(pairs, list) or not 1 <= len(pairs) <= max_pairs: raise ValueError('unsupported_pair_count')
        pp = [{'object_id': p['object_id'], 'destination_region': p['destination_region'], 'relation': relation_for(p['destination_region'])} for p in pairs]
        comp = compile_resolved_pairs(objects=req['objects'], initial_states=req['initial_states'], pairs=pp)
        flag, state, errs = sim.simulate_actions(req['initial_states'], comp['actions'])
        sc = scoring.evaluate_final_state(state, {'initial_state': req['initial_states'], 'goal_state': task['final_goal'],
                                                  'spatial_reference': 'camera', 'camera_perspective': 'front'})
        row.update(success=sc['Plan Success Rate'] == 1, strict=bool(sc['all_success']), progress=float(sc['Progress Rate']),
                   exact=sorted((p['object_id'], p['destination_region']) for p in pairs) == gold_pairs(req['activity_id']))
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        row['failure'] = str(e)[:120]
    return row


def gold_pairs(activity_id):
    return sorted((g[1], g[2]) for g in goals()[activity_id]['interest_objects_final_states'])
