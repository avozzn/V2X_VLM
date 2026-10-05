"""V2 association experiment. No cooperative labels or identity inputs.

All variants use raw dimensions and causal compensated centers. L1 removes
only the second distance gate. L2 allows unknown motion for messages <=100ms
only with cost<=0.45 and BEV IoU>=0.1. These are engineering defaults, not
calibrated confidence. Ambiguous or old unknown-motion pairs remain separate.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .core import pair_features
from .stable_fusion import process


def associate_relaxed(vehicle, infra, age, age_aware=False):
    """Return candidate pairs, accepted pairs and geometric decisions."""
    if not vehicle or not infra:
        return [], [], []
    bev, _, distance, _ = pair_features(vehicle,infra)
    gates = np.array([2. if x['class']==0 else 1. for x in vehicle])[:,None]
    same = np.array([x['class'] for x in vehicle])[:,None] == np.array([x['class'] for x in infra])[None,:]
    yaw = np.arccos(np.clip(np.abs(np.cos(np.array([x['box'][6] for x in vehicle])[:,None]
                                        -np.array([x['box'][6] for x in infra])[None,:])),0,1))
    pedestrian = np.array([x['class']==1 for x in vehicle])[:,None]
    cost = np.where(pedestrian,.7,.6)*distance/gates + .3*(1-bev) + np.where(pedestrian,0,.1)*yaw/(np.pi/2)
    allowed = same & (distance <= gates)
    legal_cost = np.where(allowed,cost,np.inf)
    augmented = np.full((len(vehicle),len(infra)+len(vehicle)),1e6)
    augmented[:,:len(infra)] = np.where(allowed,cost,1e6)
    augmented[np.arange(len(vehicle)),len(infra)+np.arange(len(vehicle))] = .7
    aa,bb = linear_sum_assignment(augmented)
    candidates = [(int(a),int(b)) for a,b in zip(aa,bb) if b<len(infra) and allowed[a,b] and cost[a,b]<.7]
    accepted, decisions = [], []
    for a,b in candidates:
        row = legal_cost[a].copy(); row[b] = np.inf
        col = legal_cost[:,b].copy(); col[a] = np.inf
        row_margin, col_margin = float(row.min()-cost[a,b]), float(col.min()-cost[a,b])
        reasons=[]
        if row_margin<.1 or col_margin<.1: reasons.append('ambiguous_or_not_mutual_best')
        # Candidate distance gate remains; no second halving of the gate.
        if vehicle[a]['class']!=1 and yaw[a,b]>np.pi/6: reasons.append('axis_heading')
        if age is None or age>1: reasons.append('old_message')
        if age is not None and age>0 and not infra[b]['motion_valid']:
            if not age_aware:
                reasons.append('invalid_motion')
            elif age > .1:
                reasons.append('unknown_motion_old_message')
            elif cost[a,b] > .45 or bev[a,b] < .1:
                reasons.append('unknown_motion_weak_geometry')
        if not reasons: accepted.append((a,b))
        decisions.append({'vehicle_index':a,'infra_index':b,'cost':float(cost[a,b]),
                          'distance_m':float(distance[a,b]),'bev_iou':float(bev[a,b]),
                          'axis_difference_rad':float(yaw[a,b]),
                          'row_margin':row_margin if np.isfinite(row_margin) else None,
                          'column_margin':col_margin if np.isfinite(col_margin) else None,
                          'accepted':not reasons,'reasons':reasons})
    return candidates,accepted,decisions


def process_relaxed(vehicle, infra, age, variant):
    if variant == 'A1':
        return process(vehicle, infra, age, 'A1')
    if variant not in ('L1','L2'):
        raise ValueError(variant)
    candidates, accepted, decisions = associate_relaxed(vehicle, infra, age, variant=='L2')
    merged={a: {'box':vehicle[a]['box'].copy(),'class':vehicle[a]['class'],
               'vehicle_index':a,'infra_index':b,'source_bits':[1,1],
               'size_policy':'select_ego_raw'} for a,b in accepted}
    outputs=[merged.get(a, {'box':x['box'].copy(),'class':x['class'],
             'vehicle_index':a,'infra_index':None,'source_bits':[1,0],
             'size_policy':'single_ego_raw'}) for a,x in enumerate(vehicle)]
    used={b for a,b in accepted}
    outputs += [{'box':x['box'].copy(),'class':x['class'],'vehicle_index':None,
                 'infra_index':b,'source_bits':[0,1],'size_policy':'single_rsu_raw'}
                for b,x in enumerate(infra) if b not in used]
    return {'variant':variant,'candidates':candidates,'accepted':accepted,
            'decisions':decisions,'outputs':outputs}
