"""Causal box processing; never reads cooperative object labels."""
import bisect
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from .core import CLASSES, native_box, pair_features, match
from .run import NativeDataset, causal_objects


class HistoryDataset(NativeDataset):
    def __init__(self, root):
        super().__init__(root)
        self.side_timelines = defaultdict(list)
        for side, frames in self.frames.items():
            for frame, row in frames.items():
                self.side_timelines[side, str(row['sequence_id'])].append(
                    (int(row['pointcloud_timestamp']), frame))
        for timeline in self.side_timelines.values():
            timeline.sort()

    def dimensions(self, side, frame, obj):
        meta = self.frames[side][frame]
        now, sequence = int(meta['pointcloud_timestamp']), str(meta['sequence_id'])
        raw = obj['box'][3:6].copy() if side == 'vehicle-side' else obj['raw_box'][3:6].copy()
        result = {'raw_lwh': raw, 'stable_lwh': raw.copy(), 'size_stable_valid': False,
                  'size_history_count': 1, 'size_mad_lwh': np.zeros(3),
                  'size_relative_mad': np.zeros(3), 'size_history': [{'frame':frame,'timestamp_us':now}],
                  'size_status': 'unsupported_class' if obj['class'] != 0 else 'insufficient_history'}
        if obj['class'] != 0:
            return result
        track = obj['track_id']
        current = [x for x in self.objects(side,frame) if str(x.get('track_id')) == track]
        if track in ('None','-1','') or len(current) != 1:
            result['size_status'] = 'invalid_current_track'; return result
        timeline = self.side_timelines[side, sequence]
        index = bisect.bisect_left(timeline, (now,'')) - 1
        sizes, history = [raw], result['size_history']
        while index >= 0 and now-timeline[index][0] <= 1_000_000 and len(sizes) < 5:
            timestamp, previous = timeline[index]
            rows = [x for x in self.objects(side, previous) if str(x.get('track_id')) == track]
            if len(rows) > 1:
                result['size_status'] = 'duplicate_history_track'; return result
            if rows:
                if CLASSES.get(rows[0]['type'].lower()) != obj['class']:
                    result['size_status'] = 'history_class_conflict'; return result
                sizes.append(native_box(rows[0])[3:6])
                history.append({'frame':previous,'timestamp_us':timestamp})
            index -= 1
        result['size_history_count'] = len(sizes)
        if len(sizes) >= 3:
            stable = np.median(sizes, axis=0)
            mad = np.median(np.abs(np.array(sizes)-stable), axis=0)
            result.update(stable_lwh=stable, size_stable_valid=True,
                          size_mad_lwh=mad,size_relative_mad=mad/stable,size_status='valid')
        return result


def prepare_objects(dataset, pair):
    vehicle, infra, selected, age = causal_objects(dataset, pair)
    for side, frame, rows in [('vehicle-side',str(pair['vehicle_frame']),vehicle),
                              ('infrastructure-side',selected,infra)]:
        for obj in rows:
            obj.update(dataset.dimensions(side,frame,obj))
            obj['source_frame'] = frame
            obj['source_sequence'] = str(dataset.frames[side][frame]['sequence_id'])
    return vehicle, infra, selected, age


def variant_objects(vehicle, infra, variant):
    stable = variant in ('A2','A3','A4')
    v, r = [], []
    for objects, target, roadside in [(vehicle,v,False),(infra,r,True)]:
        for obj in objects:
            box = obj['spatial_box'].copy() if roadside and variant == 'A0' else obj['box'].copy()
            if stable:
                box[3:6] = obj['stable_lwh']
            target.append({**obj,'box':box})
    return v,r


def soft_associate(vehicle, infra, age):
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
        if distance[a,b]>gates[a,0]/2: reasons.append('center_distance')
        if vehicle[a]['class']!=1 and yaw[a,b]>np.pi/6: reasons.append('axis_heading')
        if age is None or age>1: reasons.append('old_message')
        if age is not None and age>0 and not infra[b]['motion_valid']: reasons.append('invalid_motion')
        if not reasons: accepted.append((a,b))
        decisions.append({'vehicle_index':a,'infra_index':b,'cost':float(cost[a,b]),
                          'distance_m':float(distance[a,b]),'bev_iou':float(bev[a,b]),
                          'axis_difference_rad':float(yaw[a,b]),
                          'row_margin':row_margin if np.isfinite(row_margin) else None,
                          'column_margin':col_margin if np.isfinite(col_margin) else None,
                          'accepted':not reasons,'reasons':reasons})
    return candidates,accepted,decisions


def fuse_size(ego, rsu, blend):
    relative = np.abs(ego['box'][3:6]-rsu['box'][3:6])/np.maximum(ego['box'][3:6],rsu['box'][3:6])
    eligible = (ego['class']==0 and ego['size_stable_valid'] and rsu['size_stable_valid']
                and (ego['size_relative_mad']<=.1).all() and (rsu['size_relative_mad']<=.1).all()
                and (relative<=.2).all())
    box = ego['box'].copy()
    if blend and eligible: box[3:6]=(ego['box'][3:6]+rsu['box'][3:6])/2
    return box, {'size_policy':'blend_stable' if blend and eligible else 'select_ego',
                 'blend_eligible':bool(eligible),'size_conflict':bool((relative>.2).any())}


def process(vehicle, infra, age, variant):
    v,r = variant_objects(vehicle,infra,variant)
    if variant in ('A0','A1','A2'):
        candidates = accepted = match(v,r,pair_features(v,r),'A',(2.,1.,1.),.1)
        decisions = [{'vehicle_index':a,'infra_index':b,'accepted':True,'reasons':[],
                      'acceptance_policy':'legacy_hard_iou'} for a,b in accepted]
    else:
        candidates,accepted,decisions=soft_associate(v,r,age)
    merged = {}
    for a,b in accepted:
        box,info=fuse_size(v[a],r[b],variant=='A4')
        merged[a]= {'box':box,'class':v[a]['class'],'vehicle_index':a,'infra_index':b,
                    'source_bits':[1,1],**info}
    outputs = [merged.get(a,{'box':x['box'],'class':x['class'],'vehicle_index':a,
                             'infra_index':None,'source_bits':[1,0],'size_policy':'single_ego'}) for a,x in enumerate(v)]
    used={b for a,b in accepted}
    outputs += [{'box':x['box'],'class':x['class'],'vehicle_index':None,'infra_index':b,
                 'source_bits':[0,1],'size_policy':'single_rsu'} for b,x in enumerate(r) if b not in used]
    return {'variant':variant,'vehicle':v,'infra':r,'candidates':candidates,
            'accepted':accepted,'decisions':decisions,'outputs':outputs}
