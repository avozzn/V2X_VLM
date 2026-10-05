"""Audit cooperative links, export causal native objects, sweep association.

Run from project root: python -m tools.association.run --output-dir <new-dir>
Uses CPU only. Cooperative links are isolated from the geometry matcher.
"""
import argparse
import bisect
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .core import CLASSES, native_box, transform_box, extrapolate, pair_features, match


def read(path):
    return json.loads(Path(path).read_text())


def calibration(path):
    row = read(path)
    row = row.get('transform', row)
    matrix = np.eye(4)
    matrix[:3, :3] = row['rotation']
    matrix[:3, 3] = np.array(row['translation']).reshape(3)
    if not np.isfinite(matrix).all() or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=2e-5):
        raise ValueError(f'Invalid rigid calibration: {path}')
    return matrix


class NativeDataset:
    def __init__(self, root):
        self.root = Path(root)
        self.frames = {}
        self.timelines = defaultdict(list)
        self.labels = {}
        self.poses = {}
        for side in ('vehicle-side', 'infrastructure-side'):
            self.frames[side] = {str(x['frame_id']): x for x in read(self.root/side/'data_info.json')}
        for row in self.frames['infrastructure-side'].values():
            self.timelines[str(row['sequence_id'])].append((int(row['pointcloud_timestamp']), str(row['frame_id'])))
        for timeline in self.timelines.values():
            timeline.sort()

    def objects(self, side, frame):
        key = (side, frame)
        if key not in self.labels:
            row = self.frames[side][frame]
            path = (self.root/side/row['label_lidar_std_path']).resolve()
            expected = (self.root/side/'label'/('lidar' if side == 'vehicle-side' else 'virtuallidar')).resolve()
            if not path.is_relative_to(expected):
                raise ValueError('Non-native label path')
            self.labels[key] = read(path)
        return self.labels[key]

    def pose(self, side, frame):
        key = (side, frame)
        if key not in self.poses:
            row = self.frames[side][frame]
            root = self.root/side
            if side == 'vehicle-side':
                pose = calibration(root/row['calib_novatel_to_world_path']) @ calibration(root/row['calib_lidar_to_novatel_path'])
            else:
                pose = calibration(root/row['calib_virtuallidar_to_world_path'])
            self.poses[key] = pose
        return self.poses[key]

    def latest_history(self, frame, obj):
        row = self.frames['infrastructure-side'][frame]
        sequence = str(row['sequence_id'])
        now = int(row['pointcloud_timestamp'])
        timeline = self.timelines[sequence]
        index = bisect.bisect_left(timeline, (now, '')) - 1
        while index >= 0 and now - timeline[index][0] <= 1_000_000:
            timestamp, previous = timeline[index]
            candidates = [x for x in self.objects('infrastructure-side', previous)
                          if str(x.get('track_id')) == str(obj.get('track_id'))]
            if len(candidates) > 1:
                return None, None, 'duplicate_history_track'
            if candidates:
                if CLASSES.get(candidates[0]['type'].lower()) != CLASSES.get(obj['type'].lower()):
                    return None, None, 'history_class_conflict'
                return previous, candidates[0], 'valid'
            index -= 1
        return None, None, 'no_recent_history'


def resolve_reference(dataset, pair, obj):
    result = {'vehicle_frame': str(pair['vehicle_frame']),
              'infra_frame': str(pair['infrastructure_frame']),
              'cooperative_track_id': obj.get('track_id'), 'from_side': obj.get('from_side')}
    errors = []
    resolved = []
    for side, short, frame_key, sequence_key in [('vehicle-side', 'veh', 'vehicle_frame', 'vehicle_sequence'),
                                                 ('infrastructure-side', 'inf', 'infrastructure_frame', 'infrastructure_sequence')]:
        frame = str(pair[frame_key])
        track, token = obj.get(short+'_track_id'), obj.get(short+'_token')
        result[short+'_track_id'], result[short+'_token'] = track, token
        if track in (None, '', '-1') or token in (None, '', '-1'):
            errors.append(short+':missing_identity'); resolved.append(None); continue
        if frame not in dataset.frames[side]:
            errors.append(short+':missing_frame'); resolved.append(None); continue
        meta = dataset.frames[side][frame]
        if str(meta['sequence_id']) != str(pair[sequence_key]) or str(obj.get(short+'_frame_id')) != frame:
            errors.append(short+':frame_sequence_conflict')
        if str(obj.get(short+'_pointcloud_timestamp')) != str(meta['pointcloud_timestamp']):
            errors.append(short+':timestamp_conflict')
        try:
            rows = dataset.objects(side, frame)
        except FileNotFoundError:
            errors.append(short+':missing_label'); resolved.append(None); continue
        by_track = [x for x in rows if str(x.get('track_id')) == str(track)]
        by_token = [x for x in rows if x.get('token') == token]
        if len(by_track) != 1 or len(by_token) != 1:
            errors.append(short+':nonunique_or_missing_identity'); resolved.append(None); continue
        if by_track[0] is not by_token[0]:
            errors.append(short+':token_track_conflict'); resolved.append(None); continue
        resolved.append(by_track[0])
    result['status'] = 'valid' if not errors else 'unknown'
    result['errors'] = errors
    if all(x is not None for x in resolved):
        result['native_classes'] = [x['type'] for x in resolved]
        result['native_lwh'] = [native_box(x)[3:6].tolist() for x in resolved]
    return result


def reference_audit(dataset, pairs, output):
    lookup, counts = {}, Counter()
    with (output/'reference_pairs.jsonl').open('w') as stream:
        for pair in pairs:
            path = dataset.root/'cooperative/label'/f"{pair['vehicle_frame']}.json"
            if not path.exists():
                counts['missing_cooperative_file'] += 1; continue
            records = [resolve_reference(dataset, pair, obj) for obj in read(path)]
            # Reject non-one-to-one cooperative correspondences as ambiguous.
            vc = Counter(str(x['veh_track_id']) for x in records if x['status'] == 'valid')
            ic = Counter(str(x['inf_track_id']) for x in records if x['status'] == 'valid')
            accepted = []
            for record in records:
                if record['status'] == 'valid' and (vc[str(record['veh_track_id'])] != 1 or ic[str(record['inf_track_id'])] != 1):
                    record['status'] = 'unknown'; record['errors'].append('nonunique_cooperative_link')
                counts[record['status']] += 1
                counts.update(record['errors'])
                if record['status'] == 'valid':
                    accepted.append(record)
                    counts['class_disagreement'] += record['native_classes'][0] != record['native_classes'][1]
                stream.write(json.dumps(record)+'\n')
            lookup[str(pair['vehicle_frame'])] = accepted
    return lookup, dict(counts)


def causal_objects(dataset, pair):
    vf, paired_inf = str(pair['vehicle_frame']), str(pair['infrastructure_frame'])
    ego_meta = dataset.frames['vehicle-side'][vf]
    ego_time = int(ego_meta['pointcloud_timestamp'])
    sequence = str(pair['infrastructure_sequence'])
    timeline = dataset.timelines[sequence]
    # Do not step beyond the paired observation, even when ego time is later.
    upper = min(ego_time, int(dataset.frames['infrastructure-side'][paired_inf]['pointcloud_timestamp']))
    index = bisect.bisect_right(timeline, (upper, '\uffff')) - 1
    selected = timeline[index][1] if index >= 0 else None
    world_to_ego = np.linalg.inv(dataset.pose('vehicle-side', vf))
    vehicle, infra = [], []
    for obj in dataset.objects('vehicle-side', vf):
        cls = CLASSES.get(obj['type'].lower())
        if cls is not None:
            vehicle.append({'track_id': str(obj['track_id']), 'token': obj['token'],
                            'class': cls, 'raw_class': obj['type'], 'box': native_box(obj)})
    if selected is None:
        return vehicle, infra, selected, None
    now = int(dataset.frames['infrastructure-side'][selected]['pointcloud_timestamp'])
    age = (ego_time-now)/1e6
    correction = np.array([float(pair['system_error_offset']['delta_x']),
                           float(pair['system_error_offset']['delta_y']), 0.])
    rows = dataset.objects('infrastructure-side', selected)
    track_counts = Counter(str(x.get('track_id')) for x in rows)
    for obj in rows:
        cls = CLASSES.get(obj['type'].lower())
        if cls is None:
            continue
        raw = native_box(obj)
        current = transform_box(raw, dataset.pose('infrastructure-side', selected))
        previous, previous_obj, status = dataset.latest_history(selected, obj)
        if str(obj.get('track_id')) in ('None', '', '-1') or track_counts[str(obj.get('track_id'))] != 1:
            previous, previous_obj, status = None, None, 'invalid_current_track'
        previous_time = int(dataset.frames['infrastructure-side'][previous]['pointcloud_timestamp']) if previous else None
        previous_world = transform_box(native_box(previous_obj), dataset.pose('infrastructure-side', previous)) if previous else None
        compensated, velocity, valid = extrapolate(current, previous_world,
                                (now-previous_time)/1e6 if previous else 0., age)
        spatial = transform_box(current, world_to_ego)
        box = transform_box(compensated, world_to_ego)
        # Released SPD calibration convention: offset added in vehicle LiDAR.
        spatial[:3] += correction; box[:3] += correction
        infra.append({'track_id': str(obj['track_id']), 'token': obj['token'], 'class': cls,
                      'raw_class': obj['type'], 'raw_box': raw, 'spatial_box': spatial,
                      'box': box, 'velocity_world': velocity, 'velocity_ego': world_to_ego[:3, :3] @ velocity,
                      'motion_valid': valid, 'motion_status': status,
                      'timestamp_us': now, 'history_frame': previous,
                      'history_timestamp_us': previous_time, 'age_s': age})
    return vehicle, infra, selected, age


def json_default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def configurations():
    for method in 'ABC':
        for vehicle_gate, small_gate, threshold in itertools.product([1., 2., 4.], [.5, 1., 2.], [0., .1, .3, .5]):
            yield method, (vehicle_gate, small_gate, small_gate), threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', default='/home/zzn/V2X_VLM/UniV2X/datasets/V2X-Seq-SPD-New')
    parser.add_argument('--planning-dir', default='data/Planning/continuous_v2_ego_gt')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'], choices=['train', 'val', 'test'])
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--limit', type=int, default=0, help='Smoke only; 0 means complete split')
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    dataset = NativeDataset(args.dataset_root)
    pairs = read(dataset.root/'cooperative/data_info.json')
    references, audit = reference_audit(dataset, pairs, output)
    by_vehicle = {str(x['vehicle_frame']): x for x in pairs}
    stats = defaultdict(Counter)
    configs = list(configurations())
    motion = Counter()
    agreement = defaultdict(list)
    input_hashes = {}
    with (output/'objects.jsonl').open('w') as stream:
        for split in args.splits:
            path = Path(args.planning_dir)/f'{split}.json'
            input_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
            samples = read(path)
            if args.limit: samples = samples[:args.limit]
            for number, sample in enumerate(samples):
                token = str(sample['token'])
                pair = by_vehicle[token]
                vehicle, infra, selected, age = causal_objects(dataset, pair)
                if int(sample['timestamp_us']) != int(dataset.frames['vehicle-side'][token]['pointcloud_timestamp']):
                    raise ValueError('Planning ego timestamp differs from native label')
                motion[split+':samples'] += 1
                motion[split+':no_message'] += selected is None
                motion.update(split+':'+x['motion_status'] for x in infra)
                # Past-track migration is reference evaluation only; matcher
                # receives class/geometry and never cooperative IDs.
                vcounts = Counter(x['track_id'] for x in vehicle)
                icounts = Counter(x['track_id'] for x in infra)
                vi = {x['track_id']: i for i, x in enumerate(vehicle) if vcounts[x['track_id']] == 1}
                ii = {x['track_id']: i for i, x in enumerate(infra) if icounts[x['track_id']] == 1}
                truth = {(vi[str(x['veh_track_id'])], ii[str(x['inf_track_id'])])
                         for x in references[token] if str(x['veh_track_id']) in vi and str(x['inf_track_id']) in ii}
                known_v, known_i = {a for a,b in truth}, {b for a,b in truth}
                stream.write(json.dumps({'split': split, 'token': token, 'vehicle_sequence': pair['vehicle_sequence'],
                    'infra_sequence': pair['infrastructure_sequence'], 'paired_infra_frame': pair['infrastructure_frame'],
                    'selected_infra_frame': selected, 'timestamp_us': sample['timestamp_us'], 'age_s': age,
                    'vehicle': vehicle, 'infra': infra}, default=json_default)+'\n')
                for a,b in truth:
                    for mode, field in [('spatial','spatial_box'),('compensated','box')]:
                        error = float(np.linalg.norm(vehicle[a]['box'][:2]-infra[b][field][:2]))
                        agreement[(split, mode, vehicle[a]['class'], 'all')].append(error)
                        bucket = '0-.1s' if age < .1 else '.1-.5s' if age < .5 else '.5-1s' if age < 1 else '1s+'
                        agreement[(split, mode, vehicle[a]['class'], bucket)].append(error)
                for mode in ('spatial', 'compensated'):
                    candidates = infra if mode == 'compensated' else [{**x, 'box':x['spatial_box']} for x in infra]
                    features = pair_features(vehicle, candidates)
                    for method, gates, threshold in configs:
                        predicted = set(match(vehicle, candidates, features, method, gates, threshold))
                        valid_predictions = {(a,b) for a,b in predicted if a in known_v or b in known_i}
                        bucket = 'no_message' if age is None else '0-.1s' if age < .1 else '.1-.5s' if age < .5 else '.5-1s' if age < 1 else '1s+'
                        for cls, age_group in [('all','all'), *[(c,'all') for c in range(3)], ('all',bucket)]:
                            selected_truth = {p for p in truth if cls == 'all' or vehicle[p[0]]['class'] == cls}
                            selected_pred = {p for p in predicted if cls == 'all' or vehicle[p[0]]['class'] == cls}
                            selected_valid = selected_pred & valid_predictions
                            stat = stats[(split, mode, method, *gates, threshold, cls, age_group)]
                            stat['reference_pairs'] += len(selected_truth)
                            stat['true_positive'] += len(selected_pred & selected_truth)
                            stat['known_false_positive'] += len(selected_valid-selected_truth)
                            stat['unknown_predictions'] += len(selected_pred-selected_valid)
                            stat['predictions'] += len(selected_pred)
                if number % 100 == 0: print(f'{split}: {number}/{len(samples)}', flush=True)
    rows = []
    for key, counts in stats.items():
        split, mode, method, vg, pg, cg, threshold, cls, age_group = key
        tp, fp = counts['true_positive'], counts['known_false_positive']
        rows.append({'split':split,'mode':mode,'method':method,'gates_m':[vg,pg,cg], 'iou_threshold':threshold,
                     'class_group':cls, 'age_group':age_group,
                     **counts, 'reference_recall':tp/counts['reference_pairs'] if counts['reference_pairs'] else None,
                     'precision_on_judgable_predictions':tp/(tp+fp) if tp+fp else None,
                     'known_false_merge_rate':fp/(tp+fp) if tp+fp else None})
    (output/'association_sweep.json').write_text(json.dumps(rows, indent=2))
    groups = [{'split':k[0], 'mode':k[1], 'class':k[2], 'age_group':k[3], 'count':len(v),
               'mean_center_difference_m':float(np.mean(v)), 'median_m':float(np.median(v)),
               'p95_m':float(np.percentile(v,95))} for k,v in agreement.items()]
    source_paths = {dataset.root/'cooperative/data_info.json',
                    dataset.root/'vehicle-side/data_info.json', dataset.root/'infrastructure-side/data_info.json'}
    source_paths.update(dataset.root/'cooperative/label'/f"{p['vehicle_frame']}.json" for p in pairs)
    for side, frame in dataset.labels:
        source_paths.add(dataset.root/side/dataset.frames[side][frame]['label_lidar_std_path'])
    for side, frame in dataset.poses:
        meta = dataset.frames[side][frame]
        keys = ['calib_lidar_to_novatel_path','calib_novatel_to_world_path'] if side == 'vehicle-side' else ['calib_virtuallidar_to_world_path']
        source_paths.update(dataset.root/side/meta[k] for k in keys)
    source_hashes = {str(p.relative_to(dataset.root)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(source_paths) if p.exists()}
    (output/'source_sha256.json').write_text(json.dumps(source_hashes, indent=2))
    manifest = {'schema':'native-causal-association-v1','args':vars(args), 'reference_audit':audit,
                'motion':dict(motion),'cross_view_agreement':groups,'planning_sha256':input_hashes,
                'native_source_hashes':'source_sha256.json',
                'source_code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('core.py')]},
                'limitations':['precision excludes unjudgable predictions; unmatched is not negative',
                    'center difference measures cross-view agreement, not independent motion ground truth',
                    '3D IoU uses yaw-oriented upright boxes; no roll/pitch cuboid intersection',
                    'released pair calibration offsets used; not a learned or GT-object correction',
                    'no reception timestamps: assume no additional transport latency',
                    'RSU-only retention not scored: missing ego ID is not verified absence',
                    'no official threshold selected; no test tuning; full candidates before ROI/top-K']}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'output':str(output),'reference_audit':audit,'motion':dict(motion)}, indent=2))


if __name__ == '__main__':
    main()
