#!/usr/bin/env python3
"""Diagnose generated planning trajectories and meta-action predictions."""

import argparse
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ACTION_PATTERN = re.compile(r"Meta\s+Action\s*:\s*([^\n\r]+)", re.IGNORECASE)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions-json', type=Path, required=True)
    parser.add_argument('--trajectories-pkl', type=Path, required=True)
    parser.add_argument('--test-json', type=Path, required=True)
    parser.add_argument(
        '--gt-trajectories', type=Path,
        default=script_dir / 'gt' / 'gt_v2x_resampled_TM_test.pkl')
    parser.add_argument(
        '--gt-mask', type=Path,
        default=script_dir / 'gt' / 'gt_v2x_resampled_Tmask_test.pkl')
    parser.add_argument('--collision-analysis', type=Path, default=None)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--predict-steps', type=int, default=9)
    parser.add_argument('--dt', type=float, default=0.5)
    parser.add_argument('--max-abs-coordinate', type=float, default=100.0)
    parser.add_argument('--max-speed', type=float, default=40.0)
    parser.add_argument('--max-acceleration', type=float, default=10.0)
    parser.add_argument('--max-backward-step', type=float, default=0.5)
    return parser.parse_args()


def load_json(path):
    with path.open('r', encoding='utf-8') as file:
        return json.load(file)


def load_pickle(path):
    with path.open('rb') as file:
        return pickle.load(file)


def normalize_action(text):
    match = ACTION_PATTERN.search(text or '')
    if not match:
        return '<PARSE_FAIL>'
    return ' '.join(match.group(1).strip().upper().split())


def as_trajectory(value, predict_steps):
    trajectory = np.asarray(value, dtype=np.float64).reshape(-1, 2)
    if trajectory.shape != (predict_steps, 2):
        raise ValueError(f'expected ({predict_steps}, 2), got {trajectory.shape}')
    return trajectory


def validate_trajectory(trajectory, args):
    reasons = []
    if not np.isfinite(trajectory).all():
        return ['non_finite']
    if np.abs(trajectory).max() > args.max_abs_coordinate:
        reasons.append('coordinate_out_of_range')

    points = np.vstack([np.zeros((1, 2)), trajectory])
    displacement = np.diff(points, axis=0)
    velocity = displacement / args.dt
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = np.diff(velocity, axis=0) / args.dt

    if (speed > args.max_speed).any():
        reasons.append('speed_out_of_range')
    if len(acceleration) and (np.linalg.norm(acceleration, axis=1) > args.max_acceleration).any():
        reasons.append('acceleration_out_of_range')
    if (displacement[:, 0] < -args.max_backward_step).any():
        reasons.append('longitudinal_backward_jump')
    return reasons


def safe_metrics(values):
    if not values:
        return {'count': 0, 'mean': None, 'median': None}
    array = np.asarray(values, dtype=np.float64)
    return {
        'count': int(array.size),
        'mean': float(array.mean()),
        'median': float(np.median(array)),
    }


def main():
    args = parse_args()
    predictions = load_json(args.predictions_json)
    trajectories = load_pickle(args.trajectories_pkl)
    test_data = load_json(args.test_json)
    gt_trajectories = load_pickle(args.gt_trajectories)
    gt_masks = load_pickle(args.gt_mask)

    prediction_text = {str(item['token']): item.get('answer', '') for item in predictions}
    gt_items = {str(item['token']): item for item in test_data}
    expected_tokens = set(gt_items)
    predicted_tokens = set(map(str, trajectories))

    collision_by_token = {}
    if args.collision_analysis and args.collision_analysis.exists():
        collision_by_token = {
            str(item['token']): bool(item.get('collision', False))
            for item in load_json(args.collision_analysis)
        }

    invalid_records = []
    invalid_tokens = set()
    invalid_reason_counts = Counter()
    per_step_errors = []
    sample_metrics = {}

    for token, value in trajectories.items():
        token = str(token)
        reasons = []
        try:
            trajectory = as_trajectory(value, args.predict_steps)
            reasons.extend(validate_trajectory(trajectory, args))
        except Exception as error:
            invalid_records.append({'token': token, 'reasons': ['invalid_shape'], 'error': str(error)})
            invalid_tokens.add(token)
            invalid_reason_counts['invalid_shape'] += 1
            continue

        if reasons:
            invalid_tokens.add(token)
            invalid_reason_counts.update(reasons)
            invalid_records.append({
                'token': token,
                'reasons': reasons,
                'trajectory': trajectory.tolist(),
            })

        if token not in gt_trajectories or token not in gt_masks:
            continue
        gt = np.asarray(gt_trajectories[token], dtype=np.float64).reshape(-1, 3)[:args.predict_steps, :2]
        mask = np.asarray(gt_masks[token], dtype=np.float64).reshape(-1, 2)[:args.predict_steps]
        errors = np.sqrt(np.sum(np.square(trajectory - gt) * mask, axis=-1))
        per_step_errors.append(errors)
        sample_metrics[token] = {
            'mean_l2': float(errors.mean()),
            'final_l2': float(errors[-1]),
            'collision': collision_by_token.get(token, False),
            'invalid': token in invalid_tokens,
        }

    errors = np.asarray(per_step_errors, dtype=np.float64)
    if errors.size:
        per_step = {
            'mean': errors.mean(axis=0).tolist(),
            'median': np.median(errors, axis=0).tolist(),
            'p90': np.percentile(errors, 90, axis=0).tolist(),
        }
        overall_mean_l2 = float(errors.mean())
        final_thresholds = {
            f'over_{threshold}m': int((errors[:, -1] > threshold).sum())
            for threshold in (5, 10, 20)
        }
    else:
        per_step = {'mean': [], 'median': [], 'p90': []}
        overall_mean_l2 = None
        final_thresholds = {}

    action_counts = Counter()
    action_correct = Counter()
    predicted_action_counts = Counter()
    confusion = defaultdict(Counter)
    action_samples = defaultdict(list)
    exact_action_correct = 0

    for token, metrics in sample_metrics.items():
        item = gt_items.get(token)
        if not item:
            continue
        gt_answer = item['conversations'][1]['value']
        gt_action = normalize_action(gt_answer)
        pred_action = normalize_action(prediction_text.get(token, ''))
        action_counts[gt_action] += 1
        predicted_action_counts[pred_action] += 1
        confusion[gt_action][pred_action] += 1
        if gt_action == pred_action:
            exact_action_correct += 1
            action_correct[gt_action] += 1
        action_samples[gt_action].append(metrics)

    labels = sorted(set(action_counts) | set(predicted_action_counts))
    per_action = {}
    for label in labels:
        tp = confusion[label][label]
        support = action_counts[label]
        predicted = predicted_action_counts[label]
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        samples = action_samples[label]
        per_action[label] = {
            'support': support,
            'predicted_count': predicted,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_l2': float(np.mean([x['mean_l2'] for x in samples])) if samples else None,
            'mean_final_l2': float(np.mean([x['final_l2'] for x in samples])) if samples else None,
            'collision_rate': float(np.mean([x['collision'] for x in samples])) if samples else None,
            'invalid_rate': float(np.mean([x['invalid'] for x in samples])) if samples else None,
        }

    parsed_count = len(predicted_tokens)
    report = {
        'inputs': {
            'predictions_json': str(args.predictions_json.resolve()),
            'trajectories_pkl': str(args.trajectories_pkl.resolve()),
            'test_json': str(args.test_json.resolve()),
        },
        'coverage': {
            'expected_unique_tokens': len(expected_tokens),
            'generated_answers': len(prediction_text),
            'parsed_trajectories': parsed_count,
            'coverage_rate': parsed_count / len(expected_tokens) if expected_tokens else 0.0,
            'missing_tokens': sorted(expected_tokens - predicted_tokens),
            'unexpected_tokens': sorted(predicted_tokens - expected_tokens),
        },
        'trajectory_metrics': {
            'evaluated_tokens': len(sample_metrics),
            'mean_l2_all_steps': overall_mean_l2,
            'per_step_l2': per_step,
            'final_l2_thresholds': final_thresholds,
            'collision_any_rate': float(np.mean([x['collision'] for x in sample_metrics.values()])) if sample_metrics else None,
        },
        'illegal_trajectories': {
            'count': len(invalid_tokens),
            'rate': len(invalid_tokens) / parsed_count if parsed_count else 0.0,
            'reason_counts': dict(invalid_reason_counts),
            'records': invalid_records,
            'thresholds': {
                'dt': args.dt,
                'max_abs_coordinate': args.max_abs_coordinate,
                'max_speed': args.max_speed,
                'max_acceleration': args.max_acceleration,
                'max_backward_step': args.max_backward_step,
            },
        },
        'action_metrics': {
            'evaluated_tokens': len(sample_metrics),
            'exact_accuracy': exact_action_correct / len(sample_metrics) if sample_metrics else 0.0,
            'exact_correct': exact_action_correct,
            'per_action': per_action,
            'confusion_matrix': {key: dict(value) for key, value in confusion.items()},
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f'Diagnostics saved to {args.output.resolve()}')
    print(f"Coverage: {report['coverage']['coverage_rate']:.2%}")
    print(f"Mean L2: {overall_mean_l2:.4f}" if overall_mean_l2 is not None else 'Mean L2: N/A')
    print(f"Illegal trajectory rate: {report['illegal_trajectories']['rate']:.2%}")
    print(f"Meta-action accuracy: {report['action_metrics']['exact_accuracy']:.2%}")


if __name__ == '__main__':
    main()
