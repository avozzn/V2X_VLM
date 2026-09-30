#!/usr/bin/env python3
"""Run inference/evaluation for multiple planning checkpoints and select a baseline."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_CHECKPOINTS = ('388', '776', '1164', '1552', '1940', '2328')


def parse_args():
    project_root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=Path,
        default=project_root / 'projects/configs/Robodrivevlm/MMdrive_v2x.py')
    parser.add_argument(
        '--checkpoint-root', type=Path,
        default=project_root / 'checkpoints/V2X_dt_9step')
    parser.add_argument('--checkpoints', nargs='+', default=list(DEFAULT_CHECKPOINTS))
    parser.add_argument(
        '--test-json', type=Path,
        default=project_root / 'data/Planning/test_dt_resampled_4.json')
    parser.add_argument(
        '--origin-model', type=Path,
        default=project_root / 'checkpoints/LLM/llava-next-interleave')
    parser.add_argument(
        '--output-dir', type=Path,
        default=project_root / 'evaluation_results/V2X_dt_9step_sweep')
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--nproc-per-node', type=int, default=1)
    parser.add_argument('--master-port', type=int, default=29420)
    parser.add_argument('--predict-steps', type=int, choices=(6, 9, 12), default=9)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--python', default=sys.executable)
    return parser.parse_args()


def checkpoint_path(root, value):
    candidate = Path(value)
    if candidate.is_dir():
        return candidate.resolve()
    name = value if value.startswith('checkpoint-') else f'checkpoint-{value}'
    return (root / name).resolve()


def run_logged(command, log_path, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log:
        log.write('\nCOMMAND: ' + ' '.join(map(str, command)) + '\n')
        log.flush()
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=env)


def load_json(path):
    with path.open('r', encoding='utf-8') as file:
        return json.load(file)


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    inference_script = project_root / 'tools/robollm/test-ddp_token.py'
    diagnostics_script = script_dir / 'trajectory_diagnostics.py'
    test_json = args.test_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()

    for required in (args.config, test_json, args.origin_model, inference_script, diagnostics_script):
        if not Path(required).exists():
            raise FileNotFoundError(required)
    if args.nproc_per_node <= 0:
        raise ValueError('--nproc-per-node must be positive.')
    if len([gpu for gpu in args.gpus.split(',') if gpu.strip()]) < args.nproc_per_node:
        raise ValueError('--gpus must list at least --nproc-per-node devices.')

    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = args.gpus
    environment['PYTHONPATH'] = str(project_root) + os.pathsep + environment.get('PYTHONPATH', '')
    summaries = []

    for index, checkpoint_value in enumerate(args.checkpoints):
        checkpoint = checkpoint_path(checkpoint_root, checkpoint_value)
        if not (checkpoint / 'adapter_model.safetensors').is_file():
            raise FileNotFoundError(f'Invalid checkpoint: {checkpoint}')

        name = checkpoint.name
        run_dir = output_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            'predictions': run_dir / 'predictions.json',
            'trajectories': run_dir / 'trajectories.pkl',
            'metrics': run_dir / 'metrics.json',
            'analysis': run_dir / 'analysis.json',
            'metadata': run_dir / 'metadata.json',
            'diagnostics': run_dir / 'diagnostics.json',
            'log': run_dir / 'run.log',
        }

        inference_complete = all(paths[key].exists() for key in ('predictions', 'trajectories', 'metrics', 'analysis', 'metadata'))
        if not (args.resume and inference_complete):
            command = [
                args.python, '-m', 'torch.distributed.run',
                '--nnodes', '1',
                '--nproc_per_node', str(args.nproc_per_node),
                '--master_addr', '127.0.0.1',
                '--master_port', str(args.master_port + index),
                str(inference_script),
                '--model_path', str(checkpoint),
                '--origin_model_path', str(args.origin_model.resolve()),
                '--result_path', str(paths['predictions']),
                '--pkl_path', str(paths['trajectories']),
                '--metrics_path', str(paths['metrics']),
                '--analysis_path', str(paths['analysis']),
                '--metadata_path', str(paths['metadata']),
                '--predict_steps', str(args.predict_steps),
            ]
            if args.max_samples is not None:
                command.extend(['--max_samples', str(args.max_samples)])
            command.extend([
                str(args.config.resolve()),
                '--cfg-options',
                f'test_dataloader.dataset.json_file={test_json}',
            ])
            print(f'[{index + 1}/{len(args.checkpoints)}] Inference: {name}')
            run_logged(command, paths['log'], env=environment)
        else:
            print(f'[{index + 1}/{len(args.checkpoints)}] Reusing inference: {name}')

        if not (args.resume and paths['diagnostics'].exists()):
            diagnostic_command = [
                args.python, str(diagnostics_script),
                '--predictions-json', str(paths['predictions']),
                '--trajectories-pkl', str(paths['trajectories']),
                '--test-json', str(test_json),
                '--collision-analysis', str(paths['analysis']),
                '--output', str(paths['diagnostics']),
                '--predict-steps', str(args.predict_steps),
            ]
            print(f'[{index + 1}/{len(args.checkpoints)}] Diagnostics: {name}')
            run_logged(diagnostic_command, paths['log'], env=environment)

        diagnostics = load_json(paths['diagnostics'])
        summary = {
            'checkpoint': name,
            'checkpoint_path': str(checkpoint),
            'coverage_rate': diagnostics['coverage']['coverage_rate'],
            'parsed_trajectories': diagnostics['coverage']['parsed_trajectories'],
            'mean_l2_all_steps': diagnostics['trajectory_metrics']['mean_l2_all_steps'],
            'collision_any_rate': diagnostics['trajectory_metrics']['collision_any_rate'],
            'illegal_trajectory_rate': diagnostics['illegal_trajectories']['rate'],
            'illegal_trajectory_count': diagnostics['illegal_trajectories']['count'],
            'meta_action_accuracy': diagnostics['action_metrics']['exact_accuracy'],
            'diagnostics_path': str(paths['diagnostics']),
        }
        summary['eligible'] = summary['coverage_rate'] >= 0.99
        summaries.append(summary)

        with (output_dir / 'sweep_summary.json').open('w', encoding='utf-8') as file:
            json.dump(summaries, file, ensure_ascii=False, indent=2)

    eligible = [item for item in summaries if item['eligible'] and item['mean_l2_all_steps'] is not None]
    ranking = sorted(
        eligible,
        key=lambda item: (
            item['mean_l2_all_steps'],
            item['collision_any_rate'] if item['collision_any_rate'] is not None else float('inf'),
            item['illegal_trajectory_rate'],
        ),
    )
    selection = {
        'selection_policy': [
            'coverage_rate >= 0.99',
            'minimum mean_l2_all_steps',
            'minimum collision_any_rate',
            'minimum illegal_trajectory_rate',
        ],
        'baseline': ranking[0] if ranking else None,
        'ranking': ranking,
        'ineligible': [item for item in summaries if not item['eligible']],
        'smoke_test': args.max_samples is not None,
    }
    with (output_dir / 'baseline_selection.json').open('w', encoding='utf-8') as file:
        json.dump(selection, file, ensure_ascii=False, indent=2)

    fieldnames = [
        'checkpoint', 'coverage_rate', 'parsed_trajectories', 'mean_l2_all_steps',
        'collision_any_rate', 'illegal_trajectory_rate', 'illegal_trajectory_count',
        'meta_action_accuracy', 'eligible', 'checkpoint_path', 'diagnostics_path',
    ]
    with (output_dir / 'sweep_summary.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    if selection['baseline']:
        print(f"Selected baseline: {selection['baseline']['checkpoint']}")
    else:
        print('No checkpoint met the baseline eligibility criteria.')
    print(f'Results saved to: {output_dir}')


if __name__ == '__main__':
    main()
