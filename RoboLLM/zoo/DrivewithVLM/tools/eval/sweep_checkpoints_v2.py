#!/usr/bin/env python3
"""Generate V2 physics validation predictions and select the best checkpoint."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT /
                        'projects/configs/Robodrivevlm/MMdrive_v2x_meta_v2_physics.py')
    parser.add_argument('--checkpoint-root', required=True, type=Path)
    parser.add_argument('--checkpoints', nargs='*', default=None,
                        help='Checkpoint names/steps; default discovers checkpoint-* directories.')
    parser.add_argument('--split-json', type=Path, default=PROJECT_ROOT /
                        'data/Planning/v2_physics/val_v2x_planning_v2.json')
    parser.add_argument('--ann-file', required=True, type=Path,
                        help='PKL backing split-json (train PKL for the fixed validation split).')
    parser.add_argument('--origin-model', type=Path, default=PROJECT_ROOT /
                        'checkpoints/LLM/llava-next-interleave')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--nproc-per-node', type=int, default=1)
    parser.add_argument('--master-port', type=int, default=29620)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--python', default=sys.executable)
    return parser.parse_args()


def checkpoint_paths(root, requested):
    if requested:
        paths = []
        for value in requested:
            candidate = Path(value)
            if not candidate.is_dir():
                name = value if value.startswith('checkpoint-') else f'checkpoint-{value}'
                candidate = root / name
            paths.append(candidate.resolve())
        return paths
    return sorted(
        (path.resolve() for path in root.glob('checkpoint-*') if path.is_dir()),
        key=lambda path: int(path.name.rsplit('-', 1)[-1]),
    )


def run_logged(command, log_path, environment):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log:
        log.write('\nCOMMAND: ' + ' '.join(map(str, command)) + '\n')
        log.flush()
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                       check=True, env=environment)


def main():
    args = parse_args()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    split_json = args.split_json.expanduser().resolve()
    ann_file = args.ann_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    inference = PROJECT_ROOT / 'tools/robollm/test-ddp_token.py'
    diagnostics = PROJECT_ROOT / 'tools/eval/trajectory_diagnostics_v2.py'
    for path in (args.config, checkpoint_root, split_json, ann_file,
                 args.origin_model, inference, diagnostics):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if args.nproc_per_node <= 0:
        raise ValueError('--nproc-per-node must be positive')
    if len([gpu for gpu in args.gpus.split(',') if gpu.strip()]) < args.nproc_per_node:
        raise ValueError('--gpus must list at least --nproc-per-node devices')

    checkpoints = checkpoint_paths(checkpoint_root, args.checkpoints)
    if not checkpoints:
        raise ValueError(f'no checkpoints found under {checkpoint_root}')
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = args.gpus
    environment['PYTHONPATH'] = str(PROJECT_ROOT) + os.pathsep + environment.get('PYTHONPATH', '')
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for index, checkpoint in enumerate(checkpoints):
        if not (checkpoint / 'adapter_model.safetensors').is_file():
            raise FileNotFoundError(f'invalid checkpoint: {checkpoint}')
        run_dir = output_dir / checkpoint.name
        predictions = run_dir / 'predictions.json'
        report_path = run_dir / 'diagnostics_v2.json'
        log_path = run_dir / 'run.log'
        auxiliary = {
            'pkl': run_dir / 'trajectories.pkl',
            'metrics': run_dir / 'legacy_metrics_skipped.json',
            'analysis': run_dir / 'legacy_analysis_skipped.json',
            'metadata': run_dir / 'metadata.json',
        }
        if not (args.resume and predictions.exists()):
            command = [
                args.python, '-m', 'torch.distributed.run', '--nnodes', '1',
                '--nproc_per_node', str(args.nproc_per_node),
                '--master_addr', '127.0.0.1',
                '--master_port', str(args.master_port + index), str(inference),
                '--model_path', str(checkpoint),
                '--origin_model_path', str(args.origin_model.resolve()),
                '--result_path', str(predictions), '--pkl_path', str(auxiliary['pkl']),
                '--metrics_path', str(auxiliary['metrics']),
                '--analysis_path', str(auxiliary['analysis']),
                '--metadata_path', str(auxiliary['metadata']),
                '--predict_steps', '9', '--skip-planning-evaluation',
            ]
            if args.max_samples is not None:
                command.extend(['--max_samples', str(args.max_samples)])
            command.extend([
                str(args.config.resolve()), '--cfg-options',
                f'test_dataloader.dataset.ann_file={ann_file}',
                f'test_dataloader.dataset.json_file={split_json}',
            ])
            print(f'[{index + 1}/{len(checkpoints)}] inference {checkpoint.name}')
            run_logged(command, log_path, environment)

        if not (args.resume and report_path.exists()):
            command = [
                args.python, str(diagnostics), '--predictions-json', str(predictions),
                '--test-json', str(split_json), '--output', str(report_path),
            ]
            print(f'[{index + 1}/{len(checkpoints)}] diagnostics {checkpoint.name}')
            run_logged(command, log_path, environment)

        with report_path.open('r', encoding='utf-8') as file:
            report = json.load(file)
        summary = {
            'checkpoint': checkpoint.name,
            'checkpoint_path': str(checkpoint),
            'coverage_rate': report['coverage']['rate'],
            'valid_predictions': report['coverage']['valid_predictions'],
            'mean_l2_all_steps': report['trajectory_metrics'].get('mean_l2_all_steps'),
            'final_step_mean_l2': (
                report['trajectory_metrics'].get('per_step_mean_l2') or [None])[-1],
            'text_lateral_macro_f1': report['text_lateral_metrics']['macro_f1'],
            'text_longitudinal_macro_f1': report['text_longitudinal_metrics']['macro_f1'],
            'lateral_consistency': report['action_consistency']['lateral_rate'],
            'longitudinal_consistency': report['action_consistency']['longitudinal_rate'],
            'target_end_speed_mae': report['continuous_metrics']['target_end_speed_mae'],
            'mean_acceleration_mae': report['continuous_metrics']['mean_acceleration_mae'],
            'report_path': str(report_path),
        }
        summary['eligible'] = summary['coverage_rate'] >= 0.99
        summaries.append(summary)
        with (output_dir / 'sweep_summary.json').open('w', encoding='utf-8') as file:
            json.dump(summaries, file, ensure_ascii=False, indent=2)

    ranking = sorted(
        (item for item in summaries if item['eligible'] and
         item['mean_l2_all_steps'] is not None),
        key=lambda item: (item['mean_l2_all_steps'], item['final_step_mean_l2']),
    )
    selection = {
        'selection_policy': [
            'coverage_rate >= 0.99', 'minimum validation mean_l2_all_steps',
            'minimum validation final_step_mean_l2 as tie-breaker',
        ],
        'best_checkpoint': ranking[0] if ranking else None,
        'ranking': ranking,
        'smoke_test': args.max_samples is not None,
    }
    with (output_dir / 'best_checkpoint_selection.json').open('w', encoding='utf-8') as file:
        json.dump(selection, file, ensure_ascii=False, indent=2)
    print('Best checkpoint:', ranking[0]['checkpoint'] if ranking else 'none')
    print('Results:', output_dir)


if __name__ == '__main__':
    main()
