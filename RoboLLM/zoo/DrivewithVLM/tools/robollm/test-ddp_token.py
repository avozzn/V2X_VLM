import argparse
import logging
import os
import os.path as osp
import torch
import torch.distributed as dist
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import DATASETS, init_default_scope
from mmengine.utils import import_modules_from_strings
from tqdm import tqdm
from mmdet3d.utils import replace_ceph_backend
from transformers import AutoProcessor, LlavaForConditionalGeneration
import json
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import pickle
import ast
import subprocess
from pathlib import Path
from torch.utils.data import Subset
from tools.eval.evaluation import load_pred_trajs_from_file,planning_evaluation
import time
def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Test a 3D detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--sync_bn',
        choices=['none', 'torch', 'mmcv'],
        default='none',
        help='convert all BatchNorm layers in the model to SyncBatchNorm '
             '(SyncBN) or mmcv.ops.sync_bn.SyncBatchNorm (MMSyncBN) layers.')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
             'specify, try to auto resume from the latest checkpoint '
             'in the work directory.')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',  # Changed default to 'pytorch' for distributed
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint."
    )
    parser.add_argument(
        "--origin_model_path",
        type=str,
        required=True,
        help="Path to origin_model."
    )
    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
        help="Path to save the results JSON file."
    )
    parser.add_argument(
        "--pkl_path",
        type=str,
        required=True,
        help="Path to save the results JSON file."
    )
    parser.add_argument('--metrics_path', type=str, required=True)
    parser.add_argument('--analysis_path', type=str, required=True)
    parser.add_argument('--metadata_path', type=str, required=True)
    parser.add_argument('--predict_steps', type=int, choices=(6, 9, 12), default=9)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--max-input-tokens', type=int, default=3968)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument(
        '--skip-planning-evaluation', action='store_true',
        help='Write generated predictions without running the legacy occupancy evaluator.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        if torch.cuda.is_available():
            backend = 'nccl'
            device = torch.device('cuda', local_rank)
            torch.cuda.set_device(device)
        else:
            backend = 'gloo'
            device = torch.device('cpu')
        dist.init_process_group(
            backend=backend,
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device, rank, world_size


def distributed_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def extract_trajectory(answer, predict_steps):
    """Extract the last Python-style list from a generated answer."""
    candidates = [line.strip() for line in answer.splitlines() if line.strip()]
    for candidate in reversed(candidates):
        try:
            trajectory = np.asarray(ast.literal_eval(candidate), dtype=np.float64)
        except (ValueError, SyntaxError):
            continue
        if trajectory.shape == (predict_steps, 2):
            if not np.isfinite(trajectory).all():
                raise ValueError('Trajectory contains NaN or Inf.')
            return trajectory
    raise ValueError(f'No trajectory with shape ({predict_steps}, 2) found.')

def main():
    args = parse_args()
    if args.max_input_tokens <= 0 or args.max_new_tokens <= 0:
        raise ValueError('Token limits must be positive.')

    # Setup distributed
    device, rank, world_size = setup_distributed()

    # Load config
    cfg = Config.fromfile(args.config)

    # Replace Ceph backend if specified
    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Determine work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # Enable automatic-mixed-precision training
    if args.amp:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # Convert BatchNorm layers
    if args.sync_bn != 'none':
        cfg.sync_bn = args.sync_bn

    # Enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # Resume training if specified
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # This entry point consumes only the dataset and the Hugging Face model.
    # Building an unused MMEngine Runner makes configs with a validation split
    # require an unrelated val loop/evaluator.
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    # Build the dataset
    dataset_cfg = cfg.test_dataloader.pop('dataset')
    datasets = DATASETS.build(dataset_cfg)
    dataset_size = len(datasets)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError('--max_samples must be positive.')
        datasets = Subset(datasets, range(min(args.max_samples, dataset_size)))

    # Wrap dataset with DistributedSampler
    if world_size > 1:
        sampler = DistributedSampler(datasets, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        sampler = None

    # DataLoader with DistributedSampler
    dataloader = DataLoader(
        dataset=datasets,
        batch_size=1,  # Adjust as needed
        sampler=sampler,
        collate_fn=lambda batch: batch[0],
        num_workers=4,  # Adjust based on your system
        pin_memory=True
    )

    # Load the model
    original_model_id = args.origin_model_path
    model_id = args.model_path
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True, 
    ).to(device)
    processor = AutoProcessor.from_pretrained(original_model_id)
    model_context_length = model.config.text_config.max_position_embeddings
    if args.max_input_tokens + args.max_new_tokens > model_context_length:
        raise ValueError(
            f'Input/output budget ({args.max_input_tokens} + '
            f'{args.max_new_tokens}) exceeds model context length '
            f'{model_context_length}.'
        )
    # Results are gathered in memory and written once, keeping predictions.json
    # valid and separate from metrics/metadata.
    collected_results = {}
    duplicate_tokens = 0
    input_token_lengths = []
    generated_token_lengths = []
    if rank == 0:
        for path in (
            args.result_path,
            args.pkl_path,
            args.metrics_path,
            args.analysis_path,
            args.metadata_path,
        ):
            Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    # Synchronize before processing
    distributed_barrier()

    # Process data
    for batch in tqdm(dataloader, desc=f"Rank {rank} Processing"):
        local_results = []  # Clear local results at the start of each iteration
        model.eval()
        with torch.no_grad():
            data = batch['llm_input']
            # Move data to the correct device
            token=data['token']
            data = {k: v.to(device) for k, v in data.items() if k != "token"}
            data['pixel_values'] = data['pixel_values'].to(torch.float16)
            
            input_token_len = data['input_ids'].shape[1]
            effective_input_len = int(data['attention_mask'][0].sum().item())
            if effective_input_len > args.max_input_tokens:
                raise ValueError(
                    f'Token {token} has {effective_input_len} input tokens, '
                    f'exceeding --max-input-tokens={args.max_input_tokens}. '
                    'Reduce perception targets rather than truncating the '
                    'prompt suffix.'
                )
            # Generate output
            output = model.generate(**data, max_new_tokens=args.max_new_tokens, do_sample=False,eos_token_id=processor.tokenizer.eos_token_id,pad_token_id=processor.tokenizer.pad_token_id)
            generated_tokens = output[0][input_token_len:]
            result = processor.decode(generated_tokens, skip_special_tokens=True)
            
            # Store results locally
            result_dict = {
                'token':token,
                # 'input': question,
                'answer': result,
                'input_tokens': effective_input_len,
                'generated_tokens': int(generated_tokens.shape[0]),
            }
            local_results.append(result_dict)

        # Gather all results to rank 0
        if world_size > 1:
            if rank == 0:
                gathered_results = [None for _ in range(world_size)]
                dist.gather_object(local_results, object_gather_list=gathered_results, dst=0)
            else:
                dist.gather_object(local_results, dst=0)
        else:
            gathered_results = [local_results]

        # Deduplicate DistributedSampler padding by token.
        if rank == 0:
            for rank_results in gathered_results:
                for item in rank_results:
                    token_key = str(item['token'])
                    if token_key in collected_results:
                        duplicate_tokens += 1
                        continue
                    item['token'] = token_key
                    collected_results[token_key] = item
                    input_token_lengths.append(item['input_tokens'])
                    generated_token_lengths.append(item['generated_tokens'])

    # Synchronize before finalizing
    distributed_barrier()

    # Rank 0 computes the average L2 distances
    if rank == 0:
        data = list(collected_results.values())
        with open(args.result_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

        traj_dict = {}
        parse_errors = []
        for line_num, item in enumerate(data, 1):
            try:
                trajectory = extract_trajectory(item['answer'], args.predict_steps)
                traj_dict[item['token']] = np.expand_dims(trajectory, axis=0)
            except Exception as e:
                parse_errors.append({
                    'line': line_num,
                    'token': item.get('token'),
                    'error': f'{type(e).__name__}: {e}',
                })

        with open(args.pkl_path, 'wb') as file:
            pickle.dump(traj_dict, file)

        if args.skip_planning_evaluation:
            evaluation_data = {"skipped": True, "reason": "requested by CLI"}
            with open(args.analysis_path, 'w', encoding='utf-8') as file:
                json.dump({"skipped": True}, file, indent=2)
        else:
            evaluation_data = planning_evaluation(
                traj_dict,
                subset=None,
                only_vehicle=False,
                predict_steps=args.predict_steps,
                analysis_output_path=args.analysis_path,
            )
        with open(args.metrics_path, 'w', encoding='utf-8') as file:
            json.dump(evaluation_data, file, indent=2)

        try:
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2], text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            git_commit = None
        metadata = {
            'checkpoint': str(Path(args.model_path).resolve()),
            'origin_model': str(Path(args.origin_model_path).resolve()),
            'config': str(Path(args.config).resolve()),
            'config_overrides': args.cfg_options or {},
            'predict_steps': args.predict_steps,
            'max_input_tokens': args.max_input_tokens,
            'max_new_tokens': args.max_new_tokens,
            'model_context_length': model_context_length,
            'input_token_length': {
                'min': min(input_token_lengths) if input_token_lengths else None,
                'max': max(input_token_lengths) if input_token_lengths else None,
                'mean': (sum(input_token_lengths) / len(input_token_lengths)) if input_token_lengths else None,
            },
            'generated_token_length': {
                'min': min(generated_token_lengths) if generated_token_lengths else None,
                'max': max(generated_token_lengths) if generated_token_lengths else None,
                'mean': (sum(generated_token_lengths) / len(generated_token_lengths)) if generated_token_lengths else None,
            },
            'dataset_size_before_limit': dataset_size,
            'requested_max_samples': args.max_samples,
            'unique_predictions': len(data),
            'parsed_trajectories': len(traj_dict),
            'duplicate_sampler_tokens': duplicate_tokens,
            'parse_errors': parse_errors,
            'world_size': world_size,
            'git_commit': git_commit,
            'command': ' '.join(os.sys.argv),
        }
        with open(args.metadata_path, 'w', encoding='utf-8') as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
        print(f"Predictions saved to {args.result_path}")
        print(f"Trajectories saved to {args.pkl_path}")
        print(f"Parse errors: {len(parse_errors)}")
        print("Test finish\n")
    distributed_barrier()

if __name__ == '__main__':
    main()
