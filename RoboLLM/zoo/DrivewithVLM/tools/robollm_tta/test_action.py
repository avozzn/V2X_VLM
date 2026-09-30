import argparse
import logging
import os
import os.path as osp
import torch
import torch.distributed as dist
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS, DATASETS
from mmengine.runner import Runner
from tqdm import tqdm
from mmdet3d.utils import replace_ceph_backend
from transformers import AutoProcessor, LlavaForConditionalGeneration
import json
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import pickle
import ast
from eval.evaluation import load_pred_trajs_from_file,planning_evaluation 
def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Test a 3D detector')
    parser.add_argument('--config1', help='train config file path')
    parser.add_argument('--config2', help='train config file path')
    parser.add_argument('--config3', help='train config file path')
    parser.add_argument('--config4', help='train config file path')
    parser.add_argument('--config5', help='train config file path')
    parser.add_argument('--config6', help='train config file path')
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

def main():
    args = parse_args()

    # Setup distributed
    device, rank, world_size = setup_distributed()
    datasets=[]
    for config in [args.config1,args.config2,args.config3,args.config4,args.config5,args.config6]:
    # Load config
        cfg = Config.fromfile(config)

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
                                    osp.splitext(osp.basename(config))[0])

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

        # Build the runner
        if 'runner_type' not in cfg:
            runner = Runner.from_cfg(cfg)
        else:
            runner = RUNNERS.build(cfg)

        # Build the dataset
        dataset_cfg = cfg.test_dataloader.pop('dataset')
        datasets.append(DATASETS.build(dataset_cfg))

    # Wrap dataset with DistributedSampler
    if world_size > 1:
        sampler = DistributedSampler(datasets, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        sampler = None

    dataloader={}
    # DataLoader with DistributedSampler
    dataloader_ALL = DataLoader(
        dataset=datasets[0],
        batch_size=1,  # Adjust as needed
        sampler=sampler,
        collate_fn=lambda batch: batch[0],
        num_workers=4,  # Adjust based on your system
        pin_memory=True
    )
    dataloader2 = DataLoader(
        dataset=datasets[1],
        batch_size=1,  # Adjust as needed
        sampler=sampler,
        collate_fn=lambda batch: batch[0],
        num_workers=4,  # Adjust based on your system
        pin_memory=True
    )
    dataloader3 = DataLoader(
        dataset=datasets[2],
        batch_size=1,  # Adjust as needed
        sampler=sampler,
        collate_fn=lambda batch: batch[0],
        num_workers=4,  # Adjust based on your system
        pin_memory=True
    )
    dataloader4 = DataLoader(
        dataset=datasets[3],
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
    # import pdb
    # pdb.set_trace()

    # Initialize result storage
    if rank == 0:
        output_file = args.result_path
        output_pkl_name=args.pkl_path
        # Clear the file if it exists and start the JSON array
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('[')  # Start of JSON array
        first_iteration = True

    # Synchronize before processing
    dist.barrier()

    # Process data
    for batchs in tqdm(zip(dataloader1,dataloader2,dataloader3,dataloader4,dataloader5,dataloader6), desc=f"Rank {rank} Processing"):
        local_results = []  # Clear local results at the start of each iteration
        model.eval()
        with torch.no_grad():
            result_dict = {}
            result_dict['token'] = batchs[0]['llm_input']['token']
            result_dict['answer'] = []
            if batchs[0]['llm_input']['token'] != batchs[1]['llm_input']['token'] or batchs[0]['llm_input']['token'] != batchs[2]['llm_input']['token'] or batchs[0]['llm_input']['token'] != batchs[3]['llm_input']['token'] or batchs[0]['llm_input']['token'] != batchs[4]['llm_input']['token'] or batchs[0]['llm_input']['token'] != batchs[5]['llm_input']['token']:
                print('token not equal')
                continue
            for batch in batchs:
                data = batch['llm_input']
                # Move data to the correct device
                token=data['token']
                data = {k: v.to(device) for k, v in data.items() if k != "token"}
                if 'pixel_values' in data:
                    data['pixel_values'] = data['pixel_values'].to(torch.float16)
                
                # Decode and preprocess the question
                question = processor.decode(data['input_ids'][0], skip_special_tokens=False)
                # Generate output
                print('question:',question)
                output = model.generate(**data, max_new_tokens=500, do_sample=False)
                
                result = processor.decode(output[0], skip_special_tokens=True).split("assistant")[-1]
                
                # Store results locally
                cur_result_dict = {
                    'input': question,
                    'answer': result,
                }
                result_dict['answer'].append(cur_result_dict)

            
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

        # Write results to the file on rank 0
        if rank == 0:
            mode = 'w' if first_iteration else 'a'
            with open(output_file, mode, encoding='utf-8') as f:
                if first_iteration:
                    f.write('[')  # Start of JSON array
                    first_iteration = False
                for r in gathered_results:
                    for item in r:
                        json.dump(item, f, ensure_ascii=False)
                        f.write(',\n')

    # Finalize the JSON array on rank 0
    if rank == 0:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(']')  # End of JSON array
        print('测试回答全部写入完毕\n')

    # Synchronize before finalizing
    dist.barrier()

    # Rank 0 computes the average L2 distances

if __name__ == '__main__':
    main()
