import argparse
import logging
import transformers
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
import deepspeed
from loaders import LOADERS
from deepspeed.ops.transformer.inference import DeepSpeedTransformerInference
from peft import PeftModel, LoraConfig, TaskType
from arguments import  LoraArguments
from train_utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)
from arguments import ModelArguments, DataArguments, TrainingArguments, LoraArguments
from supported_models import MODULE_KEYWORDS

import pickle
import ast
from eval.evaluation import load_pred_trajs_from_file,planning_evaluation 
def parse_args():
    parser2 = transformers.HfArgumentParser(
        (ModelArguments)
    )
    parser = argparse.ArgumentParser(description='Distributed Test a 3D detector')
    parser.add_argument('--config', help='train config file path')
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
    parser.add_argument(
        "--use_lora",
        type=str,
        required=True,
        help=""
    )
    parser.add_argument(
        "--q_lora",
        required=True,
        help=""
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        required=True,
        help=""
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        required=True,
        help=""
    )
    parser.add_argument(
        "--lora_bias",
        required=True,
        help=""
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        required=True,
        help=""
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        required=True,
        help=""
    )
    args, remain= parser.parse_known_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    model_args = parser2.parse_args_into_dataclasses(remain)
    return args,model_args



def main():
    args,model_args = parse_args()
    model_args=model_args[0]
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])


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

    # Build the runner
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    # Build the dataset
    dataset_cfg = cfg.test_dataloader.pop('dataset')
    datasets = DATASETS.build(dataset_cfg)



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
    model = LlavaForConditionalGeneration.from_pretrained(
        original_model_id, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True, 
    )
    
    llm_keys = MODULE_KEYWORDS[model_args.model_family_id]["llm"]
    named_modules = {n: m for n, m in model.named_modules()}
    lora_modules = []
    if args.use_lora:
        lora_modules.extend(find_all_linear_names(named_modules, llm_keys))
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_modules,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type="CAUSAL_LM",
            inference_mode=True,
        )

    model_id = args.model_path
    model = PeftModel.from_pretrained(model, model_id, config=lora_config)
    processor = AutoProcessor.from_pretrained(original_model_id)



    ds_engine = deepspeed.init_inference(model,
                                        dtype="fp16",
                                        checkpoint=None,
                                        replace_with_kernel_inject=True)
    
    device = next(ds_engine.module.parameters()).device


    # Initialize result storage
    if rank == 0:
        output_pkl_name=args.pkl_path
        output_file = args.result_path
        # Clear the file if it exists and start the JSON array
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('[')  # Start of JSON array
        first_iteration = True

    # Synchronize before processing
    dist.barrier()

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
            
            # Decode and preprocess the question
            question = processor.decode(data['input_ids'][0], skip_special_tokens=False)
            # Generate output
            
            output = ds_engine._generate(**data, max_new_tokens=500, do_sample=False)
            
            result = processor.decode(output[0], skip_special_tokens=True)
            
            # Store results locally
            result_dict = {
                'token':token,
                'input': question,
                'answer': result,
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
    if rank == 0:
        with open(output_file, 'r', encoding='utf-8') as file:
            try:
                # Fix the JSON array
                file.seek(0)
                content = file.read()
                if content.endswith(',\n]'):
                    content = content.rstrip(',\n]') + ']'
                    with open(output_file, 'w', encoding='utf-8') as write_file:
                        write_file.write(content)
                data = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
                data = []

        if not data:
            print("No data to process for L2 distance.")
            return


        traj_dict = {}

        exist_dict = {}
        for item in data:
            # 获取 token
            token = item['token']
            # 提取轨迹信息
            traj = item['answer'].split("\n")[-1]
            # 将字符串转换为列表
            traj = ast.literal_eval(traj)
            # 将列表转换为 NumPy 数组
            traj = np.array(traj)
            # 将轨迹数据添加到 traj_dict 中
            traj_dict[token] = np.expand_dims(traj, axis=0)

        exist_dict.update(traj_dict)

        # 将更新后的字典保存到 result.pkl 文件中
        with open(output_pkl_name, 'wb') as f:
            pickle.dump(exist_dict, f)
        pred_trajs_dict = load_pred_trajs_from_file(output_pkl_name)
        evaluation_data=planning_evaluation(pred_trajs_dict, subset=None, only_vehicle=False)
        with open(output_file, 'a', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=4)
        print("Test finish\n")
    dist.barrier()

if __name__ == '__main__':
    main()
