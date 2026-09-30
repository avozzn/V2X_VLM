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
from loaders import LOADERS
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import pickle
from peft import PeftModel, LoraConfig, TaskType
import ast
from eval.evaluation import load_pred_trajs_from_file,planning_evaluation 
from tools.TTA_utils.inference_tta import test
from tools.TTA_utils.TTA_config import tta_config,num_iters
from transformers import Trainer
from arguments import ModelArguments, DataArguments, TrainingArguments, LoraArguments
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tools.TTA_utils.LLM_TTA import TTA
from train_utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)
from collators import COLLATORS
from accelerate.utils import DistributedType, DeepSpeedPlugin
from supported_models import MODULE_KEYWORDS
import os
import gc
import os
os.environ["WANDB_PROJECT"]= "lmms-ft"
os.environ["WANDB_MODE"] = "offline"
def parse_args():
    parser2 = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
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
    parser.add_argument('--local_rank','--local-rank', type=int, default=0)
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
        "--corr_image_type",
        type=str,
        help=""
    )
    parser.add_argument(
        "--corr_severity",
        type=str,
        help=""
    )
    parser.add_argument(
        "--corr_ego_type",
        type=str,
        help=""
    )
    parser.add_argument(
        "--corr_prompt_type",
        type=str,
        help=""
    )
    parser.add_argument(
        "--pkl_path",
        type=str,
        required=True,
        help="Path to save the results JSON file."
    )
    args,remain = parser.parse_known_args()
    model_args, data_args, training_args, lora_args = parser2.parse_args_into_dataclasses(remain)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    return args,model_args, data_args, training_args, lora_args

def calculate_l2(output_file,output_pkl_name):
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
    i=0
    for line_num, item in enumerate(data, 1):
        try:
            # 获取 token
            token = item['token']
            # 提取轨迹信息
            traj = item['answer'].split("\n")[-1]
            # 将字符串转换为列表
            traj = ast.literal_eval(traj)
            # 将列表转换为 NumPy 数组
            traj = np.array(traj)
            # 将轨迹数据添加到 traj_dict 中
            assert traj.shape == (6,2)
            traj_dict[token] = np.expand_dims(traj, axis=0)
        except Exception as e:
            # 打印详细的错误信息（包括错误类型和具体描述）
            print(f"Error processing line {line_num}: {type(e).__name__} - {str(e)}")
            i=i+1
            continue  # 跳过当前数据项，继续处理下一个
    print(f"error num:{i}")
    errornum=f"error num:{i}"
    exist_dict.update(traj_dict)

    # 将更新后的字典保存到 result.pkl 文件中
    with open(output_pkl_name, 'wb') as f:
        pickle.dump(exist_dict, f)
    pred_trajs_dict = load_pred_trajs_from_file(output_pkl_name)
    evaluation_data=planning_evaluation(pred_trajs_dict, subset=None, only_vehicle=False)
    with open(output_file, 'a', encoding='utf-8') as f:
        json.dump(evaluation_data, f, indent=4)
        json.dump(errornum,f,indent=4)
    print("Test finish\n")


def build_dataloader(args,config,world_size,rank,test=False):
    
    cfg = Config.fromfile(config)

    # Replace Ceph backend if specifieds
    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher

    # Determine work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs/TTA',
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
    if args.corr_image_type is not None:
        test_pipeline=dataset_cfg.pipeline
        corruption_severity_dict={
            args.corr_image_type:int(args.corr_severity),
        }
        if args.corr_image_type !="dark_sim":
            for item in test_pipeline:
                if isinstance(item, dict) and item.get('type') == 'Processed_sensor':
                    item['corruption_severity_dict'] = corruption_severity_dict
                    item['vis_root']="vis/tta/"
                    break
        test_pipeline.insert(1, {
        'type': 'image_corruption',
        'corruption_severity_dict': corruption_severity_dict    
        })


    if args.corr_ego_type is not None:
        test_pipeline=dataset_cfg.pipeline
        for idx,item in enumerate(test_pipeline):
            if isinstance(item, dict) and item.get('type') == 'Processed_sensor':
                test_pipeline.insert(idx + 1, {
                'type': 'Processed_sensor_corruption',
                'sensor_corruption_level': args.corr_severity  
                })
                break

    if args.corr_prompt_type is not None:
        test_pipeline=dataset_cfg.pipeline
        test_pipeline.insert(-1, {
                'type': 'prompt_corruption',
                'attack_type': args.corr_severity  
                })



    

    datasets = DATASETS.build(dataset_cfg)

    if not test:
        sampler = DistributedSampler(datasets, num_replicas=world_size, rank=rank, shuffle=True,seed=147)
    else:
        sampler = DistributedSampler(datasets, num_replicas=world_size, rank=rank, shuffle=False)

    if not test:
        dataloader= DataLoader(
        dataset=datasets,
        batch_size=1,  # Adjust as needed
        sampler=sampler,
        collate_fn=lambda batch: batch[0],
        num_workers=4,  # Adjust based on your system
        pin_memory=True
    ) 
    else:
        dataloader= DataLoader(
        dataset=datasets,
        batch_size=1,  # Adjust as needed
        sampler=sampler,
        collate_fn=lambda batch: batch[0],
        num_workers=4,  # Adjust based on your system
        pin_memory=True
    ) 


    return dataloader


def main():
    args,model_args, data_args, training_args, lora_args=parse_args()



    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED


    # Load the model



    loader = LOADERS[model_args.model_family_id](
        model_hf_path=model_args.model_hf_path,
        model_local_path=model_args.model_local_path,
        compute_dtype=compute_dtype,
        use_flash_attn=training_args.use_flash_attn,
        device_map=device
    )

    model, tokenizer, processor, load_config = loader.load()
    tokenizer.model_max_length = training_args.model_max_length


    original_model_id = args.origin_model_path
    
    output_file = args.result_path
    output_pkl_name=args.pkl_path

    # lora preparation
    vision_encoder_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_encoder"]
    if not training_args.train_vision_encoder:
        rank0_print(f"Vision encoder is freezed... including:")
        for module in vision_encoder_keys:
            rank0_print(f"\t{module}")
            eval(f"model.{module}").requires_grad_(False)

    vision_projector_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_projector"]
    if not training_args.train_vision_projector:
        rank0_print(f"Vision projector is freezed... including:")
        for module in vision_projector_keys:
            rank0_print(f"\t{module}")
            eval(f"model.{module}").requires_grad_(False)

    # other components preparation (e.g., image_newline, vision_resampler)
    # we will just freeze these
    if "others" in MODULE_KEYWORDS[model_args.model_family_id]:
        rank0_print(f"Other multimodal component is freezed... including:")
        for other_key in MODULE_KEYWORDS[model_args.model_family_id]["others"]:
            rank0_print(f"\t{other_key}")
            eval(f"model.{other_key}").requires_grad_(False)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()


    llm_keys = MODULE_KEYWORDS[model_args.model_family_id]["llm"]
    if not (lora_args.use_lora or (training_args.train_vision_encoder and lora_args.use_vision_lora)):
        rank0_print("No LoRA enabled...")        
    else:
        named_modules = {n: m for n, m in model.named_modules()}
        lora_modules = []
        full_modules = []
        
        if lora_args.use_lora:
            rank0_print("LoRA for LLM enabled...")
            lora_modules.extend(find_all_linear_names(named_modules, llm_keys))
            
        else:
            rank0_print("LLM will be fully trained...")
            full_modules.extend(llm_keys)
        
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_modules,
            modules_to_save=full_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            init_lora_weights=False, 
            task_type="CAUSAL_LM",
        )
        model_id = args.model_path

        model = PeftModel.from_pretrained(model, model_id, config=lora_config)

        for name, param in model.named_parameters():
            if "lora" in name: 
                param.requires_grad = True
                
    data_loaders={type:build_dataloader(args,config,world_size,rank) for type,config in tta_config.items()}

    test_dataloader=build_dataloader(args,args.config,world_size,rank,test=True)

    data_collator = COLLATORS[model_args.model_family_id](
        config=load_config,
        tokenizer=tokenizer,
        processor=processor,
        mask_question_tokens=training_args.mask_question_tokens
    )

    trainer = TrainerWithCustomSampler(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=None
    )
    
    TTA(model,data_loaders,rank,num_iters,processor,trainer)

    

    torch.cuda.empty_cache()
    for key in list(data_loaders.keys()):
            del data_loaders[key]

    gc.collect()
        


    test(model,test_dataloader,rank,world_size,output_file,processor)


    if rank == 0:
        calculate_l2(output_file,output_pkl_name)

    

if __name__ == '__main__':
    main()
