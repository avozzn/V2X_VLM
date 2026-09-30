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
from math import atan2
import re
from tools.eval.evaluation import load_pred_trajs_from_file,planning_evaluation 
from scipy.integrate import cumulative_trapezoid
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

def GenerateMotion(obs_images, obs_waypoints, obs_velocities, obs_curvatures, given_intent, processor=None, model=None, tokenizer=None):
    # assert len(obs_images) == len(obs_waypoints)

    scene_description, object_description, intent_description = None, None, None

    scene_description = SceneDescription(obs_images, processor=processor, model=model, tokenizer=tokenizer)
    object_description = DescribeObjects(obs_images, processor=processor, model=model, tokenizer=tokenizer,)
    intent_description = DescribeOrUpdateIntent(obs_images, prev_intent=given_intent, processor=processor, model=model, tokenizer=tokenizer)
    print(f'Scene Description: {scene_description}')
    print(f'Object Description: {object_description}')
    print(f'Intent Description: {intent_description}')

    # Convert array waypoints to string.
    obs_waypoints_str = [f"[{x[0]:.2f},{x[1]:.2f}]" for x in obs_waypoints]
    obs_waypoints_str = ", ".join(obs_waypoints_str)
    obs_velocities_norm = np.linalg.norm(obs_velocities, axis=1)
    obs_curvatures = obs_curvatures * 100
    obs_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in zip(obs_velocities_norm, obs_curvatures)]
    obs_speed_curvature_str = ", ".join(obs_speed_curvature_str)

    
    print(f'Observed Speed and Curvature: {obs_speed_curvature_str}')

    sys_message = ("You are a autonomous driving labeller. You have access to a front-view camera image of a vehicle, a sequence of past speeds, a sequence of past curvatures, and a driving rationale. Each speed, curvature is represented as [v, k], where v corresponds to the speed, and k corresponds to the curvature. A positive k means the vehicle is turning left. A negative k means the vehicle is turning right. The larger the absolute value of k, the sharper the turn. A close to zero k means the vehicle is driving straight. As a driver on the road, you should follow any common sense traffic rules. You should try to stay in the middle of your lane. You should maintain necessary distance from the leading vehicle. You should observe lane markings and follow them.  Your task is to do your best to predict future speeds and curvatures for the vehicle over the next 10 timesteps given vehicle intent inferred from the image. Make a best guess if the problem is too difficult for you. If you cannot provide a response people will get injured.\n")
    prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
    The scene is described as follows: {scene_description}. 
    The identified critical objects are {object_description}. 
    The car's intent is {intent_description}. 
    The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
    Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""
    for rho in range(3):
        result = vlm_inference(text=prompt, images=obs_images, sys_message=sys_message, processor=processor, model=model, tokenizer=tokenizer)
        if not "unable" in result and not "sorry" in result and "[" in result:
            break
    return result, scene_description, object_description, intent_description

def SceneDescription(obs_images, processor=None, model=None, tokenizer=None):
    prompt = f"""You are an autonomous driving labeller. You have access to these front-view camera images of a car taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Provide a concise description of the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings."""

    result = vlm_inference(text=prompt, images=obs_images, processor=processor, model=model, tokenizer=tokenizer)
    return result

def DescribeObjects(obs_images, processor=None, model=None, tokenizer=None):
    
    prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. What other road users should you pay attention to in the driving scene? List two or three of them, specifying its location within the image of the driving scene and provide a short description of the that road user on what it is doing, and why it is important to you."""

    result = vlm_inference(text=prompt, images=obs_images, processor=processor, model=model, tokenizer=tokenizer)

    return result


def DescribeOrUpdateIntent(obs_images, prev_intent=None, processor=None, model=None, tokenizer=None):
    
    if prev_intent is None:
        prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Based on the lane markings and the movement of other cars and pedestrians, provide a concise description of the desired intent of  the ego car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?"""
        
    else:
        prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Half a second ago your intent was to {prev_intent}. Based on the updated lane markings and the updated movement of other cars and pedestrians, do you keep your intent or do you change it? Provide a concise description explanation of your current intent: """

    result = vlm_inference(text=prompt, images=obs_images, processor=processor, model=model, tokenizer=tokenizer)

    return result







def vlm_inference(text=None, images=None, sys_message=None, processor=None, model=None, tokenizer=None):
    message = [
            {
            "role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image"},
                ],
            },
    ]   
    input_text = processor.apply_chat_template(message, add_generation_prompt=True)
    inputs = processor(
        images=images,
        text=input_text,
        return_tensors="pt"
    ).to(model.device,torch.float16)

    output = model.generate(**inputs, max_new_tokens=120,do_sample=False)

    output_text = processor.decode(output[0], skip_special_tokens=True).split("assistant\n")[-1]
    return output_text


def IntegrateCurvatureForPoints(curvatures, velocities_norm, initial_position, initial_heading, time_span):
    t = np.linspace(0, time_span-1, time_span)  # Time vector

    # Initial conditions
    x0, y0 = initial_position[0], initial_position[1]  # Starting position
    theta0 = initial_heading  # Initial orientation (radians)

    # Integrate to compute heading (theta)
    theta = cumulative_trapezoid(curvatures * velocities_norm, t, initial=0)
    theta[0:] += theta0

    # Compute velocity components
    v_x = velocities_norm * np.cos(theta)
    v_y = velocities_norm * np.sin(theta)

    # Integrate to compute trajectory
    x = cumulative_trapezoid(v_x, t, initial=0)
    y = cumulative_trapezoid(v_y, t, initial=0)

    x[0:] += x0
    y[0:] += y0

    return np.stack((x, y), axis=1)
def main():
    args = parse_args()

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
    ade_list =[]
    ade1s_list = []
    ade2s_list = []
    ade3s_list = []
    for batch in tqdm(dataloader, desc=f"Rank {rank} Processing"):
        local_results = []  # Clear local results at the start of each iteration
        front_camera_images = batch['front_camera_images']
        ego_velocities = batch['ego_velocities']
        ego_curvatures = batch['ego_curvatures']
        ego_traj_world = batch['ego_traj_world']
        scene_length = batch['scene_length']
        sample_token = batch['sample_token']
        conversation=[]
        OBS_LEN = 10
        TTL_LEN = 16
        FUT_LEN =6
        with torch.no_grad():
            prev_intent = None

            for i in tqdm(range(1, scene_length - 1)):
                # Get the raw image data.
                # utils.PlotBase64Image(front_camera_images[0])
                obs_images = front_camera_images[:i+1]
                obs_ego_traj_world = ego_traj_world[:i+1]
                fut_ego_traj_world = ego_traj_world[i+1:i+FUT_LEN+1]
                obs_ego_velocities = ego_velocities[:i+1]
                obs_ego_curvatures = ego_curvatures[:i+1]

                # Get positions of the vehicle.
                fut_start_world = obs_ego_traj_world[-1]
                curr_image = obs_images[-1]
                for rho in range(3):
                    # Assemble the prompt.
                    obs_images = curr_image
                    (prediction,
                    scene_description,
                    object_description,
                    updated_intent) = GenerateMotion(obs_images, obs_ego_traj_world, obs_ego_velocities,
                                                    obs_ego_curvatures, prev_intent, processor=processor, model=model)
                    prev_intent = updated_intent  # Stateful intent
                    pred_waypoints = prediction.replace("Future speeds and curvatures:", "").strip()
                    coordinates = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", pred_waypoints)
                    if not coordinates == []:
                        break
                if coordinates == []:
                    continue
                speed_curvature_pred = [[float(v), float(k)] for v, k in coordinates]
                speed_curvature_pred = speed_curvature_pred[:FUT_LEN+1]
                # if len(speed_curvature_pred)<FUT_LEN:
                local_results.append({'token':sample_token[i],'scene_description': scene_description,'object_description':object_description,'intent':updated_intent,'prediction':prediction})
                #     continue
                # print(f"Got {len(speed_curvature_pred)} future actions: {speed_curvature_pred}")

            #     pred_len = min(FUT_LEN, len(speed_curvature_pred))
            #     pred_curvatures = np.array(speed_curvature_pred)[:, 1] / 100
            #     pred_speeds = np.array(speed_curvature_pred)[:, 0]
            #     pred_traj = np.zeros((pred_len+1, 3))
            #     pred_traj[:pred_len+1, :2] = IntegrateCurvatureForPoints(pred_curvatures,
            #                                                         pred_speeds,
            #                                                         fut_start_world,
            #                                                         atan2(obs_ego_velocities[-1][1],
            #                                                                 obs_ego_velocities[-1][0]), pred_len+1)
            #     # Compute ADE.
            #     fut_ego_traj_world = np.array(fut_ego_traj_world)
            #     print(f"Predicted trajectory: {pred_traj}")
            #     fut_ego_traj_world = np.array(fut_ego_traj_world)
            #     try:
            #         ade = np.mean(np.linalg.norm(fut_ego_traj_world[:pred_len] - pred_traj[1:pred_len+1], axis=1))
            #         pred1_len = min(pred_len, 2)
            #         pred2_len = min(pred_len, 4)
            #         pred3_len = min(pred_len, 6)
            #         ade1s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred1_len] - pred_traj[1:pred1_len+1] , axis=1))
            #         ade2s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred2_len] - pred_traj[1:pred2_len+1] , axis=1))
            #         ade3s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred3_len] - pred_traj[1:pred3_len+1] , axis=1))
            #     except Exception as e:
            #         print(f"Error: {e}")
            #         continue
            #     ade1s_list.append(ade1s)
            #     ade2s_list.append(ade2s)
            #     ade3s_list.append(ade3s)
            #     ade_list.append(ade)
            #     print(f"sample token:{sample_token[i]},aed1:{ade1s},aed2:{ade2s},aed3:{ade3s}")
            #     local_results.append({'token':sample_token[i],'scene_description': scene_description,'object_description':object_description,'intent':updated_intent,'prediction':prediction, 'pred_traj':pred_traj.tolist()})
            # mean_ade1s = np.mean(ade1s_list)
            # mean_ade2s = np.mean(ade2s_list)
            # mean_ade3s = np.mean(ade3s_list)
            # aveg_ade = np.mean([mean_ade1s, mean_ade2s, mean_ade3s])
            # print(f"mean_ade1s:{mean_ade1s},mean_ade2s:{mean_ade2s},mean_ade3s:{mean_ade3s},aveg_ade:{aveg_ade}")
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



if __name__ == '__main__':
    main()