import base64
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
import os.path
import re
import argparse
from datetime import datetime
from math import atan2
from eval.evaluation import load_pred_trajs_from_file,planning_evaluation
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from nuscenes import NuScenes
from pyquaternion import Quaternion
from scipy.integrate import cumulative_trapezoid
import json
from tools.eval.emma_utils import  IntegrateCurvatureForPoints
from nuscenes.utils.geometry_utils import transform_matrix
OBS_LEN = 10
FUT_LEN = 6
TTL_LEN = OBS_LEN + FUT_LEN

def transform_traj(traj=None,sample_token_list=None,nusc=None):
    sample = nusc.get('sample', sample_token_list[0])
    sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    ego_fut_trajs = []
    for i in range(7):
        traj_curr=traj[i]
        if i<len(sample_token_list)-1:
            sample_token=sample_token_list[i]
            sample = nusc.get('sample', sample_token)
            cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            sd_ep_camera = nusc.get("ego_pose", cam_front_data["ego_pose_token"])
            sd_cs_camera = nusc.get("calibrated_sensor", cam_front_data["calibrated_sensor_token"])
        if i>=len(sample_token_list)-1:
            sample_token=sample_token_list[-1]
            sample = nusc.get('sample', sample_token)
            cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            sd_ep_camera = nusc.get("ego_pose", cam_front_data["ego_pose_token"])
            sd_cs_camera = nusc.get("calibrated_sensor", cam_front_data["calibrated_sensor_token"])
        sd_ep_camera['translation']=traj_curr
        global_from_ego_camera = transform_matrix(sd_ep_camera["translation"], Quaternion(sd_ep_camera["rotation"]), inverse=False)
        ego_from_sensor_camera = transform_matrix(sd_cs_camera["translation"], Quaternion(sd_cs_camera["rotation"]), inverse=False)
        pose_camera = global_from_ego_camera.dot(ego_from_sensor_camera)
        traj_curr=pose_camera[:3,3]
        ego_fut_trajs.append(traj_curr)
    ego_fut_trajs=np.array(ego_fut_trajs)
    ego_fut_trajs = ego_fut_trajs - np.array(pose_record['translation'])
    rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
    ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
    ego_fut_trajs = ego_fut_trajs- np.array(cs_record['translation'])
    rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
    ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
    ego_fut_trajs=np.expand_dims(ego_fut_trajs[1:,:2], axis=0)
    return ego_fut_trajs
if __name__ == '__main__':
    ade05s_list=[]
    ade1s_list = []
    ade15s_list=[]
    ade2s_list = []
    ade25s_list=[]
    ade3s_list = []
    # Load the dataset
    nusc = NuScenes(version='v1.0-trainval', dataroot='/home/ldc/Projects/RoboLLM/zoo/MMDrive/data/nuscenes', verbose=True)
    with open('/home/ldc/Projects/RoboLLM/zoo/MMDrive/data/openemma/scene_keys.json', 'r') as scene_list:
        scenes = json.load(scene_list)
    print(f"Number of scenes: {len(scenes)}")
    error_num=0
    traj_dict={}
    for scene in scenes:
        token = scene
        scene=nusc.get('scene', token)
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        sample_token = []
        curr_sample_token = first_sample_token
        name = scene['name']
        description = scene['description']
        if name in ["scene-0103", "scene-1077"]:
            continue
        # Get all image and pose in this scene
        front_camera_images = []
        ego_poses = []
        camera_params = []
        curr_sample_token = first_sample_token
        while True:
            sample_token.append(curr_sample_token)
            sample = nusc.get('sample', curr_sample_token)
            cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
            ego_poses.append(pose)
            camera_params.append(nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token']))
            if curr_sample_token == last_sample_token:
                break
            curr_sample_token = sample['next']

        scene_length = len(sample_token)
        print(f"Scene {name} has {scene_length} frames")

        if scene_length < TTL_LEN:
            print(f"Scene {name} has less than {TTL_LEN} frames, skipping...")
            continue

        ## Compute interpolated trajectory.
        # Get the velocities of the ego vehicle.
        ego_poses_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]
        ego_poses_world = np.array(ego_poses_world)
        ego_velocities = np.zeros_like(ego_poses_world)
        ego_velocities[1:] = ego_poses_world[1:] - ego_poses_world[:-1]
        ego_velocities[0] = ego_velocities[1]

        # Get the curvature of the ego vehicle.
        ego_velocities_norm = np.linalg.norm(ego_velocities, axis=1)


        # Get the waypoints of the ego vehicle.
        ego_traj_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]

        prev_intent = None
        cam_images_sequence = []
        for i in range(1, scene_length - 1):
            # Get the raw image data.
            # utils.PlotBase64Image(front_camera_images[0])
            obs_ego_traj_world = ego_traj_world[:i+1]
            fut_ego_traj_world = ego_traj_world[i+1:i+FUT_LEN+1]
            obs_ego_velocities = ego_velocities[:i+1]
            sample_token_list = sample_token[i:i+FUT_LEN+1]
            sample_token_cur=sample_token[i]
            # Get positions of the vehicle.
            fut_start_world = obs_ego_traj_world[-1]
            #/home/ldc/Projects/RoboLLM/zoo/MMDrive/vis/emma_fog_3_20250330_051150.json
            #/home/ldc/Projects/RoboLLM/zoo/MMDrive/vis/emma_fog_5_20250329_040715.json
            #/home/ldc/Projects/RoboLLM/zoo/MMDrive/vis/emma_motion_1_20250328_025337.json
            #/home/ldc/Projects/RoboLLM/zoo/MMDrive/vis/emma_motion_3_20250327_023150.json
            #/home/ldc/Projects/RoboLLM/zoo/MMDrive/vis/emma_motion_5_20250326_010744.json
            # /home/ldc/Projects/RoboLLM/zoo/MMDrive/vis/emma_rain_1_20250328_032624.json
            #/home/ldc/Projects/RoboLLM/zoo/MMDrive/vis/emma_rain_3_20250327_023116.json
            #/home/ldc/Projects/RoboLLM/zoo/MMDrive/vis/emma_rain_5__20250326_010337.json
            # print("file_path",file_path)
            file_path = '/home/ldc/Projects/RoboLLM/zoo/MMDrive/vis/result_emma_overwrite_rules_20250405_043237.json'

            with open(file_path, 'r') as f:
                input_data = json.load(f)
            answer=None
            for item in input_data:
                if item['token'] == sample_token[i]:
                    answer = item['prediction']
                    break
            if answer==None:
                continue
            prediction=answer
            pred_waypoints = prediction.replace("Future speeds and curvatures:", "").strip()
            coordinates = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", pred_waypoints)
            speed_curvature_pred = [[float(v), float(k)] for v, k in coordinates]
            if len(speed_curvature_pred)<FUT_LEN+1:
                error_num=error_num+1
                continue
            speed_curvature_pred = speed_curvature_pred[:FUT_LEN+1]
            
            pred_len = FUT_LEN+1
            pred_curvatures = np.array(speed_curvature_pred)[:, 1] / 100
            pred_speeds = np.array(speed_curvature_pred)[:, 0]
            pred_traj = np.zeros((pred_len, 3))
            pred_traj[:pred_len, :2] = IntegrateCurvatureForPoints(pred_curvatures,
                                                                   pred_speeds,
                                                                   fut_start_world,
                                                                   atan2(obs_ego_velocities[-1][1],
                                                                         obs_ego_velocities[-1][0]), pred_len)
            fut_ego_traj_world = np.array(fut_ego_traj_world)
            try:                
                if len(fut_ego_traj_world) < 6:
                    # 复制最后一个点，填充到 (6, 3)
                    last_point = fut_ego_traj_world[-1]  # 取最后一个点
                    num_pad = 6 - len(fut_ego_traj_world)  # 需要填充的数量
                    padded_points = np.tile(last_point, (num_pad, 1))  # 复制为 (num_pad, 3)
                    fut_ego_traj_world = np.vstack([fut_ego_traj_world, padded_points])  # 拼接成 (6, 3)
                pred1_len = min(pred_len, 2)
                pred2_len = min(pred_len, 4)
                pred3_len = min(pred_len, 6)
                ade05s = np.mean(np.linalg.norm(fut_ego_traj_world[:1] - pred_traj[1:2] , axis=1))
                ade1s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred1_len] - pred_traj[1:pred1_len+1] , axis=1))
                ade15s = np.mean(np.linalg.norm(fut_ego_traj_world[:3] - pred_traj[1:4] , axis=1))
                ade2s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred2_len] - pred_traj[1:pred2_len+1] , axis=1))
                ade25s = np.mean(np.linalg.norm(fut_ego_traj_world[:5] - pred_traj[1:6] , axis=1))
                ade3s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred3_len] - pred_traj[1:pred3_len+1] , axis=1))
            except Exception as e:
                print(f"Error: {e}")
                continue
            ade05s_list.append(ade05s)
            ade1s_list.append(ade1s)
            ade15s_list.append(ade15s)
            ade2s_list.append(ade2s)
            ade25s_list.append(ade25s)
            ade3s_list.append(ade3s)
            print(f"sample token:{sample_token[i]},ade1:{ade1s},ade2:{ade2s},ade3:{ade3s}")
            pred_traj_transform=transform_traj(traj=pred_traj,sample_token_list=sample_token_list,nusc=nusc)
            traj_dict[sample_token_cur] = pred_traj_transform
        mean_ade05s = np.mean(ade05s_list)
        mean_ade1s = np.mean(ade1s_list)
        mean_ade15s = np.mean(ade15s_list)
        mean_ade2s = np.mean(ade2s_list)
        mean_ade25s = np.mean(ade25s_list)
        mean_ade3s = np.mean(ade3s_list)
        aveg_ade = np.mean([mean_ade1s, mean_ade2s, mean_ade3s])
        print(f"mean_ade0.5s:{mean_ade05s},mean_ade1s:{mean_ade1s},mean_ade1.5s:{mean_ade15s},mean_ade2s:{mean_ade2s},mean_ade2.5s:{mean_ade25s},mean_ade3s:{mean_ade3s},aveg_ade:{aveg_ade}")
    mean_ade05s = np.mean(ade05s_list)
    mean_ade1s = np.mean(ade1s_list)
    mean_ade15s = np.mean(ade15s_list)
    mean_ade2s = np.mean(ade2s_list)
    mean_ade25s = np.mean(ade25s_list)
    mean_ade3s = np.mean(ade3s_list)
    aveg_ade = np.mean([mean_ade1s, mean_ade2s, mean_ade3s])
    print(f"len of list:{len(ade1s_list)}")
    print(f"mean_ade0.5s:{mean_ade05s},mean_ade1s:{mean_ade1s},mean_ade1.5s:{mean_ade15s},mean_ade2s:{mean_ade2s},mean_ade2.5s:{mean_ade25s},mean_ade3s:{mean_ade3s},aveg_ade:{aveg_ade}")
    evaluation_data=planning_evaluation(traj_dict, subset=None, only_vehicle=False)
    print(f"error num{error_num}")