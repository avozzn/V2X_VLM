#----------------------------------------------------------------#
# UniV2X: End-to-End Autonomous Driving through V2X Cooperation  #
# Source code: https://github.com/AIR-THU/UniV2X                 #
# Copyright (c) DAIR-V2X. All rights reserved.                   #
# Modified from UniAD (https://github.com/OpenDriveLab/UniAD)    #
#----------------------------------------------------------------#
import copy
import numpy as np
import torch
import mmcv
from mmengine.registry import DATASETS
from mmcv.transforms import to_tensor
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.datasets import NuScenesDataset
from os import path as osp
# NuScenes 官方库
from nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.eval.common.config import config_factory
from nuscenes.prediction import convert_local_coords_to_global
import tempfile
import random
import pickle
import json
from prettytable import PrettyTable
from mmengine.fileio import FileClient 
from mmdet3d.datasets import Det3DDataset
from typing import List, Dict, Union, Any
# from projects.mmdet3d_plugin.datasets.eval_utils.nuscenes_eval import NuScenesEval_custom,TrackingEval_custom
# from projects.mmdet3d_plugin.datasets.eval_utils.nuscenes_eval_motion import MotionEval
# from projects.mmdet3d_plugin.datasets.data_utils.data_utils import lidar_nusc_box_to_global, obtain_map_info, output_to_nusc_box, output_to_nusc_box_det
from projects.mmdet3d_plugin.datasets.data_utils.spd_trajectory_api import SPDTraj
from projects.mmdet3d_plugin.datasets.data_utils.planning_metadata_v2 import (
    generate_planning_metadata_v2)
from projects.mmdet3d_plugin.datasets.data_utils.vector_map import VectorizedLocalMap
# from projects.mmdet3d_plugin.datasets.data_utils.rasterize import preprocess_map, draw_lane_on_image
from projects.mmdet3d_plugin.datasets.eval_utils.map_api import NuScenesMap
# from projects.mmdet3d_plugin.datasets.data_utils.trajectory_api import NuScenesTraj
#####
import cv2
import os
from typing import Callable, List, Union
class_names_nuscenes_mappings = {
    'Car': 'car',
    'Truck': 'car',
    'Van': 'car',
    'Bus': 'car',
    'Motorcyclist': 'bicycle',
    'Cyclist': 'bicycle',
    'Tricyclist': 'bicycle',
    'Barrowlist': 'bicycle',
    'Pedestrian': 'pedestrian',
    'TrafficCone': 'traffic_cone',
    'car': 'car',
    'bicycle': 'bicycle',
    'pedestrian': 'pedestrian',
    'traffic_cone': 'traffic_cone'
}

@DATASETS.register_module()
class SpdVehicleE2EVLMDataset(NuScenesDataset):
    r"""NuScenes E2E Dataset.

    This dataset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self,
                 #robollm
                data_root: str=None,
                ann_file: str=None,
                json_file:str=None,
                v2x_side: str = None,
                task:str = None,
                pipeline: List[Union[dict, Callable]] = [],
                test_mode: bool = False,
                metainfo: dict = None,

                queue_length=4,
                bev_size=(200, 200),
                patch_size=(102.4, 102.4),
                canvas_size=(200, 200),
                overlap_test=False,
                predict_steps=9,
                planning_steps=9,
                past_steps=4,
                fut_steps=4,
                use_nonlinear_optimizer=False,
                lane_ann_file=None,
                eval_mod=None,
                inference_wo_label=False,
                command_file=None,
                # For debug
                is_debug=False,
                len_debug=30,

                # Occ dataset
                enbale_temporal_aug=False,
                occ_receptive_field=3,
                occ_n_future=4,
                occ_n_future_only_occ=4,
                occ_filter_invalid_sample=False,
                occ_filter_by_valid_flag=False,
                file_client_args =dict(backend='disk'),
                split_datas_file="",
                class_range=None,
                new_range_100=False,
                other_agent_names=[],
                *args,
                **kwargs):
        # robollm
        self.data_root = data_root
        self.ann_file = data_root + ann_file
        self.CLASSES = metainfo['classes']
        self.inference_wo_label = inference_wo_label
        self.command_file = command_file
        self.split_datas_file = split_datas_file
        self.v2x_side = v2x_side
        if self.v2x_side not in ['vehicle_side', 'infrastructure_side', 'cooperative', '']:
            raise Exception('v2x_side is not correct with {}'.format(self.v2x_side))
        self.file_client_args = file_client_args
        self.file_client = FileClient(**file_client_args)

        self.tmp_dataset_type = 'spd'
        self.with_velocity = True ####### 
        if self.tmp_dataset_type not in ['spd', 'nuscenes']:
            raise Exception('tmp_dataset_type is not correct with {}'.format(self.tmp_dataset_type))

        self.is_debug = is_debug
        self.len_debug = len_debug
        self.task=task
        
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.predict_steps = predict_steps
        self.planning_steps = planning_steps
        self.past_steps = past_steps
        self.fut_steps = fut_steps
        self.scene_token = None
        # self.lane_infos = self.load_annotations(lane_ann_file) \
        #     if lane_ann_file else None
        self.eval_mod = eval_mod
        self.visualize_locations =['yizhuang06', 'yizhuang08', 'yizhuang09', 'yizhuang10', 'yizhuang13', 'yizhuang16']
        self.use_nonlinear_optimizer = use_nonlinear_optimizer

        self.nusc = NuScenes(version='v1.0-trainval',
                             dataroot= self.data_root, verbose=True)


        self.map_num_classes = 3
        #自车长宽
        if canvas_size[0] == 50:
            self.thickness = 1
        elif canvas_size[0] == 200:
            self.thickness = 2
        else:
            assert False
        self.angle_class = 36
        self.patch_size = patch_size
        self.canvas_size = canvas_size
        if self.tmp_dataset_type == 'spd':
            self.nusc_maps = {
                    'yizhuang06': NuScenesMap(dataroot=self.data_root, map_name='yizhuang06'),
                    'yizhuang08': NuScenesMap(dataroot=self.data_root, map_name='yizhuang08'),
                    'yizhuang09': NuScenesMap(dataroot=self.data_root, map_name='yizhuang09'),
                    'yizhuang10': NuScenesMap(dataroot=self.data_root, map_name='yizhuang10'),
                    'yizhuang13': NuScenesMap(dataroot=self.data_root, map_name='yizhuang13'),
                    'yizhuang16': NuScenesMap(dataroot=self.data_root, map_name='yizhuang16')
            }
        # else:
        #     self.nusc_maps = {
        #         'boston-seaport': NuScenesMap(dataroot=self.data_root, map_name='boston-seaport'),
        #         'singapore-hollandvillage': NuScenesMap(dataroot=self.data_root, map_name='singapore-hollandvillage'),
        #         'singapore-onenorth': NuScenesMap(dataroot=self.data_root, map_name='singapore-onenorth'),
        #         'singapore-queenstown': NuScenesMap(dataroot=self.data_root, map_name='singapore-queenstown'),
        #     }
        # 将地图转换为向量格式
        self.vector_map = VectorizedLocalMap(
            self.data_root,
            patch_size=self.patch_size,
            canvas_size=self.canvas_size,
            dataset_type=self.tmp_dataset_type)
        
        self.enbale_temporal_aug = enbale_temporal_aug
        assert self.enbale_temporal_aug is False

        self.occ_receptive_field = occ_receptive_field  # past + current
        self.occ_n_future = occ_n_future  # future only
        self.occ_filter_invalid_sample = occ_filter_invalid_sample
        self.occ_filter_by_valid_flag = occ_filter_by_valid_flag
        self.occ_only_total_frames = 7  # NOTE: hardcode, not influenced by planning
        self.occ_only_total_frames = occ_n_future_only_occ + occ_receptive_field   # 7 

        self.class_range=class_range
        self.new_range_100 = new_range_100
        self.other_agent_names = other_agent_names
        # if self.tmp_dataset_type == 'spd':
        #     #输出每个物体的未来轨迹、未来 6 秒内轨迹的有效性掩码、和过去两秒历史轨迹和历史轨迹的有效性掩码
        #     self.traj_api = SPDTraj(self.nusc,
        #                                 self.predict_steps,
        #                                 self.planning_steps,
        #                                 self.past_steps,
        #                                 self.fut_steps,
        #                                 self.with_velocity,
        #                                 self.CLASSES,
        #                                 self.box_mode_3d,
        #                                 self.use_nonlinear_optimizer)
            
        ##json_file
        self.json_file=json_file
        
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)
       
        
    def __len__(self)-> int:
            return self.length
    #根据时间戳对 data_list 进行排序 读取pkl文件中的参数信息 存储到self.data_list里 在NuScenesDataset的__init__里调用

    # def parse_data_info(self, raw_data_info: dict) -> dict:
    # # NOTE: We skip the parent's annotation parsing and use our custom logic.
    #     data_info = raw_data_info.copy()
    #     if 'lidar_path' in data_info:
    #         data_info['pts_filename'] = data_info.pop('lidar_path')
    #     data_info['instances'] = []
    #     data_info = super(NuScenesDataset, self).parse_data_info(data_info)
    #     return data_info

    def load_data_list(self) -> List[Dict]:
        """Load annotations from annotation file and potentially merge with prompt data.

        This function overrides the base function (BaseDataset.load_data_list) 
        to ensure the SPD data is loaded correctly and then processed with VLM prompts.
        """
        print("Loading data list...") ### 第一步
        data_list = super().load_data_list()
        
        data_list = list(sorted(data_list, key=lambda e: e['timestamp']))

        
        # output_dir= "/home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM/tools/eval/gt"
        # self.create_v2x_gt_files(data_list,output_dir)
        if self.task == 'planning' or self.task == 'vqa':
            data_list = self.process_json_file(data_list, self.json_file)

        self.length = len(data_list)
        return data_list

    def create_v2x_gt_files(self, data_list: List[Dict],output_dir):
        """
        遍历数据集，将所有 SDC 轨迹收集到字典中，
        然后一次性保存为 .pkl 文件。
        
        Args:
            dataset: 你实例化的数据集对象。
            output_dir: 保存 .pkl 文件的目录。
        """
        self.traj_api = SPDTraj(self.nusc,
                                        self.predict_steps,
                                        self.planning_steps,
                                        self.past_steps,
                                        self.fut_steps,
                                        self.with_velocity,
                                        self.CLASSES,
                                        self.box_mode_3d,
                                        self.use_nonlinear_optimizer)
            
        # 这两个大字典用来收集所有数据
        gt_trajs_dict = {}
        gt_trajs_mask_dict = {}
        for info in data_list:
            token = info['token']
            sdc_vel = self.traj_api.sdc_vel_info[token][2:] # 自车速度
            past_vel = self.traj_api.sdc_vel_info[token][:2] # 自车过去1秒的速度
            gt_sdc_past_traj, gt_sdc_past_traj_mask,sdc_planning, sdc_planning_mask, command = self.traj_api.get_sdc_planning_label(
            info['token'])
            if command == -1:
                command_vlm = np.array([-1])
            else:
                command_vlm = generate_planning_metadata_v2(
                    sdc_planning=sdc_planning,
                    planning_mask=sdc_planning_mask,
                    current_velocity=np.asarray(sdc_vel, dtype=np.float64),
                    dt=0.5,
                )
            
            # 3. 将数据存入大字典
            if token is None or sdc_planning is None or sdc_planning_mask is None:
                continue
            else:
                gt_trajs_dict[token] = sdc_planning.astype(np.float32)
                gt_trajs_mask_dict[token] = sdc_planning_mask.astype(np.bool_)
                # print(f"Token: {token} | SDC Planning Shape: {gt_trajs_dict[token].shape} | Mask Shape: {gt_trajs_mask_dict[token].shape}")

        # --- 4. (循环结束后) 创建目录并保存文件 ---
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        # 保存 gt_traj.pkl
        gt_traj_path = os.path.join(output_dir, "gt_v2x_traj.pkl")
        with open(gt_traj_path, 'wb') as f:
            pickle.dump(gt_trajs_dict, f)
        print(f"Successfully saved {len(gt_trajs_dict)} SDC trajectories to {gt_traj_path}")

        # 保存 gt_traj_mask.pkl
        gt_traj_mask_path = os.path.join(output_dir, "gt_v2x_traj_mask.pkl")
        with open(gt_traj_mask_path, 'wb') as f:
            pickle.dump(gt_trajs_mask_dict, f)
        print(f"Successfully saved {len(gt_trajs_mask_dict)} SDC trajectory masks to {gt_traj_mask_path}")

    def process_json_file(self, data_list: List[Dict], json_file: str) -> List[Dict]:
        """处理 train_v2x.json 文件，并将 Q&A pairs 添加到匹配的 data_list 样本中。"""
        print(f"Processing JSON file: {json_file}")

        with open(json_file, 'r') as json_file:
            json_data = json.load(json_file)

        qa_pairs_by_images = {}
        for item_group in json_data: # item_group 是 json 文件中的一个样本
            
            # 提取 conversations
            qa_info = []
            if 'conversations' in item_group:
                try:
                    # human/question 是 item_group['conversations'][0]['value']
                    question = item_group['conversations'][0]['value'] 
                    # gpt/answer 是 item_group['conversations'][1]['value']
                    answer = item_group['conversations'][1]['value']
                    qa_info.append(question)
                    qa_info.append(answer)
                except (IndexError, KeyError):
                    continue
            
            # 提取图像路径 (vehicle-side 和 infrastructure-side)
            image_paths = item_group.get('image', [])
            if len(image_paths) == 2 and qa_info:
                veh_path = image_paths[0]
                inf_path = image_paths[1]
                key = (veh_path, inf_path)
                
                if key not in qa_pairs_by_images:
                    qa_pairs_by_images[key] = []
                qa_pairs_by_images[key].append(qa_info)


        # 3. 过滤 data_list 并附加处理后的 Q&A 数据
        filtered_data_list = []
        
        for info in data_list:
    
            veh_img_path_rel = info['images'].get('VEHICLE_CAM_FRONT', {}).get('img_path', '')
            
            inf_img_path_rel = ""
            other_agent_key = 'model_other_agent_inf'
            if other_agent_key in info.get('other_agent_info_dict', {}):
                inf_img_path_rel = info['other_agent_info_dict'][other_agent_key]['images'].get('INF_CAM_FRONT', {}).get('img_path', '')
                
            # 提取 JSON 的 key (相对路径)
            matching_key = (veh_img_path_rel, inf_img_path_rel)
            
            if matching_key in qa_pairs_by_images:
                # 将 Q&A 对添加到 info，并为每个 Q&A 对创建新样本
                for qa_pair in qa_pairs_by_images[matching_key]:
                    new_info = copy.deepcopy(info) # 深度复制，防止修改原数据结构
                    
                    # 存储 Q&A 对
                    new_info['QA_pairs'] = qa_pair  
                    new_info['input_images']=[]
                    # # 存储所有图像路径 (与您的 prepare_train_data 逻辑匹配)
                    # new_info['img_filename'] = [veh_img_path_rel] ####只存储车辆侧图像路径
                    # print("new_info['img_filename']:",new_info['img_filename'])
                    
                    filtered_data_list.append(new_info) 

        assert filtered_data_list, "filtered_data_list is empty! Check if JSON file paths match PKL file paths."
        return filtered_data_list
    
    def prepare_data(self, index: int) -> Union[dict, None]:
        """Data preparation for both training and testing stage.

        Called by `__getitem__`  of dataset.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict or None: Data dict of the corresponding index.

        """
        # print(f"Preparing data for index: {index}")
        ori_input_dict = self.get_data_info(index)

        # deepcopy here to avoid inplace modification in pipeline.
        input_dict = copy.deepcopy(ori_input_dict)

        # box_type_3d (str): 3D box type.
        input_dict['box_type_3d'] = self.box_type_3d
        # box_mode_3d (str): 3D box mode.
        input_dict['box_mode_3d'] = self.box_mode_3d

        input_dict['conversations']=["",""]

        
        
        
        # pre-pipline return None to random another in `__getitem__`
        if not self.test_mode and self.filter_empty_gt:
            if len(input_dict['ann_info']['gt_labels_3d']) == 0:
                return None
      


        # prompt = generate_full_prompt(input_dict) 

        example = self.pipeline(input_dict)
        return example

# 原本正常数据处理 parse_ann_info会被注视掉
    # def parse_ann_info(self, info: dict) -> dict:
    #     """Parse ann_info for nuScenes dataset.

    #     Args:
    #         info (dict): Data information of single sample.

    #     """
    #     # print("Parsing annotation info for sample token:", info['token'])
    #     if self.use_valid_flag:
    #         mask = info['valid_flag']
    #     else:
    #         mask = info['num_lidar_pts'] > 0

    #     ann_info = super().parse_ann_info(info)
    #     gt_velocity = info['gt_velocity'][mask]
    #     prev_token = info['prev']
    #     sdc_vel = self.traj_api.sdc_vel_info[info['token']]
    #     if prev_token != '' and prev_token in self.traj_api.sdc_vel_info:
    #         past_vel = self.traj_api.sdc_vel_info[prev_token][:2]
    #     else:
    #         # 场景第一帧，无法计算历史加速度，给一个默认值(和当前一样)
    #         past_vel = sdc_vel[:2]
    #     gt_sdc_bbox, gt_sdc_label = self.traj_api.generate_sdc_info(sdc_vel)
    #     gt_sdc_past_traj, gt_sdc_past_traj_mask,sdc_planning, sdc_planning_mask, command = self.traj_api.get_sdc_planning_label(
    #         info['token'])
        
    #     if command == -1:
    #         command_vlm = np.array([-1])
    #     else:
    #         command_vlm is now produced by generate_planning_metadata_v2.
        

    #     sdc_planning, sdc_planning_mask, command = self.traj_api.get_sdc_planning_label(
    #         info['token']) #生成规划指令
        
    #     ann_info = dict(
    #             gt_sdc_bbox=gt_sdc_bbox,
    #             gt_sdc_label=gt_sdc_label,
    #             sdc_planning=sdc_planning,
    #             sdc_planning_mask=sdc_planning_mask,
    #             command_vlm = command_vlm,
    #             command=command,
    #             gt_sdc_past_traj = gt_sdc_past_traj,
    #             gt_sdc_past_traj_mask = gt_sdc_past_traj_mask,
    #             past_vel = past_vel,
    #             gt_velocity = gt_velocity,
    #         )
    #     return ann_info


    def parse_data_info(self, data_info: dict) -> Union[dict, List[dict]]:
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.
        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
                - qa_info (dict): qainfo.
        """

        # if self.inference_wo_label:
        #     return self.get_data_info_wo_label(index, agent_name)

        data_info = super().parse_data_info(data_info)
        ##地图信息
        location = self.nusc.get('log', self.nusc.get(
            'scene', data_info['scene_token'])['log_token'])['location']
        
        # graph_nodes = self.vector_map.vectors_to_graph(
        #                                                 location,
        #                                                 data_info['ego2global_translation'],
        #                                                 data_info['ego2global_rotation']
                                                    # )

        l2e_r = data_info['lidar2ego_rotation'] ##
        l2e_t = data_info['lidar2ego_translation'] ##
        e2g_r = data_info['ego2global_rotation'] ##
        e2g_t = data_info['ego2global_translation']##
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        l2g_r_mat = l2e_r_mat.T @ e2g_r_mat.T
        l2g_t = l2e_t @ e2g_r_mat.T + e2g_t
        
        data_info.update(
            dict(
                l2g_r_mat=l2g_r_mat.astype(np.float32),
                l2g_t=l2g_t.astype(np.float32)))
        veh2inf_r = None
        veh2inf_t = None
        if 'VehLidar2InfLidar_rotation' in data_info:            
            veh2inf_r = data_info['VehLidar2InfLidar_rotation']
            
        if 'VehLidar2InfLidar_translation' in data_info:
            veh2inf_t = data_info['VehLidar2InfLidar_translation']

        elif 'other_agent_info_dict' in data_info and \
             'model_other_agent_inf' in data_info['other_agent_info_dict']:
            inf_info = data_info['other_agent_info_dict']['model_other_agent_inf']
            if 'VehLidar2InfLidar_rotation' in inf_info:
                veh2inf_r = inf_info['VehLidar2InfLidar_rotation']
                veh2inf_t = inf_info['VehLidar2InfLidar_translation']
        
        veh2inf_rt = np.eye(4)
        if veh2inf_r is not None and veh2inf_t is not None:
            veh2inf_rt[:3, :3] = veh2inf_r
            veh2inf_rt[:3, 3] = veh2inf_t
            veh2inf_rt = veh2inf_rt.T
            
        data_info.update(dict(veh2inf_rt=veh2inf_rt.astype(np.float32)))

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_data_info in data_info['images'].items():
                image_paths.append(cam_data_info['img_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_data_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_data_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_data_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)
            if 'other_agent_info_dict' in data_info and \
                'model_other_agent_inf' in data_info['other_agent_info_dict']:
                    
                    inf_info = data_info['other_agent_info_dict']['model_other_agent_inf']
                    
                    # 检查是否有 INF_CAM_FRONT
                    if 'images' in inf_info and 'INF_CAM_FRONT' in inf_info['images']:
                        inf_cam_data = inf_info['images']['INF_CAM_FRONT']
                        
                        # 1. 添加路径
                        image_paths.append(inf_cam_data['img_path'])
                        
                        # 2. 计算矩阵 (逻辑与自车完全一致，因为 .pkl 生成时已经转换好了 sensor2lidar)
                        # 注意：这里的 'lidar' 指的是路侧的 Virtual Lidar
                        lidar2cam_r = np.linalg.inv(inf_cam_data['sensor2lidar_rotation'])
                        lidar2cam_t = inf_cam_data['sensor2lidar_translation'] @ lidar2cam_r.T
                        lidar2cam_rt = np.eye(4)
                        lidar2cam_rt[:3, :3] = lidar2cam_r.T
                        lidar2cam_rt[3, :3] = -lidar2cam_t
                        
                        intrinsic = inf_cam_data['cam_intrinsic']
                        viewpad = np.eye(4)
                        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

                        lidar2img_rt = viewpad @ lidar2cam_rt.T @ veh2inf_rt.T
                        # 3. 追加到列表
                        lidar2img_rts.append(lidar2img_rt)
                        cam_intrinsics.append(viewpad)
                        lidar2cam_rts.append(lidar2cam_rt.T)
            
            data_info.update(
                dict(
                    # img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            if 'sdc_planning' in data_info['ann_info'].keys():
                data_info['sdc_planning'] = data_info['ann_info']['sdc_planning']
                data_info['sdc_planning_mask'] = data_info['ann_info']['sdc_planning_mask']

        rotation = Quaternion(data_info['ego2global_rotation'])
        translation = data_info['ego2global_translation']
        can_bus = data_info['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
    
        data_info.update(dict(can_bus=can_bus)) ### 
        
        
      
        return data_info
    
    def get_data_info(self, index):

        data_info = super().get_data_info(index)
        
        image_info = []
        
        if 'VEHICLE_CAM_FRONT' in data_info['images']:
            cam_dict = data_info['images']['VEHICLE_CAM_FRONT']
            if 'img_path' in cam_dict:
                image_info.append(cam_dict['img_path'])

            # --- 2. 提取路侧端图像路径 (Infrastructure Cams) ---
            other_agent_key = 'model_other_agent_inf'
            if other_agent_key in data_info.get('other_agent_info_dict', {}):
                inf_info = data_info['other_agent_info_dict'][other_agent_key]
            
        
                if 'INF_CAM_FRONT' in inf_info.get('images', {}): 
                    cam_dict = inf_info['images']['INF_CAM_FRONT']
                    if 'img_path' in cam_dict:
                        image_info.append(cam_dict['img_path'])

        data_info['img_filename'] = image_info

        # prompt = generate_full_prompt(data_info['ann_info'],data_info['sample_idx'],image_info) 
        # data_info['prompt_data'] = prompt
        
        return data_info
        

    
def generate_full_prompt(processed_data, token, can_bus, img_filename,
                         traj_only=False, output_contract='legacy'):

    # print("token",token)
    
    if processed_data['command'] == -1:
        print(f"Sample {token} skipped due to invalid trajectory (Blacklist).")
        return None
    
    user_message = generate_user_message(processed_data, can_bus)
    assitant_message = generate_assistant_message(
        processed_data, traj_only=traj_only, output_contract=output_contract)
    
    final_prompt_dict = {
        "token": str(token),
        "image": img_filename, # 图像路径列表
        "conversations": [
            {
                "from": "human",
                # 注意：user_message 已经包含了我们自定义的所有 Ego-States, Perception等信息
                "value": user_message 
            },
            {
                "from": "gpt",
                # assistant_message 包含了 Trajectory: [...]
                "value": assitant_message 
            }
        ]
    }
    # print("final_prompt_dict",final_prompt_dict)
    return final_prompt_dict

def generate_user_message(data_dict, perception_range=50.0, short=True):
    # perception_range 建议稍微调大一点 (例如 50m)，20m 对于高速场景太短了

    user_message  = f"\n"
    """
    Perception and Prediction Outputs:
        object_boxes: [N, 7]  -> (x, y, z, l, w, h, yaw) in Lidar Frame (X-Front, Y-Left)
        object_names: [N]
        object_velocity: [N, 2] -> (vx, vy)
        object_fut_mask: [N, 6]
    """
    object_boxes = data_dict['gt_bboxes_3d'].tensor
    object_names = data_dict['gt_names']
    object_velocity = data_dict['gt_velocity']

    user_message += f"Perception and Prediction:\n"
    num_objects = object_boxes.shape[0]

    count = 0
    for i in range(num_objects):
        if object_boxes[i, 0] < -10.0:
            continue
            
        # --- 修改 2: 距离过滤 ---
        # 过滤掉超出感知范围的物体 (X 或 Y 超过范围)
        if (np.abs(object_boxes[i, :2]) > perception_range).any(): 
            continue

        count += 1
        object_name = object_names[i]
        ox, oy = object_boxes[i, :2]
        vx, vy = object_velocity[i]
        
        # --- 修改 3: 增加自然语言方位描述 (对 LLM 非常重要) ---
        # 将坐标转换为相对方位，辅助模型理解
        rel_pos = ""
        if ox > 0: rel_pos += "Front"
        else:      rel_pos += "Rear"
        
        if oy > 1.5:   rel_pos += "-Left" # 这里的阈值可以根据车道宽度调整
        elif oy < -1.5: rel_pos += "-Right"
        else:           rel_pos += "-Center" # 正前方或正后方

        # Short 模式：只包含当前位置和终点
        user_message += f" - {object_name} at {rel_pos} ({ox.item():.1f}, {oy.item():.1f})m, vel ({vx.item():.1f}, {vy.item():.1f})m/s, "

    if count == 0:
        user_message += " - No relevant objects detected nearby.\n"
    
 ####改到这里
    """
    Ego-States:
        
    """

    sdc_bbox_data = data_dict['gt_sdc_bbox'].data.tensor.numpy()[0]
    vx = float(sdc_bbox_data[7])
    vy = float(sdc_bbox_data[8])
    ego_length = float(sdc_bbox_data[3])
    ego_width = float(sdc_bbox_data[4])
    dt = 0.5
    # print("vx_past, vy_past:", data_dict['past_vel'])
    vx_past = float(data_dict['past_vel'][0])
    vy_past = float(data_dict['past_vel'][1])
    
    # Calculate acceleration based on the difference between current and past velocity
    ax = (vx - vx_past) / dt
    ay = (vy - vy_past) / dt
   
    gt_sdc_past_traj = data_dict['gt_sdc_past_traj']
    # print("sdc_bbox_data:", sdc_bbox_data)
    # print("gt_sdc_past_traj:", gt_sdc_past_traj)
    # Check if there are at least two historical points to calculate acceleration
    user_message += f"Ego-States:\n"
    user_message += f" - Dimensions (L, W): ({ego_length:.2f}, {ego_width:.2f})\n" 
    user_message += f" - Velocity (vx,vy): ({vx:.2f},{vy:.2f})\n"
    # user_message += f" - Heading Angular Velocity (v_yaw): ({v_yaw:.2f})\n"
    user_message += f" - Acceleration (ax,ay): ({ax:.2f},{ay:.2f})\n"
    # user_message += f" - Current Position of the Ego Vehicle(in Global Coordinates)(x, y, z): ({pos_x:.2f}, {pos_y:.2f}, {pos_z:.2f})"
    # user_message += f" - Heading Speed: ({vhead:.2f})\n"

    
    xh1, yh1 = gt_sdc_past_traj[0][0]
    xh2, yh2 = gt_sdc_past_traj[0][1]
    xh3, yh3 = gt_sdc_past_traj[0][2]
    xh4, yh4 = gt_sdc_past_traj[0][3]
    user_message += f"Historical Trajectory (last 2 seconds):"
    user_message += f" [({xh1:.2f},{yh1:.2f}), ({xh2:.2f},{yh2:.2f}), ({xh3:.2f},{yh3:.2f}), ({xh4:.2f},{yh4:.2f})]\n"
        
    
    
    # print("Generated User Message:\n", user_message)

    return user_message

def generate_assistant_message(data_dict, traj_only=False, output_contract='legacy'):

    gt_sdc_fut_traj = data_dict['sdc_planning'][0]
    assitant_message = ""

    planning_metadata = data_dict['command_vlm']
    if not isinstance(planning_metadata, dict) or planning_metadata.get('schema_version') != '2.0':
        raise ValueError('command_vlm must contain planning metadata schema 2.0')
    assitant_message += f"Lateral Action: {planning_metadata['lateral_action']}\n"
    assitant_message += f"Longitudinal Action: {planning_metadata['longitudinal_action']}\n"
    if output_contract == 'legacy':
        assitant_message += f"Target End Speed: {planning_metadata['target_end_speed_mps']:.2f} m/s\n"
        assitant_message += f"Mean Acceleration: {planning_metadata['mean_acceleration_mps2']:.2f} m/s^2\n"
    elif output_contract != 'physics':
        raise ValueError(f'unknown planning output contract: {output_contract}')
    trajectory_points = []
    # print("data_dict['gt_sdc_fut_traj']:", data_dict['gt_sdc_fut_traj'])

    for t in range(gt_sdc_fut_traj.shape[0]):
        x, y = gt_sdc_fut_traj[t][:2]
        trajectory_points.append(f"({float(x):.2f},{float(y):.2f})") 
    
    # Format the final message
    trajectory_string = ", ".join(trajectory_points)

    if not traj_only:
        assitant_message += "Trajectory:\n"
    
    assitant_message += f"[{trajectory_string}]"

    # assitant_message += f"[ {x1:.2f},{x2:.2f},{x3:.2f},{x4:.2f},{x5:.2f},{x6:.2f},{y1:.2f},{y2:.2f},{y3:.2f},{y4:.2f},{y5:.2f},{y6:.2f} ]"
    return assitant_message
