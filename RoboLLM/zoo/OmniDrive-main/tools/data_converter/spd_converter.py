# Copyright (c) OpenMMLab. All rights reserved.
import os
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from tools.dataset_converter.calib_i2v import trans_lidar_i2v
from mmdet3d.core.bbox import points_cam2img
from mmdet3d.datasets import NuScenesDataset
from scipy.linalg import polar
import json
from tqdm import tqdm
import uuid

def iterative_closest_point(A, num_iterations=100):
    R = A.copy()

    for _ in range(num_iterations):
        U, _ = polar(R)
        R = U

    return R
# -----------------------------------------------------------
# 新增辅助函数：用于 2D 投影 (复刻 OmniDrive 逻辑)
# -----------------------------------------------------------
def post_process_coords(corner_coords, imsize=(1920, 1080)):
    # 注意：请根据 SPD 数据集的实际图片分辨率修改 imsize
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])
        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])
        return min_x, min_y, max_x, max_y
    else:
        return None

def view_points(points, view, normalize: bool):
    # 简单的投影函数: 3D -> 2D
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
    return points

CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

nus_attributes = ('', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

class Box3D():
    def __init__(self):
        self.center = None
        self.wlh = None
        self.orientation_yaw_pitch_roll = None
        self.name = None
        self.token = None
        self.instance_token = None
        self.track_id = None
        self.prev_token = None
        self.next_token = None
        self.timestamp = None
        self.visibility = None
        self.gt_velocity = None
        self.prev = None
        self.next = None


def get_cam_intr(calib_path):
    try:
        intr = np.array(load_json(calib_path)['P']).reshape(3, 4)[:, :3]
    except:
        intr = np.array(load_json(calib_path)['cam_K']).reshape(3, 3)

    return intr

def get_box_corners(center, wlh, yaw):
    """
    根据中心、尺寸和偏航角计算3D框的8个角点 (Lidar/Ego 坐标系)
    Args:
        center: [x, y, z]
        wlh: [w, l, h] (NuScenes/MMDet3D 格式通常对应 dy, dx, dz)
        yaw: float (yaw angle, radians)
    Returns:
        corners: (3, 8) numpy array
    """
    w, l, h = wlh
    
    # 建立以原点为中心的角点 (x: front/back, y: left/right, z: up/down)
    # 这里的定义参考 NuScenes/OpenMMLab 的坐标系定义: x前, y左, z上
    x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
    z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    # 旋转 (绕Z轴)
    c = np.cos(yaw)
    s = np.sin(yaw)
    rot_mat = np.array([[c, -s, 0],
                        [s,  c, 0],
                        [0,  0, 1]])
    corners = np.dot(rot_mat, corners)

    # 平移
    corners[0, :] += center[0]
    corners[1, :] += center[1]
    corners[2, :] += center[2]
    
    return corners

def mul_matrix(rotation_1, translation_1, rotation_2, translation_2):
    rotation_1 = np.matrix(rotation_1)
    translation_1 = np.matrix(translation_1).reshape(3, 1)
    rotation_2 = np.matrix(rotation_2)
    translation_2 = np.matrix(translation_2).reshape(3, 1)

    rotation = rotation_2 * rotation_1
    translation = rotation_2 * translation_1 + translation_2
    rotation = np.array(rotation)
    translation = np.array(translation).reshape(3)

    return rotation, translation


visibility_mappings = {
    0: 4,
    1: 3,
    2: 2,
    3: 1
}


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

def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)

    return data

def gen_token(*args):
    token_name = ''
    for value in args:
        token_name += str(value)
    token = uuid.uuid3(uuid.NAMESPACE_DNS, token_name)
    return str(token)

def create_spd_infos_coop(root_path,
                          out_path,
                          v2x_side,
                          split_path,
                          can_bus_root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10,
                          split_part='train',
                          flag_save=True,
                          use_valid_flag=True):
    """Create info file of spd dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'vehicle-side'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    out_path = osp.join(out_path, v2x_side)

    ## Step 0: load neccesary data
    coop_data_info_path = osp.join(root_path, 'cooperative/data_info.json')
    coop_data_infos = load_json(coop_data_info_path)

    veh_data_info_path = osp.join(root_path, 'vehicle-side/data_info.json')
    veh_data_infos = load_json(veh_data_info_path)

    inf_data_info_path = osp.join(root_path, 'infrastructure-side/data_info.json')
    inf_data_infos = load_json(inf_data_info_path)

    split_data_path = split_path    
    split_data = load_json(split_data_path)

    train_scenes = split_data['batch_split']['train']
    val_scenes = split_data['batch_split']['val']

    train_spd_infos = []
    val_spd_infos = []
    spd_infos = []
    ## Generate  sample_info_mappings, secene_frame_mappings, total_annotations, instance_token_mappings
    sample_infos, sample_info_mappings = _generate_sample_infos_coop(coop_data_infos,veh_data_infos,inf_data_infos)
    secene_frame_mappings = _get_secene_frame_mappings(sample_info_mappings)
    total_annotations = _get_total_annotations_coop(root_path,coop_data_infos,sample_info_mappings)
    instance_token_mappings = _get_instance_token_mappings(total_annotations, sample_info_mappings)

    # get lidar2ego info
    lidar_ego_global_infos = get_lidar_ego_global_infos(osp.join(root_path,'vehicle-side'), veh_data_infos, v2x_side)
    
    ## interpolate boxes for unvisible objects
    total_annotations =  _generate_unvisible_annotations("cooperative",sample_info_mappings,secene_frame_mappings,instance_token_mappings,total_annotations,lidar_ego_global_infos)

    ## update instance_token_mappings
    instance_token_mappings = _get_instance_token_mappings(total_annotations, sample_info_mappings)

    ## add velocity and prev/next, update total_annotations and instance_token_mappings
    total_annotations, instance_token_mappings = _add_annotation_velocity_prev_next(total_annotations, instance_token_mappings, lidar_ego_global_infos)

    for coop_data_info in tqdm(coop_data_infos):
        ## Step 1: build basic information
        veh_frame_id = coop_data_info['vehicle_frame']
        inf_frame_id = coop_data_info['infrastructure_frame']
        
        sample_token = veh_frame_id
        sample_info = sample_info_mappings[sample_token]

        assert veh_frame_id == sample_info['token']
        assert inf_frame_id == sample_info['token_inf']
        ##自车端
        info = {
            'token': sample_info['token'],
            'frame_idx': sample_info['frame_idx'],
            'scene_token': sample_info['scene_token'],
            'location': sample_info['location'],
            'timestamp': sample_info['timestamp'],
            'prev': sample_info['prev'], #上一帧的 token
            'next': sample_info['next'], # 下一帧的 token
        }
        ##基础设施端关键信息
        other_agent_info = {
            'token': sample_info['token_inf'],
            'frame_idx': sample_info['frame_idx'],
            'scene_token': sample_info['scene_token'],
            'location': sample_info['location'],
            'timestamp': sample_info['timestamp_inf'],
            'prev': sample_info_mappings[sample_info['prev']]['token_inf'] if sample_info['prev'] else '',
            'next': sample_info_mappings[sample_info['next']]['token_inf'] if sample_info['next'] else '',
            'system_error_offset':sample_info['system_error_offset']
        }

        ## Step 2: build camera sensor infos

        # 2.1 准备自车 Lidar 位姿 (OmniDrive 标准)
        ego_vehicle_data_info = get_single_sample_info(veh_frame_id, veh_data_infos)

        info['lidar2ego_rotation'] = lidar_ego_global_infos[sample_token]['lidar2ego_rotation']
        info['lidar2ego_translation'] = lidar_ego_global_infos[sample_token]['lidar2ego_translation']
        info['ego2global_rotation'] = lidar_ego_global_infos[sample_token]['ego2global_rotation']
        info['ego2global_translation'] = lidar_ego_global_infos[sample_token]['ego2global_translation']
        info['pts_filename'] = os.path.join('vehicle-side', ego_vehicle_data_info['pointcloud_path'].replace('pcd','bin'))
        
        info['cams'] = {} # 初始化 cams 字典
        cam_key = 'VEHICLE_CAM_FRONT'  
        cam_info_v = {}
        cam_info_v['data_path'] = os.path.join('vehicle-side', ego_vehicle_data_info['image_path'])
        
        # 读取外参 (Lidar_V -> Cam_V)
        calib_lidar2cam_v = load_json(osp.join(root_path, 'vehicle-side', ego_vehicle_data_info['calib_lidar_to_camera_path']))
        
        # 计算 sensor2lidar (Cam_V -> Lidar_V)
        # 注意：OmniDrive/NuScenes 要求 sensor2lidar 是 T_sensor_to_lidar
        # 我们的 calib_lidar2cam 通常是 Lidar -> Cam，所以需要求逆
        l2c_r_v = np.array(calib_lidar2cam_v['rotation'])
        l2c_t_v = np.array(calib_lidar2cam_v['translation'])
        
        # Inverse: Cam -> Lidar
        c2l_r_v = np.linalg.inv(l2c_r_v)
        c2l_t_v = - np.dot(c2l_r_v, l2c_t_v)
        
        cam_info_v['sensor2lidar_rotation'] = c2l_r_v
        cam_info_v['sensor2lidar_translation'] = c2l_t_v
        
        # 读取内参
        cam_info_v['cam_intrinsic'] = get_cam_intr(osp.join(root_path, 'vehicle-side', ego_vehicle_data_info['calib_camera_intrinsic_path']))
        
        # 存入 info
        info['cams'][cam_key] = cam_info_v


        # Step 2.2: build roadside sensor info
        # build lidar info
        cam_key_i = 'INF_CAM_FRONT'
        inf_data_info = get_single_sample_info(inf_frame_id, inf_data_infos)
        
        cam_info_i = {}
        cam_info_i['data_path'] = os.path.join('infrastructure-side', inf_data_info['image_path'])

        sys_offset = sample_info['system_error_offset']
        # 路径准备
        path_inf_l2w = osp.join(root_path, 'infrastructure-side', inf_data_info['calib_virtuallidar_to_world_path'])
        path_veh_l2n = osp.join(root_path, 'vehicle-side', ego_vehicle_data_info['calib_lidar_to_novatel_path'])
        path_veh_n2w = osp.join(root_path, 'vehicle-side', ego_vehicle_data_info['calib_novatel_to_world_path'])
        # r_i2v, t_i2v 即 R_inf_to_veh, T_inf_to_veh
        r_i2v, t_i2v = trans_lidar_i2v(path_inf_l2w, path_veh_l2n, path_veh_n2w, sys_offset)

        # 2. 获取 Cam_I -> Lidar_I (路侧相机外参)
        calib_l2c_i = load_json(osp.join(root_path, 'infrastructure-side', inf_data_info['calib_virtuallidar_to_camera_path']))
        l2c_r_i = np.array(calib_l2c_i['rotation'])
        l2c_t_i = np.array(calib_l2c_i['translation'])

        # 求逆得到 Cam -> Lidar
        c2l_r_i = np.linalg.inv(l2c_r_i)
        c2l_t_i = - np.dot(c2l_r_i, l2c_t_i)

        # 3. 链式变换: Cam_I -> Lidar_V
        # 公式: T_total = T_i2v * T_c2l
        # P_veh = R_i2v * (R_c2l * P_cam + T_c2l) + T_i2v
        #       = (R_i2v * R_c2l) * P_cam + (R_i2v * T_c2l + T_i2v)

        c2v_r = np.dot(r_i2v, c2l_r_i) # 旋转矩阵相乘
        c2v_t = np.dot(r_i2v, c2l_t_i.reshape(3,1)).flatten() + t_i2v.flatten() # 平移向量变换

        # 4. 存入 OmniDrive 格式
        cam_info_i['sensor2lidar_rotation'] = c2v_r
        cam_info_i['sensor2lidar_translation'] = c2v_t
        cam_info_i['cam_intrinsic'] = get_cam_intr(osp.join(root_path, 'infrastructure-side', inf_data_info['calib_camera_intrinsic_path']))

        info['cams'][cam_key_i] = cam_info_i

        # UniV2X TODO: complete this part
        info['sweeps'] = {} ############ 这个部分待补充 ##########
        info['can_bus'] = np.zeros(13) ############ 这个部分待补充 ##########
        other_agent_info['sweeps'] = {} 
        other_agent_info['can_bus'] = np.zeros(18)

        ##计算自车速度
        timestamp_T_ns = info['timestamp']

        # build egolidar2lidar info
        veh_l2e_r = np.array(Quaternion(info['lidar2ego_rotation']).rotation_matrix)
        veh_l2e_t = np.array(info['lidar2ego_translation']).reshape(3)
        veh_e2g_r = np.array(Quaternion(info['ego2global_rotation']).rotation_matrix)
        veh_e2g_t = np.array(info['ego2global_translation']).reshape(3)

        
        # 预先计算T帧的 逆矩阵转置 (Global -> Ego -> Lidar)
        e2g_r_T_inv_T = np.linalg.inv(veh_e2g_r).T
        l2e_r_T_inv_T = np.linalg.inv(veh_l2e_r).T
        
        # --- A. 计算 sdc_vel (即 T -> T+1 的速度, 转换到 T 的Lidar坐标系) ---
        sdc_vel_lidar = np.array([0.0, 0.0], dtype=np.float32)
        # 使用 info['next']
        next_sample_token_T_plus_1 = info['next']

        if next_sample_token_T_plus_1:
            # 获取下一帧(T+1)的信息 (需要访问全局字典)
            timestamp_T_plus_1_ns = sample_info_mappings[next_sample_token_T_plus_1]['timestamp']
            veh_e2g_t_plus_1 = lidar_ego_global_infos[next_sample_token_T_plus_1]['ego2global_translation']
            
            # 1. 计算全局速度 (T -> T+1)
            dt_sec = (timestamp_T_plus_1_ns - timestamp_T_ns) / 1e6
            if dt_sec > 1e-3:
                vel_global = (veh_e2g_t_plus_1 - veh_e2g_t) / dt_sec
                
                # 2. 转换到当前帧(T)的Lidar坐标系
                vel_ego_T = vel_global @ e2g_r_T_inv_T
                vel_lidar_T = vel_ego_T @ l2e_r_T_inv_T
                sdc_vel_lidar = vel_lidar_T[:2].astype(np.float32)

        # --- B. 计算 past_vel (即 T-1 -> T 的速度, 转换到 T-1 的Lidar坐标系) ---
        past_vel_lidar = np.array([0.0, 0.0], dtype=np.float32)
        # 使用 info['prev']
        prev_sample_token_T_minus_1 = info['prev']

        if prev_sample_token_T_minus_1:
            # 获取上一帧(T-1)的信息 (需要访问全局字典)
            timestamp_T_minus_1_ns = sample_info_mappings[prev_sample_token_T_minus_1]['timestamp']
            veh_e2g_t_minus_1_global = lidar_ego_global_infos[prev_sample_token_T_minus_1]['ego2global_translation']

            # 获取上一帧(T-1)的变换矩阵 (需要访问全局字典)
            l2e_r_T_minus_1_mat = Quaternion(lidar_ego_global_infos[prev_sample_token_T_minus_1]['lidar2ego_rotation']).rotation_matrix
            e2g_r_T_minus_1_mat = Quaternion(lidar_ego_global_infos[prev_sample_token_T_minus_1]['ego2global_rotation']).rotation_matrix
            
            # 预先计算T-1帧的 逆矩阵转置 (Global -> Ego -> Lidar)
            e2g_r_T_minus_1_inv_T = np.linalg.inv(e2g_r_T_minus_1_mat).T
            l2e_r_T_minus_1_inv_T = np.linalg.inv(l2e_r_T_minus_1_mat).T

            # 1. 计算全局速度 (T-1 -> T)
            dt_past_sec = (timestamp_T_ns - timestamp_T_minus_1_ns) / 1e6
            if dt_past_sec > 1e-3:
                vel_global_past = (veh_e2g_t - veh_e2g_t_minus_1_global) / dt_past_sec

                # 2. 转换到上一帧(T-1)的Lidar坐标系
                vel_ego_T_minus_1 = vel_global_past @ e2g_r_T_minus_1_inv_T
                vel_lidar_T_minus_1 = vel_ego_T_minus_1 @ l2e_r_T_minus_1_inv_T
                past_vel_lidar = vel_lidar_T_minus_1[:2].astype(np.float32)

        # 2. 生成 SDC BBox 和 Label (使用 sdc_vel_lidar)
        #
        psudo_sdc_bbox = np.array([0.0, 0.0, 0.0, 1.73, 4.08, 1.56, -np.pi]) 
        
        # 将 sdc_vel_lidar (T -> T+1 的速度) 拼接
        gt_sdc_bbox_np = np.concatenate([psudo_sdc_bbox, sdc_vel_lidar], axis=-1).astype(np.float32)
        gt_sdc_names_3d = ['car']
        gt_sdc_labels_3d = []
        for cat in gt_sdc_names_3d:
            if cat in CLASSES:
                gt_sdc_labels_3d.append(CLASSES.index(cat))
            else:
                gt_sdc_labels_3d.append(-1)
        gt_sdc_labels_3d = np.array(gt_sdc_labels_3d)
        
        # 3. 将新生成的信息存入 info 字典
        info['gt_sdc_bbox'] = gt_sdc_bbox_np  # (9,) array, 包含 T -> T+1 的速度
        info['gt_sdc_label'] = gt_sdc_labels_3d # (1,) array
        info['past_vel'] = past_vel_lidar      # (2,) array, 包含 T-1 -> T 的速度
        info['sdc_velocity'] = sdc_vel_lidar  # (2,) array, 包含 T -> T+1 的速度
        # print("SDC Velocity (Lidar Coord) - Past Vel: {}, SDC Vel: {}".format(past_vel_lidar, sdc_vel_lidar))
        ## Step 3: build annotation information
        annotations = total_annotations[sample_token]

        boxes = []
        for anno_token in annotations.keys():
            annotation = annotations[anno_token]
            box3d = Box3D()
            box3d.center = [annotation['3d_location']['x'], annotation['3d_location']['y'],
                                            annotation['3d_location']['z']]
            box3d.wlh = [annotation['3d_dimensions']['w'], annotation['3d_dimensions']['l'],
                                            annotation['3d_dimensions']['h']]
            box3d.orientation_yaw_pitch_roll = annotation['rotation']
            box3d.name = annotation['type']
            box3d.token = annotation['token']
            box3d.instance_token = annotation['instance_token']
            box3d.track_id = int(annotation['track_id'])
            box3d.timestamp = float(sample_info['timestamp'])
            box3d.visibility = visibility_mappings[annotation['occluded_state']]
            box3d.gt_velocity = annotation['gt_velocity']
            box3d.prev = annotation['prev']
            box3d.next = annotation['next']

            boxes.append(box3d)

        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation_yaw_pitch_roll
                            for b in boxes]).reshape(-1, 1)
        
        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
        names = np.array([b.name for b in boxes])
        
        instance_tokens = np.array([b.instance_token for b in boxes])
        instance_inds = np.array([b.track_id for b in boxes])
        box_tokens = np.array([b.token for b in boxes])
        timestamps = np.array([b.timestamp for b in boxes])
        visibility_tokens = np.array([b.visibility for b in boxes])
        gt_velocity = np.array([b.gt_velocity for b in boxes])
        prev_anno_tokens = np.array([b.prev for b in boxes])
        next_anno_tokens = np.array([b.next for b in boxes])

        # TODO: complete this part
        valid_flag = np.array([True for b in boxes])
        num_lidar_pts = np.array([1 for b in boxes]) ############ 这个部和omni的不一样 ##########
        gt_labels_3d = []
        for cat in names:
            if cat in CLASSES: # 使用 CLASSES 列表
                gt_labels_3d.append(CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        ##处理instance信息 
        gt_2dbboxes_cams = []
        gt_3dbboxes_cams = []
        centers2d_cams = []
        gt_2dbboxes_ignore_cams = []
        gt_2dlabels_cams = []
        depths_cams = []
        visibilities = []

        # 遍历所有相机 (包括 'VEHICLE_CAM_FRONT' 和 'INF_CAM_FRONT')
        for cam_key, cam_info in info['cams'].items():
            
            # 1. 确定图像尺寸
            # 根据 SPD 数据集说明，自车和路侧图像分辨率可能不同
            if 'INF' in cam_key:
                width, height = 1920, 1080 # [Config] 请确认为 SPD 路侧真实分辨率
            else:
                width, height = 1920, 1080 # [Config] 请确认为 SPD 自车真实分辨率

            # 2. 获取 Veh-Lidar -> Camera 的变换矩阵
            # info['cams'] 中存储的是 sensor2lidar (Cam -> Veh-Lidar)
            # 我们需要其逆矩阵：Veh-Lidar -> Cam
            c2vl_r = cam_info['sensor2lidar_rotation'] 
            c2vl_t = cam_info['sensor2lidar_translation']
            
            # 求逆: R_inv = R^T, T_inv = -R^T * T
            vl2c_r = c2vl_r.T
            vl2c_t = -np.dot(vl2c_r, c2vl_t)
            
            camera_intrinsic = cam_info['cam_intrinsic']

            # 初始化当前相机的列表
            cur_gt_2dbboxes = []
            cur_gt_3dbboxes = [] 
            cur_centers2d = []
            cur_gt_2dlabels = []
            cur_depths = []
            cur_visibilities = []
            cur_gt_2dbboxes_ignore = []

            for i, box3d in enumerate(boxes):
                # 过滤掉未知类别
                if gt_labels_3d[i] == -1:
                    continue
                
                # --- A. 准备 3D 数据 (在 Vehicle-Lidar 坐标系) ---
                # gt_boxes[i]: [x, y, z, w, l, h, yaw]
                # 注意：这里的 gt_boxes 已经是处理好的 Lidar 坐标系下的框
                box_center_lidar = gt_boxes[i][:3]
                box_dims_lidar = gt_boxes[i][3:6] # w, l, h
                box_yaw_lidar = gt_boxes[i][6]

                # 获取 8 个角点 (Vehicle-Lidar 坐标系)
                corners_lidar = get_box_corners(box_center_lidar, box_dims_lidar, box_yaw_lidar) # Shape: (3, 8)

                # --- B. 坐标变换: Veh-Lidar -> Current Camera ---
                # 这一步对于路侧相机同样适用，因为 vl2c_r 已经包含了 InfraLidar->VehLidar 的逆变换
                corners_cam = np.dot(vl2c_r, corners_lidar) + vl2c_t.reshape(3, 1)

                # --- C. 深度过滤 ---
                # 检查是否在相机前方 (z > 0.1)
                # 策略：只要有一个角点在前方，就尝试保留；或者用中心点判断
                if np.all(corners_cam[2, :] <= 0.1):
                    continue
                
                # 计算 Camera 坐标系下的中心点
                center_cam = np.dot(vl2c_r, box_center_lidar) + vl2c_t
                depth = center_cam[2]
                
                # 如果中心点在背后太远，通常视为不可见
                if depth <= 0.1:
                    continue

                # --- D. 投影到图像 (3D -> 2D Pixel) ---
                # view_points 包含内参乘法和透视除法
                points_img = view_points(corners_cam, camera_intrinsic, normalize=True)[:2, :] # Shape: (2, 8)

                # --- E. 裁剪与筛选 (post_process_coords) ---
                points_img_list = points_img.T.tolist()
                bbox2d = post_process_coords(points_img_list, imsize=(width, height))

                if bbox2d is None:
                    continue
                
                min_x, min_y, max_x, max_y = bbox2d
                
                # 过滤过小的框
                w_2d = max_x - min_x
                h_2d = max_y - min_y
                if w_2d < 2 or h_2d < 2:
                    continue

                # --- F. 保存结果 ---
                # 1. 2D BBox
                cur_gt_2dbboxes.append([min_x, min_y, max_x, max_y])
                
                # 2. Labels
                cur_gt_2dlabels.append(gt_labels_3d[i])
                
                # 3. Centers2D (投影后的中心点像素坐标) & Depth
                center_img = view_points(center_cam.reshape(3, 1), camera_intrinsic, normalize=True).reshape(3)
                cur_centers2d.append(center_img[:2]) 
                cur_depths.append(depth)
                
                # 4. 3D BBox in Camera Coords (可选，用于 Mono3D)
                # 简单近似：将 Lidar 系的 yaw 转换到 Cam 系
                v_lidar_heading = np.array([np.cos(box_yaw_lidar), np.sin(box_yaw_lidar), 0])
                v_cam_heading = np.dot(vl2c_r, v_lidar_heading)
                yaw_cam = -np.arctan2(v_cam_heading[0], v_cam_heading[2]) 
                bbox3d_cam = np.concatenate([center_cam, box_dims_lidar, [yaw_cam]])
                cur_gt_3dbboxes.append(bbox3d_cam)

                # 5. Visibility (如果有)
                cur_visibilities.append(visibility_tokens[i])

            # 将当前相机的结果转为 numpy 并存入总列表
            gt_2dbboxes_cams.append(np.array(cur_gt_2dbboxes, dtype=np.float32))
            gt_2dlabels_cams.append(np.array(cur_gt_2dlabels, dtype=np.int64))
            centers2d_cams.append(np.array(cur_centers2d, dtype=np.float32))
            depths_cams.append(np.array(cur_depths, dtype=np.float32))
            gt_3dbboxes_cams.append(np.array(cur_gt_3dbboxes, dtype=np.float32))
            gt_2dbboxes_ignore_cams.append(np.array(cur_gt_2dbboxes_ignore, dtype=np.float32))
            visibilities.append(np.array(cur_visibilities))

        # 将生成的2D信息更新到 info 字典
        info.update(dict(
            bboxes2d=gt_2dbboxes_cams,
            bboxes3d_cams=gt_3dbboxes_cams,
            labels2d=gt_2dlabels_cams,
            centers2d=centers2d_cams,
            depths=depths_cams,
            bboxes_ignore=gt_2dbboxes_ignore_cams,
            visibilities=visibilities
        ))

        info['instances'] = []
        for i in range(len(boxes)):
            # 这里必须使用原始数组的元素，而不是 LiDARInstance3DBoxes 对象
            instance_dict = {
                # 核心 Annotation 信息 (Det3DDataset/NuScenesDataset 强依赖)
                'bbox_3d': gt_boxes[i].astype(np.float32),      # (7,) NumPy Array
                'bbox_label_3d': int(gt_labels_3d[i]),          # int
                'velocity': gt_velocity[i].astype(np.float32),  # (2,) NumPy Array (用于 with_velocity)
                # 其他 NuScenes 风格的/自定义信息
                'gt_names': names[i],
                'gt_inds': int(instance_inds[i]),
                'gt_ins_tokens': instance_tokens[i],
                'bbox_3d_isvalid': bool(valid_flag[i]),              # bool
                'num_lidar_pts': int(num_lidar_pts[i]),         # int
            }
            info['instances'].append(instance_dict)

        info['anno_tokens'] = box_tokens
        info['timestamps'] = timestamps
        info['visibility_tokens'] = visibility_tokens
        info['prev_anno_tokens'] = prev_anno_tokens
        info['next_anno_tokens'] = next_anno_tokens
        info['instance_tokens'] = instance_tokens
        
        
        ## Step X: save spd infos
        info['other_agent_info_dict']['model_other_agent_inf'] = other_agent_info

        # if len(train_spd_infos) == 0: # 仅打印第一个训练样本
        #     print("\n--- DEBUG: 第一个样本的结构 (info) ---")
            
        #     # 打印顶层所有键
        #     print(f"Keys in 'info' dictionary: {info.keys()}")
            
        #     # 打印 instances 列表的长度
        #     num_instances = len(info.get('instances', []))
        #     print(f"实例数量 (info['instances'] length): {num_instances}")
            
        #     # 打印第一个实例的详细结构 (如果存在)
        #     if num_instances > 0:
        #         first_instance = info['instances'][0]
        #         print(f"第一个实例 (info['instances'][0]) 的键: {first_instance.keys()}")

        #     print("-----------------------------------------------------")

        #     assert""

        if ego_vehicle_data_info['sequence_id'] in train_scenes:
            train_spd_infos.append(info)
        elif ego_vehicle_data_info['sequence_id'] in val_scenes:
            val_spd_infos.append(info)

        spd_infos.append(info)

        if flag_save:
            # metadata = dict(version=version)
            # data = dict(data_list=train_spd_infos,  metainfo=metadata)
            # info_path = osp.join(out_path,
            #                     '{}_infos_temporal_train_sdc.pkl'.format(info_prefix))
            # mm_dump(data, info_path)

            metadata = dict(version=version)
            data = dict(data_list=val_spd_infos,  metainfo=metadata)
            info_val_path = osp.join(out_path,
                                    '{}_infos_temporal_val_sdc.pkl'.format(info_prefix))
            mm_dump(data, info_val_path)
        

    return total_annotations, sample_info_mappings, spd_infos

def write_json(data, path):
    with open(path, mode="w") as f:
        json.dump(data, f, indent=2)

def get_single_sample_info(frame_id, data_infos):
    sample_info = {}
    for data in data_infos:
        if data['frame_id'] == frame_id:
            sample_info = data
            break
    return sample_info

def _generate_sample_infos_coop(coop_data_infos,veh_data_infos,inf_data_infos):
    """Get the prev and next sample token for a given `sample_data_token`.
    Args:
        data_infos (list): data_infos loaded from data_info.json file.
    Return:
        list[dict]: List of sample info
        dict: mapping sample token to sample info   
    """
    veh_sample_mappings = {}
    inf_sample_mappings = {}  
    coop_sample_mappings = {}  
    scene_data_dict = {}

    for data_info in coop_data_infos:
        veh_frame_id = data_info['vehicle_frame']
        inf_frame_id = data_info['infrastructure_frame']

        assert data_info['vehicle_sequence'] == data_info['infrastructure_sequence']

        veh_sample_mappings[veh_frame_id] = get_single_sample_info(veh_frame_id, veh_data_infos)
        inf_sample_mappings[inf_frame_id] = get_single_sample_info(inf_frame_id, inf_data_infos)
        coop_sample_mappings[veh_frame_id] = data_info

        scene_token = data_info['vehicle_sequence']
        if scene_token not in scene_data_dict.keys():
            scene_data_dict[scene_token] = []

        scene_data_dict[scene_token].append(veh_frame_id)

    sample_infos = []
    for scene_token in scene_data_dict.keys():
        scene_data_dict[scene_token].sort()

        for idx in range(len(scene_data_dict[scene_token])):
            info = {}
            if idx == 0:
                info['prev'] = ''
            else:
                info['prev'] = scene_data_dict[scene_token][idx - 1]
            
            if idx == len(scene_data_dict[scene_token]) - 1:
                info['next'] = ''
            else:
                info['next'] = scene_data_dict[scene_token][idx + 1]
            
            veh_frame_id = scene_data_dict[scene_token][idx]
            inf_frame_id = coop_sample_mappings[veh_frame_id]['infrastructure_frame']

            veh_sample_info = veh_sample_mappings[veh_frame_id]
            inf_sample_info = inf_sample_mappings[inf_frame_id]
            coop_sample_info = coop_sample_mappings[veh_frame_id]

            info['token'] = veh_sample_info['frame_id']
            info['timestamp'] = float(veh_sample_info['pointcloud_timestamp'])
            info['image_timestamp'] = float(veh_sample_info['image_timestamp'])
            info['scene_token'] = veh_sample_info['sequence_id']
            info['location'] = veh_sample_info['intersection_loc']
            info['frame_idx'] = idx

            info['token_inf'] = inf_sample_info['frame_id']
            info['timestamp_inf'] = float(inf_sample_info['pointcloud_timestamp'])
            info['image_timestamp_inf'] = float(inf_sample_info['image_timestamp'])

            info['system_error_offset'] = coop_sample_info['system_error_offset']

            sample_infos.append(info)
    
    sample_info_mappings = {}
    for sample_info in sample_infos:
        sample_token = sample_info['token']
        sample_info_mappings[sample_token] = sample_info
    
    return sample_infos, sample_info_mappings

def _generate_sample_infos(data_infos):
    """Get the prev and next sample token for a given `sample_data_token`.
    Args:
        data_infos (list): data_infos loaded from data_info.json file.
    Return:
        list[dict]: List of sample info
        dict: mapping sample token to sample info   
    """
    sample_mappings = {}
    scene_data_dict = {}
    for data_info in data_infos:
        sample_token = data_info['frame_id']
        sample_mappings[sample_token] = data_info

        scene_token = data_info['sequence_id']
        if scene_token not in scene_data_dict.keys():
            scene_data_dict[scene_token] = []

        scene_data_dict[scene_token].append(sample_token)

    sample_infos = []
    for scene_token in scene_data_dict.keys():
        scene_data_dict[scene_token].sort()

        for idx in range(len(scene_data_dict[scene_token])):
            info = {}
            if idx == 0:
                info['prev'] = ''
            else:
                info['prev'] = scene_data_dict[scene_token][idx - 1]

            if idx == len(scene_data_dict[scene_token]) - 1:
                info['next'] = ''
            else:
                info['next'] = scene_data_dict[scene_token][idx + 1]

            sample_token = scene_data_dict[scene_token][idx]
            sample_info = sample_mappings[sample_token]
            info['token'] = sample_info['frame_id']
            info['timestamp'] = float(sample_info['pointcloud_timestamp'])
            info['image_timestamp'] = float(sample_info['image_timestamp'])
            info['scene_token'] = sample_info['sequence_id']
            info['location'] = sample_info['intersection_loc']
            info['frame_idx'] = idx

            sample_infos.append(info)

    sample_info_mappings = {}
    for sample_info in sample_infos:
        sample_token = sample_info['token']
        sample_info_mappings[sample_token] = sample_info

    return sample_infos, sample_info_mappings

def _get_total_annotations_coop(root_path,data_infos,sample_info_mappings):
    total_annotations = {}
    for data_info in data_infos:
        sample_token = data_info['vehicle_frame']
        scene_token = sample_info_mappings[sample_token]['scene_token']

        annotation_path = osp.join(root_path, 'cooperative/label', sample_token+'.json')
        annotations = load_json(annotation_path)
        total_annotations[sample_token] = {}
        for annotation in annotations:
            anno_token = annotation['token']
            annotation["instance_token"] = gen_token(annotation['track_id'],scene_token)
            annotation['type'] = class_names_nuscenes_mappings[annotation['type']]
            total_annotations[sample_token][anno_token] = annotation
    return total_annotations



 # 为完全被遮挡的物体生成轨迹和标注
def _generate_unvisible_annotations(source_name, sample_info_mappings, secene_frame_mappings, instance_token_mappings,
                                    total_annotations,lidar_ego_global_infos):
    """Generate annotations for totally occluded objects and make trajectory complete.
    Args:
        root_path: data root
        data_infos: (list): data_infos loaded from data_info.json file.
    Return:
        dict[dict]: {'sample_token': {'anno_token': }}
    """
    ## Interpolate box   
    for instance_token in instance_token_mappings.keys():
        cur_instance_samples = instance_token_mappings[instance_token]
        cur_scene_token = cur_instance_samples[0]['scene_token']
        cur_scene_token_end = cur_instance_samples[-1]['scene_token']
        assert cur_scene_token == cur_scene_token_end

        for ii in range(len(cur_instance_samples) - 1):
            cur_frame_idx = cur_instance_samples[ii]['frame_idx'] + 1
            while cur_frame_idx != cur_instance_samples[ii + 1]['frame_idx']:
                # linear interpolation
                loc_ii_0 = cur_instance_samples[ii]['annotation']['3d_location']
                loc_ii_1 = cur_instance_samples[ii + 1]['annotation']['3d_location']
                rot_ii_0 = cur_instance_samples[ii]['annotation']['rotation']
                rot_ii_1 = cur_instance_samples[ii + 1]['annotation']['rotation']

                timestamp_ii_0 = cur_instance_samples[ii]['timestamp']
                timestamp_ii_1 = cur_instance_samples[ii + 1]['timestamp']

                cur_sample_token = secene_frame_mappings[(cur_scene_token, cur_frame_idx)]
                # cur_time_stamp = sample_data_mappings[cur_sample_token]['pointcloud_timestamp']
                cur_timestamp = sample_info_mappings[cur_sample_token]['timestamp']

                # if cur_timestamp == '1626155888.384136':
                #     cur_timestamp = cur_timestamp

                sample_token_0 = cur_instance_samples[ii]['sample_token']
                sample_token_1 = cur_instance_samples[ii+1]['sample_token']
                
                cur_loc = loc_linear_interpolation(loc_ii_0, loc_ii_1, timestamp_ii_0, timestamp_ii_1, cur_timestamp,
                                                   lidar_ego_global_infos[sample_token_0],lidar_ego_global_infos[sample_token_1],
                                                   lidar_ego_global_infos[cur_sample_token])
                cur_rot = rot_linear_interpolation(rot_ii_0, rot_ii_1, timestamp_ii_0, timestamp_ii_1, cur_timestamp)
                cur_anno_token = gen_token(source_name, cur_sample_token, str(cur_loc['x']), str(cur_loc['y']), str(cur_loc['z']))

                cur_instance_sample_anno = {
                    "token": cur_anno_token,
                    "type": cur_instance_samples[ii]['annotation']['type'],
                    "track_id": cur_instance_samples[ii]['annotation']['track_id'],
                    "truncated_state": 0,
                    "occluded_state": 3,
                    "3d_dimensions": cur_instance_samples[ii]['annotation']['3d_dimensions'],
                    "3d_location": cur_loc,
                    "rotation": cur_rot,
                    "instance_token": instance_token
                }

                total_annotations[cur_sample_token][cur_anno_token] = cur_instance_sample_anno

                cur_frame_idx = cur_frame_idx + 1

    return total_annotations


def loc_linear_interpolation(loc_ii_0, loc_ii_1, timestamp_ii_0, timestamp_ii_1, cur_timestamp, \
                             lidar_ego_global_info_0,lidar_ego_global_info_1,cur_lidar_ego_global_info):
    """Use linear interpolation to estimate the 3d location for occluded objects.
    """
    timestamp_ii_0 = float(timestamp_ii_0) / 1e6
    timestamp_ii_1 = float(timestamp_ii_1) / 1e6
    cur_timestamp = float(cur_timestamp) / 1e6

    #cvt to global
    center_0 = np.array([loc_ii_0['x'], loc_ii_0['y'], loc_ii_0['z']])
    # lidar2ego
    center_0 = np.dot(Quaternion(lidar_ego_global_info_0['lidar2ego_rotation']).rotation_matrix, center_0)   \
                        + np.array(lidar_ego_global_info_0['lidar2ego_translation'])
    # ego2global
    center_0 = np.dot(Quaternion(lidar_ego_global_info_0['ego2global_rotation']).rotation_matrix, center_0)  \
                        + np.array(lidar_ego_global_info_0['ego2global_translation'])

    #cvt to global
    center_1 = np.array([loc_ii_1['x'], loc_ii_1['y'], loc_ii_1['z']])
    # lidar2ego
    center_1 = np.dot(Quaternion(lidar_ego_global_info_1['lidar2ego_rotation']).rotation_matrix, center_1)   \
                        + np.array(lidar_ego_global_info_1['lidar2ego_translation'])
    # ego2global
    center_1 = np.dot(Quaternion(lidar_ego_global_info_1['ego2global_rotation']).rotation_matrix, center_1)  \
                        + np.array(lidar_ego_global_info_1['ego2global_translation'])
    
    # global interpolation
    cur_center = (center_1 - center_0) / (timestamp_ii_1 - timestamp_ii_0) * (cur_timestamp - timestamp_ii_0) + center_0

    # cur sesor data interpolation
    cur_ego2global_translation = cur_lidar_ego_global_info['ego2global_translation']
    cur_ego2global_rotation = Quaternion(cur_lidar_ego_global_info['ego2global_rotation'])

    # global to ego, ego to lidar
    global2ego_r = np.linalg.inv(cur_ego2global_rotation.rotation_matrix)
    global2ego_t = - np.array(cur_ego2global_translation).reshape(1, 3) @ global2ego_r.T
    global2ego_t = global2ego_t.reshape(3)

    ego2lidar_r = np.linalg.inv(Quaternion(cur_lidar_ego_global_info['lidar2ego_rotation']).rotation_matrix)
    ego2lidar_t =  - np.array(cur_lidar_ego_global_info['lidar2ego_translation']).reshape(1, 3) @ ego2lidar_r.T
    ego2lidar_t = ego2lidar_t.reshape(3)

    # global2ego
    cur_center = np.dot(global2ego_r, cur_center) + global2ego_t 
    # ego2lidar
    cur_center = np.dot(ego2lidar_r, cur_center) + ego2lidar_t 
                       
    cur_loc = {}
    cur_loc['x'] = cur_center[0]
    cur_loc['y'] = cur_center[1]
    cur_loc['z'] = cur_center[2]

    return cur_loc


def rot_linear_interpolation(rot_ii_0, rot_ii_1, timestamp_ii_0, timestamp_ii_1, cur_timestamp):
    """Use linear interpolation to estimate the rotation for occluded objects.
    """
    timestamp_ii_0 = float(timestamp_ii_0) / 1e6
    timestamp_ii_1 = float(timestamp_ii_1) / 1e6
    cur_timestamp = float(cur_timestamp) / 1e6

    time_ratio = (cur_timestamp - timestamp_ii_0) / (timestamp_ii_1 - timestamp_ii_0)
    diff = rot_ii_1 - rot_ii_0 + pi
    if diff < 0:
        diff = diff + pi
    elif diff > 2*pi:
        diff = diff -3*pi
    else:
        diff = diff - pi
    
    cur_rot = time_ratio * diff + rot_ii_0
    if cur_rot < -pi:
        cur_rot = cur_rot + 2*pi
    if cur_rot > 2*pi:
        cur_rot = cur_rot - 2*pi
    
    return cur_rot

def _add_annotation_velocity_prev_next(total_annotations, instance_token_mappings, lidar_ego_global_infos):
    """Generate velocity and prev/next token for annotations.
    Args:
        total_annotations: added occluded annotations
        sample_info_mappings
        data_infos
    """
    ## Generate Velocity and Successors
    for instance_token in instance_token_mappings.keys():
        cur_instance_samples = instance_token_mappings[instance_token]
        cur_scene_token = cur_instance_samples[0]['scene_token']
        
        for ii in range(len(cur_instance_samples)):
            if ii == 0:
                prev_anno_token = ''
            else:
                prev_anno_token = cur_instance_samples[ii - 1]['annotation']['token']

            if ii == len(cur_instance_samples) - 1:
                next_anno_token = ''
            else:
                next_anno_token = cur_instance_samples[ii + 1]['annotation']['token']

            if ii == len(cur_instance_samples) - 1:
                gt_velocity = [0, 0]
            else:
                loc_ii_0 = cur_instance_samples[ii]['annotation']['3d_location']
                loc_ii_1 = cur_instance_samples[ii + 1]['annotation']['3d_location']

                sample_token_0 = cur_instance_samples[ii]['sample_token']
                sample_token_1 = cur_instance_samples[ii+1]['sample_token']

                info_0 = lidar_ego_global_infos[sample_token_0]
                info_1 = lidar_ego_global_infos[sample_token_1]

                # 原始坐标 (假设输入就在 Lidar 坐标系下，如果是 Ego 下请调整)
                # center_0 是 t0 时刻物体在 Lidar0 下的位置
                # center_1 是 t1 时刻物体在 Lidar1 下的位置
                center_0_lidar = np.array([loc_ii_0['x'], loc_ii_0['y'], loc_ii_0['z']])
                center_1_lidar = np.array([loc_ii_1['x'], loc_ii_1['y'], loc_ii_1['z']])

                def to_global(p_lidar, info):
                    # Lidar -> Ego
                    p_ego = np.dot(Quaternion(info['lidar2ego_rotation']).rotation_matrix, p_lidar) + \
                            np.array(info['lidar2ego_translation'])
                    # Ego -> Global
                    p_global = np.dot(Quaternion(info['ego2global_rotation']).rotation_matrix, p_ego) + \
                               np.array(info['ego2global_translation'])
                    return p_global

                center_0_global = to_global(center_0_lidar, info_0)
                center_1_global = to_global(center_1_lidar, info_1)

                # --- 4. 计算 Global 下的绝对速度 ---
                timestamp_ii_0 = float(cur_instance_samples[ii]['timestamp']) / 1e6
                timestamp_ii_1 = float(cur_instance_samples[ii + 1]['timestamp']) / 1e6
                time_delta = timestamp_ii_1 - timestamp_ii_0
                
                if time_delta < 1e-3: 
                    v_global = np.zeros(3)
                else:
                    # 这里的位移是物体在地球上的真实位移
                    v_global = (center_1_global - center_0_global) / time_delta

                # --- 5. 将 Global 速度向量转回 T0 时刻的 Lidar 坐标系 ---
                # 注意：只旋转，不平移！
                
                # Global -> Ego (逆旋转)
                q_ego2global = Quaternion(info_0['ego2global_rotation'])
                v_ego = np.dot(q_ego2global.rotation_matrix.T, v_global) 
                
                # Ego -> Lidar (逆旋转)
                q_lidar2ego = Quaternion(info_0['lidar2ego_rotation'])
                v_lidar = np.dot(q_lidar2ego.rotation_matrix.T, v_ego)

                gt_velocity = v_lidar[:2] # [vx, vy]

            # 赋值
            instance_token_mappings[instance_token][ii]['annotation']['gt_velocity'] = gt_velocity
            instance_token_mappings[instance_token][ii]['annotation']['prev'] = prev_anno_token
            instance_token_mappings[instance_token][ii]['annotation']['next'] = next_anno_token

    return total_annotations, instance_token_mappings

def _get_secene_frame_mappings(sample_info_mappings):
    secene_frame_mappings = {}
    for sample_token in sample_info_mappings.keys():
        scene_token = sample_info_mappings[sample_token]['scene_token']
        frame_idx = sample_info_mappings[sample_token]['frame_idx']
        secene_frame_mappings[(scene_token, frame_idx)] = sample_token

    return secene_frame_mappings


def _get_instance_token_mappings(total_annotations, sample_info_mappings):
    instance_token_mappings = {}

    for sample_token in total_annotations.keys():
        annotations = total_annotations[sample_token]
        scene_token = sample_info_mappings[sample_token]['scene_token']
        frame_idx = sample_info_mappings[sample_token]['frame_idx']
        timestamp = sample_info_mappings[sample_token]['timestamp']

        for anno_token in annotations.keys():
            annotation = annotations[anno_token]
            instance_token = annotation["instance_token"]
            if instance_token not in instance_token_mappings.keys():
                instance_token_mappings[instance_token] = []
            instance_token_mappings[instance_token].append({
                'scene_token': scene_token,
                'frame_idx': frame_idx,
                'sample_token': sample_token,
                'timestamp': timestamp,
                'annotation': annotation})

    # sorted by frame_idx, for downstream usage
    for instance_token in instance_token_mappings.keys():
        sorted(instance_token_mappings[instance_token], key=lambda annotation: annotation['frame_idx'])

    return instance_token_mappings


def generate_json_maps_files(data_root, version='v1.0-mini'):
    json_types = ['category', 'attribute', 'visibility', 'instance', 'sensor', 'calibrated_sensor',
                  'ego_pose', 'log', 'scene', 'sample', 'sample_data', 'sample_annotation', 'map']

    import shutil
    if not os.path.exists(osp.join(data_root, version)):
        tmp_nuscenes_json_root = '/data/ad_sharing/datasets/nuScenes/nuScenes_v1.0-mini/v1.0-mini'
        shutil.copytree(tmp_nuscenes_json_root, osp.join(data_root, version))

    if not os.path.exists(osp.join(data_root, 'maps')):
        tmp_nuscenes_map_root = '/data/ad_sharing/datasets/nuScenes/nuScenes_v1.0-mini/maps'
        shutil.copytree(tmp_nuscenes_map_root, osp.join(data_root, 'maps'))

def get_lidar_ego_global_infos(root_path,data_infos,v2x_side):
    lidar_ego_global_infos = {}
    for data_info in tqdm(data_infos):
        ## Step 1: build basic information
        sample_token = data_info['frame_id']
        lidar_ego_global_infos[sample_token] = {}

        if v2x_side == 'infrastructure-side':
            lidar_ego_global_infos[sample_token]['lidar2ego_rotation'] = np.array(list(Quaternion(matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))))
            lidar_ego_global_infos[sample_token]['lidar2ego_translation'] = np.array([0, 0, 0]).reshape(3)

            calib_virtuallidar2global_path = osp.join(root_path, data_info['calib_virtuallidar_to_world_path'])
            calib_virtuallidar2global = load_json(calib_virtuallidar2global_path)

            virtuallidar2word_rotation = np.array(calib_virtuallidar2global['rotation'])
            approx_rotation_matrix = iterative_closest_point(virtuallidar2word_rotation)

            lidar_ego_global_infos[sample_token]['ego2global_rotation'] = np.array(list(Quaternion(matrix=approx_rotation_matrix)))
            lidar_ego_global_infos[sample_token]['ego2global_translation'] = np.array(calib_virtuallidar2global['translation']).reshape(3)
        else:
            calib_lidar2ego_path = osp.join(root_path, data_info['calib_lidar_to_novatel_path'])
            calib_lidar2ego = load_json(calib_lidar2ego_path)
            lidar_ego_global_infos[sample_token]['lidar2ego_rotation'] = np.array(list(Quaternion(matrix=np.array(calib_lidar2ego['transform']['rotation']))))
            lidar_ego_global_infos[sample_token]['lidar2ego_translation'] = np.array(calib_lidar2ego['transform']['translation']).reshape(3)

            calib_ego2global_path = osp.join(root_path, data_info['calib_novatel_to_world_path'])
            calib_ego2global = load_json(calib_ego2global_path)
            lidar_ego_global_infos[sample_token]['ego2global_rotation'] = np.array(list(Quaternion(matrix=np.array(calib_ego2global['rotation']))))
            lidar_ego_global_infos[sample_token]['ego2global_translation'] = np.array(calib_ego2global['translation']).reshape(3)
    
    return lidar_ego_global_infos

