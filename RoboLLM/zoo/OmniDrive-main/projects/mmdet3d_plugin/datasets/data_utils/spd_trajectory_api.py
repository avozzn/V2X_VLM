#----------------------------------------------------------------#
# UniV2X: End-to-End Autonomous Driving through V2X Cooperation  #
# Source code: https://github.com/AIR-THU/UniV2X                 #
# Copyright (c) DAIR-V2X. All rights reserved.                   #
# Modified from UniAD (https://github.com/OpenDriveLab/UniAD)    #
#----------------------------------------------------------------#

import numpy as np
from nuscenes.prediction import (PredictHelper)
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor
from typing import Dict, Tuple, Any, List, Union
from scipy import interpolate

def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw

def angle_of_rotation(yaw: float) -> float:
    """
    Given a yaw angle (measured from x axis), find the angle needed to rotate by so that
    the yaw is aligned with the y axis (pi / 2).
    :param yaw: Radians. Output of quaternion_yaw function.
    :return: Angle in radians.
    """
    return (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)

def make_2d_rotation_matrix(angle_in_radians: float) -> np.ndarray:
    """
    Makes rotation matrix to rotate point in x-y plane counterclockwise
    by angle_in_radians.
    """

    return np.array([[np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                     [np.sin(angle_in_radians), np.cos(angle_in_radians)]])

def convert_global_coords_to_local_correct(coordinates: np.ndarray,
                                   translation: Tuple[float, float, float],
                                   rotation: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Converts global coordinates to coordinates in the frame given by the rotation quaternion and
    centered at the translation vector. The rotation is meant to be a z-axis rotation.
    :param coordinates: x,y locations. array of shape [n_steps, 2].
    :param translation: Tuple of (x, y, z) location that is the center of the new frame.
    :param rotation: Tuple representation of quaternion of new frame.
        Representation - cos(theta / 2) + (xi + yi + zi)sin(theta / 2).
    :return: x,y locations in frame stored in array of share [n_times, 2].
    """
    yaw = angle_of_rotation(quaternion_yaw(Quaternion(rotation)))

    transform = make_2d_rotation_matrix(angle_in_radians=-yaw)

    coords = (coordinates - np.atleast_2d(np.array(translation)[:2])).T

    return np.dot(transform, coords).T[:, :2]


def convert_local_coords_to_global_correct(coordinates: np.ndarray,
                                   translation: Tuple[float, float, float],
                                   rotation: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Converts local coordinates to global coordinates.
    :param coordinates: x,y locations. array of shape [n_steps, 2]
    :param translation: Tuple of (x, y, z) location that is the center of the new frame
    :param rotation: Tuple representation of quaternion of new frame.
        Representation - cos(theta / 2) + (xi + yi + zi)sin(theta / 2).
    :return: x,y locations stored in array of share [n_times, 2].
    """
    yaw = angle_of_rotation(quaternion_yaw(Quaternion(rotation)))

    transform = make_2d_rotation_matrix(angle_in_radians=yaw)

    return np.dot(transform, coordinates.T).T[:, :2] + np.atleast_2d(np.array(translation)[:2])

Record = Dict[str, Any]


class SPDTraj(object):
    def __init__(self,
                 nusc,
                 predict_steps,
                 planning_steps,
                 past_steps,
                 fut_steps,
                 with_velocity,
                 CLASSES,
                 box_mode_3d,
                 use_nonlinear_optimizer=False):
        self.tmp_dataset_type = 'spd'
        if self.tmp_dataset_type not in ['spd', 'nuscenes']:
            raise Exception('tmp_dataset_type is not correct with {}'.format(self.tmp_dataset_type))
        super().__init__()
        self.nusc = nusc
        self.prepare_sdc_vel_info()
        self.predict_steps = predict_steps
        self.planning_steps = planning_steps
        self.past_steps = past_steps
        self.fut_steps = fut_steps
        self.with_velocity = with_velocity
        self.CLASSES = CLASSES
        self.box_mode_3d = box_mode_3d
        self.predict_helper = PredictHelper(self.nusc)
        self.use_nonlinear_optimizer = use_nonlinear_optimizer

    def get_future_for_agent(self, instance_token: str, sample_token: str,
                             seconds: float, in_agent_frame: bool,
                             just_xy: bool = True) -> Union[List[Record], np.ndarray]:
        
        return self._get_past_or_future_for_agent(instance_token, sample_token, seconds,
                                                  in_agent_frame, direction='next', just_xy=just_xy)

    def get_past_for_agent(self, instance_token: str, sample_token: str,
                           seconds: float, in_agent_frame: bool,
                           just_xy: bool = True) -> Union[List[Record], np.ndarray]:
       
        return self._get_past_or_future_for_agent(instance_token, sample_token, seconds,
                                                  in_agent_frame, direction='prev', just_xy=just_xy)
    
    def _get_past_or_future_for_agent(self, instance_token: str, sample_token: str,
                                      seconds: float, in_agent_frame: bool,
                                      direction: str,
                                      just_xy: bool = True) -> Union[List[Record], np.ndarray]:
        """
        Helper function to reduce code duplication between get_future and get_past for agent.
        :param instance_token: Instance of token.
        :param sample_token: Sample token for instance.
        :param seconds: How many seconds of data to retrieve.
        :param in_agent_frame: Whether to rotate the coordinates so the
            heading is aligned with the y-axis. Only relevant if just_xy = True.
        :param direction: 'next' for future or 'prev' for past.
        :return: array of shape [n_timesteps, 2].
        """
        starting_annotation = self.predict_helper.get_sample_annotation(instance_token, sample_token)
        sequence = self.predict_helper._iterate(starting_annotation, seconds, direction)

        if not just_xy:
            return sequence

        coords = np.array([r['translation'][:2] for r in sequence])

        if coords.size == 0:
            return coords

        if in_agent_frame:
            coords = convert_global_coords_to_local_correct(coords,
                                                    starting_annotation['translation'],
                                                    starting_annotation['rotation'])

        return coords

    def get_traj_label(self, sample_token, ann_tokens, sensor_type='LIDAR_TOP'):
        sd_rec = self.nusc.get('sample', sample_token)
        fut_traj_all = []
        fut_traj_valid_mask_all = []
        past_traj_all = []	
        past_traj_valid_mask_all = []
        _, boxes, _ = self.nusc.get_sample_data(sd_rec['data'][sensor_type], selected_anntokens=ann_tokens)
        for i, ann_token in enumerate(ann_tokens):
            box = boxes[i]
            instance_token = self.nusc.get('sample_annotation', ann_token)['instance_token']
            fut_traj_local = self.get_future_for_agent(instance_token, sample_token, seconds=self.predict_steps/2, in_agent_frame=True)
            past_traj_local = self.get_past_for_agent(instance_token, sample_token, seconds=2, in_agent_frame=True)
            fut_traj = np.zeros((self.predict_steps, 2))
            fut_traj_valid_mask = np.zeros((self.predict_steps, 2))
            past_traj = np.zeros((self.past_steps + self.fut_steps, 2))		
            past_traj_valid_mask = np.zeros((self.past_steps + self.fut_steps, 2))
            if fut_traj_local.shape[0] > 0:
                if self.use_nonlinear_optimizer:
                    trans = box.center
                else:
                    trans = np.array([0, 0, 0])
                rot = Quaternion(matrix=box.rotation_matrix)
                fut_traj_scence_centric = convert_local_coords_to_global_correct(fut_traj_local, trans, rot) 
                fut_traj[:fut_traj_scence_centric.shape[0], :] = fut_traj_scence_centric
                fut_traj_valid_mask[:fut_traj_scence_centric.shape[0], :] = 1
            if past_traj_local.shape[0] > 0:			
                if self.use_nonlinear_optimizer:
                    trans = box.center
                else:
                    trans = np.array([0, 0, 0])	
                rot = Quaternion(matrix=box.rotation_matrix)		
                past_traj_scence_centric = convert_local_coords_to_global_correct(past_traj_local, trans, rot) 		
                past_traj[:past_traj_scence_centric.shape[0], :] = past_traj_scence_centric		
                past_traj_valid_mask[:past_traj_scence_centric.shape[0], :] = 1

                if fut_traj_local.shape[0] > 0:
                    fut_steps = min(self.fut_steps, fut_traj_scence_centric.shape[0])
                    past_traj[self.past_steps:self.past_steps+fut_steps, :] = fut_traj_scence_centric[:fut_steps]
                    past_traj_valid_mask[self.past_steps:self.past_steps+fut_steps, :] = 1

            fut_traj_all.append(fut_traj)		
            fut_traj_valid_mask_all.append(fut_traj_valid_mask)		
            past_traj_all.append(past_traj)		
            past_traj_valid_mask_all.append(past_traj_valid_mask)		
        if len(ann_tokens) > 0:		
            fut_traj_all = np.stack(fut_traj_all, axis=0)		
            fut_traj_valid_mask_all = np.stack(fut_traj_valid_mask_all, axis=0)		
            past_traj_all = np.stack(past_traj_all, axis=0)		
            past_traj_valid_mask_all = np.stack(past_traj_valid_mask_all, axis=0)	
        else:		
            fut_traj_all = np.zeros((0, self.predict_steps, 2))		
            fut_traj_valid_mask_all = np.zeros((0, self.predict_steps, 2))		
            past_traj_all = np.zeros((0, self.predict_steps, 2))		
            past_traj_valid_mask_all = np.zeros((0, self.predict_steps, 2))		
        return fut_traj_all, fut_traj_valid_mask_all, past_traj_all, past_traj_valid_mask_all
    
    

    def get_vel_transform_mats(self, sample, sensor_type='LIDAR_TOP'):
        sd_rec = self.nusc.get('sample_data', sample['data'][sensor_type])
        cs_record = self.nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])

        l2e_r = cs_record['rotation']
        l2e_t = cs_record['translation']
        e2g_r = pose_record['rotation']
        e2g_t = pose_record['translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        return l2e_r_mat, e2g_r_mat

    def get_vel_and_time(self, sample, sensor_type='LIDAR_TOP'):
        lidar_token = sample['data'][sensor_type]
        lidar_top = self.nusc.get('sample_data', lidar_token)
        pose = self.nusc.get('ego_pose', lidar_top['ego_pose_token'])
        xyz = pose['translation']
        timestamp = sample['timestamp']
        return xyz, timestamp
        
    def prepare_sdc_vel_info(self):
        # generate sdc velocity info for all samples
        # Note that these velocity values are converted from 
        # global frame to lidar frame
        # as aligned with bbox gts

        self.sdc_vel_info = {}
        for scene in self.nusc.scene:
            scene_token = scene['token']

            # we cannot infer vel for the last sample, therefore we skip it
            last_sample_token = scene['last_sample_token']
            sample_token = scene['first_sample_token']
            sample = self.nusc.get('sample', sample_token)
            xyz, time = self.get_vel_and_time(sample)
            while sample['token'] != last_sample_token:
                next_sample_token = sample['next']
                next_sample = self.nusc.get('sample', next_sample_token)
                next_xyz, next_time = self.get_vel_and_time(next_sample)
                dc = np.array(next_xyz) - np.array(xyz) 
                dt = (next_time - time) / 1e6
                vel = dc/dt
                # Global Velocity (物理事实)
                vel_global_mag = np.linalg.norm(vel)
                # global frame to lidar frame
                l2e_r_mat, e2g_r_mat = self.get_vel_transform_mats(sample)
                vel = vel @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                vel = vel[:2]


                self.sdc_vel_info[sample['token']] = vel
                xyz, time = next_xyz, next_time
                sample = next_sample

            # set first sample's vel equal to second sample's
            last_sample = self.nusc.get('sample', last_sample_token)
            second_last_sample_token = last_sample['prev']
            self.sdc_vel_info[last_sample_token] = self.sdc_vel_info[second_last_sample_token]                

    def generate_sdc_info(self, sdc_vel, as_lidar_instance3d_box=False):
        # sdc dim from https://forum.nuscenes.org/t/dimensions-of-the-ego-vehicle-used-to-gather-data/550
        # psudo_sdc_bbox = np.array([0.0, 0.0, 0.0, 1.73, 4.08, 1.56,-np.pi])
        psudo_sdc_bbox = np.array([0.0, 0.0, 0.0, 4.08, 1.73, 1.56, 0.0]) ###################
        if self.with_velocity:
            psudo_sdc_bbox = np.concatenate([psudo_sdc_bbox, sdc_vel], axis=-1)
        gt_bboxes_3d = np.array([psudo_sdc_bbox]).astype(np.float32)
        gt_names_3d = ['car']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        
        if as_lidar_instance3d_box:
            # if we do not want the batch the box in to DataContrainer
            return gt_bboxes_3d

        gt_labels_3d = DC(to_tensor(gt_labels_3d))
        gt_bboxes_3d = DC(gt_bboxes_3d, cpu_only=True)

        return gt_bboxes_3d, gt_labels_3d

    
    def get_sdc_traj_label(self, sample_token, sensor_type='LIDAR_TOP'):
        sd_rec = self.nusc.get('sample', sample_token)
        lidar_top_data_start = self.nusc.get('sample_data', sd_rec['data'][sensor_type])
        ego_pose_start = self.nusc.get('ego_pose', lidar_top_data_start['ego_pose_token'])
        start_time = self.nusc.get('sample', sample_token)['timestamp'] / 1e6
        target_times = [start_time + (i + 1) * 0.5 for i in range(self.predict_steps)]
        sdc_fut_traj_global = []

        traj_anchors = []

        curr_rec = sd_rec

        while True:
            # 获取当前 sample 的 ego_pose
            curr_lidar_token = curr_rec['data'][sensor_type]
            curr_lidar_data = self.nusc.get('sample_data', curr_lidar_token)
            curr_pose = self.nusc.get('ego_pose', curr_lidar_data['ego_pose_token'])
            
            ts = curr_rec['timestamp'] / 1e6
            trans = np.array(curr_pose['translation'])
            rot = Quaternion(curr_pose['rotation'])
            
            traj_anchors.append((ts, trans, rot))
            
            # 如果收集的时间已经覆盖了我们要预测的最远时间，就停止
            if ts > target_times[-1] + 1.0: 
                break
            if curr_rec['next'] == '':
                break
            curr_rec = self.nusc.get('sample', curr_rec['next'])

        # 2. 对每个目标时间点进行插值
        for t_target in target_times:
            # 找到 t_target 前后的两个锚点
            prev_anchor = None
            next_anchor = None
            
            for i in range(len(traj_anchors) - 1):
                if traj_anchors[i][0] <= t_target <= traj_anchors[i+1][0]:
                    prev_anchor = traj_anchors[i]
                    next_anchor = traj_anchors[i+1]
                    break
            
            if prev_anchor is None or next_anchor is None:
                # 找不到区间（比如超出了场景结束时间），填充 0 或最后一点
                # 这里为了保持 shape 统一，我们跳出，由 mask 控制有效性
                break
                
            # 计算插值比例 alpha
            t0, p0, r0 = prev_anchor
            t1, p1, r1 = next_anchor
            
            # 防止除以0
            if t1 - t0 < 1e-4:
                interp_trans = p0
            else:
                alpha = (t_target - t0) / (t1 - t0)
                
                # 位置线性插值
                interp_trans = (1 - alpha) * p0 + alpha * p1
                
                
            sdc_fut_traj_global.append(interp_trans[:2]) # 只取 xy

        # 3. 构造输出
        sdc_fut_traj_all = np.zeros((1, self.predict_steps, 2))
        sdc_fut_traj_valid_mask_all = np.zeros((1, self.predict_steps, 2))
        
        n_valid_timestep = len(sdc_fut_traj_global)
        if n_valid_timestep > 0:
            sdc_fut_traj = np.stack(sdc_fut_traj_global, axis=0)
            
            # === 🛡️ 纯净版 DEBUG: 没有任何修正 ===
            # 我们只看原始数据里的 Quaternion 和 位移 到底差多少度
            
            # 1. 原始记录的 Yaw
            raw_rot = Quaternion(ego_pose_start['rotation'])
            correction_quat = Quaternion(axis=[0, 0, 1], radians=np.pi / 2)
            final_rot = raw_rot * correction_quat
            raw_heading_yaw = quaternion_yaw(final_rot)
            
            # 2. 实际移动的 Yaw
            move_vec = sdc_fut_traj[0] - ego_pose_start['translation'][:2]
            move_dist = np.linalg.norm(move_vec)
            move_yaw = np.arctan2(move_vec[1], move_vec[0])
            
            # 3. 计算原始偏差
            raw_diff = move_yaw - raw_heading_yaw
            raw_diff_deg = np.degrees((raw_diff + np.pi) % (2 * np.pi) - np.pi)
            
            sdc_fut_traj = convert_global_coords_to_local_correct(
                coordinates=sdc_fut_traj,
                translation=ego_pose_start['translation'],
                rotation=list(final_rot),
            )
            # print("sdc_fut_traj:",sdc_fut_traj)

            sdc_fut_traj_all[:,:n_valid_timestep,:] = sdc_fut_traj
            sdc_fut_traj_valid_mask_all[:,:n_valid_timestep,:] = 1

            

        return sdc_fut_traj_all, sdc_fut_traj_valid_mask_all
    
    def get_sdc_planning_label(self, sample_token):
        sd_rec = self.nusc.get('sample', sample_token)
    
        # 记录初始时间戳
        start_timestamp = sd_rec['timestamp']
        curr_time_us = start_timestamp
        
        # 获取初始帧的变换矩阵 (保持原有的坐标系逻辑)
        l2e_r_mat_init, l2e_t_init, e2g_r_mat_init, e2g_t_init = self.get_l2g_transform(sd_rec)

        # --- 1. 收集锚点 (Anchors) ---
        # 格式: (relative_time, x, y, yaw)
        # 我们显式添加 t=0 时刻的点 (在 initial ego frame 下坐标为 0,0,0)
        search_range_start = -self.past_steps * 0.5 - 1.0
        search_range_end = self.planning_steps * 0.5 + 1.0
        traj_anchors = [(0.0, 0.0, 0.0, 0.0)]

        curr_node = sd_rec
        
        while curr_node['next'] != '':
            curr_node = self.nusc.get('sample', curr_node['next'])
            rel_time = (curr_node['timestamp'] - curr_time_us) / 1e6
            
            if rel_time > search_range_end:
                break
                
            # 坐标变换 (Global -> Init Ego)
            pose = self._get_pose_in_init_frame(curr_node, l2e_r_mat_init, l2e_t_init, e2g_r_mat_init, e2g_t_init)
            traj_anchors.append([rel_time, *pose])

        # --- 1.2 向前搜索 (Past) ---
        curr_node = sd_rec
        while curr_node['prev'] != '':
            curr_node = self.nusc.get('sample', curr_node['prev'])
            rel_time = (curr_node['timestamp'] - curr_time_us) / 1e6
            
            if rel_time < search_range_start:
                break
            
            # 坐标变换
            pose = self._get_pose_in_init_frame(curr_node, l2e_r_mat_init, l2e_t_init, e2g_r_mat_init, e2g_t_init)
            traj_anchors.append([rel_time, *pose])
            
        # 按时间排序
        traj_anchors = np.array(sorted(traj_anchors, key=lambda x: x[0])) # (N, 4)
        
        # --- 2. 插值与补全 (参考参考代码逻辑) ---
        
        # 目标时间点 (Past + Future)
        # Past: [-2.0, -1.5, -1.0, -0.5] (注意顺序)
        t_past_targets = np.linspace(-self.past_steps * 0.5, -0.5, self.past_steps)
        # Future: [0.5, 1.0, ..., 4.5]
        t_fut_targets = np.linspace(0.5, self.planning_steps * 0.5, self.planning_steps)
        
        # 合并所有目标时间点，方便一次性处理
        all_targets = np.concatenate([t_past_targets, t_fut_targets])
        gt_past_traj = np.zeros((1, self.past_steps, 2))
        gt_planning = np.zeros((1, self.planning_steps, 3)) # x, y, yaw
        
        # print("traj_anchors",len(traj_anchors))
        if len(traj_anchors) < 6:
            print(f"  [ERROR] Sample Token: {sample_token} Too Few Valid Resampled Points: {t_fut_targets.shape[0]}")
            gt_past_traj_mask = np.zeros((1, self.past_steps, 2))
            gt_planning_mask = np.zeros((1, self.planning_steps, 2))
            command = np.array([-1])
            return gt_past_traj, gt_past_traj_mask, gt_planning, gt_planning_mask, command
        else:
            valid_times = traj_anchors[:, 0]
            valid_coords = traj_anchors[:, 1:] # x, y, yaw
            
            # 使用 scipy 的 interp1d 进行插值
            # fill_value="extrapolate" 实现了简单的线性外推 (Linear Extrapolation)
            # 这对应了参考代码中用首尾速度补全的逻辑
            
            # 处理 Yaw 的周期性 (Unwrap)
            valid_coords[:, 2] = np.unwrap(valid_coords[:, 2])
            
            # 构建插值函数
            # kind='linear' 对应参考代码中的 interp1d
            interpolator = interpolate.interp1d(
                valid_times, 
                valid_coords, 
                axis=0, 
                kind='linear', 
                fill_value="extrapolate", # 关键：开启外推
                assume_sorted=True
            )
            
            # 生成所有目标点的坐标
            predicted_coords = interpolator(all_targets) # (Total_Steps, 3)
            
        # 分离过去和未来
        past_traj_result = predicted_coords[:self.past_steps]
        fut_traj_result = predicted_coords[self.past_steps:]
        
        # Past Traj
        gt_past_traj_mask = np.ones((1, self.past_steps, 2)) # 只要生成了就认为是有效的（因为包含补全）
        gt_past_traj[0] = past_traj_result[:, :2] # 只取 x, y
        
        # Future Traj (Planning)
        gt_planning_mask = np.ones((1, self.planning_steps, 2))
        gt_planning[0] = fut_traj_result
        
        # Command 生成 (基于未来轨迹终点)
        final_y = gt_planning[0, -1, 1]
        if final_y <= -2:
            command = 0 # RIGHT
        elif final_y >= 2:
            command = 1 # LEFT
        else:
            command = 2 # FORWARD
            
        return gt_past_traj, gt_past_traj_mask, gt_planning, gt_planning_mask, command
       
    def _get_pose_in_init_frame(self, sample_rec, l2e_r_init, l2e_t_init, e2g_r_init, e2g_t_init):
        """
        辅助函数：将任意帧的自车位置转换到初始帧的自车坐标系下。
        完全复用 get_sdc_planning_label 的矩阵变换逻辑。
        """
        # 获取当前帧的变换
        l2e_r_curr, l2e_t_curr, e2g_r_curr, e2g_t_curr = self.get_l2g_transform(sample_rec)
        
        # 生成 Box (位于原点)
        # 这里不需要速度信息，只关心位姿
        bbox = self.generate_sdc_info(np.zeros(2), as_lidar_instance3d_box=True)
        
        # 变换链
        bbox.rotate(l2e_r_curr.T)
        bbox.translate(l2e_t_curr)
        
        bbox.rotate(e2g_r_curr.T)
        bbox.translate(e2g_t_curr)
        
        bbox.translate(- e2g_t_init)
        m1 = np.linalg.inv(e2g_r_init)
        bbox.rotate(m1.T)
        
        bbox.translate(- l2e_t_init)
        m2 = np.linalg.inv(l2e_r_init)
        bbox.rotate(m2.T)
        
        # 提取 x, y, yaw
        pose = bbox.tensor.squeeze(0)[[0, 1, 6]].numpy()
        return pose
    
    def get_l2g_transform(self, sample, sensor_type='LIDAR_TOP'):
        sd_rec = self.nusc.get('sample_data', sample['data'][sensor_type])
        cs_record = self.nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])

        l2e_r = cs_record['rotation']
        l2e_t = np.array(cs_record['translation'])
        e2g_r = pose_record['rotation']
        e2g_t = np.array(pose_record['translation'])
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        return l2e_r_mat, l2e_t, e2g_r_mat, e2g_t
    
    def generate_meta_action(self, sdc_planning, sdc_vel, dt=0.5):
        
        meta_action = ""
        # 注意：这里假设 sdc_planning 已经包含 batch 维度，如没有请移除 [0]
        fut_traj = sdc_planning[0, :, :2] 
        
        # --- 2. 计算速度 (Speed Meta) ---
        constant_eps = 0.5  # 速度阈值 (m/s)

        # A. 获取当前速度 (Current Velocity)
        if np.ndim(sdc_vel) > 0 and np.size(sdc_vel) > 1:
            cur_velo = np.linalg.norm(sdc_vel)
        else:
            cur_velo = float(sdc_vel)

        # B. 计算末端速度 (End Velocity)
        # 通过未来轨迹最后两个有效点的距离除以时间步长 dt
        dist_last_step = np.linalg.norm(fut_traj[-1] - fut_traj[-2])
        end_velo = dist_last_step / dt

        # C. 速度逻辑判断 (全部大写)
        if cur_velo < constant_eps and end_velo < constant_eps:
            speed_meta = "STOP"
        elif end_velo < constant_eps:
            speed_meta = "A DECELERATION TO ZERO"
        elif np.abs(end_velo - cur_velo) < constant_eps:
            speed_meta = "A CONSTANT SPEED"
        else:
            if cur_velo > end_velo:
                if cur_velo > 2 * end_velo:
                    speed_meta = "A QUICK DECELERATION"
                else:
                    speed_meta = "A DECELERATION"
            else:
                if end_velo > 2 * cur_velo:
                    speed_meta = "A QUICK ACCELERATION"
                else:
                    speed_meta = "AN ACCELERATION"
        
        if speed_meta == "STOP":
            meta_action += (speed_meta + "\n")
            return meta_action 
        else:
            # 阈值设定 (米)
            forward_th = 2.0       
            lane_changing_th = 4.0 

            # --- 核心修改区域：基于 X前/Y左 坐标系 ---
            
            # 1. 提取横向坐标 (Y轴, Index 1)
            lat_coords = fut_traj[:, 1]
            
            # 2. 检查是否有明显的横向偏移
            if (np.abs(lat_coords) < forward_th).all():
                behavior_meta = "MOVE FORWARD"
            else:
                # 取最后一个点的 Y 坐标来判断最终意图
                last_lat = lat_coords[-1]
                
                # Y > 0 代表向左 (Left)
                if last_lat > 0: 
                    if np.abs(last_lat) > lane_changing_th:
                        behavior_meta = "TURN LEFT"
                    else:
                        behavior_meta = "CHANGE LANE TO LEFT"
                
                # Y < 0 代表向右 (Right)
                elif last_lat < 0: 
                    if np.abs(last_lat) > lane_changing_th:
                        behavior_meta = "TURN RIGHT"
                    else:
                        behavior_meta = "CHANGE LANE TO RIGHT"
                
                else:
                    behavior_meta = "MOVE FORWARD"

            meta_action += (behavior_meta + " WITH " + speed_meta + "\n")
        
        return meta_action
