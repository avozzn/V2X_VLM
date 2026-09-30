#----------------------------------------------------------------#
# UniV2X: End-to-End Autonomous Driving through V2X Cooperation  #
# Source code: https://github.com/AIR-THU/UniV2X                 #
# Copyright (c) DAIR-V2X. All rights reserved.                   #
# Modified from UniAD (https://github.com/OpenDriveLab/UniAD)    #
#----------------------------------------------------------------#

import numpy as np
from nuscenes.prediction import (PredictHelper)
from mmdet3d.structures import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmengine.structures import InstanceData as DC
from mmcv.transforms import to_tensor
from typing import Dict, Tuple, Any, List, Union
from scipy import interpolate
from .planning_metadata_v2 import generate_planning_metadata_v2

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

    def get_sdc_traj_label_ori(self, sample_token, sensor_type='LIDAR_TOP'):
        sd_rec = self.nusc.get('sample', sample_token)
        lidar_top_data_start = self.nusc.get('sample_data', sd_rec['data'][sensor_type])
        ego_pose_start = self.nusc.get('ego_pose', lidar_top_data_start['ego_pose_token'])

        sdc_fut_traj = []
        for _ in range(self.predict_steps):
            next_annotation_token = sd_rec['next']
            if next_annotation_token=='':
                break
            sd_rec = self.nusc.get('sample', next_annotation_token)
            lidar_top_data_next = self.nusc.get('sample_data', sd_rec['data'][sensor_type])
            ego_pose_next = self.nusc.get('ego_pose', lidar_top_data_next['ego_pose_token'])
            sdc_fut_traj.append(ego_pose_next['translation'][:2])  # global xy pos of sdc at future step i
        
        sdc_fut_traj_all = np.zeros((1, self.predict_steps, 2))
        sdc_fut_traj_valid_mask_all = np.zeros((1, self.predict_steps, 2))
        n_valid_timestep = len(sdc_fut_traj)
        if n_valid_timestep>0:
            sdc_fut_traj = np.stack(sdc_fut_traj, axis=0)  #(t,2)
            sdc_fut_traj = convert_global_coords_to_local_correct(
                coordinates=sdc_fut_traj,
                translation=ego_pose_start['translation'],
                rotation=ego_pose_start['rotation'],
            )
            sdc_fut_traj_all[:,:n_valid_timestep,:] = sdc_fut_traj
            sdc_fut_traj_valid_mask_all[:,:n_valid_timestep,:] = 1

        

        return sdc_fut_traj_all, sdc_fut_traj_valid_mask_all
    
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
        curr_time_us = sd_rec['timestamp']
        
        # 获取初始帧的变换矩阵 (保持原有的坐标系逻辑)
        l2e_r_mat_init, l2e_t_init, e2g_r_mat_init, e2g_t_init = self.get_l2g_transform(sd_rec)

        # Past and future anchors must stay separate.  In particular, future
        # poses must never be used to synthesize an input history trajectory.
        search_range_start = -self.past_steps * 0.5 - 1.0
        search_range_end = self.planning_steps * 0.5 + 1.0
        current_anchor = [0.0, 0.0, 0.0, 0.0]
        past_anchors = [current_anchor]
        future_anchors = [current_anchor]

        curr_node = sd_rec
        while curr_node['next'] != '':
            curr_node = self.nusc.get('sample', curr_node['next'])
            rel_time = (curr_node['timestamp'] - curr_time_us) / 1e6
            if rel_time > search_range_end:
                break
            pose = self._get_pose_in_init_frame(curr_node, l2e_r_mat_init, l2e_t_init, e2g_r_mat_init, e2g_t_init)
            future_anchors.append([rel_time, *pose])

        curr_node = sd_rec
        while curr_node['prev'] != '':
            curr_node = self.nusc.get('sample', curr_node['prev'])
            rel_time = (curr_node['timestamp'] - curr_time_us) / 1e6
            if rel_time < search_range_start:
                break
            pose = self._get_pose_in_init_frame(curr_node, l2e_r_mat_init, l2e_t_init, e2g_r_mat_init, e2g_t_init)
            past_anchors.append([rel_time, *pose])

        past_anchors = np.asarray(sorted(past_anchors, key=lambda x: x[0]), dtype=np.float64)
        future_anchors = np.asarray(sorted(future_anchors, key=lambda x: x[0]), dtype=np.float64)
        t_past_targets = np.linspace(-self.past_steps * 0.5, -0.5, self.past_steps)
        t_fut_targets = np.linspace(0.5, self.planning_steps * 0.5, self.planning_steps)
        gt_past_traj = np.zeros((1, self.past_steps, 2))
        gt_planning = np.zeros((1, self.planning_steps, 3))

        # Causal history: interpolate inside the observed past range and use
        # the earliest two causal anchors for older, missing timestamps.  If
        # this is the first frame of a scene, no causal velocity is available,
        # so history stays at zero and its observation mask stays false.
        if len(past_anchors) >= 2:
            past_coords = past_anchors[:, 1:].copy()
            past_coords[:, 2] = np.unwrap(past_coords[:, 2])
            past_interpolator = interpolate.interp1d(
                past_anchors[:, 0], past_coords, axis=0, kind='linear',
                fill_value='extrapolate', assume_sorted=True)
            past_traj_result = past_interpolator(t_past_targets)
            gt_past_traj[0] = past_traj_result[:, :2]

        earliest_observed_time = past_anchors[0, 0] if len(past_anchors) > 1 else 0.0
        observed_mask = t_past_targets >= earliest_observed_time - 1e-6
        gt_past_traj_mask = np.repeat(observed_mask[None, :, None], 2, axis=2).astype(np.float32)

        # Future labels are generated independently.  Preserve the previous
        # end-of-scene extrapolation behaviour, but never feed these anchors to
        # the history interpolator above.
        if len(future_anchors) < 2:
            print(f"  [ERROR] Sample Token: {sample_token} Too Few Valid Resampled Points: {t_fut_targets.shape[0]}")
            gt_planning_mask = np.zeros((1, self.planning_steps, 2))
            command = np.array([-1])
            return gt_past_traj, gt_past_traj_mask, gt_planning, gt_planning_mask, command

        future_coords = future_anchors[:, 1:].copy()
        future_coords[:, 2] = np.unwrap(future_coords[:, 2])
        future_interpolator = interpolate.interp1d(
            future_anchors[:, 0], future_coords, axis=0, kind='linear',
            fill_value='extrapolate', assume_sorted=True)
        fut_traj_result = future_interpolator(t_fut_targets)
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
    
    # def get_sdc_planning_label_ori(self, sample_token):
    #     sd_rec = self.nusc.get('sample', sample_token)
    #     l2e_r_mat_init, l2e_t_init, e2g_r_mat_init, e2g_t_init = self.get_l2g_transform(sd_rec)
        

    #     planning = []
    #     for _ in range(self.planning_steps):
    #         next_annotation_token = sd_rec['next']
    #         if next_annotation_token=='':
    #             break
    #         sd_rec = self.nusc.get('sample', next_annotation_token)
    #         l2e_r_mat_curr, l2e_t_curr, e2g_r_mat_curr, e2g_t_curr = self.get_l2g_transform(sd_rec)  # (lidar to global at current frame)
            
    #         # bbox of sdc under current lidar frame
    #         next_bbox3d = self.generate_sdc_info(self.sdc_vel_info[next_annotation_token], as_lidar_instance3d_box=True)

    #         # to bbox under curr ego frame
    #         next_bbox3d.rotate(l2e_r_mat_curr.T)
    #         next_bbox3d.translate(l2e_t_curr)

    #         # to bbox under world frame
    #         next_bbox3d.rotate(e2g_r_mat_curr.T)
    #         next_bbox3d.translate(e2g_t_curr)

    #         # to bbox under initial ego frame, first inverse translate, then inverse rotate 
    #         next_bbox3d.translate(- e2g_t_init)
    #         m1 = np.linalg.inv(e2g_r_mat_init)
    #         next_bbox3d.rotate(m1.T)

    #         # to bbox under curr ego frame, first inverse translate, then inverse rotate
    #         next_bbox3d.translate(- l2e_t_init)
    #         m2 = np.linalg.inv(l2e_r_mat_init)
    #         next_bbox3d.rotate(m2.T)
            
    #         planning.append(next_bbox3d)

    #     planning_all = np.zeros((1, self.planning_steps, 3))
    #     planning_mask_all = np.zeros((1, self.planning_steps, 2))
    #     n_valid_timestep = len(planning)
    #     if n_valid_timestep>0:
    #         planning = [p.tensor.squeeze(0) for p in planning]
    #         planning = np.stack(planning, axis=0)  # (valid_t, 9)
    #         planning = planning[:, [0,1,6]]  # (x, y, yaw)
    #         planning_all[:,:n_valid_timestep,:] = planning
    #         planning_mask_all[:,:n_valid_timestep,:] = 1

    #     mask = planning_mask_all[0].any(axis=1)
    #     if mask.sum() == 0:
    #         command = 2 #'FORWARD'
    #     elif planning_all[0, mask][-1][1] <= -2:
    #         command = 0 #'RIGHT' 
    #     elif planning_all[0, mask][-1][1] >= 2:
    #         command = 1 #'LEFT'
    #     else:
    #         command = 2 #'FORWARD'

        
    #     return planning_all, planning_mask_all, command
    

    

    # def generate_meta_action(self,planning_all, planning_mask_all):

    # # """
    # # 根据规划轨迹计算速度变化和转向行为，生成细致的 Meta Action 文本指令。

    # # :param planning_all: 规划轨迹 [(x, y, yaw)]。
    # # :param planning_mask_all: 轨迹有效性掩码。
    # # :return: 文本形式的 Meta Action (例如: "ACCELERATE LEFT TURN")。
    # # """

    # # 提取有效的规划轨迹点
    #     mask = planning_mask_all[0].any(axis=1)
    #     if mask.sum() == 0 :
    #         return "FORWARD" # 如果轨迹点太少，无法计算速度，则判断为直行 保底

    #     # 获取有效轨迹
    #     valid_traj = planning_all[0, mask] # 形状: (n_valid_steps, 3)
    #     if valid_traj.shape[0] < 2:
    #     # 如果轨迹点少于2个，无法计算速度，设置一个默认值（例如 0）
    #         initial_velo = 0.0
    #         final_velo = 0.0
    #         # 或者您也可以在这里打印一个警告，来追踪是哪条数据出了问题
    #         print(f"Warning: Trajectory has only {valid_traj.shape[0]} points, cannot compute initial velocity.")
    #     else:
    #         # 只有在轨迹至少有两个点时，才执行原始计算
    #         initial_velo = np.linalg.norm(valid_traj[1, :2] - valid_traj[0, :2])
    #         final_velo = np.linalg.norm(valid_traj[-1, :2] - valid_traj[-2, :2])
        
    #     constant_eps = 0.5 # 速度阈值 0.5 (m/0.5s)
        
    #     if initial_velo < constant_eps and final_velo < constant_eps:
    #         speed_meta = "STOP"
    #     elif np.abs(final_velo - initial_velo) < constant_eps:
    #         speed_meta = "CONSTANT SPEED"
    #     elif final_velo > initial_velo:
    #         if final_velo > 1.5 * initial_velo: # 终点速度明显大于初始速度
    #             speed_meta = "QUICK ACCELERATION"
    #         else:
    #             speed_meta = "ACCELERATION"
    #     else: # final_velo < initial_velo
    #         if initial_velo > 1.5 * final_velo: # 初始速度明显大于终点速度
    #             speed_meta = "QUICK DECELERATION"
    #         else:
    #             speed_meta = "DECELERATION"
                
    #     # --- 2. 行为分析 (基于终点位置) ---
        
    #     # 获取终点位置 (x, y) #######正常为x 但univ2x为y
    #     end_x, end_y = valid_traj[-1, :2]
            
    #     # 假设在规划终点（3秒后），横向位移超过 2 米视为转向，否则视为直行
    #     straight_threshold = 1.0
    #     turn_threshold = 2.0
        
    #     behavior_meta = ""
        
    #     if speed_meta == "STOP":
    #         behavior_meta = "" # 停车时不谈方向
        
    #     # === 核心方向判定逻辑 ===
    #     elif np.abs(end_y) < straight_threshold: 
    #         # 横向位移极小 (< 0.5m)
    #         behavior_meta = "MOVE FORWARD"
            
    #     elif end_y > 0: # Y > 0 -> 左侧 (Left)
    #         if end_y > turn_threshold:
    #             behavior_meta = "TURN LEFT"      # 大幅度左转 (>2m)
    #         else:
    #             behavior_meta = "LANE CHANGE LEFT" # 小幅度左移 (0.5~2m)
                
    #     else: # Y < 0 -> 右侧 (Right)
    #         if end_y < -turn_threshold:
    #             behavior_meta = "TURN RIGHT"     # 大幅度右转 (<-2m)
    #         else:
    #             behavior_meta = "LANE CHANGE RIGHT" # 小幅度右移 (-0.5~-2m)
    #     # --- 3. 组合指令 ---
    #     if speed_meta == "STOP":
    #         return speed_meta.upper()
    #     else:
    #         # 组合：例如 "ACCELERATION" + "TURN LEFT" -> "ACCELERATION TURN LEFT"
    #         return (speed_meta + " " + behavior_meta).strip().upper()
