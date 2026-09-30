
from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS
from PIL import Image
from nuscenes import NuScenes
import os
import numpy as np
from .emmautils import EstimateCurvatureFromTrajectory, IntegrateCurvatureForPoints
from math import atan2
import random
@TRANSFORMS.register_module()
class LoadMultiViewImageFromFiles4Clip_sensor_corruption(LoadMultiViewImageFromFiles):
    def __init__(self,dataroot=None,sensor_corruption_level=None,**kwargs):
        super(LoadMultiViewImageFromFiles4Clip_sensor_corruption, self).__init__(**kwargs)
        self.sensor_corruption_level=sensor_corruption_level
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
        self.dataroot=dataroot

    def transform(self, results) -> dict:
        token=results['scene_token']
        scene = self.nusc.get('scene', token)
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        front_camera_images = []
        ego_poses = []
        curr_sample_token = first_sample_token
        while True:
            sample = self.nusc.get('sample', curr_sample_token)
            # Get the front camera image of the sample.
            cam_front_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
            front_camera_images.append(Image.open(os.path.join(self.nusc.dataroot, cam_front_data['filename'])).convert("RGB"))
            pose = self.nusc.get('ego_pose', cam_front_data['ego_pose_token'])
            ego_poses.append(pose)
            # Advance the pointer.
            if curr_sample_token == last_sample_token:
                break
            curr_sample_token = sample['next']
        scene_length = len(front_camera_images)
        ego_poses_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]
        ego_poses_world = np.array(ego_poses_world)
        ego_poses_world = self.cor_trajectory(traj=ego_poses_world)
        ego_velocities = np.zeros_like(ego_poses_world)
        ego_velocities[1:] = ego_poses_world[1:] - ego_poses_world[:-1]
        ego_velocities[0] = ego_velocities[1]
        ego_curvatures = EstimateCurvatureFromTrajectory(ego_poses_world)
        i=results['num']
        curr_image = front_camera_images[i]
        obs_ego_velocities = ego_velocities[:i+1]
        obs_ego_curvatures = ego_curvatures[:i+1]
        obs_velocities_norm = np.linalg.norm(obs_ego_velocities, axis=1)
        obs_curvatures = obs_ego_curvatures * 100
        obs_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in zip(obs_velocities_norm, obs_curvatures)]
        obs_speed_curvature_str = ", ".join(obs_speed_curvature_str)
        scene_description=results['scene_description']
        object_description=results['object_description']
        intent_description=results['Intent']
        results['prompt'] = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
        The scene is described as follows: {scene_description}. 
        The identified critical objects are {object_description}. 
        The car's intent is {intent_description}. 
        The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
        Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""        
        results['image']=curr_image
        return results

    def cor_trajectory(self,traj=None):
        traj = np.array(traj)
        if self.sensor_corruption_level==None:
            return traj
        t=0.5
        # 计算位移
        delta = np.diff(traj, axis=0)

        # 计算加速度
        accel = (2 * delta) / t**2
        if self.sensor_corruption_level=='low':
            self.Cxy=0; self.Cxz=0;self.Cyx=0;self.Cyz=0; self.Czx=0;self.Czy=0;self.mean=0;self.std_dev=0.388;self.bias_1=random.uniform(0, 0.1);self.bias_2=random.uniform(0, 0.1);self.bias_3=random.uniform(0, 0.1)
        elif self.sensor_corruption_level=='mid':
            self.Cxy=random.uniform(0, 0.1);self.Cxz=random.uniform(0, 0.1);self.Cyx=random.uniform(0, 0.1);self.Cyz=random.uniform(0, 0.1);self.Czx=random.uniform(0, 0.1);self.Czy=random.uniform(0, 0.1);self.mean=0;self.std_dev=0.388;self.bias_1=random.uniform(0.1, 1);self.bias_2=random.uniform(0.1, 1);self.bias_3=random.uniform(0.1, 1)
        elif self.sensor_corruption_level=='high':
            self.Cxy=random.uniform(0, 1);self.Cxz=random.uniform(0, 1);self.Cyx=random.uniform(0, 1);self.Cyz=random.uniform(0, 1);self.Czx=random.uniform(0, 1);self.Czy=random.uniform(0, 1);self.mean=0;self.std_dev=0.388;self.bias_1=random.uniform(1, 9);self.bias_2=random.uniform(1, 9);self.bias_3=random.uniform(1, 9)
        ###create corruption method
        self.bias_vector = np.array([self.bias_1, self.bias_2, self.bias_3])
        self.noise_vector = np.abs(np.random.normal(self.mean, self.std_dev, 3))
        self.Gauss=np.random.normal(self.mean, self.std_dev, 1)
        self.coefficient_matrix=np.array([
                [1, self.Cxy, self.Cxz],
                [self.Cyx, 1, self.Cyz],
                [self.Czx, self.Czy, 1]
            ])
        # 生成噪声向量
        noise_vector = np.random.normal(self.mean, self.std_dev, 3)
        accel_c = np.dot(accel, self.coefficient_matrix) + self.bias_vector + noise_vector
        accel_c[:, -1] = 0  # 将最后一个维度（z 方向）设置为 0
        # 计算修改后的位移
        move = accel_c * t**2 / 2
        # 反推出修改后的轨迹点
        end_point = traj[-1]
        corruption_trajectory = [end_point]
        for delta in reversed(move):
            prev_point = corruption_trajectory[-1] - delta
            corruption_trajectory.append(prev_point)
        # 反转轨迹点列表并转换为列表
        corruption_trajectory = np.array(corruption_trajectory[::-1])
        
        return corruption_trajectory