import json
from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS
from PIL import Image
from mmcv.transforms.base import BaseTransform
import numpy as np
import sys
import os.path as osp
# sample_json_file = '/home/ldc/Projects/RoboLLM/zoo/MMDrive/data/nuscenes/v1.0-trainval/sample.json'
# scene_json_file = '/home/ldc/Projects/RoboLLM/zoo/MMDrive/data/nuscenes/v1.0-trainval/scene.json'

@TRANSFORMS.register_module()
class Load_vct(BaseTransform):
    def __init__(self,sample_json_file=None,scene_json_file=None,**kwargs):
        super(Load_vct, self).__init__(**kwargs)
        self.sample_json_file=sample_json_file
        self.scene_json_file=scene_json_file
    
    def transform(self, results):
        sample_token = results['token']
        
        with open(self.sample_json_file, 'r') as f:
            sample_data = json.load(f)
        
        # 2. 从 sample.json 查找对应的 scene_token
        scene_token = None
        for sample in sample_data:
            if sample['token'] == sample_token:
                scene_token = sample['scene_token']
                break
        
        if scene_token is None:
            raise ValueError(f"Token {sample_token} not found in {self.sample_json_file}")

        # 3. 加载 scene.json
        with open(self.scene_json_file, 'r') as f:
            scene_data = json.load(f)

        # 4. 在 scene.json 中查找对应的 name
        scene_name = None
        for scene in scene_data:
            if scene['token'] == scene_token:
                scene_name = scene['name']
                break
        
        if scene_name is None:
            raise ValueError(f"Scene with token {scene_token} not found in {self.scene_json_file}")
        
        
        scene_file = f"/home/ldc/Projects/RoboLLM/zoo/MMDrive/projects/Openemma/prepare/VCT/{scene_name}.json"
    
        if not osp.exists(scene_file):
            raise FileNotFoundError(f"The scene file {scene_file} does not exist.")
        
        with open(scene_file, "r") as f:
            all_data= json.load(f)
        sample_data = all_data[sample_token]
        
        obs_ego_velocities = np.array(sample_data["past_ego_velocities"])
        obs_ego_curvatures = np.array(sample_data["past_ego_curvatures"])
        fut_ego_velocities = np.array(sample_data["future_ego_velocities"])
        fut_ego_curvatures = np.array(sample_data["future_ego_curvatures"])
        obs_ego_trajectory = np.array(sample_data["past_ego_trajectory"])
        fut_ego_trajectory = np.array(sample_data["future_ego_trajectory"])
        
        # 计算速度的范数
        obs_velocities_norm = np.linalg.norm(obs_ego_velocities, axis=1)
        fut_velocities_norm = np.linalg.norm(fut_ego_velocities, axis=1)


        # 将曲率乘以 100
        obs_ego_curvatures *= 100
        fut_ego_curvatures *= 100

        # 格式化 obs_speed_curvature_str
        obs_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in zip(obs_velocities_norm, obs_ego_curvatures)]
        obs_speed_curvature_str = ", ".join(obs_speed_curvature_str)

        # 格式化 fut_speed_curvature_str
        fut_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in zip(fut_velocities_norm, fut_ego_curvatures)]
        fut_speed_curvature_str = ", ".join(fut_speed_curvature_str)

        obs_ego_trajectory_str = [f"[{x[0]:.2f},{x[1]:.2f}]" for x in obs_ego_trajectory]
        obs_ego_trajectory_str = ", ".join(obs_ego_trajectory_str)

        fut_ego_trajectory_str = [f"[{x[0]:.2f},{x[1]:.2f}]" for x in fut_ego_trajectory]
        fut_ego_trajectory_str = ", ".join(fut_ego_trajectory_str)
        
        results['obs_speed_curvature_str'] = obs_speed_curvature_str
        results['fut_speed_curvature_str'] = fut_speed_curvature_str
        results['fut_ego_trajectory_str']  = fut_ego_trajectory_str
        
        return results



# # 从 JSON 文件加载数据


# # 选择一个 sample_token，这里假设我们使用第一个样本 token
# sample_token = "c8ae3fcd7e9c48418c7a3b3c5df524cd"
# sample_data = data[sample_token]

# # 提取相关数据
# obs_ego_velocities = np.array(sample_data["past_ego_velocities"])
# obs_ego_curvatures = np.array(sample_data["past_ego_curvatures"])
# fut_ego_velocities = np.array(sample_data["future_ego_velocities"])
# fut_ego_curvatures = np.array(sample_data["future_ego_curvatures"])
# obs_ego_trajectory = np.array(sample_data["past_ego_trajectory"])
# fut_ego_trajectory = np.array(sample_data["future_ego_trajectory"])

# # 计算速度的范数
# obs_velocities_norm = np.linalg.norm(obs_ego_velocities, axis=1)
# fut_velocities_norm = np.linalg.norm(fut_ego_velocities, axis=1)


# # 将曲率乘以 100
# obs_ego_curvatures *= 100
# fut_ego_curvatures *= 100

# # 格式化 obs_speed_curvature_str
# obs_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in zip(obs_velocities_norm, obs_ego_curvatures)]
# obs_speed_curvature_str = ", ".join(obs_speed_curvature_str)

# # 格式化 fut_speed_curvature_str
# fut_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in zip(fut_velocities_norm, fut_ego_curvatures)]
# fut_speed_curvature_str = ", ".join(fut_speed_curvature_str)

# obs_ego_trajectory_str = [f"[{x[0]:.2f},{x[1]:.2f}]" for x in obs_ego_trajectory]
# obs_ego_trajectory_str = ", ".join(obs_ego_trajectory_str)

# fut_ego_trajectory_str = [f"[{x[0]:.2f},{x[1]:.2f}]" for x in fut_ego_trajectory]
# fut_ego_trajectory_str = ", ".join(fut_ego_trajectory_str)

# # 输出结果
# print("obs_speed_curvature_str")
# print(obs_speed_curvature_str)

# print("fut_speed_curvature_str")
# print(fut_speed_curvature_str)

# # 过去轨迹
# print("obs_ego_trajectory")
# print(obs_ego_trajectory_str)

# print("fut_ego_trajectory_str")
# print(fut_ego_trajectory_str)