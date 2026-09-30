from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS
from PIL import Image
from nuscenes import NuScenes
import time
from mmcv.transforms.base import BaseTransform
import numpy as np
import ast
import random
@TRANSFORMS.register_module()
class LoadMultiViewImageFromFiles4Clip_sensor_corruption(LoadMultiViewImageFromFiles):
    def __init__(self,dataroot=None,sensor_corruption_level=None,**kwargs):
        super(LoadMultiViewImageFromFiles4Clip_sensor_corruption, self).__init__(**kwargs)
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
        self.sensor_corruption_level=sensor_corruption_level
        self.dataroot=dataroot
        
    def transform(self, results) -> dict:
        token=results['token']
        current_sample = self.nusc.get('sample', token)
        prev_token=current_sample['prev']
        prev_sample=self.nusc.get('sample',prev_token)
        prev_prev_token=prev_sample['prev']
        prev_prev_sample=self.nusc.get('sample',prev_prev_token)
        current_image=self.nusc.get('sample_data', current_sample['data']['CAM_FRONT'])['filename']
        prev_image=self.nusc.get('sample_data', prev_sample['data']['CAM_FRONT'])['filename']
        prev_prev_image=self.nusc.get('sample_data', prev_prev_sample['data']['CAM_FRONT'])['filename']
        input_images=[]
        image_sources=[]
        image_sources.append(self.dataroot+'/'+current_image)
        image_sources.append(self.dataroot+'/'+prev_image)
        image_sources.append(self.dataroot+'/'+prev_prev_image)
        for image_path in image_sources:
            input_images.append(
                Image.open(image_path).convert("RGB")
                )
        results['input_images']=input_images
        results['questions']=[]
        results['answers']=[]
        for context in results['qa_list']:
            if context['role']=='user':
                results['questions'].append(context['content'][0]['text'])
            else:
                results['answers'].append(context['content'][0]['text'])
        results['questions'].append('Based on the above decisions and analysis, please plan the 3s future trajectory of the ego vehicle.')
        results['questions'][3]=self.sensor_corruption(question=results['questions'][3])
        return results
    
    def sensor_corruption(self,question):
        question='Ego-States:\nHistorical Trajectory (last 2 seconds): [(0.00,0.07), (0.00,0.05), (0.00,0.03), (0.00,0.00)]\nNavigation:MAINTAIN.\nPlease make a summary of the current scene based on the information provided and your previous analysis.'
        traj=question.split("Historical Trajectory (last 2 seconds): ")[1].split("\n")[0]
        traj = ast.literal_eval(traj)
        corrupution_trajectory= self.cor_trajectory(traj=traj)
        formatted_traj = ", ".join([f"({x:.2f},{y:.2f})" for x, y in corrupution_trajectory])
        new_traj_str = f"[{formatted_traj}]"
        prefix, suffix = question.split("Historical Trajectory (last 2 seconds): ")
        new_text = prefix + f"Historical Trajectory (last 2 seconds): {new_traj_str}\n" + suffix.split("\n", 1)[1]
        return new_text
    def cor_trajectory(self,traj=None):
        traj = np.array(traj)
        traj=[(x, y, 0.0) for x, y in traj]
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
        corruption_trajectory=np.array(corruption_trajectory)[:, :2]
        # 反转轨迹点列表并转换为列表
        corruption_trajectory = np.array(corruption_trajectory[::-1])
        
        return corruption_trajectory