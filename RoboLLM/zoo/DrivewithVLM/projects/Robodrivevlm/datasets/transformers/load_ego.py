from mmdet3d.registry import TRANSFORMS
from PIL import Image
from mmcv.transforms.base import BaseTransform
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from PIL import Image
from typing import  Union
from transformers.utils import logging
logger = logging.get_logger(__name__)
Number = Union[int, float]
from .Templates  import System_prompt,System_prompt_no_image,System_prompt_no_sensor
import numpy as np
import random
@TRANSFORMS.register_module()
class Load_EGO(BaseTransform):
    def __init__(self,load_image=True,load_sensor=True):
        self.load_image=load_image
        self.load_sensor=load_sensor

    def transform(self, results):
        v0 = results['v0']
        acc_x=results['acc_x']
        acc_y=results['acc_y']
        can_bus = results['can_bus']
        pos_x, pos_y, pos_z = can_bus[0], can_bus[1], can_bus[2]
        if self.load_image:
            infra_pixel = results['infra_ego_pixel']
            
            # curvature (positive: turn left)
            # steering = results['steering']
            is_visible = results['is_visible']
            if is_visible and infra_pixel is not None:
                # 情况 A：自车可见
                try:
                    u, v = infra_pixel
                    ego_message = f"- Ego-Vehicle Position in Roadside Camera View: Visible at pixel coordinates ({u}, {v}).\n"
                except (ValueError, TypeError):
                    # 防止 infra_pixel 格式不对导致崩溃
                    ego_message = "- Ego-Vehicle Position in Roadside Camera View: Visible (Coordinates unavailable).\n"
            
            else: 
                # 情况 B：自车不可见
                # 检查 infra_pixel 是否意外存储了深度值 (z)，如果是数字则打印，否则只说不可见
                if isinstance(infra_pixel, (int, float)):
                    ego_message = f"- Ego-Vehicle Position in Roadside Camera View: Not Visible (Behind Camera, Depth z={infra_pixel:.1f}m).\n"
                else:
                    ego_message = "- Ego-Vehicle Position in Roadside Camera View: Not Visible (Out of frame or Occluded).\n"
            # ego_message += f" - Acceleration (ax,ay): ({acc_x:.2f},{acc_y:.2f})"
            # ego_message = f" - Current Position(in Global Coordinates)(x, y, z): ({pos_x:.2f}, {pos_y:.2f}, {pos_z:.2f})"
        
            # # # ego_message += f" - Steering: ({steering:.2f})"
            # results['system_prompt'] = results['system_prompt'] + results['QA_pairs'][0]
            results['system_prompt'] = results['system_prompt'] + results['QA_pairs'][0]+ ego_message
        else:
            results['system_prompt'] = results['system_prompt'] + results['QA_pairs'][0]
        # with open(f"//home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM/vis/visualization_train/visualization_train.txt", "a") as file:
        #     file.write(ego_message + "\n")
        return results
    
    def locate_message(self,utimes, utime):
        i = np.searchsorted(utimes, utime)
        if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
            i -= 1
        return i
    
# class Load_EGO(BaseTransform):
#     def __init__(self,load_sensor=True):
#         self.load_sensor=load_sensor

#     def transform(self, results):
#         results['system_prompt'] = results['system_prompt']+results['QA_pairs'][0]
#         return results

    
@TRANSFORMS.register_module()
class Load_EGO_corruption(BaseTransform):
    def __init__(self,load_image=True,load_sensor=True,corruption_rate='none',corruption_type=[]):
        self.load_image=load_image
        self.load_sensor=load_sensor

    def transform(self, results):
        
        infos=results
        pose_msgs = infos['pose_msgs']
        steer_msgs = infos['steer_msgs']
        try:
            # 提取时间戳列表
            pose_uts = [msg['utime'] for msg in pose_msgs]
            steer_uts = [msg['utime'] for msg in steer_msgs]
            ref_utime = infos['timestamp']*1000000
            # print(pose_uts)
            # print(ref_utime)
            # 尝试找到最近的pose_index
            pose_index = self.locate_message(pose_uts, ref_utime)
            # print(pose_index)
        except Exception as e:
            # 输出pose_uts和ref_utime，并打印异常信息
            print(f"Error occurred in locate_message:\npose_uts: {pose_uts}\nref_utime: {ref_utime}")
            print(f"Exception: {e}")
        pose_data = pose_msgs[pose_index]
        steer_index = self.locate_message(steer_uts, ref_utime)
        steer_data = steer_msgs[steer_index]
        v0 = pose_data["vel"][0]  # [0] means longitudinal velocity  m/s
        acc_x=pose_data['accel'][0]
        acc_y=pose_data['accel'][1]
        # curvature (positive: turn left)
        steering = steer_data["value"]
        if 'speed' in self.corruption_type:
            if self.corruption_rate=='low':
                speed_corruption_rate=random.uniform(0,5)
            elif self.corruption_rate=='medium':
                speed_corruption_rate=random.uniform(5,10)
            elif self.corruption_rate=='high':
                speed_corruption_rate=random.uniform(10,20)
            else:
                speed_corruption_rate=1
        if 'steering' in self.corruption_type:
            steering=random.uniform(-7.7,6.3)
        v0=v0*speed_corruption_rate
        acc_x=acc_x*speed_corruption_rate
        acc_y=acc_y*speed_corruption_rate
        steering=-1 * steering / 2.588
        ego_message = f"- Heading Speed: ({v0:.2f})"
        ego_message += f" - Acceleration (ax,ay): ({acc_x:.2f},{acc_y:.2f})"
        ego_message += f" - Steering: ({steering:.2f})"
        results['system_prompt'] = results['system_prompt']+ego_message
        return results
    
    def locate_message(self,utimes, utime):
        i = np.searchsorted(utimes, utime)
        if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
            i -= 1
        return i