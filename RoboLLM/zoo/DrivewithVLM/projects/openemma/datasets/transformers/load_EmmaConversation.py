import json
from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS
from PIL import Image
from mmcv.transforms.base import BaseTransform
import numpy as np
import sys
from .Templates  import System_prompt

@TRANSFORMS.register_module()
class Load_EmmaConversation(BaseTransform):
    
    def transform(self, results):
        
        
        system_prompt=System_prompt[0]
        results['question']=[]
        results['question'].append("You are a autonomous driving labeller. You have access to these front-view camera images of a car taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Provide a concise description of the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings.")
        results['question'].append("You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. What other road users should you pay attention to in the driving scene? List two or three of them, specifying its location within the image of the driving scene and provide a short description of the that road user on what it is doing, and why it is important to you.")
        obs_speed_curvature_str=results['obs_speed_curvature_str']
        Cot_prompt=f'''The scene is described as follows: {scene_description}.\nThe identified critical objects are {object_description}.\n The car's intent is {intent_description}.\n The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}.\n Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:'''
        results['question'].append(results['system_prompt'] +'\n' + Cot_prompt)
        return results

