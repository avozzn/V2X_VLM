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
from .Templates  import System_prompt,System_prompt_no_image,System_prompt_no_sensor,System_prompt_no_camera,System_prompt_no_ego,System_prompt_no_text,System_prompt_only_camera,System_prompt_only_ego,System_prompt_only_Lidar
import numpy as np
import random
@TRANSFORMS.register_module()
class Load_system_prompt(BaseTransform):
    def __init__(self,prompt_choose=None):
        self.prompt_choose=prompt_choose
    def transform(self, results):
        if self.prompt_choose=='no_image':
            results['system_prompt']=System_prompt_no_image[0]
        if self.prompt_choose=='no_sensor':
            results['system_prompt']=System_prompt_no_sensor[0]
            results['input_images'] = results['input_images'][:3]
        if self.prompt_choose=='no_camera':
            results['system_prompt']=System_prompt_no_camera[0]
        if self.prompt_choose=='no_ego':
            results['system_prompt']=System_prompt_no_ego[0]
        if self.prompt_choose=='no_text':
            results['system_prompt']=System_prompt_no_text[0]
        if self.prompt_choose=='only_camera':
            results['system_prompt']=System_prompt_only_camera[0]
            results['input_images'] = results['input_images'][:3] 
        if self.prompt_choose=='only_lidar':
            results['system_prompt']=System_prompt_only_Lidar[0]
            results['input_images']=[results['input_images'][3]]
        if self.prompt_choose=='only_ego':
            results['system_prompt']=System_prompt_only_ego[0]
            results['input_images']=[]
        elif self.prompt_choose==None:
            results['system_prompt']=System_prompt[0]
        return results