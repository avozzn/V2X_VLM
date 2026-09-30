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
from .Templates import *


# results['class_dict']
# {
#     'car': '<c0>',
#     'truck': '<c1>',
#     'construction_vehicle': '<c2>',
#     'bus': '<c3>',
#     'trailer': '<c4>',
#     'barrier': '<c5>',
#     'motorcycle': '<c6>',
#     'bicycle': '<c7>',
#     'pedestrian': '<c8>',
#     'traffic_cone': '<c9>'
# }




@TRANSFORMS.register_module()
class Load_detection_prompt(BaseTransform):
    def __init__(self,task_type,class_name):
        super(Load_detection_prompt, self).__init__()

        self.task_type=task_type
        self.class_name=class_name

    def transform(self, results):

        
        Detection = globals()[self.task_type]

        class_dict = {class_name: f"<c{index}>" for index, class_name in enumerate(self.class_name)}

        results['class_dict']=class_dict

        class_str = "{" + ", ".join([f"'{k}': {v}" for k, v in class_dict.items()]) + "}"

        Detection[0] = Detection[0].replace("<class>", class_str)


        
        results['conversations'][0]=results['conversations'][0]+Detection[0]

        return results
    
