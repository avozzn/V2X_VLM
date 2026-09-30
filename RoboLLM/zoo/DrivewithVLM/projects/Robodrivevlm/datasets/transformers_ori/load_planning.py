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
from .Templates import System_prompt,Planning 
@TRANSFORMS.register_module()
class Load_planning_prompt(BaseTransform):
    def __init__(self):
        super(Load_planning_prompt, self).__init__()

    def transform(self, results):


        
        
        results['conversations'][0]=results['conversations'][0]+Planning[0]
        results['conversations'][1]=results['conversations'][1]+results['QA_pairs'][1]
        print("conversations:",results['conversations'])

        return results
    
@TRANSFORMS.register_module()
class Load_planning_prompt_test(BaseTransform):
    def __init__(self):
        super(Load_planning_prompt_test, self).__init__()

    def transform(self, results):


        results['conversations'][0]=results['conversations'][0]+Planning[0]

        results['label']=results['conversations'][1]+results['QA_pairs'][1]
        results['conversations'][1]=''
        return results