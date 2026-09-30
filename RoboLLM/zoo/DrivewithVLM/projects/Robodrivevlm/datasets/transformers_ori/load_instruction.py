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
from .Templates import System_prompt,Instruction,COT,Scenario_Description,COT_2
@TRANSFORMS.register_module()
class Load_Instruction(BaseTransform):
        
    def transform(self, results):

        results['system_prompt'] = results['system_prompt']+Instruction[0]

        return results
        
@TRANSFORMS.register_module()       
class Load_COT(BaseTransform):
    def __init__(self,task='Detection'):
        self.task=task
        
    
        
    def transform(self, results):
        
        if self.task== 'Detection':
            results['conversations'][0] = results['conversations'][0]+COT[0]
        if self.task=='Scene':
            results['conversations'][0] = results['conversations'][0]+COT_2[0]

        return results


@TRANSFORMS.register_module()  
class Load_Scene(BaseTransform):
        
    def transform(self, results):

        results['conversations'][0] = results['conversations'][0]+Scenario_Description[0]

        return results