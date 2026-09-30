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
from .Templates import Planning, PlanningPhysics, System_prompt


def planning_prompt(output_contract):
    if output_contract == 'legacy':
        return Planning[0]
    if output_contract == 'physics':
        return PlanningPhysics[0]
    raise ValueError(f'unknown planning output contract: {output_contract}')


@TRANSFORMS.register_module()
class Load_planning_prompt(BaseTransform):
    def __init__(self, output_contract='legacy'):
        super(Load_planning_prompt, self).__init__()
        self.output_contract = output_contract

    def transform(self, results):
        results['conversations'][0] += planning_prompt(self.output_contract)
        results['conversations'][1]=results['conversations'][1]+results['QA_pairs'][1]
        # print("conversations:",results['conversations'])

        return results
    
@TRANSFORMS.register_module()
class Load_planning_prompt_test(BaseTransform):
    def __init__(self, output_contract='legacy'):
        super(Load_planning_prompt_test, self).__init__()
        self.output_contract = output_contract

    def transform(self, results):


        results['conversations'][0] += planning_prompt(self.output_contract)

        return results
