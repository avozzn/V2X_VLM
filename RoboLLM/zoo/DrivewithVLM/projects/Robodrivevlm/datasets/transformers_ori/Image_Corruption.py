
from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS
from PIL import Image
from mmcv.transforms.base import BaseTransform
import numpy as np
import sys
import os
sys.path.append('//home/ldc/Projects/RoboLLM/corruptions')

from image_corruption.Camera_Method import CameraMethods
@TRANSFORMS.register_module()
class image_corruption(BaseTransform):
    def __init__(self,corruption_severity_dict=
            {
                'sun_sim':5,
            }):
            self.corruption_severity_dict=corruption_severity_dict

            self.corruption=CameraMethods(corruption_severity_dict=corruption_severity_dict)
            self.first_key = next(iter(corruption_severity_dict.keys()))
            self.first_key = next(iter(corruption_severity_dict.keys()))
            self.first_value = corruption_severity_dict[self.first_key]
            print("self.corruption",self.first_key)
            

        
    def transform(self, results) -> dict:
        image_inputs=results['input_images']

        image_inputs=[np.array(img) for img in image_inputs]

        image_inputs=self.corruption(image_inputs)

        image_inputs=[Image.fromarray(img) for img in image_inputs ]

        results['input_images']=image_inputs



        for i, image in enumerate(image_inputs):

            image.save(f"vis/visulization/output_{self.first_key}_{self.first_value}_{i}.jpg") 

        return results