
from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS
from PIL import Image
from mmcv.transforms.base import BaseTransform
import numpy as np
import sys
sys.path.append('//home/ldc/Projects/RoboLLM/corruptions')

from image_corruption.Camera_Method import CameraMethods
@TRANSFORMS.register_module()
class image_corruption(BaseTransform):
    def __init__(self,corruption_severity_dict=
            {
                'sun_sim':5,
            }):

            self.corruption=CameraMethods(corruption_severity_dict=corruption_severity_dict)
            

        
    def transform(self, results) -> dict:
        image_inputs=results['front_camera_images']

        image_inputs=[np.array(img) for img in image_inputs]

        image_inputs=self.corruption(image_inputs)

        image_inputs=[Image.fromarray(img) for img in image_inputs ]

        results['front_camera_images']=image_inputs

        for i, image in enumerate(image_inputs):

            image.save(f"vis/output_{i}.jpg") 

        return results
    


@TRANSFORMS.register_module()
class image_corruption_fog(BaseTransform):
    def __init__(self,corruption_severity_dict=
            {
                'fog_sim':5,
            }):

            self.corruption=CameraMethods(corruption_severity_dict=corruption_severity_dict)
            

        
    def transform(self, results) -> dict:
        image_inputs=results['front_camera_images']

        image_inputs=[np.array(img) for img in image_inputs]

        image_inputs=self.corruption(image_inputs)

        image_inputs=[Image.fromarray(img) for img in image_inputs ]

        results['front_camera_images']=image_inputs

        for i, image in enumerate(image_inputs):

            image.save(f"vis/dark/output_{i}.jpg") 

        return results


@TRANSFORMS.register_module()
class image_corruption_snow(BaseTransform):
    def __init__(self,corruption_severity_dict=
            {
                'snow_sim':5,
            }):

            self.corruption=CameraMethods(corruption_severity_dict=corruption_severity_dict)
            

        
    def transform(self, results) -> dict:
        image_inputs=results['front_camera_images']

        image_inputs=[np.array(img) for img in image_inputs]

        image_inputs=self.corruption(image_inputs)

        image_inputs=[Image.fromarray(img) for img in image_inputs ]

        results['front_camera_images']=image_inputs

        for i, image in enumerate(image_inputs):

            image.save(f"vis/snow/output_{i}.jpg") 

        return results


@TRANSFORMS.register_module()
class image_corruption_sun(BaseTransform):
    def __init__(self,corruption_severity_dict=
            {
                'sun_sim':5,
            }):

            self.corruption=CameraMethods(corruption_severity_dict=corruption_severity_dict)
            

        
    def transform(self, results) -> dict:
        image_inputs=results['front_camera_images']

        image_inputs=[np.array(img) for img in image_inputs]

        image_inputs=self.corruption(image_inputs)

        image_inputs=[Image.fromarray(img) for img in image_inputs ]

        results['front_camera_images']=image_inputs

        for i, image in enumerate(image_inputs):

            image.save(f"vis/sun/output_{i}.jpg") 

        return results