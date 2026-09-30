import numpy as np
import sys
from mmdet.datasets.builder import PIPELINES
sys.path.append('//home/ldc/Projects/RoboLLM/corruptions')

from image_corruption.Camera_Method import CameraMethods

@PIPELINES.register_module()
class image_corruption(object):
    def __init__(self, corruption_severity_dict=
    {
        'sun_sim': 5,
    }):
        self.corruption = CameraMethods(corruption_severity_dict=corruption_severity_dict)

    def  __call__(self, results) -> dict:
        image_inputs = results['img']

        image_inputs = [np.array(img) for img in image_inputs]

        image_inputs = self.corruption(image_inputs)

        image_inputs = [Image.fromarray(img) for img in image_inputs]

        results['img'] = image_inputs

        print("corruption_severity_dict",self.corruption)

        for i, image in enumerate(image_inputs):
            image.save(f"vis/output/{i}.jpg")

        return results

    def __repr__(self):
        """Return a string representation of the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(corruption_severity_dict={self.corruption.severity_dict})"
        return repr_str