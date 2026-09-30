
from mmdet3d.datasets.pipelines import LoadMultiViewImageFromFiles
from mmdet.datasets.builder import PIPELINES
from PIL import Image
from mmdet3d.datasets.pipelines import DefaultFormatBundle
import numpy as np
import sys
sys.path.append('//home/ldc/Projects/RoboLLM/corruptions')

from image_corruption.Camera_Method import CameraMethods
@PIPELINES.register_module()
class image_corruption(DefaultFormatBundle):
    def __init__(self,corruption_severity_dict=
            {
                'sun_sim':5,
            }):

            self.corruption=CameraMethods(corruption_severity_dict=corruption_severity_dict)
            print("corruption_severity_dict",corruption_severity_dict)

        
    def __call__(self, results) -> dict:
        image_inputs=results['img']
         # 转换为numpy数组列表（输入验证）
        image_inputs = [np.array(img) for img in image_inputs]
        # print("===== 输入图像信息 =====")
        # # for i, img in enumerate(image_inputs):
        #     print(f"图像 {i}:")
        #     print(f"  形状: {img.shape}")       # 期望 (H, W, 3)
        #     print(f"  类型: {img.dtype}")       # 期望 uint8
        #     print(f"  像素范围: {np.min(img)}-{np.max(img)}")  # 期望 0-255
        #     print("-" * 40)

        image_inputs=[np.array(img) for img in image_inputs]
        # print("===== 腐蚀后图像信息 =====")
        # for i, img in enumerate(image_inputs):
        #     print(f"图像 {i}:")
        #     print(f"  形状: {img.shape}")       # 期望 (H, W, 3)
        #     print(f"  类型: {img.dtype}")       # 期望 uint8
        #     print(f"  像素范围: {np.min(img)}-{np.max(img)}")  # 期望 0-255
        #     print("-" * 40)

        image_inputs=self.corruption(image_inputs)
        
        image_inputs=[Image.fromarray(img) for img in image_inputs ]

        results['img']=image_inputs
       
        for i, image in enumerate(image_inputs):

            image.save(f"vis/output_{i}.jpg") 
             
        return results