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

import numpy as np


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



def filter_bboxes(bboxes, threshold=30):
    selected_bboxes = []
    for bbox in bboxes:
        x_center, x_size, y_center, y_size,yaw,label = bbox
        
        # 计算中心点到原点的距离
        distance = np.sqrt(x_center**2 + y_center**2)
        
        # 如果距离小于或等于30米，则保留该bbox
        if distance <= threshold:
            selected_bboxes.append(bbox)
    
    return selected_bboxes

@TRANSFORMS.register_module()
class Load_detection_gt(BaseTransform):
    def __init__(self):
        super(Load_detection_gt, self).__init__()

    def transform(self, results):

        labels=results['ann_info']['gt_bboxes_labels']
        
        bbox_3d=results['ann_info']['gt_bboxes_3d'].tensor.numpy()[:, :7]

        bbox_3d = bbox_3d[:, [0, 3, 1, 4, 6]]

        bbox_3d = np.hstack((bbox_3d, labels.reshape(-1, 1)))

        condition = (bbox_3d[:, 0] > 0)

        bbox_3d = bbox_3d[condition]

        bbox_3d = filter_bboxes(bbox_3d, threshold=10)

        bbox_3d= np.round(bbox_3d, 1)

        if bbox_3d.size==0:
            gt='There are no objects within ten meters ahead belonging to <class>.' 
        else :
            gt=convert_str(bbox_3d)


        results['conversations'][1]=results['conversations'][1]+gt+'\n'

        
        return results
    


def convert_str(bbox_3d):

        
    # 组合成元组，并将类别替换为 <cX> 格式

    bev_bboxes = [
    (row[0],row[1],row[2],row[3],row[4],f"<c{int(row[5])}>")
    for row in bbox_3d
    ]
    bev_bboxes_text="BEV Bounding Box:\n" + str(bev_bboxes)
    return bev_bboxes_text