
from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS
from PIL import Image
from mmcv.transforms.base import BaseTransform

import torch

import copy

@TRANSFORMS.register_module()
class Load_vqa_prompt(BaseTransform):
    def __init__(self, key):
        # 确保 self.key 是一个字符串而不是列表
        if isinstance(key, list) and len(key) == 1:
            self.key = key[0]
        else:
            self.key = key



    def transform(self, results):
                # 从 results 中提取问答对
        results['conversations']=results[self.key]

        return results





def save_results_to_txt(results, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
