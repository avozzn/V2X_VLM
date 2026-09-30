import numpy as np
import torch
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import BasePoints
from mmdet3d.datasets.transforms import Pack3DDetInputs

@TRANSFORMS.register_module()
class PETRFormatBundle3D(Pack3DDetInputs):
    """
    针对 MMDetection 3.x / MMDet3D 1.x 适配的 PETR 格式化模块。
    移除了对 mmengine.utils.to_tensor 的依赖，直接使用 torch 原生转换。
    """

    def __init__(self, 
                 class_names, 
                 collect_keys, 
                 with_gt=True, 
                 with_label=True,
                 task_2d=False):
        # 在 3.x 中，Pack3DDetInputs 会自动处理 'img' 
        # 我们通过 super().__init__ 告诉它还要额外打包哪些自定义键到 inputs 中
        super(PETRFormatBundle3D, self).__init__(keys=['img'] + collect_keys)
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label
        self.collect_keys = collect_keys
        self.task_2d = task_2d

    def transform(self, results):
        """
        数据流水线的核心转换逻辑。
        """
        # 1. 处理点云数据 (BasePoints -> Tensor)
        if 'points' in results:
            if isinstance(results['points'], BasePoints):
                results['points'] = results['points'].tensor
            elif not isinstance(results['points'], torch.Tensor):
                results['points'] = torch.as_tensor(results['points'])

        # 2. 处理 collect_keys 中的数据 (如 coords, timestamps 等)
        for key in self.collect_keys:
            if key in results:
                # 时间戳使用 float64 保证精度，其余默认 float32
                dtype = torch.float64 if key in ['timestamp', 'img_timestamp'] else torch.float32
                # 使用 torch.as_tensor 替代 to_tensor，更安全且无依赖
                results[key] = torch.as_tensor(np.array(results[key]), dtype=dtype)
        
        # 3. 处理车道线点 (针对 PETR 扩展)
        if 'lane_pts' in results:
            results['lane_pts'] = torch.as_tensor(np.array(results['lane_pts']), dtype=torch.float32)

        # 4. 处理 VLM (多模态) 相关输入
        for key in ['input_ids', 'vlm_labels']:
            if key in results:
                # 如果是 list 则保持，如果是 array 则转 Tensor
                if isinstance(results[key], (np.ndarray, list)):
                    results[key] = torch.as_tensor(results[key])

        # 5. 处理 Ground Truth (适配 3.x 的标签映射逻辑)
        if self.with_gt:
            # 5.1 过滤无效物体 (Masking)
            if 'gt_bboxes_3d_mask' in results:
                mask = results['gt_bboxes_3d_mask']
                # 确保 mask 是布尔索引
                if isinstance(mask, np.ndarray):
                    mask = mask.astype(bool)
                
                for key in ['gt_bboxes_3d', 'gt_names_3d', 'centers2d', 'depths', 'gt_names']:
                    if key in results:
                        results[key] = results[key][mask]

            # 5.2 类别名称转 ID (Label Mapping)
            if self.with_label:
                if 'gt_names_3d' in results:
                    results['gt_labels_3d'] = torch.tensor([
                        self.class_names.index(n) for n in results['gt_names_3d']
                    ], dtype=torch.long)
                
                if 'gt_names' in results:
                    results['gt_labels'] = torch.tensor([
                        self.class_names.index(n) for n in results['gt_names']
                    ], dtype=torch.long)

        # 6. 调用父类 Pack3DDetInputs 完成最终打包
        # 父类会将 results['img'] 和 results[collect_keys] 放入 'inputs'
        # 将标注信息（如 gt_bboxes_3d）放入 'data_samples'
        packed_results = super(PETRFormatBundle3D, self).transform(results)
        
        return packed_results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(class_names={self.class_names}, '
        repr_str += f'collect_keys={self.collect_keys}, with_gt={self.with_gt}, with_label={self.with_label})'
        return repr_str