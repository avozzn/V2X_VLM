# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Union

from mmdet3d.datasets import NuScenesDataset

from mmdet3d.registry import DATASETS
import json
import copy

@DATASETS.register_module()
class NuScenesMMDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict]): Pipeline used for data processing.
            Defaults to [].
        test_mode (bool): Store `True` when building test or val dataset.
    """
    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 json_file:str=None,
                 blacklist: str=None,
                 task:str = None,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 **kwargs) -> None:
        self.json_file=json_file
        self.blacklist=blacklist
        self.length=0
        self.task=task
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self):
        # 调用父类的方法，获取初步解析结果
        data_list = super().load_data_list()
        print("data_list:",data_list) ######
        if self.task == 'planning' or self.task == 'vqa':
            data_list=self.process_json_file(data_list, self.json_file)
        if self.task !='vqa':
            data_list =self.filter_empty_sweep(data_list)
        self.length=len(data_list)
        return data_list

    def process_json_file(self, data_infos, json_file):
     
        """处理 json_file 并将相关信息添加到 data_infos 中."""
        print(f"Processing JSON file: {json_file}")
        with open(json_file, 'r') as json_file:
            json_data = json.load(json_file)
        # 提取 JSON 中的 image 和 conversations 数据
        cam_front_qa_pairs = {}
        for item_group in json_data:
            # 从 conversations 中提取 Q&A 对
            if 'conversations' in item_group:
                try:
                    qa_info=[]
                    question = item_group['conversations'][0]['value']
                    answer = item_group['conversations'][1]['value']
                    qa_info.append(question)
                    qa_info.append(answer)
                except (IndexError, KeyError) as e:
                    print(f"Error extracting Q&A pair: {e}")
                    continue

            # 提取 CAM_FRONT 路径
            cam_front_path = None
            if 'image' in item_group:
                for image_path in item_group['image']:
                    if 'CAM_FRONT' in image_path:
                        cam_front_path = image_path
                        break  
            
            # 同时存在 CAM_FRONT 路径和 Q&A 对时，将其存储
            if cam_front_path and qa_info:
                if cam_front_path not in cam_front_qa_pairs:
                    cam_front_qa_pairs[cam_front_path] = []
                cam_front_qa_pairs[cam_front_path].append(qa_info)
        # 过滤 data_infos 并附加处理后的 Q&A 数据
        filtered_data_infos = []
        for info in data_infos:
            if 'CAM_FRONT' in info['images']['CAM_FRONT']['img_path']:

                cam_front_path = 'data/nuscenes/samples/CAM_FRONT/'+info['images']['CAM_FRONT']['img_path']
                
                if cam_front_path in cam_front_qa_pairs:
                    # 将 Q&A 对添加到 info，并生成新的条目
                    for qa_pair in cam_front_qa_pairs[cam_front_path]:
                        new_info = info.copy()  # 复制当前 info 条目
                        new_info['QA_pairs'] = qa_pair  # 添加 Q&A 对
                        new_info['input_images']=[]
                        filtered_data_infos.append(new_info)  # 加入结果列表

        assert filtered_data_infos, "List should not be empty"
            
       
        return filtered_data_infos


    def filter_empty_sweep(self,filtered_data_infos):
        empty_sweeps_count = 0
        black_list_count = 0
        blacklist_path = self.blacklist
        with open(blacklist_path, 'r') as f:
            blacklist_tokens = json.load(f)

        # 创建一个新的列表来存储非空'sweeps'且不在黑名单中的数据
        filtered_data_infos_non_empty = []

        # 遍历'filtered_data_infos'中的每一项，检查'sweeps'是否为空和是否在黑名单中
        for num, info in enumerate(filtered_data_infos):
            if info['token'] in blacklist_tokens:
                
                black_list_count += 1 
                continue 

            if not info.get('lidar_sweeps'):  
                
                empty_sweeps_count += 1  
            else:
                # 如果'sweeps'不为空且不在黑名单中，将该项添加到新列表中
                filtered_data_infos_non_empty.append(info)

        # 输出'sweeps'为空的总数量
        print(f"Total number of empty 'sweeps': {empty_sweeps_count}")
        print(f"Total number of black token: {black_list_count}")
        # 更新'filtered_data_infos'为只包含非空'sweeps'且不在黑名单中的数据
        filtered_data_infos = filtered_data_infos_non_empty
        # 在处理结束时输出 filtered_data_infos 的数据对数量
        print(f"Total number of data pairs in filtered_data_infos: {len(filtered_data_infos)}")

        return filtered_data_infos
    

    def __len__(self) -> int:
        return self.length
    
    def prepare_data(self, index: int) -> Union[dict, None]:
        """Data preparation for both training and testing stage.

        Called by `__getitem__`  of dataset.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict or None: Data dict of the corresponding index.
        """
        ori_input_dict = self.get_data_info(index)

        # deepcopy here to avoid inplace modification in pipeline.
        input_dict = copy.deepcopy(ori_input_dict)

        # box_type_3d (str): 3D box type.
        input_dict['box_type_3d'] = self.box_type_3d
        # box_mode_3d (str): 3D box mode.
        input_dict['box_mode_3d'] = self.box_mode_3d

        input_dict['conversations']=["",""]

        
        
        
        # pre-pipline return None to random another in `__getitem__`
        if not self.test_mode and self.filter_empty_gt:
            if len(input_dict['ann_info']['gt_labels_3d']) == 0:
                return None
      




        example = self.pipeline(input_dict)
        return example


