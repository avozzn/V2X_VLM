# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Union

from mmdet3d.datasets import NuScenesDataset

from mmdet3d.registry import DATASETS
import json
import copy
from nuscenes import NuScenes
@DATASETS.register_module()
class NuScenesMMDataset_ddp(NuScenesDataset):
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
                 scene_list: str=None,
                 sample_list: str=None,
                 task:str = None,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 **kwargs) -> None:
        self.json_file=json_file
        self.scene_list=scene_list
        self.sample_list=sample_list
        self.length=0
        self.task=task
        self.nusc=NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=True)
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self):
        # 调用父类的方法，获取初步解析结果
        data_list = super().load_data_list()
        if self.task == 'planning' or self.task == 'vqa':
            data_list=self.process_json_file(data_list, self.json_file,self.scene_list,self.sample_list)
        self.length=len(data_list)
        return data_list

    def process_json_file(self, data_infos, json_file,scene_list,sample_list):
        filtered_data_infos=[]
        """处理 json_file 并将相关信息添加到 data_infos 中."""
        print(f"Processing JSON file: {json_file}")
        with open(scene_list, 'r') as scene_list:
            scene_list = json.load(scene_list)
        with open(sample_list,'r') as sample_list:
            sample_list=json.load(sample_list)
        data_list = []
        for scene_token in scene_list:
            scene=self.nusc.get('scene', scene_token)
            first_sample_token = scene['first_sample_token']
            last_sample_token = scene['last_sample_token']
            name = scene['name']
            if name in ["scene-0103", "scene-1077"]:
                continue
            sample_token=[]
            curr_sample_token = first_sample_token
            while True:
                sample_token.append(curr_sample_token)
                sample = self.nusc.get('sample', curr_sample_token)
                if curr_sample_token == last_sample_token:
                    break
                curr_sample_token = sample['next']
            scene_length = len(sample_token)
            for i in range(1, scene_length-1):
                sample_token_cur=sample_token[i]
                data=None
                for item in sample_list:
                    if item['token']==sample_token_cur:
                        data=item
                        break
                if data==None:
                    continue
                Intent=None
                if i>1:
                    for item in sample_list:
                        if item['token']==sample_token[i-1]:
                            Intent=item['intent']
                            break
                data_list.append({'sample_token':data['token'],'scene_description':data['scene_description'],'object_description':data['object_description'],'Intent':Intent,'num':i,'scene_token':scene_token})
        return data_list



    def filter_empty_sweep(self,filtered_data_infos):
        empty_sweeps_count = 0
        
        blacklist_path = self.blacklist
        with open(blacklist_path, 'r') as f:
            blacklist_tokens = json.load(f)

        # 创建一个新的列表来存储非空'sweeps'且不在黑名单中的数据
        filtered_data_infos_non_empty = []

        # 遍历'filtered_data_infos'中的每一项，检查'sweeps'是否为空和是否在黑名单中
        for num, info in enumerate(filtered_data_infos):
            if info['token'] in blacklist_tokens:
                
                empty_sweeps_count += 1 
                continue 

            if not info.get('lidar_sweeps'):  
                
                empty_sweeps_count += 1  
            else:
                # 如果'sweeps'不为空且不在黑名单中，将该项添加到新列表中
                filtered_data_infos_non_empty.append(info)

        # 输出'sweeps'为空的总数量
        print(f"Total number of empty 'sweeps': {empty_sweeps_count}")

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

        
        
        
        # pre-pipline return None to random another in `__getitem__`
      
        example = self.pipeline(input_dict)
        return example


    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if not self._fully_initialized:
            print(
                'Please call `full_init()` method manually to accelerate '
                'the speed.',
                logger='current',
                )
            self.full_init()

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data
        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue
            return data
        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')