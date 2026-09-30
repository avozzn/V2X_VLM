# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Union

from mmdet3d.datasets import NuScenesDataset

from mmdet3d.registry import DATASETS
import json
import copy
from nuscenes import NuScenes
import pickle

from projects.mmdet3d_plugin.datasets.data_utils.spd_trajectory_api import SPDTraj
from projects.mmdet3d_plugin.datasets.data_utils.vector_map import VectorizedLocalMap
from projects.mmdet3d_plugin.datasets.eval_utils.map_api import NuScenesMap

#####
@DATASETS.register_module()
class NuScenesMMDatasetV2X(NuScenesDataset):
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
                 v2x_side: str = None,
                 task:str = None,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 **kwargs) -> None:
        self.json_file=json_file
        self.length=0
        self.task=task
        self.v2x_side = v2x_side
        if self.v2x_side not in ['vehicle_side', 'infrastructure_side', 'cooperative', '']:
            raise Exception('v2x_side is not correct with {}'.format(self.v2x_side))
        self.nusc = NuScenes(version=self.version,
                             dataroot=self.data_root, verbose=True)
        self.nusc_maps = {
                    'yizhuang06': NuScenesMap(dataroot=self.data_root, map_name='yizhuang06'),
                    'yizhuang08': NuScenesMap(dataroot=self.data_root, map_name='yizhuang08'),
                    'yizhuang09': NuScenesMap(dataroot=self.data_root, map_name='yizhuang09'),
                    'yizhuang10': NuScenesMap(dataroot=self.data_root, map_name='yizhuang10'),
                    'yizhuang13': NuScenesMap(dataroot=self.data_root, map_name='yizhuang13'),
                    'yizhuang16': NuScenesMap(dataroot=self.data_root, map_name='yizhuang16')
            }
        # 将地图转换为向量格式
        self.vector_map = VectorizedLocalMap(
            self.data_root,
            patch_size=self.patch_size,
            canvas_size=self.canvas_size,
            dataset_type=self.tmp_dataset_type)
        #输出每个物体的未来轨迹、未来 6 秒内轨迹的有效性掩码、和过去两秒历史轨迹和历史轨迹的有效性掩码
        self.traj_api = SPDTraj(self.nusc,
                                        self.predict_steps,
                                        self.planning_steps,
                                        self.past_steps,
                                        self.fut_steps,
                                        self.with_velocity,
                                        self.CLASSES,
                                        self.box_mode_3d,
                                        self.use_nonlinear_optimizer)


        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)
        
    def __len__(self) -> int:
        return len(self.data_infos)
    
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.
        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        if self.file_client_args['backend'] == 'disk':
            # data_infos = mmcv.load(ann_file)
            data = pickle.loads(self.file_client.get(ann_file))
            data_infos = list(
                sorted(data['infos'], key=lambda e: e['timestamp']))
            data_infos = data_infos[::self.load_interval]
            self.metadata = data['metadata']
            self.version = self.metadata['version']
        elif self.file_client_args['backend'] == 'petrel':
            data = pickle.loads(self.file_client.get(ann_file))
            data_infos = list(
                sorted(data['infos'], key=lambda e: e['timestamp']))
            data_infos = data_infos[::self.load_interval]
            self.metadata = data['metadata']
            self.version = self.metadata['version']
        else:
            assert False, 'Invalid file_client_args!'
        return data_infos
    
    def load_data_list(self):
        # 调用父类的方法，获取初步解析结果
        if self.task == 'planning' or self.task == 'vqa':
            data_list=self.process_json_file(self.data_infos, self.json_file)
        # if self.task !='vqa':
        #     data_list =self.filter_empty_sweep(data_list)
        self.length=len(data_list)
        return data_list

    def process_json_file(self, data_infos, json_file):
        
        """处理 json_file 并将相关信息添加到 data_infos 中."""
        print(f"Processing JSON file: {json_file}")
        with open(json_file, 'r') as json_file:
            json_data = json.load(json_file)
        # 提取 JSON 中的 image 和 conversations 数据
        vehicle_qa_pairs = {}
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

            # 提取 vehicle 路径
            vehicle_path = None
            if 'image' in item_group:
                for image_path in item_group['image']:
                    if 'vehicle-side' in image_path:
                        vehicle_path = image_path
                        break  
            
            # 同时存在 vehicle 路径和 Q&A 对时，将其存储
            if vehicle_path and qa_info:
                if vehicle_path not in vehicle_qa_pairs:
                    vehicle_qa_pairs[vehicle_path] = []
                vehicle_qa_pairs[vehicle_path].append(qa_info)
        # 过滤 data_infos 并附加处理后的 Q&A 数据
        filtered_data_infos = []
        image_info = []
        for info in data_infos:
            if 'cams' in info:
                for _, cam_info in info['cams'].items():
                    if 'data_path' in cam_info:
                        print("cam_info['data_path']:",cam_info['data_path'])
                        vehicle_path = cam_info['data_path']
                        image_info.append(cam_info['data_path'])
                 
                if vehicle_path in vehicle_qa_pairs:
                    # 将 Q&A 对添加到 info，并生成新的条目
                    for qa_pair in vehicle_qa_pairs[vehicle_path]:
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


