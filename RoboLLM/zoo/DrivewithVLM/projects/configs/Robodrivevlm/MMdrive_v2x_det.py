class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
# 定义收集合并
vlm_keys = ['input_ids', 'labels', 'attention_mask', 'pixel_values']
det_keys = ['gt_bboxes_3d', 'gt_labels_3d']
geo_keys = ['lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'ego_pose']
collect_keys = ['lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv', 'command', 'can_bus']
metainfo = dict(classes=class_names)
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False) ###### univ2x
default_scope = 'mmdet3d'
inf_point_cloud_range = [0, -51.2, -5.0, 102.4, 51.2, 3.0]
inf_post_center_range = [-10.0, -61.2, -10.0, 112.4, 61.2, 10.0]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
post_center_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]

## occflow setting used in V2X
occ_n_future = 4
occ_n_future_plan = 10 #####
occ_n_future_max = max([occ_n_future, occ_n_future_plan]) #####

### traj prediction args used in v2x###
predict_steps = 9
predict_modes = 6
fut_steps = 4
past_steps = 4
use_nonlinear_optimizer = True

### planning used in v2x###
planning_steps = 9
use_col_optim = True

### other settings used in v2x###
train_gt_iou_threshold = 0.3

### Occ args ###
occflow_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
}
ida_aug_conf = {
        "resize_lim": (0.37, 0.45),
        "final_dim": (320, 640),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": False,
    }
llm_path = 'ckpts/'
voxel_size = [0.2, 0.2, 8]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    visualization=dict(type='Det3DVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
custom_imports = dict(
    imports=['projects.Robodrivevlm'], allow_failed_imports=False)

#------------------------------------------------------------------------------------------------------------------#
#                                                Dataloader Config (Hybrid)                                                                                         #
#------------------------------------------------------------------------------------------------------------------#
dataset_type = "SpdVehicleE2EVLMDataset" ##############
data_root = "data/V2X-Seq-SPD-New/cooperative/"
info_root = "datainfo/infos/V2X-Seq-SPD-New/cooperative/"
project_root='./'
####### 怎么改？
data_prefix = dict(
    pts='vehicle-side/velodyne',
    pts_semantic_mask='v1.0-trainval',
    CAM_FRONT='vehicle-side/image',
    CAM_FRONT_LEFT='infrastructure-side/image',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT')

backend_args = None
input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True
)
ann_file_train = info_root + f"spd_infos_temporal_train_sdc.pkl"
ann_file_val = info_root + f"spd_infos_temporal_val_sdc.pkl"
ann_file_test = info_root + f"spd_infos_temporal_val_sdc.pkl"
json_file_train = project_root + 'data/Planning//train_v2x_dt_VM_re4.json' # VLM 问答对/指令文件
json_file_test = project_root + 'data/Planning/test_v2x_resampled_4.json'
split_datas_file = project_root + 'data/V2X-Seq-SPD-New/datainfo/split_datas/cooperative-split-data-spd.json'
v2x_side = 'cooperative'
other_agent_names = ['model_other_agent_inf']
##PhotoMetricDistortionMultiViewImage 数据增强 可以加在LoadMultiViewImageFromFiles4Clip后面
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles4Clip', only_vehicle = False, test_mode=False, dataroot=data_root, to_float32=True),
    # dict(type='Visualize3DBoxOnImage',img_root=data_root,save_dir=project_root+'vis_image/'),  # 可视化 3D 框
    # dict(type='GenerateOccFlowLabels', grid_conf=occflow_grid_conf, ignore_index=255, only_vehicle=True,
    #     filter_invisible=False),  # NOTE: Currently vis_token is not in pkl
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
        with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
   # --- 核心引入：Omni 的几何增强逻辑 ---
    # 这个算子会根据旋转/缩放同时更新图像和 lidar2img 投影矩阵，非常关键
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='ResizeMultiview3D', img_scale=[(640, 640)], keep_ratio=False, multiscale_mode='value'),
    
    dict(type='Processed_sensor',dataroot=data_root, vis_root=project_root+'vis/'),
    dict(type='Load_system_prompt',test_mode=False,prompt_choose=None),
    # dict(type='Load_Instruction'),
    dict(type='Load_EGO',load_sensor=True, load_image=False),
    dict(type='Load_planning_prompt'),
    # PETRFormatBundle3D 会把 gt_bboxes_3d 转为 Tensor 格式
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys),
    dict(type='LLava_interleave_Process',model_hf_path=project_root+'checkpoints/LLM/llava-next-interleave',mode='train'),
    dict(
    type='Pack3DDetInputs',
    keys= vlm_keys + det_keys + geo_keys,
    meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 
               'scale_factor', 'box_mode_3d', 'box_type_3d', 'scene_token')
)

]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles4Clip', only_vehicle = False, test_mode=False, dataroot=data_root, to_float32=True),
    dict(type='Visualize3DBoxOnImage',img_root=data_root,save_dir=project_root+'vis_test_image/'),  # 可视化 3D 框
    dict(type='Processed_sensor',dataroot= data_root, vis_root=project_root+'vis/',load_image=False),
    dict(type='Load_system_prompt',test_mode=True,prompt_choose=None),
    dict(type='Load_EGO',load_sensor=True,load_image=False),
    # dict(type='Load_Instruction'),
    dict(type='Load_planning_prompt_test'),
    dict(type='LLava_interleave_Process',model_hf_path=project_root+'checkpoints/LLM/llava-next-interleave')
]

# 使用 V2X 的 ann_file 路径

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file= ann_file_train,
        json_file=json_file_train,
        task='planning', 
        metainfo=metainfo,   
        pipeline=train_pipeline,
        test_mode=False,
        modality=input_modality,
        v2x_side=v2x_side,
        other_agent_names=other_agent_names,
        ))




test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=False), # 测试集通常不 shuffle 原来shuffle true
    dataset=dict(
        type=dataset_type, # SPDE2EVLMDataset
        data_root=data_root, # V2X 根目录
        data_prefix=data_prefix, # 保留 nuScenes 格式
        ann_file=ann_file_val, # V2X 的 val pkl 文件
        json_file=json_file_test, # VLM 的指令文件
        task='planning',    
        metainfo=metainfo,
        pipeline=test_pipeline,
        test_mode=True, # 确保设置为 True
        v2x_side=v2x_side,
        # V2X 特有参数
        modality=input_modality,
        )
)



vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),
    }),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=24,
        by_epoch=True,
        eta_min=1e-6,
        convert_to_iter_based=True)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)

test_cfg = dict()
test_evaluator=dict()

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))

model = dict(
    type='ResNet'
)