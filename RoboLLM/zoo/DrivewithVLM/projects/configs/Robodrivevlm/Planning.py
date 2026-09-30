default_scope = 'mmdet3d'
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

metainfo = dict(classes=class_names)


default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=-1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
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


dataset_type = 'NuScenesMMDataset'
data_root='data/nuscenes'
project_root='./'

data_prefix = dict(
    sweeps='sweeps',
    pts='samples/LIDAR_TOP',
    pts_semantic_mask='lidarseg/v1.0-trainval',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT')

backend_args = None

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles4Clip', dataroot=data_root,to_float32=True),
    dict(type='Processed_sensor',dataroot=data_root, vis_root=project_root+'vis/'),
    dict(type='Load_system_prompt',prompt_choose=None),
    dict(type='Load_EGO'),
    dict(type='Load_Instruction'),
    dict(type='Load_planning_prompt'),
    dict(type='LLava_interleave_Process',model_hf_path=project_root+'checkpoints/LLM/llava-next-interleave',mode='train')
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles4Clip', dataroot=data_root,to_float32=True),
    dict(type='Processed_sensor',dataroot=data_root, vis_root=project_root+'vis/',load_image=True),
    dict(type='Load_system_prompt',prompt_choose=None),
    dict(type='Load_EGO',load_image=True),
    dict(type='Load_Instruction'),
    dict(type='Load_planning_prompt'),
    dict(type='LLava_interleave_Process',model_hf_path=project_root+'checkpoints/LLM/llava-next-interleave')
]


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
        ann_file='nuscenes_infos_train.pkl',
        json_file=project_root+'data/Planning/train-llava-1.json',
        blacklist=project_root+'data/Planning/blacklist_tokens.json',
        task='planning', 
        metainfo=metainfo,   
        pipeline=train_pipeline,
        test_mode=False))




test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='nuscenes_infos_val.pkl',
        json_file=project_root+'data/Planning/train-llava-1.json',
        blacklist=project_root+'data/Planning/blacklist_tokens.json',
        task='planning',    
        metainfo=metainfo,
        pipeline=test_pipeline,
        test_mode=False))


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