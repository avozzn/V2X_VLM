import os

_base_ = ['./MMdrive_v2x.py']

univ2x_info_root = os.environ.get(
    'UNIV2X_INFO_ROOT',
    '/home/zzn/V2X_VLM/UniV2X/data/infos/V2X-Seq-SPD-New/cooperative')
train_info = os.path.join(univ2x_info_root, 'spd_infos_temporal_train_sdc.pkl')
val_info = os.path.join(univ2x_info_root, 'spd_infos_temporal_val_sdc.pkl')

# Meta Action V2 keeps the E0 configuration intact and switches only the
# prompt datasets. The official train PKL backs both scene-disjoint splits.
train_dataloader = dict(
    dataset=dict(
        ann_file=train_info,
        json_file='./data/Planning/v2/train_v2x_planning_v2.json'))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='SpdVehicleE2EVLMDataset',
        data_root='data/V2X-Seq-SPD-New/cooperative/',
        data_prefix=_base_.data_prefix,
        ann_file=train_info,
        json_file='./data/Planning/v2/val_v2x_planning_v2.json',
        task='planning',
        metainfo=_base_.metainfo,
        pipeline=_base_.train_pipeline,
        test_mode=False,
        modality=_base_.input_modality,
        v2x_side='cooperative',
        other_agent_names=['model_other_agent_inf'],
    ))

test_dataloader = dict(
    dataset=dict(
        ann_file=val_info,
        json_file='./data/Planning/v2/test_v2x_planning_v2.json'))
