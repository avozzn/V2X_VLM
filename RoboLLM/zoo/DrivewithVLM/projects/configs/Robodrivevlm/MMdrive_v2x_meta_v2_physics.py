_base_ = ['./MMdrive_v2x_meta_v2.py']

# Same scene split and metadata labels as V2. The language target contains only
# lateral action, longitudinal action and trajectory; continuous motion values
# are derived deterministically during evaluation. Separate pipelines preserve
# the original V2 prompt contract.
physics_train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles4Clip', only_vehicle=False,
         test_mode=False, dataroot=_base_.data_root, to_float32=True),
    dict(type='Processed_sensor', dataroot=_base_.data_root,
         vis_root=_base_.project_root + 'vis/'),
    dict(type='Load_system_prompt', test_mode=False, prompt_choose=None),
    dict(type='Load_EGO', load_sensor=True, load_image=False),
    dict(type='Load_planning_prompt', output_contract='physics'),
    dict(type='LLava_interleave_Process',
         model_hf_path=_base_.project_root + 'checkpoints/LLM/llava-next-interleave',
         mode='train', max_length=_base_.train_max_length),
]

physics_test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles4Clip', only_vehicle=False,
         test_mode=False, dataroot=_base_.data_root, to_float32=True),
    dict(type='Visualize3DBoxOnImage', img_root=_base_.data_root,
         save_dir=_base_.project_root + 'vis_test_image/'),
    dict(type='Processed_sensor', dataroot=_base_.data_root,
         vis_root=_base_.project_root + 'vis/', load_image=False),
    dict(type='Load_system_prompt', test_mode=True, prompt_choose=None),
    dict(type='Load_EGO', load_sensor=True, load_image=False),
    dict(type='Load_planning_prompt_test', output_contract='physics'),
    dict(type='LLava_interleave_Process',
         model_hf_path=_base_.project_root + 'checkpoints/LLM/llava-next-interleave',
         max_length=_base_.test_max_input_length),
]

train_dataloader = dict(
    dataset=dict(
        json_file='./data/Planning/v2_physics/train_v2x_planning_v2.json',
        pipeline=physics_train_pipeline))

val_dataloader = dict(
    dataset=dict(
        json_file='./data/Planning/v2_physics/val_v2x_planning_v2.json',
        pipeline=physics_train_pipeline))

test_dataloader = dict(
    dataset=dict(
        json_file='./data/Planning/v2_physics/test_v2x_planning_v2.json',
        pipeline=physics_test_pipeline))
