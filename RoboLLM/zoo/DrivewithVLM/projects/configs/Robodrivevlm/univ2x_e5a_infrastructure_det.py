"""E5-A D2: infrastructure-only UniV2X detection/tracking baseline on SPD.

Run this config from the UniV2X repository with ``tools/train.py``.  Unlike
the cooperative pkl, this official infrastructure-side info file provides GT
in the infrastructure LiDAR frame, which matches this model's pc_range.
"""

_base_ = [
    "/home/zzn/V2X_VLM/UniV2X/projects/configs_e2e_univ2x/"
    "univ2x_sub_inf_e2e_track.py"
]

e5a_experiment = "infrastructure_only"

# Allow AMP to back off on overflow instead of a fixed scale 512.
fp16 = dict(_delete_=True, loss_scale=dict(init_scale=32.0, growth_interval=2000))

model_ego_agent = dict(
    seg_head=None,
    load_from="ckpts/bevformer_r101_dcn_24ep.pth",
)

_infrastructure_data_root = "datasets/V2X-Seq-SPD-New/infrastructure-side/"
_infrastructure_info_root = "data/infos/V2X-Seq-SPD-New/infrastructure-side/"

# Keep this path deliberately detection/tracking-only.  The inherited E2E
# pipeline requests future occupancy labels and local maps, neither of which is
# part of E5-A's per-agent detector comparison.
_infrastructure_train_pipeline = [
    dict(type="LoadMultiViewImageFromFilesInCeph", to_float32=True,
         file_client_args=dict(backend="disk"), img_root=_infrastructure_data_root),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="LoadAnnotations3D_E2E", with_bbox_3d=True, with_label_3d=True,
         with_attr_label=False, with_future_anns=False, with_ins_inds_3d=True,
         ins_inds_add_1=True),
    dict(type="ObjectRangeFilterTrack",
         point_cloud_range=[0.0, -51.2, -5.0, 102.4, 51.2, 3.0]),
    dict(type="ObjectNameFilterTrack", classes=[
        "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
        "motorcycle", "bicycle", "pedestrian", "traffic_cone",
    ]),
    dict(type="NormalizeMultiviewImage",
         mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=[
        "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
        "motorcycle", "bicycle", "pedestrian", "traffic_cone",
    ]),
    dict(type="CustomCollect3D", keys=[
        "gt_bboxes_3d", "gt_labels_3d", "gt_inds", "img", "timestamp",
        "l2g_r_mat", "l2g_t", "gt_fut_traj", "gt_fut_traj_mask",
        "gt_past_traj", "gt_past_traj_mask", "gt_sdc_bbox", "gt_sdc_label",
    ], meta_keys=(
        "filename", "ori_shape", "img_shape", "lidar2img", "pad_shape",
        "scale_factor", "box_mode_3d", "box_type_3d", "img_norm_cfg",
        "sample_idx", "prev_idx", "next_idx", "scene_token", "can_bus",
    )),
]

data = dict(
    train=dict(
        type="SPDE2EDetectionDataset",
        data_root=_infrastructure_data_root,
        ann_file=_infrastructure_info_root + "spd_infos_temporal_train.pkl",
        pipeline=_infrastructure_train_pipeline,
        other_agent_names=[],
    ),
    val=dict(
        type="SPDE2EDetectionDataset",
        data_root=_infrastructure_data_root,
        ann_file=_infrastructure_info_root + "spd_infos_temporal_val.pkl",
        other_agent_names=[],
    ),
    test=dict(
        type="SPDE2EDetectionDataset",
        data_root=_infrastructure_data_root,
        ann_file=_infrastructure_info_root + "spd_infos_temporal_val.pkl",
        other_agent_names=[],
    ),
)
