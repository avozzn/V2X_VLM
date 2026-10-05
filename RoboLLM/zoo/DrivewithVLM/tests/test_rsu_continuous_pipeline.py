"""F0/F1 scientific control and input/target isolation regression tests."""
import json

import numpy as np
import pytest
import torch

from tools.continuous.data import collate_features, observation_prompt
from tools.continuous.model import (
    ContinuousPlanner, FrozenLlavaReadout, last_valid_hidden, waypoint_loss,
)


def test_legacy_rsu_planner_is_disabled():
    for mode in ("oracle", "detector"):
        model = ContinuousPlanner(8, mode, head_dim=16)
        with pytest.raises(RuntimeError, match="bypass is disabled"):
            model(torch.randn(2, 8), torch.zeros(2, 16, 13),
                  torch.zeros(2, 16, dtype=torch.bool))


def test_masked_loss_and_batch_gradient():
    pred = torch.ones(2, 9, 2, requires_grad=True)
    target = torch.zeros_like(pred)
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[:, 3:] = False
    target[:, 3:] = float("nan")
    loss = waypoint_loss(pred, target, mask)
    torch.testing.assert_close(loss, torch.tensor(.5))
    loss.backward()
    assert pred.grad[0, :3].abs().sum() > 0 and pred.grad[1, :3].abs().sum() > 0
    assert pred.grad[:, 3:].count_nonzero() == 0


def test_readout_left_right_padding():
    hidden = torch.arange(24.).reshape(2, 6, 2)
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
    torch.testing.assert_close(last_valid_hidden(hidden, mask), torch.stack([hidden[0, 5], hidden[1, 2]]))
    with pytest.raises(ValueError):
        last_valid_hidden(hidden, mask[:, :3])


def test_save_load(tmp_path):
    model = ContinuousPlanner(8, "zero", head_dim=16).eval()
    ego = torch.randn(2, 8)
    model.save(tmp_path)
    loaded = ContinuousPlanner.load(tmp_path).eval()
    torch.testing.assert_close(model(ego)["trajectory"], loaded(ego)["trajectory"])


def test_observation_prompt_ignores_all_answers_and_rsu():
    sample = {"planning_targets": {"ego_velocity_xy_mps": [3., 0.],
              "ego_acceleration_xy_mps2": [0., 0.],
              "history_xy": [[-1., 0.]], "history_mask": [True]},
              "ego_state": {"dimensions_lw_m": [4., 2.]},
              "ego_perception": {"schema": "ego-perception-v1", "source": "vehicle_gt",
                                 "coordinate_frame": "current_ego_lidar", "objects": []}}
    expected = observation_prompt(sample)
    sample.update(conversations=["LEAK"], planning_labels_v2={"action": "LEAK"}, agents={"geometry": "LEAK"})
    sample["planning_targets"]["future_xy"] = [[999., 999.]]
    assert observation_prompt(sample) == expected
    batch = collate_features([{"h_ego": torch.zeros(3), "index": 0},
                              {"h_ego": torch.ones(3), "index": 1}])
    assert batch["h_ego"].shape == (2, 3) and batch["h_ego"][1].sum() == 3


def test_native_vehicle_gt_and_shared_formatter(tmp_path):
    from tools.continuous.ego_perception import load_vehicle_gt, perception_text
    root = tmp_path / "vehicle-side"
    labels = root / "label/lidar"
    labels.mkdir(parents=True)
    row = {"type": "Car", "3d_location": {"x": 8., "y": -2., "z": 1.},
           "3d_dimensions": {"l": 4., "w": 2., "h": 1.5}, "rotation": .2,
           "velocity": [999., 999.]}
    (labels / "000001.json").write_text(json.dumps([row]))
    frame = {"frame_id": "000001", "pointcloud_timestamp": "100",
             "label_lidar_std_path": "label/lidar/000001.json"}
    gt = load_vehicle_gt(frame, root, 100)
    assert gt["objects"][0]["size_lwh_m"] == [4., 2., 1.5]
    assert gt["objects"][0]["heading_rad"] == .2
    text = perception_text(gt)
    assert "velocity_xy_mps=unknown" in text and "999" not in text
    assert "position_xyz=(8.00, -2.00, 1.00)" in text
    # GT and detector share formatting; provenance stays outside VLM text.
    gt["source"] = "vehicle_detector"
    assert perception_text(gt) == text
    with pytest.raises(ValueError, match="current ego"):
        load_vehicle_gt(frame, root, 99)
    frame["label_lidar_std_path"] = "../cooperative/label/000001.json"
    with pytest.raises(ValueError, match="vehicle-side"):
        load_vehicle_gt(frame, root, 100)


def test_ego_perception_changes_prompt_and_observation_signature():
    import copy
    from tools.continuous.data import observation_signature
    sample = {"token": "t", "ego_image": "vehicle-side/image/t.jpg", "timestamp_us": 100,
              "planning_targets": {"ego_velocity_xy_mps": [3., 0.],
              "ego_acceleration_xy_mps2": [0., 0.], "history_xy": [], "history_mask": []},
              "ego_state": {"dimensions_lw_m": [4., 2.]},
              "ego_perception": {"schema": "ego-perception-v1", "source": "vehicle_gt",
              "coordinate_frame": "current_ego_lidar", "timestamp_us": 100,
              "source_frame": "t", "objects": []}}
    altered = copy.deepcopy(sample)
    altered["ego_perception"]["objects"] = [{"class": "car", "position_xyz_m": [8., 0., 0.],
        "size_lwh_m": [4., 2., 1.], "heading_rad": 0., "velocity_valid": False,
        "velocity_xy_mps": None, "confidence": None}]
    assert observation_prompt(altered) != observation_prompt(sample)
    assert observation_signature([altered]) != observation_signature([sample])
    altered = copy.deepcopy(sample)
    altered["agents"] = {"geometry": "RSU cannot change the ego feature cache"}
    assert observation_signature([altered]) == observation_signature([sample])


def test_real_tiny_llava_legacy_merge_readout():
    """Use actual installed HF merge/decoder on CPU, not a mocked backbone."""
    from transformers import LlavaConfig, LlavaForConditionalGeneration
    config = LlavaConfig(
        vision_config={"model_type": "siglip_vision_model", "hidden_size": 16,
                       "intermediate_size": 32, "num_hidden_layers": 1,
                       "num_attention_heads": 2, "image_size": 8, "patch_size": 4},
        text_config={"model_type": "qwen2", "vocab_size": 40, "hidden_size": 16,
                     "intermediate_size": 32, "num_hidden_layers": 1,
                     "num_attention_heads": 2, "num_key_value_heads": 2,
                     "pad_token_id": 0}, image_token_index=39,
        vision_feature_select_strategy="full", vision_feature_layer=-1,
        image_seq_length=4, pad_token_id=0)
    backbone = LlavaForConditionalGeneration(config)
    extractor = FrozenLlavaReadout(backbone).train()
    assert not backbone.training
    ids = torch.tensor([[0, 39, 4, 5], [39, 6, 7, 8]])
    features, length = extractor(input_ids=ids, attention_mask=ids.ne(0).long(),
                                 pixel_values=(images := torch.randn(2, 3, 8, 8)))
    assert length == 7 and features.shape == (2, 16)
    assert torch.isfinite(features).all() and not features.requires_grad
    single, _ = extractor(input_ids=ids[:1, 1:], attention_mask=torch.ones(1, 3, dtype=torch.long),
                          pixel_values=images[:1])
    torch.testing.assert_close(features[:1], single, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError):
        extractor(input_ids=ids, labels=ids)


def test_geometry_rotation_dimensions_and_yaw():
    from tools.continuous.prepare import transform_geometry
    # Physical heading x+, stored encoding -pi/2, dimensions w=2/l=4/h=1.
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = np.array([10., 20., 0.])
    ego_center = np.array([3., 4., 0.])
    infra_center = rotation @ ego_center + translation
    boxes = [np.r_[infra_center, 2., 4., 1., -np.pi/2]]
    geometry, mask = transform_geometry(boxes, ["car"], rotation, translation)
    np.testing.assert_allclose(geometry[0, :6], [3., 4., 0., 4., 2., 1.], atol=1e-6)
    np.testing.assert_allclose(geometry[0, 6:8], [-1., 0.], atol=1e-6)
    assert mask.sum() == 1
