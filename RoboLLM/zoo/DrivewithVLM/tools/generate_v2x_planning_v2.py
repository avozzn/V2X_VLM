#!/usr/bin/env python3
"""Generate deduplicated V2X planning V2 train/val/test prompt data."""

import argparse
import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projects.mmdet3d_plugin.datasets.data_utils.planning_metadata_v2 import (  # noqa: E402
    generate_planning_metadata_v2,
)


DT = 0.5
HISTORY_TIMES = np.asarray([-2.0, -1.5, -1.0, -0.5], dtype=np.float64)
FUTURE_TIMES = np.arange(1, 10, dtype=np.float64) * DT


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-info", required=True, type=Path)
    parser.add_argument("--test-info", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-scene-ratio", type=float, default=0.15)
    return parser.parse_args()


def load_info(path):
    with path.open("rb") as file:
        payload = pickle.load(file)
    rows = payload.get("data_list", payload.get("infos"))
    if rows is None:
        raise KeyError(f"{path} must contain data_list or infos")
    by_token = {str(item["token"]): item for item in rows}
    if len(by_token) != len(rows):
        raise ValueError(f"{path} contains duplicate tokens")
    return rows, by_token


def lidar_to_global(info):
    lidar_to_ego = np.eye(4, dtype=np.float64)
    lidar_to_ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    lidar_to_ego[:3, 3] = np.asarray(info["lidar2ego_translation"], dtype=np.float64)
    ego_to_global = np.eye(4, dtype=np.float64)
    ego_to_global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
    ego_to_global[:3, 3] = np.asarray(info["ego2global_translation"], dtype=np.float64)
    return ego_to_global @ lidar_to_ego


def relative_position(current_to_global, other):
    transform = np.linalg.inv(current_to_global) @ lidar_to_global(other)
    return transform[:2, 3]


def collect_chain(current, by_token, direction, limit_seconds):
    current_time = float(current["timestamp"])
    current_to_global = lidar_to_global(current)
    anchors = [[0.0, 0.0, 0.0]]
    node = current
    while node.get(direction, ""):
        other = by_token.get(str(node[direction]))
        if other is None or other.get("scene_token") != current.get("scene_token"):
            break
        relative_time = (float(other["timestamp"]) - current_time) / 1e6
        if abs(relative_time) > limit_seconds:
            break
        position = relative_position(current_to_global, other)
        anchors.append([relative_time, position[0], position[1]])
        node = other
    return np.asarray(sorted(anchors, key=lambda row: row[0]), dtype=np.float64)


def interpolate_observed(anchors, target_times):
    result = np.zeros((len(target_times), 2), dtype=np.float64)
    observed = np.zeros(len(target_times), dtype=np.int64)
    if len(anchors) < 2:
        return result, observed
    minimum, maximum = anchors[0, 0], anchors[-1, 0]
    valid = (target_times >= minimum - 1e-6) & (target_times <= maximum + 1e-6)
    for axis in range(2):
        result[valid, axis] = np.interp(
            target_times[valid], anchors[:, 0], anchors[:, axis + 1])
    observed[valid] = 1
    return result, observed


def causal_ego_state(info, past_anchors):
    """Use the measured current velocity and causal history for acceleration."""
    velocity = np.asarray(info.get("sdc_velocity", [0.0, 0.0]), dtype=np.float64)[:2]
    acceleration = np.zeros(2, dtype=np.float64)
    if len(past_anchors) >= 2:
        current, previous = past_anchors[-1], past_anchors[-2]
        current_dt = current[0] - previous[0]
        previous_velocity = (current[1:3] - previous[1:3]) / current_dt
        acceleration = (velocity - previous_velocity) / current_dt
    if velocity.shape != (2,) or not np.isfinite(velocity).all():
        raise ValueError("sdc_velocity must contain two finite values")
    return velocity, acceleration


def image_paths(info):
    vehicle = info.get("images", {}).get("VEHICLE_CAM_FRONT", {}).get("img_path")
    infrastructure = (
        info.get("other_agent_info_dict", {})
        .get("model_other_agent_inf", {})
        .get("images", {})
        .get("INF_CAM_FRONT", {})
        .get("img_path")
    )
    return [vehicle, infrastructure]


def perception_text(info, perception_range=50.0):
    fragments = []
    for instance in info.get("instances", []):
        if not instance.get("bbox_3d_isvalid", True):
            continue
        box = np.asarray(instance.get("bbox_3d", []), dtype=np.float64)
        velocity = np.asarray(instance.get("velocity", [0.0, 0.0]), dtype=np.float64)
        if box.size < 2 or box[0] < -10.0 or np.any(np.abs(box[:2]) > perception_range):
            continue
        x, y = box[:2]
        longitudinal = "Front" if x > 0 else "Rear"
        lateral = "Left" if y > 1.5 else "Right" if y < -1.5 else "Center"
        name = str(instance.get("gt_names", "object"))
        fragments.append(
            f" - {name} at {longitudinal}-{lateral} ({x:.1f}, {y:.1f})m, "
            f"vel ({velocity[0]:.1f}, {velocity[1]:.1f})m/s,")
    if not fragments:
        return "Perception and Prediction:\n - No relevant objects detected nearby.\n"
    return "Perception and Prediction:\n" + " ".join(fragments) + "\n"


def ego_dimensions(info):
    box = np.asarray(info.get("gt_sdc_bbox", []), dtype=np.float64)
    if box.size < 5:
        raise ValueError("missing gt_sdc_bbox dimensions")
    return float(box[4]), float(box[3])


def build_question(info, history, history_mask, velocity, acceleration):
    length, width = ego_dimensions(info)
    history_text = ", ".join(f"({x:.2f},{y:.2f})" for x, y in history)
    mask_text = ", ".join(str(int(value)) for value in history_mask)
    return (
        "\n" + perception_text(info)
        + "Ego-States:\n"
        + f" - Dimensions (L, W): ({length:.2f}, {width:.2f})\n"
        + f" - Velocity (vx,vy): ({velocity[0]:.2f},{velocity[1]:.2f})\n"
        + f" - Acceleration (ax,ay): ({acceleration[0]:.2f},{acceleration[1]:.2f})\n"
        + f"Historical Trajectory (last 2 seconds): [{history_text}]\n"
        + f"Historical Observation Mask: [{mask_text}]\n"
    )


def rounded_metadata(metadata):
    result = dict(metadata)
    for key, value in result.items():
        if isinstance(value, (float, np.floating)):
            result[key] = round(float(value), 6)
    return result


def build_answer(metadata, trajectory):
    points = ", ".join(f"({x:.2f},{y:.2f})" for x, y in trajectory)
    return (
        f"Lateral Action: {metadata['lateral_action']}\n"
        f"Longitudinal Action: {metadata['longitudinal_action']}\n"
        f"Trajectory:\n[{points}]"
    )


def build_samples(rows, by_token):
    samples = []
    excluded = Counter()
    history_masks = Counter()
    for info in sorted(rows, key=lambda item: (str(item.get("scene_token")), item["timestamp"])):
        token = str(info["token"])
        images = image_paths(info)
        if len(images) != 2 or any(not path for path in images):
            excluded["missing_dual_images"] += 1
            continue
        past = collect_chain(info, by_token, "prev", 3.0)
        future = collect_chain(info, by_token, "next", 5.5)
        history, history_mask = interpolate_observed(past, HISTORY_TIMES)
        trajectory, future_mask = interpolate_observed(future, FUTURE_TIMES)
        if not future_mask.all():
            excluded["incomplete_future_4_5s"] += 1
            continue
        velocity, acceleration = causal_ego_state(info, past)
        planning_mask = np.ones((1, 9, 2), dtype=np.float32)
        metadata = generate_planning_metadata_v2(
            sdc_planning=trajectory[None, ...],
            planning_mask=planning_mask,
            current_velocity=velocity.astype(np.float64),
            dt=DT,
        )
        metadata = rounded_metadata(metadata)
        samples.append({
            "token": token,
            "scene_token": str(info.get("scene_token", "")),
            "image": images,
            "planning_labels_v2": metadata,
            "conversations": [
                {"from": "human", "value": build_question(
                    info, history, history_mask, velocity, acceleration)},
                {"from": "gpt", "value": build_answer(metadata, trajectory)},
            ],
        })
        history_masks["".join(map(str, history_mask.tolist()))] += 1
    return samples, excluded, history_masks


def split_train(samples, ratio, seed):
    if not 0.0 < ratio < 1.0:
        raise ValueError("--val-scene-ratio must be between 0 and 1")
    scenes = sorted({sample["scene_token"] for sample in samples})
    if len(scenes) < 2:
        raise ValueError("at least two train scenes are required")
    random.Random(seed).shuffle(scenes)
    val_count = max(1, round(len(scenes) * ratio))
    val_scenes = set(scenes[:val_count])
    train = [sample for sample in samples if sample["scene_token"] not in val_scenes]
    val = [sample for sample in samples if sample["scene_token"] in val_scenes]
    return train, val, sorted(set(scenes) - val_scenes), sorted(val_scenes)


def action_counts(samples, key):
    return dict(sorted(Counter(
        sample["planning_labels_v2"][key] for sample in samples).items()))


def numeric_summary(samples, key):
    values = np.asarray([sample["planning_labels_v2"][key] for sample in samples], dtype=np.float64)
    if not values.size:
        return {"min": None, "median": None, "p90": None, "max": None}
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def threshold_audit(samples):
    metadata = [sample["planning_labels_v2"] for sample in samples]
    return {
        "final_lateral_within_0_5m_of_2m": sum(
            abs(abs(item["final_lateral_offset_m"]) - 2.0) <= 0.5 for item in metadata),
        "final_heading_within_3deg_of_15deg": sum(
            abs(abs(item["final_heading_change_deg"]) - 15.0) <= 3.0 for item in metadata),
        "end_speed_within_0_2mps_of_stop_threshold": sum(
            abs(item["target_end_speed_mps"] - 0.5) <= 0.2 for item in metadata),
        "acceleration_within_0_05mps2_of_action_threshold": sum(
            abs(abs(item["mean_acceleration_mps2"]) - 0.2) <= 0.05 for item in metadata),
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    train_rows, train_by_token = load_info(args.train_info)
    test_rows, test_by_token = load_info(args.test_info)
    train_all, train_excluded, train_masks = build_samples(train_rows, train_by_token)
    test, test_excluded, test_masks = build_samples(test_rows, test_by_token)
    train, val, train_scenes, val_scenes = split_train(
        train_all, args.val_scene_ratio, args.seed)
    test_scenes = sorted({sample["scene_token"] for sample in test})
    if set(train_scenes) & set(val_scenes) or (set(train_scenes) | set(val_scenes)) & set(test_scenes):
        raise ValueError("scene leakage detected across train/val/test")

    names = {
        "train": "train_v2x_planning_v2.json",
        "val": "val_v2x_planning_v2.json",
        "test": "test_v2x_planning_v2.json",
    }
    for split, samples in (("train", train), ("val", val), ("test", test)):
        write_json(args.output_dir / names[split], samples)

    manifest = {
        "schema_version": "2.0",
        "output_contract_version": "2.1-action-trajectory-physics",
        "seed": args.seed,
        "val_scene_ratio": args.val_scene_ratio,
        "source": {"train_info": str(args.train_info.resolve()), "test_info": str(args.test_info.resolve())},
        "files": names,
        "scenes": {"train": train_scenes, "val": val_scenes, "test": test_scenes},
    }
    audit = {
        "schema_version": "2.0",
        "output_contract_version": "2.1-action-trajectory-physics",
        "source_counts": {"official_train": len(train_rows), "official_test": len(test_rows)},
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "unique_tokens": {
            "train": len({x['token'] for x in train}),
            "val": len({x['token'] for x in val}),
            "test": len({x['token'] for x in test}),
        },
        "excluded": {"official_train": dict(train_excluded), "official_test": dict(test_excluded)},
        "history_masks": {"official_train": dict(train_masks), "official_test": dict(test_masks)},
        "lateral_actions": {split: action_counts(samples, "lateral_action") for split, samples in (("train", train), ("val", val), ("test", test))},
        "longitudinal_actions": {split: action_counts(samples, "longitudinal_action") for split, samples in (("train", train), ("val", val), ("test", test))},
        "continuous_distributions": {
            split: {
                key: numeric_summary(samples, key)
                for key in ("current_speed_mps", "target_end_speed_mps", "mean_acceleration_mps2", "final_lateral_offset_m", "final_heading_change_deg")
            }
            for split, samples in (("train", train), ("val", val), ("test", test))
        },
        "near_threshold_counts": {
            split: threshold_audit(samples)
            for split, samples in (("train", train), ("val", val), ("test", test))
        },
        "scene_overlap": {"train_val": [], "train_test": [], "val_test": []},
    }
    write_json(args.output_dir / "planning_v2_split_manifest.json", manifest)
    write_json(args.output_dir / "planning_v2_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
