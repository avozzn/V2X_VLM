#!/usr/bin/env python3
"""Build V2X planning prompts with causal ego history.

The source JSON remains authoritative for sample selection, image paths and
future trajectory answers.  Ego motion and history are rebuilt from the info
PKL using only the current sample and its ``prev`` chain.
"""

import argparse
import copy
import json
import pickle
import re
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion


HISTORY_TIMES = np.asarray([-2.0, -1.5, -1.0, -0.5], dtype=np.float64)
DIMENSIONS_RE = re.compile(
    r"Dimensions \(L, W\): \(([-+\d.]+),\s*([-+\d.]+)\)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--info-pkl", required=True, type=Path)
    parser.add_argument("--source-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def lidar_to_global(info):
    """Return a column-vector homogeneous transform, LiDAR -> global."""
    lidar_to_ego = np.eye(4, dtype=np.float64)
    lidar_to_ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    lidar_to_ego[:3, 3] = np.asarray(info["lidar2ego_translation"], dtype=np.float64)

    ego_to_global = np.eye(4, dtype=np.float64)
    ego_to_global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
    ego_to_global[:3, 3] = np.asarray(info["ego2global_translation"], dtype=np.float64)
    return ego_to_global @ lidar_to_ego


def collect_causal_anchors(current, by_token):
    current_to_global = lidar_to_global(current)
    global_to_current = np.linalg.inv(current_to_global)
    current_timestamp = float(current["timestamp"])
    anchors = [[0.0, 0.0, 0.0]]
    node = current

    while node.get("prev", ""):
        previous = by_token.get(node["prev"])
        if previous is None or previous["scene_token"] != current["scene_token"]:
            break
        relative_time = (float(previous["timestamp"]) - current_timestamp) / 1e6
        if relative_time < HISTORY_TIMES[0] - 1.0:
            break
        previous_in_current = global_to_current @ lidar_to_global(previous)
        anchors.append([
            relative_time,
            previous_in_current[0, 3],
            previous_in_current[1, 3],
        ])
        node = previous

    return np.asarray(sorted(anchors, key=lambda item: item[0]), dtype=np.float64)


def expected_image_paths(info):
    vehicle = info.get("images", {}).get("VEHICLE_CAM_FRONT", {}).get("img_path")
    infrastructure = (
        info.get("other_agent_info_dict", {})
        .get("model_other_agent_inf", {})
        .get("images", {})
        .get("INF_CAM_FRONT", {})
        .get("img_path")
    )
    return [vehicle, infrastructure]


def interpolate_causal_history(anchors):
    """Interpolate observed history and extrapolate older points causally."""
    history = np.zeros((len(HISTORY_TIMES), 2), dtype=np.float64)
    if len(anchors) >= 2:
        times = anchors[:, 0]
        for axis in range(2):
            history[:, axis] = np.interp(HISTORY_TIMES, times, anchors[:, axis + 1])
            older = HISTORY_TIMES < times[0]
            if np.any(older):
                slope = (anchors[1, axis + 1] - anchors[0, axis + 1]) / (
                    times[1] - times[0]
                )
                history[older, axis] = anchors[0, axis + 1] + slope * (
                    HISTORY_TIMES[older] - times[0]
                )

    earliest_observed = anchors[0, 0] if len(anchors) >= 2 else 0.0
    mask = (HISTORY_TIMES >= earliest_observed - 1e-6).astype(np.int64)
    return history, mask


def causal_ego_state(anchors):
    """Estimate current velocity and acceleration from backward differences."""
    velocity = np.zeros(2, dtype=np.float64)
    acceleration = np.zeros(2, dtype=np.float64)
    if len(anchors) >= 2:
        current, previous = anchors[-1], anchors[-2]
        dt_current = current[0] - previous[0]
        velocity = (current[1:3] - previous[1:3]) / dt_current
        if len(anchors) >= 3:
            previous_previous = anchors[-3]
            dt_previous = previous[0] - previous_previous[0]
            previous_velocity = (
                previous[1:3] - previous_previous[1:3]
            ) / dt_previous
            acceleration = (velocity - previous_velocity) / dt_current
    return velocity, acceleration


def build_question(source_question, history, history_mask, velocity, acceleration):
    dimensions_match = DIMENSIONS_RE.search(source_question)
    if dimensions_match is None:
        raise ValueError("source question is missing ego dimensions")
    length, width = map(float, dimensions_match.groups())
    history_text = ", ".join(
        f"({point[0]:.2f},{point[1]:.2f})" for point in history
    )
    mask_text = ", ".join(str(int(value)) for value in history_mask)
    return (
        "\nEgo-States:\n"
        f" - Dimensions (L, W): ({length:.2f}, {width:.2f})\n"
        f" - Velocity (vx,vy): ({velocity[0]:.2f},{velocity[1]:.2f})\n"
        f" - Acceleration (ax,ay): ({acceleration[0]:.2f},{acceleration[1]:.2f})\n"
        f"Historical Trajectory (last 2 seconds): [{history_text}]\n"
        f"Historical Observation Mask: [{mask_text}]\n"
    )


def main():
    args = parse_args()
    with args.info_pkl.open("rb") as file:
        info_data = pickle.load(file)
    data_list = info_data.get("data_list", info_data.get("infos"))
    if data_list is None:
        raise KeyError("info PKL must contain 'data_list' or 'infos'")
    by_token = {str(item["token"]): item for item in data_list}

    with args.source_json.open("r", encoding="utf-8") as file:
        source_items = json.load(file)

    output_items = []
    mask_counts = {observed: 0 for observed in range(len(HISTORY_TIMES) + 1)}
    for source_item in source_items:
        token = str(source_item["token"])
        if token not in by_token:
            raise KeyError(f"token {token} is missing from {args.info_pkl}")
        if len(source_item.get("image", [])) != 2:
            raise ValueError(f"token {token} does not have exactly two images")
        if source_item["image"] != expected_image_paths(by_token[token]):
            raise ValueError(f"token {token} image paths do not match the info PKL")
        conversations = source_item.get("conversations", [])
        if len(conversations) < 2:
            raise ValueError(f"token {token} does not have a question and answer")

        anchors = collect_causal_anchors(by_token[token], by_token)
        history, history_mask = interpolate_causal_history(anchors)
        velocity, acceleration = causal_ego_state(anchors)
        item = copy.deepcopy(source_item)
        item["conversations"][0]["value"] = build_question(
            conversations[0]["value"], history, history_mask,
            velocity, acceleration)
        output_items.append(item)
        mask_counts[int(history_mask.sum())] += 1

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as file:
        json.dump(output_items, file, ensure_ascii=False, indent=2)

    print(f"Wrote {len(output_items)} samples to {args.output_json}")
    print("Observed history point counts:", mask_counts)


if __name__ == "__main__":
    main()
