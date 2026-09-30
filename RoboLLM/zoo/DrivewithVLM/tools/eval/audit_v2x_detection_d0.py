#!/usr/bin/env python3
"""Audit SPD cooperative 3D-detection labels and vehicle/infrastructure frames.

The official UniV2X cooperative info file stores detection GT in the vehicle
LiDAR frame.  This script verifies the vehicle-to-infrastructure calibration,
counts valid GT in each agent's BEV range, and writes a JSON report.  It never
changes the input data.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


VEHICLE_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
INFRASTRUCTURE_RANGE = (0.0, -51.2, -5.0, 102.4, 51.2, 3.0)
INFRA_AGENT = "model_other_agent_inf"


def in_xy_range(points: np.ndarray, point_cloud_range: tuple[float, ...]) -> np.ndarray:
    return (
        (points[:, 0] >= point_cloud_range[0])
        & (points[:, 0] <= point_cloud_range[3])
        & (points[:, 1] >= point_cloud_range[1])
        & (points[:, 1] <= point_cloud_range[4])
    )


def vehicle_to_infrastructure(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply the column-vector vehicle-LiDAR -> infrastructure-LiDAR transform."""
    return points @ rotation.T + translation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Audit at most this many frames; 0 means the full info file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.ann_file.open("rb") as handle:
        payload = pickle.load(handle)

    if not {"infos", "metadata"}.issubset(payload):
        raise ValueError(
            f"Expected UniV2X infos/metadata format, got keys {sorted(payload)}. "
            "Use spd_infos_temporal_train.pkl, not the DrivewithVLM *_sdc.pkl."
        )

    infos = payload["infos"][: args.limit or None]
    totals = Counter()
    round_trip_errors: list[float] = []
    orthogonality_errors: list[float] = []
    determinant_errors: list[float] = []
    sample_examples: list[dict[str, object]] = []

    for info in infos:
        other = info.get("other_agent_info_dict", {}).get(INFRA_AGENT)
        if other is None:
            totals["missing_infrastructure_agent"] += 1
            continue
        if "VehLidar2InfLidar_rotation" not in other or "VehLidar2InfLidar_translation" not in other:
            totals["missing_vehicle_to_infrastructure_calibration"] += 1
            continue

        rotation = np.asarray(other["VehLidar2InfLidar_rotation"], dtype=np.float64)
        translation = np.asarray(other["VehLidar2InfLidar_translation"], dtype=np.float64)
        boxes = np.asarray(info["gt_boxes"], dtype=np.float64)
        valid = np.asarray(info["valid_flag"], dtype=bool)
        centers_vehicle = boxes[valid, :3]
        centers_infrastructure = vehicle_to_infrastructure(centers_vehicle, rotation, translation)
        vehicle_mask = in_xy_range(centers_vehicle, VEHICLE_RANGE)
        infrastructure_mask = in_xy_range(centers_infrastructure, INFRASTRUCTURE_RANGE)

        totals["frames_with_calibration"] += 1
        totals["valid_gt_boxes"] += int(len(centers_vehicle))
        totals["vehicle_range_boxes"] += int(vehicle_mask.sum())
        totals["infrastructure_range_boxes_after_transform"] += int(infrastructure_mask.sum())
        totals["infrastructure_range_boxes_if_untransformed"] += int(
            in_xy_range(centers_vehicle, INFRASTRUCTURE_RANGE).sum()
        )

        if len(centers_vehicle):
            recovered = (centers_infrastructure - translation) @ rotation
            round_trip_errors.append(float(np.abs(recovered - centers_vehicle).max()))
        orthogonality_errors.append(float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord=np.inf)))
        determinant_errors.append(float(abs(np.linalg.det(rotation) - 1.0)))

        if len(sample_examples) < 3:
            sample_examples.append(
                {
                    "token": info["token"],
                    "valid_box_count": int(len(centers_vehicle)),
                    "vehicle_range_boxes": int(vehicle_mask.sum()),
                    "infrastructure_range_boxes_after_transform": int(infrastructure_mask.sum()),
                    "infrastructure_range_boxes_if_untransformed": int(
                        in_xy_range(centers_vehicle, INFRASTRUCTURE_RANGE).sum()
                    ),
                    "first_vehicle_center": centers_vehicle[0].round(6).tolist() if len(centers_vehicle) else None,
                    "first_infrastructure_center": centers_infrastructure[0].round(6).tolist()
                    if len(centers_infrastructure)
                    else None,
                }
            )

    frames = totals["frames_with_calibration"]
    report = {
        "purpose": "D0 coordinate/GT audit for UniV2X cooperative 3D detection",
        "ann_file": str(args.ann_file),
        "metadata": payload["metadata"],
        "frames_requested": len(infos),
        "vehicle_point_cloud_range": VEHICLE_RANGE,
        "infrastructure_point_cloud_range": INFRASTRUCTURE_RANGE,
        "calibration_semantics": "VehLidar2InfLidar maps vehicle-LiDAR centers to infrastructure-LiDAR centers",
        "counts": dict(totals),
        "mean_valid_boxes_per_calibrated_frame": totals["valid_gt_boxes"] / frames if frames else None,
        "mean_vehicle_range_boxes_per_calibrated_frame": totals["vehicle_range_boxes"] / frames if frames else None,
        "mean_infrastructure_range_boxes_after_transform_per_calibrated_frame": (
            totals["infrastructure_range_boxes_after_transform"] / frames if frames else None
        ),
        "mean_infrastructure_range_boxes_if_untransformed_per_calibrated_frame": (
            totals["infrastructure_range_boxes_if_untransformed"] / frames if frames else None
        ),
        "max_vehicle_to_infrastructure_to_vehicle_round_trip_error_m": max(round_trip_errors, default=None),
        "max_rotation_orthogonality_error": max(orthogonality_errors, default=None),
        "max_rotation_determinant_error": max(determinant_errors, default=None),
        "examples": sample_examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
