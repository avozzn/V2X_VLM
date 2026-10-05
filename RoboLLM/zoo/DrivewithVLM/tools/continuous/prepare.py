"""Rebuild exact existing split with numeric targets and independent RSU GT.

Never reads GT cooperative objects into model inputs. Default speed policy is
explicitly disabled for ALL agents: source GT velocity may use future frames.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from tools.generate_v2x_planning_v2 import (
    build_samples, image_paths, lidar_to_global, load_info,
)
from tools.continuous.ego_perception import load_vehicle_gt

CLASSES = {"car": 0, "truck": 0, "bus": 0, "van": 0,
           "pedestrian": 1, "bicycle": 2, "motorcycle": 2,
           "cyclist": 2, "tricyclist": 2, "motorcyclist": 2, "barrowlist": 2}


def load_infrastructure_gt(frame, infrastructure_root, timestamp_us):
    """Read current native RSU labels; PKL interpolation/velocity is never an input."""
    if int(frame["pointcloud_timestamp"]) != int(timestamp_us):
        raise ValueError("Infrastructure label timestamp differs from selected source")
    root = Path(infrastructure_root).resolve()
    path = (root / frame["label_lidar_std_path"]).resolve()
    if not path.is_relative_to(root / "label/virtuallidar"):
        raise ValueError("Infrastructure GT must come from label/virtuallidar")
    content = path.read_bytes()
    rows = json.loads(content)
    boxes, names = [], []
    for row in rows:
        xyz = [float(row["3d_location"][axis]) for axis in "xyz"]
        wlh = [float(row["3d_dimensions"][axis]) for axis in "wlh"]
        yaw = float(row["rotation"])
        if not np.isfinite(xyz + wlh + [yaw]).all() or min(wlh) <= 0:
            raise ValueError("Invalid native infrastructure GT box")
        # Existing transform_geometry consumes UniV2X's stored box convention.
        boxes.append(xyz + wlh + [-yaw - np.pi / 2])
        names.append(row["type"].lower())
    provenance = {"annotation_source": "infrastructure_native_lidar_gt",
                  "label_path": str(path), "label_sha256": hashlib.sha256(content).hexdigest(),
                  "annotation_tokens": [row.get("token") for row in rows],
                  "track_ids": [row.get("track_id") for row in rows],
                  "native_object_count": len(rows),
                  "image_timestamp_us": int(frame["image_timestamp"])}
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 7), names, provenance


def transform_geometry(boxes, names, rotation, translation, k=16, roi=50.):
    """UniV2X stored box=[xyz,w,l,h,-annotation_yaw-pi/2].

    Undo converter yaw encoding to recover physical length-axis heading.
    Calibration is vehicle -> infra: p_inf=p_veh @ R.T+t.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5):
        raise ValueError("Invalid rigid rotation")
    if not np.isclose(np.linalg.det(rotation), 1., atol=2e-5):
        raise ValueError("Calibration must be a proper rotation")
    if not np.isfinite(boxes).all() or not np.isfinite(translation).all():
        raise ValueError("Nonfinite source geometry")
    centers = (boxes[:, :3] - translation) @ rotation
    theta = -boxes[:, 6] - np.pi / 2
    directions = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(len(theta))]) @ rotation
    yaw = np.arctan2(directions[:, 1], directions[:, 0])
    states = []
    for i, name in enumerate(names):
        cls = CLASSES.get(str(name).lower())
        if cls is None or np.linalg.norm(centers[i, :2]) > roi:
            continue
        if (boxes[i, 3:6] <= 0).any():
            raise ValueError("Nonpositive box dimensions")
        one_hot = np.eye(3)[cls]
        states.append(np.r_[centers[i], boxes[i, [4, 3, 5]],
                            np.sin(yaw[i]), np.cos(yaw[i]), 0., 0., one_hot])
    states.sort(key=lambda row: (np.linalg.norm(row[:2]), *row.tolist()))
    geometry = np.zeros((k, 13), dtype=np.float32)
    mask = np.zeros(k, dtype=bool)
    count = min(k, len(states))
    if count:
        geometry[:count] = np.asarray(states[:count])
        mask[:count] = True
    return geometry, mask


def prepare_split(split, samples_json, coop_info, infra_info, output, k, roi,
                  vehicle_root, vehicle_frames, ego_max_objects,
                  infrastructure_root, infrastructure_frames):
    requested = json.loads(Path(samples_json).read_text())
    rows, by_token = load_info(coop_info)
    rebuilt, excluded, _ = build_samples(rows, by_token)
    numeric = {item["token"]: item for item in rebuilt}
    with open(infra_info, "rb") as stream:
        payload = pickle.load(stream)
    infra = {str(row["token"]): row for row in payload["infos"]}
    result = []
    age_values = []
    for old in requested:
        token = str(old["token"])
        if token not in numeric:
            raise ValueError(f"Cannot rebuild authoritative target for {token}")
        item = numeric[token]
        info = by_token[token]
        ego_perception = load_vehicle_gt(vehicle_frames[token], vehicle_root,
                                         info["timestamp"], ego_max_objects, roi)
        other = info["other_agent_info_dict"]["model_other_agent_inf"]
        source = infra[str(other["token"])]
        # Dataset native pairing sometimes observes infra after the ego time.
        # Select only the latest paired/previous infra observation available
        # by ego time. Never use a later frame just because it was paired.
        while source is not None and float(source["timestamp"]) > float(info["timestamp"]):
            previous = infra.get(str(source.get("prev", "")))
            if previous is not None and previous["scene_token"] != source["scene_token"]:
                previous = None
            source = previous
        if source is None:
            geometry, mask = np.zeros((k, 13), dtype=np.float32), np.zeros(k, dtype=bool)
            age = None
            provenance = {"annotation_source": "infrastructure_native_lidar_gt",
                          "label_path": None, "native_object_count": 0}
        else:
            # Express the previous source frame in the native paired infra
            # frame before applying the audited vehicle->infra calibration.
            source_to_pair = np.linalg.inv(lidar_to_global(other)) @ lidar_to_global(source)
            source_frame = infrastructure_frames[str(source["token"])]
            if str(source_frame["sequence_id"]) != str(source["scene_token"]):
                raise ValueError("Infrastructure native label scene differs from selected source")
            boxes, names, provenance = load_infrastructure_gt(
                source_frame, infrastructure_root, source["timestamp"])
            boxes[:, :3] = boxes[:, :3] @ source_to_pair[:3, :3].T + source_to_pair[:3, 3]
            theta = -boxes[:, 6] - np.pi / 2
            direction = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(len(theta))]) @ source_to_pair[:3, :3].T
            boxes[:, 6] = -np.arctan2(direction[:, 1], direction[:, 0]) - np.pi / 2
            geometry, mask = transform_geometry(
                boxes, names, other["VehLidar2InfLidar_rotation"],
                other["VehLidar2InfLidar_translation"], k, roi)
            age = (float(info["timestamp"]) - float(source["timestamp"])) / 1e6
            age_values.append(age)
        result.append({
            "token": token, "scene_token": item["scene_token"],
            "ego_image": image_paths(info)[0], "timestamp_us": info["timestamp"],
            "planning_targets": item["planning_targets"],
            "ego_perception": ego_perception,
            "ego_state": {"dimensions_lw_m": [float(info["gt_sdc_bbox"][4]),
                                               float(info["gt_sdc_bbox"][3])]},
            "agents": {"source": "oracle", "geometry": geometry.tolist(),
                       **provenance,
                       "mask": mask.tolist(), "source_token": str(source["token"]) if source else None,
                       "timestamp_us": source["timestamp"] if source else None, "age_s": age,
                       "velocity_policy": "disabled_for_all_agents",
                       "velocity_valid": False,
                       "coordinate_frame": "current_ego_lidar",
                       "heading": "physical_length_axis_radians",
                       "pairing": "latest_available_by_ego_time_no_motion_compensation"},
        })
    path = Path(output) / f"{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False))
    return {"samples": len(result), "scenes": sorted({x["scene_token"] for x in result}),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "agent_count": sum(sum(x["agents"]["mask"]) for x in result),
            "ego_object_count": sum(len(x["ego_perception"]["objects"]) for x in result),
            "ego_empty_samples": sum(not x["ego_perception"]["objects"] for x in result),
            "native_rsu_object_count": sum(x["agents"]["native_object_count"] for x in result),
            "signed_age_min_max_s": [min(age_values), max(age_values)] if age_values else [None, None],
            "future_paired_frames": sum(a < 0 for a in age_values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-dir", default="data/Planning/v2_physics")
    parser.add_argument("--info-root", default="/home/zzn/V2X_VLM/UniV2X/data/infos/V2X-Seq-SPD-New")
    parser.add_argument("--output-dir", default="data/Planning/continuous_v3_native_gt")
    parser.add_argument("--dataset-root", default="data/V2X-Seq-SPD-New")
    parser.add_argument("--ego-max-objects", type=int, default=32)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--roi", type=float, default=50.)
    args = parser.parse_args()
    if args.k <= 0 or args.roi <= 0 or args.ego_max_objects <= 0:
        parser.error("k and roi must be positive")
    out = Path(args.output_dir)
    if any((out / f"{s}.json").exists() for s in ("train", "val", "test")):
        raise FileExistsError("Use a fresh output directory to preserve dataset versions")
    root = Path(args.info_root)
    vehicle_root = Path(args.dataset_root) / "vehicle-side"
    vehicle_frames = {r["frame_id"]: r for r in json.loads((vehicle_root / "data_info.json").read_text())}
    infrastructure_root = Path(args.dataset_root) / "infrastructure-side"
    infrastructure_frames = {r["frame_id"]: r for r in json.loads((infrastructure_root / "data_info.json").read_text())}
    manifest = {"schema": "continuous-v3-native-gt", "k": args.k, "roi_m": args.roi,
                "ego_perception_source": "vehicle_gt", "ego_max_objects": args.ego_max_objects,
                "rsu_perception_source": "infrastructure_native_lidar_gt",
                "rsu_info_usage": "frame_time_pose_only_never_gt_boxes",
                "ego_velocity_policy": "unknown_no_future_tracks",
                "velocity_policy": "disabled_for_all_agents", "splits": {}}
    for split in ("train", "val", "test"):
        source_split = "val" if split == "test" else "train"
        manifest["splits"][split] = prepare_split(
            split, Path(args.planning_dir) / f"{split}_v2x_planning_v2.json",
            root / f"cooperative/spd_infos_temporal_{source_split}_sdc.pkl",
            root / f"infrastructure-side/spd_infos_temporal_{source_split}.pkl",
            out, args.k, args.roi, vehicle_root, vehicle_frames, args.ego_max_objects,
            infrastructure_root, infrastructure_frames)
    scenes = [set(manifest["splits"][s]["scenes"]) for s in ("train", "val", "test")]
    if any(scenes[i] & scenes[j] for i in range(3) for j in range(i)):
        raise ValueError("Scene leakage")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
