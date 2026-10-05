"""Audited vehicle-only perception contract and shared GT/detector formatter."""
import json
import hashlib
import math
from pathlib import Path

FRAME = "current_ego_lidar"
SCHEMA = "ego-perception-v1"
CLASSES = {"car", "truck", "bus", "van", "pedestrian", "cyclist",
           "bicycle", "motorcycle", "motorcyclist", "tricyclist"}


def load_vehicle_gt(frame, vehicle_root, ego_timestamp_us, max_objects=32, roi=50.):
    """Native vehicle LiDAR labels only; no cooperative boxes or future tracks."""
    timestamp = int(frame["pointcloud_timestamp"])
    if timestamp != int(ego_timestamp_us):
        raise ValueError("Vehicle GT must be the current ego LiDAR frame")
    root = Path(vehicle_root).resolve()
    path = (root / frame["label_lidar_std_path"]).resolve()
    if not path.is_relative_to(root / "label/lidar"):
        raise ValueError("Vehicle GT must come from vehicle-side/label/lidar")
    objects = []
    content = path.read_bytes()
    native_rows = json.loads(content)
    for row in native_rows:
        name = row["type"].lower()
        if name not in CLASSES:
            continue
        xyz = [float(row["3d_location"][axis]) for axis in "xyz"]
        lwh = [float(row["3d_dimensions"][axis]) for axis in "lwh"]
        yaw = float(row["rotation"])
        if not all(math.isfinite(v) for v in xyz + lwh + [yaw]) or min(lwh) <= 0:
            raise ValueError("Invalid native vehicle GT box")
        if math.hypot(*xyz[:2]) > roi:
            continue
        objects.append({"class": name, "position_xyz_m": xyz, "size_lwh_m": lwh,
                        "heading_rad": yaw, "velocity_xy_mps": None,
                        "velocity_valid": False, "confidence": None,
                        "annotation_token": row.get("token"), "track_id": row.get("track_id"),
                        "occluded_state": row.get("occluded_state"),
                        "truncated_state": row.get("truncated_state")})
    objects.sort(key=lambda obj: (math.hypot(*obj["position_xyz_m"][:2]),
                                 obj["class"], *obj["position_xyz_m"], obj["heading_rad"]))
    return {"schema": SCHEMA, "source": "vehicle_gt", "coordinate_frame": FRAME,
            "source_frame": frame["frame_id"], "timestamp_us": timestamp,
            "image_timestamp_us": int(frame["image_timestamp"]) if "image_timestamp" in frame else None,
            "label_sha256": hashlib.sha256(content).hexdigest(),
            "native_object_count": len(native_rows),
            "label_path": str(path), "roi_m": roi, "max_objects": max_objects,
            "velocity_policy": "unknown_no_future_tracks",
            "objects": objects[:max_objects], "objects_before_topk": len(objects)}


def perception_text(perception):
    """The text contains observations, never source-specific Oracle hints."""
    if (perception.get("schema") != SCHEMA or
            perception.get("coordinate_frame") != FRAME or
            perception.get("source") not in {"vehicle_gt", "vehicle_detector"}):
        raise ValueError("Unsupported ego perception contract")
    lines = ["Ego perception (vehicle sensors; positions in current ego LiDAR meters):"]
    if not perception["objects"]:
        return lines[0] + "\n - No objects reported within the configured ROI.\n"
    for obj in perception["objects"]:
        xyz, lwh, yaw = obj["position_xyz_m"], obj["size_lwh_m"], obj["heading_rad"]
        if len(xyz) != 3 or len(lwh) != 3 or not all(
                math.isfinite(v) for v in xyz + lwh + [yaw]) or min(lwh) <= 0:
            raise ValueError("Invalid ego perception geometry")
        velocity = "unknown"
        if obj["velocity_valid"]:
            values = obj["velocity_xy_mps"]
            if values is None or len(values) != 2 or not all(math.isfinite(v) for v in values):
                raise ValueError("Invalid observed object velocity")
            velocity = f"({values[0]:.2f}, {values[1]:.2f})"
        confidence = obj.get("confidence")
        if confidence is not None and (not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("Invalid detector confidence")
        score = "unknown" if confidence is None else f"{confidence:.3f}"
        lines.append(f" - {obj['class']}: position_xyz=({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}); "
                     f"size_lwh=({lwh[0]:.2f}, {lwh[1]:.2f}, {lwh[2]:.2f}); "
                     f"heading_rad={yaw:.3f}; velocity_xy_mps={velocity}; confidence={score}.")
    return "\n".join(lines) + "\n"
