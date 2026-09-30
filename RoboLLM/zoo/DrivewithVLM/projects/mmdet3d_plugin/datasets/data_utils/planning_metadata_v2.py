"""Pure NumPy planning metadata and physics helpers."""

import numpy as np


PLANNING_METADATA_SCHEMA_VERSION = "2.0"


def generate_planning_metadata_v2(
        *, sdc_planning, planning_mask, current_velocity, dt=0.5):
    """Derive actions and continuous motion labels from a 9-point trajectory."""
    trajectory = np.asarray(sdc_planning, dtype=np.float64)
    mask = np.asarray(planning_mask)
    velocity = np.asarray(current_velocity)

    if trajectory.ndim == 3 and trajectory.shape[0] == 1:
        trajectory = trajectory[0]
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if trajectory.ndim != 2 or trajectory.shape[0] != 9 or trajectory.shape[1] < 2:
        raise ValueError(
            f"sdc_planning must have shape [1, 9, 2+] or [9, 2+], got {trajectory.shape}")
    if mask.shape != (9, 2):
        raise ValueError(
            f"planning_mask must have shape [1, 9, 2] or [9, 2], got {mask.shape}")
    if not np.isfinite(trajectory[:, :2]).all():
        raise ValueError("sdc_planning contains non-finite coordinates")
    if not np.all(mask.astype(bool)):
        raise ValueError("Meta Action V2 requires all 9 future trajectory points to be observed")
    if velocity.dtype.kind not in "fc" or velocity.shape != (2,):
        raise TypeError("current_velocity must be a length-2 floating-point vector, not a mask")
    velocity = velocity.astype(np.float64)
    if not np.isfinite(velocity).all() or dt <= 0:
        raise ValueError("current_velocity and dt must be finite; dt must be positive")

    xy = trajectory[:, :2]
    points = np.vstack([np.zeros((1, 2), dtype=np.float64), xy])
    displacements = np.diff(points, axis=0)
    segment_speeds = np.linalg.norm(displacements, axis=1) / dt
    current_speed = float(np.linalg.norm(velocity))
    target_end_speed = float(np.median(segment_speeds[-2:]))
    speed_times = np.arange(10, dtype=np.float64) * dt
    speed_profile = np.concatenate([[current_speed], segment_speeds])
    mean_acceleration = float(np.polyfit(speed_times, speed_profile, 1)[0])

    if target_end_speed < 0.5:
        longitudinal_action = "STOP"
    elif mean_acceleration > 0.2 and target_end_speed - current_speed > 0.5:
        longitudinal_action = "ACCELERATE"
    elif mean_acceleration < -0.2 and current_speed - target_end_speed > 0.5:
        longitudinal_action = "DECELERATE"
    else:
        longitudinal_action = "KEEP_SPEED"

    valid_headings = np.unwrap(
        np.arctan2(displacements[:, 1], displacements[:, 0]))[segment_speeds >= 0.5]
    if valid_headings.size >= 2:
        final_heading = float(np.median(valid_headings[-2:]))
        cumulative_curvature = float(np.sum(np.abs(np.diff(valid_headings))))
    elif valid_headings.size == 1:
        final_heading = float(valid_headings[-1])
        cumulative_curvature = 0.0
    else:
        final_heading = 0.0
        cumulative_curvature = 0.0

    final_lateral_offset = float(xy[-1, 1])
    total_displacement = float(np.linalg.norm(xy[-1]))
    if total_displacement < 1.0:
        lateral_action = "KEEP_LANE"
    elif abs(final_heading) >= np.deg2rad(15.0):
        lateral_action = "TURN_LEFT" if final_heading > 0 else "TURN_RIGHT"
    elif abs(final_lateral_offset) >= 2.0:
        lateral_action = (
            "LANE_CHANGE_LEFT" if final_lateral_offset > 0 else "LANE_CHANGE_RIGHT")
    else:
        lateral_action = "KEEP_LANE"

    return {
        "schema_version": PLANNING_METADATA_SCHEMA_VERSION,
        "lateral_action": lateral_action,
        "longitudinal_action": longitudinal_action,
        "current_speed_mps": current_speed,
        "target_end_speed_mps": target_end_speed,
        "mean_acceleration_mps2": mean_acceleration,
        "final_lateral_offset_m": final_lateral_offset,
        "final_heading_change_deg": float(np.degrees(final_heading)),
        "cumulative_heading_change_deg": float(np.degrees(cumulative_curvature)),
    }
