#!/usr/bin/env python3
"""Model-agnostic metrics for numeric ego-frame trajectory predictions."""

import argparse
import json
from pathlib import Path

import numpy as np


HORIZONS_S = (1.0, 2.0, 3.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-json", required=True, type=Path)
    parser.add_argument("--ground-truth-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def keyed_rows(payload, label):
    if isinstance(payload, dict):
        payload = payload.get("predictions" if label == "predictions" else "samples", payload)
    if isinstance(payload, dict):
        payload = list(payload.values())
    if not isinstance(payload, list):
        raise ValueError(f"{label} JSON must contain a list of rows")
    rows = {}
    for row in payload:
        if not isinstance(row, dict) or "token" not in row:
            raise ValueError(f"every {label} row must be an object with token")
        token = str(row["token"])
        if token in rows:
            raise ValueError(f"duplicate {label} token: {token}")
        rows[token] = row
    return rows


def as_trajectory(value, name):
    trajectory = np.asarray(value, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError(f"{name} must have shape [T,2], got {trajectory.shape}")
    return trajectory


def summarize(values):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else None


def main():
    args = parse_args()
    gt_rows = keyed_rows(read_json(args.ground_truth_json), "ground truth")
    pred_rows = keyed_rows(read_json(args.predictions_json), "predictions")
    per_horizon = {f"{h:g}s": [] for h in HORIZONS_S}
    per_step = {}
    records, missing, invalid = [], [], []

    for token, gt_row in gt_rows.items():
        prediction = pred_rows.get(token)
        if prediction is None:
            missing.append(token)
            continue
        targets = gt_row.get("planning_targets", {})
        if not targets:
            # Supports legacy generated JSON: trajectory is in the assistant answer.
            invalid.append({"token": token, "reason": "missing planning_targets"})
            continue
        try:
            gt = as_trajectory(targets["future_xy"], "future_xy")
            gt_times = np.asarray(targets["future_times_s"], dtype=np.float64)
            gt_mask = np.asarray(targets["future_mask"], dtype=bool).reshape(-1)
            pred = as_trajectory(prediction.get("future_xy", prediction.get("trajectory")), "prediction")
            pred_times = np.asarray(prediction.get("future_times_s", gt_times), dtype=np.float64)
            pred_mask = np.asarray(prediction.get("future_mask", np.ones(len(pred), dtype=bool)), dtype=bool).reshape(-1)
            if len(gt) != len(gt_times) or len(gt) != len(gt_mask):
                raise ValueError("ground-truth trajectory, times, and mask lengths differ")
            if len(pred) != len(pred_times) or len(pred) != len(pred_mask):
                raise ValueError("prediction trajectory, times, and mask lengths differ")
            if not np.isfinite(gt_times).all() or not np.isfinite(pred_times).all():
                raise ValueError("timestamps contain non-finite values")
            if np.any(np.diff(pred_times) <= 0) or np.any(np.diff(gt_times) <= 0):
                raise ValueError("timestamps must be strictly increasing")
            if not np.isfinite(pred).all():
                # Preserve finite points; a non-finite point is simply not a valid prediction.
                pred_mask &= np.isfinite(pred).all(axis=1)
        except Exception as error:
            invalid.append({"token": token, "reason": f"{type(error).__name__}: {error}"})
            continue

        row_errors = {}
        for index, time_s in enumerate(gt_times):
            match = np.flatnonzero(np.isclose(pred_times, time_s, atol=1e-4, rtol=0.0))
            key = f"{time_s:g}s"
            bucket = per_step.setdefault(key, [])
            if gt_mask[index] and len(match) and pred_mask[match[0]]:
                error = float(np.linalg.norm(pred[match[0]] - gt[index]))
                bucket.append(error)
                row_errors[key] = error

        for horizon in HORIZONS_S:
            gt_idx = np.flatnonzero(np.isclose(gt_times, horizon, atol=1e-4, rtol=0.0))
            pred_idx = np.flatnonzero(np.isclose(pred_times, horizon, atol=1e-4, rtol=0.0))
            if len(gt_idx) and len(pred_idx):
                gi, pi = int(gt_idx[0]), int(pred_idx[0])
                if gt_mask[gi] and pred_mask[pi]:
                    per_horizon[f"{horizon:g}s"].append(float(np.linalg.norm(pred[pi] - gt[gi])))
        records.append({"token": token, "valid_horizons": row_errors})

    horizon_metrics = {name: {"valid_samples": len(vals), "mean_l2_m": summarize(vals)}
                       for name, vals in per_horizon.items()}
    main_values = [item["mean_l2_m"] for item in horizon_metrics.values() if item["mean_l2_m"] is not None]
    report = {
        "metric_schema_version": "1.0",
        "coordinate_frame": "current_ego_lidar_xy_m",
        "coverage": {
            "ground_truth_samples": len(gt_rows),
            "prediction_samples": len(pred_rows),
            "matched_samples": len(records),
            "missing_predictions": len(missing),
            "invalid_predictions": len(invalid),
            "valid_at_all_primary_horizons": sum(
                all(name in row["valid_horizons"] for name in per_horizon) for row in records),
            "missing_tokens": missing,
            "invalid": invalid,
        },
        "l2_at_horizon": horizon_metrics,
        "mean_l2_1_2_3s_m": float(np.mean(main_values)) if len(main_values) == 3 else None,
        "fde_3s_m": horizon_metrics["3s"]["mean_l2_m"],
        "fde_4_5s_m": summarize(per_step.get("4.5s", [])),
        "mean_l2_all_valid_steps_m": summarize([v for values in per_step.values() for v in values]),
        "per_step_l2": {key: {"valid_samples": len(values), "mean_l2_m": summarize(values)}
                        for key, values in sorted(per_step.items(), key=lambda pair: float(pair[0][:-1]))},
        "sample_diagnostics": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
