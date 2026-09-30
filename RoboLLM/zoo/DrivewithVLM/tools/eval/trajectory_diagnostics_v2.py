#!/usr/bin/env python3
"""Evaluate action + trajectory outputs with deterministic physics metrics."""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projects.mmdet3d_plugin.datasets.data_utils.planning_metadata_v2 import (  # noqa: E402
    generate_planning_metadata_v2,
)


LATERAL = {"KEEP_LANE", "TURN_LEFT", "TURN_RIGHT", "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT"}
LONGITUDINAL = {"ACCELERATE", "KEEP_SPEED", "DECELERATE", "STOP"}
PATTERNS = {
    "lateral_action": re.compile(r"^Lateral Action:\s*([A-Z_]+)\s*$", re.MULTILINE),
    "longitudinal_action": re.compile(r"^Longitudinal Action:\s*([A-Z_]+)\s*$", re.MULTILINE),
}
POINT_RE = re.compile(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-json", required=True, type=Path)
    parser.add_argument("--test-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--collision-analysis", type=Path)
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_answer(text):
    parsed = {}
    for key, pattern in PATTERNS.items():
        match = pattern.search(text or "")
        if match is None:
            raise ValueError(f"missing {key}")
        parsed[key] = match.group(1)
    if parsed["lateral_action"] not in LATERAL:
        raise ValueError(f"unknown lateral action {parsed['lateral_action']}")
    if parsed["longitudinal_action"] not in LONGITUDINAL:
        raise ValueError(f"unknown longitudinal action {parsed['longitudinal_action']}")
    trajectory_text = (text or "").split("Trajectory:", 1)
    if len(trajectory_text) != 2:
        raise ValueError("missing Trajectory")
    trajectory = np.asarray(POINT_RE.findall(trajectory_text[1]), dtype=np.float64)
    if trajectory.shape != (9, 2) or not np.isfinite(trajectory).all():
        raise ValueError(f"trajectory must be finite [9,2], got {trajectory.shape}")
    parsed["trajectory"] = trajectory
    return parsed


def derive_metadata(trajectory, current_speed_mps):
    return generate_planning_metadata_v2(
        sdc_planning=trajectory,
        planning_mask=np.ones((9, 2), dtype=np.uint8),
        current_velocity=np.asarray([current_speed_mps, 0.0], dtype=np.float64),
        dt=0.5,
    )


def classification_report(records, key, labels, prediction_key="pred"):
    confusion = defaultdict(Counter)
    for record in records:
        confusion[record["gt_labels"][key]][record[prediction_key][key]] += 1
    per_class = {}
    for label in sorted(labels):
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(row[label] for row in confusion.values())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
    active = [label for label, value in per_class.items() if value["support"] or any(
        record[prediction_key][key] == label for record in records)]
    accuracy = np.mean([
        record["gt_labels"][key] == record[prediction_key][key] for record in records
    ]) if records else 0.0
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(np.mean([per_class[label]["f1"] for label in active])) if active else 0.0,
        "per_class": per_class,
        "confusion_matrix": {label: dict(row) for label, row in confusion.items()},
    }


def main():
    args = parse_args()
    predictions = {str(item["token"]): item.get("answer", "") for item in load_json(args.predictions_json)}
    test_items = {str(item["token"]): item for item in load_json(args.test_json)}
    collisions = {}
    if args.collision_analysis and args.collision_analysis.exists():
        collisions = {str(item["token"]): bool(item.get("collision")) for item in load_json(args.collision_analysis)}

    records, parse_errors = [], []
    for token, item in test_items.items():
        if token not in predictions:
            continue
        try:
            pred = parse_answer(predictions[token])
            gt_text = parse_answer(item["conversations"][1]["value"])
            gt_labels = item["planning_labels_v2"]
            if gt_labels.get("schema_version") != "2.0":
                raise ValueError("planning_labels_v2 must use schema 2.0")
            derived = derive_metadata(pred["trajectory"], gt_labels["current_speed_mps"])
        except Exception as error:
            parse_errors.append({"token": token, "error": f"{type(error).__name__}: {error}"})
            continue
        step_l2 = np.linalg.norm(pred["trajectory"] - gt_text["trajectory"], axis=1)
        records.append({
            "token": token, "pred": pred, "derived": derived,
            "gt_labels": gt_labels, "step_l2": step_l2,
            "collision": collisions.get(token),
        })

    errors = np.asarray([record["step_l2"] for record in records], dtype=np.float64)
    trajectory_metrics = {}
    if errors.size:
        trajectory_metrics = {
            "mean_l2_all_steps": float(errors.mean()),
            "per_step_mean_l2": errors.mean(axis=0).tolist(),
            "per_step_median_l2": np.median(errors, axis=0).tolist(),
            "per_step_p90_l2": np.percentile(errors, 90, axis=0).tolist(),
            "final_l2_over_5m": int((errors[:, -1] > 5).sum()),
            "final_l2_over_10m": int((errors[:, -1] > 10).sum()),
        }
    collision_values = [record["collision"] for record in records if record["collision"] is not None]
    report = {
        "coverage": {
            "expected": len(test_items), "generated": len(predictions),
            "valid_predictions": len(records),
            "rate": len(records) / len(test_items) if test_items else 0.0,
            "parse_errors": parse_errors,
        },
        "trajectory_metrics": trajectory_metrics,
        "text_lateral_metrics": classification_report(records, "lateral_action", LATERAL),
        "text_longitudinal_metrics": classification_report(records, "longitudinal_action", LONGITUDINAL),
        "derived_lateral_metrics": classification_report(records, "lateral_action", LATERAL, "derived"),
        "derived_longitudinal_metrics": classification_report(records, "longitudinal_action", LONGITUDINAL, "derived"),
        "action_consistency": {
            "lateral_rate": float(np.mean([
                record["pred"]["lateral_action"] == record["derived"]["lateral_action"]
                for record in records])) if records else None,
            "longitudinal_rate": float(np.mean([
                record["pred"]["longitudinal_action"] == record["derived"]["longitudinal_action"]
                for record in records])) if records else None,
        },
        "continuous_metrics": {
            "target_end_speed_mae": float(np.mean([
                abs(record["derived"]["target_end_speed_mps"] - record["gt_labels"]["target_end_speed_mps"])
                for record in records])) if records else None,
            "mean_acceleration_mae": float(np.mean([
                abs(record["derived"]["mean_acceleration_mps2"] - record["gt_labels"]["mean_acceleration_mps2"])
                for record in records])) if records else None,
        },
        "collision_any_rate": float(np.mean(collision_values)) if collision_values else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
