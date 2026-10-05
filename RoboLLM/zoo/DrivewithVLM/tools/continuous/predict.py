"""Reload a saved continuous planner and export evaluator-compatible trajectories."""
import argparse
import json
from pathlib import Path

import torch

from tools.continuous.data import FeatureDataset
from tools.continuous.model import ContinuousPlanner
from tools.continuous.train import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("Prediction output exists")
    model = ContinuousPlanner.load(args.checkpoint).to(args.device)
    dataset = FeatureDataset(args.samples, args.features, model.settings["rsu_mode"], args.limit)
    saved = json.loads((Path(args.checkpoint) / "planner.json").read_text())["metadata"].get("features")
    if saved:
        for key in ["model", "prompt_version", "ego_perception_sources", "readout", "model_config_sha256", "preprocessor_sha256", "dtype", "transformers", "weight_files"]:
            if saved[key] != dataset.metadata[key]:
                raise ValueError(f"Feature contract differs from checkpoint: {key}")
    report, prediction, _, _ = evaluate(model, dataset, torch.device(args.device), args.batch_size)
    rows = [{"token": sample["token"], "coordinate_frame": "current_ego_lidar_xy_m",
             "future_times_s": sample["planning_targets"]["future_times_s"],
             "trajectory": xy.tolist()} for sample, xy in zip(dataset.samples, prediction)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(rows))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
