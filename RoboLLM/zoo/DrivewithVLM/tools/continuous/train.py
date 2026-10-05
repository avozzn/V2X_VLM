"""Train and evaluate independently initialized same-size F0/F1/F2 heads.

Frozen ego feature extraction is separate; training needs no VLM on GPU.
No synthetic feature results are accepted as real F0 learning evidence.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.continuous.data import FeatureDataset, collate_features
from tools.continuous.model import ContinuousPlanner, waypoint_loss


def metrics(prediction, target, valid):
    errors = np.linalg.norm(prediction - np.where(valid[..., None], target, prediction), axis=-1)
    counts = valid.sum(0)
    per_step = np.divide((errors * valid).sum(0), counts,
                         out=np.full(len(counts), np.nan), where=counts > 0)
    indices = [1, 3, 5]
    return {"l2_1_2_3s": per_step[indices].tolist(),
            "avg_l2_1_2_3s": float(np.nanmean(per_step[indices])),
            "mean_l2_9steps": float((errors * valid).sum() / valid.sum()),
            "fde_4_5s": float(per_step[-1]), "per_step_l2": per_step.tolist(),
            "valid_counts": counts.tolist(),
            "prediction_std_xy_by_step": prediction.std(0).tolist()}


def baseline_metrics(train, val):
    target = np.asarray([s["planning_targets"]["future_xy"] for s in val.samples])
    valid = np.asarray([s["planning_targets"]["future_mask"] for s in val.samples], dtype=bool)
    rows = np.asarray([s["planning_targets"]["future_xy"] for s in train.samples])
    masks = np.asarray([s["planning_targets"]["future_mask"] for s in train.samples], dtype=bool)
    mean = np.where(masks[..., None], rows, 0).sum(0) / masks.sum(0)[:, None].clip(1)
    velocity = np.asarray([s["planning_targets"]["ego_velocity_xy_mps"] for s in val.samples])
    velocity_valid = np.asarray([s["planning_targets"]["ego_velocity_valid"] for s in val.samples], dtype=bool)
    times = np.asarray([s["planning_targets"]["future_times_s"] for s in val.samples])
    straight = np.zeros_like(velocity)
    straight[:, 0] = np.linalg.norm(velocity, axis=-1)
    report = {"train_mean": metrics(np.broadcast_to(mean, target.shape), target, valid),
              "speed_valid_samples": int(velocity_valid.sum())}
    if velocity_valid.any():
        for name, speed in [("constant_velocity", velocity), ("straight", straight)]:
            report[name] = metrics((times[..., None] * speed[:, None])[velocity_valid],
                                   target[velocity_valid], valid[velocity_valid])
    return report, velocity_valid


@torch.no_grad()
def evaluate(model, dataset, device, batch_size):
    model.eval()
    predictions, targets, masks = [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=collate_features):
        output = model(batch["h_ego"].to(device), batch["agent_geometry"].to(device),
                       batch["agent_mask"].to(device))
        predictions.append(output["trajectory"].cpu().numpy())
        targets.append(batch["future_xy"].numpy())
        masks.append(batch["future_mask"].numpy())
    pred, target, valid = map(np.concatenate, (predictions, targets, masks))
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite predictions")
    return metrics(pred, target, valid), pred, target, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-samples", required=True)
    parser.add_argument("--val-samples", required=True)
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--val-features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rsu-mode", choices=["zero", "oracle", "detector"], default="zero")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.rsu_mode != "zero":
        parser.error("Cached M1 RSU bypass disabled. Use python -m tools.continuous.online train")
    destination = Path(args.output_dir)
    if destination.exists():
        raise FileExistsError("Use an independent fresh run directory")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    train = FeatureDataset(args.train_samples, args.train_features, args.rsu_mode, args.limit)
    val = FeatureDataset(args.val_samples, args.val_features, args.rsu_mode, args.limit)
    train_scenes = {s["scene_token"] for s in train.samples}
    if train_scenes & {s["scene_token"] for s in val.samples}:
        raise ValueError("Train/validation scenes overlap")
    for key in ["model", "prompt_version", "ego_perception_sources", "readout", "model_config_sha256", "preprocessor_sha256", "dtype", "transformers", "weight_files"]:
        if train.metadata[key] != val.metadata[key]:
            raise ValueError(f"Train/val frozen feature contract differs: {key}")
    device = torch.device(args.device)
    model = ContinuousPlanner(train.metadata["ego_dim"], args.rsu_mode,
                              head_dim=args.head_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_features)
    baselines, speed_valid = baseline_metrics(train, val)
    destination.mkdir(parents=True)
    metadata = {"arguments": vars(args), "features": train.metadata,
                "validation_features": val.metadata, "baselines": baselines,
                "normalization": "fixed physical geometry scales; target in meters",
                "loss": "waypoint Smooth-L1 beta=1m; coordinate mean; valid-point mean",
                "velocity_policy": sorted({s["agents"]["velocity_policy"] for s in train.samples}),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "effective_parameters": sum(p.numel() for n, p in model.named_parameters()
                                            if args.rsu_mode != "zero" or not n.startswith("agent_encoder"))}
    (destination / "run.json").write_text(json.dumps(metadata, indent=2))
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, points = 0., 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["h_ego"].to(device), batch["agent_geometry"].to(device),
                           batch["agent_mask"].to(device))
            loss = waypoint_loss(output["trajectory"], batch["future_xy"].to(device),
                                 batch["future_mask"].to(device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite train loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            count = int(batch["future_mask"].sum())
            loss_sum += float(loss.detach()) * count
            points += count
        report, pred, target, valid = evaluate(model, val, device, args.batch_size)
        if speed_valid.any():
            report["speed_valid_subset"] = metrics(pred[speed_valid], target[speed_valid], valid[speed_valid])
        row = {"epoch": epoch, "train_loss": loss_sum / points, "validation": report}
        with (destination / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(f"epoch={epoch} loss={row['train_loss']:.5f} val_L2={report['avg_l2_1_2_3s']:.5f}", flush=True)
        if report["avg_l2_1_2_3s"] < best:
            best = report["avg_l2_1_2_3s"]
            model.save(destination / "best", {**metadata, "epoch": epoch, "validation": report})
            predictions = [{"token": s["token"], "scene_token": s["scene_token"],
                            "coordinate_frame": "current_ego_lidar_xy_m",
                            "future_times_s": s["planning_targets"]["future_times_s"],
                            "trajectory": trajectory.tolist()}
                           for s, trajectory in zip(val.samples, pred)]
            (destination / "val_predictions.json").write_text(json.dumps(predictions))
            (destination / "best_metrics.json").write_text(json.dumps(report, indent=2))
    model.save(destination / "last", metadata)


if __name__ == "__main__":
    main()
