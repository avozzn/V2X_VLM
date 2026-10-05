"""Strict observation-only prompts and numeric batching for cached features."""
import json
import hashlib
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tools.continuous.ego_perception import perception_text

PROMPT_VERSION = "ego-perception-observation-v2"


def observation_prompt(sample):
    targets = sample["planning_targets"]
    # Do not consume conversations, planning_labels or any RSU field.
    velocity = targets["ego_velocity_xy_mps"]
    history = targets["history_xy"]
    history_mask = targets["history_mask"]
    return ("You are an autonomous driving planner. Use the ego camera, vehicle perception, "
            "and observed ego state to plan a safe trajectory.\n"
            "Coordinates: current ego LiDAR frame; x forward, y left, z up; meters.\n"
            "Predict nine future waypoints at 0.5-second intervals, through 4.5 seconds.\n"
            + perception_text(sample["ego_perception"])
            + f"Ego dimensions (length, width; m): {sample['ego_state']['dimensions_lw_m']}.\n"
            f"Current ego-frame velocity (m/s): {velocity}.\n"
            f"Observed ego acceleration (m/s^2): {targets['ego_acceleration_xy_mps2']}.\n"
            f"Past ego-frame positions (m): {history}.\n"
            f"Past observation mask: {history_mask}.\n"
            "Represent the scene for predicting nine future ego waypoints.")


def observation_signature(samples):
    identity = [{"token": s["token"], "image": s["ego_image"],
                 "timestamp_us": s["timestamp_us"], "prompt": observation_prompt(s),
                 "ego_perception_source": s["ego_perception"]["source"],
                 "ego_perception_frame": s["ego_perception"].get("source_frame"),
                 "ego_perception_timestamp_us": s["ego_perception"]["timestamp_us"]}
                for s in samples]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


class FeatureDataset(Dataset):
    def __init__(self, samples_path, features_path, rsu_mode="zero", limit=None):
        self.samples = json.loads(Path(samples_path).read_text())
        if limit:
            self.samples = self.samples[:limit]
        if not self.samples:
            raise ValueError("Empty dataset")
        payload = torch.load(features_path, map_location="cpu", weights_only=True)
        self.metadata = payload["metadata"]
        self.features = payload["features"]
        self.rsu_mode = rsu_mode
        if self.metadata.get("prompt_version") != PROMPT_VERSION:
            raise ValueError("Rebuild ego features for the updated perception prompt")
        if self.metadata.get("ego_perception_sources") != sorted({
                s["ego_perception"]["source"] for s in self.samples}):
            raise ValueError("Ego perception source differs from the feature cache")
        expected = hashlib.sha256("\n".join(observation_prompt(s) for s in self.samples).encode()).hexdigest()
        if self.metadata.get("prompt_sha256") != expected:
            raise ValueError("Stale feature cache: observation prompt/order changed")
        if self.metadata.get("observation_sha256") != observation_signature(self.samples):
            raise ValueError("Stale feature cache: ego image/token/timestamp changed")
        if len({s["token"] for s in self.samples}) != len(self.samples):
            raise ValueError("Duplicate sample tokens")
        for sample in self.samples:
            if int(sample["ego_perception"]["timestamp_us"]) != int(sample["timestamp_us"]):
                raise ValueError("Ego perception must match the current observation")
            target = sample["planning_targets"]
            if target.get("coordinate_frame") != "current_ego_lidar_xy_m":
                raise ValueError("Unsupported target coordinate frame")
            times = torch.tensor(target["future_times_s"])
            if times.shape != (9,) or not torch.allclose(times, torch.arange(1, 10) * .5):
                raise ValueError("First-stage evaluator requires nine 0.5s future steps")
            agents = sample["agents"]
            if agents["coordinate_frame"] != "current_ego_lidar":
                raise ValueError("Agent coordinate contract mismatch")
            if agents["velocity_policy"] != "disabled_for_all_agents":
                raise ValueError("First-stage runs require the unified velocity-disabled setting")
            if sample["token"] not in self.features:
                raise ValueError(f"Missing feature for {sample['token']}")
            if rsu_mode != "zero" and sample["agents"]["source"] != rsu_mode:
                raise ValueError("Oracle/detector source mismatch")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        target = sample["planning_targets"]
        agents = sample["agents"]
        return {
            "h_ego": self.features[sample["token"]].float(),
            "agent_geometry": torch.tensor(agents["geometry"], dtype=torch.float32),
            "agent_mask": torch.tensor(agents["mask"], dtype=torch.bool),
            "future_xy": torch.tensor(target["future_xy"], dtype=torch.float32),
            "future_mask": torch.tensor(target["future_mask"], dtype=torch.bool),
            "index": index,
        }


def collate_features(instances):
    # Stack every instance, never silently discard batch[1:].
    return {key: torch.stack([item[key] for item in instances])
            if key != "index" else torch.tensor([item[key] for item in instances])
            for key in instances[0]}
