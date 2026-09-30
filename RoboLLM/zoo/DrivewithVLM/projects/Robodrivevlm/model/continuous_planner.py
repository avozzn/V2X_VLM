"""Small continuous waypoint head and masked losses for trajectory planning."""

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


class ContinuousTrajectoryHead(nn.Module):
    """Regress ego-frame XY displacements from a per-sample planning latent.

    ``planning_latent`` is expected to be the hidden state at a dedicated
    planning token (shape ``[B, hidden_size]``). The returned waypoints are
    cumulative sums of per-step displacements from the current ego origin.
    """

    def __init__(self, hidden_size: int, horizon: int = 9, mlp_size: int = 512):
        super().__init__()
        if hidden_size <= 0 or horizon <= 0 or mlp_size <= 0:
            raise ValueError("hidden_size, horizon, and mlp_size must be positive")
        self.horizon = horizon
        self.displacement_mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, mlp_size),
            nn.GELU(),
            nn.Linear(mlp_size, horizon * 2),
        )

    def forward(self, planning_latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        if planning_latent.ndim != 2:
            raise ValueError(f"planning_latent must be [B,H], got {tuple(planning_latent.shape)}")
        displacement = self.displacement_mlp(planning_latent).reshape(
            planning_latent.shape[0], self.horizon, 2)
        return {
            "displacement_xy": displacement,
            "trajectory_xy": displacement.cumsum(dim=1),
        }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    denominator = mask.sum()
    if denominator.item() == 0:
        return anchor.sum() * 0.0
    return (values * mask).sum() / denominator


def continuous_trajectory_loss(
    predictions: Dict[str, torch.Tensor],
    target_xy: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    endpoint_weight: float = 1.0,
    displacement_weight: float = 0.0,
    beta: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Compute masked waypoint, final-valid-point, and optional delta losses.

    ``target_mask`` is true only where both XY coordinates are valid. Delta
    supervision additionally requires both adjacent waypoints to be valid.
    All losses are zero (with a valid gradient path) for an entirely masked
    batch rather than producing NaN.
    """
    trajectory = predictions["trajectory_xy"]
    displacement = predictions["displacement_xy"]
    if target_xy.ndim != 3 or target_xy.shape[-1] != 2:
        raise ValueError(f"target_xy must be [B,T,2], got {tuple(target_xy.shape)}")
    expected = trajectory.shape
    if tuple(target_xy.shape) != tuple(expected) or tuple(displacement.shape) != tuple(expected):
        raise ValueError(
            f"prediction and target shapes must match [B,T,2], got trajectory={tuple(expected)}, "
            f"displacement={tuple(displacement.shape)}, target={tuple(target_xy.shape)}")
    valid = torch.as_tensor(target_mask, device=target_xy.device, dtype=torch.bool)
    if valid.ndim == 3 and valid.shape[-1] == 1:
        valid = valid[..., 0]
    if tuple(valid.shape) != tuple(target_xy.shape[:2]):
        raise ValueError(f"target_mask must be [B,T], got {tuple(valid.shape)}")
    valid = valid & torch.isfinite(target_xy).all(dim=-1)
    safe_target = torch.where(valid.unsqueeze(-1), target_xy, torch.zeros_like(target_xy))

    waypoint_raw = F.smooth_l1_loss(trajectory, safe_target, reduction="none", beta=beta).mean(dim=-1)
    waypoint_loss = _masked_mean(waypoint_raw, valid, trajectory)

    batch_size, horizon = valid.shape
    positions = torch.arange(horizon, device=valid.device).expand(batch_size, -1)
    last_index = torch.where(valid, positions, -1).max(dim=1).values
    has_endpoint = last_index >= 0
    if has_endpoint.any():
        rows = torch.arange(batch_size, device=valid.device)[has_endpoint]
        indices = last_index[has_endpoint]
        endpoint_raw = F.smooth_l1_loss(
            trajectory[rows, indices], safe_target[rows, indices], reduction="none", beta=beta
        ).mean(dim=-1)
        endpoint_loss = endpoint_raw.mean()
    else:
        endpoint_loss = trajectory.sum() * 0.0

    target_delta = torch.cat((safe_target[:, :1], safe_target[:, 1:] - safe_target[:, :-1]), dim=1)
    delta_mask = valid.clone()
    if horizon > 1:
        delta_mask[:, 1:] &= valid[:, :-1]
    displacement_raw = F.smooth_l1_loss(
        displacement, target_delta, reduction="none", beta=beta).mean(dim=-1)
    displacement_loss = _masked_mean(displacement_raw, delta_mask, displacement)

    total = waypoint_loss + endpoint_weight * endpoint_loss + displacement_weight * displacement_loss
    return {
        "loss": total,
        "waypoint_loss": waypoint_loss,
        "endpoint_loss": endpoint_loss,
        "displacement_loss": displacement_loss,
    }
