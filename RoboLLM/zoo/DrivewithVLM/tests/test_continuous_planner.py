import importlib.util
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "projects/Robodrivevlm/model/continuous_planner.py"
SPEC = importlib.util.spec_from_file_location("continuous_planner", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ContinuousTrajectoryHead = MODULE.ContinuousTrajectoryHead
continuous_trajectory_loss = MODULE.continuous_trajectory_loss


def test_head_outputs_cumulative_ego_waypoints():
    head = ContinuousTrajectoryHead(hidden_size=8, horizon=3, mlp_size=16)
    output = head(torch.randn(2, 8))
    assert output["displacement_xy"].shape == (2, 3, 2)
    assert output["trajectory_xy"].shape == (2, 3, 2)
    assert torch.allclose(output["trajectory_xy"], output["displacement_xy"].cumsum(dim=1))


def test_masked_loss_trains_each_sample_and_ignores_invalid_waypoints():
    trajectory = torch.zeros((2, 3, 2), requires_grad=True)
    displacement = torch.zeros((2, 3, 2), requires_grad=True)
    targets = torch.tensor([
        [[1.0, 0.0], [float("nan"), float("nan")], [99.0, 99.0]],
        [[0.0, 0.0], [1.0, 1.0], [2.0, 1.0]],
    ])
    mask = torch.tensor([[True, False, False], [True, True, True]])

    losses = continuous_trajectory_loss(
        {"trajectory_xy": trajectory, "displacement_xy": displacement},
        targets,
        mask,
        displacement_weight=0.1,
    )
    losses["loss"].backward()

    assert torch.isfinite(losses["loss"])
    assert trajectory.grad[0, 0].abs().sum() > 0
    assert trajectory.grad[1].abs().sum() > 0
    assert trajectory.grad[0, 1:].abs().sum() == 0
