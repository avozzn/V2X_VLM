"""Same-size F0/F1/F2 planner and observation-only frozen LLaVA readout."""
import json
import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# Reuse the existing repository head without importing the MMDetection
# registry-heavy projects.Robodrivevlm package in this isolated CPU path.
_head_path = Path(__file__).resolve().parents[2] / "projects/Robodrivevlm/model/continuous_planner.py"
_spec = importlib.util.spec_from_file_location("continuous_waypoint_head", _head_path)
_head_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_head_module)
ContinuousTrajectoryHead = _head_module.ContinuousTrajectoryHead


def last_valid_hidden(hidden, mask):
    """Gather after image expansion; works with left/right/irregular padding."""
    if hidden.shape[:2] != mask.shape or not mask.bool().any(1).all():
        raise ValueError("Readout requires the merged nonempty attention mask")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    indices = positions.masked_fill(~mask.bool(), -1).max(1).values
    return hidden[torch.arange(len(hidden), device=hidden.device), indices]


class FrozenLlavaReadout(nn.Module):
    """Capture decoder final-normalized output and its actual merged mask.

    The local 4.45.2 LLaVA forwards the merged mask into language_model.model.
    Hooks avoid requesting every decoder hidden layer. No assistant labels are
    accepted. One-logit output avoids materializing full sequence vocabulary logits.
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, **inputs):
        if "labels" in inputs:
            raise ValueError("Frozen readout accepts observations, never answer labels")
        captured = {}
        decoder = self.backbone.language_model.model

        def before(module, args, kwargs):
            captured["mask"] = kwargs["attention_mask"]

        def after(module, args, output):
            hidden = output.last_hidden_state
            captured["readout"] = last_valid_hidden(hidden, captured["mask"])
            captured["merged_length"] = hidden.shape[1]

        pre = decoder.register_forward_pre_hook(before, with_kwargs=True)
        post = decoder.register_forward_hook(after)
        try:
            self.backbone(**inputs, use_cache=False, return_dict=True,
                          output_hidden_states=False, num_logits_to_keep=1)
        finally:
            pre.remove()
            post.remove()
        if "readout" not in captured:
            raise RuntimeError("Unsupported LLaVA decoder; readout hook not invoked")
        return captured["readout"].float(), captured["merged_length"]


class ContinuousPlanner(nn.Module):
    """F0/F1/F2 share all parameters; only the source of h_rsu changes."""
    def __init__(self, ego_dim, rsu_mode="zero", agent_dim=256, head_dim=512,
                 horizon=9, beta=1.0):
        super().__init__()
        if rsu_mode not in {"zero", "oracle", "detector"}:
            raise ValueError("rsu_mode must be zero/oracle/detector")
        self.settings = dict(ego_dim=ego_dim, rsu_mode=rsu_mode,
                             agent_dim=agent_dim, head_dim=head_dim,
                             horizon=horizon, beta=beta)
        # xyz / lwh / sin-cos / velocity / class: physical scaling, not val fit.
        self.register_buffer("geometry_scale", torch.tensor(
            [50., 50., 5., 10., 10., 5., 1., 1., 20., 20., 1., 1., 1.]))
        self.agent_encoder = nn.Sequential(nn.Linear(13, agent_dim), nn.GELU(),
                                           nn.Linear(agent_dim, agent_dim))
        self.trajectory_head = ContinuousTrajectoryHead(
            hidden_size=ego_dim + agent_dim, horizon=horizon, mlp_size=head_dim)

    def forward(self, h_ego, agent_geometry=None, agent_mask=None):
        if self.settings["rsu_mode"] != "zero":
            raise RuntimeError("Legacy M1 RSU planner bypass is disabled; use tools.continuous.online")
        if h_ego.ndim != 2 or h_ego.shape[1] != self.settings["ego_dim"]:
            raise ValueError("Unexpected ego feature dimension")
        if not torch.isfinite(h_ego).all():
            raise ValueError("Ego features must be finite")
        # Historical F0 reload only: preserve its D+256 head with a constant zero block.
        # RSU modes are rejected above; object geometry never reaches this head.
        h_rsu = h_ego.new_zeros((len(h_ego), self.settings["agent_dim"]))
        output = self.trajectory_head(torch.cat([h_ego, h_rsu], -1))
        return {"delta_xy": output["displacement_xy"], "trajectory": output["trajectory_xy"]}

    def save(self, directory, metadata=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "planner.pt")
        (directory / "planner.json").write_text(json.dumps(
            {"settings": self.settings, "metadata": metadata or {}}, indent=2))

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        settings = json.loads((directory / "planner.json").read_text())["settings"]
        model = cls(**settings)
        model.load_state_dict(torch.load(directory / "planner.pt", map_location="cpu",
                                         weights_only=True))
        return model


def waypoint_loss(prediction, target, mask, beta=1.0):
    """Only cumulative waypoint Smooth-L1, global valid-point mean under DDP."""
    valid = mask.bool()
    if prediction.shape != target.shape or valid.shape != target.shape[:2]:
        raise ValueError("Waypoint shapes do not match")
    if not valid.any(1).all() or not torch.isfinite(target[valid]).all():
        raise ValueError("Every sample needs finite valid future targets")
    safe = torch.where(valid[..., None], target, prediction.detach())
    numerator = (F.smooth_l1_loss(prediction, safe, reduction="none", beta=beta)
                 .mean(-1) * valid).sum()
    count = valid.sum().to(prediction.dtype)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(count)
        # DDP averages local gradients, so compensate to form a global mean.
        return numerator * torch.distributed.get_world_size() / count.clamp_min(1)
    return numerator / count.clamp_min(1)
