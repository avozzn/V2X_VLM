"""I1: retain LLaVA image merge, inject RSU soft tokens before language decoding.

Frozen base/optional frozen driving adapter; gradients through the language model
train the RSU encoder and dedicated planning embedding. No RSU planner bypass.
"""
import json
from pathlib import Path

import torch
from torch import nn

from tools.continuous.model import ContinuousTrajectoryHead

INTERFACE = 'i1-rsu-before-vlm-v1'


class I1Planner(nn.Module):
    def __init__(self, backbone, rsu_mode='zero', agent_dim=256, head_dim=512):
        super().__init__()
        if rsu_mode not in {'zero', 'oracle', 'detector'}:
            raise ValueError('Invalid RSU source')
        self.backbone = backbone.requires_grad_(False).eval()
        base = backbone.get_base_model() if hasattr(backbone, 'get_base_model') else backbone
        self.language_dim = base.config.text_config.hidden_size
        self.settings = dict(rsu_mode=rsu_mode, agent_dim=agent_dim, head_dim=head_dim)
        self.agent_encoder = nn.Sequential(nn.Linear(13, agent_dim), nn.GELU(),
                                           nn.Linear(agent_dim, self.language_dim))
        self.plan_embedding = nn.Parameter(torch.empty(self.language_dim))
        self.rsu_embedding = nn.Parameter(torch.empty(self.language_dim))
        nn.init.normal_(self.plan_embedding, std=.02)
        nn.init.normal_(self.rsu_embedding, std=.02)
        self.register_buffer('geometry_scale', torch.tensor(
            [50., 50., 5., 10., 10., 5., 1., 1., 20., 20., 1., 1., 1.]))
        self.trajectory_head = ContinuousTrajectoryHead(
            hidden_size=self.language_dim, horizon=9, mlp_size=head_dim)

    @property
    def decoder(self):
        base = self.backbone.get_base_model() if hasattr(self.backbone, 'get_base_model') else self.backbone
        return base.language_model.model

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def assemble(self, embeddings, attention_mask, geometry, agent_mask):
        if attention_mask.ndim != 2 or embeddings.shape[:2] != attention_mask.shape:
            raise ValueError('Expected actual merged image/text embeddings and 2D mask')
        if not attention_mask.bool().any(1).all():
            raise ValueError('Empty observation')
        if geometry.shape != (*agent_mask.shape, 13) or len(geometry) != len(embeddings):
            raise ValueError('Invalid object batch')
        valid = agent_mask.bool()
        if self.settings['rsu_mode'] == 'zero':
            valid = torch.zeros_like(valid)
        if not torch.isfinite(geometry[valid]).all():
            raise ValueError('Nonfinite valid RSU geometry')
        clean = torch.where(valid[..., None], geometry, 0.)
        agent_tokens = self.agent_encoder(clean / self.geometry_scale)
        rows, indices = [], []
        for i in range(len(embeddings)):
            observation = embeddings[i, attention_mask[i].bool()]
            objects = (agent_tokens[i, valid[i]] + self.rsu_embedding).to(embeddings.dtype)
            row = torch.cat([observation, objects, self.plan_embedding.to(embeddings.dtype)[None]])
            rows.append(row)
            indices.append(len(row) - 1)
        merged = nn.utils.rnn.pad_sequence(rows, batch_first=True)
        lengths = torch.tensor([len(row) for row in rows], device=merged.device)
        mask = torch.arange(merged.shape[1], device=merged.device)[None] < lengths[:, None]
        positions = torch.arange(merged.shape[1], device=merged.device)[None].expand(len(rows), -1)
        return merged, mask, positions, torch.tensor(indices, device=merged.device)

    def forward(self, inputs, agent_geometry, agent_mask):
        if 'labels' in inputs or inputs.get('past_key_values') is not None:
            raise ValueError('Observation-only forward; no answers or KV cache')
        captured = {}

        def inject(module, args, kwargs):
            if args or kwargs.get('inputs_embeds') is None or kwargs.get('input_ids') is not None:
                raise ValueError('Unsupported LLaVA decoder interface')
            embeds, mask, positions, index = self.assemble(
                kwargs['inputs_embeds'], kwargs['attention_mask'], agent_geometry, agent_mask)
            kwargs = dict(kwargs, inputs_embeds=embeds, attention_mask=mask,
                          position_ids=positions, cache_position=None)
            captured['index'] = index
            captured['lengths'] = mask.sum(1)
            return args, kwargs

        def read(module, args, output):
            hidden = output.last_hidden_state
            captured['hidden'] = hidden[torch.arange(len(hidden), device=hidden.device), captured['index']]

        before = self.decoder.register_forward_pre_hook(inject, with_kwargs=True)
        after = self.decoder.register_forward_hook(read)
        try:
            # Intentionally NOT no_grad: new input tokens need gradients through the frozen decoder.
            self.backbone(**inputs, use_cache=False, return_dict=True,
                          output_hidden_states=False, num_logits_to_keep=1)
        finally:
            before.remove()
            after.remove()
        head = self.trajectory_head(captured['hidden'].float())
        return {'trajectory': head['trajectory_xy'], 'delta_xy': head['displacement_xy'],
                'wp_hidden': captured['hidden'], 'merged_lengths': captured['lengths']}

    def save(self, directory, metadata):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        state = {key: value for key, value in self.state_dict().items() if not key.startswith('backbone.')}
        torch.save(state, directory / 'i1.pt')
        (directory / 'i1.json').write_text(json.dumps(
            dict(interface=INTERFACE, settings=self.settings, metadata=metadata), indent=2))

    def restore(self, directory):
        config = json.loads((Path(directory) / 'i1.json').read_text())
        if config['interface'] != INTERFACE or config['settings'] != self.settings:
            raise ValueError('Wrong input/checkpoint interface')
        result = self.load_state_dict(torch.load(Path(directory) / 'i1.pt',
                                                 map_location='cpu', weights_only=True), strict=False)
        if result.unexpected_keys or any(not key.startswith('backbone.') for key in result.missing_keys):
            raise ValueError('Incomplete I1 checkpoint')
