"""Real tiny HF LLaVA integration: image merge, RSU causality and frozen-base gradients."""
import copy

import pytest
import torch
from transformers import LlavaConfig, LlavaForConditionalGeneration

from tools.continuous.online_model import I1Planner
from tools.continuous.model import ContinuousPlanner, waypoint_loss


def backbone(local_family=False):
    config = LlavaConfig(
        vision_config=dict(model_type='clip_vision_model', hidden_size=16, intermediate_size=32,
                           num_hidden_layers=2, num_attention_heads=2, image_size=8, patch_size=4),
        text_config=dict(model_type='llama', vocab_size=40, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=256, pad_token_id=0),
        image_token_index=39, image_seq_length=4, vision_feature_layer=-1)
    if local_family:
        from transformers import Qwen2Config, SiglipVisionConfig
        config.text_config = Qwen2Config(vocab_size=40, hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=2, num_attention_heads=2,
                                        num_key_value_heads=2, max_position_embeddings=256, pad_token_id=0)
        config.vision_config = SiglipVisionConfig(hidden_size=16, intermediate_size=32,
                                                 num_hidden_layers=2, num_attention_heads=2,
                                                 image_size=8, patch_size=4)
        config.vision_feature_select_strategy = 'full'
    return LlavaForConditionalGeneration(config).eval()


def inputs():
    # Legacy image expansion with unequal left-padded context lengths.
    return dict(input_ids=torch.tensor([[0, 1, 39, 3, 4], [1, 39, 3, 4, 5]]),
                attention_mask=torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
                pixel_values=torch.randn(2, 3, 8, 8))


@pytest.mark.parametrize("local_family", [False, True])
def test_real_llava_rsu_inside_decoder_and_gradients(local_family):
    torch.manual_seed(12)
    model = I1Planner(backbone(local_family), 'oracle', agent_dim=8, head_dim=16)
    batch = inputs()
    geometry = torch.randn(2, 3, 13)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    geometry[~mask] = float('nan')
    seen = {}
    def capture(module, args, kwargs):
        seen['embeddings'] = kwargs['inputs_embeds'].detach().clone()
        seen['mask'] = kwargs['attention_mask'].clone()
    hook = model.decoder.register_forward_pre_hook(capture, with_kwargs=True)
    # Injection runs after the capture hook registered here; lengths are returned by the model.
    output = model(batch, geometry, mask)
    hook.remove()
    assert output['trajectory'].shape == (2, 9, 2)
    assert output['merged_lengths'].tolist() == [10, 9]  # merged visual/text + objects + plan
    changed = geometry.clone()
    changed[0, 0, 0] += 20
    altered = model(batch, changed, mask)
    assert not torch.allclose(output['wp_hidden'][0], altered['wp_hidden'][0])
    torch.testing.assert_close(output['wp_hidden'][1], altered['wp_hidden'][1])
    image_changed = {**batch, 'pixel_values': batch['pixel_values'] + 1}
    assert not torch.allclose(output['wp_hidden'], model(image_changed, geometry, mask)['wp_hidden'])
    padded = torch.cat([geometry, torch.full((2, 4, 13), float('nan'))], 1)
    padded_mask = torch.cat([mask, torch.zeros(2, 4, dtype=torch.bool)], 1)
    torch.testing.assert_close(output['trajectory'], model(batch, padded, padded_mask)['trajectory'])
    loss = waypoint_loss(output['trajectory'], torch.zeros(2, 9, 2), torch.ones(2, 9, dtype=torch.bool))
    loss.backward()
    assert model.agent_encoder[0].weight.grad.abs().sum() > 0
    assert model.plan_embedding.grad.abs().sum() > 0
    assert model.trajectory_head is not None
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert len(model.decoder._forward_pre_hooks) == 0


def test_zero_and_empty_rsu_match_and_reload(tmp_path):
    torch.manual_seed(3)
    base = backbone()
    oracle = I1Planner(copy.deepcopy(base), 'oracle', agent_dim=8, head_dim=16)
    zero = I1Planner(copy.deepcopy(base), 'zero', agent_dim=8, head_dim=16)
    zero.load_state_dict(oracle.state_dict())
    batch = inputs()
    geometry = torch.full((2, 3, 13), float('nan'))
    mask = torch.zeros(2, 3, dtype=torch.bool)
    expected = oracle(batch, geometry, mask)['trajectory']
    torch.testing.assert_close(expected, zero(batch, geometry, mask)['trajectory'])
    oracle.save(tmp_path / 'checkpoint', {'model': 'tiny-test-only'})
    restored = I1Planner(copy.deepcopy(base), 'oracle', agent_dim=8, head_dim=16)
    restored.restore(tmp_path / 'checkpoint')
    torch.testing.assert_close(expected, restored(batch, geometry, mask)['trajectory'])
    with pytest.raises(ValueError, match='Observation-only'):
        oracle({**batch, 'labels': batch['input_ids']}, geometry, mask)
    assert len(oracle.decoder._forward_pre_hooks) == 0


def test_old_rsu_bypass_rejected():
    with pytest.raises(RuntimeError, match='bypass is disabled'):
        ContinuousPlanner(8, 'oracle')(torch.randn(1, 8), torch.zeros(1, 16, 13),
                                       torch.ones(1, 16, dtype=torch.bool))
