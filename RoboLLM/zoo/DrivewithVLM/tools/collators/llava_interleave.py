import re
from typing import Dict, List, Sequence, Union

import numpy as np
import PIL
import torch
from transformers.image_utils import get_image_size, to_numpy_array
from transformers.models.llava.processing_llava import LlavaProcessorKwargs
from transformers.utils import logging

from . import register_collator
from .base import BaseDataCollator


logger = logging.get_logger(__name__)


# slightly different from https://huggingface.co/llava-hf/llava-interleave-qwen-0.5b-hf/blob/main/chat_template.json
# to include <|im_end|> of assistant's response as labels
template = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n'}}"
    "{# Render all images first #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<image>' }}"
    "{% endfor %}"
    "{# Render all text next #}"
    "{% if message['role'] != 'assistant' %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ '\n' + content['text'] }}"
    "{% endfor %}"
    "{% else %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{% generation %}"
    "{{ '\n' + content['text'] }}"
    "{{'<|im_end|>' + '\n'}}"
    "{% endgeneration %}"
    "{% endfor %}"
    "{% endif %}"
    "{% if message['role'] != 'assistant' %}"
    "{{'<|im_end|>' + '\n'}}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


@register_collator("llava-interleave")
class LLaVAInterleaveDataCollator(BaseDataCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # The MMDet3D dataset keeps calibration, annotations and metadata in
        # the pipeline result.  Some of those values are classes (notably
        # ``box_type_3d``), and Accelerate interprets any object exposing
        # ``.to`` as a tensor-like value.  Passing the full result therefore
        # crashes before model.forward.  Only return inputs accepted by the
        # LLaVA model, while allowing optional processor outputs used by newer
        # checkpoints.
        instance = instances[0]
        model_input_keys = {
            "input_ids",
            "attention_mask",
            "labels",
            "pixel_values",
            "image_sizes",
            "pixel_attention_mask",
            "position_ids",
        }
        batch = {
            key: value for key, value in instance.items()
            if key in model_input_keys
        }
        required = {"input_ids", "attention_mask", "labels", "pixel_values"}
        missing = required - batch.keys()
        if missing:
            raise KeyError(f"LLaVA pipeline output is missing model inputs: {sorted(missing)}")
        if not all(isinstance(value, torch.Tensor) for value in batch.values()):
            invalid = {
                key: type(value).__name__ for key, value in batch.items()
                if not isinstance(value, torch.Tensor)
            }
            raise TypeError(f"LLaVA model inputs must be tensors, got: {invalid}")
        return batch
