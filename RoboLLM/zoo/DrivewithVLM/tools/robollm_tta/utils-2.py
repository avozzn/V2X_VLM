import math
from typing import List, Dict, Optional

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from torch import nn
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
import transformers
from transformers import Trainer
from transformers.trainer import has_length
from typing import Any, Dict, List, Optional, Union

class NoTextOnlyBatchSampler(Sampler):
    r"""
    Sampler that tries its best to sample batches such that no batch has only 
    text (unimodal) data. This is necessary for training with deepspeed. 
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        is_text_only: Optional[List[bool]] = None,
        generator=None,
    ):
        if is_text_only is None:
            raise ValueError("`is_text_only` must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.is_text_only = is_text_only
        self.generator = generator
        self.mega_batch_size = batch_size * world_size

    def __len__(self):
        return len(self.is_text_only)

    def __iter__(self):
        # mm: multimodal, entry that has both text and image/video
        # uni: unimodal, entry that has only text
        mm_indices = [i for i, is_text_only in enumerate(self.is_text_only) if not is_text_only]
        uni_indices = [i for i, is_text_only in enumerate(self.is_text_only) if is_text_only]

        num_batches = math.ceil((len(mm_indices) + len(uni_indices)) / self.mega_batch_size)
        if len(mm_indices) < num_batches:
            raise ValueError(
                f"{len(mm_indices)} multimodal entries, {len(num_batches)} batches. "
                "Not enough multimodal data in the dataset, or the batch size is too small. " 
                "There will be at least one batch that is text-only, which doesn't work with deepspeed. "
                "Try increasing the batch size first."
            )

        # shuffle indices
        mm_indices = [mm_indices[i] for i in torch.randperm(len(mm_indices), generator=None).tolist()]
        uni_indices = [uni_indices[i] for i in torch.randperm(len(uni_indices), generator=None).tolist()]

        # distribute indices into batches
        num_uni_indices_in_mega_batch = [len(uni_indices) // num_batches] * num_batches
        for i in range(len(uni_indices) % num_batches):
            num_uni_indices_in_mega_batch[i] += 1
        
        mega_batches = []
        cur_uni_index = 0
        cur_mm_index = 0
        for i, num_uni_indices in enumerate(num_uni_indices_in_mega_batch):
            mega_batch = []
            mega_batch.extend(uni_indices[cur_uni_index:cur_uni_index + num_uni_indices])
            cur_uni_index += num_uni_indices
            assert len(mega_batch) < self.mega_batch_size

            if i < num_batches - 1:
                increment = self.mega_batch_size - len(mega_batch)
                mega_batch.extend(
                    mm_indices[cur_mm_index:cur_mm_index + increment]
                )
                cur_mm_index += increment
            else: # last batch
                mega_batch.extend(mm_indices[cur_mm_index:])
                assert len(mega_batch) <= self.mega_batch_size, "Last batch is too big."
            
            mega_batches.append(mega_batch)
        
        mega_batch_indices = torch.randperm(len(mega_batches), generator=self.generator)
        mega_batches = [mega_batches[i] for i in mega_batch_indices]
        indices = [i for mega_batch in mega_batches for i in mega_batch]
        return iter(indices)


class TrainerWithCustomSampler(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 添加 dataloader 属性
        self.sample_count = 0
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        is_text_only = [False]*self.train_dataset.length
        return NoTextOnlyBatchSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            is_text_only=is_text_only,
        )
    
    def _get_eval_sampler(self, eval_dataset: torch.utils.data.Dataset) -> Optional[torch.utils.data.Sampler]:
        is_text_only = eval_dataset.is_text_only
        return NoTextOnlyBatchSampler(
            self.args.eval_batch_size,
            world_size=self.args.world_size,
            is_text_only=is_text_only,
        )
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        # 获取当前批次的样本数量，这里假设通过inputs['input_ids'].size(0)获取，需根据实际情况调整
        batch_size = inputs['input_ids'].size(0)
        self.sample_count += batch_size

        if self.is_sagemaker_mp_enabled():
            loss_mb = self.smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        # 每经过100个样本，执行测试并输出结果
        self.sample_count=self.sample_count+1
        if self.sample_count % 100 == 0:
            test_result = self.test(model,inputs)  # 这里假设你有一个名为test的方法来进行测试并返回结果
            print(f"After {self.sample_count} samples, test result: {test_result}")

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if self.is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif self.is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif self.is_torch_musa_available():
                torch.musa.empty_cache()
            elif self.is_torch_npu_available():
                torch.npu.empty_cache()
            elif self.is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [self.OptimizerNames.LOMO, self.OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps
    def test_sp(self,model,inputs):
        model.eval()

        with torch.no_grad():
            question=self.processor.decode(inputs['input_ids'][0][2:], skip_special_tokens=False)
            question=question.split("\n\nMeta Action:")[0]
            result = self.processor(text=question, return_tensors='pt')
            inputs['input_ids']=result['input_ids']
            inputs['attention_mask']=result['attention_mask']
            del inputs['labels']
            output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            print(question)
            print(self.processor.decode(output[0][2:], skip_special_tokens=True))
        model.train()
    # def get_train_dataloader(self):
    #     return self.accelerator.prepare(self.train_dataloader)
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     """
    #     How the loss is computed by Trainer. By default, all models return the loss in the first element.

    #     Subclass and override for custom behavior.
    #     """
    #     import pdb
    #     pdb.set_trace()
    #     for key, value in inputs.items():
    #             inputs[key] = value[0]
    #     inputs['pixel_values']=inputs['pixel_values'].to(torch.bfloat16)
    #     if self.label_smoother is not None and "labels" in inputs:
    #         labels = inputs.pop("labels")
    #     else:
    #         labels = None
    #     outputs = model(**inputs)
    #     # Save past state if it exists
    #     # TODO: this needs to be fixed and made cleaner later.
    #     if self.args.past_index >= 0:
    #         self._past = outputs[self.args.past_index]

    #     if labels is not None:
    #         unwrapped_model = self.accelerator.unwrap_model(model)
    #         if self._is_peft_model(unwrapped_model):
    #             model_name = unwrapped_model.base_model.model._get_name()
    #         else:
    #             model_name = unwrapped_model._get_name()
    #         if model_name in self.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
    #             loss = self.label_smoother(outputs, labels, shift_labels=True)
    #         else:
    #             loss = self.label_smoother(outputs, labels)
    #     else:
    #         if isinstance(outputs, dict) and "loss" not in outputs:
    #             raise ValueError(
    #                 "The model did not return a loss from the inputs, only the following keys: "
    #                 f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
    #             )
    #         # We don't use .loss here since the model may return tuples instead of ModelOutput.
    #         loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    #     return (loss, outputs) if return_outputs else loss

def find_all_linear_names(named_modules: Dict, target_modules: List[str]):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in named_modules.items():
        if not any([module_name in name for module_name in target_modules]):
            continue

        if isinstance(module, cls):
            lora_module_names.add(name)

    for name in list(lora_module_names):
        if 'lm_head' in name: # needed for 16-bit
            lora_module_names.remove(name)

    return list(lora_module_names)


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args)


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)