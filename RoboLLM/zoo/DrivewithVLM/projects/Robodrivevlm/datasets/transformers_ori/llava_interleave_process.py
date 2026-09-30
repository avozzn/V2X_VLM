from mmdet3d.registry import TRANSFORMS
from PIL import Image
from mmcv.transforms.base import BaseTransform
from transformers import AutoProcessor, AutoConfig
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from PIL import Image
import re
from typing import List, Union
import PIL
import torch
from transformers.image_utils import get_image_size, to_numpy_array
from transformers.models.llava.processing_llava import LlavaProcessorKwargs
from transformers.utils import logging
logger = logging.get_logger(__name__)
Number = Union[int, float]


template = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n'}}"
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




@TRANSFORMS.register_module()
class LLava_interleave_Process(BaseTransform):
    def __init__(self,model_hf_path=None,mode='test'):
        
        self.mask_answer_tokens=None
        self.model_hf_path=model_hf_path
        self.model_local_path=self.model_hf_path
        self.processor = AutoProcessor.from_pretrained(self.model_hf_path)
        self.tokenizer = self.processor.tokenizer
        self.config = AutoConfig.from_pretrained(self.model_local_path)
        self.IGNORE_TOKEN_ID=-100
        self.mask_question_tokens=True
        self.PAD_TOKEN_ID=self.tokenizer.pad_token_id
        self.mode=mode



    def transform(self, results):
        
        '''
        results['conversations']=
        [
            {
                "system_prompt": "You are a helpful assistant.",
                "video": "path/to/video1.mp4",
                "conversations": [
                    {
                        "from": "human",
                        "value": "<video>What is this video about?"
                    },
                    {
                        "from": "gpt",
                        "value": "This video shows a baby crying."
                    },
                ]
            }
        ]
        
        '''
        output_kwargs = self.processor._merge_kwargs(
            LlavaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
        )

        vision_inputs = dict()
        images: List[List[PIL.Image.Image]] = [x for x in results["input_images"]]
        if len(images) > 0:
            vision_inputs.update(**self.processor.image_processor(images, return_tensors="pt", **output_kwargs["images_kwargs"]))

        # some parsing
        # 提取图像列表
        images: List[List[PIL.Image.Image]] = []
        if "input_images" in results:
            images = [results["input_images"]]  # 将单个字段转换为列表格式

        # 提取 system_prompt
        system_prompts: List[Union[str, None]] = []
        if "system_prompt" in results:
            system_prompts = [results["system_prompt"]]



        # 提取对话内容
        if self.mode=='test':
            results["conversations"][1]=''
        conversations: List[List] = []
        if "conversations" in results:
            conversations = [results["conversations"]]



        # constants
        max_len = 1024
        image_token_id = self.config.image_token_index
        patch_size = self.processor.patch_size
        vision_feature_select_strategy = self.processor.vision_feature_select_strategy
       
        input_ids = []
        labels = []
        attention_mask=[]

        for system_prompt, cur_images, cur_convs in zip(system_prompts, images, conversations):
            cur_num_images = 0
            cur_input_ids = []
            cur_labels = []
            cur_text = []
            if system_prompt is not None:
                cur_text.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                })
            
            for i, text in enumerate(cur_convs):
                if i % 2 == 0:
                    num_images = len([m.start() for m in re.finditer("<image>", text)])
                    cur_num_images += num_images

                    # .strip(): whitespaces and newlines are handled by chat_template
                    text = text.replace("<image>", "").strip()

                    cur_text.append({
                        "role": "user",
                        "content": [{"type": "text", "text": text}] + \
                            [{"type": "image"}] * num_images
                    })
                else:
                    cur_text.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": text}]
                    })
            

            temp = self.processor.apply_chat_template(
                cur_text,
                chat_template=template,
                add_generation_prompt=False,
                tokenize=True,
                return_assistant_tokens_mask=True,
                return_dict=True,
                return_tensors="pt",
                truncation=False # the assistant tokens mask seems wrong when truncation is enabled
            )
            cur_input_ids = temp["input_ids"]
            cur_assistant_masks = torch.tensor(temp["assistant_masks"], dtype=torch.bool).unsqueeze(0)

            # expand image tokens
            temp_vision_inputs = self.processor.image_processor(cur_images, return_tensors="pt")
            if temp_vision_inputs.get("pixel_values") is not None:
                if patch_size is not None and vision_feature_select_strategy is not None:
                    # Replace the image token with the expanded image token sequence
                    pixel_values = temp_vision_inputs["pixel_values"]
                    height, width = get_image_size(to_numpy_array(pixel_values[0]))
                    num_image_tokens = (height // patch_size) * (width // patch_size) + 1
                    if vision_feature_select_strategy == "default":
                        num_image_tokens -= 1

                    repeat = torch.where(cur_input_ids == image_token_id, num_image_tokens, 1).squeeze()
                    cur_input_ids = cur_input_ids.repeat_interleave(repeat, dim=1)
                    cur_assistant_masks = cur_assistant_masks.repeat_interleave(repeat, dim=1)
                else:
                    logger.warning_once(
                        "Expanding inputs for image tokens in LLaVa should be done in processing. "
                        "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
                        "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
                        "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
                    )

            # manual truncation
            if cur_input_ids.shape[1] > max_len:
                cur_input_ids = cur_input_ids[:, :max_len]
                cur_assistant_masks = cur_assistant_masks[:, :max_len]
            cur_labels = cur_input_ids.clone()

            if self.mask_question_tokens:
                assert cur_labels.shape == cur_assistant_masks.shape, "Label and mask shapes do not match"
                cur_labels = torch.where(cur_assistant_masks, cur_labels, self.IGNORE_TOKEN_ID)


            # mask labels
            if self.mask_answer_tokens:
                cur_input_ids = torch.where(~cur_assistant_masks, cur_input_ids, self.PAD_TOKEN_ID)


            assert cur_input_ids.shape == cur_labels.shape, "Input and label shapes do not match"

            # padding
            if cur_input_ids.shape[1] < max_len:
                cur_input_ids = torch.cat([
                    cur_input_ids,
                    torch.full(
                        (cur_input_ids.shape[0], max_len - cur_input_ids.shape[1]),
                        self.PAD_TOKEN_ID,
                        dtype=cur_input_ids.dtype,
                        device=cur_input_ids.device
                    )
                ], dim=1)
                cur_labels = torch.cat([
                    cur_labels,
                    torch.full(
                        (cur_labels.shape[0], max_len - cur_labels.shape[1]),
                        self.IGNORE_TOKEN_ID,
                        dtype=cur_labels.dtype,
                        device=cur_labels.device
                    )
                ], dim=1)
                

            cur_attention_mask=cur_input_ids.ne(self.PAD_TOKEN_ID)
            cur_attention_mask = cur_attention_mask | (cur_labels != self.IGNORE_TOKEN_ID)

            input_ids.append(cur_input_ids)
            labels.append(cur_labels)
            attention_mask.append(cur_attention_mask)
        

        input_ids = torch.cat(input_ids)
        labels = torch.cat(labels)
        attention_mask=torch.cat(attention_mask)




        if self.mode=='train':
            del results
            results=dict(
                **vision_inputs,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
            return results
        elif self.mode=='test':
            token=results['token']
            question = self.processor.decode(input_ids[0], skip_special_tokens=False)
            question = question.split("<|im_end|>\n<|endoftext|>")[0]
            process = self.processor(text=question, return_tensors='pt')
            input_ids = process['input_ids']
            attention_mask = process['attention_mask']
            results['llm_input']=dict(
                token=token,
                **vision_inputs,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            return results
    


@TRANSFORMS.register_module()
class LLava_interleave_Process_no_image(BaseTransform):
    def __init__(self,model_hf_path=None,mode='test'):
        
        self.mask_answer_tokens=None
        self.model_hf_path=model_hf_path
        self.model_local_path=self.model_hf_path
        self.processor = AutoProcessor.from_pretrained(self.model_hf_path)
        self.tokenizer = self.processor.tokenizer
        self.config = AutoConfig.from_pretrained(self.model_local_path)
        self.IGNORE_TOKEN_ID=-100
        self.mask_question_tokens=True
        self.PAD_TOKEN_ID=self.tokenizer.pad_token_id
        self.mode=mode
        



    def transform(self, results):
        '''
        results['conversations']=
        [
            {
                "system_prompt": "You are a helpful assistant.",
                "video": "path/to/video1.mp4",
                "conversations": [
                    {
                        "from": "human",
                        "value": "<video>What is this video about?"
                    },
                    {
                        "from": "gpt",
                        "value": "This video shows a baby crying."
                    },
                ]
            }
        ]
        '''
        output_kwargs = self.processor._merge_kwargs(
            LlavaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
        )



        # 提取 system_prompt
        system_prompts: List[Union[str, None]] = []
        if "system_prompt" in results:
            system_prompts = [results["system_prompt"]]
        


        # 提取对话内容
        if self.mode=='test':
            results["conversations"][1]=''
        conversations: List[List] = []
        if "conversations" in results:
            conversations = [results["conversations"]]


        # constants
        max_len = 1024


        input_ids = []
        labels = []
        attention_mask=[]

        for system_prompt, cur_convs in zip(system_prompts, conversations):
            cur_num_images = 0
            cur_input_ids = []
            cur_labels = []

            cur_text = []
            if system_prompt is not None:
                cur_text.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                })
            
            for i, text in enumerate(cur_convs):
                if i % 2 == 0:
                    num_images = len([m.start() for m in re.finditer("<image>", text)])
                    cur_num_images += num_images

                    # .strip(): whitespaces and newlines are handled by chat_template
                    text = text.replace("<image>", "").strip()

                    cur_text.append({
                        "role": "user",
                        "content": [{"type": "text", "text": text}] + \
                            [{"type": "image"}] * num_images
                    })
                else:
                    cur_text.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": text}]
                    })
            

            temp = self.processor.apply_chat_template(
                cur_text,
                chat_template=template,
                add_generation_prompt=False,
                tokenize=True,
                return_assistant_tokens_mask=True,
                return_dict=True,
                return_tensors="pt",
                truncation=False # the assistant tokens mask seems wrong when truncation is enabled
            )
            cur_input_ids = temp["input_ids"]
            cur_assistant_masks = torch.tensor(temp["assistant_masks"], dtype=torch.bool).unsqueeze(0)


            

            # manual truncation
            if cur_input_ids.shape[1] > max_len:
                cur_input_ids = cur_input_ids[:, :max_len]
                cur_assistant_masks = cur_assistant_masks[:, :max_len]
            cur_labels = cur_input_ids.clone()

            if self.mask_question_tokens:
                assert cur_labels.shape == cur_assistant_masks.shape, "Label and mask shapes do not match"
                cur_labels = torch.where(cur_assistant_masks, cur_labels, self.IGNORE_TOKEN_ID)


            # mask labels
            if self.mask_answer_tokens:
                cur_input_ids = torch.where(~cur_assistant_masks, cur_input_ids, self.PAD_TOKEN_ID)


            assert cur_input_ids.shape == cur_labels.shape, "Input and label shapes do not match"

            # padding
            if cur_input_ids.shape[1] < max_len:
                cur_input_ids = torch.cat([
                    cur_input_ids,
                    torch.full(
                        (cur_input_ids.shape[0], max_len - cur_input_ids.shape[1]),
                        self.PAD_TOKEN_ID,
                        dtype=cur_input_ids.dtype,
                        device=cur_input_ids.device
                    )
                ], dim=1)
                cur_labels = torch.cat([
                    cur_labels,
                    torch.full(
                        (cur_labels.shape[0], max_len - cur_labels.shape[1]),
                        self.IGNORE_TOKEN_ID,
                        dtype=cur_labels.dtype,
                        device=cur_labels.device
                    )
                ], dim=1)
                

            cur_attention_mask=cur_input_ids.ne(self.PAD_TOKEN_ID)
            cur_attention_mask = cur_attention_mask | (cur_labels != self.IGNORE_TOKEN_ID)

            input_ids.append(cur_input_ids)
            labels.append(cur_labels)
            attention_mask.append(cur_attention_mask)
        

        input_ids = torch.cat(input_ids)
        labels = torch.cat(labels)
        attention_mask=torch.cat(attention_mask)

        if self.mode=='train':
            del results
            results=dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
            return results
        elif self.mode=='test':
            token=results['token']
            question = self.processor.decode(input_ids[0], skip_special_tokens=False)
            question = question.split("<|im_end|>\n<|endoftext|>")[0]
            process = self.processor(text=question, return_tensors='pt')
            input_ids = process['input_ids']
            attention_mask = process['attention_mask']
            results['llm_input']=dict(
                token=token,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return results
    

