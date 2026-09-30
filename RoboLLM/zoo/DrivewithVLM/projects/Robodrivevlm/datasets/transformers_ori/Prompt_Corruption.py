from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS
from PIL import Image
from mmcv.transforms.base import BaseTransform
import numpy as np
import sys
import random
sys.path.append('//home/ldc/Projects/RoboLLM/corruptions')
# 字符级-函数
from prompt_corruption.prompt_methods import character_level_attack
# 恶意攻击-豆包
from prompt_corruption.prompt_methods import generate_malice_level_attack
# 同义词-豆包
# from prompt_corruption.prompt_methods import generate_word_level_attack
# # promptQA攻击
# from prompt_corruption.prompt_methods import prompt_qa_attack
from prompt_corruption.prompt_methods import malice_role_change_attack
from prompt_corruption.prompt_methods import malice_injection_change_attack
from prompt_corruption.prompt_methods import malice_delete_attack,malice_overwrite_rulescode_attack,malice_overwrite_rules_attack,generate_conversation_level_attack
attack_map = {
    'character': character_level_attack,
    # 'word': generate_word_level_attack,
    # 'promptQA':prompt_qa_attack,
    'role_change': malice_role_change_attack,
    'injection':malice_injection_change_attack,
    'delete':malice_delete_attack,
    'overwrite_rulescode':malice_overwrite_rulescode_attack,
    'overwrite_rules':malice_overwrite_rules_attack,
    'generate_conversation':generate_conversation_level_attack
}


@TRANSFORMS.register_module()
class prompt_corruption(BaseTransform):
    def __init__(self, attack_type):
        assert attack_type in attack_map, f"Unknown attack type: {attack_type}"
        self.attack_type = attack_type
        self.method = attack_map[attack_type]
        super().__init__()

    def transform(self, results,attack_type=None):
        
        if self.attack_type == 'promptQA':
            promptQA = results['conversations'][0]
            corruption_promptQA = self.method(promptQA)
            results['conversations'][0] = corruption_promptQA
            print("conversations:",results['conversations'])
        elif self.attack_type == 'character':
            system_prompt = results['system_prompt']
            system_corruption_prompt,count = self.method(system_prompt)
            results['system_prompt'] = system_corruption_prompt
        elif self.attack_type=='generate_conversation':
            promptQA = results['conversations'][0]
            corruption_promptQA = self.method(promptQA)
            results['conversations'][0] = corruption_promptQA
            print("conversations:",results['conversations'])
        elif self.attack_type=='overwrite_rules':
            promptQA = results['conversations'][0]
            corruption_promptQA = self.method(promptQA)
            results['conversations'][0] = corruption_promptQA
            # print("conversations:",results['conversations'])
        elif self.attack_type=='injection':
            operation = random.choice(["system_prompt", "conversation"])
            if operation == 'system_prompt':
                system_prompt = results['system_prompt']
                system_corruption_prompt = self.method(system_prompt,operation)
                results['system_prompt'] = system_corruption_prompt
                print(system_corruption_prompt)
            elif operation == 'conversation':
                promptQA = results['conversations'][0]
                corruption_promptQA = self.method(promptQA,operation)
                results['conversations'][0] = corruption_promptQA
            # print("conversations:",results['conversations'])
        else:
            system_prompt = results['system_prompt']
            system_corruption_prompt = self.method(system_prompt)
            
            if self.attack_type == 'word':
                if system_corruption_prompt.count("\n\n") >= 2:
                    second_newline_pos = system_corruption_prompt.find("\n\n",system_corruption_prompt.find("\n\n") + 2)
                    system_corruption_prompt = system_corruption_prompt[:second_newline_pos]
            
            
            results['system_prompt'] = system_corruption_prompt
            # print("results['system_prompt']:",system_corruption_prompt)

        
        # with open(f"/home/ldc/Projects/RoboLLM/zoo/MMDrive/vis/prompt/nolidar_{self.attack_type}_attack_output.txt", "a") as file:
        #     file.write(system_corruption_prompt + "\n")

        return results
