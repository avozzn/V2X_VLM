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
# promptQA攻击
# from prompt_corruption.prompt_methods import prompt_qa_attack
from prompt_corruption.prompt_methods import malice_role_change_attack
from prompt_corruption.prompt_methods import malice_injection_drivevlm_attack
from prompt_corruption.prompt_methods import malice_delete_attack,malice_overwrite_rulescode_attack,malice_overwrite_rules_attack,generate_conversation_level_attack
attack_map = {
    'character': character_level_attack,
    # 'word': generate_word_level_attack,
    # 'promptQA':prompt_qa_attack,
    'role_change': malice_role_change_attack,
    'injection':malice_injection_drivevlm_attack,
    'delete':malice_delete_attack,
    'overwrite_rulescode':malice_overwrite_rulescode_attack,
    'overwrite_rules':malice_overwrite_rules_attack,
    'conversation':generate_conversation_level_attack

}

@TRANSFORMS.register_module()
class prompt_corruption(BaseTransform):
    def __init__(self, attack_type, severity):
        assert attack_type in attack_map, f"Unknown attack type: {attack_type}"
        self.attack_type = attack_type
        self.method = attack_map[attack_type]
        print("attack_type in methods : ",attack_type)
        super().__init__()

    def transform(self, results,attack_type=None):
        if self.attack_type == 'promptQA':
            promptQA = results['conversations'][0]
            corruption_promptQA = self.method(promptQA)
            results['conversations'][0] = corruption_promptQA
            # print(results['conversations'][0])    
        else:
            corrupted_questions = []
            for i, q in enumerate(results['questions']):
                # 转换为字符串并去括号
                q_clean = q.strip('[]')
                if self.attack_type == 'word':
                    is_first = (i == 0)
                    # 执行攻击
                    corrupted_q = self.method(q_clean,is_first)
                    # print("count_q",count)
                elif self.attack_type == 'character':
                    corrupted_q,count_q = self.method(q_clean)
                    # print("count_q",count_q)
                elif self.attack_type == 'delete':
                    is_first = (i == 0)
                    # 执行攻击
                    if is_first:
                        corrupted_q = q_clean
                    else:
                        corrupted_q = self.method(q_clean,is_first)
                elif self.attack_type == 'role_change':
                    is_first = (i == 0)
                    # 执行攻击
                    if is_first:
                        corrupted_q = self.method(q_clean)
                    else:
                        corrupted_q = q_clean
                elif self.attack_type == 'overwrite_rulescode' or self.attack_type =='overwrite_rules':
                    is_last = (i == len(results['questions']) - 1)
                    if is_last:
                         corrupted_q = self.method(q_clean)
                    else:
                        corrupted_q = q_clean
                elif self.attack_type == 'injection':
                    operation = random.choice(["is_first", "is_middle","is_last"])
                    if operation == "is_first":
                        is_first= (i == 0)
                        if is_first:
                            corrupted_q = self.method(q_clean,operation)
                        else: 
                            corrupted_q = q_clean
                    if operation == "is_middle":
                        is_middle = (i ==4)
                        if is_middle:
                            corrupted_q = self.method(q_clean,operation)
                        else: 
                            corrupted_q = q_clean
                    if operation == "is_last":
                        is_last = (i == len(results['questions']) - 1)
                        if is_last:
                            corrupted_q = self.method(q_clean,operation)
                        else: 
                            corrupted_q = q_clean
                else:
                    corrupted_q = self.method(q_clean)
                corrupted_questions.append(corrupted_q)
            results['questions'] = corrupted_questions
            print("results['questions']",results['questions'])
            
            corrupted_answers = []
            for a in results['answers']:
                a_clean = a.strip('[]')
                if self.attack_type == 'character':
                    corrupted_a ,_= self.method(a_clean)
                else:
                    corrupted_a = a_clean
                corrupted_answers.append(corrupted_a)
            results['answers'] = corrupted_answers
            # print("results['answers']",results['answers'])


        # with open(f"/home/ldc/Projects/RoboLLM/zoo/MMDrive/vis/prompt_drivevlm_{self.attack_type}_attack_output.txt", "a") as file:
        #     file.write("System Prompt:\n")
        #     file.write(results['questions'])
        #     file.write(results['answers'])
        return results
