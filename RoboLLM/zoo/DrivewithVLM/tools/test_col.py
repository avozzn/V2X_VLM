from eval.evaluation import load_pred_trajs_from_file,planning_evaluation
import json
import pickle
import ast
import numpy as np
output_file='/home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM/vis/result_test_dt_re_20251217_142636.json'
pred_trajs_dict = load_pred_trajs_from_file('/home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM/vis/pkl/result_test_dt_re_20251217_142636.pkl')
evaluation_data=planning_evaluation(pred_trajs_dict, subset=None, only_vehicle=False)
with open(output_file, 'a', encoding='utf-8') as f:
    json.dump(evaluation_data, f, indent=4)
print("Test finish\n")