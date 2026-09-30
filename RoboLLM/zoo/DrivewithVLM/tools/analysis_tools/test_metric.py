import json
import numpy as np
import ast
from eval.evaluation import load_pred_trajs_from_file,planning_evaluation
output_file = '/home/ldc/Projects/RoboLLM/zoo/MMDrive/vis/result_tta_sun_sim_5_20250515_150716.json'



with open(output_file, 'r', encoding='utf-8') as file:
    file.seek(0)
    content = file.read()
    data = json.loads(content)
    traj_dict = {}

    exist_dict = {}
    i=0
    for line_num, item in enumerate(data, 1):
        try:
            # 获取 token
            token = item["token"]
            # 提取轨迹信息
            traj = item['answer'].split('\n')[-1]
            assert "None" not in traj            # 将字符串转换为列表
            traj = ast.literal_eval(traj)
            assert None not in traj
            # 将列表转换为 NumPy 数组
            traj = np.array(traj)
            # 将轨迹数据添加到 traj_dict 中
            assert traj.shape == (9, 2)
            
            traj_dict[token] = np.expand_dims(traj, axis=0)
        except Exception as e:
            # 打印详细的错误信息（包括错误类型和具体描述）
            print(f"Error processing line {line_num}: {type(e).__name__} - {str(e)}")
            i=i+1
            continue
    print(f"error num:{i}\n")
    evaluation_data=planning_evaluation(traj_dict, subset=None, only_vehicle=False)
    print("Test finish\n")
