import json
import numpy as np
from scipy.integrate import cumulative_trapezoid
from math import atan2
import sys
import os.path as osp
import math

def IntegrateCurvatureForPoints(curvatures, velocities_norm, initial_position, initial_heading, time_span):
    # Convert lists to numpy arrays for element-wise multiplication
    curvatures = np.array(curvatures)
    velocities_norm = np.array(velocities_norm)
    
    t = np.linspace(0, time_span, time_span)  # Time vector
    
    # Initial conditions
    x0, y0 = initial_position[0], initial_position[1]  # Starting position
    theta0 = initial_heading  # Initial orientation (radians)

    # Integrate to compute heading (theta)
    theta = cumulative_trapezoid(curvatures * velocities_norm, t, initial=0)
    theta[0:] += theta0

    # Compute velocity components
    v_x = velocities_norm * np.cos(theta)
    v_y = velocities_norm * np.sin(theta)

    # Integrate to compute trajectory
    x = cumulative_trapezoid(v_x, t, initial=0)
    y = cumulative_trapezoid(v_y, t, initial=0)

    x[0:] += x0
    y[0:] += y0

    return np.stack((x, y), axis=1)
def pred_traj_VCT(file_path,sample_json_file,scene_json_file):
    # 读取输入的 JSON 文件
    with open(file_path, 'r') as f:
        input_data = json.load(f)
    
    output_data = []  # 存储处理后的输出数据
    
    # 遍历 JSON 文件中的每一项
    for item in input_data:
        token = item['token']
        answer = item['answer']
         
        
        # =======================================
        
        with open(sample_json_file, 'r') as f:
            sample_data = json.load(f)
        
        # 2. 从 sample.json 查找对应的 scene_token
        scene_token = None
        for sample in sample_data:
            if sample['token'] == token:
                scene_token = sample['scene_token']
                break
        
        if scene_token is None:
            raise ValueError(f"Token {token} not found in {sample_json_file}")

        # 3. 加载 scene.json
        with open(scene_json_file, 'r') as f:
            scene_data = json.load(f)

        # 4. 在 scene.json 中查找对应的 name
        scene_name = None
        for scene in scene_data:
            if scene['token'] == scene_token:
                scene_name = scene['name']
                break
        
        if scene_name is None:
            raise ValueError(f"Scene with token {scene_token} not found in {scene_json_file}")
        
        
        scene_file = f"/home/ldc/Projects/RoboLLM/zoo/MMDrive/projects/Openemma/prepare/VCT/{scene_name}.json"
    
        if not osp.exists(scene_file):
            raise FileNotFoundError(f"The scene file {scene_file} does not exist.")
        
        with open(scene_file, "r") as f:
            all_data= json.load(f)
        sample_data = all_data[token]
        
        obs_ego_trajectory = np.array(sample_data["past_ego_trajectory"])
        obs_ego_velocities = np.array(sample_data["past_ego_velocities"])
        initial_position = obs_ego_trajectory[-1]
        initial_heading = atan2(obs_ego_velocities[-1][1],obs_ego_velocities[-1][0])
        # =======================================
        
        # 提取最后一个 "Future speeds and curvatures:" 后的数据
        if 'Future speeds and curvatures:' in answer:
            # 根据 "Future speeds and curvatures:" 分割字符串
            parts = answer.split('Future speeds and curvatures:')
            
            # 获取最后一个部分
            speeds_and_curvatures_str = parts[-1].strip()

            # 将字符串转换为包含速度和曲率的对
            speeds_and_curvatures = []
            skip_entry = False 
            for pair in speeds_and_curvatures_str.split('],'):
                pair = pair.replace('[', '').replace(']', '').strip()
                if pair:
                    try:
                        values = pair.split(',')
                        if len(values) == 2:  # 确保每个分割出的对包含两个元素
                            speed, curvature = map(float, values)
                            if math.isnan(speed) or math.isnan(curvature):
                                print(token)
                                print(f"skip entry error formert ：{pair}")
                                skip_entry = True
                                break
                            speeds_and_curvatures.append([speed, curvature])
                        else:
                            # 记录格式错误的数据，跳过整个条目
                            print(token)
                            print(f"skip entry error formert ：{pair}")
                            skip_entry = True
                            break
                    except ValueError:
                        # 如果无法转换为浮点数，跳过整个条目
                        print(token)
                        print(f"skip entry can't convert：{pair}")
                        skip_entry = True
                        break

            if skip_entry:
                continue  

            velocities_norm = [sc[0] for sc in speeds_and_curvatures]  # 速度
            curvatures = [sc[1] for sc in speeds_and_curvatures]  # 曲率
            
            if len(curvatures) != len(velocities_norm):
                print(f"length problem token：{token}")
                continue
            
            
            time_span = len(curvatures)  
            
            trajectory = IntegrateCurvatureForPoints(curvatures, velocities_norm, initial_position, initial_heading, time_span)
            

            output_data.append({
                'token': token,
                'predicted_trajectory': trajectory.tolist(),
                'predicted_speeds_and_curvatures': speeds_and_curvatures
            })
    
    return json.dumps(output_data, indent=4)

# # Example usage
# file_path = '/home/ldc/Projects/RoboLLM/zoo/MMDrive/vis/result_20250226_194229.json'
# sample_json_file='/home/ldc/Projects/RoboLLM/zoo/MMDrive/data/nuscenes/v1.0-trainval/sample.json'
# scene_json_file='/home/ldc/Projects/RoboLLM/zoo/MMDrive/data/nuscenes/v1.0-trainval/scene.json'
# output_json = pred_traj_VCT(file_path,sample_json_file,scene_json_file)

# # Optionally save the output to a new file
# output_file_path = '/home/ldc/Projects/RoboLLM/zoo/MMDrive/projects/Openemma/prepare/pred_traj_20250226_194229.json'
# with open(output_file_path, 'w') as f:
#     f.write(output_json)

# print("Processing complete. Output saved to:", output_file_path)