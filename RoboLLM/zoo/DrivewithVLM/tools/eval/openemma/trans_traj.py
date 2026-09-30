import json
import numpy as np


def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def get_ego_pose_token(sample_data, token):
    for sample in sample_data:
        if sample['sample_token'] == token:
            return sample['ego_pose_token']
    return None


def get_ego_pose(ego_pose_data, ego_pose_token):
    for ego_pose in ego_pose_data:
        if ego_pose['token'] == ego_pose_token:
            translation = ego_pose['translation'] # 三元数
            rotation = ego_pose['rotation']  # 假设是四元数
            return translation, rotation
    return None, None


def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    return R


def convert_to_local_coordinates(global_points, translation, rotation):
    
    relative_points = global_points - translation

   
    R = quaternion_to_rotation_matrix(rotation)

   
    local_points = np.dot(R.T, relative_points.T).T
    return local_points


def process_trajectory(pred_traj_file, sample_data_file, ego_pose_file):

    pred_data = read_json(pred_traj_file)
    sample_data = read_json(sample_data_file)
    ego_pose_data = read_json(ego_pose_file)
    

    converted_trajectories = []


    for item in pred_data:
        token = item['token']
        pred_trajectory = item['predicted_trajectory']
        

        ego_pose_token = get_ego_pose_token(sample_data, token)
        

 
        translation, rotation = get_ego_pose(ego_pose_data, ego_pose_token)
        
        if translation is None or rotation is None:
            print(f"ego_pose_token {ego_pose_token} not found in ego_pose_data")
            continue
        

        local_trajectory = []
        for point in pred_trajectory:
            point_3d = point + [0.0]
            # print(point_3d)

            local_point = convert_to_local_coordinates(np.array(point_3d), translation, rotation)
            local_trajectory.append([local_point[0],local_point[1]]) 
        # print(token)
            

        converted_trajectories.append({
            "token": token,
            "traj": local_trajectory
        })
    
    return converted_trajectories

# # 输入文件路径
# pred_traj_file = '/home/ldc/Projects/RoboLLM/zoo/MMDrive/projects/Openemma/prepare/pred_traj_20250226_194229.json'  # 你的预测轨迹文件路径
# sample_data_file = '/home/ldc/Projects/RoboLLM/zoo/MMDrive/data/nuscenes/v1.0-trainval/sample_data.json'  
# ego_pose_file = '/home/ldc/Projects/RoboLLM/zoo/MMDrive/data/nuscenes/v1.0-trainval/ego_pose.json'

# # 处理并获取转换后的轨迹数据
# converted_trajectories = process_trajectory(pred_traj_file, sample_data_file, ego_pose_file)

# # 输出转换后的结果
# output_file = '/home/ldc/Projects/RoboLLM/zoo/MMDrive/projects/Openemma/prepare/converted_trajectories.json'  # 输出路径
# with open(output_file, 'w') as f:
#     json.dump(converted_trajectories, f, indent=4)

# print(f"Converted trajectories saved to {output_file}")
