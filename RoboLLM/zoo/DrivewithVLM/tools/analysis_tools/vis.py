import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from nuscenes import NuScenes
import mmcv
from pyquaternion import Quaternion
import sys
sys.path.append('//home/ldc/Projects/RoboLLM/corruptions')
from image_corruption.Camera_Method import CameraMethods
class image_corruption():
    def __init__(self,corruption_severity_dict=
            {
                'fog_sim':3,
            }):
            self.corruption_severity_dict=corruption_severity_dict

            self.corruption=CameraMethods(corruption_severity_dict=corruption_severity_dict)
            self.first_key = next(iter(corruption_severity_dict.keys()))
            self.first_key = next(iter(corruption_severity_dict.keys()))
            self.first_value = corruption_severity_dict[self.first_key]
            print("self.corruption",self.first_key)

class CamParams:
    def __init__(self, K, R, t, frame_id=None):
        """Construct camera parameter object

        Args:
            K (3x3 matrix): Camera intrinsics
            R (3x3 matrix): Camera rotation
            t (3x1 vector): Camera translation
            frame_id: Reference frame id

            K: 相机内参矩阵（3×3），定义了焦距、光心等参数。
            R: 旋转矩阵（3×3），用于相机坐标变换。
            t: 平移向量（3×1），定义相机在某一参考坐标系中的位置。
            frame_id: 可选参数，表示相机参考的坐标系（例如“ego”或“lidar”）
        """
        self.K = K
        self.R = R
        self.t = t.reshape(3, 1)
        # Affine Transform 计算 仿射变换矩阵 A，用于将世界坐标转换为相机坐标
        self.A = np.column_stack((self.R.T, - self.R.T @ self.t))
        # Projection Matrix 计算投影矩阵 P：
        # P = K @ A 组合了相机的内参和外参（R 和 t），用于将 3D 点投影到 2D 图像坐标。
        self.P = K @ self.A
        self.frame_id = frame_id




def read_cam_params(nusc, sample_token, cam_name, ref_frame='ego'):
    """
    读取 nuScenes 相机参数，支持 ego / lidar 坐标参考系

    Args:
        nusc (NuScenes): NuScenes 实例
        sample_token (str): 某个 sample 的 token
        cam_name (str): 相机名，例如 "CAM_FRONT"
        ref_frame (str): 参考坐标系，可选 'ego' 或 'lidar'

    Returns:
        CamParams
    """
    sample = nusc.get('sample', sample_token)
    cam_data = nusc.get('sample_data', sample['data'][cam_name])
    calibrated = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    cam_intrinsic = np.array(calibrated['camera_intrinsic'])

    sensor2ego_t = np.array(calibrated['translation'])  # 3,
    sensor2ego_q = Quaternion(calibrated['rotation'])   # 四元数 (w, x, y, z)
    sensor2ego_R = sensor2ego_q.rotation_matrix         # 3x3

    if ref_frame == 'ego':
        rotation = sensor2ego_R
        translation = sensor2ego_t
    elif ref_frame == 'lidar':
        lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

        # Lidar to ego pose
        ego2lidar_q = Quaternion(lidar_calib['rotation'])
        ego2lidar_R = ego2lidar_q.rotation_matrix
        ego2lidar_t = np.array(lidar_calib['translation'])

        rotation = ego2lidar_R.T @ sensor2ego_R
        translation = ego2lidar_R.T @ (sensor2ego_t - ego2lidar_t)
    else:
        raise ValueError(f"Unsupported ref_frame: {ref_frame}")

    return CamParams(cam_intrinsic, rotation, translation, ref_frame)
## 投影 3D 点到 2D
def proj_3d_point(point, cam_params):
    point = np.append(point, 1)
    # print(cam_params.A @ point)
    point = cam_params.P @ point
    if point[2] > 0:
        point = point[:2]/point[2]
        point = point.astype(np.int16)
        return point
    else:
        return None  # Point is behind the camera



def draw_traj(img, traj, cam_params, color=(0, 255, 0), car_width=-1, car_length=-1, thickness=2):
    projected_traj = []
    traj_3d = []
    polygon_sides = [[], []]

    for i, pt in enumerate(traj):
        pt3d_corrected = np.array([pt[1], -pt[0], 0])
        traj_3d.append(pt3d_corrected)
        _pt = proj_3d_point(pt3d_corrected, cam_params)
        if _pt is not None:
            projected_traj.append(_pt)
            cv2.circle(img, (_pt[0], _pt[1]), thickness*3, color, thickness=-1)

            if car_width > 0:
                n = traj_3d[i] - \
                    traj_3d[i-1] if i > 0 else np.array([1.0, 0])
                n = n / np.linalg.norm(n)
                n_rt = np.array([n[1], -n[0], 0])
                polygon_sides[0].append(proj_3d_point(
                    pt3d_corrected + n_rt * car_width / 2.0, cam_params))
                polygon_sides[1].append(proj_3d_point(
                    pt3d_corrected - n_rt * car_width / 2.0, cam_params))
                if i == len(traj) - 1 and car_length > 0:
                    polygon_sides[0].append(proj_3d_point(
                        pt3d_corrected + n_rt * car_width / 2.0 + n * car_length / 2, cam_params))
                    polygon_sides[1].append(proj_3d_point(
                        pt3d_corrected - n_rt * car_width / 2.0 + n * car_length / 2, cam_params))
        else:
            print("WARNING: Traj point behind!")

    projected_traj = np.array(projected_traj, dtype=int)

    for i in range(len(projected_traj)-1):
        cv2.line(img, tuple(projected_traj[i]), tuple(
            projected_traj[i+1]), color, thickness)

    if polygon_sides[0] and polygon_sides[1]:
        occupied_polygon = np.array(
            polygon_sides[0] + list(reversed(polygon_sides[1])), dtype=int)
        frame = np.zeros_like(img)
        cv2.fillPoly(frame, [occupied_polygon], color)
        alpha = 0.5
        mask = frame.astype(bool)
        img[mask] = cv2.addWeighted(img, alpha, frame, 1 - alpha, 0)[mask]

    return img


if __name__ == "__main__":
    import json
    import os
    
    current_dir='/home/ldc/Projects/RoboLLM/zoo/MMDrive/tools/VIS_PLANNING'
    car_width = 1.73  # Renault Zoe
    car_length = 4.084  # Renault Zoe
    dataroot='data/nuscenes/'

    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes')


    outputs = mmcv.load("/home/ldc/Projects/RoboLLM/zoo/MMDrive/vis/pkl/result_tta_sun_sim_5_20250520_170430.pkl")

    for token in outputs:
        

        sample = nusc.get('sample', token)


        cam_token = sample['data']['CAM_FRONT']

 
        sample_data = nusc.get('sample_data', cam_token)

    
        img_path = dataroot+sample_data['filename']

        img = cv2.imread(img_path)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

        ic=image_corruption()
        img=ic.corruption(img_rgb)
        img= cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


        cam_lidar_params = read_cam_params(nusc, token, "CAM_FRONT", ref_frame='lidar')
        cam_ego_params = read_cam_params(nusc, token, "CAM_FRONT", ref_frame='ego')
        
        trajs = outputs[token][0]



        draw_traj(img, trajs, cam_ego_params,
                car_width=car_width, car_length=car_length)
        cv2.imwrite(os.path.join(current_dir,f"output_{token}.png"), img)


with open(current_dir+'/gt/gt_traj.pkl','rb') as f:
    gt_trajs_dict = pickle.load(f)



for index, token in enumerate(tqdm(pred_trajs_dict.keys())):
    gt_trajectory =  torch.tensor(gt_trajs_dict[token])
    gt_trajectory = gt_trajectory.to(device)

    gt_traj_mask = torch.tensor(gt_trajs_mask_dict[token])
    gt_traj_mask = gt_traj_mask.to(device)

    output_trajs =  torch.tensor(pred_trajs_dict[token])
    output_trajs = output_trajs.reshape(gt_traj_mask.shape)
    output_trajs = output_trajs.to(device)