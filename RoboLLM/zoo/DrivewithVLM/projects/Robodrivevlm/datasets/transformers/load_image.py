
from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmcv.transforms.base import BaseTransform
from mmdet3d.registry import TRANSFORMS
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import torch
import cv2

@TRANSFORMS.register_module()
class LoadMultiViewImageFromFiles4Clip(LoadMultiViewImageFromFiles):
    def __init__(self,test_mode,dataroot=None,only_vehicle=False,**kwargs):
        super(LoadMultiViewImageFromFiles4Clip, self).__init__(**kwargs)
        self.dataroot=dataroot
        self.test_mode=test_mode
        self.only_vehicle = only_vehicle

    def transform(self, results) -> dict:

        image_sources = []

        if self.only_vehicle and 'img_filename' in results:
            for relative_path in results['img_filename']:
                # 这里判断路径中是否包含 vehicle-side
                if 'vehicle-side' in relative_path:
                    if self.dataroot:
                        image_sources.append(os.path.join(self.dataroot, relative_path))
                    else:
                    # 如果没有 dataroot，则直接使用相对路径，这可能导致文件找不到
                        image_sources.append(relative_path)
            
            
        elif 'img_filename' in results and not self.only_vehicle and results['img_filename']:
            
            # 将每个相对路径与 self.dataroot 拼接起来
            for relative_path in results['img_filename']:
                # 检查 self.dataroot 是否存在，如果存在则拼接
                if self.dataroot:
                    image_sources.append(os.path.join(self.dataroot, relative_path))
                else:
                    # 如果没有 dataroot，则直接使用相对路径，这可能导致文件找不到
                    image_sources.append(relative_path)
        
        # 3. 如果 results['img_filename'] 不存在或为空，退回到硬编码逻辑（可选，用于兼容老代码）
        else:
            print("Warning: 'img_filename' not found in results. Falling back to hardcoded paths.")

        results['input_images'] = []
        for image_path in image_sources:
            try:
                results['input_images'].append(
                    Image.open(image_path).convert("RGB")
                )
            except FileNotFoundError as e:
                print(f"Error loading image: {image_path}. Check dataroot and file path.")
        
        # print("results after image")
        # print(image_sources)
        return results
    
@TRANSFORMS.register_module()
class LoadMultiViewImageFromFiles4Clip_EGO(LoadMultiViewImageFromFiles):
    def __init__(self,test_mode,dataroot=None,**kwargs):
        super(LoadMultiViewImageFromFiles4Clip_EGO, self).__init__(**kwargs)
        self.dataroot=dataroot
        self.test_mode=test_mode

    def transform(self, results) -> dict:
        image_sources = []
        if 'img_filename' in results and results['img_filename']:
            for relative_path in results['img_filename']:
                if self.dataroot:
                    image_sources.append(os.path.join(self.dataroot, relative_path))
                else:
                    image_sources.append(relative_path)
        else:
            print("Warning: 'img_filename' not found. Falling back.")

        results['input_images'] = []
        results['infra_ego_pixel'] = None
        results['is_visible'] = False
        # veh2inf_rt: 自车雷达 -> 路侧雷达 [4, 4]
        veh2inf_rt = results.get('veh2inf_rt', None)
        
        if isinstance(veh2inf_rt, torch.Tensor):
            veh2inf_rt = veh2inf_rt.numpy()
            
        if isinstance(veh2inf_rt, np.ndarray) and veh2inf_rt.ndim == 3:
            veh2inf_rt = veh2inf_rt[0] # (1, 4, 4) -> (4, 4)

        veh2inf_rt = veh2inf_rt.T ###
        for i, image_path in enumerate(image_sources):
            try:
                img = Image.open(image_path).convert("RGB")
                
                # 判断是否为路侧图像
                is_infra_image = 'infrastructure-side' in image_path
                
                if is_infra_image and veh2inf_rt is not None:
                    draw = ImageDraw.Draw(img)
                    
                    # --- 🚀 修复 2: 获取当前视角的矩阵并清洗维度 ---
                    # 必须使用 [i] 获取当前摄像头的参数
                    lidar2cam = results['lidar2cam'][i]
                    
                    intrinsic = results['cam_intrinsic'][i]
                    
                    # 转换为 numpy
                    if isinstance(lidar2cam, torch.Tensor): 
                        lidar2cam = lidar2cam.numpy()
                    elif isinstance(lidar2cam, list):
                        lidar2cam = np.array(lidar2cam)

                    if isinstance(intrinsic, torch.Tensor): 
                        intrinsic = intrinsic.numpy()
                    elif isinstance(intrinsic, list):
                        intrinsic = np.array(intrinsic)
                    
                    # 关键修复：如果矩阵形状是 (1, 4, 4)，必须变为 (4, 4)
                    if lidar2cam.ndim == 3: lidar2cam = lidar2cam[0]
                    if intrinsic.ndim == 3: intrinsic = intrinsic[0]
                    
                    # 2. 定义自车中心点 (在自车雷达坐标系下为原点)
                    # 形状 [4, 1]
                    ego_center_local = np.array([0, 0, 0, 1]).reshape(4, 1)
                    
                    # 3. 坐标变换链
                    # Step A: 自车雷达 -> 路侧雷达
                    # 此时 veh2inf_rt 是 (4, 4), ego_center_local 是 (4, 1) -> 结果 (4, 1)
                    ego_in_inf_lidar = veh2inf_rt @ ego_center_local
                    
                    # Step B: 路侧雷达 -> 路侧相机
                    # 此时 lidar2cam 是 (4, 4), ego_in_inf_lidar 是 (4, 1) -> 结果 (4, 1)
                    ego_in_inf_cam = lidar2cam @ ego_in_inf_lidar
                   
                    # Step C: 相机坐标系 -> 像素平面 (透视投影)
                    # 提取 xyz (前3维)
                    # 此时 ego_in_inf_cam 形状是 (4, 1)，直接切片取值
                    cam_x = ego_in_inf_cam[0, 0]
                    cam_y = ego_in_inf_cam[1, 0]
                    cam_z = ego_in_inf_cam[2, 0]
                    
                    # 过滤掉相机后方的点 (z < 0 通常表示在相机后面)
                    if cam_z > 0:
                        # 使用内参投影
                        # intrinsic (4, 4) @ ego_in_inf_cam (4, 1) -> uv_homo (4, 1)
                        uv_homo = intrinsic @ ego_in_inf_cam 
                        
                        # 归一化 (u/z, v/z)
                        # --- 🚀 修复 3: 正确的索引访问 ---
                        # uv_homo 形状是 (4, 1)
                        u = uv_homo[0, 0] / cam_z
                        v = uv_homo[1, 0] / cam_z
                        pixel_offset_y = 1500 / cam_z 
                        
                        # 修正后的中心点
                        center_u = u
                        center_v = v + pixel_offset_y
                        
                        # 4. 绘制标记
                        W, H = img.size
                        results['infra_ego_pixel'] = [round(float(u), 1), round(float(v), 1)]
                        if 0 <= center_u < W and 0 <= center_v < H:
                            results['is_visible'] = True
                            if not self.test_mode:
                                r = max(8, min(20, int(400 / cam_z))) 
                                
                                # 1. 画一个实心红点 (Core)
                                draw.ellipse((center_u-r, center_v-r, center_u+r, center_v+r), fill="red")
                                
                                # 2. 画一个白色外圈 (让红点在深色背景下也明显)
                                draw.ellipse((center_u-r, center_v-r, center_u+r, center_v+r), outline="white", width=2)
                                
                                # 3. 再画一个更大的红色空心圈 (Target style)
                                r_outer = r + 5
                                draw.ellipse((center_u-r_outer, center_v-r_outer, center_u+r_outer, center_v+r_outer), outline="red", width=3)

                                # --- 绘制文字 ---
                                font_size = max(60, min(150, int(r * 7)))
                                try:
                                    # 尝试加载字体，如果不行就用默认
                                    font = ImageFont.truetype("arial.ttf", font_size)
                                except IOError:
                                    font = ImageFont.load_default()
                                
                                # 文字位置：点上方
                                text = "EGO"
                                
                                # 如果您的 PIL 版本支持 textbbox (Pillow >= 9.2.0)
                                if hasattr(draw, "textbbox"):
                                    bbox = draw.textbbox((0, 0), text, font=font)
                                    text_w = bbox[2] - bbox[0]
                                    text_h = bbox[3] - bbox[1]
                                else:
                                    # 旧版本兼容
                                    text_w, text_h = draw.textsize(text, font=font)
                                
                                text_x = center_u - text_w / 2
                                text_y = center_v - r_outer - text_h - 10
                                
                                padding = 10
                                
                                # 画大背景框
                                draw.rectangle(
                                    (text_x - padding, text_y - padding, 
                                    text_x + text_w + padding, text_y + text_h + padding), 
                                    fill="white", 
                                    outline="red", # 加个红边框更醒目
                                    width=3
                                )
                                # 画文字
                                draw.text((text_x, text_y), text, font=font, fill="red")
                        else:
                            results['infra_ego_pixel'] = cam_z

                    # print(f"--- Debugging Image {i} ({'Infrastructure' if is_infra_image else 'Vehicle'}) ---")
                    # print(f"  Veh2Inf_RT exists: {veh2inf_rt is not None}")

                    # print(f"  Lidar2Cam Shape: {lidar2cam.shape}") # 应该是 (4, 4)
                    # print(f"  Intrinsic Shape: {intrinsic.shape}") # 应该是 (4, 4)
                    # print(f"  Ego Center (Local): {ego_center_local.flatten()}") # [0, 0, 0, 1]
                    # print(f"  Ego in Inf Lidar: {ego_in_inf_lidar.flatten()[:3]}") # 看一下数值是否合理
                    # print(f"  Ego in Inf Cam (xyz): {ego_in_inf_cam.flatten()[:3]}") # 重点看 z (第三个值)
                    # print(f"  Projected Pixel (u, v): {u:.1f}, {v:.1f}")
                    # print(f"  Image Size: {W}x{H}")

                results['input_images'].append(img)
                
                
            except FileNotFoundError as e:
                print(f"Error loading image: {image_path}")

        # # 保存可视化结果 (拼接图像)
        # if 'input_images' in results and len(results['input_images']) >= 2 and not self.test_mode:
        #     try:
        #         veh_img = results['input_images'][0]
        #         inf_img = results['input_images'][1]

        #         target_height = veh_img.height
        #         aspect_ratio = inf_img.width / inf_img.height
        #         new_width = int(target_height * aspect_ratio)
        #         inf_img_resized = inf_img.resize((new_width, target_height))

        #         total_width = veh_img.width + inf_img_resized.width
        #         combined_img = Image.new('RGB', (total_width, target_height))

        #         combined_img.paste(veh_img, (0, 0))
        #         combined_img.paste(inf_img_resized, (veh_img.width, 0))

        #         token = results.get('token', results.get('sample_idx'))
                
        #         save_dir = "vis/visualization"
        #         if not os.path.exists(save_dir):
        #             os.makedirs(save_dir)

        #         save_path = os.path.join(save_dir, f"{token}.jpg")
        #         combined_img.save(save_path)

            # except Exception as e:
            #     print(f"[Warning] Failed to save visualization: {e}")
                
        return results
    
@TRANSFORMS.register_module()
class Visualize3DBoxOnImage(BaseTransform):

    def __init__(self, 
                 img_root='',
                 save_dir='./vis_debug_vehicle',
                 max_distance=50.0):
        super().__init__()
        self.img_root = img_root
        self.save_dir = save_dir
        self.max_distance = max_distance
        
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    def project_lidar_to_img_two_steps(self, points_3d, lidar2cam, cam2img):
        """
        分两步投影：Lidar -> Cam -> Img
        """
        N = len(points_3d)
        points_4d = np.concatenate([points_3d, np.ones((N, 1))], axis=-1)
        
        # 1. Lidar -> Camera
        # 注意：这里假设输入的 lidar2cam 是标准的变换矩阵 (4x4)
        # points_4d 是 (N, 4)，所以用 @ matrix.T
        points_cam_4d = points_4d @ lidar2cam.T
        points_cam_3d = points_cam_4d[..., :3]
        
        # 2. 深度过滤 (Z > 0.1)
        depth = points_cam_3d[..., 2]
        valid_mask = depth > 0.1
        
        # 3. Camera -> Image
        points_img_3d = points_cam_4d @ cam2img.T
        points_2d = points_img_3d[..., :2] / (points_img_3d[..., 2:3] + 1e-5)
        
        return points_2d, valid_mask

    def draw_box(self, img, corners_2d, valid_mask, color=(0, 255, 0), thickness=2):
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        H, W = img.shape[:2]
        lines_drawn = 0
        
        # 绘制角点
        for i, (u, v) in enumerate(corners_2d):
            if valid_mask[i]:
                # 0号点(左后底角)画红点区分方向
                radius = 4 if i == 0 else 2
                cv_color = (0, 0, 255) if i == 0 else (0, 255, 255)
                cv2.circle(img, (int(u), int(v)), radius, cv_color, -1)

        # 绘制连线
        for start, end in edges:
            if valid_mask[start] and valid_mask[end]:
                pt1 = (int(corners_2d[start][0]), int(corners_2d[start][1]))
                pt2 = (int(corners_2d[end][0]), int(corners_2d[end][1]))
                
                # 简单的图像范围检查
                if -W < pt1[0] < 2*W and -H < pt1[1] < 2*H:
                    cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
                    lines_drawn += 1
        
        return lines_drawn

    def is_box_in_image(self, points, mask, W, H):
        valid_points = points[mask]
        if len(valid_points) == 0: return False
        u = valid_points[:, 0]
        v = valid_points[:, 1]
        return np.any((u >= 0) & (u < W) & (v >= 0) & (v < H))

    def transform(self, results) -> dict:
        # 1. 获取标注框 (GT BBoxes)
        # 优先从 results 获取处理好的 bbox，如果没有再尝试 ann_info
        gt_bboxes = results.get('gt_bboxes_3d', None)
        if gt_bboxes is None and 'ann_info' in results:
             gt_bboxes = results['ann_info'].get('gt_bboxes_3d', None)
        
        bboxes_corners = []
        if gt_bboxes is not None:
             if 'mmcv.parallel.data_container.DataContainer' in str(type(gt_bboxes)):
                 gt_bboxes = gt_bboxes.data
             
             if hasattr(gt_bboxes, 'corners'):
                 bboxes_corners = gt_bboxes.corners.numpy()
             elif isinstance(gt_bboxes, np.ndarray):
                 # 如果是原始 numpy (N, 7)，这里暂不处理，假设已经有 corners
                 pass

        # 2. 遍历图像路径进行可视化
        filename = results.get('img_filename', [])
        
        for i, img_path in enumerate(filename):
            # --- 过滤 1: 仅处理车端图像 ('vehicle-side') ---
            # 注意：这里的过滤逻辑应与你的 LoadImage 保持一致
            if 'vehicle-side' not in img_path:
                continue

            full_img_path = os.path.join(self.img_root, img_path)
            
            # 使用 cv2 读取图像用于画图 (BGR)
            # 注意：这里是独立读取，不干扰其他 pipeline 加载进模型的 Tensor
            vis_img = cv2.imread(full_img_path)
            if vis_img is None:
                continue
                
            H, W = vis_img.shape[:2]
            print(f"Visualizing on image: {full_img_path} (W: {W}, H: {H}) with {len(bboxes_corners)} boxes.")
            
            # 3. 获取投影矩阵
            l2c = None
            c2i = None
            
            # 外参 (Lidar -> Camera)
            if 'lidar2cam' in results: l2c = results['lidar2cam'][i]
            elif 'lidar2cam_rts' in results: l2c = results['lidar2cam_rts'][i]
            
            # 内参 (Camera Intrinsic)
            if 'cam_intrinsic' in results: c2i = results['cam_intrinsic'][i]
            elif 'cam2img' in results: c2i = results['cam2img'][i]
            elif 'cam_intrinsics' in results: c2i = results['cam_intrinsics'][i]

            # 转换 tensor -> numpy
            if l2c is not None and isinstance(l2c, torch.Tensor): l2c = l2c.numpy()
            if c2i is not None and isinstance(c2i, torch.Tensor): c2i = c2i.numpy()
            
            # Fallback 矩阵 (Lidar -> Img)
            l2i = None
            if 'lidar2img' in results:
                l2i = results['lidar2img'][i]
                if isinstance(l2i, torch.Tensor): l2i = l2i.numpy()

            # 4. 遍历并绘制 3D 框
            if len(bboxes_corners) > 0:
                for corners_3d in bboxes_corners:
                    # --- 过滤 2: 距离过滤 ---
                    center_3d = corners_3d.mean(axis=0)
                    dist = np.linalg.norm(center_3d[:2])
                    if dist > self.max_distance:
                        continue

                    points_2d = None
                    valid_mask = None
                    
                    # 方案 A: 分两步投影 (推荐)
                    if l2c is not None and c2i is not None:
                        points_2d, valid_mask = self.project_lidar_to_img_two_steps(corners_3d, l2c, c2i)
                    
                    # 方案 B: Fallback 使用 lidar2img
                    elif l2i is not None:
                        corners_4d = np.concatenate([corners_3d, np.ones((8, 1))], axis=-1)
                        pts_homo = corners_4d @ l2i.T
                        z = pts_homo[:, 2]
                        valid_mask = z > 0.1
                        points_2d = pts_homo[:, :2] / (z[:, None] + 1e-5)
                    
                    # 绘制
                    if points_2d is not None and np.any(valid_mask):
                        if self.is_box_in_image(points_2d, valid_mask, W, H):
                            self.draw_box(vis_img, points_2d, valid_mask)
            
            # 5. 保存图片
            if self.save_dir:
                sample_token = results.get('sample_idx', 'unknown')
                # 简化文件名，替换路径分隔符
                view_name = os.path.basename(img_path).split('.')[0]
                save_name = f"{sample_token}_{view_name}.jpg"
                save_path = os.path.join(self.save_dir, save_name)
                cv2.imwrite(save_path, vis_img)

            if 'input_images' in results and len(results['input_images']) > i:
                # OpenCV (BGR) 转为 RGB
                vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                # 转为 PIL Image (Clip 需要的格式)
                vis_img_pil = Image.fromarray(vis_img_rgb)
                # 替换原有图像
                results['input_images'][i] = vis_img_pil
                # print(f"replaced image {i}")

        return results
    

