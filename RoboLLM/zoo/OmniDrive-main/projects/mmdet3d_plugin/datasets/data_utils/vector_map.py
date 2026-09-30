import numpy as np
# from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from projects.mmdet3d_plugin.datasets.eval_utils.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString
import os
import json
CLASS2LABEL = {
    'road_divider': 0,
    'lane_divider': 0,
    'ped_crossing': 1,
    'contours': 2,
    'others': -1
}

class VectorizedLocalMap(object):
    def __init__(self,
                 dataroot,
                 patch_size,
                 canvas_size,
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 normalize=False,
                 fixed_num=-1,
                 dataset_type='nuScenes'):
        '''
        Args:
            fixed_num = -1 : no fixed num
        '''
        super().__init__()
        self.data_root = dataroot
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.nusc_maps = {}
        self.map_explorer = {}
        if dataset_type not in ['spd', 'nuScenes']:
            raise ("Error", dataset_type)
        if  dataset_type == 'spd':
            self.MAPS = ['yizhuang06', 'yizhuang08', 'yizhuang09', 'yizhuang10', 'yizhuang13','yizhuang16']
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(dataroot=self.data_root, map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        self.patch_size = patch_size
        self.canvas_size = canvas_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.normalize = normalize
        self.fixed_num = fixed_num

    def _parse_coord_str(self, coord):
        """
        [新增辅助工具] 解析 yizhuang06.json 特有的 "(x, y)" 字符串格式
        """
        if isinstance(coord, str):
            # 去除括号并分割
            parts = coord.strip("()").split(",")
            return [float(parts[0]), float(parts[1])]
        return coord # 如果已经是列表则直接返回
    
    def _global_to_ego(self, points, translation, rotation):
        """
        [新增辅助工具] 坐标转换：全局 -> 自车
        """
        # 1. 平移 (Global - Ego_Translation)
        points_3d = np.hstack((points, np.zeros((points.shape[0], 1)))) 
        points_centered = points_3d - np.array(translation)
        
        # 2. 旋转 (Inverse Rotation: Global -> Ego)
        # NuScenes 的 rotation 是 Ego->Global，所以要乘逆矩阵
        q = Quaternion(rotation)
        q_inv = q.inverse
        
        # 对点集进行旋转
        # q_inv.rotate 方法通常一次处理一个向量，这里用矩阵乘法加速
        # R_inv * P_centered^T
        rot_mat_inv = q_inv.rotation_matrix
        points_ego_3d = np.dot(points_centered, rot_mat_inv.T)
        
        return points_ego_3d[:, :2]
    
    def _resolve_lane_data(self, nusc_map, raw_map_data, token):
        """
        尝试从 map_api 或 原始 JSON 中提取 geometry 和 topology
        """
        # 来源 A: map_api (标准途径)
        record_api = nusc_map.get('lane', token) or nusc_map.get('lane_connector', token)
        
        # 来源 B: 原始 JSON (针对 Yizhuang06 这种特殊格式)
        # 注意：Yizhuang06 的 Key 是 'LANE'，ID 是 key
        record_raw = None
        if raw_map_data and 'lane' in raw_map_data:
            record_raw = raw_map_data['lane'].get(token)
        
        # --- 1. 提取 Centerline ---
        centerline = None
        
        # 尝试 1: 原始 JSON 的显式字段 (字符串格式 "(x,y)")
        if record_raw and 'centerline' in record_raw:
            raw_cl = record_raw['centerline']
            if len(raw_cl) > 0:
                # 解析字符串
                centerline = np.array([self._parse_coord_str(p) for p in raw_cl])
        print("centerline1",centerline)
        # 尝试 2: map_api 的显式字段 (如果它成功加载了的话)
        if centerline is None and record_api and 'centerline' in record_api:
            raw_cl = record_api['centerline']
            if len(raw_cl) > 0:
                centerline = np.array([self._parse_coord_str(p) for p in raw_cl])
                print("centerline2",centerline)
                
        # 尝试 3: 标准 Arcline 离散化
        if centerline is None:
            try:
                cl_list = nusc_map.discretize_lane(token, resolution_meters=1.0)
                if len(cl_list) > 0:
                    centerline = np.array(cl_list)
                print("centerline3",centerline)
                
            except:
                pass
                
        # --- 2. 提取 Topology ---
        # 优先用原始 JSON 的数据，因为 map_api 可能没加载进去
        source = record_raw if record_raw else record_api
        if not source:
            return None, {}, {} # 没找到记录
            
        topo = {
            'l_id': source.get('l_neighbor_id') or source.get('left_lane_token'),
            'r_id': source.get('r_neighbor_id') or source.get('right_lane_token'),
            'preds': source.get('predecessors') or source.get('incoming_tokens', []),
            'succs': source.get('successors') or source.get('outgoing_tokens', []),
            'type': source.get('lane_type', 'CITY_DRIVING')
        }
        
        return centerline, topo, source
    
    def vectors_to_graph(self, location, ego2global_translation, ego2global_rotation):
        
        # --- 1. 准备数据源 ---
        if location not in self.map_explorer:
            print(f"[Error] Location {location} not found in map_explorer")
            return []
        
        nusc_map = self.map_explorer[location].map_api
        
        # [关键]：手动加载一次原始 JSON，作为 map_api 的备胎
        if not hasattr(self, '_raw_map_cache'):
            self._raw_map_cache = {}
            
        if location not in self._raw_map_cache:
            # 拼凑路径，根据你的目录结构可能需要调整
            # 尝试标准路径和 expansion 路径
            candidates = [
                os.path.join(self.data_root, 'maps', 'expansion', f'{location}.json'),
                os.path.join(self.data_root, f'{location}.json'),
                os.path.join(nusc_map.dataroot, 'maps', 'expansion', f'{location}.json')
            ]
            raw_data = {}
            for p in candidates:
                if os.path.exists(p):
                    print(f"[Info] Loading Raw Map from: {p}")
                    try:
                        with open(p, 'r') as f:
                            raw_data = json.load(f)
                        break
                    except: pass
            self._raw_map_cache[location] = raw_data
            
        raw_map_data = self._raw_map_cache[location]
        # ================== DEBUG START ==================
        print("\n" + "="*30 + f" DEBUG: {location} " + "="*30)
        print(f"[DEBUG] raw_map_data Type: {type(raw_map_data)}")
        
        if raw_map_data is None:
            print("[ERROR] raw_map_data is None!")
        
        elif isinstance(raw_map_data, dict):
            keys = list(raw_map_data.keys())
            print(f"[DEBUG] Top-level Keys (first 10): {keys[:10]}")
            
            # 自动寻找可能的车道数据 Key
            target_key = None
            if 'LANE' in raw_map_data: target_key = 'LANE'
            elif 'lane' in raw_map_data: target_key = 'lane'
            elif 'lane_connector' in raw_map_data: target_key = 'lane_connector'
            
            if target_key:
                container = raw_map_data[target_key]
                print(f"[DEBUG] Found key '{target_key}' containing type: {type(container)}")
                
                # 提取一个样本进行解剖
                sample_item = None
                if isinstance(container, list) and len(container) > 0:
                    print(f"[DEBUG] Structure is LIST. Total items: {len(container)}")
                    sample_item = container[0]
                elif isinstance(container, dict) and len(container) > 0:
                    print(f"[DEBUG] Structure is DICT. Total items: {len(container)}")
                    sample_item = list(container.values())[0]
                
                if sample_item:
                    print(f"[DEBUG] --- Sample Item in '{target_key}' ---")
                    print(f"  Keys: {list(sample_item.keys())}")
                    
                    if 'centerline' in sample_item:
                        cl_val = sample_item['centerline']
                        print(f"  'centerline' type: {type(cl_val)}")
                        print(f"  'centerline' content (preview): {cl_val[:3] if isinstance(cl_val, list) else cl_val}")
                    else:
                        print("  [WARNING] 'centerline' key NOT FOUND in this item.")
                        
                    if 'polygon_token' in sample_item:
                         print(f"  'polygon_token': {sample_item['polygon_token']}")
            else:
                print("[WARNING] No 'lane' or 'LANE' key found in top-level.")

        print("="*80 + "\n")
        # ================== DEBUG END ==================

        # --- 2. 空间搜索 ---
        ego_pos = np.array(ego2global_translation[:2])
        search_radius = max(self.patch_size) / 2.0 
        
        # 使用 map_api 进行快速索引 (假设 ID 索引是正常的)
        layers = ['lane', 'lane_connector']
        nearby_tokens = nusc_map.get_records_in_radius(ego_pos[0], ego_pos[1], search_radius, layers)
        all_lane_tokens = nearby_tokens.get('lane', []) + nearby_tokens.get('lane_connector', [])
        
        # [兜底] 如果 map_api 连 token 都没搜到 (说明 map_api 加载彻底失败)，则暴力遍历 Raw JSON
        if len(all_lane_tokens) == 0 and 'LANE' in raw_map_data:
            print("[Warning] map_api returned 0 tokens. Falling back to brute-force search in Raw JSON.")
            for token, info in raw_map_data['LANE'].items():
                # 简单粗暴取第一个点做距离判断
                if 'centerline' in info and len(info['centerline']) > 0:
                    pt = self._parse_coord_str(info['centerline'][0])
                    if np.linalg.norm(np.array(pt) - ego_pos) < search_radius:
                        all_lane_tokens.append(token)

        token2idx = {token: i for i, token in enumerate(all_lane_tokens)}
        graph_nodes = []

        # --- 3. 构建节点 ---
        for i, token in enumerate(all_lane_tokens):
            # 调用上面的解析函数
            centerline, topo, record = self._resolve_lane_data(nusc_map, raw_map_data, token)
            
            if centerline is None or len(centerline) == 0:
                # print(f"[Warning] Still no geometry for {token}")
                continue

            # 坐标转换
            centerline_ego = self._global_to_ego(centerline, ego2global_translation, ego2global_rotation)
            
            # 拓扑索引转换
            def get_indices(tokens, type_id=0):
                if not tokens: return []
                if isinstance(tokens, str): tokens = [tokens]
                res = []
                for t in tokens:
                    if t in token2idx:
                        res.append([token2idx[t], type_id])
                return res

            # 组装
            node = {
                'id': token,
                'xy': centerline_ego,
                'yaw': np.zeros(len(centerline_ego)), # 简化 yaw，后续可计算
                'left': get_indices(topo['l_id']),
                'right': get_indices(topo['r_id']),
                'prev': get_indices(topo['preds']),
                'follow': get_indices(topo['succs']),
                'type': 0,
                'signal': []
            }
            
            # 补充 Yaw 计算
            if len(centerline_ego) > 1:
                diff = centerline_ego[1:] - centerline_ego[:-1]
                yaws = np.arctan2(diff[:, 1], diff[:, 0])
                yaws = np.concatenate([yaws, yaws[-1:]])
                node['yaw'] = yaws

            graph_nodes.append(node)
            
        return graph_nodes
    
    def gen_vectorized_samples(self, location, ego2global_translation, ego2global_rotation):
        map_pose = ego2global_translation[:2]
        rotation = Quaternion(ego2global_rotation)

        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        line_geom = self.get_map_geom(patch_box, patch_angle, self.line_classes, location)
        line_vector_dict = self.line_geoms_to_vectors(line_geom)

        ped_geom = self.get_map_geom(patch_box, patch_angle, self.ped_crossing_classes, location)
        # ped_vector_list = self.ped_geoms_to_vectors(ped_geom)
        ped_vector_list = self.line_geoms_to_vectors(ped_geom)['ped_crossing']

        polygon_geom = self.get_map_geom(patch_box, patch_angle, self.polygon_classes, location)
        poly_bound_list = self.poly_geoms_to_vectors(polygon_geom)

        vectors = []
        for line_type, vects in line_vector_dict.items():
            for line, length in vects:
                vectors.append((line.astype(float), length, CLASS2LABEL.get(line_type, -1)))

        for ped_line, length in ped_vector_list:
            vectors.append((ped_line.astype(float), length, CLASS2LABEL.get('ped_crossing', -1)))

        for contour, length in poly_bound_list:
            vectors.append((contour.astype(float), length, CLASS2LABEL.get('contours', -1)))

        # filter out -1
        filtered_vectors = []
        for pts, pts_num, type in vectors:
            if type != -1:
                filtered_vectors.append({
                    'pts': pts,
                    'pts_num': pts_num,
                    'type': type
                })

        return filtered_vectors

    def get_map_geom(self, patch_box, patch_angle, layer_names, location):
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.map_explorer[location]._get_layer_line(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_ped_crossing_line(patch_box, patch_angle, location)
                # geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
        return map_geom

    def _one_type_line_geom_to_vectors(self, line_geom):
        line_vectors = []
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_vectors.append(self.sample_pts_from_line(single_line))
                elif line.geom_type == 'LineString':
                    line_vectors.append(self.sample_pts_from_line(line))
                else:
                    raise NotImplementedError
        return line_vectors

    def poly_geoms_to_vectors(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def line_geoms_to_vectors(self, line_geom):
        line_vectors_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_vectors = self._one_type_line_geom_to_vectors(a_type_of_lines)
            line_vectors_dict[line_type] = one_type_vectors

        return line_vectors_dict

    def ped_geoms_to_vectors(self, ped_geom):
        ped_geom = ped_geom[0][1]
        union_ped = ops.unary_union(ped_geom)
        if union_ped.geom_type != 'MultiPolygon':
            union_ped = MultiPolygon([union_ped])

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for ped_poly in union_ped:
            # rect = ped_poly.minimum_rotated_rectangle
            ext = ped_poly.exterior
            if not ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def get_ped_crossing_line(self, patch_box, patch_angle, location):
        def add_line(poly_xy, idx, patch, patch_angle, patch_x, patch_y, line_list):
            points = [(p0, p1) for p0, p1 in zip(poly_xy[0, idx:idx + 2], poly_xy[1, idx:idx + 2])]
            line = LineString(points)
            line = line.intersection(patch)
            if not line.is_empty:
                line = affinity.rotate(line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                line = affinity.affine_transform(line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(line)

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        line_list = []
        records = getattr(self.nusc_maps[location], 'ped_crossing')
        for record in records:
            polygon = self.map_explorer[location].extract_polygon(record['polygon_token'])
            poly_xy = np.array(polygon.exterior.xy)
            dist = np.square(poly_xy[:, 1:] - poly_xy[:, :-1]).sum(0)
            x1, x2 = np.argsort(dist)[-2:]

            add_line(poly_xy, x1, patch, patch_angle, patch_x, patch_y, line_list)
            add_line(poly_xy, x2, patch, patch_angle, patch_x, patch_y, line_list)

        return line_list

    def sample_pts_from_line(self, line):
        if self.fixed_num < 0:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
        else:
            # fixed number of points, so distance is line.length / self.fixed_num
            distances = np.linspace(0, line.length, self.fixed_num)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)

        if self.normalize:
            sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])

        num_valid = len(sampled_points)

        if not self.padding or self.fixed_num > 0:
            # fixed num sample can return now!
            return sampled_points, num_valid

        # fixed distance sampling need padding!
        num_valid = len(sampled_points)

        if self.fixed_num < 0:
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate([sampled_points, padding], axis=0)
            else:
                sampled_points = sampled_points[:self.num_samples, :]
                num_valid = self.num_samples

            if self.normalize:
                sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])
                num_valid = len(sampled_points)

        return sampled_points, num_valid
