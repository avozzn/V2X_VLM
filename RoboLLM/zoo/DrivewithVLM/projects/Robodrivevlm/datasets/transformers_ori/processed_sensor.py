from nuscenes.can_bus.can_bus_api import NuScenesCanBus

from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS
from PIL import Image
from matplotlib.axes import Axes  
import os.path as osp 
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import numpy as np
from nuscenes.utils.geometry_utils import view_points
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from mmcv.transforms.base import BaseTransform
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from PIL import Image
import io
from typing import Union
import numpy as np
from transformers.utils import logging
from matplotlib.colors import ListedColormap, BoundaryNorm
logger = logging.get_logger(__name__)
Number = Union[int, float]


import sys
sys.path.append('//home/ldc/Projects/RoboLLM/corruptions')
from lidar_corruption.Lidar_Method import LidarMethods

@TRANSFORMS.register_module()
class Processed_sensor(BaseTransform):
    def __init__(self,dataroot=None,vis_root=None,load_image=True,corruption_severity_dict=None):
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
        self.nusc_can_bus = NuScenesCanBus(dataroot=dataroot)
        self.vis_root=vis_root
        self.load_image=load_image


        ###create corruption method
        self.corruption_severity_dict=corruption_severity_dict
        if self.corruption_severity_dict is not None:
            self.corruption=LidarMethods(corruption_severity_dict=corruption_severity_dict)


        
    def transform(self, results):
        sample_data_token=results['token']
        prev_token=self.nusc.get('sample', sample_data_token)['prev']
        prev_sample=self.nusc.get('sample',prev_token)
        prev_lidar_token=prev_sample['data']['LIDAR_TOP']
        prev_sd_record = self.nusc.get('sample_data', prev_lidar_token)
        prev_cs_record = self.nusc.get('calibrated_sensor', prev_sd_record['calibrated_sensor_token'])
        prev_pose_record = self.nusc.get('ego_pose', prev_sd_record['ego_pose_token'])
        lidar_data_token=results['lidar_sweeps'][0]['sample_data_token']
        if self.load_image==True:
            lidar_image=self.render_lidar(sample_data_token=lidar_data_token,out_path=self.vis_root+'output_image.png',ground_threshold=-1.5,prev_cs_record=prev_cs_record,prev_pose_record=prev_pose_record)
            radar_image=self.render_radar(sample_data_token=sample_data_token,out_path=self.vis_root+'output_image1.png')
            results['input_images'].append(lidar_image)
            results['input_images'].append(radar_image)
        sample = self.nusc.get("sample", sample_data_token)
        scene_token = sample["scene_token"]
        ref_scene = self.nusc.get("scene", scene_token)
        pose_msgs = self.nusc_can_bus.get_messages(ref_scene['name'],'pose')
        steer_msgs = self.nusc_can_bus.get_messages(ref_scene['name'], 'steeranglefeedback')
        results['pose_msgs']=pose_msgs
        results['steer_msgs']=steer_msgs
        try:
            pose_uts = [msg['utime'] for msg in pose_msgs]
            steer_uts = [msg['utime'] for msg in steer_msgs]
            ref_utime = results['timestamp']*1000000
            pose_index = self.locate_message(pose_uts, ref_utime)
        except Exception as e:
            print(f"Error occurred in locate_message:\npose_uts: {pose_uts}\nref_utime: {ref_utime}")
            print(f"Exception: {e}")
        pose_data = pose_msgs[pose_index]
        steer_index = self.locate_message(steer_uts, ref_utime)
        steer_data = steer_msgs[steer_index]
        v0 = pose_data["vel"][0]  # [0] means longitudinal velocity  m/s
        acc_x=pose_data['accel'][0]
        acc_y=pose_data['accel'][1]
        # curvature (positive: turn left)
        steering = steer_data["value"]
        steering=-1 * steering / 2.588
        results['v0']=v0
        results['accel']=pose_data['accel']
        results['acc_x']=acc_x
        results['acc_y']=acc_y
        results['steering']=steering
        return results

    def render_lidar(self,
                        sample_data_token: str,
                        axes_limit: float = 50,
                        ax: Axes = None,
                        nsweeps: int = 1,
                        out_path: str = None,
                        use_flat_vehicle_coordinates: bool = True,
                        show_lidarseg: bool = False,
                        lidarseg_preds_bin_path: str = None,
                        verbose: bool = True,
                        show_panoptic: bool = False,
                        ground_threshold=None,
                        prev_cs_record=None,
                        prev_pose_record=None) -> None:
        """
        Render sample data onto axis.
        :param sample_data_token: Sample_data token.
        :param with_anns: Whether to draw box annotations.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param axes_limit: Axes limit for lidar and radar (measured in meters).
        :param ax: Axes onto which to render.
        :param nsweeps: Number of sweeps for lidar and radar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
            aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
            can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
            setting is more correct and rotates the plot by ~90 degrees.
        :param show_lidarseg: When set to True, the lidar data is colored with the segmentation labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
        :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
            or the list is empty, all classes will be displayed.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :param verbose: Whether to display the image after it is rendered.
        :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
            If show_lidarseg is True, show_panoptic will be set to False.
        """


        sd_record = self.nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']


        sample_rec = self.nusc.get('sample', sd_record['sample_token'])
        chan = sd_record['channel']
        ref_chan = 'LIDAR_TOP'
        ref_sd_token = sample_rec['data'][ref_chan]
        ref_sd_record = self.nusc.get('sample_data', ref_sd_token)

        if sensor_modality == 'lidar':
            if show_lidarseg or show_panoptic:
                gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
                assert hasattr(self.nusc, gt_from), f'Error: nuScenes-{gt_from} not installed!'

                # Ensure that lidar pointcloud is from a keyframe.
                assert sd_record['is_key_frame'], \
                    'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'

                assert nsweeps == 1, \
                    'Error: Only pointclouds which are keyframes have lidar segmentation labels; nsweeps should ' \
                    'be set to 1.'

                # Load a single lidar point cloud.
                pcl_path = osp.join(self.nusc.dataroot, ref_sd_record['filename'])
                pc = LidarPointCloud.from_file(pcl_path)
            else:
                # Get aggregated lidar point cloud in lidar frame.
                pc, times = LidarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan,
                                                                    nsweeps=nsweeps)




        # By default we render the sample_data top down in the sensor frame.
        # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
        # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
        if use_flat_vehicle_coordinates:
            # Retrieve transformation matrices for reference point cloud.
            cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
            pose_record = self.nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
            ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                            rotation=Quaternion(cs_record["rotation"]))
                ### corruption
            if self.corruption_severity_dict is not None:
               
                lidar={
                    "points":pc.points,
                    "lidar2ego":cs_record,
                    "ego2global":pose_record,
                    "prev_lidar2ego":prev_cs_record,
                    "prev_ego2global":prev_pose_record
                }            
                pc.points=self.corruption(lidar)

            # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
            ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            rotation_vehicle_flat_from_vehicle = np.dot(
                Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                Quaternion(pose_record['rotation']).inverse.rotation_matrix)
            vehicle_flat_from_vehicle = np.eye(4)
            vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
            viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
        else:
            viewpoint = np.eye(4)

        # Init axes.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(20, 20))

        # Render map if requested.
        # if underlay_map:
        #     assert use_flat_vehicle_coordinates, 'Error: underlay_map requires use_flat_vehicle_coordinates, as ' \
        #                                             'otherwise the location does not correspond to the map!'
        #     self.render_ego_centric_map(sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax)

        # Show point cloud.
        if ground_threshold!=None:
            pc.points = pc.points[:, pc.points[2, :] > ground_threshold]
        points = view_points(pc.points[:3, :], viewpoint, normalize=False)
        heights = pc.points[2, :]  # 获取所有点的Z坐标
        boundaries = [-np.inf, -2, -1, 0, 1, 2, 3, 4, np.inf]
        colors = ['gray','red', 'black', 'purple', 'blue', 'green', 'orange', 'yellow']
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(boundaries, cmap.N)
        # dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
        # colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
        
        point_scale = 0.2 if sensor_modality == 'lidar' else 3.0
        # ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)
        ax.scatter(points[0, :], points[1, :],c=heights, cmap=cmap, norm=norm, s=point_scale)

        # Show ego vehicle.
        ax.plot(0, 0, 'x', color='red')

        # Limit visible range.
        ax.set_xlim(-axes_limit, axes_limit)
        ax.set_ylim(-axes_limit, axes_limit)

        ax.axis('off')
        ax.set_title('{} {labels_type}'.format(
            sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
        ax.set_aspect('equal')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image = image.convert('RGB')
        # buf.close()
        
        
        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)

        if verbose:
            plt.show()
            
        plt.close()
        
        # image.save('/root/autodl-tmp/trush/output_lidar.png')
        # image=self.transform(image)
        
        return image
    
    def render_radar(self,sample_data_token,out_path: str = None):
        
        record = self.nusc.get('sample',sample_data_token)

        # 分离雷达数据
        radar_data = {}
        for channel, token in record['data'].items():
            sd_record = self.nusc.get('sample_data', token)
            if sd_record['sensor_modality'] == 'radar':
                radar_data[channel] = token

        # 如果存在雷达数据，则进行可视化
        if len(radar_data) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(20, 20))
            for i, (_, sd_token) in enumerate(radar_data.items()):
                self.nusc.render_sample_data(sd_token, with_anns=False, box_vis_level=False, ax=ax, nsweeps=1,verbose=False,underlay_map=False,axes_limit=50)
        ax.set_title('Fused RADARs')
        plt.tight_layout()
        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
            
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image = image.convert('RGB')
        # buf.close()
        plt.close()
                
        # image.save('/root/autodl-tmp/trush/output_radar.png')
        # image = self.transform(image)
        return image
    
    def locate_message(self,utimes, utime):
        i = np.searchsorted(utimes, utime)
        if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
            i -= 1
        return i