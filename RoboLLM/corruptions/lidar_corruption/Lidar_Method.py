import mmcv
import warnings
from copy import deepcopy

from scipy.spatial.transform import Rotation 

import warnings
from copy import deepcopy

import mmcv
import numpy as np


import torch
from .Lidar_corruptions import gaussian_noise,lidar_crosstalk_noise,density_dec_global,cutout_local,uniform_noise,impulse_noise,scene_glare_noise,rain_sim,fog_sim,snow_sim,spatial_alignment_noise,fulltrajectory_noise, temporal_alignment_noise



def format_list_float_06(l) -> None:
    for index, value in enumerate(l):
        l[index] = float('%.6f' % value)
    return l

def load_points(pts_filename):
    """Private function to load point clouds data.

    Args:
        pts_filename (str): Filename of point clouds data.

    Returns:
        np.ndarray: An array containing point clouds data.
    """
    points = np.fromfile(pts_filename, dtype=np.float32)
    return points

def convert_to_egopose(translation,rotation):
    t = np.array(translation)
    R = Rotation.from_quat(rotation).as_matrix()
    pose = np.eye(4); pose[:3, :3] = R; pose[:3, 3] = t
    return pose 
        
        
        
        
        
class LidarMethods(object):
    """Test-time augmentation with corruptions.

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple]: Images scales for resizing.
        pts_scale_ratio (float | list[float]): Points scale ratios for
            resizing.
        flip (bool, optional): Whether apply flip augmentation.
            Defaults to False.
        flip_direction (str | list[str], optional): Flip augmentation
            directions for images, options are "horizontal" and "vertical".
            If flip_direction is list, multiple flip augmentations will
            be applied. It has no effect when ``flip == False``.
            Defaults to "horizontal".
        pcd_horizontal_flip (bool, optional): Whether apply horizontal
            flip augmentation to point cloud. Defaults to True.
            Note that it works only when 'flip' is turned on.
        pcd_vertical_flip (bool, optional): Whether apply vertical flip
            augmentation to point cloud. Defaults to True.
            Note that it works only when 'flip' is turned on.
    """

    def __init__(self,
                 corruption_severity_dict=
                    {
                        'snow_sim':2,
                    },
                 ):
        
    
        self.corruption_severity_dict = corruption_severity_dict
        
    

    def __call__(self, results):
        #lidar corruptions

        if 'gaussian_noise' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['gaussian_noise']
           # aug_pl = pl[:,:3]
            points_aug = gaussian_noise(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl


        if 'lidar_crosstalk_noise' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['lidar_crosstalk_noise']
            aug_pl = pl[:,:3]
           # aug_pl = pl[:,:3]
            points_aug = lidar_crosstalk_noise(pl.numpy(), severity)
            
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

            

        if 'density_dec_global' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['density_dec_global']
            # aug_pl = pl[:,:3]
            points_aug = density_dec_global(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        if 'density_dec_local' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['density_dec_local']
            # aug_pl = pl[:,:3]
            points_aug = density_dec_local(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl
        
        if 'cutout_local' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['cutout_local']
            # aug_pl = pl[:,:3]
            points_aug = cutout_local(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl



        if 'uniform_noise' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['uniform_noise']
            # aug_pl = pl[:,:3]
            points_aug = uniform_noise(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        if 'upsampling' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['upsampling']
            # aug_pl = pl[:,:3]
            points_aug = upsampling(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        if 'background_noise' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['background_noise']
           # aug_pl = pl[:,:3]
            points_aug = background_noise(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl
        
        if 'impulse_noise' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['impulse_noise']
           # aug_pl = pl[:,:3]
            points_aug = impulse_noise(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        if 'layer_del' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['layer_del']
            # aug_pl = pl[:,:3]
            points_aug = layer_del(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        if 'rain_sim' in self.corruption_severity_dict:
            import numpy as np
            severity = self.corruption_severity_dict['rain_sim']
            points_aug = rain_sim(results['points'].T, severity)
            pl = torch.from_numpy(points_aug)
            pl=pl[:, :4].numpy().T

        if 'fog_sim' in self.corruption_severity_dict:
            import numpy as np
            severity = self.corruption_severity_dict['fog_sim']
            points_aug = fog_sim(results['points'].T, severity)
            pl = torch.from_numpy(points_aug)
            pl=pl[:, :4].numpy().T


        if 'snow_sim' in self.corruption_severity_dict:
            import numpy as np
            severity = self.corruption_severity_dict['snow_sim']
            points_aug = snow_sim(results['points'].T, severity)
            pl = torch.from_numpy(points_aug)
            pl=pl[:, :4].numpy().T


        if 'motion_sim' in self.corruption_severity_dict:
            import numpy as np
            severity = self.corruption_severity_dict['motion_sim']
            fir_ego_pose = format_list_float_06(results['ego2global']['translation']+results['ego2global']['rotation'])
            fir_sen_glo = format_list_float_06(results['lidar2ego']['translation']+results['lidar2ego']['rotation'])

            # sec_sweeps = results['sweeps_back'][0]
            sec_ego_pose = format_list_float_06(results['prev_ego2global']['translation']+results['prev_ego2global']['rotation'])
            sec_sen_glo = format_list_float_06(results['prev_lidar2ego']['translation']+results['prev_lidar2ego']['rotation'])

            pc_pose = np.array([fir_ego_pose,fir_sen_glo,sec_ego_pose,sec_sen_glo])

            points_aug = fulltrajectory_noise(results['points'].T, pc_pose,severity)
            pl =points_aug
            pl=pl[:, :4].numpy().T

        
        if 'sun_sim' in self.corruption_severity_dict:
            import numpy as np
            severity = self.corruption_severity_dict['sun_sim']
            points_aug = scene_glare_noise(results['points'].T, severity)
            pl = torch.from_numpy(points_aug)
            pl=pl[:, :4].numpy().T

        if 'dark_sim' in self.corruption_severity_dict:
            pl = results['points']
            



        return pl