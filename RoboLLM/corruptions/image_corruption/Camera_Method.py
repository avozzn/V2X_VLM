import mmcv
import warnings
from copy import deepcopy




import warnings
from copy import deepcopy

import mmcv
import numpy as np


from .Camera_corruptions import ImageAddSun,ImageAddSnow,ImageAddFog,ImageAddRain,ImageAddGaussianNoise,ImageAddImpulseNoise,ImageAddUniformNoise,ImageAddDark
from .Camera_corruptions import ImageBBoxOperation
from .Camera_corruptions import ImageMotionBlurFrontBack, ImageMotionBlurLeftRight


import torch


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

def remove_close(points, radius=1.0):
    """Removes point too close within a certain radius from origin.

    Args:
        points (np.ndarray): Sweep points.
        radius (float): Radius below which points are removed.
            Defaults to 1.0.

    Returns:
        np.ndarray: Points after removing.
    """
    if isinstance(points, np.ndarray):
        points_numpy = points
    else:
        raise NotImplementedError
    x_filt = np.abs(points_numpy[:, 0]) < radius
    y_filt = np.abs(points_numpy[:, 1]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    return points[not_close]









class CameraMethods(object):
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
                        'sun_sim':2,
                    },
                 ):


        self.corruption_severity_dict = corruption_severity_dict

        if 'sun_sim' in self.corruption_severity_dict:
            # 注意这个只是加图像噪声，没有点云干扰
           
            severity = self.corruption_severity_dict['sun_sim']
            self.sun_sim = ImageAddSun(severity,seed=2022)
            
        if 'snow_sim' in self.corruption_severity_dict:
            # 注意这个只是加图像噪声，没有点云干扰
            severity = self.corruption_severity_dict['snow_sim']
            self.snow_sim = ImageAddSnow(severity, seed=2022)

        if 'fog_sim' in self.corruption_severity_dict:
            # 注意这个只是加图像噪声，没有点云干扰
            severity = self.corruption_severity_dict['fog_sim']
            self.fog_sim = ImageAddFog(severity, seed=2022)
            
        if 'rain_sim' in self.corruption_severity_dict:
            # 注意这个只是加图像噪声，没有点云干扰
            severity = self.corruption_severity_dict['rain_sim']
            self.rain_sim = ImageAddRain(severity, seed=2022)


        if 'motion_sim' in self.corruption_severity_dict:
            # 注意这个只是加图像噪声，没有点云干扰
            severity = self.corruption_severity_dict['motion_sim']
            self.motion_sim_leftright = ImageMotionBlurLeftRight(severity)
            self.motion_sim_frontback = ImageMotionBlurFrontBack(severity)

        if 'dark_sim' in self.corruption_severity_dict:
            # 注意这个只是加图像噪声，没有点云干扰
            severity = self.corruption_severity_dict['dark_sim']
            self.dark_sim = ImageAddDark(severity, seed=2022)

        if 'gauss_sim' in self.corruption_severity_dict:
            # 注意这个只是加图像噪声，没有点云干扰
            severity = self.corruption_severity_dict['gauss_sim']
            self.gauss_sim = ImageAddGaussianNoise(severity, seed=2022)
        
        if 'impulse_sim' in self.corruption_severity_dict:
            # 注意这个只是加图像噪声，没有点云干扰
            severity = self.corruption_severity_dict['impulse_sim']
            self.impulse_sim = ImageAddImpulseNoise(severity, seed=2022)

        if 'uniform_sim' in self.corruption_severity_dict:
            # 注意这个只是加图像噪声，没有点云干扰
            severity = self.corruption_severity_dict['uniform_sim']
            self.uniform_sim = ImageAddUniformNoise(severity)



        if 'object_motion_sim' in self.corruption_severity_dict:
            # for kitti and nus
            severity = self.corruption_severity_dict['object_motion_sim']
            self.object_motion_sim_frontback = ImageBBoxMotionBlurFrontBack(
                severity=severity,
                corrput_list=[0.02 * i for i in range(1, 6)],
            )
            self.object_motion_sim_leftright = ImageBBoxMotionBlurLeftRight(
                severity=severity,
                corrput_list=[0.02 * i for i in range(1, 6)],
            )









    def __call__(self, img_list):
        
        """img 一定为list,RGB格式!!!!!!!!!!"""


        """Call function to augment common fields in results.

        Args:
            results (dict): Result dict contains the data to augment.

        Returns:
            dict: The result dict contains the data that is augmented with
                different scales and flips.
        """

        img_result=[]

        if 'sun_sim' in self.corruption_severity_dict:
            for img in  img_list:
                cur_img = self.sun_sim(
                    image=img,
                    # watch_img=True,
                    # file_path='2.jpg'
                )
                img_result.append(cur_img)


        if 'snow_sim' in self.corruption_severity_dict:
            for img in img_list:
                cur_img = self.snow_sim(
                image = img
                )
                img_result.append(cur_img)
            


        if 'fog_sim' in self.corruption_severity_dict:
            for img in img_list:
                cur_img = self.fog_sim(
                image = img
                )
                img_result.append(cur_img)



        if 'rain_sim' in self.corruption_severity_dict:
            for img in img_list:
                cur_img = self.rain_sim(
                image = img
                )
                img_result.append(cur_img)


        if 'dark_sim' in self.corruption_severity_dict:
            for img in img_list:
                cur_img = self.dark_sim(
                image = img
                )
                img_result.append(cur_img)

        if 'motion_sim' in self.corruption_severity_dict:

            # 判断是否是 nus 数据集
            '''
            nuscenes:
            0    CAM_FRONT,
            1    CAM_FRONT_RIGHT,
            2    CAM_FRONT_LEFT,
            3    CAM_BACK,
            4    CAM_BACK_LEFT,
            5    CAM_BACK_RIGHT
            '''

            num = len(img_list)
            for i in range(num):
                if i % 3 == 0:
                    cur_img = self.motion_sim_frontback(image=img_list[i])
                else:
                    cur_img = self.motion_sim_leftright(image=img_list[i])
                img_result.append(cur_img)
        


        if 'gauss_sim' in self.corruption_severity_dict:
            for img in img_list:
                cur_img = self.gauss_sim(
                image = img
                )
                img_result.append(cur_img)

        if 'impulse_sim' in self.corruption_severity_dict:
            for img in img_list:
                cur_img = self.impulse_sim(
                image = img
                )
                img_result.append(cur_img)

        if 'uniform_sim' in self.corruption_severity_dict:
            for img in img_list:
                cur_img = self.uniform_sim(
                image = img
                )
                img_result.append(cur_img)

        
        


        


        return img_result

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip}, '
        repr_str += f'pts_scale_ratio={self.pts_scale_ratio}, '
        repr_str += f'flip_direction={self.flip_direction})'
        return repr_str

