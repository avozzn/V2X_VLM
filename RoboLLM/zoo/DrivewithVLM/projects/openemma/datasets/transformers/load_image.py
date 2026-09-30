
from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS
from PIL import Image
from nuscenes import NuScenes
import os
import numpy as np
from .emmautils import EstimateCurvatureFromTrajectory, IntegrateCurvatureForPoints
from math import atan2
@TRANSFORMS.register_module()
class LoadMultiViewImageFromFiles4Clip(LoadMultiViewImageFromFiles):
    def __init__(self,dataroot=None,**kwargs):
        super(LoadMultiViewImageFromFiles4Clip, self).__init__(**kwargs)
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
        self.dataroot=dataroot

    def transform(self, results) -> dict:
        token=results['scene_token']
        results['sample_token']=[]
        scene = self.nusc.get('scene', token)
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        name = scene['name']
        description = scene['description']
        front_camera_images = []
        ego_poses = []
        camera_params = []
        curr_sample_token = first_sample_token
        while True:
            results['sample_token'].append(curr_sample_token)
            sample = self.nusc.get('sample', curr_sample_token)

            # Get the front camera image of the sample.
            cam_front_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
            front_camera_images.append(Image.open(os.path.join(self.nusc.dataroot, cam_front_data['filename'])).convert("RGB"))
            pose = self.nusc.get('ego_pose', cam_front_data['ego_pose_token'])
            ego_poses.append(pose)
            camera_params.append(self.nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token']))

            # Advance the pointer.
            if curr_sample_token == last_sample_token:
                break
            curr_sample_token = sample['next']
        scene_length = len(front_camera_images)
        ego_poses_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]
        ego_poses_world = np.array(ego_poses_world)
        ego_velocities = np.zeros_like(ego_poses_world)
        ego_velocities[1:] = ego_poses_world[1:] - ego_poses_world[:-1]
        ego_velocities[0] = ego_velocities[1]
        ego_curvatures = EstimateCurvatureFromTrajectory(ego_poses_world)
        ego_velocities_norm = np.linalg.norm(ego_velocities, axis=1)
        estimated_points = IntegrateCurvatureForPoints(ego_curvatures, ego_velocities_norm, ego_poses_world[0],
                                                       atan2(ego_velocities[0][1], ego_velocities[0][0]), scene_length)
        ego_traj_world = [ego_poses[t]['translation'][:3] for t in range(len(ego_poses))]
        results['front_camera_images']=front_camera_images
        results['ego_velocities']=ego_velocities
        results['ego_curvatures']=ego_curvatures
        results['ego_traj_world']=ego_traj_world
        results['scene_length']=scene_length
        results['name']=name
        results['description']=description
        return results


@TRANSFORMS.register_module()
class LoadMultiViewImageFromFiles4Clip_test(LoadMultiViewImageFromFiles):
    def __init__(self,dataroot=None,**kwargs):
        super(LoadMultiViewImageFromFiles4Clip_test, self).__init__(**kwargs)
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
        self.dataroot=dataroot
        
    def transform(self, results) -> dict:
        token=results['token']
        current_sample = self.nusc.get('sample', token)
        prev_token=current_sample['prev']
        prev_sample=self.nusc.get('sample',prev_token)
        prev_prev_token=prev_sample['prev']
        prev_prev_sample=self.nusc.get('sample',prev_prev_token)
        current_image=self.nusc.get('sample_data', current_sample['data']['CAM_FRONT'])['filename']
        prev_image=self.nusc.get('sample_data', prev_sample['data']['CAM_FRONT'])['filename']
        prev_prev_image=self.nusc.get('sample_data', prev_prev_sample['data']['CAM_FRONT'])['filename']
        input_images=[]
        image_sources=[]
        image_sources.append(self.dataroot+'/'+current_image)
        image_sources.append(self.dataroot+'/'+prev_image)
        image_sources.append(self.dataroot+'/'+prev_prev_image)
        for image_path in image_sources:
            input_images.append(
                Image.open(image_path).convert("RGB")
                )
        results['input_images']=input_images
        results['questions']=[]
        results['answers']=[]
        for context in results['qa_list']:
            if context['role']=='user':
                results['questions'].append(context['content'][0]['text'])
            else:
                results['answers'].append(context['content'][0]['text'])
        results['questions'].append('Based on the above decisions and analysis, please plan the 3s future trajectory of the ego vehicle.')
        return results