
from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS
from PIL import Image
from nuscenes import NuScenes
import time
from mmcv.transforms.base import BaseTransform
@TRANSFORMS.register_module()
class LoadMultiViewImageFromFiles4Clip(LoadMultiViewImageFromFiles):
    def __init__(self,dataroot=None,**kwargs):
        super(LoadMultiViewImageFromFiles4Clip, self).__init__(**kwargs)
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
        answer={
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": results['answer_5']
                    }
                ]
            }
        results['qa_list'].append(answer)
        results['conversations']=[]
        for context in results['qa_list']:
            results['conversations'].append(context['content'][0]['text'])
        results['conversations'].append('Based on the above decisions and analysis, please plan the 3s future trajectory of the ego vehicle.')
        results['conversations'].append(results['answer_6'])
        # print("results after image")
        # print(results)
        
        # with open("/home/ldc/Projects/RoboLLM/zoo/MMDrive/results.txt", "a") as file:
        #     file.write("results after image\n")  # 先写入'results'标题
        #     file.write(str(results)) 
        #     file.write("\n=================\n")
        return results
    
@TRANSFORMS.register_module()
class LoadMultiViewImageFromFiles4Clip_qwen(BaseTransform):
    def __init__(self,dataroot=None,**kwargs):
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
            input_images.append(image_path)
        results['input_images']=input_images
        results['questions']=[]
        results['answers']=[]
        for context in results['qa_list']:
            if context['role']=='user':
                results['questions'].append(context['content'][0]['text'].replace("<image>", ""))
            else:
                results['answers'].append(context['content'][0]['text'])
        results['questions'].append('Based on the above decisions and analysis, please plan the 3s future trajectory of the ego vehicle.')
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