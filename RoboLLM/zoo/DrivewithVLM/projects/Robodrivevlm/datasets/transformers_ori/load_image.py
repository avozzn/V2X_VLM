
from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS
from PIL import Image

@TRANSFORMS.register_module()
class LoadMultiViewImageFromFiles4Clip(LoadMultiViewImageFromFiles):
    def __init__(self,dataroot=None,**kwargs):
        super(LoadMultiViewImageFromFiles4Clip, self).__init__(**kwargs)
        self.dataroot=dataroot

    def transform(self, results) -> dict:
        image_sources=[]
        image_sources.append(self.dataroot+'/samples/CAM_FRONT/'+results['images']['CAM_FRONT']['img_path'])
        image_sources.append(self.dataroot+'/samples/CAM_FRONT_RIGHT/'+results['images']['CAM_FRONT_RIGHT']['img_path'])
        image_sources.append(self.dataroot+'/samples/CAM_FRONT_LEFT/'+results['images']['CAM_FRONT_LEFT']['img_path'])
        # image_sources.append(self.dataroot+'/samples/CAM_BACK/'+results['images']['CAM_BACK']['img_path'])
        # image_sources.append(self.dataroot+'/samples/CAM_BACK_LEFT/'+results['images']['CAM_BACK_LEFT']['img_path'])
        # image_sources.append(self.dataroot+'/samples/CAM_BACK_RIGHT/'+results['images']['CAM_BACK_RIGHT']['img_path'])
        for image_path in image_sources:
            results['input_images'].append(
                Image.open(image_path).convert("RGB")
                )

        # print("results after image")
        # print(results)
        
        # with open("/home/ldc/Projects/RoboLLM/zoo/MMDrive/results.txt", "a") as file:
        #     file.write("results after image\n")  # 先写入'results'标题
        #     file.write(str(results)) 
        #     file.write("\n=================\n")
        return results