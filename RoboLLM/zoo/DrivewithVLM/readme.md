


```
## train

bash tools/robollm/dist_train.sh projects/configs/Robodrivevlm/MMdrive_v2x.py 4


bash tools/dist_train.sh projects/configs/MMdrive/Planning_no_sensor.py 6 


###detection train
bash tools/dist_train.sh projects/configs/MMdrive/Detection.py 6



bash tools/dist_train.sh projects/configs/MMdrive/COT.py 1
```




```
test指令

bash tools/robollm/dist_test.sh projects/configs/Robodrivevlm/MMdrive_v2x.py 4

<!-- lyx zzn -->
bash tools/dist_test_ddp.sh projects/configs/MMdrive/Planning_prompt_corruption.py 1


###lidar corruption
bash tools/dist_test_ddp.sh projects/configs/MMdrive/Planning_lidar.py 1
    

###detection vis
bash tools/dist_test_vis.sh projects/configs/MMdrive/Detection.py 1



### image corruption
bash tools/dist_test_ddp.sh projects/configs/MMdrive/Planning_image_corruption.py.py 1



bash tools/dist_test_vis.sh projects/configs/MMdrive/COT.py 1
bash tools/dist_test_ddp.sh projects/configs/MMdrive/COT.py 2


bash tools/dist_test_ddp_deepspeed.sh projects/configs/MMdrive/COT.py /home/ldc/Projects/RoboLLM/zoo/MMDrive/checkpoints/llava-interleave-qwen-7b_lora-True_qlora-False/checkpoint-26000 1



tta 
bash tools/dist_test_tta.sh projects/configs/MMdrive/COT.py 1 

 ```


openemma
bash tools/dist_train.sh projects/configs/Openemma/Planning.py 1
bash tools/dist_test_ddp_openemma.sh projects/configs/Openemma/Planning.py 1
bash tools/dist_test_ddp_openemma.sh projects/configs/Openemma/Planning_image_cor.py 1

drivevlm
bash tools/dist_test_drivevlm.sh projects/configs/drivevlm/Planning.py 2
bash tools/dist_test_drivevlm_cor.sh projects/configs/drivevlm/Planning_image_cor.py 3
bash tools/dist_test_drivevlm_prompt.sh projects/configs/drivevlm/Planning_prompt.py 8
bash tools/dist_test_drivevlm_prompt.sh projects/configs/drivevlm/Planning_prompt.py 2




corruption test No_lidar:

bash tools/dist_test_ddp.sh projects/configs/MMdrive/No_Sensor.py 1

 --model_path='/home/ldc/Projects/RoboLLM/zoo/MMDrive/checkpoints/final/checkpoint-26848' 
 修改为/home/ldc/Projects/RoboLLM/zoo/MMDrive/checkpoints/without_sensor/checkpoint-8950
 bash tools/dist_test_ddp_image.sh projects/configs/MMdrive/No_Sensor_image_cor.py 1
 bash tools/dist_test_ddp_prompt.sh projects/configs/MMdrive/No_Sensor_prompt.py 1
 bash tools/dist_test_ddp_image.sh projects/configs/MMdrive/No_Sensor_image_cor.py 4



emma
 bash tools/dist_test_emma.sh projects/configs/open_emma_new/openemma.py 1
 bash tools/dist_test_emma_image.sh projects/configs/open_emma_new/openemma_cor.py 3

 prompt_cor
 bash tools/dist_test_ddp_prompt.sh projects/configs/MMdrive/COT_prompt_cor.py 1
 bash tools/dist_test_tta_prompt.sh projects/configs/MMdrive_TTA/ALL_prompt.py 4
 bash tools/dist_test_emma_image.sh projects/configs/open_emma_new/openemma_cor.py 3



sensor cor
bash tools/dist_test_ddp_sensor_corruption.sh projects/configs/MMdrive/MMdrive_sensor_cor.py 4
 bash tools/dist_test_emma_cor.sh projects/configs/openemma_new/openemma_sensor.py 2





 backbone
  bash tools/dist_test_qwen.sh projects/configs/drivevlm/Planning_qwen.py 1





  bash tools/dist_test_tta_prompt.sh projects/configs/MMdrive/MMdrive.py 2





bash tools/dist_test_ddp_all.sh projects/configs/MMdrive/MMdrive.py 4