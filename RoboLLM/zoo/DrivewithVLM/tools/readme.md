RobodriveVLM模型ckpt路径：/home/ldc/Projects/RoboLLM/zoo/DrivewithVLM/checkpoints/MMdrive/checkpoint-8950
DriveVLM模型ckpt路径：/home/ldc/Projects/RoboLLM/zoo/DrivewithVLM/checkpoints/drivevlm/checkpoint-13358
Openemma模型ckpt路径：/home/ldc/Projects/RoboLLM/zoo/DrivewithVLM/checkpoints/Openemma_train/checkpoint-27666
llava_interleave模型ckpt路径：/home/ldc/Projects/RoboLLM/zoo/DrivewithVLM/checkpoints/LLM/llava-next-interleave


1.RobodriveVLM训练脚本
bash tools/robollm/dist_train.sh projects/configs/Robodrivevlm/Planning.py n
2.RobodriveVLM测试脚本
#无corruption测试
bash tools/robollm/dist_test_ddp.sh projects/configs/Robodrivevlm/Planning.py n (无corruption测试)

#图像corruption测试
bash tools/robollm/dist_test_ddp_image.sh projects/configs/Robodrivevlm/Planning.py n

#prompt corruption测试
bash tools/robollm/dist_test_ddp_prompt.sh projects/configs/Robodrivevlm/Planning.py n

3.DriveVLM测试脚本
#无corruption测试
bash tools/drivevlm/dist_test_drivevlm.sh projects/configs/drivevlm/Planning.py n

#图像corruption测试
bash tools/drivevlm/dist_test_drivevlm_image.sh projects/configs/drivevlm/Planning.py n

#prompt corruption测试
bash tools/drivevlm/dist_test_drivevlm_prompt.sh projects/configs/drivevlm/Planning.py n

4.Openemma测试脚本
#无corruption测试
bash tools/openemma/dist_test_emma.sh projects/configs/openemma/openemma.py 1 （建议单卡测试，分布式测试易出现同步问题导致进程被killed）

#图像corruption测试
bash tools/openemma/dist_test_emma_image.sh projects/configs/openemma/openemma.py 1 

#prompt corruption测试
bash tools/openemma/dist_test_emma_prompt.sh projects/configs/openemma/openemma.py 1 

#openemma批量评测脚本
bash tools/openemma/dist_test_emma_prompt_val.sh