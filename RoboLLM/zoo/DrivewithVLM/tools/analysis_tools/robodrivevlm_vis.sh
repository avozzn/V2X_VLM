export PYTHONPATH=/home/ldc/Projects/RoboLLM/zoo/MMDrive:$PYTHONPATH

python ./tools/analysis_tools/visualize/run.py \
    --predroot /home/ldc/Projects/RoboLLM/zoo/MMDrive/vis/pkl/result_20250216_222844.pkl\
    --out_folder /home/ldc/Projects/RoboLLM/zoo/MMDrive/tools/VIS_PLANNING \
    --demo_video /home/ldc/Projects/RoboLLM/zoo/MMDrive/tools/VIS_PLANNING \
    --project_to_cam True