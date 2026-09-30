#!/usr/bin/env bash

WORK_DIR="/home/ldc/Projects/RoboLLM/zoo/MMDrive/work_dirs/openemma_image/"
CONFIG="/home/ldc/Projects/RoboLLM/zoo/MMDrive/tools/test_emma.py"
TEST_PATHS=(
    "emma_fog_1_20250331_061943.json"
    "emma_fog_3_20250330_051150.json"
    "emma_fog_5_20250329_040715.json"
    "emma_motion_1_20250328_025337.json"
    "emma_motion_3_20250327_023150.json"
    "emma_motion_5_20250326_010744.json"
    "emma_rain_1_20250328_032624.json"
    "emma_rain_3_20250327_023116.json"
    "emma_rain_5_20250326_010337.json"
    "emma_dark_3_20250327_023319.json"
    "emma_dark_5_20250326_010014.json"
    "emma_dark_sim_1_20250328_024444.json"
    "emma_snow_1_20250331_095833.json"
    "emma_snow_3_20250330_063015.json"
    "emma_snow_5_20250329_040646.json"
    "emma_sun_1.json"
    "emma_sun_3_20250330_052515.json"
    "emma_sun_5_20250329_040707.json"
)
# 设置JSON文件所在目录路径
file_path="/home/ldc/Projects/RoboLLM/zoo/MMDrive/vis/emma_eval/"

# 遍历目录下所有.json文件（支持子目录）
for ((i=0; i<${#TEST_PATHS[@]}; i++)); do
    echo "正在测试文件: ${TEST_PATHS[i]}"
    grep -q "new_suffix =" $CONFIG || { echo "Error: 'new_suffix' not found in $CONFIG"; exit 1; }
    sed -i "s/new_suffix = \".*\"/new_suffix = \"${TEST_PATHS[i]}\"/" $CONFIG
    python test_emma.py  \
    2>&1 | tee ${WORK_DIR}/test._${TEST_PATHS[i]}
done

echo "所有测试完成"