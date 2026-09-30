#!/usr/bin/env bash
RESULTS_BASEDIR="./results_planning_only"
WORK_DIR="/home/ldc/Projects/RoboLLM/zoo/OmniDrive-main/work_dirs/"
TEST_PATHS=(
    "fog_sim_1_20250315_164220"
    "fog_sim_3_20250315_164220"
    "fog_sim_5_20250315_164220"
    "motion_sim_1_20250315_164220"
    "motion_sim_3_20250315_164220"
    "motion_sim_5_20250315_164220"
    "rain_sim_1_20250315_164220"
    "rain_sim_3_20250315_164220"
    "rain_sim_5_20250315_164220"
    "snow_sum_1_20250315_164220"
    "snow_sum_3_20250315_164220"
    "snow_sum_5_20250315_164220"
    "sun_sim_1_20250315_164220"
    "sun_sim_3_20250315_164220"
    "sun_sim_5_20250315_164220"
)

# ==== 批量测试主循环 ====
for ((i=0; i<${#TEST_PATHS[@]}; i++)); do
    SAVE_PATH="${RESULTS_BASEDIR}/${TEST_PATHS[i]}/"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    # 执行测试命令
    python ./evaluation/eval_planning.py \
        --pred_path="$SAVE_PATH" \
        2>&1 | tee "${WORK_DIR}logs/eval_$i_$TIMESTAMP.log"
done