import json
import os

def json_to_txt(json_file):
    # 获取文件名（不带扩展名）
    base_name = os.path.splitext(json_file)[0]
    txt_file = base_name + ".txt"

    try:
        # 读取 JSON 文件
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        import pdb
        pdb.set_trace()
        # 写入 TXT 文件
        with open(txt_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"成功将 JSON 内容保存到 {txt_file}")
    except Exception as e:
        print(f"读取或写入文件时出错: {e}")

# 示例用法
json_to_txt("/home/ldc/Projects/RoboLLM/zoo/OmniDrive-main/test/MMmask_eva_lane_det_vlm_cor/Thu_Mar_20_18_58_42_2025/pts_bbox/results_nusc.json")
