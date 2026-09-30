import torch
from torch import Tensor
from tqdm import tqdm
import pickle
import json
from pathlib import Path
import os

# 假设 metric 在当前目录或子目录下，保持原引用不变
from .metric import PlanningMetric

current_dir = os.path.dirname(os.path.abspath(__file__))

def check_data_integrity(token, pred, gt, mask):
    """
    Data integrity check helper function.
    Returns: (is_valid, error_msg)
    """
    # 1. Check for NaN or Inf
    if not torch.isfinite(pred).all():
        return False, f"Prediction contains NaN or Inf."
    if not torch.isfinite(gt).all():
        return False, f"Ground Truth contains NaN or Inf."
    
    # 2. Check for extreme outliers (e.g. coordinates > 1000m)
    # Only check masked valid regions for GT
    valid_gt = gt[mask.bool()]
    if valid_gt.numel() > 0 and valid_gt.abs().max() > 1000:
        return False, f"Ground Truth contains extreme values > 1000m. Max: {valid_gt.abs().max().item():.2f}"

    return True, ""

def planning_evaluation(
    pred_trajs_dict,
    subset=None,
    only_vehicle=True,
    predict_steps=9,
    analysis_output_path=None,
):
    """
    Args:
        predict_steps (int): 预测步数。
                             12 -> 6.0s
                             9  -> 4.5s
                             6  -> 3.0s
    """
    # 根据输入的 predict_steps 设定时间步长
    ts = predict_steps 
    
    device = torch.device('cuda')

    # metric 初始化时传入当前的 ts
    metric_planning_val = PlanningMetric(ts).to(device)     
    metric_planning_val._enable_sync=False
    
    if only_vehicle:
        with open('gt/planing_gt_segmentation_test.pkl','rb') as f:
            gt_occ_map = pickle.load(f)
        for token in gt_occ_map.keys():
            if not isinstance(gt_occ_map[token], torch.Tensor):
                gt_occ_map[token] = torch.tensor(gt_occ_map[token])
            print("gt_occ_map[token]",token) 
    else:
        with open(current_dir+'/gt/planing_gt_segmentation_test.pkl','rb') as f:
            gt_occ_map_woP = pickle.load(f)
        for token in gt_occ_map_woP.keys():
            if not isinstance(gt_occ_map_woP[token], torch.Tensor):
                gt_occ_map_woP[token] = torch.tensor(gt_occ_map_woP[token])
        gt_occ_map = gt_occ_map_woP

    with open(current_dir+'/gt/gt_v2x_resampled_TM_test.pkl','rb') as f:
        gt_trajs_dict = pickle.load(f)

    with open(current_dir+'/gt/gt_v2x_resampled_Tmask_test.pkl','rb') as f:
        gt_trajs_mask_dict = pickle.load(f)

    print_first_token_details = True 
    if subset:
        test_tokens = subset
    
    analysis_records = []
    
    print(f"Start evaluating {len(pred_trajs_dict)} tokens...")
    for index, token in enumerate(tqdm(pred_trajs_dict.keys())):
        if token not in gt_occ_map:
                print(f"Warning: Token {token} found in predictions, but NOT in gt_occ_map. Skipping.")
                continue 
        
        gt_trajectory = torch.tensor(gt_trajs_dict[token])
        gt_trajectory = gt_trajectory.to(device)

        # 确保维度对其
        if gt_trajectory.shape[-1] == 3:
            gt_trajectory = gt_trajectory[:, :, :2]

        gt_traj_mask = torch.tensor(gt_trajs_mask_dict[token])
        gt_traj_mask = gt_traj_mask.to(device)

        output_trajs = torch.tensor(pred_trajs_dict[token])
        
        output_trajs = output_trajs.reshape(1, -1, 2)

        output_trajs = output_trajs.to(device)

        occupancy: Tensor = gt_occ_map[token]
        occupancy = occupancy.to(device)

        # Occupancy is expressed in the current vehicle lidar/ego frame:
        # x points forward, y points left, and the spatial layout is [row(y), col(x)].
        # Its temporal layout is [current, future_1, ..., future_N], whereas the
        # predicted trajectory starts at future_1, so frame 0 must be skipped.
        if occupancy.dim() != 4:
            raise ValueError(
                f"Occupancy for token {token} must have shape [B, T, H, W], "
                f"but got {tuple(occupancy.shape)}."
            )
        if occupancy.shape[1] < ts + 1:
            raise ValueError(
                f"Occupancy for token {token} contains {occupancy.shape[1]} frames, "
                f"but {ts + 1} are required (current frame + {ts} future frames)."
            )

        pred_t = output_trajs[:, :ts]
        gt_t = gt_trajectory[:, :ts]
        mask_t = gt_traj_mask[:, :ts]
        occ_t = occupancy[:, 1:ts + 1]
        # ======================================================
        # 🛡️ VALIDATION CODE ADDED HERE 🛡️
        # ======================================================
        is_valid, err_msg = check_data_integrity(token, pred_t, gt_t, mask_t)
        
        if not is_valid:
            print(f"\n[Data Integrity Alert] Token {token} Skipped!")
            print(f"Reason: {err_msg}")
            # Debug print
            if "Ground Truth" in err_msg:
                print(f"GT Traj (First 5): {gt_t[0, :5]}")
                print(f"Mask (First 5): {mask_t[0, :5]}")
            continue

        if gt_traj_mask.dim() < 2:
            print(f"\n[Debug Warning] Skipping token {token}: gt_traj_mask is a 0-dim scalar, not an array.")
            continue 

        if print_first_token_details:
            print(f"\n--- 🕵️ DEBUG: 第一个有效 Token ({token}) 数据检查 ---")
            print(f"[传递给 Metric 的 Shape]")
            # 这里打印切片后的形状
            print(f"  - pred_traj shape:     {output_trajs[:, :ts].shape}")
            print("pred_traj",output_trajs[:, :ts])
            print(f"  - gt_traj shape:       {gt_trajectory[:, :ts].shape}")
            print(f"  - occupancy shape:     {occ_t.shape}")
            print(f"  - gt_mask shape:       {gt_traj_mask[:, :ts].shape}")
            print_first_token_details = False 

        # 轨迹 future_1 对齐 occupancy future_1（occupancy 的索引 1）。
        metric_planning_val(pred_t, gt_t, occ_t, token, mask_t)
        
        ###分析数据
        if hasattr(metric_planning_val, 'last_L2_raw') and metric_planning_val.last_L2_raw is not None:
            # 计算该 Token 在当前时间步范围内的平均 L2
            curr_l2_tensor = metric_planning_val.last_L2_raw 
            avg_l2_val = curr_l2_tensor.mean().item()
            final_l2_val = curr_l2_tensor[0, -1].item() # 终点误差
            
            # 计算碰撞
            curr_col_tensor = metric_planning_val.last_col_raw
            is_collision = (curr_col_tensor.sum() > 0).item()
            
            analysis_records.append({
                'token': token,
                'avg_l2': avg_l2_val,
                'final_l2': final_l2_val,
                'collision': is_collision,
                'collision_steps': curr_col_tensor.numpy().tolist()
            })
    print("\n" + "="*60)
    print(" 📊 失败案例与整体分布分析 (Analysis Report) 📊")
    print("="*60)
    if len(analysis_records) > 0:
        import numpy as np
        
        # 1. L2 距离分布分析
        l2_values = [r['avg_l2'] for r in analysis_records]
        l2_np = np.array(l2_values)
        
        mean_l2 = np.mean(l2_np)
        median_l2 = np.median(l2_np)
        std_l2 = np.std(l2_np)
        max_l2 = np.max(l2_np)
        min_l2 = np.min(l2_np)
        
        print(f"\n[L2 误差分布]")
        print(f"  样本数:   {len(l2_np)}")
        print(f"  平均值 (Mean):   {mean_l2:.4f} m")
        print(f"  中位数 (Median): {median_l2:.4f} m")
        print(f"  标准差 (StdDev): {std_l2:.4f} m")
        print(f"  最大值 (Max):    {max_l2:.4f} m")
        print(f"  最小值 (Min):    {min_l2:.4f} m")
        
        # 判断偏差类型
        print(f"\n[诊断结论]")
        if mean_l2 > median_l2 * 1.3:
            print("  ⚠️ 发现: Mean 显著大于 Median。")
            print("  -> 说明存在【少数极端差的案例】(Outliers) 拉高了整体误差。")
            print("  -> 建议重点检查下方的 'Top Worst Cases'。")
        else:
            print("  ⚠️ 发现: Mean 与 Median 接近。")
            print("  -> 说明是【系统性偏差】(Systematic Bias)，大部分预测都不太准。")
            print("  -> 建议检查模型收敛情况、坐标系转换或 Loss 权重。")

        # 2. 输出最差的 10 个案例
        K = 10
        sorted_by_l2 = sorted(analysis_records, key=lambda x: x['avg_l2'], reverse=True)
        
        print(f"\n[L2 误差最大的 Top {K} 个 Token]")
        print(f"{'Token':<35} | {'Avg L2 (m)':<12} | {'Final L2 (m)':<12} | {'Collision?'}")
        print("-" * 80)
        for r in sorted_by_l2[:K]:
            print(f"{r['token']:<35} | {r['avg_l2']:.4f}         | {r['final_l2']:.4f}         | {r['collision']}")

        # 3. 碰撞分析
        col_cases = [r for r in analysis_records if r['collision']]
        print(f"\n[碰撞分析]")
        print(f"  碰撞总数: {len(col_cases)} / {len(analysis_records)} ({len(col_cases)/len(analysis_records)*100:.2f}%)")
        if len(col_cases) > 0:
            print(f"  碰撞案例示例 (Token):")
            for i, r in enumerate(col_cases[:5]):
                print(f"    {i+1}. {r['token']} (Steps: {r['collision_steps']})")
                
        # 保存到 JSON 以便后续画图
        analysis_output_path = analysis_output_path or 'evaluation_analysis_dump.json'
        with open(analysis_output_path, 'w') as f:
            json.dump(analysis_records, f, indent=4)
        print(f"\n✅ 详细分析数据已保存至 '{analysis_output_path}'")

    print("="*60 + "\n")
    results = {}

    scores = {
            'obj_col': metric_planning_val.obj_col / metric_planning_val.total,
            'obj_box_col': metric_planning_val.obj_box_col / metric_planning_val.total,
            'L2' : metric_planning_val.L2 / metric_planning_val.total
        }
    
    # 计算每一步的指标
    for i in range(ts):
        for key, value in scores.items():
            results['plan_'+key+'_{}s'.format((i+1)/2)]=value[:i+1].mean()
            
    # Print results in table
    print(f"gt collision: {metric_planning_val.gt_collision}")
    headers = ["Method", "L2 (m)", "Collision (%)"]
    
    evaluation_data = {}

    # =========================================================
    # Case 1: 6 Steps (3.0s)
    # =========================================================
    if predict_steps == 6:
        sub_headers = ["0.5s", "1s", "1.5s", "2s", "2.5s", "3s", "Avg."]
        
        # Avg 计算: 通常计算 1s, 2s, 3s 的平均值
        avg_l2 = (results['plan_L2_1.0s'] + results['plan_L2_2.0s'] + results['plan_L2_3.0s']) / 3.
        avg_col = (results['plan_obj_box_col_1.0s'] + results['plan_obj_box_col_2.0s'] + results['plan_obj_box_col_3.0s']) / 3. * 100

        method = ("DriveAgent", 
                "{:.2f}".format(results['plan_L2_0.5s']),
                "{:.2f}".format(results['plan_L2_1.0s']),
                "{:.2f}".format(results['plan_L2_1.5s']),
                "{:.2f}".format(results['plan_L2_2.0s']),
                "{:.2f}".format(results['plan_L2_2.5s']),
                "{:.2f}".format(results['plan_L2_3.0s']),
                "{:.2f}".format(avg_l2), # Avg L2
                
                "{:.2f}".format(results['plan_obj_box_col_0.5s']*100),
                "{:.2f}".format(results['plan_obj_box_col_1.0s']*100),
                "{:.2f}".format(results['plan_obj_box_col_1.5s']*100),
                "{:.2f}".format(results['plan_obj_box_col_2.0s']*100),
                "{:.2f}".format(results['plan_obj_box_col_2.5s']*100),
                "{:.2f}".format(results['plan_obj_box_col_3.0s']*100),
                "{:.2f}".format(avg_col) # Avg Col
        )
        
        # 打印格式：1 Method + 7 L2 + 7 Col = 15 列
        fmt_str = "{:<15} " + "{:<6} " * 14
        
        evaluation_data = {
            "L2_0.5": results['plan_L2_0.5s'].item(), "L2_1": results['plan_L2_1.0s'].item(),
            "L2_1.5": results['plan_L2_1.5s'].item(), "L2_2": results['plan_L2_2.0s'].item(),
            "L2_2.5": results['plan_L2_2.5s'].item(), "L2_3": results['plan_L2_3.0s'].item(),
            "Collision_0.5": results['plan_obj_box_col_0.5s'].item() * 100, "Collision_1": results['plan_obj_box_col_1.0s'].item() * 100,
            "Collision_1.5": results['plan_obj_box_col_1.5s'].item() * 100, "Collision_2": results['plan_obj_box_col_2.0s'].item() * 100,
            "Collision_2.5": results['plan_obj_box_col_2.5s'].item() * 100, "Collision_3": results['plan_obj_box_col_3.0s'].item() * 100,
        }

    # =========================================================
    # Case 2: 9 Steps (4.5s)
    # =========================================================
    elif predict_steps == 9:
        sub_headers = ["0.5s", "1s", "1.5s", "2s", "2.5s", "3s", "3.5s", "4s", "4.5s", "Avg."]
        
        # Avg 计算: 计算 1s, 2s, 3s, 4s 的平均值 (或者包含4.5，这里按整秒算)
        # 也可以改为 (1+2+3+4+4.5)/5，此处按整秒计算：
        avg_l2 = (results['plan_L2_1.0s'] + results['plan_L2_2.0s'] + results['plan_L2_3.0s'] + results['plan_L2_4.0s']) / 4.
        avg_col = (results['plan_obj_box_col_1.0s'] + results['plan_obj_box_col_2.0s'] + results['plan_obj_box_col_3.0s'] + results['plan_obj_box_col_4.0s']) / 4. * 100

        method = ("DriveAgent", 
                "{:.2f}".format(results['plan_L2_0.5s']), "{:.2f}".format(results['plan_L2_1.0s']),
                "{:.2f}".format(results['plan_L2_1.5s']), "{:.2f}".format(results['plan_L2_2.0s']),
                "{:.2f}".format(results['plan_L2_2.5s']), "{:.2f}".format(results['plan_L2_3.0s']),
                "{:.2f}".format(results['plan_L2_3.5s']), "{:.2f}".format(results['plan_L2_4.0s']),
                "{:.2f}".format(results['plan_L2_4.5s']),
                "{:.2f}".format(avg_l2),
                
                "{:.2f}".format(results['plan_obj_box_col_0.5s']*100), "{:.2f}".format(results['plan_obj_box_col_1.0s']*100),
                "{:.2f}".format(results['plan_obj_box_col_1.5s']*100), "{:.2f}".format(results['plan_obj_box_col_2.0s']*100),
                "{:.2f}".format(results['plan_obj_box_col_2.5s']*100), "{:.2f}".format(results['plan_obj_box_col_3.0s']*100),
                "{:.2f}".format(results['plan_obj_box_col_3.5s']*100), "{:.2f}".format(results['plan_obj_box_col_4.0s']*100),
                "{:.2f}".format(results['plan_obj_box_col_4.5s']*100),
                "{:.2f}".format(avg_col)
        )
        # 1 Method + 10 L2 + 10 Col = 21 cols
        fmt_str = "{:<15} " + "{:<6} " * 20

        evaluation_data = {
            "L2_0.5": results['plan_L2_0.5s'].item(), "L2_1": results['plan_L2_1.0s'].item(),
            "L2_1.5": results['plan_L2_1.5s'].item(), "L2_2": results['plan_L2_2.0s'].item(),
            "L2_2.5": results['plan_L2_2.5s'].item(), "L2_3": results['plan_L2_3.0s'].item(),
            "L2_3.5": results['plan_L2_3.5s'].item(), "L2_4": results['plan_L2_4.0s'].item(),
            "L2_4.5": results['plan_L2_4.5s'].item(),
            
            "Collision_0.5": results['plan_obj_box_col_0.5s'].item() * 100, "Collision_1": results['plan_obj_box_col_1.0s'].item() * 100,
            "Collision_1.5": results['plan_obj_box_col_1.5s'].item() * 100, "Collision_2": results['plan_obj_box_col_2.0s'].item() * 100,
            "Collision_2.5": results['plan_obj_box_col_2.5s'].item() * 100, "Collision_3": results['plan_obj_box_col_3.0s'].item() * 100,
            "Collision_3.5": results['plan_obj_box_col_3.5s'].item() * 100, "Collision_4": results['plan_obj_box_col_4.0s'].item() * 100,
            "Collision_4.5": results['plan_obj_box_col_4.5s'].item() * 100,
        }

    # =========================================================
    # Case 3: 12 Steps (6.0s) - 默认情况
    # =========================================================
    else: # predict_steps == 12
        sub_headers = ["0.5s", "1s", "1.5s", "2s", "2.5s", "3s", "3.5s", "4s", "4.5s", "5s","5.5s","6s", "Avg."]
        
        avg_l2 = (results['plan_L2_1.0s'] + results['plan_L2_2.0s'] + results['plan_L2_3.0s'] +
                  results['plan_L2_4.0s'] + results['plan_L2_5.0s'] + results['plan_L2_6.0s']) / 6.
        
        avg_col = (results['plan_obj_box_col_1.0s'] + results['plan_obj_box_col_2.0s'] + results['plan_obj_box_col_3.0s'] +
                   results['plan_obj_box_col_4.0s'] + results['plan_obj_box_col_5.0s']  + results['plan_obj_box_col_6.0s']) / 6. * 100

        method = ("DriveAgent", 
                "{:.2f}".format(results['plan_L2_0.5s']), "{:.2f}".format(results['plan_L2_1.0s']),
                "{:.2f}".format(results['plan_L2_1.5s']), "{:.2f}".format(results['plan_L2_2.0s']),
                "{:.2f}".format(results['plan_L2_2.5s']), "{:.2f}".format(results['plan_L2_3.0s']),
                "{:.2f}".format(results['plan_L2_3.5s']), "{:.2f}".format(results['plan_L2_4.0s']),
                "{:.2f}".format(results['plan_L2_4.5s']), "{:.2f}".format(results['plan_L2_5.0s']),
                "{:.2f}".format(results['plan_L2_5.5s']), "{:.2f}".format(results['plan_L2_6.0s']),
                "{:.2f}".format(avg_l2),
                
                "{:.2f}".format(results['plan_obj_box_col_0.5s']*100), "{:.2f}".format(results['plan_obj_box_col_1.0s']*100),
                "{:.2f}".format(results['plan_obj_box_col_1.5s']*100), "{:.2f}".format(results['plan_obj_box_col_2.0s']*100),
                "{:.2f}".format(results['plan_obj_box_col_2.5s']*100), "{:.2f}".format(results['plan_obj_box_col_3.0s']*100),
                "{:.2f}".format(results['plan_obj_box_col_3.5s']*100), "{:.2f}".format(results['plan_obj_box_col_4.0s']*100),
                "{:.2f}".format(results['plan_obj_box_col_4.5s']*100), "{:.2f}".format(results['plan_obj_box_col_5.0s']*100),
                "{:.2f}".format(results['plan_obj_box_col_5.5s']*100), "{:.2f}".format(results['plan_obj_box_col_6.0s']*100),
                "{:.2f}".format(avg_col)
        )
        # 1 Method + 13 L2 + 13 Col = 27 cols
        fmt_str = "{:<15} " + "{:<5} " * 26
        
        evaluation_data = {
            "L2_0.5": results['plan_L2_0.5s'].item(), "L2_1": results['plan_L2_1.0s'].item(),
            "L2_1.5": results['plan_L2_1.5s'].item(), "L2_2": results['plan_L2_2.0s'].item(),
            "L2_2.5": results['plan_L2_2.5s'].item(), "L2_3": results['plan_L2_3.0s'].item(),
            "L2_3.5": results['plan_L2_3.5s'].item(), "L2_4": results['plan_L2_4.0s'].item(),
            "L2_4.5": results['plan_L2_4.5s'].item(), "L2_5": results['plan_L2_5.0s'].item(),
            "L2_5.5": results['plan_L2_5.5s'].item(), "L2_6": results['plan_L2_6.0s'].item(),

            "Collision_0.5": results['plan_obj_box_col_0.5s'].item() * 100, "Collision_1": results['plan_obj_box_col_1.0s'].item() * 100,
            "Collision_1.5": results['plan_obj_box_col_1.5s'].item() * 100, "Collision_2": results['plan_obj_box_col_2.0s'].item() * 100,
            "Collision_2.5": results['plan_obj_box_col_2.5s'].item() * 100, "Collision_3": results['plan_obj_box_col_3.0s'].item() * 100,
            "Collision_3.5": results['plan_obj_box_col_3.5s'].item() * 100, "Collision_4": results['plan_obj_box_col_4.0s'].item() * 100,
            "Collision_4.5": results['plan_obj_box_col_4.5s'].item() * 100, "Collision_5": results['plan_obj_box_col_5.0s'].item() * 100,
            "Collision_5.5": results['plan_obj_box_col_5.5s'].item() * 100, "Collision_6": results['plan_obj_box_col_6.0s'].item() * 100,
        }

    print("\n")
    print("VAD evaluation:")
    # Headers 只有3个主标题
    print("{:<15} {:<85} {:<85}".format(*headers)) 
    
    # 根据步骤数生成的 sub_headers (时间点)
    # fmt_str 在上面已经定义好了，用来对齐列
    print(fmt_str.format("", *sub_headers, *sub_headers))
    print(fmt_str.format(*method))
    
    return evaluation_data

def load_pred_trajs_from_file(path):
    with open(path, "rb") as f:
        pred_trajs_dict = pickle.load(f)
    return pred_trajs_dict

if __name__ == "__main__":
    pred_trajs_dict = load_pred_trajs_from_file("result.pkl")
    # 示例调用：可以修改 predict_steps=6 或 9 或 12
    planning_evaluation(pred_trajs_dict, subset=None, only_vehicle=False, predict_steps=12)
