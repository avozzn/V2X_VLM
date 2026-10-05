"""Render reproducible evaluation tables and figures for frozen A0-A4 runs."""
import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .core import footprint,native_box


def read(path):return json.loads(Path(path).read_text())


def assoc(summary,split,variant,stage='accepted'):
    return next(x for x in summary['association'] if x['split']==split and x['variant']==variant
                and x['stage']==stage and x['class_group']=='all' and x['age_group']=='all')


def geom(summary,split,variant,kind,group='all'):
    return next((x for x in summary['geometry'] if x['split']==split and x['variant']==variant
                 and x['kind']==kind and x['group']==group),{'count':0})


def cases(trainval,out):
    objects={x['token']:x for x in map(json.loads,(trainval/'objects.jsonl').open()) if x['split']=='val'}
    decisions={}
    for row in map(json.loads,(trainval/'decisions.jsonl').open()):
        if row['split']=='val':decisions[row['token'],row['variant']]=row
    dataset=Path(read(trainval/'manifest.json')['args']['dataset_root'])
    chosen=[]
    # Reference identities select illustration subjects only, never algorithm decisions.
    for token,frame in objects.items():
        cor=read(dataset/'cooperative/label'/f'{token}.json')
        lookup={(str(x.get('veh_track_id')),str(x.get('inf_track_id'))):x for x in cor if x.get('from_side')=='coop'}
        for d in decisions[token,'A3']['decisions']:
            a,b=d['vehicle_index'],d['infra_index'];v,r=frame['vehicle'][a],frame['infra'][b]
            target=lookup.get((v['track_id'],r['track_id']))
            if target is None:continue
            if not d['accepted'] and not any(x[0]=='rejected_true_pair' for x in chosen):
                chosen.append(('rejected_true_pair',token,a,b,target,d['reasons']))
        for output in decisions[token,'A4']['outputs']:
            if output['size_policy']!='blend_stable':continue
            a,b=output['vehicle_index'],output['infra_index'];v,r=frame['vehicle'][a],frame['infra'][b]
            target=lookup.get((v['track_id'],r['track_id']))
            if target is not None and not any(x[0]=='size_blend' for x in chosen):chosen.append(('size_blend',token,a,b,target,[]))
        if len(chosen)>=2:break
    fig,axes=plt.subplots(1,max(1,len(chosen)),figsize=(12,5),squeeze=False)
    records=[]
    for ax,(kind,token,a,b,target,reasons) in zip(axes[0],chosen):
        frame=objects[token];v,r=frame['vehicle'][a],frame['infra'][b]
        boxes=[('cor',native_box(target),'black'),('ego raw',np.array(v['box']),'tab:blue'),
               ('RSU spatial',np.array(r['spatial_box']),'tab:orange'),('RSU compensated',np.array(r['box']),'tab:green')]
        if kind=='size_blend':
            fused=next(x for x in decisions[token,'A4']['outputs'] if x['vehicle_index']==a and x['infra_index']==b)
            boxes.append(('A4 fused',np.array(fused['box']),'tab:red'))
        for label,box,color in boxes:
            vertices=np.array(footprint(box).exterior.coords)
            ax.plot(vertices[:,0],vertices[:,1],label=label,color=color,linewidth=1.7)
        ax.set_aspect('equal');ax.set_xlabel('ego X (m)');ax.set_ylabel('ego Y (m)');ax.legend(fontsize=8)
        ax.set_title(f'{kind}: {token}\n'+','.join(reasons),fontsize=10)
        records.append({'case':kind,'token':token,'vehicle_index':a,'infra_index':b,'reasons':reasons,
                        'age_s':frame['age_s'],'ego_raw_lwh':v['raw_lwh'],'ego_stable_lwh':v['stable_lwh'],
                        'rsu_raw_lwh':r['raw_lwh'],'rsu_stable_lwh':r['stable_lwh'],'cor_lwh':native_box(target)[3:6].tolist()})
    fig.tight_layout();fig.savefig(out/'cases.png',dpi=180);plt.close(fig)
    (out/'cases.json').write_text(json.dumps(records,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trainval-dir',required=True);p.add_argument('--test-dir',required=True);p.add_argument('--output-dir',required=True)
    args=p.parse_args();root=Path(args.trainval_dir);testroot=Path(args.test_dir);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    trainval=read(root/'summary.json');test=read(testroot/'summary.json');selection=read(root/'frozen_selection.json')
    versions=['A0','A1','A2','A3','A4'];fig,axes=plt.subplots(2,2,figsize=(12,8))
    x=np.arange(5)
    axes[0,0].bar(x-.18,[assoc(trainval,'val',v,'candidates')['recall']*100 for v in versions],.36,label='Candidates')
    axes[0,0].bar(x+.18,[assoc(trainval,'val',v)['recall']*100 for v in versions],.36,label='Accepted')
    axes[0,0].set_xticks(x,versions);axes[0,0].set_ylabel('Reference recall (%)');axes[0,0].set_title('Validation association');axes[0,0].legend()
    axes[0,1].bar(['Spatial','Compensated'],[geom(trainval,'val',v,'rsu')['center_error_m']['mean'] for v in ['A0','A1']])
    axes[0,1].set_title('RSU vs cor: mean 3D center error');axes[0,1].set_ylabel('m')
    changes={(v['side'],v['mode']):v for v in trainval['temporal_size_change'] if v['split']=='val'}
    axes[1,0].bar(np.array([0,1])-.18,[changes[s,'raw']['mean_m'] for s in ['ego','rsu']],.36,label='Raw')
    axes[1,0].bar(np.array([0,1])+.18,[changes[s,'stable']['mean_m'] for s in ['ego','rsu']],.36,label='Stable')
    axes[1,0].set_xticks([0,1],['Ego','RSU']);axes[1,0].set_ylabel('Mean |delta lwh| (m)');axes[1,0].set_title('Temporal size variation');axes[1,0].legend()
    axes[1,1].bar(versions,[geom(trainval,'val',v,'fused_common')['size_mae_m']['mean'] for v in versions])
    axes[1,1].set_ylabel('Mean |lwh error| (m)');axes[1,1].set_title('Same matched cohort vs cor (ego inherited)')
    fig.tight_layout();fig.savefig(out/'overview.png',dpi=180);fig.savefig(out/'overview.pdf');plt.close(fig);cases(root,out)
    lines=['# 尺寸稳定、软关联与尺寸融合：cooperative验证报告','',
           '日期：2026-10-04。A0–A4已实现，完整train929/val157与冻结后的test480已完成；CPU实验，无VLM训练。','',
           '**结论：时间补偿有效，历史尺寸稳定降低波动；当前软关联/高可信参数未通过召回与精确率验收，尺寸平均也未改善cor一致性。冻结保留A1，不把整套方法宣称为有效。**','',
           '![实验概览](overview.png)','', '## 1. 算法与对照','',
           '双端仅用本端sequence/track过去1s内最多5次观测，车辆至少3次时以中位数稳定lwh；路端沿用因果平面恒速补偿。软代价为距离0.6、BEV IoU0.3、轴向0.1，行人为0.7/0.3；dummy0.7。高可信要求双向最佳、margin≥0.1、距离门限的一半、轴向≤30°、age≤1s且正age运动有效。匹配后保留ego中心/yaw。双端尺寸历史有效、相对MAD≤10%、各维差≤20%才平均，否则选ego尺寸。','',
           '| 组 | 尺寸稳定 | 补偿 | 关联 | 配对输出尺寸 |','|---|---|---|---|---|',
           '| A0 | 无 | 无 | 旧硬IoU0.1 | ego当前 |','| A1 | 无 | 有 | 同A0 | ego当前 |',
           '| A2 | 有 | 有 | 同A0 | ego稳定 |','| A3 | 有 | 有 | 软关联＋高可信 | ego稳定 |','| A4 | 有 | 有 | 与A3同配对 | 条件平均 |','',
           '候选距离固定车辆2m、小目标1m。A0/A1/A2 accepted沿用旧匹配；A3/A4 accepted通过新增高可信检查。因此同时报告candidate与accepted，明确新增验收造成的覆盖损失。完整宽候选，不做ROI/top-K截断。','',
           '## 2. 关联结果','', '| split | 组 | 候选召回 | 最终召回 | 可判定精确率 | 错配 | 未知配对 |','|---|---|---:|---:|---:|---:|---:|']
    for summary,split in [(trainval,'train'),(trainval,'val'),(test,'test')]:
        for version in versions:
            a=assoc(summary,split,version);c=assoc(summary,split,version,'candidates')
            lines.append(f"| {split} | {version} | {c['recall']*100:.2f}% | {a['recall']*100:.2f}% | {a['precision_judgable']*100:.2f}% | {a['known_false_positive']} | {a['unknown_predictions']} |")
    lines+=['','精确率仅覆盖有cor身份依据的预测，未知配对不能计为正确或错误；参考覆盖与单端来源统计在summary中。验证集场景bootstrap（1000次，seed42）时间补偿A1−A0召回增量95%区间：']
    for row in trainval['bootstrap']:
        if row['split']=='val' and row['metric']=='accepted_association' and row.get('before')=='A0':
            ci=row['delta_95percent_ci']['recall'];lines.append(f"{ci[0]*100:.2f}～{ci[1]*100:.2f}个百分点，共{row['scene_count']}个scene。")
    lines+=['','## 3. 尺寸稳定与融合','', '| split | 端 | 原始尺寸变化/m | 稳定后/m | 降幅 |','|---|---|---:|---:|---:|']
    for summary,split in [(trainval,'val'),(test,'test')]:
        rows={(x['side'],x['mode']):x for x in summary['temporal_size_change'] if x['split']==split}
        for side in ['ego','rsu']:
            a,b=rows[side,'raw']['mean_m'],rows[side,'stable']['mean_m']
            lines.append(f'| {split} | {side} | {a:.4f} | {b:.4f} | {(1-b/a)*100:.2f}% |')
    lines+=['','尺寸变化为本次选中原生观测同track相邻时间的lwh绝对变化均值，包含未启用稳定的其他类别；不是车辆实测尺寸精度。','',
            '| split | 同一正确配对队列 | A3尺寸MAE/m | A4尺寸MAE/m | A3 P95/m | A4 P95/m |','|---|---:|---:|---:|---:|---:|']
    for summary,split in [(trainval,'val'),(test,'test')]:
        a,b=[geom(summary,split,v,'fused_A3_A4') for v in ['A3','A4']]
        lines.append(f"| {split} | {a['count']} | {a['size_mae_m']['mean']:.5f} | {b['size_mae_m']['mean']:.5f} | {a['size_mae_m']['p95']:.5f} | {b['size_mae_m']['p95']:.5f} |")
    inherited=geom(trainval,'val','A3','fused_A3_A4','ego_inherited');allg=geom(trainval,'val','A3','fused_A3_A4')
    lines+=['',f"验证集此融合队列{allg['count']}个对象，{inherited['count']}个cor尺寸与ego原始尺寸完全一致（绝对容差1e-6m）。因此A0/A1固定共同队列尺寸MAE=0是标注继承关系，不是算法恢复真实尺寸。非ego继承队列为空时明确记未评估，不能据此推断融合的物理尺寸精度。",'',
            '![案例](cases.png)','', '案例身份仅用于评估展示；cor框未进入算法。case元数据见cases.json。完整几何统计含cor单端来源对象、类别、age、继承子集；融合误差比较固定正确配对队列。','',
            '## 4. 冻结政策、验证与产物','',f"冻结政策：**{selection['selected_variant']}**。选择仅使用validation；frozen_selection.json在test之前发布。test使用同一算法hash运行五组，不根据test改规则。冻结规则要求相对A1可判定精确率/召回均不下降、共同队列cor尺寸均值/P95不下降且至少一项严格改善。A2/A3/A4未满足，保留已有时间补偿基线。",'',
            '测试与完整输出检查详见integrity.json：包括未来/跨scene历史、尺寸异常、180°翻转、密集歧义、dummy、冲突回退、同配对融合、未知预测隔离，以及不读取cor标签的运行验证。','',
            f'train/val产物：`{root.resolve()}`。test产物：`{testroot.resolve()}`。objects.jsonl不含cor对象字段；decisions.jsonl为纯算法结果；evaluation.jsonl单独记录cor评估；summary.json含分组与scene-bootstrap；source_sha256.json和manifest.json记录来源及代码身份。','',
            '当前仅独立离线算法接口，未替换规划缓存、未训练T1/S1、未重评Detector AP。后续应先分析软关联拒绝的真实配对，改进高可信覆盖；物理尺寸准确性需要独立复核参考。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Report: {out}/report.md')


if __name__=='__main__':main()
