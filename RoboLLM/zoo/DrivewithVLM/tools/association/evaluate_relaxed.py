"""Train/val-only V2 association ablation from audited native observations."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from .relaxed_fusion import process_relaxed
from .run import json_default
from .evaluate_stable import association_counts, metrics, bootstrap_counts, age_group

VARIANTS=('A1','L1','L2')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',default='data/Planning/stable_fusion_v1_trainval_20261004')
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();source=Path(args.source_dir);out=Path(args.output_dir)
    # Validate cache identity before creating published results.
    manifest=json.loads((source/'manifest.json').read_text())
    if set(manifest['args']['splits']) != {'train','val'}:
        parser.error('Only audited train/val cache is allowed; no test tuning')
    root=Path(manifest['args']['dataset_root'])
    hashes=json.loads((source/'source_sha256.json').read_text())
    for name,digest in hashes.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Native source changed: '+name)
    truth=defaultdict(set)
    for line in (source/'evaluation.jsonl').open():
        x=json.loads(line)
        if x['variant']=='A1' and x['kind']=='ego':
            truth[x['split'],x['token']].add((x['vehicle_index'],x['infra_index']))
    baseline={}
    for line in (source/'decisions.jsonl').open():
        x=json.loads(line)
        if x['variant']=='A1':baseline[x['split'],x['token']]=x['accepted']
    out.mkdir(parents=True,exist_ok=False)
    counts=defaultdict(Counter);scenes=defaultdict(Counter);rejections=defaultdict(Counter);samples=Counter()
    with (out/'decisions.jsonl').open('w') as stream:
        for number,line in enumerate((source/'objects.jsonl').open()):
            x=json.loads(line);split=x['split'];token=x['token'];samples[split]+=1
            if split not in ('train','val'):raise ValueError('Unexpected split')
            v,r=x['vehicle'],x['infra']
            for obj in v+r:
                for key in ('box','raw_box','spatial_box','stable_lwh','raw_lwh','size_relative_mad'):
                    if key in obj:obj[key]=np.array(obj[key],dtype=float)
            target=truth[split,token];known_v={a for a,b in target};known_r={b for a,b in target}
            for variant in VARIANTS:
                result=process_relaxed(v,r,x['age_s'],variant)
                if variant=='A1':assert [list(p) for p in result['accepted']]==baseline[split,token]
                for stage in ('candidates','accepted'):
                    pred=result[stage];c=association_counts(pred,target)
                    counts[split,variant,stage,'all','all'].update(c)
                    counts[split,variant,stage,'all',age_group(x['age_s'])].update(c)
                    for cls in range(3):
                        counts[split,variant,stage,str(cls),'all'].update(association_counts(
                            [p for p in pred if v[p[0]]['class']==cls],[p for p in target if v[p[0]]['class']==cls]))
                    if stage=='accepted':scenes[split,x['scene'],variant].update(c)
                for d in result['decisions']:
                    p=(d['vehicle_index'],d['infra_index'])
                    tag='true' if p in target else 'false' if p[0] in known_v or p[1] in known_r else 'unknown'
                    # Identity classification stays in summary, never in algorithm decisions.
                    for reason in d['reasons']:rejections[split,variant,tag][reason]+=1
                for obj in result['outputs']:
                    assert np.isfinite(obj['box']).all() and (obj['box'][3:6]>0).all()
                stream.write(json.dumps({'split':split,'token':token,**result},default=json_default,allow_nan=False)+'\n')
            if number%100==0:print(f'{number}: {dict(samples)}',flush=True)
    association=[dict(split=k[0],variant=k[1],stage=k[2],class_group=k[3],age_group=k[4],**metrics(c)) for k,c in counts.items()]
    ci=[]
    for split in ('train','val'):
        subset={(s,v):c for (sp,s,v),c in scenes.items() if sp==split}
        for before,after in [('A1','L1'),('L1','L2'),('A1','L2')]:
            ci.append({'split':split,**bootstrap_counts(subset,before,after)})
    overall={x['variant']:x for x in association if x['split']=='val' and x['stage']=='accepted' and x['class_group']=='all' and x['age_group']=='all'}
    base=overall['A1'];selected='A1'
    for variant in ('L1','L2'):
        row=overall[variant]
        if row['precision_judgable']>=base['precision_judgable'] and row['recall']>base['recall']:selected=variant
    summary={'samples':dict(samples),'association':association,'bootstrap':ci,
             'rejections':[{'split':k[0],'variant':k[1],'reference_status':k[2],'counts':dict(c)} for k,c in rejections.items()],
             'selected_variant':selected,'selection_split':'val','test_used':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    inputs=['objects.jsonl','evaluation.jsonl','decisions.jsonl','manifest.json','source_sha256.json']
    code=['relaxed_fusion.py','evaluate_relaxed.py','stable_fusion.py','core.py','run.py','evaluate_stable.py']
    (out/'manifest.json').write_text(json.dumps({'schema':'relaxed-association-v2','args':vars(args),
        'input_sha256':{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in inputs},
        'code_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in code},
        'native_sources_verified':len(hashes),'raw_dimensions_for_all_variants':True,
        'unknown_motion_short_age_s':.1,'unknown_motion_max_cost':.45,'unknown_motion_min_bev_iou':.1,
        'limits':['GT track Oracle history','unknown predictions excluded from precision','no test','no VLM','engineering defaults not calibrated confidence']},indent=2))
    lines=['# 放宽关联规则：train/val对照','',
           'A1为旧补偿基线；L1使用软匹配，取消验收距离减半，运动无效仍拒绝；L2进一步允许age≤0.1s且cost≤0.45、BEV IoU≥0.1的未知运动配对。三组均使用原始尺寸，匹配输出保留ego原始框；不做尺寸稳定/平均。', '',
           '| split | 组 | 候选召回 | 最终召回 | 可判定精确率 | 错配 | 未知 |',
           '|---|---|---:|---:|---:|---:|---:|']
    for row in association:
        if row['stage']=='accepted' and row['class_group']=='all' and row['age_group']=='all':
            cand=next(z for z in association if z['split']==row['split'] and z['variant']==row['variant'] and z['stage']=='candidates' and z['class_group']=='all' and z['age_group']=='all')
            lines.append(f"| {row['split']} | {row['variant']} | {cand['recall']:.2%} | {row['recall']:.2%} | {row['precision_judgable']:.2%} | {row['known_false_positive']} | {row['unknown_predictions']} |")
    lines += ['',f'验证集规则选择：**{selected}**。只有精确率不低于A1且召回严格提升才替换，否则保留A1。没有使用test。', '',
              '旧A3使用稳定尺寸，本轮L1/L2使用原始尺寸，因此与旧63.87%的差异不应全归因于验收门限；本轮L1→L2才是单独消息年龄政策消融。', '',
              '完整类别/消息年龄、拒绝原因与场景bootstrap在summary.json；输入与代码hash在manifest.json。算法不读取cor身份，评估器从已审计cor队列读取参考；不将未知配对算正确。', '',
              '已验证A1逐帧配对与原报告完全一致，所有输出有限且尺寸为正。宽候选接口尚未替换VLM缓存或训练。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print('Complete:',out, 'selected:',selected,flush=True)

if __name__=='__main__':main()
