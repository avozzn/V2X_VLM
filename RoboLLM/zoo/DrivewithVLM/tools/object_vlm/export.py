"""Export shared A1 ROI objects, without reading cooperative object labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from collections import Counter
import numpy as np
from tools.association.run import NativeDataset, causal_objects, json_default
from tools.association.stable_fusion import process
from .data import SCHEMA,validate


def select_objects(outputs,age,roi=50.,ego_k=32,rsu_k=16):
    groups=[[],[]];counts=Counter()
    for x in outputs:
        box=np.asarray(x['box']);distance=float(np.linalg.norm(box[:2]))
        if distance>roi:counts['outside_roi']+=1;continue
        group=0 if x['source_bits'][0] else 1
        groups[group].append((distance,tuple(box),x['vehicle_index'] if x['vehicle_index'] is not None else -1,
                              x['infra_index'] if x['infra_index'] is not None else -1,x))
    chosen=[]
    for group,limit in zip(groups,[ego_k,rsu_k]):
        group.sort(key=lambda x:x[:4]);counts['budget_truncated']+=max(0,len(group)-limit)
        chosen += [x[-1] for x in group[:limit]]
    geometry=[];source=[];quality=[];audit=[]
    for x in chosen:
        box=np.asarray(x['box']);cls=x['class'];g=[*box[:6],float(np.sin(box[6])),float(np.cos(box[6])),0.,0.,*([0.]*3)]
        g[10+cls]=1.;geometry.append(g);source.append(x['source_bits'])
        has_rsu=bool(x['source_bits'][1]);quality.append([float(age) if has_rsu and age is not None else 0.,float(not has_rsu or age is not None),0.])
        audit.append({k:x[k] for k in ('vehicle_index','infra_index','source_bits')})
    return {'geometry':geometry,'source_bits':source,'quality':quality},audit,counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',default='/home/zzn/V2X_VLM/UniV2X/datasets/V2X-Seq-SPD-New')
    p.add_argument('--planning-dir',default='data/Planning/continuous_v3_native_gt')
    p.add_argument('--output-dir',required=True);args=p.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    dataset=NativeDataset(args.dataset_root);pairs=json.loads((dataset.root/'cooperative/data_info.json').read_text())
    by_frame={str(x['vehicle_frame']):x for x in pairs};counts={};planning_hash={}
    for split in ('train','val','test'):
        path=Path(args.planning_dir)/f'{split}.json';samples=json.loads(path.read_text());planning_hash[split]=hashlib.sha256(path.read_bytes()).hexdigest();rows=[];audits=[];stat=Counter()
        for s in samples:
            token=str(s['token']);pair=by_frame[token];v,r,selected,age=causal_objects(dataset,pair)
            assert int(s['timestamp_us'])==int(dataset.frames['vehicle-side'][token]['pointcloud_timestamp'])
            for obj in v+r:
                obj.update(raw_lwh=obj['box'][3:6].copy(),stable_lwh=obj['box'][3:6].copy(),size_stable_valid=False,size_relative_mad=np.zeros(3))
            result=process(v,r,age,'A1');objects,audit,c=select_objects(result['outputs'],age);stat.update(c)
            t=s['planning_targets'];state={'dimensions_lw_m':s['ego_state']['dimensions_lw_m'],
                'velocity_xy_mps':t['ego_velocity_xy_mps'],'velocity_valid':t['ego_velocity_valid'],
                'acceleration_xy_mps2':t['ego_acceleration_xy_mps2'],'history_xy':t['history_xy'],
                'history_mask':t['history_mask'],'history_times_s':t['history_times_s']}
            row={'schema':SCHEMA,'token':token,'scene_token':s['scene_token'],'timestamp_us':int(s['timestamp_us']),
                 'observation':{'ego_image':s['ego_image'],'state':state,'objects':objects},
                 'target':{k:t[k] for k in ['future_xy','future_mask','future_times_s']}}
            validate(row);rows.append(row)
            audits.append({'token':token,'selected_infra_frame':selected,'age_s':age,'has_rsu_message':selected is not None,
                           'objects':audit,'vehicle_observations':v,'infra_observations':r})
            stat['samples']+=1;stat['objects']+=len(objects['geometry']);stat['dual_objects']+=sum(x==[1,1] for x in objects['source_bits'])
            stat['rsu_objects']+=sum(x==[0,1] for x in objects['source_bits']);stat['empty_objects']+=not bool(objects['geometry'])
        for name,payload in [(f'{split}.json',rows),(f'{split}_audit.json',audits)]:
            tmp=out/(name+'.tmp');tmp.write_text(json.dumps(payload,default=json_default,allow_nan=False));os.replace(tmp,out/name)
        counts[split]=dict(stat);print(split,dict(stat),flush=True)
    paths={dataset.root/'cooperative/data_info.json'}
    for side in dataset.frames:paths.add(dataset.root/side/'data_info.json')
    for side,frame in dataset.labels:paths.add(dataset.root/side/dataset.frames[side][frame]['label_lidar_std_path'])
    for side,frame in dataset.poses:
        for key in (['calib_lidar_to_novatel_path','calib_novatel_to_world_path'] if side=='vehicle-side' else ['calib_virtuallidar_to_world_path']):paths.add(dataset.root/side/dataset.frames[side][frame][key])
    manifest={'schema':SCHEMA,'args':vars(args),'counts':counts,'policy':'A1 raw dimensions + causal motion compensation',
              'roi_m':50,'ego_budget':32,'extra_rsu_budget':16,'velocity_input':'disabled_for_both_sides',
              'planning_sha256':planning_hash,'native_source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},
              'split_sha256':{sp:hashlib.sha256((out/f'{sp}.json').read_bytes()).hexdigest() for sp in counts},
              'code_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('tools/association/core.py'),Path('tools/association/run.py'),Path('tools/association/stable_fusion.py')]},
              'limits':['GT track Oracle history','no extra transmission latency assumption','no cooperative object labels read','test exported but not evaluated']}
    tmp=out/'manifest.json.tmp';tmp.write_text(json.dumps(manifest,indent=2));os.replace(tmp,out/'manifest.json')

if __name__=='__main__':main()
