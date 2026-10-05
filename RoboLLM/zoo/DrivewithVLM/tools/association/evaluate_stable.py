"""A0-A4 cooperative-only evaluation, isolated from algorithm inputs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .core import native_box, pair_features
from .run import read, reference_audit, json_default, resolve_reference
from .stable_fusion import HistoryDataset, prepare_objects, process

VARIANTS = ['A0','A1','A2','A3','A4']


def age_group(age):
    return 'no_message' if age is None else '0-.1s' if age<.1 else '.1-.5s' if age<.5 else '.5-1s' if age<1 else '1s+'


def association_counts(predicted, truth):
    truth=set(map(tuple,truth))
    known_v,known_r = {a for a,b in truth},{b for a,b in truth}
    predicted=set(map(tuple,predicted))
    judgable={p for p in predicted if p[0] in known_v or p[1] in known_r}
    return Counter(reference_pairs=len(truth),true_positive=len(predicted&truth),
                   known_false_positive=len(judgable-truth),unknown_predictions=len(predicted-judgable),
                   predictions=len(predicted))


def metrics(counts):
    tp,fp=counts['true_positive'],counts['known_false_positive']
    return {**counts,'recall':tp/counts['reference_pairs'] if counts['reference_pairs'] else None,
            'precision_judgable':tp/(tp+fp) if tp+fp else None,
            'unknown_fraction':counts['unknown_predictions']/counts['predictions'] if counts['predictions'] else None}


def geometry(prediction, target):
    obj={'box':prediction,'class':0};gt={'box':target,'class':0}
    bev,volume,_,_=pair_features([obj],[gt])
    absolute=np.abs(prediction[3:6]-target[3:6])
    return {'center_error_m':float(np.linalg.norm(prediction[:3]-target[:3])),
            'axis_error_rad':float(np.arccos(np.clip(abs(np.cos(prediction[6]-target[6])),0,1))),
            'lwh_abs_error_m':absolute.tolist(),'lwh_relative_error':(absolute/target[3:6]).tolist(),
            'size_mae_m':float(absolute.mean()),'bev_iou':float(bev[0,0]),'iou3d':float(volume[0,0])}


def summarize(values):
    if not values:return {'count':0}
    keys=['center_error_m','axis_error_rad','size_mae_m','bev_iou','iou3d']
    return {'count':len(values),**{key:{'mean':float(np.mean([v[key] for v in values])),
                                  'median':float(np.median([v[key] for v in values])),
                                  'p95':float(np.percentile([v[key] for v in values],95))} for key in keys},
            'mean_lwh_abs_error_m':np.mean([v['lwh_abs_error_m'] for v in values],axis=0).tolist(),
            'mean_lwh_relative_error':np.mean([v['lwh_relative_error'] for v in values],axis=0).tolist()}


def bootstrap_counts(scene_counts, before, after):
    scenes=sorted({k[0] for k in scene_counts})
    if len(scenes)<2:return {'scene_count':len(scenes),'ci':None}
    rng=np.random.default_rng(42);deltas=defaultdict(list)
    for _ in range(1000):
        sampled=rng.integers(0,len(scenes),len(scenes));totals=[]
        for variant in [before,after]:
            total=Counter()
            for i in sampled:total.update(scene_counts[scenes[i],variant])
            totals.append(metrics(total))
        for key in ['recall','precision_judgable']:
            if all(x[key] is not None for x in totals):deltas[key].append(totals[1][key]-totals[0][key])
    return {'scene_count':len(scenes),'after':after,'before':before,
            'delta_95percent_ci':{k:np.percentile(v,[2.5,97.5]).tolist() for k,v in deltas.items()}}


def bootstrap_geometry(rows, before, after):
    # Pair observations AND scene resampling are shared across both policies.
    by_identity=defaultdict(dict)
    for row in rows:by_identity[(row['scene'],row['token'],row['vehicle_index'],row['infra_index'])][row['variant']]=row
    grouped=defaultdict(list)
    for key,values in by_identity.items():
        if before in values and after in values:grouped[key[0]].append((values[before]['size_mae_m'],values[after]['size_mae_m']))
    scenes=sorted(grouped)
    if len(scenes)<2:return {'scene_count':len(scenes),'ci':None}
    arrays={s:np.array(grouped[s]) for s in scenes};rng=np.random.default_rng(42);deltas=[]
    for _ in range(1000):
        sample=np.concatenate([arrays[scenes[i]] for i in rng.integers(0,len(scenes),len(scenes))])
        deltas.append(float(np.mean(sample[:,1]-sample[:,0])))
    return {'scene_count':len(scenes),'paired_observations':sum(len(x) for x in arrays.values()),
            'after':after,'before':before,'mean_size_error_delta_95percent_ci':np.percentile(deltas,[2.5,97.5]).tolist()}


def evaluate(args):
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    dataset=HistoryDataset(args.dataset_root)
    pairs=read(dataset.root/'cooperative/data_info.json');by_vehicle={str(x['vehicle_frame']):x for x in pairs}
    references,audit=({}, {}) if args.no_evaluation else reference_audit(dataset,pairs,out)
    counts=defaultdict(Counter);scene_counts=defaultdict(Counter);geometry_rows=[];groups=defaultdict(list)
    temporal=defaultdict(list);previous={};objects_count=Counter();source_paths=set();planning_hashes={}
    if not args.no_evaluation:
        source_paths.update(dataset.root/'cooperative/label'/f"{p['vehicle_frame']}.json" for p in pairs)
    def emit(stream,value):stream.write(json.dumps(value,default=json_default,allow_nan=False)+'\n')
    with (out/'objects.jsonl').open('w') as objects_stream, (out/'decisions.jsonl').open('w') as decision_stream, (out/'evaluation.jsonl').open('w') as evaluation_stream:
        for split in args.splits:
            sample_path=Path(args.planning_dir)/f'{split}.json';planning_hashes[split]=hashlib.sha256(sample_path.read_bytes()).hexdigest()
            samples=read(sample_path)
            if args.limit:samples=samples[:args.limit]
            for number,sample in enumerate(samples):
                token=str(sample['token']);pair=by_vehicle[token]
                v,r,selected,age=prepare_objects(dataset,pair)
                assert int(sample['timestamp_us'])==int(dataset.frames['vehicle-side'][token]['pointcloud_timestamp'])
                scene=str(pair['vehicle_sequence'])
                objects_count[split+':samples']+=1;objects_count[split+':no_message']+=selected is None
                for side,rows in [('ego',v),('rsu',r)]:
                    objects_count.update(split+':'+side+':size_'+x['size_status'] for x in rows)
                    for obj in rows:
                        now=int(dataset.frames['vehicle-side' if side=='ego' else 'infrastructure-side'][obj['source_frame']]['pointcloud_timestamp'])
                        assert all(x['timestamp_us']<=now for x in obj['size_history'])
                        if obj['size_status']=='invalid_current_track':continue
                        identity=(split,side,obj['source_sequence'],obj['track_id'])
                        last=previous.get(identity)
                        if last and last[0]<now:
                            for mode,key in [('raw','raw_lwh'),('stable','stable_lwh')]:
                                temporal[split,side,mode].append(float(np.abs(obj[key]-last[1][key]).mean()))
                        previous[identity]=(now,obj)
                emit(objects_stream,{'split':split,'token':token,'scene':scene,'timestamp_us':sample['timestamp_us'],
                                     'selected_infra_frame':selected,'age_s':age,'vehicle':v,'infra':r})
                variants={variant:process(v,r,age,variant) for variant in VARIANTS}
                assert variants['A3']['accepted']==variants['A4']['accepted']
                for variant,result in variants.items():
                    emit(decision_stream,{'split':split,'token':token,'variant':variant,
                                         'candidates':result['candidates'],'accepted':result['accepted'],
                                         'decisions':result['decisions'],'outputs':result['outputs']})
                if args.no_evaluation:continue
                vc=Counter(x['track_id'] for x in v);rc=Counter(x['track_id'] for x in r)
                vi={x['track_id']:i for i,x in enumerate(v) if vc[x['track_id']]==1}
                ri={x['track_id']:i for i,x in enumerate(r) if rc[x['track_id']]==1}
                cor_path=dataset.root/'cooperative/label'/f'{token}.json';source_paths.add(cor_path)
                cor=read(cor_path)
                valid_links={(str(x['veh_track_id']),str(x['inf_track_id'])) for x in references[token]}
                targets={}
                for obj in cor:
                    identity=(str(obj.get('veh_track_id')),str(obj.get('inf_track_id')))
                    if identity in valid_links and identity[0] in vi and identity[1] in ri:
                        targets[vi[identity[0]],ri[identity[1]]]=obj
                truth=set(targets)
                for variant,result in variants.items():
                    for stage in ['candidates','accepted']:
                        pred=set(result[stage]);sc=association_counts(pred,truth)
                        counts[split,variant,stage,'all','all'].update(sc)
                        if stage=='accepted':scene_counts[split,scene,variant].update(sc)
                        for cls in range(3):
                            counts[split,variant,stage,str(cls),'all'].update(association_counts(
                                [p for p in pred if v[p[0]]['class']==cls],[p for p in truth if v[p[0]]['class']==cls]))
                        counts[split,variant,stage,'all',age_group(age)].update(sc)
                        for flag in [True,False]:
                            selected_truth=[p for p in truth if bool(v[p[0]]['size_stable_valid'] and r[p[1]]['size_stable_valid'])==flag]
                            selected_pred=[p for p in pred if bool(v[p[0]]['size_stable_valid'] and r[p[1]]['size_stable_valid'])==flag]
                            counts[split,variant,stage,'all','dual_stable_'+str(flag)].update(association_counts(selected_pred,selected_truth))
                    correct_outputs={(x['vehicle_index'],x['infra_index']):x for x in result['outputs'] if x['source_bits']==[1,1]}
                    for indexes in truth:
                        a,b=indexes;target=native_box(targets[indexes]);inherited=bool(np.allclose(target[3:6],v[a]['raw_lwh'],rtol=0,atol=1e-6))
                        for side,prediction in [('ego',result['vehicle'][a]['box']),('rsu',result['infra'][b]['box'])]:
                            g=geometry(prediction,target)
                            row={'split':split,'scene':scene,'token':token,'variant':variant,'kind':side,
                                 'vehicle_index':a,'infra_index':b,'class':v[a]['class'],'age_group':age_group(age),
                                 'cor_from_side':targets[indexes]['from_side'],'cor_size_equals_ego':inherited,**g}
                            emit(evaluation_stream,row);groups[split,variant,side,'all'].append(g)
                            groups[split,variant,side,'ego_inherited' if inherited else 'not_ego_inherited'].append(g)
                            groups[split,variant,side,'class_'+str(v[a]['class'])].append(g)
                            groups[split,variant,side,'age_'+age_group(age)].append(g)
                        if indexes in correct_outputs:
                            g=geometry(correct_outputs[indexes]['box'],target)
                            row={'split':split,'scene':scene,'token':token,'variant':variant,'kind':'fused',
                                 'vehicle_index':a,'infra_index':b,'class':v[a]['class'],'age_group':age_group(age),
                                 'cor_size_equals_ego':inherited,'size_conflict':correct_outputs[indexes]['size_conflict'],**g}
                            geometry_rows.append(row);emit(evaluation_stream,row)
                    # Audit single-source cor objects separately, not as negative association pairs.
                    for side,rows,prefix,frame in [('ego',result['vehicle'],'veh',token),('rsu',result['infra'],'inf',str(pair['infrastructure_frame']))]:
                        valid_targets={}
                        native=dataset.objects('vehicle-side' if side=='ego' else 'infrastructure-side',frame)
                        for obj in cor:
                            if obj.get('from_side')!=prefix:continue
                            matches=[x for x in native if str(x.get('track_id'))==str(obj.get(prefix+'_track_id')) and x.get('token')==obj.get(prefix+'_token')]
                            if len(matches)==1 and str(obj.get(prefix+'_frame_id'))==frame:valid_targets[str(obj[prefix+'_track_id'])]=obj
                        row_counts=Counter(x['track_id'] for x in rows)
                        for obj in rows:
                            if obj['track_id'] in valid_targets and row_counts[obj['track_id']]==1:
                                g=geometry(obj['box'],native_box(valid_targets[obj['track_id']]))
                                groups[split,variant,side,'single_source_cor'].append(g)
                                emit(evaluation_stream,{'split':split,'token':token,'variant':variant,'kind':side+'_single_source','track_id':obj['track_id'],**g})
                if number%100==0:print(f'{split}: {number}/{len(samples)}',flush=True)
    # Compare fused geometry on identical correctly matched object cohorts.
    cohort=defaultdict(dict)
    for row in geometry_rows:cohort[row['split'],row['token'],row['vehicle_index'],row['infra_index']][row['variant']]=row
    common=[]
    for identity,rows in cohort.items():
        if all(x in rows for x in VARIANTS):
            for variant,row in rows.items():
                groups[identity[0],variant,'fused_common','all'].append(row)
                groups[identity[0],variant,'fused_common','ego_inherited' if row['cor_size_equals_ego'] else 'not_ego_inherited'].append(row)
                common.append(row)
        if 'A3' in rows and 'A4' in rows:
            for variant in ['A3','A4']:
                groups[identity[0],variant,'fused_A3_A4','all'].append(rows[variant])
                groups[identity[0],variant,'fused_A3_A4','ego_inherited' if rows[variant]['cor_size_equals_ego'] else 'not_ego_inherited'].append(rows[variant])
                groups[identity[0],variant,'fused_A3_A4','size_conflict_'+str(rows[variant]['size_conflict'])].append(rows[variant])
    association=[dict(split=k[0],variant=k[1],stage=k[2],class_group=k[3],age_group=k[4],**metrics(c)) for k,c in counts.items()]
    geom_summary=[dict(split=k[0],variant=k[1],kind=k[2],group=k[3],**summarize(values)) for k,values in groups.items()]
    ci=[]
    if not args.no_evaluation:
        for split in args.splits:
            scenes={(s,variant):c for (sp,s,variant),c in scene_counts.items() if sp==split}
            for before,after in zip(VARIANTS[:-1],VARIANTS[1:]):
                ci.append({'split':split,'metric':'accepted_association',**bootstrap_counts(scenes,before,after)})
                ci.append({'split':split,'metric':'pairwise_common_fused_size_error',**bootstrap_geometry([x for x in geometry_rows if x['split']==split],before,after)})
    summary={'association':association,'geometry':geom_summary,'temporal_size_change':[
        {'split':k[0],'side':k[1],'mode':k[2],'count':len(x),'mean_m':float(np.mean(x)),'p95_m':float(np.percentile(x,95))} for k,x in temporal.items()],
        'bootstrap':ci,'object_counts':dict(objects_count),'reference_audit':audit}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    source_paths.update(dataset.root/side/'data_info.json' for side in dataset.frames)
    source_paths.add(dataset.root/'cooperative/data_info.json')
    for side,frame in dataset.labels:source_paths.add(dataset.root/side/dataset.frames[side][frame]['label_lidar_std_path'])
    for side,frame in dataset.poses:
        keys=['calib_lidar_to_novatel_path','calib_novatel_to_world_path'] if side=='vehicle-side' else ['calib_virtuallidar_to_world_path']
        source_paths.update(dataset.root/side/dataset.frames[side][frame][k] for k in keys)
    hashes={str(p.relative_to(dataset.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_paths)}
    (out/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
    (out/'manifest.json').write_text(json.dumps({'schema':'causal-stable-fusion-v1','args':vars(args),'planning_sha256':planning_hashes,
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        'limitations':['Oracle native track IDs for within-side history only','cor geometry is a benchmark proxy, not independent measured dimensions',
                       'unknown associations excluded from precision denominator','upright yaw-only 3D IoU','no additional transport latency',
                       'common fused cohort may be smaller than entire reference population','single-source cor lacks proof of other-side invisibility']},indent=2))
    print(f'Complete: {out}',flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',default='/home/zzn/V2X_VLM/UniV2X/datasets/V2X-Seq-SPD-New')
    parser.add_argument('--planning-dir',default='data/Planning/continuous_v3_native_gt')
    parser.add_argument('--splits',nargs='+',choices=['train','val','test'],default=['train','val'])
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--no-evaluation',action='store_true')
    parser.add_argument('--frozen-selection',help='Required for test; records pre-test policy and parameter identity')
    args=parser.parse_args()
    if 'test' in args.splits and not args.frozen_selection:parser.error('Freeze validation policy before test')
    if args.frozen_selection:
        selection=read(args.frozen_selection)
        digest=hashlib.sha256(Path(__file__).with_name('stable_fusion.py').read_bytes()).hexdigest()
        if selection['algorithm_sha256']!=digest:parser.error('Algorithm differs from frozen selection')
    evaluate(args)


if __name__=='__main__':main()
