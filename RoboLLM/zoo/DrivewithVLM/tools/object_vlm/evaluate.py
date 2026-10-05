"""Freeze validation-selected checkpoints, evaluate test, summarize T1/S1.

No training or test-driven checkpoint selection. GPU prediction is only run by
an explicit user-launched script; freeze/preflight is CPU-only.
"""
import argparse
import hashlib
import json
import os
import math
import sys
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from .data import read_rows,load_processor,prompt
from .model import load_model,INTERFACE
from .run import base_identity,evaluate,metrics
from .distributed import evaluate_distributed


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def input_length_audit(rows,processor,config,training_limit,max_new_tokens=512):
    """Only observation prompts; never inspect test trajectory answers."""
    vision=config['vision_config']
    image_tokens=(vision['image_size']//vision['patch_size'])**2
    if config['vision_feature_select_strategy']=='default':image_tokens-=1
    lengths={}
    for label,representation in [('T1','text'),('S1','soft')]:
        values=[]
        for row in rows:
            messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt(row['observation'],representation)}]}]
            text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            ids=processor.tokenizer(text,add_special_tokens=False)['input_ids']
            if ids.count(config['image_token_index'])!=1:raise ValueError('Unexpected image placeholder count')
            n=len(ids)+image_tokens-1
            if representation=='soft':
                if ids.count(processor.tokenizer.convert_tokens_to_ids('<v2x_objects>'))!=1:raise ValueError('Unexpected object placeholder count')
                n+=len(row['observation']['objects']['geometry'])-1
            values.append(n)
        lengths[label]={'max_input_tokens':max(values),'over_training_limit':sum(n>training_limit for n in values)}
    maximum=max(r['max_input_tokens'] for r in lengths.values())
    limit=max(training_limit,math.ceil(maximum/1024)*1024)
    supported=config['text_config']['max_position_embeddings']
    if limit+max_new_tokens>supported:raise ValueError('Input plus generation budget exceeds model context')
    return {'experiments':lengths,'training_max_length':training_limit,'evaluation_max_length':limit,
            'model_context_limit':supported,'max_new_tokens':max_new_tokens,'uses_test_targets':False}


def selected_checkpoint(training):
    training=Path(training)
    complete=json.loads((training/'complete.json').read_text())
    best=json.loads((training/'best_checkpoint.json').read_text())
    history=[json.loads(x) for x in (training/'history.jsonl').read_text().splitlines()]
    if complete['epochs']!=6 or complete['steps']!=1398 or len(history)!=6:raise ValueError('Formal six-epoch training is incomplete')
    selected=max(history,key=lambda x:(x['coverage'],-x['avg_l2_1_2_3s'] if x['avg_l2_1_2_3s'] is not None else -float('inf')))
    if selected['epoch']!=best['epoch']:raise ValueError('Best checkpoint differs from validation selection')
    checkpoint=(training/f"epoch_{best['epoch']}").resolve()
    if Path(best['path']).resolve()!=checkpoint:raise ValueError('Checkpoint path mismatch')
    config=json.loads((checkpoint/'interface.json').read_text())
    if config['interface']!=INTERFACE:raise ValueError('Unsupported checkpoint interface')
    files=[checkpoint/'interface.json',checkpoint/'adapter/adapter_config.json',checkpoint/'adapter/adapter_model.safetensors']
    if config['representation']=='soft':files.append(checkpoint/'objects.pt')
    return {'checkpoint':str(checkpoint),'epoch':best['epoch'],'representation':config['representation'],
            'validation_metrics':best['metrics'],'max_length':config['max_length'],
            'checkpoint_sha256':{str(p):digest(p) for p in files}}


def freeze(args):
    out=Path(args.output_dir)
    if out.exists():raise FileExistsError(f'Use a new evaluation directory: {out}')
    cache=Path(args.cache).resolve();manifest=json.loads((cache/'manifest.json').read_text())
    rows=read_rows(cache/'test.json')
    if len(rows)!=480:raise ValueError('Expected full 480-frame test split')
    if digest(cache/'test.json')!=manifest['split_sha256']['test']:raise ValueError('Test cache hash mismatch')
    selected={};runs={}
    for label,representation in [('T1','text'),('S1','soft')]:
        training=Path(args.run_root)/(label+'_train');runs[label]=json.loads((training/'run.json').read_text())
        selected[label]=selected_checkpoint(training)
        if selected[label]['representation']!=representation:raise ValueError('Wrong representation')
        if runs[label]['shared_manifest_sha256']!=digest(cache/'manifest.json'):raise ValueError('Evaluation cache differs from training cache')
        if runs[label]['base_identity']!=base_identity(runs[label]['args']['base_model']):raise ValueError('Base model changed since training')
        # Verify the runtime implementations against the ones used for training.
        for name in ['data.py','model.py','run.py','distributed.py']:
            if digest(Path(__file__).with_name(name))!=runs[label]['code_sha256'][name]:raise ValueError('Training/inference implementation changed: '+name)
    if runs['T1']['configuration']!=runs['S1']['configuration']:raise ValueError('Training configurations differ')
    if runs['T1']['base_identity']!=runs['S1']['base_identity'] or runs['T1']['args']['image_root']!=runs['S1']['args']['image_root']:raise ValueError('Base models or image sources differ')
    base=runs['T1']['args']['base_model']
    audit=input_length_audit(rows,load_processor(base),json.loads((Path(base)/'config.json').read_text()),max(s['max_length'] for s in selected.values()))
    print('Observation length audit: '+json.dumps(audit),flush=True)
    record={'schema':'t1-s1-validation-frozen-test-v1','selection_split':'val','evaluation_split':'test',
            'test_used_for_selection':False,'samples':len(rows),'cache':str(cache),
            'cache_manifest_sha256':digest(cache/'manifest.json'),'test_sha256':digest(cache/'test.json'),
            'experiments':selected,'base_model':str(Path(runs['T1']['args']['base_model']).resolve()),
            'base_identity':runs['T1']['base_identity'],'image_root':runs['T1']['args']['image_root'],
            'input_length_audit':audit,'evaluation_max_length':audit['evaluation_max_length'],
            'generation':{'do_sample':False,'max_new_tokens':512,'use_cache':True,'parser':'strict JSON finite [9,2]'},
            'inference_code_sha256':{name:digest(Path(__file__).with_name(name)) for name in ['data.py','model.py','run.py','distributed.py','evaluate.py']}}
    out.mkdir(parents=True,exist_ok=False)
    (out/'frozen_selection.json').write_text(json.dumps(record,indent=2))
    print(f"Frozen validation checkpoints: T1 epoch{selected['T1']['epoch']}, S1 epoch{selected['S1']['epoch']}; test=480. Output: {out}",flush=True)


def predict(args):
    record=json.loads(Path(args.selection).read_text());selected=record['experiments'][args.experiment]
    for path,expected in selected['checkpoint_sha256'].items():
        if digest(path)!=expected:raise ValueError('Frozen checkpoint changed: '+path)
    if digest(Path(record['cache'])/'test.json')!=record['test_sha256']:raise ValueError('Frozen test cache changed')
    if base_identity(record['base_model'])!=record['base_identity']:raise ValueError('Frozen base changed')
    for name,expected in record['inference_code_sha256'].items():
        if digest(Path(__file__).with_name(name))!=expected:raise ValueError('Frozen inference code changed: '+name)
    world=int(os.environ.get('WORLD_SIZE','1'));rank=0
    if not torch.cuda.is_available():raise RuntimeError('Real GPU evaluation requires CUDA')
    args.device='cuda:'+os.environ.get('LOCAL_RANK','0');torch.cuda.set_device(torch.device(args.device))
    if world>1:
        dist.init_process_group('nccl',timeout=timedelta(hours=12));rank=dist.get_rank()
    out=Path(args.output_dir)
    if rank==0:out.mkdir(parents=True,exist_ok=False)
    if world>1:dist.barrier()
    rows=read_rows(Path(record['cache'])/'test.json');args.representation=selected['representation'];args.image_root=record['image_root']
    processor=load_processor(record['base_model'])
    model=load_model(record['base_model'],processor,args.representation,args.device,torch.bfloat16,selected['max_length'],selected['checkpoint'])
    # Checkpoint loading restores the training guard. This is an inference-only
    # guard extension within the base model context; weights and prompts unchanged.
    model.max_length=record['evaluation_max_length']
    model.requires_grad_(False).eval()
    torch.cuda.reset_peak_memory_stats()
    if world>1:report,predictions=evaluate_distributed(model,rows,processor,args,evaluate,metrics)
    else:report,predictions=evaluate(model,rows,processor,args)
    if rank==0:
        report.update(experiment=args.experiment,split='test',checkpoint=selected['checkpoint'],epoch=selected['epoch'],world_size=world,collision_rate=None,evaluation_max_length=model.max_length)
        (out/'predictions.json').write_text(json.dumps(predictions))
        (out/'metrics.json').write_text(json.dumps(report,indent=2))
        (out/'complete.json').write_text(json.dumps({'samples':len(rows),'split':'test','experiment':args.experiment,'selection_sha256':digest(args.selection)},indent=2))
        print(json.dumps(report,indent=2),flush=True)
    if world>1:dist.barrier()


def summarize(args):
    root=Path(args.output_dir);record=json.loads((root/'frozen_selection.json').read_text());reports={}
    for label in ['T1','S1']:
        done=json.loads((root/label/'complete.json').read_text())
        if done['samples']!=480 or done['selection_sha256']!=digest(root/'frozen_selection.json'):raise ValueError('Incomplete/incompatible evaluation')
        reports[label]=json.loads((root/label/'metrics.json').read_text())
    before,after=reports['T1']['avg_l2_1_2_3s'],reports['S1']['avg_l2_1_2_3s']
    change=(before-after)/before if before is not None and after is not None and before>0 else None
    result={'split':'test','experiments':reports,'relative_avg_l2_reduction':change,
            'limitations':['single seed42','Oracle native objects','valid predictions only; compare coverage','no collision evaluation','no ego-only baseline; not a causal RSU benefit estimate']}
    (root/'comparison.json').write_text(json.dumps(result,indent=2))
    lines=['# T1/S1 test评测','', 'Checkpoint仅按validation冻结，test不用于重新选epoch。','',
           '| 实验 | epoch | L2@1s | L2@2s | L2@3s | 平均L2 | FDE@4.5s | 解析成功 | 平均输入tokens |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x):return f'{x:.4f}' if x is not None else '未评测'
    for label,r in reports.items():
        values=r.get('l2_1_2_3s',[None]*3)
        lines.append('| '+ ' | '.join([label,str(r['epoch']),*[fmt(x) for x in values],fmt(r['avg_l2_1_2_3s']),fmt(r.get('fde_4_5s')),f"{r['parsed']}/{r['samples']}",f"{r['mean_input_tokens']:.1f}"])+' |')
    lines += ['', 'L2/FDE单位米，仅有效预测计入误差；解析失败不可隐去。单seed、独立GT Oracle对象结果，不证明RSU相对ego-only收益。collision未评测。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','predict','summarize'])
    p.add_argument('--run-root',default='work_dirs/T1_S1_objects_4gpu_smoke_20261004')
    p.add_argument('--cache',default='data/Planning/dual_object_a1_v1_20261004')
    p.add_argument('--output-dir',required=True);p.add_argument('--selection');p.add_argument('--experiment',choices=['T1','S1'])
    args=p.parse_args()
    if args.action=='predict' and (not args.selection or not args.experiment):p.error('predict requires --selection and --experiment')
    {'freeze':freeze,'predict':predict,'summarize':summarize}[args.action](args)

if __name__=='__main__':
    try:main()
    except BaseException:
        # PyTorch's exception hook calls get_rank after finally destroys NCCL.
        sys.excepthook=sys.__excepthook__
        raise
    finally:
        if dist.is_initialized():dist.destroy_process_group()
