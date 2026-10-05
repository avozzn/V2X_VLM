"""Four-rank data-parallel training with exact global batches of four.

Gradients are summed explicitly, rather than padded DistributedSampler batches:
929 samples yields 232 full batches and one singleton, with no duplicate/drop.
An idle rank in the final batch still participates in gradient collectives.
"""
import math
import os
import time
from datetime import timedelta
from pathlib import Path
import json
import numpy as np
import torch
import torch.distributed as dist
from .data import build_batch

GLOBAL_BATCH=4


def initialize(action):
    world=int(os.environ.get('WORLD_SIZE','1'))
    if world==1:return 0,1
    if action!='train':raise ValueError('torchrun is supported for formal training only')
    if world not in (2,4):raise ValueError('Global batch4 supports two or four ranks')
    if not torch.cuda.is_available():raise RuntimeError('Distributed 7B training requires CUDA')
    local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
    dist.init_process_group(backend='nccl',timeout=timedelta(hours=12))
    return dist.get_rank(),dist.get_world_size()


def local_indexes(global_indexes,rank,world):
    return list(global_indexes)[rank::world]


def sum_gradients(parameters):
    """Losses are pre-divided by actual global count; SUM is the desired mean.

Preserve grad=None if a parameter was unused on EVERY rank. This matters for
empty-object batches: Adam momentum must not advance solely due to padding.
"""
    parameters=list(parameters)
    if not parameters:return
    device=parameters[0].device
    presence=torch.tensor([p.grad is not None for p in parameters],device=device,dtype=torch.int32)
    dist.all_reduce(presence,op=dist.ReduceOp.MAX)
    active=[p for p,used in zip(parameters,presence.cpu().tolist()) if used]
    if not active:return
    flat=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).float() for p in active])
    dist.all_reduce(flat,op=dist.ReduceOp.SUM)
    offset=0
    for p in active:
        value=flat[offset:offset+p.numel()].view_as(p).to(p.dtype)
        if p.grad is None:p.grad=value.clone()
        else:p.grad.copy_(value)
        offset+=p.numel()


def summarize_predictions(rows,predictions,seconds,input_lengths,peak_memory,metrics_fn):
    by_token={p['token']:p for p in predictions}
    if len(by_token)!=len(rows) or set(by_token)!={r['token'] for r in rows}:raise ValueError('Distributed validation coverage mismatch')
    ordered=[by_token[r['token']] for r in rows]
    valid=[(r,p) for r,p in zip(rows,ordered) if p['trajectory'] is not None]
    report={'samples':len(rows),'parsed':len(valid),'coverage':len(valid)/len(rows),
            'parse_failures':len(rows)-len(valid),'seconds':seconds,
            'mean_input_tokens':float(np.mean(input_lengths)),'max_input_tokens':max(input_lengths),
            'peak_gpu_memory_bytes':peak_memory}
    if valid:
        report.update(metrics_fn(np.asarray([p['trajectory'] for r,p in valid]),
            np.asarray([r['target']['future_xy'] for r,p in valid]),np.asarray([r['target']['future_mask'] for r,p in valid],bool)))
        report['fde_3s']=report['l2_1_2_3s'][2]
    else:report['avg_l2_1_2_3s']=None
    return report,ordered


def evaluate_distributed(model,rows,processor,args,evaluate_fn,metrics_fn):
    rank=dist.get_rank();world=dist.get_world_size()
    subset=rows[rank::world]
    started=time.monotonic();local_report,predictions=evaluate_fn(model,subset,processor,args)
    # Carry input lengths with each prediction; keep original-frame ordering on rank0.
    payload={'predictions':predictions,'input_lengths':[p['input_tokens'] for p in predictions],
             'seconds':time.monotonic()-started,'peak_memory':local_report['peak_gpu_memory_bytes']}
    gathered=[None]*world if rank==0 else None
    dist.gather_object(payload,gathered,dst=0)
    if rank!=0:return None,None
    return summarize_predictions(rows,[p for g in gathered for p in g['predictions']],max(g['seconds'] for g in gathered),
                                 [n for g in gathered for n in g['input_lengths']],max(g['peak_memory'] for g in gathered),metrics_fn)


def train(model,rows,val,processor,args,out,optimizer,parameters,evaluate_fn,metrics_fn,gradients_fn):
    rank=dist.get_rank();world=dist.get_world_size()
    # Base is the same local checkpoint; broadcast every trainable parameter as
    # an explicit startup guarantee, including newly initialized object weights.
    for p in parameters:dist.broadcast(p.data,src=0)
    steps=math.ceil(len(rows)/GLOBAL_BATCH)*args.epochs;warmup=max(1,int(steps*.03))
    def factor(step):return (step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor);step=0;start_epoch=0;best=None
    if args.checkpoint:
        ckpt=Path(args.checkpoint);settings=json.loads((ckpt/'interface.json').read_text())
        start_epoch=int(settings['metadata'].get('epoch',0));step=int(settings['metadata'].get('step',0))
        if not start_epoch:raise ValueError('Formal resume requires epoch checkpoint')
        previous=json.loads((ckpt.parent/'run.json').read_text())
        current=json.loads((out/'run.json').read_text())
        if previous['shared_manifest_sha256']!=current['shared_manifest_sha256'] or previous['args']['epochs']!=args.epochs or previous['args']['seed']!=args.seed:
            raise ValueError('Resume cache/budget/seed differs')
        if previous['configuration'].get('world_size',1)!=world:raise ValueError('Resume world size differs')
        state=torch.load(ckpt/'training.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        rng=state['rank_rng'][rank];torch.set_rng_state(rng['torch_rng']);torch.cuda.set_rng_state(rng['cuda_rng'])
        old=settings['metadata']['metrics'];best=((old['coverage'],-old['avg_l2_1_2_3s'] if old['avg_l2_1_2_3s'] is not None else -float('inf')),start_epoch)
    for epoch in range(start_epoch,args.epochs):
        order=np.random.default_rng(args.seed+epoch).permutation(len(rows));model.train();optimizer.zero_grad();total=0.;started=time.monotonic()
        for offset in range(0,len(order),GLOBAL_BATCH):
            indexes=order[offset:offset+GLOBAL_BATCH]
            for index in local_indexes(indexes,rank,world):
                batch=build_batch([rows[int(index)]],processor,args.image_root,args.representation,training=True,device=args.device,dtype=torch.bfloat16)
                result=model(**batch)
                if not torch.isfinite(result['loss']):raise ValueError('Nonfinite distributed loss')
                (result['loss']/len(indexes)).backward();total+=float(result['loss'])
            sum_gradients(parameters)
            if step==0 and rank==0:(out/'gradient_checks.json').write_text(json.dumps(gradients_fn(model),indent=2))
            torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step();optimizer.zero_grad();step+=1
            if step%10==0:
                aggregate=torch.tensor(total,device=args.device,dtype=torch.float64);dist.all_reduce(aggregate)
                if rank==0:print(f'epoch={epoch+1} step={step}/{steps} train_loss={float(aggregate)/(offset+len(indexes)):.4f}',flush=True)
        aggregate=torch.tensor(total,device=args.device,dtype=torch.float64);dist.all_reduce(aggregate)
        report,pred=evaluate_distributed(model,val,processor,args,evaluate_fn,metrics_fn)
        rng={'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
        all_rng=[None]*world if rank==0 else None;dist.gather_object(rng,all_rng,dst=0)
        if rank==0:
            report.update(epoch=epoch+1,steps=step,train_loss=float(aggregate)/len(rows),epoch_seconds=time.monotonic()-started)
            ckpt=out/f'epoch_{epoch+1}';model.save(ckpt,processor,{'epoch':epoch+1,'step':step,'metrics':report,'world_size':world})
            torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rank_rng':all_rng},ckpt/'training.pt')
            (out/f'val_epoch_{epoch+1}.json').write_text(json.dumps({'metrics':report,'predictions':pred},indent=2))
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
            score=(report['coverage'],-report['avg_l2_1_2_3s'] if report['avg_l2_1_2_3s'] is not None else -float('inf'))
            if best is None or score>best[0]:
                best=(score,epoch+1);(out/'best_checkpoint.json').write_text(json.dumps({'epoch':epoch+1,'path':str(ckpt),'metrics':report,'selection':'coverage then validation avg L2'},indent=2))
        dist.barrier()
    if rank==0:(out/'complete.json').write_text(json.dumps({'epochs':args.epochs,'steps':step,'best_epoch':best[1],'world_size':world,'test_used':False},indent=2))
    dist.barrier()
