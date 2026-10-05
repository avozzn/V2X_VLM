"""T1/S1 training, smoke, length audit and validation-only autoregressive evaluation."""
import argparse
import hashlib
import json
import math
import os
import torch.distributed as dist
import random
import time
from pathlib import Path
import numpy as np
import torch
from .data import read_rows,load_processor,build_batch,prompt,answer,parse_trajectory
from .model import load_model
from tools.continuous.train import metrics

def base_identity(path):
    root=Path(path)
    return {'config_sha256':hashlib.sha256((root/'config.json').read_bytes()).hexdigest(),
            'weights':[{'name':p.name,'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in sorted(root.glob('*.safetensors'))]}



def audit_lengths(rows,processor,base_path):
    config=json.loads((Path(base_path)/'config.json').read_text());v=config['vision_config']
    image_tokens=(v['image_size']//v['patch_size'])**2
    if config['vision_feature_select_strategy']=='default':image_tokens-=1
    maxima={};counts={}
    for representation in ('text','soft'):
        lengths=[]
        for r in rows:
            messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt(r['observation'],representation)}]},
                      {'role':'assistant','content':[{'type':'text','text':answer(r)}]}]
            text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
            prefix=processor.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True)
            if not text.startswith(prefix):raise ValueError('Incompatible assistant template')
            n=len(processor.tokenizer(prefix,add_special_tokens=False)['input_ids'])+len(processor.tokenizer(text[len(prefix):],add_special_tokens=False)['input_ids'])+image_tokens-1
            if representation=='soft':n+=len(r['observation']['objects']['geometry'])-1
            lengths.append(n)
        maxima[representation]=max(lengths);counts[representation]={'max':max(lengths),'median':float(np.median(lengths)),'p95':float(np.percentile(lengths,95))}
    max_length=max(4096,math.ceil(max(maxima.values())/1024)*1024)
    if max_length>config['text_config'].get('max_position_embeddings',32768):raise ValueError('Inputs exceed model context')
    return {'image_tokens':image_tokens,'max_length':max_length,'lengths':counts,'samples':len(rows),'test_used':False}


@torch.no_grad()
def evaluate(model,rows,processor,args):
    model.eval();predictions=[];parsed=[];accepted=[];lengths=[];start=time.monotonic()
    for i,r in enumerate(rows):
        batch=build_batch([r],processor,args.image_root,args.representation,device=args.device,dtype=torch.bfloat16)
        ids=model.generate(**batch,max_new_tokens=512,eos_token_id=processor.tokenizer.eos_token_id,pad_token_id=processor.tokenizer.pad_token_id)
        text=processor.tokenizer.decode(ids[0],skip_special_tokens=True);trajectory=parse_trajectory(text)
        predictions.append({'token':r['token'],'text':text,'trajectory':trajectory,'future_times_s':r['target']['future_times_s'],'input_tokens':model.last_lengths[0]})
        lengths.extend(model.last_lengths)
        if trajectory is not None:parsed.append(trajectory);accepted.append(r)
        if i%25==0:print('eval',i,'/',len(rows),'parsed',len(parsed),flush=True)
    report={'samples':len(rows),'parsed':len(parsed),'coverage':len(parsed)/len(rows),'parse_failures':len(rows)-len(parsed),
            'seconds':time.monotonic()-start,'mean_input_tokens':float(np.mean(lengths)),
            'max_input_tokens':max(lengths),'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None}
    if parsed:
        report.update(metrics(np.array(parsed),np.array([r['target']['future_xy'] for r in accepted]),np.array([r['target']['future_mask'] for r in accepted],bool)))
        report['fde_3s']=report['l2_1_2_3s'][2]
    else:report['avg_l2_1_2_3s']=None
    return report,predictions


def gradients(model):
    groups={'object':[],'lora':[],'frozen_vision':[]}
    for name,p in model.named_parameters():
        group='object' if name.startswith('object_encoder.') else 'lora' if 'lora_' in name else 'frozen_vision' if 'vision_tower' in name or 'multi_modal_projector' in name else None
        if group is not None and p.grad is not None:
            if not torch.isfinite(p.grad).all():raise ValueError('Nonfinite gradient '+name)
            groups[group].append(float(p.grad.norm()))
    if not any(x>0 for x in groups['lora']):raise ValueError('No LoRA gradient')
    if model.object_encoder and not any(x>0 for x in groups['object']):raise ValueError('No object gradient')
    if groups['frozen_vision']:raise ValueError('Vision stack unexpectedly trained')
    return {name:{'tensors':len(values),'nonzero':sum(x>0 for x in values)} for name,values in groups.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['audit','smoke','train','predict'])
    p.add_argument('--cache',default='data/Planning/dual_object_a1_v1_20261004')
    p.add_argument('--base-model',default='checkpoints/LLM/llava-next-interleave')
    p.add_argument('--image-root',default='/home/zzn/V2X_VLM/UniV2X/datasets/V2X-Seq-SPD-New')
    p.add_argument('--representation',choices=['text','soft'],default='soft')
    p.add_argument('--device',default='cuda:0');p.add_argument('--output-dir',required=True)
    p.add_argument('--distributed-smoke',action='store_true',help='Four-rank 9 train/8 val probe, one epoch; never a formal score')
    p.add_argument('--checkpoint');p.add_argument('--epochs',type=int,default=6);p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    from .distributed import initialize
    rank,world=initialize(args.action)
    if world>1:args.device='cuda:'+os.environ['LOCAL_RANK']
    out=Path(args.output_dir)
    if rank==0:out.mkdir(parents=True,exist_ok=False)
    if world>1:dist.barrier()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    rows=read_rows(Path(args.cache)/'train.json');val=read_rows(Path(args.cache)/'val.json')
    manifest=json.loads((Path(args.cache)/'manifest.json').read_text())
    for split in ('train','val'):
        if hashlib.sha256((Path(args.cache)/f'{split}.json').read_bytes()).hexdigest()!=manifest['split_sha256'][split]:raise ValueError('Shared cache changed')
    processor=load_processor(args.base_model);audit=audit_lengths(rows+val,processor,args.base_model)
    if args.distributed_smoke:
        if args.action!='train' or world!=4 or args.checkpoint:raise ValueError('Four-rank smoke requires fresh torchrun train with four ranks')
        rows=sorted(rows,key=lambda r:len(r['observation']['objects']['geometry']),reverse=True)[:9]
        val=val[:8];args.epochs=1
    if rank==0:
        (out/'length_audit.json').write_text(json.dumps(audit,indent=2))
        (out/'run.json').write_text(json.dumps({'args':vars(args),'length_audit':audit,'shared_manifest_sha256':hashlib.sha256((Path(args.cache)/'manifest.json').read_bytes()).hexdigest(),
            'base_identity':base_identity(args.base_model),
            'configuration':{'effective_batch':4,'world_size':world,'microbatch_per_rank':1,'gradient_accumulation_steps':4//world,'lr':2e-4,'weight_decay':0,'warmup_ratio':.03,'scheduler':'cosine','lora_r':8,'lora_alpha':8,'lora_dropout':0,'vision_frozen':True,'seed':args.seed,'test_used':False},
            'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}},indent=2))
    if world>1:dist.barrier()
    if args.action=='audit':print(json.dumps(audit,indent=2));return
    if not torch.cuda.is_available():raise RuntimeError('GPU required for real 7B smoke/train; CPU tests are separate')
    free_bytes,_=torch.cuda.mem_get_info(torch.device(args.device))
    if free_bytes<35*1024**3:
        raise RuntimeError(f'Insufficient free GPU memory ({free_bytes/1024**3:.1f} GiB); wait for idle hardware, never evict another job')
    model=load_model(args.base_model,processor,args.representation,args.device,torch.bfloat16,audit['max_length'],args.checkpoint)
    model.language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.backbone.enable_input_require_grads()
    parameters=[x for x in model.parameters() if x.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=0)
    if args.action=='smoke':
        # Use eight train observations, including object-rich examples.
        subset=sorted(rows,key=lambda r:len(r['observation']['objects']['geometry']),reverse=True)[:8]
        model.train();optimizer.zero_grad();losses=[];first_batch=None
        for r in subset:
            batch=build_batch([r],processor,args.image_root,args.representation,training=True,device=args.device,dtype=torch.bfloat16)
            if first_batch is None:first_batch=batch
            result=model(**batch)
            if not torch.isfinite(result['loss']):raise ValueError('Nonfinite smoke loss')
            (result['loss']/8).backward();losses.append(float(result['loss']))
        g=gradients(model);optimizer.step();optimizer.zero_grad();model.eval()
        # Changing geometry must influence logits while image and other observations are held fixed.
        sensitivity=None
        if args.representation=='soft':
            with torch.no_grad():
                before=model(**first_batch,return_logits=True)['answer_logits'];modified=dict(first_batch);modified['geometry']=first_batch['geometry'].clone();modified['geometry'][...,0]+=5
                after=model(**modified,return_logits=True)['answer_logits'];sensitivity=float((before-after).abs().max())
                if not sensitivity>0:raise ValueError('Object inputs have no logit effect')
        report,predictions=evaluate(model,subset,processor,args)
        model.save(out/'checkpoint',processor,{'action':'smoke','gradient_checks':g})
        del optimizer,parameters,first_batch,result,batch,model
        if args.representation=='soft':del before,after,modified
        torch.cuda.empty_cache()
        restored=load_model(args.base_model,processor,args.representation,args.device,torch.bfloat16,audit['max_length'],out/'checkpoint')
        restored.eval();batch=build_batch([subset[0]],processor,args.image_root,args.representation,device=args.device,dtype=torch.bfloat16)
        ids=restored.generate(**batch,max_new_tokens=512,eos_token_id=processor.tokenizer.eos_token_id,pad_token_id=processor.tokenizer.pad_token_id)
        text=processor.tokenizer.decode(ids[0],skip_special_tokens=True)
        if text!=predictions[0]['text']:raise ValueError('Restored generation differs')
        (out/'smoke.json').write_text(json.dumps({'losses':losses,'gradients':g,'object_logit_sensitivity':sensitivity,'reload_generation_equal':True,'evaluation':report,'predictions':predictions},indent=2));return
    if args.action=='predict':
        if not args.checkpoint:raise ValueError('Prediction requires checkpoint')
        report,pred=evaluate(model,val,processor,args);(out/'metrics.json').write_text(json.dumps(report,indent=2));(out/'predictions.json').write_text(json.dumps(pred));return
    if world>1:
        from .distributed import train
        train(model,rows,val,processor,args,out,optimizer,parameters,evaluate,metrics,gradients)
        return
    steps=math.ceil(len(rows)/4)*args.epochs;warmup=max(1,int(steps*.03))
    def lr_factor(step):
        return (step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor);best=None;step=0;start_epoch=0
    if args.checkpoint:
        ckpt=Path(args.checkpoint)
        settings=json.loads((ckpt/'interface.json').read_text())
        start_epoch=int(settings['metadata'].get('epoch',0));step=int(settings['metadata'].get('step',0))
        if not start_epoch:raise ValueError('Formal resume requires an epoch checkpoint; never initialize formal training from smoke')
        previous=json.loads((ckpt.parent/'run.json').read_text())
        if previous['shared_manifest_sha256']!=hashlib.sha256((Path(args.cache)/'manifest.json').read_bytes()).hexdigest() or previous['args']['epochs']!=args.epochs or previous['args']['seed']!=args.seed:
            raise ValueError('Resume cache/budget/seed differs')
        oldmetrics=settings['metadata']['metrics']
        best=((oldmetrics['coverage'],-oldmetrics['avg_l2_1_2_3s'] if oldmetrics['avg_l2_1_2_3s'] is not None else -float('inf')),start_epoch)
        if start_epoch:
            state=torch.load(ckpt/'training.pt',map_location=args.device,weights_only=False)
            optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
            torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    for epoch in range(start_epoch,args.epochs):
        order=np.random.default_rng(args.seed+epoch).permutation(len(rows));model.train();optimizer.zero_grad()
        total=0.;start=time.monotonic()
        for offset in range(0,len(order),4):
            indexes=order[offset:offset+4]
            for idx in indexes:
                batch=build_batch([rows[int(idx)]],processor,args.image_root,args.representation,training=True,device=args.device,dtype=torch.bfloat16)
                result=model(**batch)
                if not torch.isfinite(result['loss']):raise ValueError('Nonfinite training loss')
                (result['loss']/len(indexes)).backward();total+=float(result['loss'])
            if step==0:(out/'gradient_checks.json').write_text(json.dumps(gradients(model),indent=2))
            torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step();scheduler.step();optimizer.zero_grad();step+=1
            if step%10==0:print(f'epoch={epoch+1} step={step}/{steps} train_loss={total/(offset+len(indexes)):.4f}',flush=True)
        report,pred=evaluate(model,val,processor,args);report.update(epoch=epoch+1,steps=step,train_loss=total/len(rows),epoch_seconds=time.monotonic()-start)
        ckpt=out/f'epoch_{epoch+1}';model.save(ckpt,processor,{'epoch':epoch+1,'step':step,'metrics':report})
        torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},ckpt/'training.pt')
        (out/f'val_epoch_{epoch+1}.json').write_text(json.dumps({'metrics':report,'predictions':pred},indent=2))
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
        # Higher coverage wins first; never hide parse failures via subset L2.
        score=(report['coverage'],-report['avg_l2_1_2_3s'] if report['avg_l2_1_2_3s'] is not None else -float('inf'))
        if best is None or score>best[0]:
            best=(score,epoch+1);(out/'best_checkpoint.json').write_text(json.dumps({'epoch':epoch+1,'path':str(ckpt),'metrics':report,'selection':'coverage then validation avg L2'},indent=2))
    (out/'complete.json').write_text(json.dumps({'epochs':args.epochs,'steps':step,'best_epoch':best[1],'test_used':False},indent=2))

if __name__=='__main__':
    try:main()
    finally:
        if dist.is_initialized():dist.destroy_process_group()
