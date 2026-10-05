"""Autoregressive object-token wrapper; image/object inputs enter the decoder once."""
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

INTERFACE='dual-object-autoregressive-v1'


class ObjectEncoder(nn.Module):
    def __init__(self,hidden):
        super().__init__()
        self.geometry=nn.Sequential(nn.Linear(13,256),nn.GELU(),nn.Linear(256,hidden))
        self.position=nn.Sequential(nn.Linear(3,256),nn.GELU(),nn.Linear(256,hidden))
        self.quality=nn.Sequential(nn.Linear(3,64),nn.GELU(),nn.Linear(64,hidden))
        self.source=nn.Embedding(3,hidden);self.norm=nn.LayerNorm(hidden)
        self.register_buffer('scale',torch.tensor([50.,50.,5.,10.,10.,5.,1.,1.,20.,20.,1.,1.,1.]))
    def forward(self,geometry,sources,quality,mask):
        if geometry.shape!=(*mask.shape,13) or sources.shape!=(*mask.shape,2) or quality.shape!=(*mask.shape,3):raise ValueError('Object shape mismatch')
        valid=mask.bool()
        if not torch.isfinite(geometry[valid]).all() or not torch.isfinite(quality[valid]).all():raise ValueError('Nonfinite valid object')
        if valid.any() and (not torch.isin(sources[valid],torch.tensor([0,1],device=sources.device)).all() or (sources[valid].sum(-1)==0).any()):raise ValueError('Bad source bits')
        g=torch.where(valid[...,None],geometry,0).to(self.scale.dtype)/self.scale
        q=torch.where(valid[...,None],quality,0).to(self.scale.dtype)
        bits=torch.where(valid[...,None],sources,0)
        indexes=(bits[...,0]+2*bits[...,1]-1).clamp(0,2)
        tokens=self.norm(self.geometry(g)+self.position(g[...,:3])+self.quality(q)+self.source(indexes))
        return tokens


class ObjectVLM(nn.Module):
    def __init__(self,backbone,marker_id,representation='soft',max_length=4096):
        super().__init__()
        if representation not in ('text','soft'):raise ValueError(representation)
        self.backbone=backbone;self.marker_id=marker_id;self.representation=representation;self.max_length=max_length
        self.hidden=backbone.config.text_config.hidden_size
        self.object_encoder=ObjectEncoder(self.hidden) if representation=='soft' else None
        self.last_lengths=None
    @property
    def language(self):return self.backbone.language_model

    def assemble(self,inputs,geometry,quality,sources,object_mask):
        if inputs.get('past_key_values') is not None:raise ValueError('Assemble only on prefill')
        ids=inputs['input_ids'];attention=inputs['attention_mask'].bool();labels=inputs.get('labels')
        image_id=self.backbone.config.image_token_index
        # Frozen vision stack is evaluated once on real images; no detached language cache.
        with torch.no_grad():
            vision=self.backbone.vision_tower(inputs['pixel_values'],output_hidden_states=True)
            feature=vision.hidden_states[self.backbone.config.vision_feature_layer]
            strategy=self.backbone.config.vision_feature_select_strategy
            if strategy=='default':feature=feature[:,1:]
            elif strategy!='full':raise ValueError(strategy)
            feature=self.backbone.multi_modal_projector(feature)
        embeds=self.backbone.get_input_embeddings()(ids)
        objects=self.object_encoder(geometry,sources,quality,object_mask).to(embeds.dtype) if self.object_encoder else None
        rows=[];target_rows=[]
        for i in range(len(ids)):
            positions=attention[i].nonzero().flatten();parts=[];targets=[];images=markers=0
            for j in positions.tolist():
                token=int(ids[i,j])
                if token==image_id:
                    part=feature[i].to(embeds.dtype);images+=1;lab=torch.full((len(part),),-100,device=ids.device,dtype=torch.long)
                elif token==self.marker_id:
                    if self.representation!='soft':raise ValueError('Text branch received soft marker')
                    part=objects[i,object_mask[i].bool()];markers+=1;lab=torch.full((len(part),),-100,device=ids.device,dtype=torch.long)
                else:
                    part=embeds[i,j:j+1];lab=labels[i,j:j+1] if labels is not None else torch.full((1,),-100,device=ids.device,dtype=torch.long)
                parts.append(part);targets.append(lab)
            if images!=1 or markers!=(1 if self.representation=='soft' else 0):raise ValueError('Expected one image and one S1 marker')
            row=torch.cat(parts);target=torch.cat(targets)
            if len(row)>self.max_length:raise ValueError(f'Expanded sequence {len(row)} exceeds {self.max_length}; never truncate')
            rows.append(row);target_rows.append(target)
        maxlen=max(map(len,rows));batch=embeds.new_zeros(len(rows),maxlen,self.hidden)
        mask=torch.zeros(len(rows),maxlen,device=ids.device,dtype=torch.long)
        targets=torch.full((len(rows),maxlen),-100,device=ids.device,dtype=torch.long)
        for i,(row,target) in enumerate(zip(rows,target_rows)):
            start=maxlen-len(row);batch[i,start:]=row;mask[i,start:]=1;targets[i,start:]=target
        position=mask.cumsum(-1)-1;position.masked_fill_(mask==0,0)
        self.last_lengths=mask.sum(1).detach().cpu().tolist()
        return {'inputs_embeds':batch,'attention_mask':mask,'position_ids':position},targets

    def forward(self,inputs,geometry,quality,sources,object_mask,return_logits=False):
        merged,targets=self.assemble(inputs,geometry,quality,sources,object_mask)
        if 'labels' not in inputs:raise ValueError('Training forward requires masked labels')
        base=self.language.get_base_model() if hasattr(self.language,'get_base_model') else self.language
        hidden=base.model(**merged,use_cache=False,return_dict=True).last_hidden_state
        h=hidden[:,:-1][targets[:,1:]!=-100];t=targets[:,1:][targets[:,1:]!=-100]
        if not len(t):raise ValueError('No supervised answer tokens')
        # CE only on answers; recomputation prevents retaining large vocab logits for all chunks.
        def chunk_loss(features,labels):
            return nn.functional.cross_entropy(base.lm_head(features).float(),labels,reduction='sum')
        loss=hidden.new_zeros((),dtype=torch.float32)
        for start in range(0,len(t),64):
            loss=loss+checkpoint(chunk_loss,h[start:start+64],t[start:start+64],use_reentrant=False)
        result={'loss':loss/len(t),'answer_tokens':len(t),'merged_lengths':self.last_lengths}
        if return_logits:result['answer_logits']=base.lm_head(h)
        return result

    @torch.no_grad()
    def generate(self,inputs,geometry,quality,sources,object_mask,**kwargs):
        if 'labels' in inputs:raise ValueError('Inference must not receive answer labels')
        merged,_=self.assemble(inputs,geometry,quality,sources,object_mask)
        # Qwen generation sees embeddings only during prefill; no object/image hook remains.
        merged.pop('position_ids') # Qwen prepare_inputs_for_generation derives/slices positions from the growing mask.
        return self.language.generate(**merged,use_cache=True,do_sample=False,**kwargs)

    def save(self,path,processor,metadata):
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        self.language.save_pretrained(path/'adapter')
        if self.object_encoder:torch.save(self.object_encoder.state_dict(),path/'objects.pt')
        processor.save_pretrained(path/'processor')
        (path/'interface.json').write_text(json.dumps({'interface':INTERFACE,'representation':self.representation,
            'marker_id':self.marker_id,'max_length':self.max_length,'metadata':metadata},indent=2))


def load_model(base_path,processor,representation,device='cpu',dtype=torch.float32,max_length=4096,checkpoint_dir=None):
    from transformers import LlavaForConditionalGeneration
    from peft import LoraConfig,get_peft_model,PeftModel
    backbone=LlavaForConditionalGeneration.from_pretrained(base_path,torch_dtype=dtype,local_files_only=True,attn_implementation='sdpa')
    # The tokenizer adds one marker; local Qwen has unused vocabulary slots.
    if len(processor.tokenizer)>backbone.get_input_embeddings().num_embeddings:
        backbone.resize_token_embeddings(len(processor.tokenizer))
    backbone.requires_grad_(False)
    if checkpoint_dir:
        settings=json.loads((Path(checkpoint_dir)/'interface.json').read_text())
        if settings['interface']!=INTERFACE or settings['representation']!=representation or settings['marker_id']!=processor.tokenizer.convert_tokens_to_ids('<v2x_objects>'):raise ValueError('Checkpoint interface mismatch')
        max_length=settings['max_length']
        backbone.language_model=PeftModel.from_pretrained(backbone.language_model,Path(checkpoint_dir)/'adapter',is_trainable=True)
    else:
        targets=[name for name,module in backbone.language_model.named_modules() if isinstance(module,nn.Linear) and name!='lm_head']
        backbone.language_model=get_peft_model(backbone.language_model,LoraConfig(r=8,lora_alpha=8,lora_dropout=0.,bias='none',target_modules=targets,task_type='CAUSAL_LM'))
    model=ObjectVLM(backbone,processor.tokenizer.convert_tokens_to_ids('<v2x_objects>'),representation,max_length)
    if checkpoint_dir and model.object_encoder:model.object_encoder.load_state_dict(torch.load(Path(checkpoint_dir)/'objects.pt',map_location='cpu',weights_only=True))
    model.to(device=device) # Keep object encoder FP32; cast its output to language dtype.
    return model
