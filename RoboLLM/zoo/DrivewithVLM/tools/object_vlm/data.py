"""Observation-only rendering and answer masking for both representations."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

MARKER='<v2x_objects>'
SCHEMA='dual-native-a1-object-v1'


def validate(row):
    if row['schema']!=SCHEMA:raise ValueError('Wrong object schema')
    obs=row['observation'];agents=obs['objects']
    g=np.asarray(agents['geometry'],float).reshape(-1,13)
    bits=np.asarray(agents['source_bits'],int).reshape(-1,2)
    quality=np.asarray(agents['quality'],float).reshape(-1,3)
    if len(g)>48 or len(g)!=len(bits) or len(g)!=len(quality):raise ValueError('Invalid object arrays')
    if not np.isfinite(g).all() or not np.isfinite(quality).all():raise ValueError('Nonfinite object')
    if len(g) and ((g[:,3:6]<=0).any() or (np.linalg.norm(g[:,:2],axis=1)>50+1e-8).any()):raise ValueError('Invalid size/ROI')
    if len(g) and (not np.isin(bits, [0,1]).all() or (bits.sum(1)==0).any()):raise ValueError('Invalid source')
    if len(g) and ((g[:,8:10]!=0).any() or (quality[:,2]!=0).any()):raise ValueError('First round object velocity disabled')
    if len(g) and (not np.isin(g[:,10:13],[0.,1.]).all() or not (g[:,10:13].sum(1)==1).all()):raise ValueError('Bad class encoding')
    if len(g) and ((quality[:,0]<0).any() or not np.isin(quality[:,1:],[0.,1.]).all()):raise ValueError('Bad quality validity')
    if len(bits) and ((bits[:,0]>0).sum()>32 or (bits[:,0]==0).sum()>16):raise ValueError('Object budget exceeded')
    t=row['target']
    if not np.isfinite(np.asarray(t['future_xy'],float)).all() or np.asarray(t['future_xy']).shape!=(9,2) or not np.allclose(t['future_times_s'],np.arange(1,10)*.5):raise ValueError('Wrong horizon')
    if not all(t['future_mask']):raise ValueError('Language labels require all nine valid points')
    return row


def prompt(observation, representation):
    """Accept observation ONLY: no target, cooperative IDs, or old object text."""
    if representation not in ('text','soft'):raise ValueError(representation)
    state=observation['state'];objects=observation['objects']
    common=('Plan using the ego camera, observed traffic objects and ego state.\n'
      'Coordinates: current ego LiDAR, x forward, y left, z up; meters. '
      'Object dimensions are length,width,height in meters.\n'
      'Object geometry fields: x,y,z,l,w,h,sin_yaw,cos_yaw,vx,vy,vehicle,pedestrian,cyclist. '
      'source_bits=[ego_seen,rsu_seen]. quality=[rsu_age_s,age_valid,velocity_valid]. '
      'Object velocity is unknown; stored zeros are placeholders.\n'
      'Ego state/history: '+json.dumps(state,separators=(',',':'))+'\n'
      'Observed objects:\n')
    if representation=='text':
        encoded=[{'geometry':g,'source_bits':s,'quality':q} for g,s,q in zip(objects['geometry'],objects['source_bits'],objects['quality'])]
        section=json.dumps(encoded,separators=(',',':'))
    else:section=MARKER
    return common+section+'\nReturn only a JSON array of nine [x,y] future waypoints at 0.5s,1s,...,4.5s.'


def answer(row):
    return json.dumps(row['target']['future_xy'],separators=(',',':'))


def read_rows(path):
    rows=json.loads(Path(path).read_text())
    if not rows or len({r['token'] for r in rows})!=len(rows):raise ValueError('Empty/duplicate rows')
    return [validate(r) for r in rows]


def load_processor(path):
    from transformers import AutoProcessor
    p=AutoProcessor.from_pretrained(path,local_files_only=True)
    p.tokenizer.add_special_tokens({'additional_special_tokens':[MARKER]})
    p.tokenizer.padding_side='left'
    if p.tokenizer.pad_token_id is None:p.tokenizer.pad_token=p.tokenizer.eos_token
    # The wrapper expands image tokens itself using the actual vision output.
    p.patch_size=None
    return p


def build_batch(rows,processor,image_root,representation,training=False,device='cpu',dtype=torch.float32):
    images=[];full=[];prefixes=[]
    for r in rows:
        with Image.open(Path(image_root)/r['observation']['ego_image']) as im:images.append(im.convert('RGB'))
        messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt(r['observation'],representation)}]}]
        prefix=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        prefixes.append(prefix)
        if training:
            text=processor.apply_chat_template(messages+[{'role':'assistant','content':[{'type':'text','text':answer(r)}]}],tokenize=False,add_generation_prompt=False)
            if not text.startswith(prefix):raise ValueError('Chat template answer boundary is not prefix-compatible')
            full.append(text)
        else:full.append(prefix)
    inputs=processor(text=prefixes,images=images,padding=True,truncation=False,return_tensors='pt')
    if training:
        # Tokenize prompt and answer separately: BPE may merge the assistant's
        # final newline with the first JSON bracket if the whole chat is tokenized.
        # Preserve the exact same prompt tokens as generation.
        sequences=[];targets=[]
        for i,(prefix,text) in enumerate(zip(prefixes,full)):
            valid=inputs['attention_mask'][i].bool()
            prefill=inputs['input_ids'][i,valid]
            suffix=processor.tokenizer(text[len(prefix):],add_special_tokens=False)['input_ids']
            suffix=torch.tensor(suffix,dtype=torch.long)
            sequences.append(torch.cat([prefill,suffix]))
            targets.append(torch.cat([torch.full_like(prefill,-100),suffix]))
        maxlen=max(map(len,sequences))
        ids=torch.full((len(rows),maxlen),processor.tokenizer.pad_token_id,dtype=torch.long)
        labels=torch.full((len(rows),maxlen),-100,dtype=torch.long)
        attention=torch.zeros_like(ids)
        for i,(sequence,target) in enumerate(zip(sequences,targets)):
            start=maxlen-len(sequence);ids[i,start:]=sequence;labels[i,start:]=target;attention[i,start:]=1
        inputs['input_ids']=ids;inputs['attention_mask']=attention;inputs['labels']=labels
    k=max(1,max(len(r['observation']['objects']['geometry']) for r in rows))
    geometry=torch.zeros(len(rows),k,13);quality=torch.zeros(len(rows),k,3)
    sources=torch.zeros(len(rows),k,2,dtype=torch.long);mask=torch.zeros(len(rows),k,dtype=torch.bool)
    for i,r in enumerate(rows):
        a=r['observation']['objects'];n=len(a['geometry'])
        if n:
            geometry[i,:n]=torch.tensor(a['geometry']);quality[i,:n]=torch.tensor(a['quality'])
            sources[i,:n]=torch.tensor(a['source_bits']);mask[i,:n]=True
    inputs={k:v.to(device=device,dtype=dtype if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
    return {'inputs':inputs,'geometry':geometry.to(device),'quality':quality.to(device),
            'sources':sources.to(device),'object_mask':mask.to(device)}


def parse_trajectory(text):
    try:
        arr=np.asarray(json.loads(text.strip()),float)
        if arr.shape!=(9,2) or not np.isfinite(arr).all():return None
        return arr.tolist()
    except (ValueError,TypeError,json.JSONDecodeError):return None
