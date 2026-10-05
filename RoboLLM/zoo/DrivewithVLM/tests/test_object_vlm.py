import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from transformers import LlavaConfig,LlavaForConditionalGeneration,Qwen2Config,SiglipVisionConfig
from tools.object_vlm.model import ObjectVLM
from tools.object_vlm.data import prompt,parse_trajectory
from tools.object_vlm.export import select_objects


def tiny_model():
    text=Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,pad_token_id=0,bos_token_id=1,eos_token_id=2)
    vision=SiglipVisionConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,image_size=8,patch_size=4)
    backbone=LlavaForConditionalGeneration(LlavaConfig(text_config=text.to_dict(),vision_config=vision.to_dict(),image_token_index=3,vision_feature_select_strategy='full',vision_feature_layer=-1))
    backbone.requires_grad_(False)
    return ObjectVLM(backbone,4,'soft',100)


def batch():
    return {'inputs':{'input_ids':torch.tensor([[1,3,5,4,5,6,7]]),'attention_mask':torch.ones(1,7,dtype=torch.long),
        'labels':torch.tensor([[-100,-100,-100,-100,-100,6,7]]),'pixel_values':torch.randn(1,3,8,8)},
        'geometry':torch.tensor([[[2.,0,0,4,2,2,0,1,0,0,1,0,0]]]),'sources':torch.tensor([[[1,1]]]),
        'quality':torch.tensor([[[.05,1,0]]]),'object_mask':torch.ones(1,1,dtype=torch.bool)}


class ObjectVLMTest(unittest.TestCase):
    def test_roi_budget_and_dual_counts_once(self):
        def item(x,source,a,b):return {'box':np.array([x,0,0,4,2,2,0.]),'class':0,'source_bits':source,'vehicle_index':a,'infra_index':b}
        objects,audit,stats=select_objects([item(10,[1,1],0,0),item(50,[0,1],None,1),item(51,[1,0],1,None)],.1,ego_k=1,rsu_k=1)
        self.assertEqual(objects['source_bits'],[[1,1],[0,1]]);self.assertEqual(stats['outside_roi'],1)
        self.assertEqual(objects['quality'],[[.1,1.,0.],[.1,1.,0.]])
    def test_prompt_no_future_or_track(self):
        obs={'state':{'velocity_xy_mps':[1,0]},'objects':{'geometry':[],'source_bits':[],'quality':[]}}
        self.assertNotIn('future_xy',prompt(obs,'text'));self.assertNotIn('track_id',prompt(obs,'soft'))
        self.assertIn('<v2x_objects>',prompt(obs,'soft'))
    def test_answer_mask_and_gradient(self):
        m=tiny_model();b=batch();merged,target=m.assemble(**b)
        self.assertEqual(int((target!=-100).sum()),2);self.assertEqual(merged['inputs_embeds'].shape[1],10)
        r=m(**b);r['loss'].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in m.object_encoder.parameters()))
        self.assertFalse(any(p.grad is not None for p in m.backbone.parameters()))
    def test_padding_mask_does_not_change_tokens(self):
        m=tiny_model();b=batch();a,_=m.assemble(**b)
        b['geometry']=torch.cat([b['geometry'],torch.full((1,1,13),float('nan'))],1)
        b['quality']=torch.cat([b['quality'],torch.full((1,1,3),float('nan'))],1)
        b['sources']=torch.cat([b['sources'],torch.zeros(1,1,2,dtype=torch.long)],1)
        b['object_mask']=torch.tensor([[True,False]])
        c,_=m.assemble(**b);torch.testing.assert_close(a['inputs_embeds'],c['inputs_embeds'])
    def test_empty_objects_and_context_failure(self):
        m=tiny_model();b=batch();b['object_mask'].zero_();merged,_=m.assemble(**b)
        self.assertEqual(merged['inputs_embeds'].shape[1],9)
        m.max_length=2
        with self.assertRaises(ValueError):m.assemble(**b)
    def test_generate_inject_once_and_cache_matches(self):
        m=tiny_model().eval();b=batch();del b['inputs']['labels']
        b['inputs']['input_ids']=b['inputs']['input_ids'][:,:-2];b['inputs']['attention_mask']=b['inputs']['attention_mask'][:,:-2]
        with patch.object(m,'assemble',wraps=m.assemble) as assembly:
            result=m.generate(**b,max_new_tokens=3,pad_token_id=0,eos_token_id=None)
            self.assertEqual(assembly.call_count,1)
        merged,_=m.assemble(**b)
        # Independent no-cache recomputation with generated token embeddings appended.
        embeds=merged['inputs_embeds'];mask=merged['attention_mask'];tokens=[]
        for _ in range(3):
            out=m.language(inputs_embeds=embeds,attention_mask=mask,use_cache=False)
            token=out.logits[:,-1].argmax(-1);tokens.append(token)
            embeds=torch.cat([embeds,m.language.get_input_embeddings()(token[:,None])],1)
            mask=torch.cat([mask,torch.ones(1,1,dtype=mask.dtype)],1)
        ref=torch.stack(tokens,1)
        torch.testing.assert_close(result,ref)
    def test_strict_parser(self):
        self.assertIsNotNone(parse_trajectory('[[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]'))
        self.assertIsNone(parse_trajectory('Here is [[0,1]]'));self.assertIsNone(parse_trajectory('[[NaN,1]]'))

if __name__=='__main__':unittest.main()

class RealProcessorTest(unittest.TestCase):
    def test_real_chat_answer_boundary(self):
        from pathlib import Path
        from tools.object_vlm.data import read_rows,load_processor,build_batch
        root=Path('/home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM')
        cache=root/'data/Planning/dual_object_a1_v1_20261004/train.json'
        if not cache.exists():self.skipTest('Shared native cache not present')
        rows=read_rows(cache)[:2];p=load_processor(root/'checkpoints/LLM/llava-next-interleave')
        for rep in ['text','soft']:
            b=build_batch(rows,p,'/home/zzn/V2X_VLM/UniV2X/datasets/V2X-Seq-SPD-New',rep,training=True)
            for i,row in enumerate(rows):
                supervised=b['inputs']['labels'][i];supervised=supervised[supervised!=-100]
                decoded=p.tokenizer.decode(supervised)
                self.assertIn(str(row['target']['future_xy'][0][0]),decoded)
                self.assertNotIn('Ego state',decoded)
            self.assertEqual(b['geometry'].shape[-1],13)

class AdapterRoundtripTest(unittest.TestCase):
    def test_lora_and_object_state_roundtrip(self):
        import tempfile,json
        from pathlib import Path
        from peft import get_peft_model,LoraConfig,PeftModel
        m=tiny_model();m.backbone.language_model=get_peft_model(m.backbone.language_model,LoraConfig(r=2,lora_alpha=2,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
        b=batch();loss=m(**b)['loss'];loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in m.named_parameters() if 'lora_' in n))
        class Processor:
            def save_pretrained(self,path):Path(path).mkdir()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model';m.save(path,Processor(),{'epoch':1})
            n=tiny_model()
            # Copy the identical frozen base, then reload the actual adapter and encoder.
            base=m.language.get_base_model()
            keys={k:v for k,v in base.state_dict().items() if 'lora_' not in k}
            keys={k.replace('.base_layer.','.'):v for k,v in keys.items()}
            n.backbone.language_model.load_state_dict(keys)
            n.backbone.vision_tower.load_state_dict(m.backbone.vision_tower.state_dict())
            n.backbone.multi_modal_projector.load_state_dict(m.backbone.multi_modal_projector.state_dict())
            n.backbone.language_model=PeftModel.from_pretrained(n.language,path/'adapter')
            n.object_encoder.load_state_dict(torch.load(path/'objects.pt',weights_only=True))
            m.eval();n.eval()
            torch.testing.assert_close(m(**b,return_logits=True)['answer_logits'],n(**b,return_logits=True)['answer_logits'])
            self.assertEqual(json.loads((path/'interface.json').read_text())['representation'],'soft')


class VariableBatchTest(unittest.TestCase):
    def test_left_padded_batch_matches_individual_answer_logits(self):
        m=tiny_model().eval();one=batch()
        a=m(**one,return_logits=True)['answer_logits']
        b={k:v.repeat(2,*([1]*(v.ndim-1))) for k,v in one.items() if k!='inputs'}
        b['inputs']={k:v.repeat(2,*([1]*(v.ndim-1))) for k,v in one['inputs'].items()}
        b['inputs']['input_ids'][1,0]=0;b['inputs']['attention_mask'][1,0]=0
        # Compare the second row separately, with its BOS removed.
        separate={k:v[1:2] for k,v in b.items() if k!='inputs'}
        separate['inputs']={k:(v[1:2,1:] if k!='pixel_values' else v[1:2]) for k,v in b['inputs'].items()}
        c=m(**separate,return_logits=True)['answer_logits']
        together=m(**b,return_logits=True)['answer_logits']
        torch.testing.assert_close(together[:2],a,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(together[2:],c,atol=1e-6,rtol=1e-5)
