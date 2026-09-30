from tqdm import tqdm
import torch
import ast
import numpy as np
from train_utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)
import gc
import torch.distributed as dist
import copy
from .TTA_Dataset import TTADataset
def inference_sample(model,batch,processor):
    model.eval()
    device=model.device
    with torch.no_grad():
        data = batch['llm_input']
        # Move data to the correct device
        token=data['token']
        data = {k: v.to(device) for k, v in data.items() if k != "token"}
        pixel_values = data.get('pixel_values', None)
        if pixel_values is not None:
            data['pixel_values'] = data['pixel_values'].to(torch.bfloat16)

        # Generate output
        output = model.generate(**data, max_new_tokens=500, do_sample=False,output_scores=True,return_dict_in_generate=True)



        #####返回的值为字符串
        result = processor.decode(output.sequences[0], skip_special_tokens=True)
        
        traj_str = result.split("\n")[-1]
            # 将字符串转换为列表
        traj = ast.literal_eval(traj_str)
            # 将列表转换为 NumPy 数组
        traj = np.array(traj)

        label=result.split("assistant\n\n")[-1]
        scores = torch.stack(output.scores, dim=0)  # 形状: [seq_len, batch_size, vocab_size]
        scores = scores.squeeze(1)  # 如果 batch_size=1，去掉 batch 维度 → [seq_len, vocab_size]

        # 计算各步最大概率
        pred_probs = torch.softmax(scores, dim=-1).max(dim=-1).values[-len(processor(text=traj_str)['input_ids'][0]):]

        geom_mean = torch.exp(torch.mean(torch.log(pred_probs)))

        return traj,geom_mean,label



'''
train
data=dict(
**vision_inputs,
input_ids=input_ids,
labels=labels,
attention_mask=attention_mask,
)
question = self.processor.decode(input_ids[0], skip_special_tokens=False)
question = question.split("<|im_end|>\n<|endoftext|>")[0]
process = self.processor(text=question, return_tensors='pt')

attention_mask?
'''
def pre_training(data,gt,processor):
    max_len=1024
    IGNORE_TOKEN_ID=-100
    input_ids=[]
    label=[]
    attention_mask=[]
    tokenizer = processor.tokenizer
    PAD_TOKEN_ID=tokenizer.pad_token_id
    question_ids=data['input_ids'][0]

    question =processor.decode(question_ids,skip_special_tokens=False)
    cur_input=question+gt



    cur_input_ids=processor(text=cur_input, return_tensors='pt')['input_ids']


    question_len = len(question_ids)
    question_mask = torch.arange(cur_input_ids.size(1)).unsqueeze(0) < question_len
    cur_labels = torch.where(question_mask,IGNORE_TOKEN_ID,cur_input_ids)

    # padding
    if cur_input_ids.shape[1] < max_len:
        cur_input_ids = torch.cat([
            cur_input_ids,
            torch.full(
                (cur_input_ids.shape[0], max_len - cur_input_ids.shape[1]),
                PAD_TOKEN_ID,
                dtype=cur_input_ids.dtype,
                device=cur_input_ids.device
            )
        ], dim=1)
        cur_labels = torch.cat([
            cur_labels,
            torch.full(
                (cur_labels.shape[0], max_len - cur_labels.shape[1]),
                IGNORE_TOKEN_ID,
                dtype=cur_labels.dtype,
                device=cur_labels.device
            )
        ], dim=1)

    cur_attention_mask=cur_input_ids.ne(PAD_TOKEN_ID)
    

    input_ids.append(cur_input_ids)
    label.append(cur_labels)
    attention_mask.append(cur_attention_mask)

    input_ids=torch.cat(input_ids, dim=1)
    label=torch.cat(label, dim=1)
    attention_mask=torch.cat(attention_mask,dim=1)
    data.pop('token')
    data['input_ids']=input_ids
    data['labels']=label
    data['attention_mask']=attention_mask

    dataset = TTADataset(data)




    return dataset




def path_distance(path1, path2):
    return np.linalg.norm(np.array(path1) - np.array(path2))
    
def select(result,key_wrong):
    if len(result)==1:
        return  key_wrong,list(result.keys())[0]
    
    ###calculate trajs
    all_trajs = [result[key]['trajs'] for key in result]
    ref_path = np.mean(all_trajs, axis=0)
    deviations = {key: path_distance(result[key]['trajs'], ref_path) for key in result}
    max_value = max(deviations.values())
    max_deviation_keys = [k for k, v in deviations.items() if v == max_value][0]
    remaining_trajs = {
    key: result[key] 
    for key in result 
    if key != max_deviation_keys 
    }
    
    if key_wrong==None and remaining_trajs != {}:
        min_prob_key = min(remaining_trajs, key=lambda k: remaining_trajs[k]['probs'])
        return max_deviation_keys,min_prob_key
    if key_wrong!=None and remaining_trajs != {}:
        min_prob_key = min(remaining_trajs, key=lambda k: remaining_trajs[k]['probs'])
        return key_wrong,min_prob_key
    if key_wrong!=None and remaining_trajs == {}:
        return key_wrong,max_deviation_keys
    if key_wrong==None and remaining_trajs == {}:
        return None,None

    


'''
dataloaders={
    key:dataloder,...
}

result={
    key: {
    probs:
    trajs:
    batch:
    }
}

train
results=dict(
**vision_inputs,
input_ids=input_ids,
labels=labels,
attention_mask=attention_mask,
)

test
results['llm_input']=dict(
token=token,
input_ids=input_ids,
attention_mask=attention_mask,
)

question = self.processor.decode(input_ids[0], skip_special_tokens=False)
question = question.split("<|im_end|>\n<|endoftext|>")[0]
process = self.processor(text=question, return_tensors='pt')

attention_mask?
'''



def TTA(model,dataloaders,rank,num_iter,processor,trainer):
    
    iterators = {key: iter(dataloader) for key, dataloader in dataloaders.items()}
    result={}
    sample={}
    remained_result={}


    for i in tqdm(range(num_iter)):
        key_pp=None
        for key in iterators:
            sample['batch']=next(iterators[key])
            flag=True
            try:
                sample['trajs'],sample['probs'],sample['label']=inference_sample(model,sample['batch'],processor)
                print(key)
                print(sample['trajs'])
                print(sample['probs'])
                assert sample['trajs'].shape == (6, 2)
            except Exception as e:
                flag=False
                print(f"Error {type(e).__name__} - {str(e)}")
            result[key]=copy.deepcopy(sample)
            remained_result[key]=copy.deepcopy(sample)
            
            if flag==False:
                key_pp=key
                result.pop(key)
        ###if all predict wrong
        skip_flag = torch.tensor(1 if len(result)<1 else 0, device=f'cuda:{rank}')
        dist.all_reduce(skip_flag, op=dist.ReduceOp.MAX)
        if skip_flag.item() == 1:
            print("all is wrong\n")
            gc.collect()
            continue

        ### result>=1

        key_worse,key_label=select(result,key_pp)
        print(f"key_worse:{key_worse}\n, rank: {rank}\n")
        print(f"key_label: {key_label}\n,rank: {rank}\n")

        skip_flag = torch.tensor(1 if key_worse==None else 0, device=f'cuda:{rank}')
        dist.all_reduce(skip_flag, op=dist.ReduceOp.MAX)
        if skip_flag.item() == 1:
            print("all is 0\n")
            gc.collect()
            continue
            

        # if key_label !="all":
        #     key_worse="all"

        data=remained_result[key_worse]['batch']['llm_input']
        dataset=pre_training(data,remained_result[key_label]['label'],processor)


        trainer.train_dataset=dataset
        trainer.train()
        trainer.model_wrapped=trainer.model


        dist.barrier()
        gc.collect()
      



            
            

