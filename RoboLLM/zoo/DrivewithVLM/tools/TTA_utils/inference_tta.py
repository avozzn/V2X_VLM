    
import torch
from tqdm import tqdm
import json
import torch.distributed as dist
import gc

def test(model,dataloader,rank,world_size,output_file,processor):

    if rank == 0:
        # Clear the file if it exists and start the JSON array
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('[')  # Start of JSON array
        first_iteration = True

    model.eval()
    dist.barrier()
    for batch in tqdm(dataloader, desc=f"Rank {rank} Processing"):
        local_results = []  # Clear local results at the start of each iteration
        device=model.device
        with torch.no_grad():
            data = batch['llm_input']
            # Move data to the correct device
            token=data['token']
            data = {k: v.to(device) for k, v in data.items() if k != "token"}
            data['pixel_values'] = data['pixel_values'].to(torch.bfloat16)
            
            # Decode and preprocess the question
            question = processor.decode(data['input_ids'][0], skip_special_tokens=False)
            # Generate output
            output = model.generate(**data, max_new_tokens=500, do_sample=False)

            result = processor.decode(output[0], skip_special_tokens=True)
            
            # Store results locally
            result_dict = {
                'token':token,
                'input': question,
                'answer': result,
            }
            local_results.append(result_dict)

            del data,output
            gc.collect()

        # Gather all results to rank 0
        if world_size > 1:
            if rank == 0:
                gathered_results = [None for _ in range(world_size)]
                dist.gather_object(local_results, object_gather_list=gathered_results, dst=0)
            else:
                dist.gather_object(local_results, dst=0)
        else:
            gathered_results = [local_results]

        # Write results to the file on rank 0
        if rank == 0:
            mode = 'w' if first_iteration else 'a'
            with open(output_file, mode, encoding='utf-8') as f:
                if first_iteration:
                    f.write('[')  # Start of JSON array
                    first_iteration = False
                for r in gathered_results:
                    for item in r:
                        json.dump(item, f, ensure_ascii=False)
                        f.write(',\n')
    # Finalize the JSON array on rank 0
    if rank == 0:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(']')  # End of JSON array
        print('测试回答全部写入完毕\n')

    dist.barrier()




