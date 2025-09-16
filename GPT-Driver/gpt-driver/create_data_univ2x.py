import pickle
import ndjson
import json
import tiktoken
from prompt_message import system_message, generate_user_message, generate_assistant_message
from nuscenes import NuScenes
from projects.mmdet3d_plugin.datasets.data_utils.spd_trajectory_api import SPDTraj
data = pickle.load(open('/home/zzn/UniV2X/data/infos/V2X-Seq-SPD-New/cooperative/spd_infos_temporal_train.pkl', 'rb'))
split_data = json.load(open('/home/zzn/UniV2X/data/split_datas/cooperative-split-data-spd.json', 'r'))

train_tokens = split_data['batch_split']['train']
val_tokens = split_data['batch_split']['val']
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
num_train_samples = len(train_tokens)
num_language_tokens = 0
num_system_tokens = 0
num_user_tokens = 0
num_assistant_tokens = 0

predict_steps = 12
planning_steps = 10
fut_steps = 4
past_steps = 4
use_nonlinear_optimizer = True
traj_only = False
CLASSES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 
                        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
nusc = NuScenes(version='v1.0-trainval',dataroot="/home/zzn/UniV2X/datasets/V2X-Seq-SPD-New/cooperative", verbose=True)
use_col_optim = True
with_velocity = True
traj_api = SPDTraj(nusc, predict_steps, planning_steps, past_steps, fut_steps, with_velocity, CLASSES, box_mode_3d, use_nonlinear_optimizer)
          
train_messages = []
for token_i, token in enumerate(train_tokens):
    if token_i >= num_train_samples:
        break 
    user_message = generate_user_message(data, token)
    assitant_message = generate_assistant_message(data, token, traj_only=traj_only)
    if len(assitant_message.split("\n")) > 6:
        print()
        print(token)
        print(system_message)
        print(user_message)
        print(assitant_message)
    num_language_tokens += len(encoding.encode(system_message))
    num_system_tokens += len(encoding.encode(system_message))
    num_language_tokens += len(encoding.encode(user_message))
    num_user_tokens += len(encoding.encode(user_message))
    num_language_tokens += len(encoding.encode(assitant_message))
    num_assistant_tokens += len(encoding.encode(assitant_message))


    train_message = {"messages": 
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}, 
            {"role": "assistant", "content": assitant_message}
        ]
    }
    train_messages.append(train_message)

print("#### Cost Summarization ####")
print(f"Number of system tokens: {num_system_tokens}")
print(f"Number of user tokens: {num_user_tokens}")
print(f"Number of assistant tokens: {num_assistant_tokens}")
print(f"Number of total tokens: {num_language_tokens}")

with open("data/train.json", "w") as f:
    ndjson.dump(train_messages, f)