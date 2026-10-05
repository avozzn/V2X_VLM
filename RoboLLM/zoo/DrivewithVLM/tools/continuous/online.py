"""Train/predict I1 online: RSU soft tokens enter VLM, never the trajectory head."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from tools.continuous.data import observation_prompt
from tools.continuous.model import waypoint_loss
from tools.continuous.online_model import I1Planner, INTERFACE
from tools.continuous.train import metrics


def read_samples(path, mode, limit):
    rows = json.loads(Path(path).read_text())
    if limit:
        rows = rows[:limit]
    if not rows or len({r['token'] for r in rows}) != len(rows):
        raise ValueError('Empty/duplicate samples')
    for row in rows:
        agents, target = row['agents'], row['planning_targets']
        if agents['coordinate_frame'] != 'current_ego_lidar':
            raise ValueError('Wrong RSU coordinates')
        if agents['velocity_policy'] != 'disabled_for_all_agents':
            raise ValueError('Use audited velocity-disabled cache')
        source_time = agents.get('timestamp_us')
        if source_time is not None and int(source_time) > int(row['timestamp_us']):
            raise ValueError('Future RSU message is not an observation')
        if mode != 'zero' and any(agents['mask']) and source_time is None:
            raise ValueError('Valid RSU objects require a source timestamp')
        if mode != 'zero' and agents['source'] != mode:
            raise ValueError('RSU source mismatch')
        if int(row['ego_perception']['timestamp_us']) != int(row['timestamp_us']):
            raise ValueError('Ego observation timestamp mismatch')
        if target['coordinate_frame'] != 'current_ego_lidar_xy_m':
            raise ValueError('Wrong trajectory coordinates')
        if not np.allclose(target['future_times_s'], np.arange(1, 10) * .5):
            raise ValueError('Expected nine half-second steps')
    return rows


def build_batch(rows, processor, image_root, device, dtype):
    images, texts = [], []
    for row in rows:
        with Image.open(Path(image_root) / row['ego_image']) as img:
            images.append(img.convert('RGB'))
        messages = [{'role': 'user', 'content': [{'type': 'image'},
                    {'type': 'text', 'text': observation_prompt(row)}]}]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    inputs = processor(text=texts, images=images, padding=True, return_tensors='pt')
    inputs = {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
              for key, value in inputs.items()}
    geometry = torch.tensor([r['agents']['geometry'] for r in rows], device=device, dtype=torch.float32)
    mask = torch.tensor([r['agents']['mask'] for r in rows], device=device, dtype=torch.bool)
    return inputs, geometry, mask


@torch.no_grad()
def evaluate_online(model, rows, processor, args, dtype):
    model.eval()
    predictions = []
    for start in range(0, len(rows), args.batch_size):
        output = model(*build_batch(rows[start:start + args.batch_size], processor,
                                    args.image_root, args.device, dtype))
        predictions.extend(output['trajectory'].cpu().tolist())
    pred = np.asarray(predictions)
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite predictions')
    report = metrics(pred, np.asarray([r['planning_targets']['future_xy'] for r in rows]),
                     np.asarray([r['planning_targets']['future_mask'] for r in rows], dtype=bool))
    exported = [{'token': r['token'], 'scene_token': r['scene_token'],
                 'coordinate_frame': 'current_ego_lidar_xy_m',
                 'future_times_s': r['planning_targets']['future_times_s'], 'trajectory': xy}
                for r, xy in zip(rows, predictions)]
    return report, exported


def fingerprint(directory):
    directory = Path(directory)
    return {'config': hashlib.sha256((directory / 'config.json').read_bytes()).hexdigest(),
            'weights': [[p.name, p.stat().st_size, p.stat().st_mtime_ns]
                        for p in sorted(directory.glob('*.safetensors'))]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'predict'])
    parser.add_argument('--model', default='checkpoints/LLM/llava-next-interleave')
    parser.add_argument('--adapter', help='Optional frozen E2 adapter; no language LoRA updates in this first interface fix')
    parser.add_argument('--image-root', default='.')
    parser.add_argument('--train-samples')
    parser.add_argument('--val-samples')
    parser.add_argument('--samples')
    parser.add_argument('--checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--rsu-mode', choices=['zero', 'oracle', 'detector'], default='zero')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--head-dim', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.lr <= 0 or (args.limit is not None and args.limit < 1):
        parser.error('Require positive batch/epochs/lr/limit')
    if args.action == 'train' and not (args.train_samples and args.val_samples):
        parser.error('train requires --train-samples and --val-samples')
    if args.action == 'predict' and not (args.samples and args.checkpoint):
        parser.error('predict requires --samples and --checkpoint')
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError('Use a fresh I1 output; old M1/cache cannot be reused')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    contract = None
    if args.action == 'predict':
        contract = json.loads((Path(args.checkpoint) / 'i1.json').read_text())
        if contract['interface'] != INTERFACE:
            raise ValueError('Not an I1 checkpoint')
        args.model = contract['metadata']['model']
        args.adapter = contract['metadata']['adapter']
        args.rsu_mode = contract['settings']['rsu_mode']
        args.head_dim = contract['settings']['head_dim']
    train = read_samples(args.train_samples, args.rsu_mode, args.limit) if args.action == 'train' else None
    val = read_samples(args.val_samples if train is not None else args.samples, args.rsu_mode, args.limit)
    if train is not None and {r['scene_token'] for r in train} & {r['scene_token'] for r in val}:
        raise ValueError('Train/val scenes overlap')
    dtype = getattr(torch, args.dtype)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.patch_size = None  # Preserve this local 4.45.2 checkpoint's legacy image merge.
    processor.tokenizer.padding_side = 'left'
    backbone = LlavaForConditionalGeneration.from_pretrained(args.model, local_files_only=True,
                            torch_dtype=dtype, low_cpu_mem_usage=True).to(args.device)
    if args.adapter:
        from peft import PeftModel
        backbone = PeftModel.from_pretrained(backbone, args.adapter, is_trainable=False)
    model = I1Planner(backbone, args.rsu_mode, head_dim=args.head_dim).to(args.device)
    metadata = {'interface': INTERFACE, 'model': str(Path(args.model).resolve()),
                'adapter': str(Path(args.adapter).resolve()) if args.adapter else None,
                'base_fingerprint': fingerprint(args.model), 'arguments': vars(args),
                'rsu_location': 'before-language-decoder', 'planner_condition': 'wp_hidden_only',
                'task_token': 'dedicated-learned-soft-embedding', 'language_lora_training': False,
                'train_sha256': hashlib.sha256(Path(args.train_samples).read_bytes()).hexdigest() if train else None}
    if args.adapter:
        adapter_file = Path(args.adapter) / 'adapter_model.safetensors'
        metadata['adapter_sha256'] = hashlib.sha256(adapter_file.read_bytes()).hexdigest()
    if contract:
        for key in ['base_fingerprint', 'adapter_sha256']:
            if contract['metadata'].get(key) != metadata.get(key):
                raise ValueError('Checkpoint backbone/adapter changed')
        model.restore(args.checkpoint)
        report, rows = evaluate_online(model, val, processor, args, dtype)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(rows))
        print(json.dumps(report, indent=2))
        return
    destination.mkdir(parents=True)
    (destination / 'run.json').write_text(json.dumps(metadata, indent=2))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = list(range(len(train)))
        random.shuffle(indices)
        total, points = 0., 0
        for start in range(0, len(indices), args.batch_size):
            batch = [train[i] for i in indices[start:start + args.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            output = model(*build_batch(batch, processor, args.image_root, args.device, dtype))
            target = torch.tensor([r['planning_targets']['future_xy'] for r in batch], device=args.device)
            mask = torch.tensor([r['planning_targets']['future_mask'] for r in batch], device=args.device, dtype=torch.bool)
            loss = waypoint_loss(output['trajectory'], target, mask)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10., error_if_nonfinite=True)
            optimizer.step()
            count = int(mask.sum())
            total += float(loss.detach()) * count
            points += count
        report, rows = evaluate_online(model, val, processor, args, dtype)
        record = dict(epoch=epoch, train_loss=total / points, validation=report)
        with (destination / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if report['avg_l2_1_2_3s'] < best:
            best = report['avg_l2_1_2_3s']
            checkpoint = destination / f'checkpoint-{epoch}'
            model.save(checkpoint, metadata)
            (destination / 'best.json').write_text(json.dumps({'checkpoint': checkpoint.name, **record}, indent=2))
            (destination / 'val_predictions.json').write_text(json.dumps(rows))
    model.save(destination / 'last', metadata)


if __name__ == '__main__':
    main()
