"""Extract frozen observation-only ego features, shared by F0/F1/F2."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from tools.continuous.data import observation_prompt, observation_signature, PROMPT_VERSION
from tools.continuous.model import FrozenLlavaReadout


def main():
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    import transformers
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="checkpoints/LLM/llava-next-interleave")
    parser.add_argument("--image-root", default="data/V2X-Seq-SPD-New")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default=None)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("batch size and threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("Feature cache exists; use a new output")
    samples = json.loads(Path(args.samples).read_text())
    if args.limit:
        samples = samples[:args.limit]
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    # Keep this checkpoint's legacy processor path: SigLIP full features has
    # no CLS token. The actual merged mask is captured at the decoder input.
    processor.patch_size = None
    processor.tokenizer.padding_side = "left"
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype) if args.dtype else (
        torch.float32 if device.type == "cpu" else torch.bfloat16)
    backbone = LlavaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, local_files_only=True,
        low_cpu_mem_usage=True).to(device)
    extractor = FrozenLlavaReadout(backbone)
    features, merged_lengths = {}, []
    for offset in range(0, len(samples), args.batch_size):
        batch = samples[offset:offset + args.batch_size]
        images, texts = [], []
        for sample in batch:
            with Image.open(Path(args.image_root) / sample["ego_image"]) as img:
                images.append(img.convert("RGB"))
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": observation_prompt(sample)}]}]
            texts.append(processor.apply_chat_template(messages, tokenize=False,
                                                        add_generation_prompt=True))
        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
        inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point()
                  else v.to(device) for k, v in inputs.items()}
        hidden, length = extractor(**inputs)
        if not torch.isfinite(hidden).all():
            raise ValueError("Nonfinite frozen features")
        for sample, feature in zip(batch, hidden.cpu()):
            features[sample["token"]] = feature
        merged_lengths.append(length)
        print(f"cached {offset + len(batch)}/{len(samples)}; merged length={length}", flush=True)
    metadata = {
        "model": str(Path(args.model).resolve()), "prompt_version": PROMPT_VERSION,
        "ego_perception_sources": sorted({s["ego_perception"]["source"] for s in samples}),
        "readout": "language_model.model.final_normalized_last_valid_context",
        "transformers": transformers.__version__, "dtype": str(dtype),
        "samples_sha256": hashlib.sha256(Path(args.samples).read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256("\n".join(observation_prompt(s) for s in samples).encode()).hexdigest(),
        "observation_sha256": observation_signature(samples),
        "model_config_sha256": hashlib.sha256((Path(args.model) / "config.json").read_bytes()).hexdigest(),
        "preprocessor_sha256": hashlib.sha256((Path(args.model) / "preprocessor_config.json").read_bytes()).hexdigest(),
        "ego_dim": hidden.shape[-1], "merged_lengths": merged_lengths,
        "tokens": list(features), "image_augmentation": "none",
        "weight_files": [{"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                         for p in sorted(Path(args.model).glob("*.safetensors"))],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "metadata": metadata}, destination)


if __name__ == "__main__":
    main()
