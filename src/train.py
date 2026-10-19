"""
Train a model to convert it from causal attention to masked diffusion

Steps:
1. Attention annealing.
2. Diffusion scoring
"""

from torch.utils.data.dataloader import DataLoader


from typing import Any, cast
import json
from pathlib import Path
import random
import time
import math
import argparse
from typed_argparse import TypedArgs

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel
from torch.utils.data import DataLoader

from src.config import Config
from src.models import MODELS
from src.mask import add_mask_token
from src.bidirectional import attention_mask_band, make_bidirectional
from src.data import PackedTokenDataset, anneal_width, build_token_stream, diffusion_loss, q_sample


class Args(TypedArgs):
    config: str


ap = argparse.ArgumentParser()
ap.add_argument("--config", type=str, required=True, help="Path to config file")
args = Args.from_argparse(ap.parse_args())

# Training config
cfg = Config.load(args.config)
spec = MODELS[cfg.spec]

out = Path(cfg.out_dir)
out.mkdir(exist_ok=True, parents=True)
log_path = out / "train_log.jsonl"

with open(out / "config.json", "w") as f:
    json.dump(cfg, f, indent=2, default=str)

random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

device = torch.device("mps")
print(f"Using device: {device}, dtype: {spec.dtype}")

#  Load model

tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(spec.hf_id))
model = AutoModelForCausalLM.from_pretrained(spec.hf_id, dtype=spec.dtype)
model.to(device)

# Mask Token setup
model, tokenizer, mask_id = add_mask_token(model, tokenizer)

make_bidirectional(model)

model.config.use_cache = False

n_params = sum(p.numel() for p in model.parameters())
print(f"base: {spec.hf_id}, params: {n_params / 1e6:.1f}M, mask_id: {mask_id}")

# Training

stream = build_token_stream(tokenizer, cfg.max_docs)
dataset = PackedTokenDataset(stream, seq_len=cfg.seq_len)
dataloader = DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    pin_memory=False,
)
print(f"data: {len(stream) / 1e6:.1f}M tokens -> {len(dataset)} blocks of {cfg.seq_len}")

# optimizer

decay, no_decay = [], []
for name, p in model.named_parameters():
    (no_decay if p.ndim < 2 else decay).append(p)

optimizer = torch.optim.AdamW(
    [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=cfg.lr,
    betas=(0.9, 0.95),
)


def lr_at(step: int) -> float:
    if step < cfg.warmup_steps:
        return (step + 1) / max(1, cfg.warmup_steps)

    prog = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


# Training Loop

step = 0
start = time.time()
running: list[float] = []

print(f"\ntraining for {cfg.steps} steps (anneal over {cfg.anneal_steps})\n")
header = f"{'step':>8} {'loss':>8} {'width':>8} {'lr':>8} {'tok/s':>8}"
print(header)
print("-" * len(header))

model.train()
done = False
while not done:
    for batch in dataloader:
        if step >= cfg.steps:
            done = True
            break

        # print("batch.shape", batch.shape)
        x_0: torch.Tensor = cast(torch.Tensor, batch.to(device))

        for g in optimizer.param_groups:
            g["lr"] = cfg.lr * lr_at(step)

        # noise
        noise = torch.randn(x_0.shape[0], 1, device=device).clamp(min=1e-3)
        x_t, mask = q_sample(x_0, noise, mask_id)

        width = anneal_width(step, cfg)

        attention_mask = attention_mask_band(cfg.seq_len, device=device, width=width)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        logits = model(input_ids=x_t, use_cache=False, attention_mask=attention_mask).logits
        loss = diffusion_loss(logits, x_0, mask, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        running.append(loss.item())
        step += 1

        if step == 1 or step % cfg.log_steps == 0:
            avg = sum(running[-cfg.log_steps :]) / min(len(running), cfg.log_steps)
            el = time.time() - start

            tps = step * cfg.batch_size * cfg.seq_len / max(el, 1e-9)

            print(f"{step:>8} {avg:>8.4f} {width:>8.1f} {cfg.lr * lr_at(step):>8.2e} {tps:>8.0f}", flush=True)
            with open(log_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "step": step,
                            "loss": avg,
                            "width": width,
                            "lr": cfg.lr * lr_at(step),
                            "tok_per_s": tps,
                        }
                    )
                    + "\n"
                )
            running = []

        if step % cfg.save_steps == 0 or step == cfg.steps:
            sd = model.state_dict()
            model.save_pretrained(out / "checkpoint", state_dict=sd)
            tokenizer.save_pretrained(out / "checkpoint")

            with open(out / "checkpoint" / "mask_id.txt", "w") as f:
                f.write(str(mask_id))

        break

elapsed = time.time() - start
print(f"\ndone in {elapsed / 60:.1f} min -> {out / 'checkpoint'}")
