"""
Train a model to convert it from causal attention to masked diffusion

Steps:
1. Attention annealing.
2. Diffusion scoring
"""

from torch.utils.data.dataloader import DataLoader


from typing import Any, cast
from dataclasses import asdict
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
from src.bidirectional import anneal_attn_mask, attention_mask_band, build_attn_mask, make_bidirectional, to_additive_mask
from src.data import (
    PackedTokenDataset,
    anneal_ratio,
    anneal_width,
    build_token_stream,
    diffusion_loss_parts,
    q_sample,
)
from src.eval_per_t import eval_per_t, unigram_ce


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
    json.dump(asdict(cfg), f, indent=2, default=str)

with open(log_path, "w") as f:
    f.write("")

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

stream = build_token_stream(tokenizer, cfg)
dataset = PackedTokenDataset(stream, seq_len=cfg.seq_len)

n_eval = min(cfg.eval_blocks, max(1, len(dataset) // 10))
train_dataset = dataset[: len(dataset) - n_eval]
eval_block_tensor = dataset.blocks[len(dataset) - n_eval :]

dataloader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    pin_memory=False,
)
print(f"data: {len(stream) / 1e6:.1f}M tokens -> {len(dataset)} blocks of {cfg.seq_len}, ({len(train_dataset)} train, {n_eval} held out for eval)")

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
running_unweighted: list[float] = []

#! ---------
unigram = unigram_ce(eval_block_tensor, model.config.vocab_size)

print(f"\ntraining for {cfg.steps} steps (anneal over {cfg.anneal_steps}, mode {cfg.anneal_mode})\n")
header = f"{'step':>8} {'loss':>8} {'unweighted':>10} {'anneal':>8} {'lr':>8} {'tok/s':>8}"
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
        while True:
            noise = torch.rand(x_0.shape[0], 1, device=device).clamp(min=cfg.t_min, max=1.0)
            x_t, mask = q_sample(x_0, noise, mask_id)
            if int(mask.sum()) > 0:
                break

        attention_mask, ratio, width = build_attn_mask(cfg, device, spec, step)

        logits = model(input_ids=x_t, use_cache=False, attention_mask=attention_mask).logits
        loss, unweighted_loss = diffusion_loss_parts(logits, x_0, mask, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        running.append(loss.item())
        running_unweighted.append(unweighted_loss.item())
        step += 1

        if step == 1 or step % cfg.log_steps == 0:
            avg = sum(running[-cfg.log_steps :]) / min(len(running), cfg.log_steps)
            avg_unweighted = sum(running_unweighted[-cfg.log_steps :]) / min(len(running_unweighted), cfg.log_steps)
            el = time.time() - start

            tps = step * cfg.batch_size * cfg.seq_len / max(el, 1e-9)

            print(
                f"{step:>8} {avg:>8.4f} {avg_unweighted:>10.4f} {ratio:>8.4f} {cfg.lr * lr_at(step):>8.2e} {tps:>8.0f}",
                flush=True,
            )
            with open(log_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "step": step,
                            "loss": avg,
                            "unweighted_loss": avg_unweighted,
                            "anneal_mode": cfg.anneal_mode,
                            "anneal_ratio": ratio,
                            "width": width,
                            "lr": cfg.lr * lr_at(step),
                            "tok_per_s": tps,
                        }
                    )
                    + "\n"
                )
            running = []
            running_unweighted = []

        if cfg.eval_every > 0 and step % cfg.eval_every == 0:
            results = eval_per_t(
                model,
                eval_block_tensor,
                cfg,
                mask_id=mask_id,
                device=device,
                unigram=unigram,
            )
            with open(log_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "kind": "eval",
                            "step": step,
                            "anneal_ratio": ratio,
                            "per_t": [asdict(r) for r in results],
                        }
                    )
                    + "\n"
                )
            summary = "  ".join(f"t={r.t}:{r.ce_mean:.3f}" for r in results)
            print(f"{'eval':>8} {summary}", flush=True)

        if step % cfg.save_steps == 0 or step == cfg.steps:
            sd = model.state_dict()
            model.save_pretrained(out / "checkpoint", state_dict=sd)
            tokenizer.save_pretrained(out / "checkpoint")

            with open(out / "checkpoint" / "mask_id.txt", "w") as f:
                f.write(str(mask_id))

elapsed = time.time() - start
print(f"\ndone in {elapsed / 60:.1f} min -> {out / 'checkpoint'}")
