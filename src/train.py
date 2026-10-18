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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel
from torch.utils.data import DataLoader

from src.models import MODELS
from src.mask import add_mask_token
from src.bidirectional import make_bidirectional
from src.data import build_token_data, q_sample

# Training config

cfg = {
    "seed": 42,
    "spec": "smollm2-135m",
    "out_dir": "runs/smollm2-135m",
    "batch_size": 16,
    "steps": 20_000,
    "anneal_steps": 1_000,
}

spec = MODELS[cfg["spec"]]

out = Path(cfg["out_dir"])
out.mkdir(exist_ok=True, parents=True)
log_path = out / "train_log.jsonl"

with open(out / "config.json", "w") as f:
    json.dump(cfg, f, indent=2, default=str)

random.seed(cfg["seed"])
torch.manual_seed(cfg["seed"])

device = torch.device("mps")
print(f"Using device: {device}, dtype: {spec.dtype}")

#  Load model

tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(spec.hf_id))
model = AutoModelForCausalLM.from_pretrained(spec.hf_id, dtype=spec.dtype)
model.to(device)

# Mask Token setup
mask_token_id = tokenizer.mask_token_id

model, tokenizer, mask_id = add_mask_token(model, tokenizer)

make_bidirectional(model)

model.config.use_cache = False

n_params = sum(p.numel() for p in model.parameters())
print(f"base: {spec.hf_id}, params: {n_params / 1e6:.1f}M, mask_id: {mask_id}")

# Training

dataset = build_token_data(
    tokenizer,
)
dataloader = DataLoader(
    dataset,
    batch_size=cfg["batch_size"],
    # shuffle=True,
    drop_last=True,
    num_workers=0,
    pin_memory=False,
)
# print(f"data: {len(dataloader) / 1e6:.1f}M")


# Training Loop

print(f"\ntraining for {cfg["steps"]} steps (anneal over {cfg["anneal_steps"]})\n")
header = f"{'step':>7} {'loss':>8} {'lr':>9}"
print(header)
print("-" * len(header))

model.train()
done = False
while not done:
    for batch in dataloader:
        print(batch)
        x_0: torch.Tensor = cast(torch.Tensor, batch.to(device))

        # noise
        noise = torch.randn(x_0.shape[0], 1, device=device).clamp(min=1e-3)
        x_t, mask = q_sample(x_0, noise, mask_id)

        # loss = model(**x_0).loss
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # model.optimizer.step()
        # model.optimizer.zero_grad()

        # if model.training:
        #     print(
        #         f"{model.global_step:>7} {loss.item():>8.2f} {model.nat_loss.item():>8.2f} {model.w.item():>7.2f} {model.alignment_loss.item():>7.2f} {model.lr:>9.2e} {model.token_loss.item():>8.2f}"
        #     )

        break
