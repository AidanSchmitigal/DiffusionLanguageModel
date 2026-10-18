"""
Train a model to convert it from causal attention to masked diffusion

Steps:
1. Attention annealing.
2. Diffusion scoring
"""

from typing import cast
import json
from pathlib import Path
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel

from src.models import MODELS
from src.mask import add_mask_token

# Training config

cfg = {
    "seed": 42,
    "spec": "smollm2-135m",
    "out_dir": "runs/smollm2-135m",
}

spec = MODELS[cfg["spec"]]

out = Path(cfg["out_dir"])
out.mkdir(exist_ok=True, parents=True)

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
