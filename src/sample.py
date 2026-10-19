import argparse
from typing import cast
from typed_argparse import TypedArgs
from pathlib import Path
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel

from src.bidirectional import make_bidirectional
from src.config import Config
from src.mask import MASK_TOKEN
from src.models import MODELS, ModelSpec


class Args(TypedArgs):
    model: str
    steps: int
    length: int


ap = argparse.ArgumentParser()
ap.add_argument("--model", help="Path to model training directory. Will check ./checkpoint for the last checkpoint.")
ap.add_argument("--steps", type=int, default=48)
ap.add_argument("--length", type=int, default=128)
args = Args.from_argparse(ap.parse_args())

device = torch.device("mps")

config_path = Path(args.model) / "config.json"
cfg = Config.load(config_path)

spec = MODELS[cfg.spec]

trained_path = Path(args.model) / "checkpoint"

tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(trained_path))
model = AutoModelForCausalLM.from_pretrained(trained_path, dtype=torch.float32)
model.to(device)

mask_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)

# model, tokenizer, mask_id = add_mask_token(model, tokenizer)

make_bidirectional(model)
model.config.use_cache = False

model.eval()

print(f"loaded {args.model} ({spec.hf_id}) mask_id={mask_id}\n")

prompt = tokenizer.encode("Hello, my name is ", return_tensors="pt").to(device)

logits = model(input_ids=prompt, use_cache=False).logits

logits = logits.clone()
logits[..., mask_id] = float("-inf")

probs = logits.softmax(dim=-1)
conf, pred = probs.max(dim=-1)

_, idx = conf[0].topk(1)

text = tokenizer.decode(pred[0, idx], skip_special_tokens=True)
print(text.strip())
