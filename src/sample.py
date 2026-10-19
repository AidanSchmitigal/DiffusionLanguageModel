import argparse
from typing import cast
from typed_argparse import TypedArgs
from pathlib import Path
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase, PreTrainedModel

from src.bidirectional import make_bidirectional
from src.config import Config
from src.mask import MASK_TOKEN
from src.models import MODELS, ModelSpec
from src.util import print_tokens


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

mask_id = int(tokenizer.convert_tokens_to_ids(MASK_TOKEN))

# model, tokenizer, mask_id = add_mask_token(model, tokenizer)

make_bidirectional(model)
model.config.use_cache = False

model.eval()

print(f"loaded {args.model} ({spec.hf_id}) mask_id={mask_id} ({MASK_TOKEN})\n")

# De-masking steps

prompt = "The history of the printing press begins with"
tokens: BatchEncoding = tokenizer(prompt, add_special_tokens=False)
ids = cast(list[int], tokens["input_ids"])
prompt_t = torch.tensor([ids], dtype=torch.long, device=device)
starting_masked = args.length - len(ids)

x = torch.full((1, args.length), mask_id, dtype=torch.long, device=device)
x[:, : prompt_t.shape[1]] = prompt_t

print("Starting", x)

step = 0
for step in range(args.steps + 1):
    logits = cast(torch.Tensor, model(input_ids=x, use_cache=False).logits)

    logits = logits.clone()
    logits[..., mask_id] = float("-inf")

    probs = logits.softmax(dim=-1)
    conf, pred = probs.max(dim=-1)

    is_mask = x == mask_id
    conf = conf.masked_fill(~is_mask, -float("inf"))

    # only unmask a small amount at a time
    expected_remaining_masks = max(0, int(round(starting_masked * (1.0 - step / (args.steps)))))
    num_masked = int(is_mask[0].sum().item())

    num_unmask = max(1, num_masked - expected_remaining_masks)

    if num_masked == 0:
        break

    k = min(num_masked, num_unmask)
    print(expected_remaining_masks, num_masked, num_unmask, k)
    _, idx = conf[0].topk(k)

    x[0, idx] = pred[0, idx]

print(f"Completed after {step} steps")

print("Ending", x)
print_tokens(x[0], tokenizer)
