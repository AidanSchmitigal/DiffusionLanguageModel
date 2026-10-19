import argparse
from typing import cast
from rich.text import Text
from typed_argparse import TypedArgs
from pathlib import Path
import json

from rich.live import Live
from rich.panel import Panel

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase, PreTrainedModel

from src.bidirectional import make_bidirectional
from src.config import Config
from src.mask import MASK_TOKEN
from src.models import MODELS, ModelSpec
from src.util import print_token_diff, print_tokens


class Args(TypedArgs):
    model: str
    steps: int
    length: int
    prefix: str
    suffix: str


ap = argparse.ArgumentParser()
ap.add_argument("--model", help="Path to model training directory. Will check ./checkpoint for the last checkpoint.")
ap.add_argument("--steps", type=int, default=48)
ap.add_argument("--length", type=int, default=128)
ap.add_argument("--prefix", type=str, default="")
ap.add_argument("--suffix", type=str, default="")
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
def token_tensor(text: str) -> torch.Tensor:
    tokens: BatchEncoding = tokenizer(text, add_special_tokens=False)
    ids = cast(list[int], tokens["input_ids"])
    t = torch.tensor([ids], dtype=torch.long, device=device)
    return t


prefix_t = token_tensor(args.prefix)
suffix_t = token_tensor(args.suffix)

x = torch.full((1, args.length), mask_id, dtype=torch.long, device=device)
x[:, : prefix_t.shape[1]] = prefix_t
x[:, -suffix_t.shape[1] :] = suffix_t

starting_masked = (x[0] == mask_id).sum().item()

# print("Starting", x)

temperature = 0.9
top_p: float | None = None

with Live(("Waiting..."), auto_refresh=False) as live:

    step = 0
    for step in range(args.steps + 1):
        logits = cast(torch.Tensor, model(input_ids=x, use_cache=False, attention_mask=None).logits)

        logits = logits.clone()
        logits[..., mask_id] = float("-inf")

        # logits = logits / temperature

        if top_p is not None:
            # sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            # probs = sorted_logits.softmax(dim=-1)
            # cumulative_probs = torch.cumsum(probs, dim=-1)
            # remove = cumulative_probs - probs > top_p

            # sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            # logits = sorted_logits.scatter(dim=-1, index=sorted_indices, src=sorted_logits)

            probs = logits.softmax(dim=-1)
            flat_probs = probs.view(-1, probs.shape[-1])
            indices = torch.multinomial(flat_probs, 1)
            pred = indices.view(*probs.shape[:-1])
            conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
        else:
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
        _, idx = conf[0].topk(k)

        x_before = x[0].clone()
        x[0, idx] = pred[0, idx]
        live.update((Text.from_ansi(print_token_diff(x_before, x[0], tokenizer))))
        live.refresh()

        # print(print_token_diff(x[0], x[0], tokenizer))
        # print_tokens(x[0], tokenizer)

    live.update((Text.from_ansi(print_token_diff(x[0], x[0], tokenizer))))
    live.refresh()

print(f"Completed after {step} steps")

# print("Ending", x)
# print_tokens(x[0], tokenizer)
