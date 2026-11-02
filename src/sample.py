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

from src.diffusion_view import DiffusionViewer
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
    temperature: float
    top_p: float | None
    viewer: bool
    seed: int
    noise_sample: str
    noise_amount: float


ap = argparse.ArgumentParser()
ap.add_argument("--model", help="Path to model training directory. Will check ./checkpoint for the last checkpoint.")
ap.add_argument("--steps", type=int, default=48)
ap.add_argument("--length", type=int, default=128)
ap.add_argument("--prefix", type=str, default="")
ap.add_argument("--suffix", type=str, default="")
ap.add_argument("--noise-sample", type=str, default="")
ap.add_argument("--noise-amount", type=float, default=0.25)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top_p", type=float, default=None)
ap.add_argument("--viewer", action="store_true", help="Enable the interactive diffusion viewer.")
ap.add_argument("--seed", type=int, default=0, help="Seed for sampling (only affects --top_p).")
args = Args.from_argparse(ap.parse_args())

torch.manual_seed(args.seed)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

config_path = Path(args.model) / "config.json"
cfg = Config.load_run(config_path)

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

print(f"loaded {args.model} ({spec.hf_id}) mask_id={mask_id} ({MASK_TOKEN})")


# De-masking steps
def token_tensor(text: str) -> torch.Tensor:
    tokens: BatchEncoding = tokenizer(text, add_special_tokens=False)
    ids = cast(list[int], tokens["input_ids"])
    t = torch.tensor([ids], dtype=torch.long, device=device)
    return t


x = torch.full((1, args.length), mask_id, dtype=torch.long, device=device)
# Smollm2 has bos_token: '<|endoftext|>' (0) and eos_token: '<|endoftext|>' (0)
# Smollm3 has eos_token: '<|im_end|>' (128012)
bos = token_tensor(" ")

truth: torch.Tensor | None = None

if args.noise_sample:
    truth = token_tensor(args.noise_sample)
    truth = torch.cat([bos, truth], dim=1)

    sample = truth.clone()
    noise = torch.rand_like(sample, dtype=torch.float32)
    masks = noise < args.noise_amount
    sample[masks] = mask_id

    set_length = min(args.length, sample.shape[1])
    x[:, :set_length] = sample[:, :set_length]
else:
    prefix_t = token_tensor(args.prefix)
    suffix_t = token_tensor(args.suffix)

    prefix_t = torch.cat([bos, prefix_t], dim=1)

    x[:, : prefix_t.shape[1]] = prefix_t
    x[:, -suffix_t.shape[1] :] = suffix_t

starting_masked = (x[0] == mask_id).sum().item()


viewer = DiffusionViewer(tokenizer, mask_id, cell_width=8, topk=5, truth=truth)
live = Live(("Waiting..."), auto_refresh=False)

with viewer if args.viewer else live:
    step = 0
    for step in range(args.steps + 1):
        with torch.no_grad():
            logits = cast(torch.Tensor, model(input_ids=x, use_cache=False, attention_mask=None).logits)

        logits = logits.clone()
        logits[..., mask_id] = float("-inf")

        logits = logits / args.temperature

        if args.top_p is not None:
            # sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            probs = logits.softmax(dim=-1)
            sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
            # cumulative_probs = torch.cumsum(probs, dim=-1)
            cumulative_probs = sorted_probs.cumsum(dim=-1)
            remove = (cumulative_probs - sorted_probs) > args.top_p
            remove[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(remove, 0.0)
            probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)

            flat = probs.view(-1, probs.shape[-1])
            flat = flat / flat.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            indices = torch.multinomial(flat, 1)
            norm = flat.view(*probs.shape)
            pred = indices.view(*probs.shape[:-1])
            conf = norm.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
            dist = norm
        else:
            probs = logits.softmax(dim=-1)
            conf, pred = probs.max(dim=-1)
            dist = probs

        conf_all = conf.clone()
        is_mask = x[:, 1:] == mask_id
        conf = conf[:, :-1].masked_fill(~is_mask, -float("inf"))
        score = conf + torch.rand_like(conf) * 1e-3

        # only unmask a small amount at a time
        expected_remaining_masks = max(0, int(round(starting_masked * (1.0 - step / (args.steps)))))
        num_masked = int(is_mask[0].sum().item())

        num_unmask = max(1, num_masked - expected_remaining_masks)

        if num_masked == 0:
            break

        k = min(num_masked, num_unmask)
        _, idx = score[0].topk(k)

        x_before = x[0].clone()
        x[0, idx + 1] = pred[0, idx]
        viewer.record(step=step, before=x_before, after=x[0], pred=pred[0].roll(1), conf=conf_all[0], probs=dist[0], committed=idx + 1)

        live.update((Text.from_ansi(print_token_diff(x_before, x[0], tokenizer))))
        live.refresh()

    viewer.finish(x[0])
    viewer.browse()
    live.update((Text.from_ansi(print_token_diff(x[0], x[0], tokenizer))))
    live.refresh()


print(f"Completed after {step + 1} steps")
