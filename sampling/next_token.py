from __future__ import annotations

import argparse
from typing import cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from src.models import MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description="Next-token prediction for a causal LM")
    parser.add_argument("--model", default="smollm2-135m", choices=sorted(MODELS))
    parser.add_argument("--text", default="Water boils at")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--steps", type=int, default=1, help="number of greedy steps to roll out")
    args = parser.parse_args()

    spec = MODELS[args.model]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(spec.hf_id))
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, dtype=spec.dtype)
    model.to(device)
    model.eval()

    inputs = tokenizer(args.text, return_tensors="pt").to(device)
    ids = inputs["input_ids"][0]
    print(f"prompt: {args.text!r}")
    print("tokens:", [tokenizer.decode([i]) for i in ids.tolist()])

    chosen: list[int] = []
    for step in range(1, args.steps + 1):
        step_ids = torch.cat([ids, torch.tensor(chosen, dtype=ids.dtype, device=ids.device)])
        with torch.no_grad():
            logits = model(input_ids=step_ids.view(1, -1)).logits[0, -1]

        probs = logits.float().softmax(dim=-1)
        top = probs.topk(args.top_k)
        print(f"\nstep {step} (context: {tokenizer.decode(step_ids.tolist())!r})")
        print(f"top {args.top_k} next-token probabilities:")
        for rank, (p, tid) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
            mark = "  <-" if rank == 1 else ""
            print(f"{rank:>2}  {tid:>7}  {p:.6f}  {tokenizer.decode([tid])!r}{mark}")

        chosen.append(top.indices[0].item())

    print(f"\ngreedy continuation: {args.text}{tokenizer.decode(chosen)!r}")


if __name__ == "__main__":
    main()
