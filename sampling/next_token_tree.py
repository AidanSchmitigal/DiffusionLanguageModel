from __future__ import annotations

import argparse
import json
import os
from typing import cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from src.models import MODELS


def expand_batch(
    model: torch.nn.Module,
    contexts: list[list[int]],
    top_k: int,
    device: torch.device,
) -> list[list[tuple[int, float]]]:
    """Return, for each context, the top-k (token_id, prob) continuations."""
    ids = torch.tensor(contexts, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[:, -1, :]
    probs = logits.float().softmax(dim=-1)
    top = probs.topk(top_k, dim=-1)
    return [
        list(zip(top.indices[i].tolist(), top.values[i].tolist()))
        for i in range(ids.shape[0])
    ]


def build_tree(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: list[int],
    top_k: int,
    depth: int,
    device: torch.device,
) -> dict:
    root: dict = {
        "token_id": None,
        "token": None,
        "prob": 1.0,
        "cum_prob": 1.0,
        "path": "",
        "chain": [],
        "children": [],
    }
    frontier: list[tuple[dict, list[int]]] = [(root, list(prompt_ids))]
    levels: list[int] = []

    for _level in range(depth):
        levels.append(len(frontier))
        contexts = [ctx for _, ctx in frontier]
        results = expand_batch(model, contexts, top_k, device)
        new_frontier: list[tuple[dict, list[int]]] = []
        for (parent, ctx), cands in zip(frontier, results):
            for tid, p in cands:
                text = tokenizer.decode([tid])
                child: dict = {
                    "token_id": tid,
                    "token": text,
                    "prob": p,
                    "cum_prob": parent["cum_prob"] * p,
                    "path": parent["path"] + text,
                    "chain": parent["chain"]
                    + [{"token_id": tid, "token": text, "prob": p}],
                    "children": [],
                }
                parent["children"].append(child)
                new_frontier.append((child, ctx + [tid]))
        frontier = new_frontier

    root["levels"] = levels
    return root


def flatten(node: dict) -> list[dict]:
    out = [node]
    for child in node["children"]:
        out.extend(flatten(child))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-k next-token tree for a causal LM")
    parser.add_argument("--model", default="smollm2-135m", choices=sorted(MODELS))
    parser.add_argument("--text", default="Water boils at")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--out", default="runs/next_token_tree.json")
    parser.add_argument("--print-depth", type=int, default=2, help="levels to print to stdout")
    args = parser.parse_args()

    spec = MODELS[args.model]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(spec.hf_id))
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, dtype=spec.dtype)
    model.to(device)
    model.eval()

    prompt_ids = tokenizer(args.text, add_special_tokens=False)["input_ids"]
    tree = build_tree(model, tokenizer, prompt_ids, args.top_k, args.depth, device)

    all_nodes = flatten(tree)
    leaves = [n for n in all_nodes if not n["children"]]
    best = sorted(leaves, key=lambda n: -n["cum_prob"])[:10]

    doc = {
        "model": spec.short_name,
        "hf_id": spec.hf_id,
        "prompt": args.text,
        "prompt_tokens": [
            {"token_id": i, "token": tokenizer.decode([i])} for i in prompt_ids
        ],
        "top_k": args.top_k,
        "depth": args.depth,
        "nodes_per_level": tree["levels"],
        "total_nodes": len(all_nodes),
        "best_leaves_by_cum_prob": [
            {
                "path": n["path"],
                "cum_prob": n["cum_prob"],
                "chain": n["chain"],
            }
            for n in best
        ],
        "tree": tree,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)

    print(f"prompt: {args.text!r}")
    print(f"nodes per level: {tree['levels']}  (total {len(all_nodes)} nodes)")
    print("\ntop-10 leaves by cumulative probability:")
    for i, n in enumerate(best, 1):
        print(f"{i:>2}  {n['cum_prob']:.6f}  {n['path']!r}")

    print(f"\ntree (first {args.print_depth} levels):")

    def show(node: dict, depth: int) -> None:
        if depth > args.print_depth:
            return
        for child in node["children"]:
            print(
                f"{'  ' * depth}{child['token']!r}  p={child['prob']:.6f}  "
                f"cum={child['cum_prob']:.6f}"
            )
            show(child, depth + 1)

    show(tree, 1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
