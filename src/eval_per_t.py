"""
Per-noise-level evaluation.

The training loss is a single number averaged over a random noise level `t`,
which makes it a poor progress signal:

  * it is dominated by the saturated high-`t` region (at t=1 the model is at
    the no-context information-theoretic floor), and
  * its divisor is the realised number of masked tokens in the batch, which
    is random, so most of the batch-to-batch variance is measurement noise.

Evals the cross entropy at each fixed noise level, on blocks that are held out,
using enough batches that the measurement noise hopefully is small enough compared
to the measurement.

python -m src.eval_per_t --run runs/smollm2-135m
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from src.bidirectional import make_bidirectional
from src.config import Config
from src.data import PackedTokenDataset, build_token_stream, q_sample
from src.mask import MASK_TOKEN
from src.models import MODELS

EVAL_NOISE_LEVELS = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0]


@dataclass
class PerTResult:
    t: float
    ce_mean: float  # mean CE over masked positions
    ce_std: float  # std of the per-batch means
    loss: float  # (1/t) * sum(CE) / n_masked - the optimized objective at this t
    unweighted: float  # sum(CE) / n_masked - mean CE per masked token
    unigram: float  # no-context unigram floor at this t
    n_masked: int  # masked tokens evaluated


def unigram_ce(blocks: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Per-token-id cross entropy of an add-one smoothed unigram model fit to
    `blocks`. This is a floor where there is no context at all. So no model can beat this.
    """

    counts = torch.bincount(blocks.reshape(-1).cpu(), minlength=vocab_size).double()
    total = counts.sum()
    probs = (counts + 1.0) / (total + vocab_size)

    return (-probs.log()).float()


@torch.no_grad()
def eval_per_t(
    model: torch.nn.Module,
    blocks: torch.Tensor,
    cfg: Config,
    *,
    mask_id: int,
    device: torch.device,
    unigram: torch.Tensor | None = None,
) -> list[PerTResult]:
    """
    Evaluate cross entropy at each noise level in `t_grid`.

    `blocks` is a (N, T) tensor of held-out token blocks.
    """

    was_training = model.training
    model.eval()

    results: list[PerTResult] = []
    n_blocks = blocks.shape[0]

    for t in EVAL_NOISE_LEVELS:
        gen = torch.Generator().manual_seed(cfg.seed + int(t * 1_000_000))

        ce_means: list[float] = []
        losses: list[float] = []
        unweighted_losses: list[float] = []

        total_ce = 0.0
        total_unigram = 0.0
        total_masked = 0

        for b in range(cfg.eval_batches):
            start = (b * cfg.batch_size) % max(1, n_blocks - cfg.batch_size)
            x_0 = blocks[start : start + cfg.batch_size].to(device)
            if x_0.shape[0] == 0:
                break

            noise = torch.full((x_0.shape[0], 1), t, device=device)
            x_t, mask = q_sample(x_0, noise, mask_id, generator=gen)

            logits = model(input_ids=x_t, use_cache=False, attention_mask=None).logits

            B, T, V = logits.shape
            shift_logits = logits[:, :-1, :]
            shift_x_0 = x_0[:, 1:]
            shift_mask = mask[:, 1:]

            ce = F.cross_entropy(
                shift_logits.reshape(-1, V),
                shift_x_0.reshape(-1),
                reduction="none",
            ).view(B, T - 1)

            n_masked = int(shift_mask.sum())
            if n_masked == 0:
                continue

            s = float(ce.masked_fill(~shift_mask, 0.0).sum())

            ce_means.append(s / n_masked)
            losses.append((1.0 / t) * s / n_masked)
            unweighted_losses.append(s / n_masked)

            total_ce += s
            total_masked += n_masked

            if unigram is not None:
                uni = unigram.to(device)[shift_x_0]
                total_unigram += float((uni * shift_mask).sum())

        if not ce_means:
            continue

        mean = sum(ce_means) / len(ce_means)
        var = sum((x - mean) ** 2 for x in ce_means) / max(1, len(ce_means) - 1)

        results.append(
            PerTResult(
                t=t,
                ce_mean=mean,
                ce_std=var**0.5,
                loss=sum(losses) / len(losses),
                unweighted=sum(unweighted_losses) / len(unweighted_losses),
                unigram=(total_unigram / total_masked) if unigram is not None else float("nan"),
                n_masked=total_masked,
            )
        )

    if was_training:
        model.train()

    return results


def eval_blocks(dataset: PackedTokenDataset, n: int) -> torch.Tensor:
    """
    The last `n` blocks of the dataset
    """

    n = max(1, min(n, len(dataset)))

    return dataset.blocks[-n:]


def print_results(results: list[PerTResult]) -> None:
    print(f"{'t':>6} {'ce':>8} {'+-':>7} {'loss(1/t)':>10} {'unweighted':>11} {'unigram':>8} {'gap':>7}")
    print("-" * 68)
    for r in results:
        gap = r.ce_mean - r.unigram
        print(f"{r.t:>6.2f} {r.ce_mean:>8.4f} {r.ce_std:>7.4f} {r.loss:>10.3f} " f"{r.unweighted:>11.4f} {r.unigram:>8.4f} {gap:>+7.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-noise-level evaluation of a trained diffusion LM")
    ap.add_argument("--run", required=True, help="Run directory (contains config.json and checkpoint/)")
    ap.add_argument("--checkpoint", default=None, help="Override the checkpoint path")
    ap.add_argument("--out", default=None, help="Write results as JSON to this path")
    args = ap.parse_args()

    run = Path(args.run)
    cfg = Config.load_run(run / "config.jso n")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    spec = MODELS[cfg.spec]

    ckpt = Path(args.checkpoint) if args.checkpoint else run / "checkpoint"

    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(ckpt))
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=spec.dtype)
    model.to(device)

    mask_id = int(tokenizer.convert_tokens_to_ids(MASK_TOKEN))

    make_bidirectional(model)
    model.config.use_cache = False

    stream = build_token_stream(tokenizer, cfg)
    dataset = PackedTokenDataset(stream, seq_len=cfg.seq_len)
    blocks = eval_blocks(dataset, args.eval_blocks if args.eval_blocks is not None else cfg.eval_blocks)

    print(f"run: {run}  checkpoint: {ckpt}  mask_id: {mask_id}")
    print(f"eval blocks: {blocks.shape[0]} (held out from training)")

    unigram = unigram_ce(blocks, model.config.vocab_size)

    results = eval_per_t(
        model,
        blocks,
        cfg,
        mask_id=mask_id,
        device=device,
        unigram=unigram,
    )

    print()
    print_results(results)

    if args.out:
        payload = {"run": str(run), "checkpoint": str(ckpt), "results": [asdict(r) for r in results]}
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
