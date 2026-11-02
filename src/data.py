from typing import cast

from datasets import load_dataset
from transformers import BatchEncoding, PreTrainedTokenizerBase

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
import os
import random

from src.config import Config

CACHE_DIR = Path("data")


def build_token_stream(tokenizer: PreTrainedTokenizerBase, cfg: Config):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{cfg.dataset.replace('/', '_')}_{cfg.dataset_config}"
    cache = CACHE_DIR / f"stream_{tag}.pt"

    if cache.exists():
        print(f"Loading cached stream from {cache}")
        return torch.load(cache)

    dataset = load_dataset(cfg.dataset, name=cfg.dataset_config, split="train", streaming=True)

    texts = []

    for row in dataset:
        if len(texts) >= cfg.max_docs:
            break

        text = row["text"]

        if len(text) > 200:
            texts.append(text)

        if len(texts) % 20000 == 0:
            print(f"read {len(texts)} docs...")

    print(f"Tokenizing {len(texts)} texts...")

    encoded = cast(BatchEncoding, tokenizer(texts, add_special_tokens=False)["input_ids"])

    bos = cast(int | list[int], tokenizer.bos_token_id)
    bos_ids: list[int] = [bos] if isinstance(bos, int) else bos
    bos_t = torch.tensor(bos_ids, dtype=torch.long)

    chunks = []
    for ids in encoded:
        chunks.append(torch.tensor(ids, dtype=torch.long))
        chunks.append(bos_t)

    stream = torch.cat(chunks)
    print(f"  data: {len(texts)} docs, {len(stream / 1e6):.1f}M tokens")

    # cahce
    torch.save(stream, cache)

    return stream


class PackedTokenDataset(Dataset[torch.Tensor]):
    """
    Chunks the 1d token stream into blocks of size `seq_len`
    """

    n_blocks: int
    seq_len: int
    blocks: torch.Tensor  # (n_blocks, seq_len)

    def __init__(self, stream: torch.Tensor, seq_len: int):
        n_blocks = len(stream) // seq_len

        self.n_blocks = n_blocks
        self.seq_len = seq_len
        self.blocks = stream[: n_blocks * seq_len].view(n_blocks, seq_len)

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.blocks[idx]


def q_sample(
    x_0: torch.Tensor,
    noise: torch.Tensor,
    mask_id: int,
    *,
    generator: torch.Generator | None = None,
):
    """
    forward process of noising the data

    The paper (2410.17891) uses
    `q(x_t|x_0) = alpha_t*x_0 + (1-alpha_t)*m`
    with alpha_t = 1-t,
    and allows a sample to end up with zero masked tokens

    x_0: (B, T_seqlen) clean tokens
    noise: (B, 1) per-example noise level t
    """

    rand = torch.rand(x_0.shape, dtype=torch.float32, generator=generator).to(x_0.device)

    mask = rand < noise

    # # put one mask in every sample
    # min_indices = rand.argmin(dim=1, keepdim=True)
    # mask = mask.scatter(dim=1, index=min_indices, value=True)

    x_t = x_0.clone()
    x_t[mask] = mask_id

    return x_t, mask


def masked_ce_parts(
    logits: torch.Tensor,
    x_0: torch.Tensor,
    mask: torch.Tensor,
):
    """
    Cross-entropy on masked positions only
    shift (logits[:, i] predicts x_0[:, i+1])

    logits: (B, T, V) model output.
    x_0: (B, T) clean tokens.
    mask: (B, T) bool, True at masked positions.
    t: (B, 1) noise level used to build `mask`.

    Returns (ce_sum_per_sample, n_masked_per_sample)
    ce_sum_per_sample is the sum of CE over that sample's masked positions.
    """

    shift_logits = logits[:, :-1, :]
    shift_x_0 = x_0[:, 1:]
    shift_mask = mask[:, 1:]

    B, T, V = shift_logits.shape

    ce = F.cross_entropy(
        shift_logits.reshape(-1, V),
        shift_x_0.reshape(-1),
        reduction="none",
    ).view(B, T)

    masked_ce = ce.masked_fill(~shift_mask, 0.0)

    return masked_ce.sum(dim=-1), shift_mask.sum(dim=-1)


def diffusion_loss(
    logits: torch.Tensor,
    x_0: torch.Tensor,
    mask: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """
    Weighted cross-entropy on masked positions only.
    The loss is a sum over masked positions of CE/t, normalised by the total
    number of masked tokens.

    logits: (B, T, V) model output.
    x_0: (B, T) clean tokens.
    mask: (B, T) bool, True at masked positions.
    t: (B, 1) noise level used to build `mask`.
    """

    ce_sum, n_masked = masked_ce_parts(logits, x_0, mask)

    return (ce_sum.unsqueeze(-1) / noise).sum() / n_masked.sum()


def diffusion_loss_parts(
    logits: torch.Tensor,
    x_0: torch.Tensor,
    mask: torch.Tensor,
    noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (optimized_loss, unweighted_loss).

    `optimized_loss` is the reference objective (aka `diffusion_loss`).
    `unweighted_loss` is the same quantity without the 1/t weighting, i.e.
    the mean cross entropy per masked token.
    """

    ce_sum, n_masked = masked_ce_parts(logits, x_0, mask)

    total = n_masked.sum()
    optimized = (ce_sum.unsqueeze(-1) / noise).sum() / total
    unweighted = ce_sum.sum() / total

    return optimized, unweighted


def anneal_ratio(step: int, cfg: Config) -> float:
    if cfg.anneal_steps <= 0:
        return 1.0

    return min(1.0, (step + 1) / cfg.anneal_steps)


def anneal_width(step: int, cfg: Config) -> float:
    full = cfg.seq_len - 1
    if cfg.anneal_steps <= 0:
        return full

    frac = min(1.0, step / cfg.anneal_steps)
    width = full * frac

    return width
