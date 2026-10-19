from typing import cast

from datasets import load_dataset
from transformers import BatchEncoding, PreTrainedTokenizerBase

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
import os
import random

CACHE_DIR = Path("data")
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
CONFIG_NAME = "sample-10BT"

max_docs = 120_000


def build_token_stream(tokenizer: PreTrainedTokenizerBase):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{DATASET_NAME.replace('/', '_')}_{CONFIG_NAME}"
    cache = CACHE_DIR / f"stream_{tag}.pt"

    if cache.exists():
        print(f"Loading cached stream from {cache}")
        return torch.load(cache)

    dataset = load_dataset(DATASET_NAME, name=CONFIG_NAME, split="train", streaming=True)

    texts = []

    for row in dataset:
        if len(texts) >= max_docs:
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


def q_sample(x_0: torch.Tensor, noise: torch.Tensor, mask_id: int):
    """
    x_0: batch, seq_len
    noise: batch, 1
    """

    # print("x_0.shape", x_0.shape)
    # print("noise.shape", noise.shape)

    rand = torch.rand_like(x_0, dtype=torch.float32)
    mask = rand < noise
    # print(rand)
    # print(mask)

    # empty = ~mask.any(dim=-1, keepdim=True)
    # print(empty)  # can have False values

    min_indices = rand.argmin(dim=1, keepdim=True)
    mask = mask.scatter(dim=1, index=min_indices, value=True)

    x_t = x_0.clone()
    x_t[mask] = mask_id

    # empty = ~mask.any(dim=-1, keepdim=True)
    # print(empty) # no false values

    return x_t, mask


def diffusion_loss(
    logits: torch.Tensor,
    x_0: torch.Tensor,
    mask: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """
    Weighted cross-entropy on masked positions only.

    logits: (B, T, V) model output.
    x_0: (B, T) clean tokens.
    mask: (B, T) bool, True at masked positions.
    t: (B, 1) noise level used to build `mask`.

    Returns the scalar loss normalised by B*T
    """
    B, T, V = logits.shape
    ce = F.cross_entropy(
        logits.reshape(-1, V),
        x_0.reshape(-1),
        reduction="none",
    ).view(B, T)

    w = mask.to(logits.dtype) / noise
    per_example = (ce * w).sum(dim=-1)

    # Examples where nothing was masked contribute nothing; drop them.
    keep = mask.any(dim=-1)
    if not keep.any():
        return logits.sum() * 0.0

    return per_example[keep].mean() / T


def anneal_width(step: int, cfg: dict) -> float:
    full = cfg["seq_len"] - 1
    if cfg["anneal_steps"] <= 0:
        return full

    frac = min(1.0, step / cfg["anneal_steps"])
    width = full * frac

    if cfg["stochastic_anneal"] and frac < 1.0:
        width = random.uniform(width / 2, width) if random.random() < 0.5 else width

    return width
