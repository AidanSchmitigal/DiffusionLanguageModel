from typing import cast

from datasets import load_dataset
from transformers import BatchEncoding, PreTrainedTokenizerBase

import torch
from pathlib import Path
import os

CACHE_DIR = Path("data")
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
CONFIG_NAME = "sample-10BT"


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
        text = row["text"]

        if len(text) > 200:
            texts.append(text)

        if len(texts) > 200:
            break

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


def q_sample(x_0: torch.Tensor, noise: torch.Tensor, mask_id: int):
    """
    x_0: batch, seq, dim
    noise: batch, 1
    """

    rand = torch.rand_like(x_0, dtype=torch.float32)
    mask = rand < noise
    print(rand)
    print(mask)

    empty = ~mask.any(dim=-1, keepdim=True)
    print(empty)
