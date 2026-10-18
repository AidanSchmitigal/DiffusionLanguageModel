from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

import torch


def build_token_data(tokenizer: PreTrainedTokenizerBase):

    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

    print(dataset[0]["text"])

    return dataset[0]["text"]


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
