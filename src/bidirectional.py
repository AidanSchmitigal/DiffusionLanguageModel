from typing import cast
from torch._tensor import Tensor
from torch.nn.modules.module import Module
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from transformers.models.llama.modeling_llama import ALL_ATTENTION_FUNCTIONS


def make_bidirectional(model: PreTrainedModel):
    """
    remove the causal mask
    make the attention annealed?
    """

    # print(f"model.config.attention_head_size: {model.config.attention_head_size}")

    print(ALL_ATTENTION_FUNCTIONS.valid_keys())

    ALL_ATTENTION_FUNCTIONS.register("my_bidirectional", dummy_attention)

    # bidirectional attention doesn't exist

    model.set_attn_implementation("my_bidirectional")

    print(f"model.config: {model.config}")
    print(f"model.config._attn_implementation: {model.config._attn_implementation}")

    pass


def dummy_attention(
    module: Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    dropout: float = 0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    softcap: float | None = None,
    is_causal: bool | None = None,
    s_aux: Tensor | None = None,
) -> tuple[Tensor, None]:

    return query, None
