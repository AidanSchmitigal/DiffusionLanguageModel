from typing import cast
import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from transformers.models.llama.modeling_llama import ALL_ATTENTION_FUNCTIONS, LlamaDecoderLayer


def make_bidirectional(model: PreTrainedModel):
    """
    remove the causal mask
    make the attention annealed?
    """

    # todo: might need to wrap sdpa to set causal to False, but could do by sending a kwarg from `model()` forward

    # print(f"model.config.attention_head_size: {model.config.attention_head_size}")

    # print(ALL_ATTENTION_FUNCTIONS.valid_keys())

    # ALL_ATTENTION_FUNCTIONS.register("my_bidirectional", dummy_attention)

    # bidirectional attention doesn't exist

    # model.set_attn_implementation("my_bidirectional")

    # print(f"model.config: {model.config}")
    print(f"model.config._attn_implementation: {model.config._attn_implementation}")

    # print(model)
    # print(model.model)
    model_layers: list[LlamaDecoderLayer] = cast(list[LlamaDecoderLayer], model.model.layers)
    # print(model_layers)
    # print(len(model_layers))

    # decoder_layer: LlamaDecoderLayer = cast(LlamaDecoderLayer, model_layers[0])
    # print(decoder_layer.self_attn)
    # print(decoder_layer.self_attn.config)
    # print(decoder_layer.self_attn.is_causal)

    for layer in model_layers:
        layer.self_attn.is_causal = False

    model.config.is_causal = False


def attention_mask_band(
    seq_len: int,
    *,
    device: torch.device,
    width: float,
) -> torch.Tensor | None:
    """
    Makes a boolean `True = attend` mask for `j <= i + width`.
    """
    if width >= seq_len - 1:
        return None

    if width <= 0:
        return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril(diagonal=0)

    keep = int(max(0, math.floor(width)))
    thick_diagonal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril(diagonal=keep)

    randoms = torch.rand_like(thick_diagonal, dtype=torch.float32)
    sets = randoms < width / seq_len
    thick_diagonal[sets] = True

    return thick_diagonal
