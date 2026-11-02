from typing import cast
import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from transformers.models.llama.modeling_llama import ALL_ATTENTION_FUNCTIONS, LlamaDecoderLayer

from config import Config
from models import ModelSpec
from src.data import (
    PackedTokenDataset,
    anneal_ratio,
    anneal_width,
    build_token_stream,
    diffusion_loss_parts,
    q_sample,
)


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


def anneal_attn_paper(
    seq_len: int,
    *,
    device: torch.device,
    ratio: float,
) -> torch.Tensor:
    """
    The paper's implementation's attention mask annealing.

    Each cell is independently selected with probability `ratio`
    Returns a 4D additive float mask (0 = attend, finfo.min = blocked),
    broadcast over the batch.
    """

    idx = torch.arange(seq_len, device=device)
    causal = idx.unsqueeze(0) <= idx.unsqueeze(1)  # [i, j] -> j <= i

    random_open = torch.rand((seq_len, seq_len), device=device) < ratio
    attend = causal | random_open

    return attend


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


def build_attn_mask(cfg: Config, device: torch.Device, spec: ModelSpec, step: int):
    """
    make the attention mask based on the config setting
    """

    ratio = anneal_ratio(step, cfg)
    # if cfg.anneal_mode == "paper":
    #     mask = anneal_attn_paper(cfg.seq_len, device=device, ratio=ratio)
    #     width = None

    if cfg.anneal_mode == "band":
        width = anneal_width(step, cfg)
        mask = attention_mask_band(cfg.seq_len, device=device, width=width)
        if mask is None:
            return None, ratio, width

    else:
        raise ValueError(f"unknown anneal_mode {cfg.anneal_mode!r}")

    mask = torch.zeros(mask.shape, dtype=spec.dtype, device=mask.device)
    mask.masked_fill_(~mask, torch.finfo(spec.dtype).min)

    return mask.unsqueeze(0).unsqueeze(0), ratio, width
