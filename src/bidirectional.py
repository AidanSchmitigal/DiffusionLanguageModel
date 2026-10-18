from typing import cast
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

    pass
