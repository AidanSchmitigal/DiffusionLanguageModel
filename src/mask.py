from typing import cast

from torch._tensor import Tensor


from torch.nn.modules.module import Module


import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

MASK_TOKEN = "[MASK]"


def add_mask_token(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, int]:
    """
    Adds a [MASK] token to the tokenizer and set its embedding to the average or something
    """

    tokenizer.add_special_tokens({"additional_special_tokens": [MASK_TOKEN]})
    model.resize_token_embeddings(len(tokenizer))

    print(f"tokenizer.special_tokens_map: {tokenizer.special_tokens_map}")

    # print(f"tokenizer.mask_token_id: {tokenizer.mask_token_id} / {len(tokenizer)}")

    # initizlize the mask token to the average
    mask_id = int(tokenizer.convert_tokens_to_ids(MASK_TOKEN))
    tokenizer.mask_token_id = mask_id
    print(f"mask_id: {mask_id}")

    with torch.no_grad():
        weight: Tensor = cast(Tensor, model.get_input_embeddings().weight)
        # print(f"weight.shape: {weight.shape}")
        emb = weight[:-1].mean(dim=0)
        weight[-1].copy_(emb)

    # TODO: gemini says I need to set the LM head here but it depends on the model. Check back if it doesn't work

    return model, tokenizer, mask_id
