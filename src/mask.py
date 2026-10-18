import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def add_mask_token(model: nn.Module, tokenizer: PreTrainedTokenizerBase) -> tuple[nn.Module, PreTrainedTokenizerBase, int]:
    """
    Adds a [MASK] token to the tokenizer and set its embedding to the average or something
    """

    print(len(tokenizer))

    return model, tokenizer, 0
