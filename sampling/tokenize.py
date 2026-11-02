import argparse
from typing import cast
from rich.text import Text
from typed_argparse import TypedArgs
from pathlib import Path
import json
from rich.live import Live
from rich.panel import Panel

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase, PreTrainedModel

from src.diffusion_view import DiffusionViewer
from src.bidirectional import make_bidirectional
from src.config import Config
from src.mask import MASK_TOKEN
from src.models import MODELS, ModelSpec
from src.util import print_token_diff, print_tokens


class Args(TypedArgs):
    config: str
    text: list[str]


ap = argparse.ArgumentParser()
ap.add_argument("--config", help="Path to model config.yaml")
ap.add_argument("text", nargs="*")
args = Args.from_argparse(ap.parse_args())

config_path = Path(args.config)
cfg = Config.load(config_path)

spec = MODELS[cfg.spec]

tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(spec.hf_id))

mask_id = int(tokenizer.convert_tokens_to_ids(MASK_TOKEN))

text = " ".join(args.text)

ids = tokenizer(text).input_ids
print(ids)
tokens = [tokenizer.decode([i]) for i in ids]
print(tokens)
