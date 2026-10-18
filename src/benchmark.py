from torch._C import device as deviceType


from typing import cast
from yachalk import chalk
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel

from src.models import MODELS, ModelSpec

device = torch.device("mps")


def benchmark_model(spec: ModelSpec, device: deviceType):
    hf_id = spec.hf_id

    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(hf_id))
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.bfloat16)
    _ = model.to(device)

    def generate(prompt: str, max_length: int):
        model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

        start = time.time()
        generated_ids: torch.LongTensor = model.generate(**model_inputs, max_length=max_length)
        end = time.time()
        num_tokens = len(generated_ids[0])
        duration = end - start
        tps = num_tokens / duration
        print(f"Time taken: {duration:.2f}s ({num_tokens} tokens at {tps:.2f} tokens/s)")

    print(f"Benchmarking {spec.short_name}")

    generate("Hello, my name is ", 30)
    generate("A great way to start a conversation is ", 100)
    generate("There a few ways to do this, but the most common is ", 300)


benchmark_model(MODELS["smollm2-135m"], device)
