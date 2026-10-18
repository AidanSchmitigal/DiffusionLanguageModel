# Download and run SmolLM2-135M

from typing import cast
from yachalk import chalk
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel

device = torch.device("mps")

# Default. 30L, d=576, GQA 9/3, tied embeddings, vocab 49152.
short_name = "smollm2-135m"
hf_id = "HuggingFaceTB/SmolLM2-135M"
family = "llama"
registry_module = "transformers.models.llama.modeling_llama"
layer_path = "model.layers"
max_seq_len = 512

tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(hf_id))
model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.float32)
_ = model.to(device)

model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to(model.device)

start = time.time()
generated_ids = model.generate(**model_inputs, max_length=300)
end = time.time()
num_tokens = len(generated_ids[0])

print(f"Time taken: {end - start}s ({num_tokens} tokens at {num_tokens /  (end - start):.2f} tokens/s)")

print('"""')
for i in range(num_tokens):
    token = tokenizer.decode(generated_ids[0][i])
    if token == "\n":
        token = "\\n"
    if i % 2 == 0:
        print(chalk.bg_white.white_bright(token), end="")
    else:
        print(chalk.bg_gray.white(token), end="")
