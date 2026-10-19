This project is an attempt to convert a regular autoregressive large language model into a diffusion language model that can do infill and then block by block diffusion generation.

I don't necessarily have performance metrics in mind, but I think I'll start by measuring the performance of a couple local LLMs to see how fast they run, and then I want to try to at least beat that generation speed. It will be hard to get a good judgment on performance in terms of intelligence, and I'm not sure modifying to a diffusion will make much of an impact on intelligence because they come from the same base model. So speed and learning are my only goals.

## Resources:

I have done a cursory search for resources and found these:

- https://arxiv.org/html/2410.17891v2 | Scaling Diffusion Language Models via Adaptation from Autoregressive Models
- https://arxiv.org/pdf/2406.07524 | Simple and Effective
  Masked Diffusion Language Models
- https://arxiv.org/pdf/2406.04329 | Simplified and Generalized
  Masked Diffusion for Discrete Data
- https://arxiv.org/pdf/2502.09992 | Large Language Diffusion Models
- https://huggingface.co/blog/ProCreations/diffusion-language-model | Diffusion Language Models: The New Paradigm
- https://neurips.cc/virtual/2025/loc/san-diego/poster/117288 | Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models
- https://huggingface.co/papers/2512.15713 | DiffusionVL: Translating Any Autoregressive Models into Diffusion Vision Language Models
- https://arxiv.org/pdf/2310.16834 | Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution

## Models I want to try adapting:

- https://huggingface.co/HuggingFaceTB/SmolLM2-135M | SmolLM2-135M
- https://huggingface.co/HuggingFaceTB/SmolLM3-3B | SmolLM3-3B
- https://huggingface.co/google/gemma-4-E2B-it | GEMMA-4-E2B-it
- https://huggingface.co/openai/gpt-oss-20b | GPT-OSS-20B

* I know nothing about these model architectures yet. Gemma might not be possible because it has vision & audio input.

## General plan

I need `[MASK]` tokens to be replaced with the correct text. We run it on a full sequence and have it predict the all the masked tokens.

1. **Add a `[MASK]` token.** We add noise to corrupt text by replacing tokens with `[MASK]`. All these tokenizers should have unused tokens we can use.
2. **Let attention look right.** Remove the causal mask so every position can see the whole block.
3. **Fine-tune on a mask-and-recover objective** (the MDLM ELBO).

The removal of the causal mask needs to be done gradually. We anneal the right-context width from 0 to the full sequence length.

Position `i` may attend to `j` iff `j <= i + w`:

```
w = 0          w = 2              w >= L-1
  1 . . .       1 1 . .         1 1 1 1
  1 1 . .       1 1 1 .         1 1 1 1
  1 1 1 .       1 1 1 1         1 1 1 1
   causal       local + future    fully bidirectional
```

## Files:

- `src/models.py`: Defines the models I want to try.
- `src/train.py`: Training code.
- `src/benchmark.py`: Simple get tps of a model.
- `src/util.py`: Utility functions.
- `src/config.py`: Configuration.
- `src/data.py`: Data loading.
- `src/mask.py`: Mask token setup.
- `src/bidirectional.py`: Bidirectional attention.

## Config

Configs are in `configs/*.yaml`.

## Running

```bash
python -m src.train --config configs/smollm2-135m.yaml
```

```bash
python -m src.sample --model runs/smollm3-3b --length 50 --prefix "In a galaxy far, far, away" --suffix " The End." --steps 35

python -m src.sample --model runs/smollm3-3b --length 25 --prefix "Water boils at" --suffix " at sea level." --steps 48

python -m src.sample --model runs/smollm3-3b --length 96 --prefix "The history of the printing press begins with" --suffix " and ends, for now, with the internet." --steps 256
```

## Training models

### SmolLM2-135M

...

### SmolLM3-3B

Using `bfloat16` because otherwise it take >64 GB of memory to train. Using `float16` caused NaN loss for some reason.

### GEMMA-4-E2B-it

...

### GPT-OSS-20B

...
