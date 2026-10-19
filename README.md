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

**16,000 steps, 65.5M tokens, 7.5h on M1 Max**

~2,400 tok/s. Uses fineweb-edu

Training loss fell **9.76 -> 4.27**.

| steps | 1-1k  | 1-2k  | 2-3k  | 3-4k  | 4-6k  | 6-8k  | 8-10k | 10-12k | 12-14k | 14-16k |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ------ | ------ | ------ |
| loss  | 5.427 | 4.588 | 4.541 | 4.504 | 4.517 | 4.502 | 4.469 | 4.442  | 4.417  | 4.407  |

The curve bends into a plateau by step 2,000 and the last 12,000 steps move the block mean from 4.504 to 4.407 (2%). Didn't learn much. I left it overnight for nothing.

average cross-entropy loss by noise level:

| t          | mean CE | share of the reported loss |
| ---------- | ------- | -------------------------- |
| 0 - 0.1    | 2.18    | 5%                         |
| 0.1 - 0.25 | 2.63    | 9%                         |
| 0.25 - 0.5 | 3.58    | 21%                        |
| 0.5 - 1.0  | 5.70    | 65%                        |

86% of the loss comes from `t > 0.25`, where most tokens are masked and the best the model can do is roughly the unigram distribution. It is trying to guess a token out of a 0 prior noisy distrobution.

Comparing against the _causal_ SmolLM2-135M base on held-out blocks (last 5% of the stream, never trained on). The AR model's next-token CE is **2.934 nats**.

| t    | AR base CE | diffusion CE | ppl  | vs AR     |
| ---- | ---------- | ------------ | ---- | --------- |
| 0.01 | 2.934      | 2.200        | 9.0  | **-0.73** |
| 0.02 | 2.934      | 2.079        | 8.0  | **-0.86** |
| 0.05 | 2.934      | 2.179        | 8.8  | **-0.76** |
| 0.10 | 2.934      | 2.358        | 10.6 | **-0.58** |
| 0.20 | 2.934      | 2.693        | 14.8 | **-0.24** |
| 0.30 | 2.934      | 3.224        | 25.1 | +0.29     |
| 0.50 | 2.934      | 4.176        | 65.1 | +1.24     |
| 1.00 | 2.934      | 7.545        | 1891 | +4.61     |

The diffusion model beats the AR model at every `t < ~0.25`, which is exactly what it should do, because it can see context on both sides. It crosses over around `t = 0.25` and degrades from there.

For reference, one of my early training runs that only went for 1,500 steps was _worse than AR at every `t`_: 4.267 at `t=0.05` vs 2.179 now.

```
prefix: Water boils at          suffix: at sea level.
temp 1.0, length 40, --steps 48 (33 masked tokens -> 34 iterations):

  [prefix] 273 degrees Fahrenheit, but it is not at sea level. The temperature
  is the amount of the water at sea. The temperature is the amount of
  water[suffix]
```

```
prefix: The history of the printing press begins with
suffix: and ends, for now, with the internet.
temp 1.0, length 96, --steps 256 (78 masked tokens -> 79 iterations):

  [prefix] the invention of the printing press in 1520. It was the first
  mass-scale printing press, and it was the first mass-scale printing press to
  be used.\nThe printing press is a printing device that uses a press to make
  a copy of a printed page. It is used to make printed books, printed
  newspapers, and other printed materials. The printing press,[suffix]
```

Grammatically coherent, but factually wrong, which is what you would expect from a 135M model. The failure mode is repetition.

```
prefix: In a galaxy far, far, away          suffix: The End.
temp 1.0, length 80, --steps 60 (69 masked tokens -> 61 iterations):

  [prefix], the rest of the universe is expanding.\nThe universe is
  expanding.\nThe universe is expanding.\nThe universe is expanding.\nThe
  universe is expanding.\nThe universe is expanding.\nThe universe is
  expanding.\nThe universe is expanding.\nThe universe is expanding.\nThe
  universe is expanding.\nThe universe is expanding.[suffix]
```

### SmolLM3-3B

Using `bfloat16` because otherwise it take >64 GB of memory to train. Using `float16` caused NaN loss for some reason.

**1500 steps, 3.1M tokens, 8.5h on M1 Max**

~2,400 tok/s. Uses fineweb-edu

Training loss fell **9.62 -> 3.87**.

| step       | 25   | 100  | 225  | 300  | 500  | 750  | 1000 | 1250 | 1500     |
| ---------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | -------- |
| SmolLM3-3B | 9.62 | 5.10 | 4.82 | 4.64 | 4.12 | 4.06 | 3.99 | 3.93 | **3.87** |

Annealing finished at step 225. Returns are diminishing over the last 500 steps, so the curve is bending toward a plateau.

Infilling produces factually correct text! Yippee. Its better than the 135M, whose output was grammatically right but very wrong:

```
prefix: Water boils at          suffix: at sea level.
temp 1.0, length 40, --steps 48 (33 masked tokens -> 34 iterations):

  [prefix] 100 degrees Celsius at sea level.\nPressure increases with
  altitude.\nWater pressure decreases with altitude.\nWater pressure increases
  with altitude.\nWater pressure decreases with altitude as water
  boils[suffix]
```

```
prefix: The history of the printing press begins with
suffix: and ends, for now, with the internet.
temp 1.0, length 96, --steps 256 (78 masked tokens -> 79 iterations):

  [prefix] the invention of movable type printing in China.The
  history of the printing press begins with the invention of movable type
  printing in China. It was invented by Bi Sheng in 1041.The history
  of the printing press begins with the invention of movable type printing in
  China. It was invented by Bi Sheng in 1041.\nThe history of the printing
  press began here,[suffix]
```

The 3B gets the facts right where the 135M does not.

The main failure is repetition, and it gets much worse with longer gap lengths. At a fixed step count, longer fills get pretty bad: `fill 96, steps 64` (65 iterations) repeats "the invention of movable type printing in China" over and over. The sampler commits one token per iteration until the schedule forces it to commit the remainder in coarse jumps, so a low step count fills a lot of masks at once. It does pick the most confident words but its still not great. Raising the step count fixes it: the same fill at 128 steps and at 256 steps both finish in 79 iterations. It still performs best when it fills 1 token per step.

### GEMMA-4-E2B-it

...

### GPT-OSS-20B

...
