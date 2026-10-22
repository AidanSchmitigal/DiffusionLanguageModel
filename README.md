This project is an attempt to convert a regular autoregressive large language model into a diffusion language model that can do infill and then block by block diffusion generation.

I don't necessarily have performance metrics in mind, but I think I'll start by measuring the performance of a couple local LLMs to see how fast they run, and then I want to try to at least beat that generation speed. It will be hard to get a good judgment on performance in terms of intelligence, and I'm not sure modifying to a diffusion will make much of an impact on intelligence because they come from the same base model. So speed and learning are my only goals.

## Resources:

I have done a cursory search for resources and found these:

- https://arxiv.org/html/2410.17891v2 | Scaling Diffusion Language Models via Adaptation from Autoregressive Models (most referenced paper)
- https://arxiv.org/html/2509.26328 | Fast-dLLM v2: Efficient Block-Diffusion LLM (unused so far)
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
- `src/sample.py`: Sampling from a checkpoint.
- `src/diffusion_view.py`: Interactive terminal viewer for the denoising steps.
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

### Viewer

`--viewer` opens an interactive trace of the denoising process. Every token is a
stack of rows:

```
│  the   │  row 1: the token currently in the sequence
│  cat   │  row 2: what the model predicts for that position at this step
│ ██████ │  row 3: confidence of that prediction
│  the   │  row 4: the original sample token (green = match, red = mismatch)
```

Row 4 is only shown when there is a ground truth to compare against, i.e. when
`--noise-sample` is used. The line under the header then reports how much of
the original sample the model reproduced (`match 90.9%  20/22 correct`).

```bash
python -m src.sample --model runs/smollm3-3b --viewer \
  --noise-sample "The history of the printing press begins with Gutenberg in Mainz and ends, for now, with the internet." \
  --noise-amount 0.3 --length 48 --steps 24
```

Keys: `←/→` previous/next step, `↑/↓` move the cursor one row, `,`/`.` one
token, `f` jump to the next token filled on this step, `g`/`G` first/last step,
`q` quit. Every step is recorded, so you can walk backwards and forwards.

## Training models

### SmolLM2-135M

**22,850 steps, 93.6M tokens, 11.0h on M1 Max**

~2,360 tok/s. Uses fineweb-edu. Target was 160,000 steps, which would have
taken ~66 more hours.

#### The plateau

Training loss went **14.96 -> 9.31**.

| steps | 0-2k  | 2-4k | 4-6k | 6-8k | 8-10k | 10-12k | 12-14k | 14-16k | 16-18k | 18-20k | 20-22k | 22-24k |
| ----- | ----- | ---- | ---- | ---- | ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| loss  | 10.24 | 9.95 | 9.73 | 9.41 | 9.32  | 9.31   | 9.36   | 9.33   | 9.37   | 9.35   | 9.34   | 9.31   |

| step     | t=0.01 | t=0.05 | t=0.1 | t=0.2 | t=0.5 | t=1.0 |
| -------- | ------ | ------ | ----- | ----- | ----- | ----- |
| 0 (ckpt) | 2.084  | 2.953  | 2.871 | 2.937 | 4.270 | 7.605 |
| 200      | 3.087  | 2.843  | 2.700 | 3.018 | 4.249 | 7.602 |
| 400      | 2.068  | 2.455  | 2.516 | 3.044 | 4.266 | 7.590 |
| 600      | 2.141  | 2.681  | 2.784 | 2.941 | 4.213 | 7.596 |

| t bucket | % of samples | % of reported loss | change in reported loss |
| -------- | ------------ | ------------------ | ----------------------- |
| 0.0-0.1  | 9.5%         | 5.4%               | -0.022                  |
| 0.1-0.25 | 14.9%        | 9.6%               | -0.032                  |
| 0.25-0.5 | 25.3%        | 21.1%              | -0.052                  |
| 0.5-1.0  | 50.2%        | 63.9%              | -0.099                  |

Comparing against the _causal_ SmolLM2-135M base on held-out blocks (last 5%
of the stream, never trained on). The AR model's next-token CE is **2.936
nats**.

| t    | AR base CE | diffusion CE | ppl  | top-1 | vs AR     |
| ---- | ---------- | ------------ | ---- | ----- | --------- |
| 0.01 | 2.936      | 2.455        | 11.7 | 56.0% | **-0.48** |
| 0.02 | 2.936      | 2.489        | 12.1 | 57.9% | **-0.45** |
| 0.05 | 2.936      | 2.576        | 13.1 | 55.3% | **-0.36** |
| 0.10 | 2.936      | 2.750        | 15.6 | 53.1% | **-0.19** |
| 0.20 | 2.936      | 3.073        | 21.6 | 48.7% | +0.14     |
| 0.30 | 2.936      | 3.479        | 32.4 | 43.6% | +0.54     |
| 0.50 | 2.936      | 4.364        | 78.6 | 32.1% | +1.43     |
| 0.70 | 2.936      | 5.489        | 242  | 19.9% | +2.55     |
| 0.90 | 2.936      | 6.735        | 841  | 8.8%  | +3.80     |
| 1.00 | 2.936      | 7.583        | 1965 | 3.7%  | +4.65     |

The diffusion model beats the AR model at every `t < ~0.15`, which is
exactly what it should do, because it can see context on both sides. It
crosses over around `t = 0.15` and degrades from there.

#### Samples

```
prefix: Water boils at          suffix: at sea level.
temp 1.0, length 40, --steps 48 (32 masked tokens -> 33 iterations):

  [prefix] the bottom of the ocean. The water is the volume
  of water in the ocean. The water is the volume of water. The water is the
  volume of water at sea level.[suffix]
```

```
prefix: The history of the printing press begins with
suffix: and ends, for now, with the internet.
temp 1.0, length 96, --steps 256 (77 masked tokens -> 78 iterations):

  [prefix] the history of the first press of the world. The first history of
  the printing press was the history of the first press. The history of the
  first press was the history of the first press. The history of the first
  press is the history of the first press. The history of the first press was
  the history of the first press. The history of the first press is the
  history, and ends, for now, with the internet.[suffix]
```

```
prefix: In a galaxy far, far, away          suffix: The End.
temp 1.0, length 80, --steps 60 (68 masked tokens -> 61 iterations):

  [prefix], in the middle of a galaxy, there is
  the end to the end from the end to the end. The end end to the end. The end
  end to the end. The end end to the end. End from the end to the end.. The
  end to the end. End to the end. End to the end. The End.[suffix]
```

The repetition failure mode is worse than the 16k run's, and the printing
press example has collapsed into a loop. Compare the 3B samples below: the
same architecture family at 3B holds facts together, the 135M does not.

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
  with altitude.\nWater pressure decreases with altitude as water boils at sea
level.[suffix]
```

```
prefix: The history of the printing press begins with
suffix: and ends, for now, with the internet.
temp 1.0, length 96, --steps 256 (78 masked tokens -> 79 iterations):

  [prefix] the invention of movable type
  printing in China.<|im_end|>The history of the printing press begins with
  the invention of movable type printing in China. It was invented by Bi Sh
  eng in 1041.<|im_end|>The history of the printing press begins with the
  invention of movable type printing in China. It was invented by Bi Sheng in
  1041.\nThe history of the printing press began here, and ends, for now, with
  the internet.[suffix]
```

```
prefix: In a galaxy far, far, away          suffix: The End.
temp 1.0, length 50, --steps 35 (39 masked tokens -> 36 iterations):

  [prefix], a long time ago, there was a young named Luke
  Skywalker. He was a good pilot and a good fighter. He was a good pilot.
  <|im_end|> was a fictional character in Star Wars: The End.[suffix]
```

The 3B gets the facts right where the 135M does not.

The main failure is repetition, and it gets much worse with longer gap lengths. At a fixed step count, longer fills get pretty bad: `fill 96, steps 64` (65 iterations) repeats "the invention of movable type printing in China" over and over. The sampler commits one token per iteration until the schedule forces it to commit the remainder in coarse jumps, so a low step count fills a lot of masks at once. It does pick the most confident words but its still not great. Raising the step count fixes it: the same fill at 128 steps and at 256 steps both finish in 79 iterations. It still performs best when it fills 1 token per step.

### GEMMA-4-E2B-it

...

### GPT-OSS-20B

...
