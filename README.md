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

The removal of the causal mask needs to be done gradually. We anneal the right-context width from 0 to the full sequence length. Plus a bit of randomeness that increases over the whole annealing duration.

Position `i` may attend to `j` iff `j <= i + w`:

```
ratio = 0          ratio = 0.5                 ratio = 1
  1 . . . . .       1 1 . . 1 .             1 1 1 1 1 1
  1 1 . . . .       1 1 1 . . 1             1 1 1 1 1 1
  1 1 1 . . .       1 1 1 1 . .             1 1 1 1 1 1
   causal          causal + random cells      fully bidirectional
```

## Files:

- `src/models.py`: Defines the models I want to try.
- `src/train.py`: Training code.
- `src/sample.py`: Sampling from a checkpoint.
- `src/eval_per_t.py`: Per-noise-level evaluation.
- `src/diffusion_view.py`: Interactive terminal viewer for the denoising steps.
- `src/benchmark.py`: Simple get tps of a model.
- `src/util.py`: Utility functions.
- `src/config.py`: Configuration.
- `src/data.py`: Data loading.
- `src/mask.py`: Mask token setup.
- `src/bidirectional.py`: Bidirectional attention and the annealing masks.

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

### SmolLM2-135M main

**40,000 steps, 163.8M tokens, 19.5h on an M1 Max.**

~2,300 tok/s. Uses fineweb-edu

Training loss fell **14.96 -> 9.31**.

| steps | 0-2k  | 2-4k | 4-6k | 6-8k | 8-10k | 10-12k | 12-14k | 14-16k | 16-18k | 18-20k | 20-22k | 22-24k |
| ----- | ----- | ---- | ---- | ---- | ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| loss  | 10.24 | 9.95 | 9.73 | 9.41 | 9.32  | 9.31   | 9.36   | 9.33   | 9.37   | 9.35   | 9.34   | 9.31   |

| step     | t=0.01 | t=0.05 | t=0.1 | t=0.2 | t=0.5 | t=1.0 |
| -------- | ------ | ------ | ----- | ----- | ----- | ----- |
| 0 (ckpt) | 2.084  | 2.953  | 2.871 | 2.937 | 4.270 | 7.605 |
| 200      | 3.087  | 2.843  | 2.700 | 3.018 | 4.249 | 7.602 |
| 400      | 2.068  | 2.455  | 2.516 | 3.044 | 4.266 | 7.590 |
| 600      | 2.141  | 2.681  | 2.784 | 2.941 | 4.213 | 7.596 |

Trained with the band attention anneal.
The anneal completed at step 2,000

It is better than the causal base model (2.936 nats next-token CE) while fewer than about 20% of tokens are hidden. At higher noise there is too little visible context
for a 135M model to reconstruct the sequence. The `t=1` result is only 0.094
nats worse than the no-context unigram floor, which is the expected saturated
end of the curve.

| masked fraction `t` | CE (nats) | perplexity | vs. AR base |
| ------------------- | --------: | ---------: | ----------: |
| 1%                  |     2.313 |       10.1 |      -0.623 |
| 5%                  |     2.464 |       11.7 |      -0.472 |
| 10%                 |     2.631 |       13.9 |      -0.305 |
| 20%                 |     2.956 |       19.2 |      +0.020 |
| 50%                 |     4.303 |       73.9 |      +1.367 |
| 80%                 |     6.033 |      417.1 |      +3.097 |
| 100%                |     7.559 |     1917.7 |      +4.623 |

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
