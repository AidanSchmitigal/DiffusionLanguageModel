This project is an attempt to convert a regular autoregressive large language model into a diffusion language model that can do infill and then block by block diffusion generation.

I don't necessarily have performance metrics in mind, but I think I'll start by measuring the performance of a couple local LLMs to see how fast they run, and then I want to try to at least beat that generation speed. It will be hard to get a good judgment on performance in terms of intelligence, and I'm not sure modifying to a diffusion will make much of an impact on intelligence because they come from the same base model. So speed and learning are my only goals.

## Resources:

I have done a cursory search for resources and found these:

- https://arxiv.org/html/2410.17891v2 | Scaling Diffusion Language Models via Adaptation from Autoregressive Models
- https://arxiv.org/pdf/2406.07524 | Simple and Effective
  Masked Diffusion Language Models
- https://arxiv.org/pdf/2502.09992 | Large Language Diffusion Models
- https://huggingface.co/blog/ProCreations/diffusion-language-model | Diffusion Language Models: The New Paradigm
- https://neurips.cc/virtual/2025/loc/san-diego/poster/117288 | Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models
- https://huggingface.co/papers/2512.15713 | DiffusionVL: Translating Any Autoregressive Models into Diffusion Vision Language Models

## Models I want to try adapting:

- https://huggingface.co/HuggingFaceTB/SmolLM2-135M | SmolLM2-135M
- https://huggingface.co/HuggingFaceTB/SmolLM3-3B | SmolLM3-3B
- https://huggingface.co/google/gemma-4-E2B-it | GEMMA-4-E2B-it
- https://huggingface.co/openai/gpt-oss-20b | GPT-OSS-20B

* I know nothing about these model architectures yet. Gemma might not be possible because it has vision & audio input.
