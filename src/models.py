from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ModelSpec:
    short_name: str
    hf_id: str
    dtype: torch.dtype


MODELS: dict[str, ModelSpec] = {
    "smollm2-135m": ModelSpec(
        short_name="smollm2-135m",
        hf_id="HuggingFaceTB/SmolLM2-135M",
        dtype=torch.float32,
    ),
    "smollm3-3b": ModelSpec(
        short_name="smollm3-3b",
        hf_id="HuggingFaceTB/SmolLM3-3B",
        dtype=torch.float32,
    ),
    "gpt-oss-20b": ModelSpec(
        short_name="gpt-oss-20b",
        hf_id="openai/gpt-oss-20b",
        dtype=torch.bfloat16,
    ),
}
