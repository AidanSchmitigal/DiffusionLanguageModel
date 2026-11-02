from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
import yaml
from pathlib import Path


@dataclass
class Config:
    # --- model ---
    spec: str = "smollm2-135m"

    # --- data ---
    dataset: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str | None = "sample-10BT"
    seq_len: int = 256
    max_docs: int = 120_000
    batch_size: int = 16

    # --- optimisation ---
    steps: int = 20_000
    lr: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    t_min: float = 1e-3

    # --- attention annealing ---
    anneal_steps: int = 2_000
    # "paper": Just use the random unmasked
    # "band":  Use the band and random numasks
    anneal_mode: str = "paper"

    # --- evaluation ---
    # Number of token blocks held out from the end of the data
    eval_blocks: int = 512
    eval_every: int = 1_000
    eval_batches: int = 60

    # --- runtime ---
    log_steps: int = 25
    save_steps: int = 500
    sample_steps: int = 1_000
    out_dir: str = "runs/default"
    seed: int = 0

    @classmethod
    def load(cls, path: str | Path) -> Config:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, str]) -> "Config":
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")

        cfg = cls(**raw)

        return cfg

    @classmethod
    def load_run(cls, path: str | Path) -> "Config":
        """
        Load a saved run's config.json.
        """
        with open(path) as f:
            raw = json.load(f)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})
