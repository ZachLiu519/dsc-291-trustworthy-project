from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    artifact_model_name: str | None = None
    dtype: str = "half"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    quantization: str | None = None
    trust_remote_code: bool = False


@dataclass(frozen=True)
class DatasetConfig:
    split: str = "harmful"
    sample_size: int = 30
    seed: int = 291


@dataclass(frozen=True)
class ArtifactConfig:
    methods: list[str] = field(default_factory=lambda: ["PAIR", "GCG"])


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    batch_size: int = 8


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    model: ModelConfig
    dataset: DatasetConfig
    artifacts: ArtifactConfig
    generation: GenerationConfig
    output_dir: Path


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    required_sections = ["experiment_name", "model", "dataset", "artifacts", "generation", "output_dir"]
    missing = [section for section in required_sections if section not in raw]
    if missing:
        raise ValueError(f"Configuration missing required section(s): {', '.join(missing)}")

    return ExperimentConfig(
        experiment_name=str(raw["experiment_name"]),
        model=ModelConfig(**_section(raw, "model")),
        dataset=DatasetConfig(**_section(raw, "dataset")),
        artifacts=ArtifactConfig(**_section(raw, "artifacts")),
        generation=GenerationConfig(**_section(raw, "generation")),
        output_dir=Path(raw["output_dir"]).expanduser(),
    )


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Configuration section '{name}' must be a mapping")
    return section
