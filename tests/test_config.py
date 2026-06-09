from pathlib import Path

import pytest

from jbb_repro.config import ExperimentConfig, load_config


def test_load_config_expands_paths_and_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experiment_name: vicuna_smoke
model:
  model_name: lmsys/vicuna-13b-v1.5
  artifact_model_name: vicuna-13b-v1.5
  dtype: half
dataset:
  split: harmful
  sample_size: 6
  seed: 17
artifacts:
  methods: [PAIR, GCG]
generation:
  max_new_tokens: 96
  temperature: 0.0
output_dir: outputs/vicuna_smoke
""".strip()
    )

    config = load_config(config_path)

    assert isinstance(config, ExperimentConfig)
    assert config.experiment_name == "vicuna_smoke"
    assert config.model.model_name == "lmsys/vicuna-13b-v1.5"
    assert config.model.artifact_model_name == "vicuna-13b-v1.5"
    assert config.model.dtype == "half"
    assert config.model.gpu_memory_utilization == 0.9
    assert config.dataset.sample_size == 6
    assert config.artifacts.methods == ["PAIR", "GCG"]
    assert config.output_dir == Path("outputs/vicuna_smoke")


def test_load_config_rejects_missing_required_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("experiment_name: bad\n")

    with pytest.raises(ValueError, match="missing required section"):
        load_config(config_path)
