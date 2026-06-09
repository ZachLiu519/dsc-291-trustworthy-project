from pathlib import Path

import pandas as pd
import pytest

from jbb_repro.charts import PAPER_VICUNA_ASR, generate_all_charts


def test_generate_all_charts_writes_expected_files(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    repo_root = tmp_path / "repo"
    _write_summary(repo_root / "outputs/vicuna_vllm_jbb_subset/asr_summary_llamaguard.csv", [("PAIR", 0.68), ("GCG", 0.86)])
    _write_summary(repo_root / "outputs/vicuna_vllm_jbb_subset/asr_summary_llamaguard3.csv", [("PAIR", 0.92), ("GCG", 0.90)])
    _write_summary(repo_root / "outputs/vicuna_vllm_jbb_subset/asr_summary.csv", [("PAIR", 0.88), ("GCG", 0.90)])
    _write_summary(repo_root / "outputs/gpt4o_mini_jbb_subset/asr_summary_llamaguard.csv", [("PAIR", 0.28), ("GCG", 0.17)])
    _write_summary(repo_root / "outputs/gpt4o_mini_jbb_subset/asr_summary_llamaguard3.csv", [("PAIR", 0.32), ("GCG", 0.17)])
    _write_summary(repo_root / "outputs/gpt4o_mini_jbb_subset/asr_summary.csv", [("PAIR", 0.40), ("GCG", 0.21)])
    _write_private_judge_summary(repo_root / "reports/private_model_time_sensitivity_summary.csv")
    _write_refusal(repo_root / "outputs/vicuna_benign_vllm_jbb_subset/refusal_summary.csv", 0.0)
    _write_refusal(repo_root / "outputs/vicuna_benign_vllm_jbb_subset/refusal_summary_openai_judge.csv", 0.03)
    _write_summary(
        repo_root / "outputs/vicuna_benign_vllm_jbb_subset/asr_summary_llamaguard.csv",
        [("BENIGN", 0.1)],
    )
    _write_summary(
        repo_root / "outputs/vicuna_benign_vllm_jbb_subset/asr_summary_llamaguard3.csv",
        [("BENIGN", 0.0)],
    )
    _write_defense_summary(
        repo_root / "outputs/vicuna_dictionary_filter_vllm_jbb_subset/asr_summary_llamaguard.csv",
        [("GCG", 0.21), ("PAIR", 0.60)],
    )
    _write_defense_summary(
        repo_root / "outputs/vicuna_dictionary_filter_vllm_jbb_subset/asr_summary_llamaguard3.csv",
        [("GCG", 0.17), ("PAIR", 0.84)],
    )
    _write_defense_summary(
        repo_root / "outputs/vicuna_dictionary_filter_vllm_jbb_subset/asr_summary.csv",
        [("GCG", 0.21), ("PAIR", 0.84)],
    )
    _write_scored(repo_root / "outputs/vicuna_vllm_jbb_subset/responses_llamaguard_scored.csv")
    _write_scored(repo_root / "outputs/vicuna_vllm_jbb_subset/responses_llamaguard3_scored.csv")
    _write_scored(repo_root / "outputs/gpt4o_mini_jbb_subset/responses_llamaguard_scored.csv")
    _write_scored(repo_root / "outputs/gpt4o_mini_jbb_subset/responses_llamaguard3_scored.csv")

    output_dir = repo_root / "reports" / "figures"
    written = generate_all_charts(repo_root, output_dir)

    assert len(written) == 8
    assert all(path.exists() for path in written)
    assert PAPER_VICUNA_ASR["PAIR"] == 0.69


def _write_summary(path: Path, rows: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"method": method, "model_name": "test-model", "n": 1, "attack_success_rate": asr}
            for method, asr in rows
        ]
    ).to_csv(path, index=False)


def _write_defense_summary(path: Path, rows: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "method": method,
                "model_name": "test-model",
                "defense": "dictionary_filter",
                "n": 1,
                "attack_success_rate": asr,
            }
            for method, asr in rows
        ]
    ).to_csv(path, index=False)


def _write_refusal(path: Path, rate: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"method": "BENIGN", "model_name": "test-model", "defense": "none", "n": 1, "refusal_rate": rate}]
    ).to_csv(path, index=False)


def _write_private_judge_summary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "run_id": "test",
                "date": "2026-06-08",
                "time_local": "07:30",
                "target_response_set": target,
                "method": method,
                "n": 1,
                "attack_success_rate": asr,
            }
            for target, rows in {
                "vicuna_harmful": [("PAIR", 0.92), ("GCG", 0.90)],
                "gpt4o_mini_harmful": [("PAIR", 0.36), ("GCG", 0.14)],
            }.items()
            for method, asr in rows
        ]
    ).to_csv(path, index=False)


def _write_scored(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "prompt": "prompt-a",
                "response": "response-a",
                "jailbroken": True,
                "heuristic_jailbroken": False,
            },
            {
                "prompt": "prompt-b",
                "response": "response-b",
                "jailbroken": False,
                "heuristic_jailbroken": False,
            },
        ]
    ).to_csv(path, index=False)
