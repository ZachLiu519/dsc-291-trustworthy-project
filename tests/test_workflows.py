import json

import pandas as pd

from jbb_repro.judges import LLAMA_GUARD_3_MODEL
from jbb_repro.workflows import (
    extract_qualitative_examples,
    score_heuristic,
    score_llamaguard,
    score_openai_judge,
    score_openai_refusal_judge,
)


class FakeLocalJudge:
    def __init__(self, model_name, batch_size):
        self.model_name = model_name
        self.batch_size = batch_size

    def classify_responses(self, prompts, responses):
        return [True for _ in responses]


def test_score_heuristic_writes_summary(tmp_path) -> None:
    responses_path = tmp_path / "responses.jsonl"
    responses_path.write_text(
        json.dumps(
            {
                "model_name": "vicuna",
                "defense": "none",
                "method": "PAIR",
                "behavior": "b",
                "goal": "g",
                "category": "c",
                "prompt": "p",
                "response": "I cannot help with that request.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = score_heuristic(responses_path)

    assert result.summary_path.exists()
    assert result.summary.iloc[0]["attack_success_rate"] == 0.0


def test_extract_qualitative_examples_writes_markdown(tmp_path) -> None:
    scored_path = tmp_path / "scored.csv"
    pd.DataFrame(
        [
            {
                "model_name": "vicuna",
                "method": "PAIR",
                "defense": "none",
                "category": "Privacy",
                "prompt": "attack",
                "response": "unsafe details",
                "jailbroken": True,
                "heuristic_jailbroken": False,
            }
        ]
    ).to_csv(scored_path, index=False)

    output_path = tmp_path / "examples.md"
    extract_qualitative_examples(scored_path, output_path)

    text = output_path.read_text(encoding="utf-8")
    assert "# Qualitative JailbreakBench Examples" in text
    assert "Successful Jailbreak" in text


def test_score_openai_judge_writes_scored_outputs(tmp_path) -> None:
    responses_path = tmp_path / "responses.jsonl"
    responses_path.write_text(
        json.dumps(
            {
                "model_name": "target-model",
                "defense": "none",
                "method": "PAIR",
                "behavior": "b",
                "goal": "g",
                "category": "c",
                "prompt": "harmful prompt",
                "response": "unsafe response",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = score_openai_judge(
        responses_path,
        output_dir=tmp_path / "api_judge",
        judge_model="gpt-4o-mini",
        api_key="test-key",
        completion_fn=lambda messages: '{"jailbroken": true}',
    )

    assert result.scored_path.name == "responses_openai_judge_scored.csv"
    assert result.summary_path.name == "asr_summary_openai_judge.csv"
    assert result.summary.iloc[0]["attack_success_rate"] == 1.0
    scored = pd.read_csv(result.scored_path)
    assert scored["judge"].tolist() == ["openai"]
    assert scored["judge_model"].tolist() == ["gpt-4o-mini"]


def test_score_openai_refusal_judge_writes_refusal_outputs(tmp_path) -> None:
    responses_path = tmp_path / "responses.jsonl"
    responses_path.write_text(
        json.dumps(
            {
                "model_name": "target-model",
                "defense": "none",
                "method": "BENIGN",
                "behavior": "b",
                "goal": "g",
                "category": "c",
                "prompt": "benign prompt",
                "response": "I cannot help with that.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = score_openai_refusal_judge(
        responses_path,
        output_dir=tmp_path / "api_refusal_judge",
        judge_model="gpt-4o-mini",
        api_key="test-key",
        completion_fn=lambda messages: '{"refused": true}',
    )

    assert result.scored_path.name == "responses_openai_refusal_judge_scored.csv"
    assert result.summary_path.name == "refusal_summary_openai_judge.csv"
    assert result.summary.iloc[0]["refusal_rate"] == 1.0
    scored = pd.read_csv(result.scored_path)
    assert scored["refused"].tolist() == [True]
    assert scored["judge"].tolist() == ["openai_refusal"]
    assert scored["judge_model"].tolist() == ["gpt-4o-mini"]


def test_score_llamaguard3_uses_distinct_output_files(tmp_path) -> None:
    responses_path = tmp_path / "responses.jsonl"
    responses_path.write_text(
        json.dumps(
            {
                "model_name": "target-model",
                "defense": "none",
                "method": "PAIR",
                "behavior": "b",
                "goal": "g",
                "category": "c",
                "prompt": "harmful prompt",
                "response": "unsafe response",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = score_llamaguard(
        responses_path,
        model=LLAMA_GUARD_3_MODEL,
        judge_cls=FakeLocalJudge,
    )

    assert result.scored_path.name == "responses_llamaguard3_scored.csv"
    assert result.summary_path.name == "asr_summary_llamaguard3.csv"
    scored = pd.read_csv(result.scored_path)
    assert scored["judge"].tolist() == ["llama_guard_3"]
    assert scored["judge_model"].tolist() == [LLAMA_GUARD_3_MODEL]
