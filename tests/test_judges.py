import pandas as pd

from jbb_repro.api_judges import (
    OpenAIJailbreakJudge,
    OpenAIRefusalJudge,
    parse_openai_judge_label,
    parse_openai_refusal_label,
)
from jbb_repro.judges import (
    LLAMA_GUARD_2_MODEL,
    LLAMA_GUARD_3_MODEL,
    apply_jailbreak_judge,
    configure_decoder_tokenizer,
    llamaguard_judge_name,
    llamaguard_output_stem,
    parse_llamaguard_label,
)


class FakeJudge:
    def classify_responses(self, prompts, responses):
        assert prompts == ["harmful goal", "another goal"]
        assert responses == ["unsafe details", "I cannot help."]
        return [True, False]


def test_apply_jailbreak_judge_adds_primary_judge_columns() -> None:
    frame = pd.DataFrame(
        {
            "prompt": ["harmful goal", "another goal"],
            "response": ["unsafe details", "I cannot help."],
        }
    )

    scored = apply_jailbreak_judge(frame, FakeJudge(), judge_name="llama_guard_2", judge_model="test-model")

    assert scored["jailbroken"].tolist() == [True, False]
    assert scored["judge"].tolist() == ["llama_guard_2", "llama_guard_2"]
    assert scored["judge_model"].tolist() == ["test-model", "test-model"]


def test_parse_llamaguard_label_detects_unsafe_prefix() -> None:
    assert parse_llamaguard_label("unsafe\nS2") is True
    assert parse_llamaguard_label("safe") is False


def test_llamaguard_metadata_distinguishes_model_versions() -> None:
    assert LLAMA_GUARD_3_MODEL == "meta-llama/Llama-Guard-3-8B"
    assert llamaguard_judge_name(LLAMA_GUARD_2_MODEL) == "llama_guard_2"
    assert llamaguard_judge_name(LLAMA_GUARD_3_MODEL) == "llama_guard_3"
    assert llamaguard_output_stem(LLAMA_GUARD_2_MODEL) == "llamaguard"
    assert llamaguard_output_stem(LLAMA_GUARD_3_MODEL) == "llamaguard3"


def test_configure_decoder_tokenizer_uses_left_padding_and_eos_pad() -> None:
    class FakeTokenizer:
        pad_token = None
        eos_token = "</s>"
        padding_side = "right"

    tokenizer = FakeTokenizer()

    configure_decoder_tokenizer(tokenizer)

    assert tokenizer.pad_token == "</s>"
    assert tokenizer.padding_side == "left"


def test_parse_openai_judge_label_accepts_json_and_plain_labels() -> None:
    assert parse_openai_judge_label('{"jailbroken": true}') is True
    assert parse_openai_judge_label('{"jailbroken": false, "reason": "refusal"}') is False
    assert parse_openai_judge_label("JAILBROKEN") is True
    assert parse_openai_judge_label("not_jailbroken") is False


def test_openai_jailbreak_judge_classifies_with_injected_completion() -> None:
    calls = []

    def complete(messages):
        calls.append(messages)
        return '{"jailbroken": true}'

    judge = OpenAIJailbreakJudge(model_name="gpt-4o-mini", api_key="test-key", completion_fn=complete)

    assert judge.classify_responses(["harmful prompt"], ["unsafe response"]) == [True]
    assert "harmful prompt" in calls[0][-1]["content"]
    assert "unsafe response" in calls[0][-1]["content"]


def test_parse_openai_refusal_label_accepts_json_and_plain_labels() -> None:
    assert parse_openai_refusal_label('{"refused": true}') is True
    assert parse_openai_refusal_label('{"refused": false, "reason": "answered"}') is False
    assert parse_openai_refusal_label("REFUSED") is True
    assert parse_openai_refusal_label("answered") is False


def test_openai_refusal_judge_classifies_with_injected_completion() -> None:
    calls = []

    def complete(messages):
        calls.append(messages)
        return '{"refused": true}'

    judge = OpenAIRefusalJudge(model_name="gpt-4o-mini", api_key="test-key", completion_fn=complete)

    assert judge.classify_responses(["benign prompt"], ["I cannot help with that."]) == [True]
    assert "benign prompt" in calls[0][-1]["content"]
    assert "I cannot help with that." in calls[0][-1]["content"]
