import pandas as pd

from jbb_repro.metrics import add_refusal_flags, summarize_attack_success, summarize_refusal_rate


def test_add_refusal_flags_detects_common_refusal_language() -> None:
    frame = pd.DataFrame(
        {
            "response": [
                "I cannot help with that request.",
                "Sure, here is a high-level benign explanation.",
            ]
        }
    )

    flagged = add_refusal_flags(frame)

    assert flagged["refused"].tolist() == [True, False]


def test_summarize_attack_success_groups_by_method_and_model() -> None:
    frame = pd.DataFrame(
        {
            "method": ["PAIR", "PAIR", "GCG", "GCG"],
            "model_name": ["vicuna", "vicuna", "vicuna", "vicuna"],
            "jailbroken": [True, False, True, True],
        }
    )

    summary = summarize_attack_success(frame)

    assert summary.to_dict("records") == [
        {"method": "GCG", "model_name": "vicuna", "n": 2, "attack_success_rate": 1.0},
        {"method": "PAIR", "model_name": "vicuna", "n": 2, "attack_success_rate": 0.5},
    ]


def test_summarize_attack_success_includes_defense_when_present() -> None:
    frame = pd.DataFrame(
        {
            "method": ["PAIR", "PAIR"],
            "model_name": ["vicuna", "vicuna"],
            "defense": ["none", "dictionary_filter"],
            "jailbroken": [True, False],
        }
    )

    summary = summarize_attack_success(frame)

    assert summary.to_dict("records") == [
        {
            "method": "PAIR",
            "model_name": "vicuna",
            "defense": "dictionary_filter",
            "n": 1,
            "attack_success_rate": 0.0,
        },
        {"method": "PAIR", "model_name": "vicuna", "defense": "none", "n": 1, "attack_success_rate": 1.0},
    ]


def test_summarize_refusal_rate_groups_benign_outputs() -> None:
    frame = pd.DataFrame(
        {
            "method": ["BENIGN", "BENIGN", "BENIGN"],
            "model_name": ["vicuna", "vicuna", "gpt-4o-mini"],
            "refused": [True, False, False],
        }
    )

    summary = summarize_refusal_rate(frame)

    assert summary.to_dict("records") == [
        {"method": "BENIGN", "model_name": "gpt-4o-mini", "n": 1, "refusal_rate": 0.0},
        {"method": "BENIGN", "model_name": "vicuna", "n": 2, "refusal_rate": 0.5},
    ]
