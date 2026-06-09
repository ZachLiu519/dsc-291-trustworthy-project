from jbb_repro.artifacts import AttackPrompt
from jbb_repro.generation import GenerationResult, batched, format_vicuna_prompt, rows_from_results


def test_format_vicuna_prompt_wraps_user_prompt_in_chat_template() -> None:
    prompt = format_vicuna_prompt("Explain safe password hygiene.")

    assert prompt.startswith("A chat between a curious user")
    assert "USER: Explain safe password hygiene." in prompt
    assert prompt.endswith("ASSISTANT:")


def test_batched_splits_items_without_dropping_remainder() -> None:
    assert list(batched([1, 2, 3, 4, 5], batch_size=2)) == [[1, 2], [3, 4], [5]]


def test_rows_from_results_keeps_prompt_metadata() -> None:
    attack_prompt = AttackPrompt(
        method="PAIR",
        behavior="Phishing",
        goal="goal",
        category="Privacy",
        prompt="attack prompt",
    )
    result = GenerationResult(prompt=attack_prompt, response="model response")

    rows = rows_from_results("lmsys/vicuna-13b-v1.5", [result])

    assert rows == [
        {
            "model_name": "lmsys/vicuna-13b-v1.5",
            "defense": "none",
            "method": "PAIR",
            "behavior": "Phishing",
            "goal": "goal",
            "category": "Privacy",
            "prompt": "attack prompt",
            "response": "model response",
        }
    ]


def test_rows_from_results_records_defense_name() -> None:
    attack_prompt = AttackPrompt(
        method="PAIR",
        behavior="Phishing",
        goal="goal",
        category="Privacy",
        prompt="attack prompt",
    )
    result = GenerationResult(prompt=attack_prompt, response="model response")

    rows = rows_from_results("lmsys/vicuna-13b-v1.5", [result], defense_name="dictionary_filter")

    assert rows[0]["defense"] == "dictionary_filter"
