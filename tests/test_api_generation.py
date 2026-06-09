from jbb_repro.api_generation import OpenAIResult, rows_from_openai_results
from jbb_repro.artifacts import AttackPrompt


def test_rows_from_openai_results_keeps_metadata_and_token_counts() -> None:
    prompt = AttackPrompt(
        method="PAIR",
        behavior="Phishing",
        goal="goal",
        category="Privacy",
        prompt="attack prompt",
    )
    result = OpenAIResult(prompt=prompt, response="model response", prompt_tokens=3, completion_tokens=4)

    rows = rows_from_openai_results("gpt-4o-mini", [result])

    assert rows == [
        {
            "model_name": "gpt-4o-mini",
            "defense": "none",
            "method": "PAIR",
            "behavior": "Phishing",
            "goal": "goal",
            "category": "Privacy",
            "prompt": "attack prompt",
            "response": "model response",
            "prompt_tokens": 3,
            "completion_tokens": 4,
        }
    ]
