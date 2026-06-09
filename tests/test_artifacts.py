import pandas as pd

from jbb_repro.artifacts import build_attack_prompts


class FakeJailbreak:
    def __init__(self, behavior: str, prompt: str | None) -> None:
        self.behavior = behavior
        self.prompt = prompt


def test_build_attack_prompts_keeps_sample_order_and_drops_missing_prompts() -> None:
    sample = pd.DataFrame(
        {
            "Behavior": ["Phishing", "Malware", "Fraud"],
            "Goal": ["goal-1", "goal-2", "goal-3"],
            "Category": ["privacy", "cyber", "fraud"],
        }
    )
    artifact_jailbreaks = [
        FakeJailbreak("Fraud", "fraud prompt"),
        FakeJailbreak("Phishing", "phishing prompt"),
        FakeJailbreak("Malware", None),
    ]

    prompts = build_attack_prompts("PAIR", sample, artifact_jailbreaks)

    assert [prompt.behavior for prompt in prompts] == ["Phishing", "Fraud"]
    assert [prompt.prompt for prompt in prompts] == ["phishing prompt", "fraud prompt"]
    assert all(prompt.method == "PAIR" for prompt in prompts)
