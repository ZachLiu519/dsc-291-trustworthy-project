from types import SimpleNamespace

import pandas as pd

import jbb_repro.datasets as datasets_module
from jbb_repro.datasets import load_jbb_behaviors


def test_load_jbb_behaviors_passes_explicit_harmful_split(monkeypatch) -> None:
    calls = []

    class FakeDataset:
        def as_dataframe(self):
            return pd.DataFrame({"Behavior": ["b"], "Goal": ["g"], "Category": ["c"]})

    def fake_read_dataset(split):
        calls.append(split)
        return FakeDataset()

    monkeypatch.setattr(
        datasets_module,
        "jbb",
        SimpleNamespace(read_dataset=fake_read_dataset),
    )

    load_jbb_behaviors("harmful")

    assert calls == ["harmful"]
