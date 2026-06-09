import pandas as pd

from jbb_repro.sampling import stratified_sample


def test_stratified_sample_covers_each_category_when_possible() -> None:
    frame = pd.DataFrame(
        {
            "Behavior": [f"behavior-{idx}" for idx in range(8)],
            "Goal": [f"goal-{idx}" for idx in range(8)],
            "Category": ["privacy", "privacy", "cyber", "cyber", "fraud", "fraud", "harm", "harm"],
        }
    )

    sample = stratified_sample(frame, sample_size=4, seed=7, category_column="Category")

    assert len(sample) == 4
    assert set(sample["Category"]) == {"privacy", "cyber", "fraud", "harm"}


def test_stratified_sample_is_deterministic_and_preserves_rows() -> None:
    frame = pd.DataFrame(
        {
            "Behavior": [f"behavior-{idx}" for idx in range(12)],
            "Goal": [f"goal-{idx}" for idx in range(12)],
            "Category": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
        }
    )

    first = stratified_sample(frame, sample_size=6, seed=13, category_column="Category")
    second = stratified_sample(frame, sample_size=6, seed=13, category_column="Category")

    assert first["Behavior"].tolist() == second["Behavior"].tolist()
    assert set(first["Behavior"]).issubset(set(frame["Behavior"]))
