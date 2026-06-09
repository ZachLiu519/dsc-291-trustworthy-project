from __future__ import annotations

import pandas as pd


def stratified_sample(
    frame: pd.DataFrame,
    sample_size: int,
    seed: int,
    category_column: str = "Category",
) -> pd.DataFrame:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if category_column not in frame.columns:
        raise ValueError(f"Missing category column: {category_column}")
    if sample_size >= len(frame):
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    grouped = list(frame.groupby(category_column, sort=True))
    selected = []
    remaining_slots = sample_size

    for category_index, (_, group) in enumerate(grouped):
        groups_left = len(grouped) - category_index
        max_for_group = min(len(group), remaining_slots - (groups_left - 1))
        quota = max(1, sample_size // len(grouped))
        n_for_group = min(max_for_group, quota)
        selected.append(group.sample(n=n_for_group, random_state=seed + category_index))
        remaining_slots -= n_for_group

    sampled = pd.concat(selected)
    if len(sampled) < sample_size:
        remainder = frame.drop(index=sampled.index)
        sampled = pd.concat(
            [
                sampled,
                remainder.sample(n=sample_size - len(sampled), random_state=seed + len(grouped)),
            ]
        )

    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
