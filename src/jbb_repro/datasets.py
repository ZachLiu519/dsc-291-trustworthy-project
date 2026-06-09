from __future__ import annotations

import jailbreakbench as jbb
import pandas as pd

from jbb_repro.sampling import stratified_sample


def load_jbb_behaviors(split: str) -> pd.DataFrame:
    dataset = jbb.read_dataset("benign" if split == "benign" else "harmful")
    return dataset.as_dataframe()


def load_sampled_behaviors(split: str, sample_size: int, seed: int) -> pd.DataFrame:
    behaviors = load_jbb_behaviors(split)
    return stratified_sample(behaviors, sample_size=sample_size, seed=seed, category_column="Category")
