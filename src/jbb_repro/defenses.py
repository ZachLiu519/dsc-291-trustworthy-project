from __future__ import annotations

import re
from functools import lru_cache

from wordfreq import iter_wordlist


def dictionary_filter_prompt(prompt: str) -> str:
    dictionary = _english_words()
    tokens = re.findall(r"\w+|[^\w\s]", prompt)
    kept = [token for token in tokens if not re.match(r"\w+", token) or token.lower() in dictionary]
    return _join_tokens(kept)


@lru_cache(maxsize=1)
def _english_words() -> frozenset[str]:
    return frozenset(iter_wordlist("en", wordlist="best"))


def _join_tokens(tokens: list[str]) -> str:
    text = " ".join(tokens)
    replacements = {
        " ,": ",",
        " .": ".",
        " !": "!",
        " ?": "?",
        " ;": ";",
        " :": ":",
        " ' s": "'s",
        "' s": "'s",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text
