from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path | None = None) -> None:
    if path is not None:
        candidates = [Path(path)]
    else:
        candidates = [
            Path("/content/.env"),
            Path(".env"),
            Path.cwd() / ".env",
        ]

    for env_path in candidates:
        if env_path.exists():
            _load_from_file(env_path)
            return


def _load_from_file(env_path: Path) -> None:
    with env_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", maxsplit=1)
            os.environ.setdefault(key.strip(), _clean_value(value.strip()))


def _clean_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
