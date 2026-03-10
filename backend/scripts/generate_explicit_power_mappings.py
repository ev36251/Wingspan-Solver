"""Generate explicit per-bird power mappings for all current cards.

Writes backend/powers/explicit_mappings.json with one explicit entry per bird.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.powers.registry import (
    clear_cache,
    get_power,
)


def _encode_value(value):
    if isinstance(value, Enum):
        return {
            "__enum__": f"{value.__class__.__module__}:{value.__class__.__name__}:{value.name}",
        }
    if isinstance(value, set):
        return {"__set__": [_encode_value(v) for v in sorted(value, key=repr)]}
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_value(v) for v in value]}
    if isinstance(value, list):
        return [_encode_value(v) for v in value]
    if isinstance(value, dict):
        return {"__dict__": [[_encode_value(k), _encode_value(v)] for k, v in value.items()]}
    return value


def main() -> None:
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()

    names = sorted(b.name for b in birds.all_birds)

    out: dict[str, dict] = {}
    for name in names:
        bird = birds.get(name)
        power = get_power(bird)
        out[name] = {
            "class": f"{power.__class__.__module__}.{power.__class__.__name__}",
            "kwargs": _encode_value(dict(getattr(power, "__dict__", {}))),
        }

    path = Path(__file__).resolve().parent.parent / "powers" / "explicit_mappings.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(out)} mappings to {path}")


if __name__ == "__main__":
    main()
