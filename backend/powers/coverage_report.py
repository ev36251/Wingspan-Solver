"""Power source coverage telemetry.

Reports how many birds resolve via strict/manual/parsed/fallback/no_power.
Optionally emits a strict-certification summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.powers.registry import get_power_source, is_strict_power_source_allowed, clear_cache


def build_report() -> dict:
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    counts: dict[str, int] = {}
    non_strict: list[str] = []
    for b in birds.all_birds:
        src = get_power_source(b)
        counts[src] = counts.get(src, 0) + 1
        if not is_strict_power_source_allowed(src):
            non_strict.append(b.name)
    return {
        "total_birds": len(birds.all_birds),
        "by_source": counts,
        "strict_certified_birds": len(birds.all_birds) - len(non_strict),
        "non_strict_birds": sorted(non_strict),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Power source coverage report")
    parser.add_argument("--out", default="reports/power_coverage.json")
    args = parser.parse_args()

    report = build_report()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

