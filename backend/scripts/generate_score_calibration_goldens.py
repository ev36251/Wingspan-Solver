"""Generate score-calibration golden fixtures for deterministic full games."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.models.enums import BoardType
from backend.rules.score_calibration import build_score_calibration_goldens


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate score calibration golden fixtures")
    parser.add_argument(
        "--out",
        default="backend/tests/fixtures/score_calibration_goldens.json",
    )
    parser.add_argument("--seed-start", type=int, default=20260310)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--board-type", choices=["base", "oceania"], default="oceania")
    parser.add_argument("--max-turns", type=int, default=240)
    parser.add_argument("--strict-rules-mode", action="store_true", default=True)
    args = parser.parse_args()

    seeds = [int(args.seed_start) + i for i in range(max(1, int(args.count)))]
    payload = build_score_calibration_goldens(
        seeds,
        players=int(args.players),
        board_type=BoardType(args.board_type),
        max_turns=int(args.max_turns),
        strict_rules_mode=bool(args.strict_rules_mode),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"wrote {len(payload.get('cases', []))} calibration cases "
        f"to {out_path}"
    )


if __name__ == "__main__":
    main()
