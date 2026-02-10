"""Replay a specific rules-certification game/step deterministically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.models.enums import BoardType
from backend.rules.certification import run_conformance_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay certification case by game index")
    parser.add_argument("--suite-seed", type=int, required=True, help="Seed used for run_conformance_suite")
    parser.add_argument("--game-index", type=int, required=True, help="1-based game index from report")
    parser.add_argument("--players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--board-type", default="oceania", choices=["base", "oceania"])
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--out", default="", help="Optional output JSON path")
    args = parser.parse_args()

    # Re-run deterministically up to the target game only.
    result = run_conformance_suite(
        games=args.game_index,
        players=args.players,
        board_type=BoardType(args.board_type),
        max_steps=args.max_steps,
        seed=args.suite_seed,
    )

    hints = result.get("move_generation_mismatch_hints", [])
    game_hints = [h for h in hints if int(h.get("game_index", -1)) == args.game_index]
    payload = {
        "meta": {
            "suite_seed": args.suite_seed,
            "game_index": args.game_index,
            "players": args.players,
            "board_type": args.board_type,
            "max_steps": args.max_steps,
        },
        "game_hints": game_hints,
        "issues": [i for i in result.get("issues", []) if int(i.get("game_index", -1)) == args.game_index],
    }

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {p}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
