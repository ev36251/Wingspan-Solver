"""Consolidated strict KPI runner.

Runs:
1) Power coverage report
2) Round-1 benchmark diagnostics
3) Strict self-play dataset smoke generation

Writes one consolidated JSON with pass/fail gate fields.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from backend.models.enums import BoardType
from backend.powers.coverage_report import build_report as build_power_coverage
from backend.solver.round1_benchmark import run_round1_benchmark
from backend.ml.self_play_dataset import generate_dataset


def run_strict_kpi(
    out_path: str,
    *,
    players: int = 2,
    board_type: BoardType = BoardType.OCEANIA,
    round1_games: int = 50,
    smoke_games: int = 10,
    max_turns: int = 120,
    seed: int | None = None,
    # Gate thresholds
    min_round1_pct_15_30: float = 0.20,
    max_round1_strict_rejected_games: int = 0,
    max_smoke_strict_rejected_games: int = 0,
) -> dict:
    coverage = build_power_coverage()
    round1 = run_round1_benchmark(
        games=round1_games,
        players=players,
        board_type=board_type,
        strict_rules_only=True,
        reject_non_strict_powers=True,
        seed=seed,
    )

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        ds_jsonl = str(tdp / "strict_smoke.jsonl")
        ds_meta = str(tdp / "strict_smoke.meta.json")
        smoke_meta = generate_dataset(
            output_jsonl=ds_jsonl,
            metadata_path=ds_meta,
            games=smoke_games,
            players=players,
            board_type=board_type,
            max_turns=max_turns,
            seed=seed,
            strict_rules_only=True,
            reject_non_strict_powers=True,
            max_round=4,
            emit_score_breakdown=True,
        )

    checks = []

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    add_check(
        "round1_pct_15_30_min",
        float(round1["totals"]["pct_15_30"]) >= float(min_round1_pct_15_30),
        f"{round1['totals']['pct_15_30']:.4f} >= {min_round1_pct_15_30:.4f}",
    )
    add_check(
        "round1_strict_rejected_games_max",
        int(round1["strict_rejected_games"]) <= int(max_round1_strict_rejected_games),
        f"{round1['strict_rejected_games']} <= {max_round1_strict_rejected_games}",
    )
    add_check(
        "smoke_strict_rejected_games_max",
        int(smoke_meta.get("strict_rejected_games", 0)) <= int(max_smoke_strict_rejected_games),
        f"{smoke_meta.get('strict_rejected_games', 0)} <= {max_smoke_strict_rejected_games}",
    )

    result = {
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "power_coverage": coverage,
        "round1_benchmark": {k: v for k, v in round1.items() if k != "rows"},
        "smoke_dataset_meta": smoke_meta,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict KPI consolidated pipeline")
    parser.add_argument("--out", default="reports/ml/strict_kpi.json")
    parser.add_argument("--players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--board-type", default="oceania", choices=["base", "oceania"])
    parser.add_argument("--round1-games", type=int, default=50)
    parser.add_argument("--smoke-games", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=120)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-round1-pct-15-30", type=float, default=0.20)
    parser.add_argument("--max-round1-strict-rejected-games", type=int, default=0)
    parser.add_argument("--max-smoke-strict-rejected-games", type=int, default=0)
    args = parser.parse_args()

    result = run_strict_kpi(
        out_path=args.out,
        players=args.players,
        board_type=BoardType(args.board_type),
        round1_games=args.round1_games,
        smoke_games=args.smoke_games,
        max_turns=args.max_turns,
        seed=args.seed,
        min_round1_pct_15_30=args.min_round1_pct_15_30,
        max_round1_strict_rejected_games=args.max_round1_strict_rejected_games,
        max_smoke_strict_rejected_games=args.max_smoke_strict_rejected_games,
    )
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

